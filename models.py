import torch
import torch.nn as nn

from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0.1):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: queries with shape of (num_windows*B, N, C)
            k: keys with shape of (num_windows*B, N, C)
            v: values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = self.window_size // 2

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn = nn.ModuleList([
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
        ])

        attn_mask1 = None
        attn_mask2 = None

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0))
        
        self.register_buffer("attn_mask1", attn_mask1)
        self.register_buffer("attn_mask2", attn_mask2)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Double Attn
        shortcut = x
        x = self.norm1(x)
        
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_1 = qkv[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)
        
        if self.shift_size > 0:
            qkv_2 = torch.roll(qkv[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C // 2)
        else:
            qkv_2 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)

        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, k1_windows, v1_windows, self.attn_mask1)
        x2 = self.attn[1](q2_windows, k2_windows, v2_windows, self.attn_mask2)
        
        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)

        if self.shift_size > 0:
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x2 = x2

        x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)
        x = self.proj(x)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        return q_windows, k_windows, v_windows


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=96, embed_dim=96, norm_layer=nn.LayerNorm):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # fixme look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)

        return x


# downsampling
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# upsampling
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpand, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=(C // 4))
        x = x.view(B, -1, C // 4)
        x= self.norm(x)

        return x


# upsampling x4
class PatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super(PatchExpand_X4, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=(C // (self.dim_scale**2)))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x


class TransformerBlock(nn.Module):
    """ A basic Swin Transformer block for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, depth, num_heads=4, window_size=7, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1, norm_layer=nn.LayerNorm):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class ColorTrans(nn.Module):
    def __init__(self, img_size=224, depths=[2, 2, 2, 2], in_chans=3, out_chans=3, patch_size=4, embed_dim=96, num_heads=[4, 4, 8, 16], mlp_ratio=4, window_size=7):
        super(ColorTrans, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # first conv layer n, 3, 224, 224 -> n, 96, 224, 224
        self.input = nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=3, stride=1, padding=1)

        # split image into non_overlapping patches
        # n, 96, 224, 224 -> n, 56 * 56, 96(B, L, C)
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.embed_dim, embed_dim=self.embed_dim
        )

        # layer1 n, 56 * 56, 96 -> n, 56 * 56, 96
        self.layer1 = TransformerBlock(
            dim=self.embed_dim, input_resolution=[self.img_size // 4, self.img_size // 4], depth=self.depths[0], \
            num_heads=self.num_heads[0], window_size=self.window_size, mlp_ratio=self.mlp_ratio
        )
        # down1 n, 56 * 56, 96 -> n, 28 * 28, 192
        self.down1 = PatchMerging(input_resolution=[self.img_size // 4, self.img_size // 4], dim=self.embed_dim)

        # layer2 n, 28 * 28, 192 -> n, 28 * 28, 192
        self.layer2 = TransformerBlock(
            dim=(self.embed_dim * 2), input_resolution=[self.img_size // 8, self.img_size // 8], depth=self.depths[1], \
            num_heads=self.num_heads[1], window_size=self.window_size, mlp_ratio=self.mlp_ratio
        )
        # down2 n, 28 * 28, 192 -> n, 14 * 14, 384
        self.down2 = PatchMerging(input_resolution=[self.img_size // 8, self.img_size // 8], dim=(self.embed_dim * 2))

        # layer3 n, 14 * 14, 384 -> n, 14 * 14, 384
        self.layer3 = TransformerBlock(
            dim=(self.embed_dim * 4), input_resolution=[self.img_size // 16, self.img_size // 16], depth=self.depths[2], \
            num_heads=self.num_heads[2], window_size=self.window_size, mlp_ratio=self.mlp_ratio
        )
        # down3 n, 14 * 14, 384 -> n, 7 * 7, 768
        self.down3 = PatchMerging(input_resolution=[self.img_size // 16, self.img_size // 16], dim=(self.embed_dim * 4))

       # layer4 n, 7 * 7, 768 -> n, 7 * 7, 768
        self.layer4 = TransformerBlock(
            dim=(self.embed_dim * 8), input_resolution=[self.img_size // 32, self.img_size // 32], depth=self.depths[3], \
            num_heads=self.num_heads[3], window_size=self.window_size, mlp_ratio=self.mlp_ratio
        )

        # up1 n, 7 * 7, 768 -> n, 14 * 14, 384
        self.up1 = PatchExpand(input_resolution=[self.img_size // 32, self.img_size // 32], dim=(self.embed_dim * 8))
        
        # n, 14 * 14, 384 -> n, 14 * 14, 384
        self.layer5 = TransformerBlock(
            dim=(self.embed_dim * 4), input_resolution=[self.img_size // 16, self.img_size // 16], depth=self.depths[-2], \
            num_heads=self.num_heads[-2], window_size=self.window_size, mlp_ratio=self.mlp_ratio
        )

        # n, 14 * 14, 384 -> n, 28 * 28, 192
        self.up2 = PatchExpand(input_resolution=[self.img_size // 16, self.img_size // 16], dim=(self.embed_dim * 4))
        
        # n, 28 * 28, 192 -> n, 28 * 28, 192
        self.layer6 = TransformerBlock(
            dim=(self.embed_dim * 2), input_resolution=[self.img_size // 8, self.img_size // 8], depth=self.depths[-3], \
            num_heads=self.num_heads[-3], window_size=self.window_size, mlp_ratio=self.mlp_ratio
        )

        # n, 28 * 28, 192 -> n, 56 * 56, 96
        self.up3 = PatchExpand(input_resolution=[self.img_size // 8, self.img_size // 8], dim=(self.embed_dim * 2))
        
        # n, 56 * 56, 96 -> n, 56 * 56, 96
        self.layer7 = TransformerBlock(
            dim=(self.embed_dim), input_resolution=[self.img_size // 4, self.img_size // 4], depth=self.depths[-4], \
            num_heads=self.num_heads[-4], window_size=self.window_size, mlp_ratio=self.mlp_ratio
        )

        # n, 56 * 56, 96 -> n, 224 * 224, 96
        self.up4 = PatchExpand_X4(input_resolution=[self.img_size // 4, self.img_size // 4], dim=self.embed_dim)

        # n, 96, 224, 224 -> n, 3, 224, 224
        self.output = nn.Conv2d(self.embed_dim, self.out_chans, kernel_size=3, stride=1, padding=1, bias=False)

        # concat linear
        self.concat_linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.concat_linear2 = nn.Linear(4 * self.embed_dim, 2 * self.embed_dim)
        self.concat_linear3 = nn.Linear(8 * self.embed_dim, 4 * self.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = self.input(x)
        patch_embed = self.patch_embed(input)
        layer1 = self.layer1(patch_embed)
        down1 = self.down1(layer1)
        layer2 = self.layer2(down1)
        down2 = self.down2(layer2)
        layer3 = self.layer3(down2)
        down3 = self.down3(layer3)
        layer4 = self.layer4(down3)
        up1 = self.up1(layer4)
        layer5 = self.layer5(up1)
        up2 = self.up2(self.concat_linear3(torch.cat([layer5, layer3], dim=-1)))
        layer6 = self.layer6(up2)   
        up3 = self.up3(self.concat_linear2(torch.cat([layer6, layer2], dim=-1)))
        layer7 = self.layer7(up3)
        up4 = self.up4(self.concat_linear1(torch.cat([layer7, layer1], dim=-1)))
        output = self.output(up4.permute(0, 2, 1).view(-1, self.embed_dim, self.img_size, self.img_size))

        return output


if __name__ == '__main__':
    x = torch.randn((2, 1, 224, 224))
    model = ColorTrans()
    print(model)
    print((model(x)).shape)

