import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image

class BlackWhite2Color(Dataset):
    def __init__(self, root, transform, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode
        
        data_dir = os.path.join(root, mode)
        self.file_list = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.mode, self.file_list[idx])
        img = Image.open(img_path)
        # W, H = img.size[0], img.size[1]

        img_gray = img.convert('L') # 1 dimension
        img_gray = [img_gray, img_gray, img_gray]
        img_gray = Image.merge("RGB", img_gray)   # 3 dimension

        img_rgb = img.convert('RGB')
        
        img_gray = self.transform(img_gray)
        img_rgb = self.transform(img_rgb)
        
        return (img_gray, img_rgb)


def data_loader(root, batch_size=1, shuffle=True, img_size=224, mode='train'):    
    transform = transforms.Compose([
                                    transforms.Resize((img_size, img_size)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                         std=(0.5, 0.5, 0.5))
                                   ])
    dset = BlackWhite2Color(root, transform, mode=mode)
    
    if batch_size == 'all':
        batch_size = len(dset)
        
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=0,
                                          drop_last=True)
    dlen = len(dset)
    
    return dloader, dlen