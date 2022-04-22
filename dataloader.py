import torch
from torch.utils.data import Dataset
import torchvision.transforms as Transforms

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
        W, H = img.size[0], img.size[1]
        
        data_l = img.convert('L') # 1 dimension
        data = [data_l, data_l, data_l]
        data = Image.merge("RGB", data)   # 3 dimension
        ground_truth = img.convert('RGB')
        
        data = self.transform(data)
        ground_truth = self.transform(ground_truth)
        
        return (data, ground_truth)


def data_loader(root, batch_size=1, shuffle=True, img_size=224, mode='train'):    
    transform = Transforms.Compose([Transforms.Resize((img_size, img_size)),
                                    Transforms.ToTensor(),
                                    Transforms.Normalize(mean=(0.5, 0.5, 0.5),
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