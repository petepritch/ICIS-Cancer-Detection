import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class ISICDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, file_ext='.jpg'):
        """ISIC Dataset implementation."""
        self.data_frame = pd.read_csv(annotations_file, low_memory=False)
        self.img_dir = img_dir
        self.transform = transform
        self.file_ext = file_ext

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_id = self.data_frame.iloc[idx, 0]
        img_name = os.path.join(self.img_dir, img_id)
        
        if not os.path.exists(img_name) and not img_name.endswith(self.file_ext):
            img_name = img_name + self.file_ext
            
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 1] # This assumes the label is in the second column
        
        if self.transform:
            image = self.transform(image)
            
        return image, label