# tumor_dataset.py
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import pandas as pd

class TumorDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Load the metadata
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get the image ID and target
        img_id = self.metadata.iloc[idx]['isic_id']
        img_name = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        try:
            # Load the image
            image = Image.open(img_name).convert('RGB')
            
            # Get the label (0 for benign, 1 for malignant)
            label = self.metadata.iloc[idx]['target']
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a placeholder image and the label
            placeholder = torch.zeros(3, 224, 224)
            return placeholder, self.metadata.iloc[idx]['target']