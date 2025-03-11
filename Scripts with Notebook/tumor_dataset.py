# tumor_dataset.py

from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import pandas as pd
import numpy as np

#have to do metadata processing inside here, so that the hidden test data is sufficiently modified
class TumorDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None,
                 include_metadata=False,
                 metadata_cols=None):
        """
        csv_file: Path to the CSV containing metadata (including 'isic_id', 'target', etc.)
        img_dir: Directory containing images named like "ISIC_XXXXXX.jpg"
        transform: Any image transform
        include_metadata: If True, __getitem__ will return (image, label, metadata_tensor)
        metadata_cols: List of columns from the CSV to include in the metadata array.
        """
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.include_metadata = include_metadata
        
        # If the user didn't specify columns, just do an empty list for now
        if metadata_cols is None:
            metadata_cols = []
        self.metadata_cols = metadata_cols
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get the image ID and target
        row = self.metadata.iloc[idx]
        img_id = row['isic_id']
        label = row['target']
        img_name = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        # Load image safely
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a placeholder image if we can't load
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # If requested, grab the numeric metadata
        if self.include_metadata and self.metadata_cols:
            # Extract columns, fill missing with 0
            meta_vals = row[self.metadata_cols].fillna(0).infer_objects(copy=False).values #infer objects to silence a python warning
            # Convert to float32 tensor
            meta_tensor = torch.tensor(meta_vals, dtype=torch.float32)
            return image, label, meta_tensor
        else:
            # Original behavior
            return image, label

    def set_metadata(self, inMetadata):
        self.metadata = inMetadata