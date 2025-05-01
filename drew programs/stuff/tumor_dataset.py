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
                 metadata_cols=None,
                 labeling_mode="standard"):
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
        self.labeling_mode = labeling_mode
        #comment below to use standard
        #self.metadataValues = self.metadata.drop(columns=['target', 'isic_id']).values
        #numeric_cols = self.metadata.select_dtypes(include=np.number).columns.tolist()
        #self.metadataValues = self.metadata[numeric_cols].fillna(0).values.astype(np.float32)
        # If the user didn't specify columns, just do an empty list for now
        #uncomment below to use standard
        feature_columns = self.metadata.columns.drop(['target', 'isic_id'])
        self.metadataValues = self.metadata[feature_columns].values.astype(np.float32)
        if metadata_cols is None:
            metadata_cols = []
        self.metadata_cols = metadata_cols
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get the image ID and target
        row = self.metadata.iloc[idx]
        img_id = row['isic_id']
        original_label = row['target']
        img_name = os.path.join(self.img_dir, f"{img_id}.jpg")
        rowValues = self.metadataValues[idx]
        
        #determine final label
        final_label = original_label
        if self.labeling_mode == "indeterminate_as_malignant" and original_label == 0:
            #check only if original label is benign (0)
            #is_indeterminate = (row['iddx_1'] == 'Indeterminate')
            has_iddx2 = pd.notna(row['iddx_2'])
            
            if has_iddx2:
              final_label = 1

        # Load image safely
        
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a placeholder image if we can't load
            image = Image.new('RGB', (224, 224), color='black')
            
        # Convert to numpy array
        image_np = np.array(image)
        # ----Apply transforms----
        if self.transform:
            transformed = self.transform(image=image_np)
            image = transformed['image']

        return image, final_label, rowValues, img_id

    def set_metadata(self, inMetadata):
        self.metadata = inMetadata