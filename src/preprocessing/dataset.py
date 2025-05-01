from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np


class TumorDataset(Dataset):
    """
    Dataset class for tumor images and metadata.
    """

    def __init__(
        self,
        csv_file,
        img_dir,
        transform=None,
        include_metadata=False,
        metadata_cols=None,
        labeling_mode="standard",
    ):
        """
        Initialize tumor dataset.

        Args:
            csv_file: Path to the CSV containing metadata
            img_dir: Directory containing images named like "ISIC_XXXXXX.jpg"
            transform: Image transforms to apply
            include_metadata: If True, __getitem__ will return (image, label, metadata)
            metadata_cols: List of columns from the CSV to include in the metadata
            labeling_mode: "standard" or "indeterminate_as_malignant"
        """
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.include_metadata = include_metadata
        self.labeling_mode = labeling_mode

        # Prepare metadata values
        feature_columns = self.metadata.columns.drop(["target", "isic_id"])
        self.metadataValues = self.metadata[feature_columns].values.astype(np.float32)

        # Set metadata columns
        if metadata_cols is None:
            metadata_cols = []
        self.metadata_cols = metadata_cols

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get image ID and target
        row = self.metadata.iloc[idx]
        img_id = row["isic_id"]
        original_label = row["target"]
        img_name = os.path.join(self.img_dir, f"{img_id}.jpg")
        rowValues = self.metadataValues[idx]

        # Determine final label based on labeling mode
        final_label = original_label
        if self.labeling_mode == "indeterminate_as_malignant" and original_label == 0:
            has_iddx2 = pd.notna(row["iddx_2"])
            if has_iddx2:
                final_label = 1

        # Load image
        try:
            image = Image.open(img_name).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return placeholder for failed loads
            image = Image.new("RGB", (224, 224), color="black")

        # Apply transforms
        image_np = np.array(image)
        if self.transform:
            transformed = self.transform(image=image_np)
            image = transformed["image"]

        return image, final_label, rowValues, img_id

    def set_metadata(self, metadata):
        """Update the dataset's metadata."""
        self.metadata = metadata
