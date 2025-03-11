from utils import *
from constants import *

class MultiModalDataset(Dataset):
  """
  Combine `ImageFeatureDataset` and `TabularDataset` for ensemble model
  - self.image_feature_dataset: ImageFeatureDataset class object
  - self.tabular_dataset: TabularDataset class object
  """
  def __init__(self, image_feature_dataset, tabular_dataset):
    self.image_feature_dataset = image_feature_dataset
    self.tabular_dataset = tabular_dataset

  def __len__(self):
    return len(self.image_feature_dataset)

  def __getitem__(self, idx):
    # Extract item key and image feature
    image_name, image_feature = self.image_feature_dataset[idx]
    # Convert item key to tabular data index
    tabular_idx = self.tabular_dataset.ids.index(image_name)
    # Extract tabular data
    _, tabular_feature, label = self.tabular_dataset[tabular_idx]

    return image_feature, tabular_feature, label