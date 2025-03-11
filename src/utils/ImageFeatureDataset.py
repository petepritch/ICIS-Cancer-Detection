from helpers import *
from constants import *

class ImageFeatureDataset(Dataset):
    """
    Extracts and stores image feature vectors processed by vision model
    - self.image_dataset: Original Image Dataset, of class `ImageDataset`
    - self.model: Vision model to extract features
    - self.device: Model device
    """
    def __init__(self, image_dataset, model, device = DEVICE):
        self.image_dataset = image_dataset  # Original Image Dataset
        self.model = model.to(device).eval()  # Ensure model is in eval mode
        self.device = device

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        image_name, image = self.image_dataset[idx]  # Get image data
        with torch.no_grad():
            feature_vector = self.model(image.unsqueeze(0).to(self.device)).squeeze()  # Extract features
        return image_name, feature_vector