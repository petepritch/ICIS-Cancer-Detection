from utils import *
from constants import *

class ImageDataset(Dataset):
  """
  Customized dataset class for loading image data from HDF5 file
  - self.file_path: Path to HDF5 file
  - self.transform: Image preprocess transformation, default Resize([224, 224]) + ToTensor()
  - self.image_names: List of image names in HDF5 file (`isic_id` in meta-data)
  """
  def __init__(self, file_path, transform = None):
    self.file_path = file_path # Image file path
    # Image preprocess transformation
    if transform:
      self.transform = transform
    else:
      self.transform = transforms.Compose(
          [
              transforms.Resize((224, 224)),
              transforms.ToTensor(),
          ]
      )

    with h5py.File(self.file_path, "r") as f:
      self.image_names = list(f.keys())

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, idx):
    image_name = self.image_names[idx] # Load image name

    with h5py.File(self.file_path, "r") as f:
      image_bytes = f[image_name][()] # Load image bytes

    image = Image.open(BytesIO(image_bytes)) # Convert bytes to image

    image = self.transform(image)

    return image_name, image