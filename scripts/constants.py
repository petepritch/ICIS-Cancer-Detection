from utils import *

# Global constants
ZIP_PATH = "/content/isic-2024-challenge.zip" # Zip file path
EXTRACT_TO = "/content/isic-2024-challenge" # Unzip target file path
IMAGE_PATH = "/content/isic-2024-challenge/train-image.hdf5" # Image HDF5 file path
META_PATH = "/content/isic-2024-challenge/train-metadata.csv" # Meta-data CSV file path
DTYPE = torch.float32 # Global data type
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Global device
BATCH_SIZE = 32
## Meta data related constants
RESPONSE = "target"
ID = "isic_id"
TRAIN_ONLY_FEATURES = [
  "lesion_id", "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5",
  "mel_mitotic_index", "mel_thick_mm", "tbp_lv_dnn_lesion_confidence"
]
CATEGORICAL_FEATURES = [
    "isic_id", "patient_id", "sex", "anatom_site_general", "image_type",
    "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple", "attribution", "copyright_license"
]
FEATURES_TO_DROP = TRAIN_ONLY_FEATURES + ["sex", "image_type", "attribution", "copyright_license", "tbp_tile_type" , "patient_id"]
FEATURES_TO_IMPUTE = ["age_approx", "anatom_site_general"]
FEATURES_TO_ENCODE = ["anatom_site_general", "tbp_lv_location", "tbp_lv_location_simple"]