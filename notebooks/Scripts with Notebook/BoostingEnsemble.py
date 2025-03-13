# BoostingEnsemble.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from preprocessing import TumorPreprocessor
from discreterotation import DiscreteRotation
from ImageModel import ImageModel
from BoostedClassifier import BoostedClassifier
from build_model import build_model  # The helper that builds arch by name
import numpy as np
import random
from torchvision import transforms

from XGBoostClassifier import XGBoostClassifier
from LightGBMClassifier import LightGBMClassifier


#  _    _                                                            _                 
# | |  | |                                                          | |                
# | |__| |_   _ _ __   ___ _ __ _ __   __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___  
# |  __  | | | | '_ \ / _ \ '__| '_ \ / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __| 
# | |  | | |_| | |_) |  __/ |  | |_) | (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \ 
# |_|  |_|\__, | .__/ \___|_|  | .__/ \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/ 
#          __/ | |             | |                                                     
#         |___/|_|             |_|            

num_workers = 8
learningrate = 0.00001
batch_size = 32
undersamplingweight = 1


# ------------------------------------------------------------------------
# 1) Set seeds
# ------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ------------------------------------------------------------------------
# 2) Device setup
# ------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------------
# 3) Data loaders for ensemble
#     We need a loader for train so XGBoost can build the dataset, and test for eval
# ------------------------------------------------------------------------
csv_file = r"C:\Users\Drew\Desktop\cancer project\train-metadata.csv"
img_dir = r"C:\Users\Drew\Desktop\cancer project\train-image\image"


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    DiscreteRotation(),
    transforms.RandomCrop(224, pad_if_needed=True),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#currently these meta_cols are unused
meta_cols = ['age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 
             'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 
             'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 
             'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 
             'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 
             'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt', 
             'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'sex_female', 'sex_male', 
             'sex_nan', 'anatom_site_general_anterior torso', 'anatom_site_general_head/neck', 'anatom_site_general_lower extremity', 
             'anatom_site_general_posterior torso', 'anatom_site_general_upper extremity', 'anatom_site_general_nan', 
             'tbp_lv_location_Head & Neck', 'tbp_lv_location_Left Arm', 'tbp_lv_location_Left Arm - Lower', 
             'tbp_lv_location_Left Arm - Upper', 'tbp_lv_location_Left Leg', 'tbp_lv_location_Left Leg - Lower', 
             'tbp_lv_location_Left Leg - Upper', 'tbp_lv_location_Right Arm', 'tbp_lv_location_Right Arm - Lower', 
             'tbp_lv_location_Right Arm - Upper', 'tbp_lv_location_Right Leg', 'tbp_lv_location_Right Leg - Lower', 
             'tbp_lv_location_Right Leg - Upper', 'tbp_lv_location_Torso Back', 'tbp_lv_location_Torso Back Bottom Third', 
             'tbp_lv_location_Torso Back Middle Third', 'tbp_lv_location_Torso Back Top Third', 'tbp_lv_location_Torso Front', 
             'tbp_lv_location_Torso Front Bottom Half', 'tbp_lv_location_Torso Front Top Half', 'tbp_lv_location_Unknown', 
             'tbp_lv_location_simple_Head & Neck', 'tbp_lv_location_simple_Left Arm', 'tbp_lv_location_simple_Left Leg', 
             'tbp_lv_location_simple_Right Arm', 'tbp_lv_location_simple_Right Leg', 'tbp_lv_location_simple_Torso Back', 
             'tbp_lv_location_simple_Torso Front', 'tbp_lv_location_simple_Unknown', 'isic_id', 'target']
#meta_cols implementation is unused currently
preprocessor = TumorPreprocessor(
    csv_file,
    img_dir,
    train_transform,
    test_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    include_metadata=True,
    metadata_cols=meta_cols
)
train_loader, test_loader = preprocessor.get_gb_dataloader()

# ------------------------------------------------------------------------
# 4) Build & Load multiple CNNs
# ------------------------------------------------------------------------
# Example: We have two saved models, one is efficientnet_b3, one is efficientnet_b0
model_paths = [
    ( "efficientnet_b3", r"C:\Users\Drew\Desktop\cancer project\models\efficientnetb3_epoch1.pth" )
    , ( "efficientnet_b0", r"C:\Users\Drew\Desktop\cancer project\models\efficientnetb0malignantweight50_epoch1.pth" )
]
  
image_models = []
for model_name, ckpt_path in model_paths:
    # Build the correct architecture
    base_cnn = build_model(model_name=model_name, pretrained=False)
    base_cnn.to(device)
    
    # Create dummy criterion, optimizer, scaler 
    # (since we won't train these models now, they're placeholders)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(base_cnn.parameters(), lr=1e-5)
    scaler = GradScaler()
    
    # Wrap in ImageModel
    img_model = ImageModel(
        model=base_cnn,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        test_loader=None,
        model_name=model_name
    )
    
    # Load saved weights
    img_model.load_model(ckpt_path)
    image_models.append(img_model)

# ------------------------------------------------------------------------
# 5) Instantiate BoostedClassifier with multiple models
# ------------------------------------------------------------------------

#xgboost
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    # For XGBoost >= 2.0:
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
#lgboost
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'device': 'gpu',
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbose': -1
}
#xgboost
boosted_classifier = BoostedClassifier(
    image_models=image_models,
    params=params,
    num_boost_round=1000
)

#lgboost
lgb_classifier = LightGBMClassifier(
    image_models=image_models,
    params=lgb_params,
    num_boost_round=1000
)
lgb_classifier.train_from_loader(train_loader)


# Evaluate LightGBM
print("LightGBM Evaluation:")
lgb_classifier.evaluate_from_loader(test_loader)
lgb_classifier.save_model(r"C:\Users\Drew\Desktop\cancer project\models\ensemble_lgb.json")

# To get column names from the underlying train dataset:
#metadata_columns = train_loader.dataset.dataset.metadata.columns.tolist()
#print(metadata_columns)




# ------------------------------------------------------------------------
# 6) Train XGBoost on the combined embeddings
# ------------------------------------------------------------------------
boosted_classifier.train_from_loader(train_loader)

# ------------------------------------------------------------------------
# 7) Evaluate on test
# ------------------------------------------------------------------------
boosted_classifier.evaluate_from_loader(test_loader)

# (Optional) Save the XGBoost model
boosted_classifier.save_model(r"C:\Users\Drew\Desktop\cancer project\models\ensemble_xgb.json")
