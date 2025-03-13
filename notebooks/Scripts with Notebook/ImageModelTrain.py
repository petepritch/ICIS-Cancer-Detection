# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 12:17:49 2025

@author: Drew
"""

# ImageModelTrain.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from preprocessing import TumorPreprocessor
from discreterotation import DiscreteRotation
from ImageModel import ImageModel
import numpy as np
import random
import timm
import torch.nn as nn

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
malignant_weight = 1


                                        
# ------------------------------------------------------------------------
# 1) Set seeds for reproducibility
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
# 3) Data loaders
# ------------------------------------------------------------------------
csv_file = r"C:\Users\Drew\Desktop\cancer project\train-metadata.csv"
img_dir = r"C:\Users\Drew\Desktop\cancer project\train-image\image"

#these transforms for imagenet1k
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    DiscreteRotation(),
   # transforms.RandomCrop(224, pad_if_needed=True),
   # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#these transforms for imagenet21k
#train_transform = transforms.Compose([
#    transforms.Resize((224, 224)),
#    transforms.RandomHorizontalFlip(p=0.5),
#    DiscreteRotation(),
    #transforms.RandomCrop(224, pad_if_needed=True),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                         std=[0.5, 0.5, 0.5])
#])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#test_transform = transforms.Compose([
#    transforms.Resize((224, 224)),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                         std=[0.5, 0.5, 0.5])
#])

preprocessor = TumorPreprocessor(
    csv_file, 
    img_dir, 
    train_transform, 
    test_transform, 
    batch_size=batch_size, 
    num_workers=num_workers,
    undersamplingweight = undersamplingweight
)
train_loader, test_loader = preprocessor.get_dataloaders()

# ------------------------------------------------------------------------
# 4) Construct and modify CNN model
# ------------------------------------------------------------------------

# Modify the final layer for binary classification. THIS IS FOR RESNET
#num_features = model.fc.in_features
#model.fc = nn.Linear(num_features, 1)  # 1 output for binary classification

#model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
#model.reset_classifier(num_classes=1)


#modify final layer, this is for efficientnet
#model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 1)
model.to(device)

# ------------------------------------------------------------------------
# 5) Create training components
# ------------------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([malignant_weight]).to(device))
optimizer = optim.Adam(model.parameters(), lr=learningrate)
scaler = GradScaler()

# ------------------------------------------------------------------------
# 6) Wrap in ImageModel
# ------------------------------------------------------------------------
image_model = ImageModel(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scaler=scaler,
    device=device,
    test_loader=test_loader,
    model_name="resnetv2"
)

# ------------------------------------------------------------------------
# 7) Train & Evaluate
# ------------------------------------------------------------------------
image_model.train(train_loader, epochs=1, verbose=True)
metrics = image_model.evaluate(verbose=True)

# ------------------------------------------------------------------------
# 8) Save the trained model checkpoint
# ------------------------------------------------------------------------
save_path = r"C:\Users\Drew\Desktop\cancer project\models\resnetv2imagenet21k1mweightuncallibrated100xsample1epoch.pth"
image_model.save_model(save_path)
print(f"Model saved to {save_path}")
