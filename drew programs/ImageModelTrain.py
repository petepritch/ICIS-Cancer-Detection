# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from preprocessing import TumorPreprocessor
from discreterotation import DiscreteRotation
from ImageModel import ImageModel
import numpy as np
import random
import xgboost as xgb

#set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    DiscreteRotation(),
    transforms.RandomCrop(224, pad_if_needed=True),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data loaders
csv_file = r"C:\Users\Drew\Desktop\cancer project\train-metadata.csv"
img_dir = r"C:\Users\Drew\Desktop\cancer project\train-image\image"
preprocessor = TumorPreprocessor(csv_file, img_dir, train_transform, test_transform, batch_size=64, num_workers=15)
train_loader, test_loader = preprocessor.get_dataloaders()

# Model setup
model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
#this is for efficientnet
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 1)
model = model.to(device)

# Modify the final layer for binary classification. THIS IS FOR RESNET
#num_features = model.fc.in_features
#model.fc = nn.Linear(num_features, 1)  # 1 output for binary classification
###MAYBE 2????


# Training components
malignant_weight = 5
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([malignant_weight]).to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scaler = GradScaler()

# Instantiate ImageModel
image_model = ImageModel(model, criterion, optimizer, scaler, device, test_loader, model_name="efficientnet_b3")

# Training
image_model.train(train_loader, epochs=1, verbose=True)

# Save model
save_path = r"C:\Users\Drew\Desktop\cancer project\models\tumor_classifier.pth"
image_model.save_model(save_path)
print("Model saved successfully!")

# Evaluation
metrics = image_model.evaluate(verbose=True)


train_loader_for_embeddings = train_loader


#xgboost stuff. NEEDS TO BE REFACTORED INTO NEW CLASS
embedding_list = []
label_list = []

for images, labels in train_loader_for_embeddings:
    embeddings = image_model.extract_embeddings(images)  # Embeddings on CPU
    embedding_list.append(embeddings.numpy())
    label_list.append(labels.numpy())

X_train = np.vstack(embedding_list)
y_train = np.concatenate(label_list)

# Prepare data
dtrain = xgb.DMatrix(X_train, label=y_train)
# Calculate class weight for imbalance:
num_negatives = np.sum(y_train == 0)
num_positives = np.sum(y_train == 1)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',  # GPU acceleration
    'max_depth': 6,
    'learning_rate': 0.01,
    'scale_pos_weight': num_negatives / num_positives,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

bst = xgb.train(params, dtrain, num_boost_round=500)

test_loader_for_embeddings = test_loader

_, test_loader_for_embeddings = preprocessor.get_dataloaders()

embedding_test_list = []
label_test_list = []

for images, labels in test_loader_for_embeddings:
    embeddings = image_model.extract_embeddings(images)
    embedding_test_list.append(embeddings.numpy())
    label_test_list.append(labels.numpy())

X_test = np.vstack(embedding_test_list)
y_test = np.concatenate(label_test_list)

dtest = xgb.DMatrix(X_test)
y_pred_probs = bst.predict(dtest)
y_pred = (y_pred_probs > 0.5).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
