# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:29:20 2025

@author: Drew
"""

# Add this to model.py or define in the new training script

    
    
    
    # train_img_meta_mutual_attention.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Import your project files
from preprocessing import TumorPreprocessor
from tumor_dataset import TumorDataset # Required by TumorPreprocessor
from utils import score, load_vit_backbone_weights # Need both utils
from encoders import ImageEncoder # Import image encoder
from attention import MultiMutualAttention # Import attention block

# Import model definitions (assuming they are in model.py or defined above/in this script)
# If defined above/in this script, you might not need these imports
from model import FinalClassificationHead # Or define locally
from model import MetadataEncoderMLP # Or define locally
from model import ImageMetadataMutualAttentionModel # Or define locally

# Standard libraries
import pandas as pd
import numpy as np
import random
import os
import time
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Albumentations for transforms (required by TumorPreprocessor)
import albumentations as A
from albumentations.pytorch import ToTensorV2

print("Imports successful.")

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Paths ---
# Use the PREPROCESSED metadata file as it contains the features the MLP was trained on
CSV_FILE = r"C:\Users\Drew\Desktop\cancer project\programs\scripts\Project Code\metadata_preprocessed.csv"
IMG_DIR = r"C:\Users\Drew\Desktop\cancer project\train-image\image"
MODEL_SAVE_DIR = r"C:\Users\Drew\Desktop\cancer project\programs\scripts\Project Code\models"
METADATA_MLP_CHECKPOINT = os.path.join(MODEL_SAVE_DIR, "metadata_mlp_best_pAUC.pth") # Path to trained MLP
# >>> Choose ONE ViT Checkpoint (or set to None to use timm pretrained) <<<
# Example: Use a previously trained ViT-Tiny checkpoint
IMAGE_CHECKPOINT = os.path.join(MODEL_SAVE_DIR, "vit_tiny1mweight10epochSTANDARD.pth") # <<< --- UPDATE THIS PATH (or set to None)
# IMAGE_CHECKPOINT = None # Use timm weights only

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Model Hyperparameters ---
# Image Encoder
VIT_MODEL_NAME = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k' # Example: ViT-Tiny
VIT_PRETRAINED = True if IMAGE_CHECKPOINT is None else False # Use timm pretrained only if no checkpoint specified
IMAGE_FEATURE_DIM = 192 # For ViT-Tiny (Check if different for your model)

# Metadata Encoder (from saved MLP checkpoint)
# These should match the saved MLP architecture loaded from checkpoint
MLP_INPUT_DIM = 37 # Or 38 if that was the final correct number
MLP_HIDDEN_DIMS = [128, 64] # Match the trained MLP
MLP_DROPOUT = 0.3 # Match the trained MLP
# Feature dim is the output of the MLP backbone (before its final classifier)
METADATA_FEATURE_DIM = MLP_HIDDEN_DIMS[-1] if MLP_HIDDEN_DIMS else MLP_INPUT_DIM # 64 in this example

# Attention Block
ATTN_FEATURE_DIMS = [IMAGE_FEATURE_DIM, METADATA_FEATURE_DIM] # e.g. [192, 64]
ATTN_NUM_HEADS = 8 # Example (ensure head_dim is reasonable)
# Try to make head_dim consistent if possible, or ensure it divides inner_dim
ATTN_HEAD_DIM = 128 # Example: results in inner_dim = 256 for Q/K/V projections

# Final Classifier
CLASSIFIER_INPUT_DIM = sum(ATTN_FEATURE_DIMS) # e.g. 192 + 64 = 256
CLASSIFIER_HIDDEN_DIM = 256 # Example
NUM_CLASSES = 1 # Binary

# --- Training Hyperparameters ---
RANDOM_SEED = 42
INITIAL_LEARNING_RATE = 1e-5 # May need lower LR for fine-tuning attention/head
WEIGHT_DECAY = 0.05
MAX_EPOCHS = 15 # Adjust as needed
PATIENCE_LIMIT = 5
WARMUP_EPOCHS = 1
T_MAX = MAX_EPOCHS - WARMUP_EPOCHS
ETA_MIN = 1e-6
BATCH_SIZE = 32 # Adjust based on GPU memory
NUM_WORKERS = 4

FREEZE_IMAGE_BACKBONE = True # Freeze ViT initially?
FREEZE_METADATA_ENCODER = True # Freeze MLP initially?

# --- Data Loading & Sampling Setup ---
# Use same settings as metadata MLP training for consistency
totalIndeterminates = 1068
totalData = 401059
totalMalignants = 393
labeling_strategy = "standard"
if labeling_strategy == "indeterminate_as_malignant":
    undersamplingweight = totalIndeterminates / (totalData - totalIndeterminates) if (totalData - totalIndeterminates) > 0 else 1
else: # strategy is standard
    undersamplingweight = totalMalignants / (totalData - totalMalignants) if (totalData - totalMalignants) > 0 else 1
malignant_weight = 1.0
print(f"Using labeling strategy: {labeling_strategy}")
print(f"Calculated undersampling weight for Sampler: {undersamplingweight:.6f}")
print(f"Using malignant_weight in BCE Loss: {malignant_weight}")

# --- Set Seeds ---
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- Data Loading ---
print("Setting up DataLoaders using TumorPreprocessor...")
image_size = 224 # ViT input size
# Use strong augmentations for image if fine-tuning ViT later
train_transform = A.Compose([
    A.Transpose(p=0.5), A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.OneOf([A.MotionBlur(blur_limit=5), A.MedianBlur(blur_limit=5), A.GaussianBlur(blur_limit=5), A.GaussNoise(p=0.7)], p=0.7),
    A.OneOf([A.OpticalDistortion(distort_limit=1.0), A.GridDistortion(num_steps=5, distort_limit=1.0), A.ElasticTransform(alpha=3)], p=0.7),
    A.CLAHE(clip_limit=4.0, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    A.Resize(image_size, image_size),
    A.CoarseDropout(max_holes=1, max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), fill_value=0, p=0.7),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Instantiate the preprocessor using the PREPROCESSED CSV
try:
    preprocessor = TumorPreprocessor(
        csv_file=CSV_FILE, # <<< Pointing to the preprocessed metadata
        img_dir=IMG_DIR,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        undersamplingweight=undersamplingweight,
        labeling_mode=labeling_strategy
    )
    train_loader, test_loader = preprocessor.get_dataloaders()
    print("DataLoaders created successfully.")
    print(f"Train loader steps per epoch: {len(train_loader)}")
    print(f"Test loader steps per epoch: {len(test_loader)}")
except FileNotFoundError:
     print(f"FATAL ERROR: Could not find CSV_FILE at {CSV_FILE}. Please verify the path.")
     exit()
except Exception as e:
     print(f"FATAL ERROR during DataLoader setup: {e}")
     exit()

# --- Model Definition & Weight Loading ---
print("Defining model components...")

# 1. Image Encoder
image_encoder = ImageEncoder(model_name=VIT_MODEL_NAME, pretrained=VIT_PRETRAINED)
if IMAGE_CHECKPOINT and os.path.exists(IMAGE_CHECKPOINT):
    print(f"Loading weights for Image Encoder from: {IMAGE_CHECKPOINT}")
    # Assuming the checkpoint is from ImageModel.py or similar PyTorch save
    load_vit_backbone_weights(image_encoder, IMAGE_CHECKPOINT, DEVICE) # Use the util function
else:
    print(f"Using timm pretrained weights for {VIT_MODEL_NAME} (no checkpoint provided or found).")
# --- Update IMAGE_FEATURE_DIM just in case ---
IMAGE_FEATURE_DIM = image_encoder.feature_dim
ATTN_FEATURE_DIMS = [IMAGE_FEATURE_DIM, METADATA_FEATURE_DIM]
CLASSIFIER_INPUT_DIM = sum(ATTN_FEATURE_DIMS)
print(f"Actual Image Feature Dim: {IMAGE_FEATURE_DIM}")
print(f"Attention Input Dims: {ATTN_FEATURE_DIMS}")
print(f"Classifier Input Dim: {CLASSIFIER_INPUT_DIM}")

# 2. Metadata Encoder
print("Instantiating ORIGINAL Metadata MLP structure for loading...")
# Instantiate the ORIGINAL MLP class first
metadata_encoder_full = MetadataEncoderMLP(
    input_dim=MLP_INPUT_DIM,
    hidden_dims=MLP_HIDDEN_DIMS,
    output_dim=NUM_CLASSES, # Output dim MUST match the saved model
    dropout_rate=MLP_DROPOUT
)

# Load the state dict from the saved checkpoint
if not os.path.exists(METADATA_MLP_CHECKPOINT):
    print(f"FATAL ERROR: Metadata MLP checkpoint not found at {METADATA_MLP_CHECKPOINT}")
    exit()
print(f"Loading Metadata MLP weights from: {METADATA_MLP_CHECKPOINT}")
# --- Add weights_only=True for security ---
checkpoint_mlp = torch.load(METADATA_MLP_CHECKPOINT, map_location=DEVICE, weights_only=False)
# Load the state dict - should work now as architectures match
metadata_encoder_full.load_state_dict(checkpoint_mlp['model_state_dict'])
print("Metadata MLP weights loaded successfully into original structure.")

# --- ADAPT FOR FEATURE EXTRACTION ---
# Get the feature dimension *before* the final layer
# Assumes the final Linear layer is the last element (-1) in self.mlp
final_layer_original = metadata_encoder_full.mlp[-1]
if isinstance(final_layer_original, nn.Linear):
     METADATA_FEATURE_DIM = final_layer_original.in_features
     print(f"Determined Metadata Feature Dim (from layer {type(final_layer_original)}): {METADATA_FEATURE_DIM}")
     print(f"Replacing final layer of loaded MLP with nn.Identity() for feature extraction.")
     metadata_encoder_full.mlp[-1] = nn.Identity() # Replace the final layer
else:
     print(f"FATAL ERROR: Could not find Linear layer at the end of the loaded MLP (found {type(final_layer_original)}). Cannot adapt for feature extraction.")
     exit()

# Assign the modified MLP as the encoder to be used
metadata_encoder = metadata_encoder_full
metadata_encoder.eval() # Set to eval mode if using for inference/features

# --- Update Dependent Dimensions AGAIN after determining METADATA_FEATURE_DIM ---
ATTN_FEATURE_DIMS = [IMAGE_FEATURE_DIM, METADATA_FEATURE_DIM]
CLASSIFIER_INPUT_DIM = sum(ATTN_FEATURE_DIMS)
print(f"Final Attention Input Dims: {ATTN_FEATURE_DIMS}")
print(f"Final Classifier Input Dim: {CLASSIFIER_INPUT_DIM}")

# --- Re-create Attention and Classifier Head with correct dimensions ---
# 3. Attention Block
print("Re-initializing Attention Block with final dimensions...")
attention_block = MultiMutualAttention(
    feature_dims=ATTN_FEATURE_DIMS,
    num_heads=ATTN_NUM_HEADS,
    head_dim=ATTN_HEAD_DIM
)

# 4. Classifier Head
print("Re-initializing Classifier Head with final dimensions...")
classifier_head = FinalClassificationHead(
    input_dim=CLASSIFIER_INPUT_DIM,
    num_classes=NUM_CLASSES,
    hidden_dim=CLASSIFIER_HIDDEN_DIM
)
# --- End Re-creation ---


# 5. Combined Model (Instantiate AFTER attention/classifier are finalized)
print("Creating combined model...")
model = ImageMetadataMutualAttentionModel(
    image_encoder,
    metadata_encoder, # Pass the adapted MLP
    attention_block,
    classifier_head
)
model.to(DEVICE)
print("Combined model created and moved to device.")

# --- Freezing Strategy ---
trainable_params = 0
total_params = 0
if FREEZE_IMAGE_BACKBONE:
    print("Freezing Image Encoder backbone...")
    for param in model.image_encoder.vit.parameters(): # Freeze the ViT part
        param.requires_grad = False
if FREEZE_METADATA_ENCODER:
    print("Freezing Metadata Encoder...")
    for param in model.metadata_encoder.parameters(): # Freeze the whole MLP
        param.requires_grad = False

print("Trainable parameters after freezing:")
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        print(f"  - {name}")
        trainable_params += param.numel()
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# --- Loss, Optimizer, Scaler, Scheduler ---
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([malignant_weight]).to(DEVICE))
# Ensure optimizer only sees trainable parameters if freezing
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)

# --- Training Loop ---
best_metric = -1.0
patience_counter = 0
best_model_filename = f"img_meta_mutual_attention_best_pAUC.pth"
best_model_path = os.path.join(MODEL_SAVE_DIR, best_model_filename)

print("\nStarting Training Loop for Image-Metadata Mutual Attention Model...")
start_time = time.time()

for epoch in range(MAX_EPOCHS):
    epoch_start_time = time.time()
    print(f"\n===== Epoch {epoch+1}/{MAX_EPOCHS} =====")

    # --- Train ---
    model.train()
    running_loss = 0.0

    for i, batch_data in enumerate(train_loader):
        # Data Handling: Get images, labels, and metadata features
        images, labels_num, metadata_numpy = batch_data

        images = images.to(DEVICE)
        labels = torch.tensor(np.array(labels_num), dtype=torch.float32).to(DEVICE).unsqueeze(1)
        metadata_tensor = torch.tensor(metadata_numpy, dtype=torch.float32).to(DEVICE)

        optimizer.zero_grad()

        # Mixed precision
        with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type=='cuda'):
            # Pass both image and metadata tensors to the combined model
            outputs = model(images, metadata_tensor)
            loss = criterion(outputs, labels)

        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected at epoch {epoch+1}, step {i+1}. Skipping batch.")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"  Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_preds_prob_val = []
    all_labels_val = []
    with torch.no_grad():
        for batch_data in test_loader:
            images, labels_num, metadata_numpy = batch_data

            images = images.to(DEVICE)
            labels = torch.tensor(np.array(labels_num), dtype=torch.float32).to(DEVICE).unsqueeze(1)
            metadata_tensor = torch.tensor(metadata_numpy, dtype=torch.float32).to(DEVICE)

            outputs = model(images, metadata_tensor) # Pass both inputs
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0) # Use image size or metadata size, should be same

            probs = torch.sigmoid(outputs)
            all_preds_prob_val.extend(probs.cpu().numpy().flatten())
            all_labels_val.extend(np.array(labels_num).flatten())

    # --- Calculate Validation Metrics ---
    total_val_samples = len(all_labels_val)
    # ...(Metric calculation logic - same as in train_metadata_mlp.py)...
    if total_val_samples == 0:
        print("Warning: No samples found in test_loader for validation.")
        avg_val_loss = float('inf')
        val_accuracy = 0.0; val_auc = 0.0; val_pauc = -1.0
    else:
        avg_val_loss = val_loss / total_val_samples
        all_labels_val_np = np.array(all_labels_val)
        all_preds_prob_val_np = np.array(all_preds_prob_val)
        all_preds_binary_val_np = (all_preds_prob_val_np > 0.5).astype(int)
        val_accuracy = accuracy_score(all_labels_val_np, all_preds_binary_val_np)
        try: val_auc = roc_auc_score(all_labels_val_np, all_preds_prob_val_np)
        except ValueError: val_auc = 0.0; print("Warning: ROC AUC score calculation error.")
        val_pauc = -1.0
        try:
            solution_df = pd.DataFrame({'row_id': range(total_val_samples),'target': all_labels_val_np})
            submission_df = pd.DataFrame({'row_id': range(total_val_samples),'target': all_preds_prob_val_np})
            val_pauc = score(solution=solution_df, submission=submission_df, row_id_column_name='row_id', min_tpr=0.80)
        except Exception as e: print(f"  Warning: Could not calculate pAUC score: {e}")

    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}, AUC: {val_auc:.4f}, pAUC: {val_pauc:.4f}")
    current_metric = val_pauc

    # --- LR Scheduling Step ---
    current_lr = -1
    if epoch < WARMUP_EPOCHS:
        warmup_lr_start = ETA_MIN
        lr_factor = (epoch + 1) / WARMUP_EPOCHS
        current_warmup_lr = warmup_lr_start + lr_factor * (INITIAL_LEARNING_RATE - warmup_lr_start)
        for param_group in optimizer.param_groups: param_group['lr'] = current_warmup_lr
        current_lr = current_warmup_lr
        print(f"  Warmup Epoch {epoch+1}/{WARMUP_EPOCHS}. Set LR to: {current_lr:.8f}")
    else:
        if epoch >= WARMUP_EPOCHS:
             scheduler.step()
             current_lr = scheduler.get_last_lr()[0]
             print(f"  Cosine Annealing Phase. Stepped Scheduler. Current LR: {current_lr:.8f}")
        else:
             current_lr = optimizer.param_groups[0]['lr']
             print(f"  First Cosine Epoch. LR: {current_lr:.8f}")

    # --- Early Stopping & Model Saving ---
    if current_metric > best_metric:
        best_metric = current_metric
        print(f"  Metric improved to {best_metric:.4f}. Saving model to {best_model_path}...")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_metric (pAUC)': best_metric,
            'val_loss': avg_val_loss,
            # Include key hyperparams for reloading architecture if needed
            'config': {
                 'vit_model_name': VIT_MODEL_NAME,
                 'image_feature_dim': IMAGE_FEATURE_DIM,
                 'mlp_input_dim': MLP_INPUT_DIM,
                 'mlp_hidden_dims': MLP_HIDDEN_DIMS,
                 'metadata_feature_dim': METADATA_FEATURE_DIM,
                 'attn_feature_dims': ATTN_FEATURE_DIMS,
                 'attn_num_heads': ATTN_NUM_HEADS,
                 'attn_head_dim': ATTN_HEAD_DIM,
                 'classifier_input_dim': CLASSIFIER_INPUT_DIM,
                 'classifier_hidden_dim': CLASSIFIER_HIDDEN_DIM
            }
        }, best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  Metric did not improve. Best pAUC: {best_metric:.4f}. Patience: {patience_counter}/{PATIENCE_LIMIT}")

    epoch_end_time = time.time()
    print(f"Epoch {epoch+1} duration: {epoch_end_time - epoch_start_time:.2f} seconds")

    if patience_counter >= PATIENCE_LIMIT:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break

print("\nTraining finished.")
total_training_time = time.time() - start_time
print(f"Total Training Time: {total_training_time / 60:.2f} minutes")

# --- Final Evaluation ---
# ...(Load best model and evaluate logic - same as in train_metadata_mlp.py)...
if os.path.exists(best_model_path):
    print(f"\nLoading best model from {best_model_path} with pAUC: {best_metric:.4f}")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    # --- Re-instantiate model architecture from config if necessary ---
    # config = checkpoint['config']
    # Recreate encoders, attention, classifier, combined model using config...
    # --- Load state dict ---
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("\nFinal evaluation using the best model on the test set:")
    # ...(Evaluation loop and metric calculation - same as validation loop)...
    # Copy the final evaluation logic from the validation block above or from train_metadata_mlp.py
    final_loss = 0.0
    all_preds_prob_final = []
    all_labels_final = []
    with torch.no_grad():
        for batch_data in test_loader:
            images, labels_num, metadata_numpy = batch_data
            images = images.to(DEVICE)
            labels = torch.tensor(np.array(labels_num), dtype=torch.float32).to(DEVICE).unsqueeze(1)
            metadata_tensor = torch.tensor(metadata_numpy, dtype=torch.float32).to(DEVICE)
            outputs = model(images, metadata_tensor) # Pass both inputs
            loss = criterion(outputs, labels)
            final_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            all_preds_prob_final.extend(probs.cpu().numpy().flatten())
            all_labels_final.extend(np.array(labels_num).flatten())

    total_final_samples = len(all_labels_final)
    if total_final_samples > 0:
        avg_final_loss = final_loss / total_final_samples
        all_labels_final_np = np.array(all_labels_final)
        all_preds_prob_final_np = np.array(all_preds_prob_final)
        all_preds_binary_final_np = (all_preds_prob_final_np > 0.5).astype(int)
        final_accuracy = accuracy_score(all_labels_final_np, all_preds_binary_final_np)
        try: final_auc = roc_auc_score(all_labels_final_np, all_preds_prob_final_np)
        except ValueError: final_auc = 0.0
        final_pauc = -1.0
        try:
            solution_df = pd.DataFrame({'row_id': range(total_final_samples), 'target': all_labels_final_np})
            submission_df = pd.DataFrame({'row_id': range(total_final_samples), 'target': all_preds_prob_final_np})
            final_pauc = score(solution=solution_df, submission=submission_df, row_id_column_name='row_id', min_tpr=0.80)
        except Exception as e: print(f"  Warning: Could not calculate final pAUC score: {e}")
        print(f"  Final Test Loss: {avg_final_loss:.4f}")
        print(f"  Final Test Accuracy: {final_accuracy:.4f}")
        print(f"  Final Test AUC: {final_auc:.4f}")
        print(f"  Final Test pAUC: {final_pauc:.4f}")
        cm = confusion_matrix(all_labels_final_np, all_preds_binary_final_np)
        plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title('Final Confusion Matrix (Image+Meta Mutual Attention)'); plt.ylabel('Actual Label'); plt.xlabel('Predicted Label'); plt.show()
    else: print("No samples in test loader for final evaluation.")

else:
    print(f"Could not find the saved best model file ({best_model_path}) to perform final evaluation.")

print("\nScript finished.")