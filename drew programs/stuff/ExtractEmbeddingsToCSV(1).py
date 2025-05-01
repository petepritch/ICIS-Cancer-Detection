# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 20:08:00 2025

@author: Drew
"""

# extract_single_embedding_careful_csv.py

import torch
import timm
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import gc


# Import necessary custom classes and functions
try:
    from ImageModel import ImageModel
    from tumor_dataset import TumorDataset
    import torch.nn as nn
    import torch.optim as optim
    from torch.cuda.amp import GradScaler # Deprecated
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as e:
    print(f"Error importing project files: {e}")
    print("Please ensure ImageModel.py, tumor_dataset.py, etc. are accessible.")
    exit()

print("Imports successful.")

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Paths ---
CSV_FILE = r"C:\Users\Drew\Desktop\cancer project\programs\scripts\Project Code\metadata_preprocessed.csv"
IMG_DIR = r"C:\Users\Drew\Desktop\cancer project\train-image\image"
MODEL_SAVE_DIR = r"C:\Users\Drew\Desktop\cancer project\programs\scripts\Project Code\models"
OUTPUT_DIR = r"C:\Users\Drew\Desktop\cancer project\programs\scripts\Project Code\embeddings" # Output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Selection ---
# *** CHOOSE ONE MODEL TO PROCESS ***
# --- Model 0 ---
#CHECKPOINT_FILENAME = "vit_tiny10mweight1epochINDETERMINATE.pth"
#TIMM_MODEL_NAME = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
#MODEL_OUTPUT_NAME = "vit_tiny_indet_1ep" # For output filename
# --- Model 1 ---
#CHECKPOINT_FILENAME = "vit_tiny1mweight10epochSTANDARD.pth"
#TIMM_MODEL_NAME = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
#MODEL_OUTPUT_NAME = "vit_tiny_std_10ep"
# --- Model 2 ---
CHECKPOINT_FILENAME = "vit_small100mweight10epochSTANDARD.pth"
TIMM_MODEL_NAME = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
MODEL_OUTPUT_NAME = "vit_small_std_10ep"
# --- Model 3 ---
#CHECKPOINT_FILENAME = "vit_small1mweight10epochINDETERMINATE.pth"
#TIMM_MODEL_NAME = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
#MODEL_OUTPUT_NAME = "vit_small_indet_1ep"
# --- New Model ---
#CHECKPOINT_FILENAME = "ViT-tiny-100mweight2epochSTANDARD.pth"
#TIMM_MODEL_NAME = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
#MODEL_OUTPUT_NAME = "vit_tiny_std_2ep_new"
# --- End Model Selection ---

CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_FILENAME)
print(f"Processing model: {MODEL_OUTPUT_NAME}")
print(f"Checkpoint: {CHECKPOINT_PATH}")

# --- Data Loading Config ---
INFERENCE_BATCH_SIZE = 64
NUM_WORKERS = 8
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.2
TARGET_COL = 'target'
ID_COL = 'isic_id'

# --- Transforms ---
image_size = 224
eval_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- Embedding Generation Function ---
@torch.no_grad()
def generate_embeddings_single(loader, extractor_model, device):
    # ... (Keep generate_embeddings_single function exactly as before) ...
    extractor_model.eval()
    all_embeddings_list = []
    all_labels = []
    all_ids = []
    num_batches = len(loader)
    for i, batch_data in enumerate(tqdm(loader, desc=f"Generating Embeddings ({device})")):
        images, labels_num, _, img_ids = batch_data
        images = images.to(device)
        features = extractor_model(images)
        if features.ndim == 4:
             features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(start_dim=1)
        all_embeddings_list.append(features.cpu().numpy())
        all_labels.extend(np.array(labels_num))
        all_ids.extend(list(img_ids))
    if not all_ids: return np.array([]), np.array([]), []
    all_embeddings_np = np.concatenate(all_embeddings_list, axis=0)
    all_labels_np = np.array(all_labels)
    return all_embeddings_np, all_labels_np, all_ids

# --- Main Execution ---

# 1. Set up DataLoaders (Consistent Split)
print("\n--- Setting up DataLoaders ---")
# ... (Keep Dataloader setup identical to previous version) ...
try:
    full_metadata_df = pd.read_csv(CSV_FILE); full_metadata_df[ID_COL] = full_metadata_df[ID_COL].astype(str)
    full_dataset = TumorDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, transform=eval_transform, labeling_mode="standard")
    indices = np.arange(len(full_dataset)); targets = full_metadata_df[TARGET_COL].values
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=targets)
    train_subset = Subset(full_dataset, train_idx); test_subset = Subset(full_dataset, test_idx)
    train_loader = DataLoader(train_subset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train subset size: {len(train_subset)}"); print(f"Test subset size: {len(test_subset)}"); print("DataLoaders created.")
except Exception as e: print(f"FATAL ERROR during DataLoader setup: {e}"); import traceback; traceback.print_exc(); exit()


# 2. Instantiate Full Model (Backbone + Head)
print(f"\n--- Preparing Full Model instance ---")
# ... (Keep Model/Head instantiation identical to previous version) ...
try:
    print(f"  Loading base model structure: {TIMM_MODEL_NAME}")
    base_model_no_head = timm.create_model(TIMM_MODEL_NAME, pretrained=False, num_classes=0)
    num_in_features = base_model_no_head.num_features; print(f"  Base model output features: {num_in_features}")
    print(f"  Rebuilding EXACT head: Linear({num_in_features}, 64) -> ReLU -> Dropout(0.5) -> Linear(64, 1)")
    rebuilt_head = nn.Sequential(nn.Linear(num_in_features, 64), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(64, 1))
    base_model_no_head.head = rebuilt_head; print("  Assigned rebuilt head to model.")
    full_model_structure = base_model_no_head.to(DEVICE)
except Exception as e: print(f"FATAL ERROR during Model instantiation or head rebuilding: {e}"); import traceback; traceback.print_exc(); exit()


# 3. Instantiate ImageModel Wrapper and Load Checkpoint (for validation)
print(f"\n--- Loading Checkpoint via ImageModel ---")
# ... (Keep ImageModel instantiation and loading identical to previous version) ...
try:
    dummy_criterion = nn.BCEWithLogitsLoss(); dummy_optimizer = optim.Adam(full_model_structure.parameters(), lr=1e-5)
    dummy_scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=False)
    image_model_wrapper = ImageModel(model=full_model_structure, criterion=dummy_criterion, optimizer=dummy_optimizer, scaler=dummy_scaler, device=DEVICE, test_loader=None, model_name=TIMM_MODEL_NAME)
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH): raise FileNotFoundError("Checkpoint not found")
    image_model_wrapper.load_model(CHECKPOINT_PATH); print("Checkpoint loaded successfully via ImageModel.")
except Exception as e: print(f"  ERROR loading checkpoint via ImageModel: {e}"); import traceback; traceback.print_exc(); exit()


# 4. Define Feature Extractor (Backbone Only)
print("\n--- Defining Feature Extractor (Backbone) ---")
# ... (Keep Feature Extractor definition identical to previous version) ...
try:
    print("Re-instantiating backbone structure...")
    feature_extractor = timm.create_model(TIMM_MODEL_NAME, pretrained=False, num_classes=0)
    loaded_full_state_dict = image_model_wrapper.model.state_dict()
    backbone_state_dict = {k: v for k, v in loaded_full_state_dict.items() if not k.startswith('head.')}
    print(f"Filtered state dict, keeping {len(backbone_state_dict)} backbone keys.")
    missing_keys, unexpected_keys = feature_extractor.load_state_dict(backbone_state_dict, strict=False)
    if missing_keys: print(f"  Warning: Missing keys in backbone: {missing_keys}")
    if unexpected_keys: print(f"  Warning: Unexpected keys in backbone: {unexpected_keys}")
    feature_extractor.to(DEVICE); feature_extractor.eval(); print("Feature extractor (backbone) prepared.")
except Exception as e: print(f"  ERROR preparing feature extractor: {e}"); import traceback; traceback.print_exc(); exit()


# 5. Generate Embeddings using the Backbone Extractor
print("\n--- Generating Embeddings for Training Set ---")
train_embeddings_np, train_labels_np, train_ids = generate_embeddings_single(train_loader, feature_extractor, DEVICE)
print(f"Generated training embeddings shape: {train_embeddings_np.shape}")

print("\n--- Generating Embeddings for Test Set ---")
test_embeddings_np, test_labels_np, test_ids = generate_embeddings_single(test_loader, feature_extractor, DEVICE)
print(f"Generated test embeddings shape: {test_embeddings_np.shape}")


# --- 6. Save Embeddings as CSV --- # <<< MODIFIED SECTION >>>
print("\n--- Saving Results as CSV ---")
print("Note: Saving as CSV may result in large files and take longer.")
feature_dim = train_embeddings_np.shape[1]
embed_cols = [f'embed_{MODEL_OUTPUT_NAME}_{j}' for j in range(feature_dim)]

# Create Training DataFrame
df_train_embed = pd.DataFrame(train_embeddings_np, columns=embed_cols)
df_train_embed[ID_COL] = train_ids
df_train_embed[TARGET_COL] = train_labels_np
df_train_embed = df_train_embed[[ID_COL, TARGET_COL] + embed_cols]

# Create Test DataFrame
df_test_embed = pd.DataFrame(test_embeddings_np, columns=embed_cols)
df_test_embed[ID_COL] = test_ids
df_test_embed[TARGET_COL] = test_labels_np
df_test_embed = df_test_embed[[ID_COL, TARGET_COL] + embed_cols]

# Define output paths with .csv extension
train_output_path = os.path.join(OUTPUT_DIR, f"train_embeddings_{MODEL_OUTPUT_NAME}.csv") # Changed extension
test_output_path = os.path.join(OUTPUT_DIR, f"test_embeddings_{MODEL_OUTPUT_NAME}.csv") # Changed extension

try:
    print(f"Saving training embeddings to: {train_output_path}")
    # Use to_csv instead of to_parquet
    df_train_embed.to_csv(train_output_path, index=False)
    print(f"Saving test embeddings to: {test_output_path}")
    # Use to_csv instead of to_parquet
    df_test_embed.to_csv(test_output_path, index=False)
    print("Embeddings saved successfully as CSV files.")
except Exception as e:
    print(f"Error saving embeddings as CSV: {e}")

# --- End Modified Section ---

# Clean up
print("\nCleaning up...")
del image_model_wrapper, feature_extractor, train_loader, test_loader, full_dataset
del df_train_embed, df_test_embed # Clear potentially large dataframes
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nScript finished.")