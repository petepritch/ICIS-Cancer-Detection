# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 20:26:39 2025

@author: Drew
"""

import torch
import timm
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import gc
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from utils import score as calculate_isic_pauc
import time
RANDOM_SEED = 42

dfTrain = pd.read_csv("C:/Users/Drew/Desktop/cancer project/programs/scripts/Project Code/embeddings/train_embeddings_vit_tiny_std_2ep_new.csv")
dfTest = pd.read_csv("C:/Users/Drew/Desktop/cancer project/programs/scripts/Project Code/embeddings/test_embeddings_vit_tiny_std_2ep_new.csv")
dfMeta = pd.read_csv(r"C:\Users\Drew\Desktop\cancer project\programs\scripts\Project Code\metadata_preprocessed.csv")

#metadata splitting
metaIndices = dfMeta.index
metaTargets = dfMeta['target']
train_meta_idx, test_meta_idx = train_test_split(metaIndices, test_size = 0.2, random_state=RANDOM_SEED, stratify=metaTargets)
dfMetaTrain = dfMeta.loc[train_meta_idx].copy()
dfMetaTest = dfMeta.loc[test_meta_idx].copy()   

dfTrainFinal = pd.merge(dfTrain, dfMetaTrain.drop(columns=["target"]), on="isic_id", how = 'inner')
dfTestFinal = pd.merge(dfTest, dfMetaTest.drop(columns=["target"]), on="isic_id", how = 'inner')

TARGET_COL = 'target'
ID_COL = 'isic_id'
y_train = dfTrain["target"]
y_test = dfTest["target"]
feature_cols = [col for col in dfTrainFinal.columns if col not in [ID_COL, TARGET_COL]]

#embedding_cols_train = [col for col in dfTrain.columns if col.startswith('embed_')]
#embedding_cols_test = [col for col in dfTest.columns if col.startswith('embed_')]

X_train = dfTrainFinal[feature_cols]
X_test = dfTestFinal[feature_cols]


##trybig block
TARGET_COL = 'target'
# ID_COL defined above
EMBEDDING_DIR = r"C:\Users\Drew\Desktop\cancer project\programs\scripts\Project Code\embeddings"
METADATA_CSV_FILE = r"C:\Users\Drew\Desktop\cancer project\programs\scripts\Project Code\metadata_preprocessed.csv"
MODEL_SAVE_DIR = r"C:\Users\Drew\Desktop\cancer project\programs\scripts\Project Code\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# *** Define the 5 Embedding Models to Ensemble ***
MODEL_NAMES_TO_USE = [
    #"vit_small_indet_1ep",      # From original set
    #"vit_small_std_10ep",       # From original set
    "vit_tiny_indet_1ep",       # From original set
    #"vit_tiny_std_10ep",        # From original set
    "vit_tiny_std_2ep_new"      # Your newly trained one
]
print(f"Using embeddings from {len(MODEL_NAMES_TO_USE)} models: {MODEL_NAMES_TO_USE}")

RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.2

FEATURES_DESCRIPTION = f"Ensemble_{len(MODEL_NAMES_TO_USE)}Embed_Meta" # For output filename

print(f"Using Metadata CSV: {METADATA_CSV_FILE}")

# --- Step 1: Load Metadata and Split ---
try:
    dfMeta = pd.read_csv(METADATA_CSV_FILE)
    dfMeta[ID_COL] = dfMeta[ID_COL].astype(str)
    print(f"Loaded dfMeta shape: {dfMeta.shape}")

    print("Splitting metadata...")
    metaIndices = dfMeta.index
    metaTargets = dfMeta[TARGET_COL].values
    train_meta_idx, test_meta_idx = train_test_split(
        metaIndices, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=metaTargets
    )
    dfMetaTrain = dfMeta.loc[train_meta_idx].copy()
    dfMetaTest = dfMeta.loc[test_meta_idx].copy()
    print(f"Split metadata: Train shape={dfMetaTrain.shape}, Test shape={dfMetaTest.shape}")

except Exception as e: print(f"Error loading/splitting metadata: {e}"); raise


# --- Step 2: Load Embeddings and Merge Sequentially ---
print("\n--- Step 2: Loading and Merging Embeddings with Metadata ---")
try:
    # Start with the split metadata as the base
    dfTrainFinal = dfMetaTrain.copy()
    dfTestFinal = dfMetaTest.copy()

    # Loop through each model name, load embeddings, and merge
    for model_name in MODEL_NAMES_TO_USE:
        print(f"  Processing: {model_name}")
        train_emb_path = os.path.join(EMBEDDING_DIR, f"train_embeddings_{model_name}.csv")
        test_emb_path = os.path.join(EMBEDDING_DIR, f"test_embeddings_{model_name}.csv")

        if not os.path.exists(train_emb_path): raise FileNotFoundError(f"Train embed not found: {train_emb_path}")
        if not os.path.exists(test_emb_path): raise FileNotFoundError(f"Test embed not found: {test_emb_path}")

        dfTrainEmbed_single = pd.read_csv(train_emb_path)
        dfTestEmbed_single = pd.read_csv(test_emb_path)
        dfTrainEmbed_single[ID_COL] = dfTrainEmbed_single[ID_COL].astype(str)
        dfTestEmbed_single[ID_COL] = dfTestEmbed_single[ID_COL].astype(str)

        # Merge train data (drop target from embedding file to avoid duplication)
        dfTrainFinal = pd.merge(
            dfTrainFinal,
            dfTrainEmbed_single.drop(columns=[TARGET_COL], errors='ignore'),
            on=ID_COL, how='inner'
        )
        # Merge test data
        dfTestFinal = pd.merge(
            dfTestFinal,
            dfTestEmbed_single.drop(columns=[TARGET_COL], errors='ignore'),
            on=ID_COL, how='inner'
        )
        print(f"    Merged {model_name}. Train shape: {dfTrainFinal.shape}, Test shape: {dfTestFinal.shape}")

    # Verification: Check final row counts (should match original split sizes)
    if len(dfTrainFinal) != len(dfMetaTrain): print("WARNING: Final Train row count mismatch!")
    if len(dfTestFinal) != len(dfMetaTest): print("WARNING: Final Test row count mismatch!")

except Exception as e: print(f"Error loading/merging data: {e}"); raise


# --- Step 3: Prepare Final X and y ---
print("\n--- Step 3: Preparing Final X and y (All Embeddings + Metadata) ---")
try:
    y_train = dfTrainFinal[TARGET_COL]
    y_test = dfTestFinal[TARGET_COL]

    # Define feature columns: all columns except ID and Target
    feature_cols = [col for col in dfTrainFinal.columns if col not in [ID_COL, TARGET_COL]]
    if not feature_cols: raise ValueError("No feature columns identified after merge.")
    num_meta_cols = len(dfMetaTrain.columns) - 2 # Exclude ID and Target from meta
    num_embed_cols = len(feature_cols) - num_meta_cols
    print(f"Selected {len(feature_cols)} total features ({num_embed_cols} embeddings + {num_meta_cols} metadata).")

    X_train = dfTrainFinal[feature_cols]
    X_test = dfTestFinal[feature_cols].copy() # Ensure using same columns

    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # --- Verification Print ---
    print("\nColumns in final X_train (first 5 + last 5):")
    print(X_train.columns.tolist()[:5] + ['...'] + X_train.columns.tolist()[-5:])
    if TARGET_COL in X_train.columns or ID_COL in X_train.columns:
         raise ValueError(f"Target or ID column found in X_train features!")
    print(f"Verified: '{TARGET_COL}' and '{ID_COL}' are NOT in X features.")
except Exception as e: print(f"Error preparing X/y: {e}"); raise





N_TRIALS_OPTUNA = 100

def lgbm_pauc_metric(y_true, y_pred_proba):
    ids = np.arange(len(y_true)); y_true_arr = np.array(y_true)
    solution_df = pd.DataFrame({ID_COL: ids, TARGET_COL: y_true_arr.astype(int)})
    submission_df = pd.DataFrame({ID_COL: ids, TARGET_COL: y_pred_proba})
    try:
        pauc_value = calculate_isic_pauc(solution=solution_df, submission=submission_df, row_id_column_name=ID_COL, min_tpr=0.80)
        return ('pAUC', pauc_value if not np.isnan(pauc_value) else 0.0, True)
    except Exception: return ('pAUC', 0.0, True)

def pauc_scorer_func(y_true, y_pred_proba):
    ids = np.arange(len(y_true)); y_true_arr = np.array(y_true)
    solution_df = pd.DataFrame({ID_COL: ids, TARGET_COL: y_true_arr.astype(int)})
    submission_df = pd.DataFrame({ID_COL: ids, TARGET_COL: y_pred_proba})
    try:
        pauc_value = calculate_isic_pauc(solution=solution_df, submission=submission_df, row_id_column_name=ID_COL, min_tpr=0.80)
        return pauc_value if not np.isnan(pauc_value) else 0.0
    except Exception: return 0.0

def objective(trial, x_train, y_train, x_val, y_val):
    """Objective function for Optuna hyperparameter optimization."""
    #sampler1 = RandomOverSampler(sampling_strategy = 0.003, random_state = RANDOM_SEED)
    sampler2 = RandomUnderSampler(sampling_strategy = 0.01, random_state = RANDOM_SEED)
    
    x_train_resampled, y_train_resampled = sampler2.fit_resample(x_train, y_train)
    #x_train_resampled, y_train_resampled = sampler2.fit_resample(x_train_resampled, y_train_resampled)
    
    # --- Define Hyperparameter Search Space ---
    lgbm_params = {
        'objective':         'binary',
        'metric':            'auc', # Use AUC for internal LGBM reporting/stopping
        'verbosity':         -1,
        'random_state':      42,
        'n_estimators':      2000, # High value, rely on early stopping
        'n_jobs':            8,
        'boosting_type':     'gbdt',
        'lambda_l1':         trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2':         trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'learning_rate':     trial.suggest_float('learning_rate', 1e-2, 0.1, log=True),
        #'num_leaves':        trial.suggest_int('num_leaves', 16, 256),
        'max_depth':         trial.suggest_int('max_depth', 4, 8),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bynode':  trial.suggest_float('colsample_bynode', 0.4, 1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq':      trial.suggest_int('bagging_freq', 1, 7),
        'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 5, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        # --- Let Optuna tune scale_pos_weight ---
        'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0.8, 4.0), # Tune weight
    }

    # --- Train and Evaluate Model for this Trial ---
    model = lgb.LGBMClassifier(**lgbm_params)

    # Fit model with early stopping based on the validation set (X_test, y_test here)
    model.fit(x_train_resampled, y_train_resampled,
              eval_set=[(x_val, y_val)],
              eval_metric='auc', # Monitor AUC for early stopping
              callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]) # Quiet stopping

    # Get the best score achieved on the validation set during training
    # Accessing score directly might depend on LGBM version, predict_proba is safer
    y_pred_proba_val = model.predict_proba(x_val)[:, 1]
    val_pauc = pauc_scorer_func(y_val, y_pred_proba_val)

    print(f"Trial {trial.number}: Test pAUC={val_pauc:.6f} (at iter {model.best_iteration_})")

    # Return the validation AUC score for Optuna to maximize
    return val_pauc

print(f"\n--- Step 4: Starting Optuna optimization ({N_TRIALS_OPTUNA} trials) ---")
study_name = "study"
study = optuna.create_study(direction='maximize', study_name=study_name)

# Pass the main train/test splits directly to the objective function
objective_with_data = lambda trial: objective(trial, X_train, y_train, X_test, y_test)

start_opt_time = time.time()
try:
    study.optimize(objective_with_data, n_trials=N_TRIALS_OPTUNA)
except Exception as e:
    print(f"\nOptuna study optimize error: {e}"); import traceback; traceback.print_exc()
end_opt_time = time.time()

print(f"\nOptuna optimization finished in {end_opt_time - start_opt_time:.2f} seconds.")
try:
    print(f"Best trial number: {study.best_trial.number}"); print(f"Best Test pAUC score during Optuna: {study.best_value:.6f}")
    print("Best hyperparameters found:"); best_params_optuna = study.best_params
    for key, value in best_params_optuna.items(): print(f"  {key}: {value}")
except Exception as e: print(f"Could not retrieve best trial info: {e}"); best_params_optuna = {}



# --- Train Final Model ---
print("\n--- Training final model ---")
# (Keep Final Model training as in lgbm_optuna_simple_pAUC_Opt_EmbedOnly.py)
if not best_params_optuna:
    print("WARNING: Optuna did not find best parameters. Using defaults with calculated weight.")
    final_params={'objective':'binary', 'metric':'auc', 'random_state':RANDOM_SEED, 'n_estimators':1000, 'learning_rate':0.05, 'n_jobs':-1}
    neg_count_train=(y_train==0).sum(); pos_count_train=(y_train==1).sum(); final_params['scale_pos_weight']=neg_count_train/pos_count_train if pos_count_train > 0 else 1
else:
    final_params = best_params_optuna.copy()
    final_params['objective'] = 'binary'; final_params['metric'] = 'None';
    final_params['random_state'] = RANDOM_SEED; final_params['n_jobs'] = -1;
    final_params['n_estimators'] = 5000
    final_params['scale_pos_weight'] = best_params_optuna.get('scale_pos_weight', 1) # Use tuned weight
print("Final model parameters:"); print(final_params)
final_model = lgb.LGBMClassifier(**final_params)
print(f"Training final model (early stopping based on Test pAUC)...")
start_final_time = time.time()
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                eval_metric=lgbm_pauc_metric if calculate_isic_pauc else 'auc', # Use pAUC callback
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=100)])
end_final_time = time.time()
print(f"Final model training took: {end_final_time - start_final_time:.2f} seconds.")
print(f"Best iteration found: {final_model.best_iteration_}")


# --- Final Evaluation ---
print("\n--- Evaluating final model on Test Set ---")
# (Keep Final Evaluation as in lgbm_optuna_simple_pAUC_Opt_EmbedOnly.py)
try:
    y_pred_proba_test = final_model.predict_proba(X_test)[:, 1]
    y_pred_binary_test = (y_pred_proba_test > 0.5).astype(int)
    final_accuracy = accuracy_score(y_test, y_pred_binary_test)
    final_auc = roc_auc_score(y_test, y_pred_proba_test)
    print(f"Test Set AUC: {final_auc:.6f}")
    if calculate_isic_pauc:
        # Get IDs from dfTestFinal (merged test data) to match y_test
        test_ids = dfTestFinal[ID_COL].tolist() # Ensure IDs align with y_test
        solution_df_test = pd.DataFrame({ID_COL: test_ids, TARGET_COL: y_test})
        submission_df_test = pd.DataFrame({ID_COL: test_ids, TARGET_COL: y_pred_proba_test})
        final_pauc = pauc_scorer_func(y_test, y_pred_proba_test) # Use direct value func
        print(f"Test Set pAUC (min_tpr=0.80): {final_pauc:.6f}")
    else: final_pauc = np.nan; print("pAUC calculation skipped.")
    print(f"Test Set Accuracy (0.5 thresh): {final_accuracy:.4f}")
    cm = confusion_matrix(y_test, y_pred_binary_test); print("Test Set Confusion Matrix:"); print(cm)
    plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.title(f'Final Test CM ()'); plt.ylabel('Actual Label'); plt.xlabel('Predicted Label'); plt.show()
except Exception as e: print(f"Error during final evaluation: {e}"); import traceback; traceback.print_exc()






