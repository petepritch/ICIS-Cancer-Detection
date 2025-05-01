import torch
import collections # To create an ordered dict if needed, though regular dict often works
from sklearn.metrics import roc_curve, auc # Removed roc_auc_score as it's commented out in the function
import pandas as pd
import numpy as np
class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80) -> float:
    # ... (rest of your score function code) ...
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    if not pd.api.types.is_numeric_dtype(submission.values):
         raise ParticipantVisibleError('Submission target column must be numeric')
    v_gt = abs(np.asarray(solution.values)-1)
    v_pred = -1.0*np.asarray(submission.values) # Requires submission to be probabilities/scores
    max_fpr = abs(1-min_tpr)
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)
    return(partial_auc)
# --- Assuming encoder_m1 and encoder_m2 are instantiated ---
# encoder_m1 = ImageEncoder(...)
# encoder_m2 = ImageEncoder(...)

# --- Function to load filtered weights ---
def load_vit_backbone_weights(encoder_module, checkpoint_path, device):
    """
    Loads weights from an ImageModel checkpoint into the ViT backbone
    of an ImageEncoder module, ignoring the final 'head' layers.

    Args:
        encoder_module (ImageEncoder): The encoder instance to load weights into.
        checkpoint_path (str): Path to the .pth checkpoint file.
        device: The torch device ('cuda' or 'cpu').
    """
    try:
        # Load the full checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        saved_state_dict = checkpoint['model_state_dict']

        # Create a new dictionary to store only backbone weights
        backbone_state_dict = collections.OrderedDict() # Or just {}
        print(f"Filtering state dict from {checkpoint_path}...")
        keys_loaded = 0
        keys_skipped = 0
        for key, value in saved_state_dict.items():
            if key.startswith('head.'):
                # Skip keys belonging to the old classification head
                # print(f"  Skipping head key: {key}")
                keys_skipped += 1
                continue
            # Add backbone keys (these should match the keys in encoder_module.vit)
            backbone_state_dict[key] = value
            keys_loaded += 1

        print(f"  Loaded {keys_loaded} backbone keys, skipped {keys_skipped} head keys.")

        # Load the filtered state dict into the ViT sub-module
        # Use strict=False initially to be safe, it will warn about missing keys (the head)
        # but shouldn't error if backbone keys match. If strict=True works, even better.
        incompatible_keys = encoder_module.vit.load_state_dict(backbone_state_dict, strict=False)
        print(f"  load_state_dict incompatible keys report: {incompatible_keys}")
        print(f"Successfully loaded backbone weights into encoder from {checkpoint_path}")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
    except KeyError as e:
        print(f"Error: Key missing in checkpoint {checkpoint_path}: {e}")
    except Exception as e:
        print(f"An error occurred loading checkpoint {checkpoint_path}: {e}")