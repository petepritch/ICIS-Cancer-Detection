import os
import torch
import shutil

def save_checkpoint(state, is_best, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
    torch.save(state, checkpoint_path)
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        shutil.copyfile(checkpoint_path, best_path)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint['epoch'], checkpoint['best_val_loss']