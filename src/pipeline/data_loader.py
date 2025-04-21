import torch
from torch.utils.data import DataLoader, random_split
from .dataset import ISICDataset
from .transforms import get_train_transforms, get_val_transforms

def get_dataloaders(config):
    """Create train and validation dataloaders."""
    # Create dataset with no transforms initially
    full_dataset = ISICDataset(
        annotations_file=config['data']['annotations_file'],
        img_dir=config['data']['img_dir'],
        file_ext=config['data']['file_ext']
    )
    
    val_size = int(len(full_dataset) * config['data']['val_split'])
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )
    
    train_transforms = get_train_transforms(config['data']['img_size'])
    val_transforms = get_val_transforms(config['data']['img_size'])
    
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader