import os
import yaml
import argparse
import torch
from pipeline.data_loader import get_dataloaders
from models.resnet import ISICResNet
from training.trainer import Trainer
from utils.visualization import plot_training_history

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(config)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create model
    model = ISICResNet(config)
    print(f"Created {config['model']['architecture']} model with {config['model']['num_classes']} output classes")
    
    # Train model
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.train()
    
    # Plot and save training history
    plot = plot_training_history(history)
    plot.savefig(os.path.join(config['paths']['log_dir'], 'training_history.png'))
    print(f"Training history saved to {config['paths']['log_dir']}/training_history.png")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC Skin Cancer Detection Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                        help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.config)
