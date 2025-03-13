import yaml
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pipeline.dataset import ISICDataset
from pipeline.transforms import get_train_transforms, get_val_transforms

def test_preprocessing_pipeline(config_path):
    """Test the preprocessing pipeline with a small batch of data."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset with transforms
    train_transforms = get_train_transforms(config['data']['img_size'])
    dataset = ISICDataset(
        annotations_file=config['data']['annotations_file'],
        img_dir=config['data']['img_dir'],
        transform=train_transforms,
        file_ext=config['data']['file_ext']
    )
    
    # Test a few samples
    num_samples = min(5, len(dataset))
    
    plt.figure(figsize=(15, 4 * num_samples))
    for i in range(num_samples):
        try:
            # Get a sample
            image, label = dataset[i]
            
            # Convert tensor to numpy for visualization
            img_np = image.numpy().transpose(1, 2, 0)
            
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            plt.subplot(num_samples, 1, i+1)
            plt.imshow(img_np)
            plt.title(f"Sample {i}, Label: {label}")
            plt.axis('off')
            
            print(f"Successfully processed sample {i}")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    
    plt.tight_layout()
    plt.savefig('preprocessing_test.png')
    print("Test results saved to preprocessing_test.png")

if __name__ == "__main__":
    test_preprocessing_pipeline("configs/config.yaml")