import logging
import os
import random
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def setup_logging(log_file=None, log_level=logging.INFO):
    """Set up logging configuration"""
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """Save configuration to YAML file"""
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(history, save_path=None):
    """Plot training history metrics"""
    plt.figure(figsize=(12, 4))

    # Plot training metrics
    if "loss" in history:
        plt.subplot(1, 3, 1)
        plt.plot(history["loss"], label="Train")
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="Validation")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.legend()

    if "accuracy" in history:
        plt.subplot(1, 3, 2)
        plt.plot(history["accuracy"], label="Train")
        if "val_accuracy" in history:
            plt.plot(history["val_accuracy"], label="Validation")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()

    if "pauc" in history:
        plt.subplot(1, 3, 3)
        plt.plot(history["pauc"], label="pAUC")
        plt.title("Partial AUC (TPR>0.8)")
        plt.xlabel("Epoch")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


class DiscreteRotation:
    """A transform that applies one of a few discrete rotations to an image"""

    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        if angle == 0:
            return img
        elif angle == 90:
            return img.transpose(1, 0).flip(0)
        elif angle == 180:
            return img.flip(0).flip(1)
        elif angle == 270:
            return img.transpose(1, 0).flip(1)
