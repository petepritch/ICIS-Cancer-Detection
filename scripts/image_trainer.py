import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import logging
from torch.cuda.amp import GradScaler
import time
from datetime import datetime
from pathlib import Path

from src.preprocessing.preprocessor import TumorPreprocessor, create_transforms
from models.image_model import (
    ImageModel,
    create_model,
    create_criterion,
    create_optimizer,
    create_scheduler,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer class that manages the entire training pipeline"""

    def __init__(self, config_path):
        """
        Initialize the trainer with config

        Args:
            config_path: Path to directory containing config files
        """
        self.config_path = config_path
        self.configs = self._load_configs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self._set_seeds()

    def _load_configs(self):
        """Load all configuration files"""
        config_files = {
            "model": os.path.join(self.config_path, "model_config.yaml"),
            "preprocessing": os.path.join(
                self.config_path, "preprocessing_config.yaml"
            ),
            "training": os.path.join(self.config_path, "training_config.yaml"),
        }

        configs = {}

        for key, path in config_files.items():
            with open(path, "r") as f:
                configs[key] = yaml.safe_load(f)

        return configs

    def _set_seeds(self, seed=42):
        """Set seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_data(self):
        """Set up data preprocessing and create dataloaders"""
        train_transform, test_transform = create_transforms(
            {**self.configs["model"], **self.configs["preprocessing"]}
        )

        preprocessor = TumorPreprocessor(
            csv_file=self.configs["preprocessing"]["data"]["csv_file"],
            img_dir=self.configs["preprocessing"]["data"]["img_dir"],
            train_transform=train_transform,
            test_transform=test_transform,
            batch_size=self.configs["training"]["training"]["batch_size"],
            num_workers=self.configs["training"]["training"]["num_workers"],
            labeling_mode=self.configs["training"]["loss"]["labeling_strategy"],
        )

        self.train_loader, self.test_loader = preprocessor.get_dataloaders()

        logger.info(
            f"Created dataloaders with batch size {self.configs['training']['training']['batch_size']}"
        )

        return self.train_loader, self.test_loader

    def setup_model(self):
        """Set up the model, criterion, optimizer, and scheduler"""
        self.model = create_model(self.configs["model"])
        self.model.to(self.device)
        self.criterion = create_criterion(self.configs["training"])
        self.criterion.to(self.device)
        self.optimizer = create_optimizer(self.configs["training"], self.model)
        self.scheduler = create_scheduler(self.configs["training"], self.optimizer)
        self.scaler = GradScaler()
        self.image_model = ImageModel(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scaler=self.scaler,
            device=self.device,
            test_loader=self.test_loader,
            model_name=self.configs["model"]["model"]["architecture"],
        )

        logger.info(
            f"Model setup complete. Using {self.configs['model']['model']['architecture']}"
        )

        return self.image_model

    def train(self, output_dir="models"):
        """Run the full training pipeline with warmup, scheduler, and early stopping"""
        os.makedirs(output_dir, exist_ok=True)

        max_epochs = self.configs["training"]["training"]["max_epochs"]
        patience_limit = self.configs["training"]["training"]["patience_limit"]
        warmup_epochs = self.configs["training"]["training"]["warmup_epochs"]
        initial_lr = self.configs["training"]["training"]["initial_learning_rate"]
        eta_min = self.configs["training"]["scheduler"]["eta_min"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.configs["model"]["model"]["architecture"]
        best_model_path = os.path.join(output_dir, f"{model_name}_{timestamp}_best.pth")

        logger.info(
            f"Starting training pipeline: {max_epochs} epochs, {patience_limit} patience"
        )

        best_metric = -1.0
        patience_counter = 0

        for epoch in range(max_epochs):
            logger.info(f"Starting epoch {epoch+1}/{max_epochs}")
            epoch_start_time = time.time()

            if epoch < warmup_epochs:
                warmup_lr_start = eta_min
                lr_factor = (epoch + 1) / warmup_epochs
                current_lr = warmup_lr_start + lr_factor * (
                    initial_lr - warmup_lr_start
                )

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_lr

                logger.info(
                    f"Warmup epoch {epoch+1}/{warmup_epochs}, LR: {current_lr:.6f}"
                )
            else:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Cosine annealing phase, LR: {current_lr:.6f}")

            self.image_model.train(self.train_loader, epochs=1, verbose=True)

            metrics = self.image_model.evaluate(verbose=True)
            val_pauc = metrics.get("pauc", -1.0)

            if val_pauc > best_metric:
                best_metric = val_pauc
                logger.info(f"Metric improved to {best_metric:.4f}. Saving model...")
                self.image_model.save_model(best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(
                    f"Metric did not improve. Best: {best_metric:.4f}. Patience: {patience_counter}/{patience_limit}"
                )

            if patience_counter >= patience_limit:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")

        logger.info("Training completed.")

        logger.info(f"Loading best model from {best_model_path}")
        self.image_model.load_model(best_model_path)
        final_metrics = self.image_model.evaluate(verbose=True)

        plot_path = os.path.join(output_dir, f"{model_name}_{timestamp}_roc.png")
        self.image_model.plot_roc_curve(save_path=plot_path)

        metrics_path = os.path.join(
            output_dir, f"{model_name}_{timestamp}_metrics.yaml"
        )
        with open(metrics_path, "w") as f:
            yaml.dump(final_metrics, f)

        logger.info(f"Training pipeline completed. Best pAUC: {best_metric:.4f}")

        return best_model_path, final_metrics
