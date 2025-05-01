import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import yaml
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class ImageModel:
    """Class for training and evaluating image-based tumor models"""

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scaler,
        device,
        test_loader=None,
        model_name="tumor_model",
    ):
        """
        Initialize the ImageModel class

        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: PyTorch optimizer
            scaler: GradScaler for mixed precision training
            device: PyTorch device
            test_loader: DataLoader for test set (optional)
            model_name: Model name for saving checkpoints
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.test_loader = test_loader
        self.model_name = model_name
        self.best_pauc = -1

    def train(self, train_loader, epochs=1, verbose=True):
        """
        Train the model for specified number of epochs

        Args:
            train_loader: DataLoader for training
            epochs: Number of epochs to train
            verbose: Whether to print progress
        """
        self.model.train()
        total_steps = len(train_loader)

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            progress_bar = (
                tqdm(enumerate(train_loader), total=total_steps)
                if verbose
                else enumerate(train_loader)
            )

            for i, (images, labels) in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass with gradient scaling
                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels.view(-1, 1))

                # Backward and optimize with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()

                # Calculate accuracy and confusion matrix metrics
                predicted = torch.sigmoid(outputs).detach() > 0.5
                total += labels.size(0)
                correct += (predicted.view(-1) == labels).sum().item()

                # Update confusion matrix
                true_positives += (
                    ((predicted.view(-1) == 1) & (labels == 1)).sum().item()
                )
                false_positives += (
                    ((predicted.view(-1) == 1) & (labels == 0)).sum().item()
                )
                true_negatives += (
                    ((predicted.view(-1) == 0) & (labels == 0)).sum().item()
                )
                false_negatives += (
                    ((predicted.view(-1) == 0) & (labels == 1)).sum().item()
                )

                # Print batch progress
                if verbose and (i + 1) % 50 == 0:
                    print(
                        f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}"
                    )

            # Calculate epoch metrics
            epoch_loss = running_loss / total_steps
            epoch_acc = 100 * correct / total

            # Calculate sensitivity and specificity
            sensitivity = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            specificity = (
                true_negatives / (true_negatives + false_positives)
                if (true_negatives + false_positives) > 0
                else 0
            )

            if verbose:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
                )
                print(f"Training True Positives: {true_positives}")
                print(f"Training False Negatives: {false_negatives}")
                print(f"Training False Positives: {false_positives}")
                print(f"Training True Negatives: {true_negatives}")
                print(f"Training Sensitivity (TPR): {sensitivity:.4f}")
                print(f"Training Specificity (TNR): {specificity:.4f}")

    def evaluate(self, test_loader=None, verbose=True):
        """
        Evaluate the model on test data

        Args:
            test_loader: DataLoader for testing (uses self.test_loader if None)
            verbose: Whether to print results

        Returns:
            Dictionary of metrics including loss, accuracy, sensitivity, specificity, and pAUC
        """
        if test_loader is None:
            if self.test_loader is None:
                raise ValueError("No test_loader provided for evaluation")
            test_loader = self.test_loader

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_outputs = []
        all_labels = []

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels.view(-1, 1))

                total_loss += loss.item()

                predicted = torch.sigmoid(outputs) > 0.5
                total += labels.size(0)
                correct += (predicted.view(-1) == labels).sum().item()

                # Store outputs and labels for AUC calculation
                all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update confusion matrix
                true_positives += (
                    ((predicted.view(-1) == 1) & (labels == 1)).sum().item()
                )
                false_positives += (
                    ((predicted.view(-1) == 1) & (labels == 0)).sum().item()
                )
                true_negatives += (
                    ((predicted.view(-1) == 0) & (labels == 0)).sum().item()
                )
                false_negatives += (
                    ((predicted.view(-1) == 0) & (labels == 1)).sum().item()
                )

        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total

        # Calculate sensitivity and specificity
        sensitivity = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        specificity = (
            true_negatives / (true_negatives + false_positives)
            if (true_negatives + false_positives) > 0
            else 0
        )

        # Calculate partial AUC (pAUC) focusing on high sensitivity region (TPR > 0.8)
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)

        try:
            fpr, tpr, _ = roc_curve(all_labels, all_outputs)
            # Partial AUC (min_tpr=0.8)
            min_tpr = 0.8
            pauc = self._calculate_pauc(fpr, tpr, min_tpr)
        except Exception as e:
            logger.error(f"Error calculating AUC: {e}")
            pauc = float("nan")

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy / 100,  # Convert to decimal
            "sensitivity": sensitivity,
            "specificity": specificity,
            "pauc": pauc,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        }

        if verbose:
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
            print(f"True Positives: {true_positives}")
            print(f"False Negatives: {false_negatives}")
            print(f"False Positives: {false_positives}")
            print(f"True Negatives: {true_negatives}")
            print(f"Sensitivity (TPR): {sensitivity:.4f}")
            print(f"Specificity (TNR): {specificity:.4f}")
            print(f"pAUC (min_tpr={min_tpr:.2f}): {pauc:.4f}")

        return metrics

    def _calculate_pauc(self, fpr, tpr, min_tpr=0.8):
        """Calculate partial AUC (area under ROC curve) for high sensitivity region"""
        # Find indices where TPR >= min_tpr
        high_sens_idx = np.where(tpr >= min_tpr)[0]

        if len(high_sens_idx) == 0:
            return 0.0  # No points above threshold

        # Get start index
        start_idx = high_sens_idx[0]
        if start_idx > 0:
            # Interpolate to get exact FPR at min_tpr
            prev_idx = start_idx - 1
            slope = (fpr[start_idx] - fpr[prev_idx]) / (tpr[start_idx] - tpr[prev_idx])
            interp_fpr = fpr[prev_idx] + slope * (min_tpr - tpr[prev_idx])

            # Use these points for pAUC calculation
            high_sens_fpr = np.concatenate(([interp_fpr], fpr[start_idx:]))
            high_sens_tpr = np.concatenate(([min_tpr], tpr[start_idx:]))
        else:
            high_sens_fpr = fpr[start_idx:]
            high_sens_tpr = tpr[start_idx:]

        # Calculate pAUC using trapezoidal rule
        pauc = np.trapz(1 - high_sens_fpr, high_sens_tpr)

        # Normalize by maximum possible area (1 - min_tpr)
        norm_pauc = pauc / (1 - min_tpr)

        return norm_pauc

    def save_model(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath):
        """Load model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler" in checkpoint and checkpoint["scaler"] and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def plot_roc_curve(self, test_loader=None, save_path=None):
        """Plot ROC curve and save if path provided"""
        if test_loader is None:
            if self.test_loader is None:
                raise ValueError("No test_loader provided for ROC curve")
            test_loader = self.test_loader

        self.model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)

                all_outputs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)

        fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
        auc = roc_auc_score(all_labels, all_outputs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.model_name}")
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

        return fpr, tpr, auc


def create_model(config):
    """Create model based on config"""
    architecture = config["model"]["architecture"]
    pretrained = config["model"]["pretrained"]
    num_classes = config["model"]["num_classes"]

    if architecture.startswith("resnetv2"):
        model = timm.create_model(architecture, pretrained=pretrained)
        model.reset_classifier(num_classes=num_classes)
    elif architecture.startswith("efficientnet"):
        model = timm.create_model(architecture, pretrained=pretrained)
        model.reset_classifier(num_classes=num_classes)
    elif architecture.startswith("vit"):
        model = timm.create_model(architecture, pretrained=pretrained)
        model.reset_classifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


def create_criterion(config):
    """Create loss function based on config"""
    loss_type = config["loss"]["type"]
    malignant_weight = config["loss"]["malignant_weight"]

    if loss_type == "bce_with_logits":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([malignant_weight]))
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def create_optimizer(config, model):
    """Create optimizer based on config"""
    lr = config["training"]["initial_learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(config, optimizer):
    """Create learning rate scheduler based on config"""
    scheduler_type = config["scheduler"]["type"]

    if scheduler_type == "cosine":
        T_max = config["scheduler"]["T_max"]
        eta_min = config["scheduler"]["eta_min"]
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
