# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:56:28 2025

@author: Drew
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
class ImageModel:
    def __init__(self, model, criterion, optimizer, scaler, device, test_loader, model_name="cnn_model"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.test_loader = test_loader
        self.model_name = model_name
        
    def train(self, train_loader, epochs=1, verbose=True):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            total = correct = 0
            train_tp = train_fn = train_fp = train_tn = 0
    
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.float().to(self.device)
    
                self.optimizer.zero_grad()
                with autocast():
                    outputs = self.model(images).squeeze()
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
    
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                running_loss += loss.item()
    
                train_tp += ((preds == 1) & (labels == 1)).sum().item()
                train_fn += ((preds == 0) & (labels == 1)).sum().item()
                train_fp += ((preds == 1) & (labels == 0)).sum().item()
                train_tn += ((preds == 0) & (labels == 0)).sum().item()
    
                if verbose and (i + 1) % 50 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
            if verbose:
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = 100 * correct / total
                sensitivity = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0.0
                specificity = train_tn / (train_tn + train_fp) if (train_tn + train_fp) > 0 else 0.0
    
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
                print(f"Training True Positives: {train_tp}")
                print(f"Training False Negatives: {train_fn}")
                print(f"Training False Positives: {train_fp}")
                print(f"Training True Negatives: {train_tn}")
                print(f"Training Sensitivity (TPR): {sensitivity:.4f}")
                print(f"Training Specificity (TNR): {specificity:.4f}")


#returns a dictionary of metrics, and displays those metrics if verbose=True which it does by default
    def evaluate(self, verbose=True):
        self.model.eval()
        total_test_loss = 0.0
        total_correct = total_samples = 0
        tp = fp = tn = fn = 0
    
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.float().to(self.device)
    
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels)
                total_test_loss += loss.item() * images.size(0)
    
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
    
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
    
        avg_loss = total_test_loss / total_samples
        accuracy = 100 * total_correct / total_samples
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
        if verbose:
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
            print(f"True Positives: {tp}")
            print(f"False Negatives: {fn}")
            print(f"False Positives: {fp}")
            print(f"True Negatives: {tn}")
            print(f"Sensitivity (TPR): {sensitivity:.4f}")
            print(f"Specificity (TNR): {specificity:.4f}")
    
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'tn': tn,
            'sensitivity': sensitivity,
            'specificity': specificity
        }


    def predict(self, images):
        """Returns binary predictions (0 or 1) for given images."""
        self.model.eval()
        images = images.to(self.device)
    
        with torch.no_grad():
            outputs = self.model(images).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float()
    
        return preds.cpu()

    def save_model(self, filepath):
        """Saves the current state of the model to a file."""
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Loads the model weights from a file."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        
    def extract_embeddings(self, images):
        """Extracts embeddings from the penultimate layer of the CNN model."""
        self.model.eval()
        images = images.to(self.device)
    
        with torch.no_grad():
            # Temporarily remove the final classifier layer
            if hasattr(self.model, 'classifier'):
                features = self.model.features(images)
                embeddings = self.model.avgpool(features)
                embeddings = torch.flatten(embeddings, 1)
            elif hasattr(self.model, 'fc'):
                embeddings = self.model.forward(images)
                embeddings = embeddings.view(embeddings.size(0), -1)
            else:
                raise NotImplementedError("Embedding extraction not defined for this architecture.")
    
        return embeddings.cpu()