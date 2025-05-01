# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:56:28 2025

@author: Drew
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from utils import *
import pandas as pd
import numpy as np
class ImageModel:
    def __init__(self, model, criterion, optimizer, scaler, device, test_loader=None, model_name="cnn_model"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.test_loader = test_loader
        self.model_name = model_name
        self.epochsTrained = 0
        
    def train(self, train_loader, epochs=1, verbose=True):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            total = correct = 0
            train_tp = train_fn = train_fp = train_tn = 0
    
            for i, (images, labels, metadataIgnore, imgidignore) in enumerate(train_loader):
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
    def evaluate(self, verbose=True, min_tpr_for_pauc=0.80): # Added min_tpr param
        if self.test_loader is None:
             print("Warning: test_loader not provided to ImageModel. Skipping evaluation.")
             return {}

        self.model.eval()
        total_test_loss = 0.0
        # --- Collect probabilities and labels ---
        all_preds_prob_list = []
        all_labels_list = []

        with torch.no_grad():
            expected_items = 4 # Assume test_loader yields 3 items like train_loader
            for batch_data in self.test_loader:
                # --- Data Handling ---
                if len(batch_data) != expected_items:
                    # Handle cases where loader might yield different items (e.g. no metadata)
                    # This depends on how test_loader was created. Let's assume 3 for now.
                    try:
                         images, labels_num, _, _ = batch_data # Try unpacking 4
                    except ValueError:
                         images, labels_num = batch_data # Fallback to 2 if needed
                else:
                     images, labels_num, _, _ = batch_data # Unpack 4, ignore metadata, ignore img id

                # Convert labels (ensure consistency with how labels are handled in train)
                # Using np.array() handles potential variations in input type better
                labels = torch.tensor(np.array(labels_num), dtype=torch.float32).to(self.device)
                # Unsqueeze if loss function expects [B, 1] output
                if self.criterion.__class__.__name__ == 'BCEWithLogitsLoss' and labels.ndim == 1 :
                    labels = labels.unsqueeze(1)

                images = images.to(self.device)

                # --- Forward Pass & Loss ---
                outputs = self.model(images) # Get raw logits
                # Squeeze if model outputs [B, 1] and criterion expects [B] - check criterion needs
                # Usually BCEWithLogitsLoss handles [B, 1] vs [B, 1] or [B] vs [B]
                # Let's assume outputs might be [B, 1], keep consistent with labels
                if outputs.ndim > 1 and outputs.shape[1] == 1 and labels.ndim == 1:
                      outputs = outputs.squeeze(1) # Make output shape match label if needed by criterion
                elif outputs.ndim == 1 and labels.ndim > 1 and labels.shape[1] == 1:
                      labels = labels.squeeze(1) # Make label shape match output if needed


                loss = self.criterion(outputs, labels)
                total_test_loss += loss.item() * images.size(0)

                # --- Store probabilities and labels for metrics ---
                # Apply sigmoid ONLY for storing probabilities
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy() # Squeeze again just in case
                # Ensure probs is iterable even for single item batches
                if probs.ndim == 0:
                    all_preds_prob_list.append(probs.item())
                else:
                    all_preds_prob_list.extend(probs)

                all_labels_list.extend(np.array(labels_num)) # Store original 0/1 labels

        # --- Calculate metrics AFTER the loop ---
        total_samples = len(all_labels_list)
        if total_samples == 0:
            print("Warning: No samples found in test_loader for evaluation.")
            return {}

        avg_loss = total_test_loss / total_samples

        # Convert collected lists to numpy arrays
        all_labels_np = np.array(all_labels_list)
        all_preds_prob_np = np.array(all_preds_prob_list)
        # Derive binary predictions from probabilities for other metrics
        all_preds_binary_np = (all_preds_prob_np > 0.5).astype(int)

        # Calculate standard metrics (TP, TN, etc.)
        tp = ((all_preds_binary_np == 1) & (all_labels_np == 1)).sum()
        fn = ((all_preds_binary_np == 0) & (all_labels_np == 1)).sum()
        fp = ((all_preds_binary_np == 1) & (all_labels_np == 0)).sum()
        tn = ((all_preds_binary_np == 0) & (all_labels_np == 0)).sum()

        accuracy = (tp + tn) / total_samples * 100 if total_samples > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall, TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # TNR

        # --- Calculate pAUC score ---
        pauc_score_val = float('nan') # Default value
        try:
            # Create DataFrames required by the score function
            solution_df = pd.DataFrame({'row_id': range(total_samples), 'target': all_labels_np})
            # IMPORTANT: score function expects prediction scores/probabilities in 'target' column
            submission_df = pd.DataFrame({'row_id': range(total_samples), 'target': all_preds_prob_np})

            # Ensure score function is imported from utils
            pauc_score_val = score(
                solution=solution_df,
                submission=submission_df,
                row_id_column_name='row_id',
                min_tpr=min_tpr_for_pauc
            )
        except Exception as e:
            print(f"Could not calculate pAUC score: {e}")

        # --- Print and Return Results ---
        if verbose:
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
            print(f"True Positives: {tp}")
            print(f"False Negatives: {fn}")
            print(f"False Positives: {fp}")
            print(f"True Negatives: {tn}")
            print(f"Sensitivity (TPR): {sensitivity:.4f}")
            print(f"Specificity (TNR): {specificity:.4f}")
            print(f"pAUC (min_tpr={min_tpr_for_pauc:.2f}): {pauc_score_val:.4f}") # Print pAUC

        # Return dictionary including pAUC
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, # Keep original metrics
            'sensitivity': sensitivity,
            'specificity': specificity,
            'pauc': pauc_score_val # Add pAUC score
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
        """
        Saves the model weights, optimizer state, and scaler state 
        into a checkpoint file.
        """
        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            # You can add other info if you like
            # e.g. 'epoch': current_epoch, 'loss': current_loss, etc.
        }
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath):
        """
        Loads the model weights, optimizer, and scaler state from a checkpoint.
        NOTE: You must ensure 'self.model' has the same architecture as when saved.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_name = checkpoint['model_name']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
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