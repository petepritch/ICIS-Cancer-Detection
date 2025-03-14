import os
import time
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from utils.metrics import calculate_metrics
from utils.checkpointing import save_checkpoint

class Trainer:

    def __init__(self, model, train_loader, val_loader, config):
        """Initialize the trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Use Cuda if available
        self.device = torch.device("cude" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        self.criterion = CrossEntropyLoss()

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training']['lr_scheduler']['factor'],
            patience=config['training']['lr_scheduler']['patience'],
            verbose=True
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self):
        """Single epoch loop."""
        self.model.train()
        epoch_loss= 0.0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Foward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pas
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        return epoch_loss / len(self.train_loader.dataset)
    
    def validate(self):
        """Validate model."""
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Foward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predictions = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))
        metrics['loss'] = val_loss / len(self.val_loader.dataset)

        return metrics
    
    def train(self):
        """Train for specifed number of epochs."""
        print(f"Training on device: {self.device}")

        for epoch in range(self.config['training']['epochs']):
            start_time = time.time()

            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)

            metrics = self.validate()
            val_loss = metrics['loss']

            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(metrics['accuracy'])
            self.history['val_precision'].append(metrics['precision'])
            self.history['val_recall'].append(metrics['recall'])
            self.history['val_f1'].append(metrics['f1'])

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Early stopping check
            if val_loss < self.best_val_loss - self.config['training']['early_stopping']['min_delta']:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=True, checkpoint_dir=self.config['paths']['checkpoint_dir'])
            
            else:
                self.patience_counter += 1
                
                # Save regular checkpoint
                if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'best_val_loss': self.best_val_loss,
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False, checkpoint_dir=self.config['paths']['checkpoint_dir'])
            

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Time: {epoch_time:.2f}s - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {metrics['accuracy']:.4f} - "
                f"Val F1: {metrics['f1']:.4f}")
            
            if self.patience_counter >= self.config['training']['early_stopping']['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
        return self.history
