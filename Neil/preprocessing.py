# preprocessing.py

import os
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from tumor_dataset import TumorDataset
from sklearn.model_selection import train_test_split

class TumorPreprocessor:
    def __init__(self, csv_file, img_dir, train_transform, test_transform,
                 batch_size=64, num_workers=15):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform
        
        # Load the metadata
        self.df = pd.read_csv(self.csv_file)
    
    def get_dataloaders(self, test_size=0.2, random_state=42, OW=100):
        # Create indices and perform stratified train/test split
        indices = self.df.index.tolist()
        labels = self.df['target'].values
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
        
        # Create separate dataset instances for train and test using their transforms
        train_dataset = TumorDataset(
            csv_file=self.csv_file,
            img_dir=self.img_dir,
            transform=self.train_transform
        )
        test_dataset = TumorDataset(
            csv_file=self.csv_file,
            img_dir=self.img_dir,
            transform=self.test_transform
        )
        
        # Create subsets based on indices
        train_subset = Subset(train_dataset, train_idx)
        test_subset = Subset(test_dataset, test_idx)
        
        # Create a weighted sampler for the training set (if needed)
        train_df = self.df.loc[train_idx]
        train_weights = np.where(train_df['target'] == 1, 1.0, 1/OW).tolist()
        train_sampler = WeightedRandomSampler(train_weights,
                                              num_samples=len(train_weights),
                                              replacement=True)
        
        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, pin_memory=True,
                                  sampler=train_sampler)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers,
                                 pin_memory=True)
        
        return train_loader, test_loader