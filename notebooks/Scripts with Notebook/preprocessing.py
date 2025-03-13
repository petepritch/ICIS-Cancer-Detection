# preprocessing.py

import os
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from tumor_dataset import TumorDataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class TumorPreprocessor:
    def __init__(self, csv_file, img_dir, train_transform, test_transform,
                 batch_size=64, num_workers=15,
                 include_metadata=False,
                 metadata_cols=None,
                 undersamplingweight = 1):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform
        
        self.include_metadata = include_metadata
        self.metadata_cols = metadata_cols if metadata_cols else []
        
        # Load the metadata
        self.df = pd.read_csv(self.csv_file)
        
        self.preprocessor = None
        self.train_columns = None
        self.undersamplingweight = undersamplingweight
        
    #this is used specifically for training image models, not gradient boosters, which do processing to metadata
    def get_dataloaders(self, test_size=0.2, random_state=42):
        # Create indices and perform stratified train/test split
        indices = self.df.index.tolist()
        labels = self.df['target'].values
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
        
        # Create separate dataset instances for train and test 

        train_dataset = TumorDataset(
            csv_file=self.csv_file,
            img_dir=self.img_dir,
            transform=self.train_transform,
            include_metadata=self.include_metadata,
            metadata_cols=self.metadata_cols
        )
        test_dataset = TumorDataset(
            csv_file=self.csv_file,
            img_dir=self.img_dir,
            transform=self.test_transform,
            include_metadata=self.include_metadata,
            metadata_cols=self.metadata_cols
        )
        
        # Create subsets based on indices
        train_subset = Subset(train_dataset, train_idx)
        test_subset = Subset(test_dataset, test_idx)
        
        # Weighted sampler if needed
        train_df = self.df.loc[train_idx]
        train_weights = np.where(train_df['target'] == 1, 1.0, self.undersamplingweight).tolist()
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


    
    # for training a gradient booster with processed metadata
   
    def get_gb_dataloader(self, test_size=0.2, random_state=42):
        """
        Similar logic to get_dataloaders, but we:
          1) Split train/test
          2) Create TumorDataset for train & test
          3) 'clean' each dataset's metadata by calling set_metadata
          4) Return final DataLoaders that yield (image, label, processed_meta)
        """
        indices = self.df.index.tolist()
        labels = self.df['target'].values
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
        
        # create dataset objects from the CSV
        train_dataset = TumorDataset(
            csv_file=self.csv_file,
            img_dir=self.img_dir,
            transform=self.train_transform,
            include_metadata=True,     
            metadata_cols=[]           
        )
        test_dataset = TumorDataset(
            csv_file=self.csv_file,
            img_dir=self.img_dir,
            transform=self.test_transform,
            include_metadata=True,
            metadata_cols=[]
        )
        

        
        # weighted sampler
        train_df_sub = self.df.loc[train_idx]
        train_weights = np.where(train_df_sub['target'] == 1, 1.0, self.undersamplingweight).tolist()
        train_sampler = WeightedRandomSampler(
            train_weights,
            num_samples=len(train_weights),
            replacement=True
        )
        
        # clean the metadata for each portion and set it on the dataset
        train_df_raw = train_dataset.metadata.loc[train_idx].copy()
        test_df_raw = test_dataset.metadata.loc[test_idx].copy()
        
        train_processed = self.fit_transform(train_df_raw)
        test_processed = self.transform(test_df_raw)
        
        # now attach the processed DataFrames to each dataset
        train_dataset.set_metadata(train_processed)

        test_dataset.set_metadata(test_processed)

        
        # create final DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader
    


    def fit_transform(self, df):
        """Preprocess training data and store transformations."""
        df = df.copy()
        df = self._drop_irrelevant_columns(df)
        df = self._drop_train_only_columns(df)
        categorical_cols, numerical_cols = self._identify_column_types(df)
        
        # Define preprocessing pipelines
        numerical_pipeline = Pipeline([
            # ("imputer", SimpleImputer(strategy="median")),
            ("imputer", KNNImputer()),
            ("scaler", StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        self.preprocessor = ColumnTransformer([
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])
        
        transformed_data = self.preprocessor.fit_transform(df)
        
        cat_feature_names = self.preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
        all_columns = numerical_cols + list(cat_feature_names)
        
        df_processed = pd.DataFrame(transformed_data, columns=all_columns)
        df_processed["isic_id"] = df["isic_id"].values
        df_processed["target"] = df["target"].values
        
        self.train_columns = df_processed.columns  # Store train columns
        return df_processed

    def transform(self, df):
        """Preprocess test data using stored transformations from training."""
        df = df.copy()
        df = self._drop_irrelevant_columns(df)
        
        transformed_data = self.preprocessor.transform(df)
        cat_feature_names = self.preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out()
        all_columns = self.train_columns[:-2]  # Exclude 'isic_id' and 'target'
        
        df_processed = pd.DataFrame(transformed_data, columns=all_columns)
        df_processed["isic_id"] = df["isic_id"].values
        df_processed["target"] = df["target"].values
        
        # Align test dataset with train columns
        df_processed = self._align_train_test_columns(df_processed)
        return df_processed

    def _drop_irrelevant_columns(self, df):
        """Remove unnecessary columns."""
        return df.drop(columns=['patient_id','image_type', 'tbp_tile_type', 'attribution', 'copyright_license'], errors="ignore")
    
    
    def _drop_train_only_columns(self,df):
        """Remove columns that are present only in the train set and not in the test set"""
        drop_train_only_columns = [
            'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',
            'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence'
            ]
        return df.drop(columns=drop_train_only_columns,errors='ignore')    
    
    
    def _identify_column_types(self, df):
        """Identify categorical and numerical columns, excluding 'isic_id' and 'target'."""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != "isic_id"]
        numerical_cols = [col for col in numerical_cols if col != "target"]
        return categorical_cols, numerical_cols    
    
    def _align_train_test_columns(self, df):
        """Ensure test data has the same columns as train data."""
        train_cols = list(self.train_columns)
        #train_cols.remove('target')
        missing_cols = set(train_cols) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        return df[train_cols]    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    