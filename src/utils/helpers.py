# General data science libraries
import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Load Kaggle token
from google.colab import userdata
import os

# Load Image
import h5py
from PIL import Image
from joblib import Parallel, delayed # Parallel processing
from io import BytesIO # Read binary image file

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# Process data on GPU
import cudf
import cupy as cp

# Scikit-learn
from sklearn.pipeline import Pipeline # General transformation pipeline
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split

# XGBoost
import xgboost as xgb

# LightGBM
import lightgbm as lgb

# CatBoost
import catboost as cb

# Oversampling
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Load pretrained vision models like `efficientnet`
import timm

# Load Huggingface model
import transformers

# Unzip .zip file
import zipfile

# Progress bar
import tqdm

__all__ = [
    # General data science libraries
    "pd", "np", "pl", "plt", "sns",

    # Load Kaggle token
    "userdata", "os",

    # Load Image
    "h5py", "Image", "Parallel", "delayed", "BytesIO",

    # PyTorch
    "torch", "Dataset", "DataLoader", "transforms", "models", "nn", "optim",

    # Process data on GPU
    "cudf", "cp",

    # Scikit-learn
    "Pipeline", "TargetEncoder", "OneHotEncoder", "MinMaxScaler",
    "SimpleImputer", "ColumnTransformer", "TransformerMixin", "BaseEstimator",

    # XGBoost
    "xgb",

    # LightGBM
    "lgb",

    # CatBoost
    "cb",

    # Oversampling
    "RandomOverSampler", "SMOTE",

    # Load pretrained vision models like `efficientnet`
    "timm",

    # Load Huggingface model
    "transformers",

    # Unzip .zip file
    "zipfile",

    # Progress bar
    "tqdm",
]