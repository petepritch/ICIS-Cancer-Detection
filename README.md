# Multimodel Transformer Framework for Skin Cancer Detection

![Model Architecture](assets/model_architecture_cropped.png)

A novel multimodal approach to skin cancer detection that combines image data and patient metadata from the SLICE-3D dataset, addressing extreme class imbalance through innovative techniques.

## Overview

This repository contains the implementation of a multimodal transformer framework for early skin cancer detection as described in our research paper. Our approach:

- Utilizes both image data and patient metadata from the SLICE-3D dataset
- Addresses extreme class imablance (1:1000) ratio using 3 key strategies:

  - Oversampling malignant tumors
  - Aysmmetric loss function
  - Syntheric data techniques (SMOTE, diffusion models, and GANs)

- Implements an ensemble architecture cobining:

  - CNN models (EfficientNet, ResNet) for image processing
  - Gradient boosting algorithms (XGBoost, LightGBM, CatBoost) for metadata classification
  - Mutual attention blocks for cross-modal feature interaction

Our experiments have demonstrated promising results, achieving up to 82% true positive rate while maintaining clinically acceptable false positive rates.

## Dataset

We use the SLICE-3D dataset, which consists of 401,059 cropped 15mm x 15mm images extracted from 3D Total Body Photos (3D-TBP). Each image has:

- 29 morphological features
- 14 metadata features describing lesion location an patient data
- Only 393 malignant cases (0.098%) among the entire dataset

## Key Features

### Preprocessing Pipeline

- Image preprocessing: Resizing to 224x224 and normalizing RGB channels
- Metadata preprocessing: Target encoding for categorical features and mode imputation for missing values

### Class Imbalance Handling

- Oversampling malignant tumors during training
- Weighting the loss of malignant tumors differently (configurable LW parameter)
- Generating synthetic malignant samples using SMOTE

### Model Architecture

- **Image Models**: Various CNN architectures (EfficientNet, ResNet) and Vision Transformers
- **Metadata Models**: Gradient boosting algorithms optimized for tabular data
- **Multimodal Fusion**: Novel mutual attention blocks connecting image and metadata features
- **Ensemble Approach**: Combining models with different sensitivity levels ("skeptical" and "naive" configurations)

### Evlaution Metrics

- Partial Area Under ROC Curve (pAUC) with TPR range of 0.8-1.0
- True Positive Rate (TPR) and False Positive Rate (FPR) analysis
- Accuracy benchmarking against Kaggle competition entries

<!-- ## Project structure

```plaintext
tumor_classification/
├── README.md
├── config/
│   ├── model_config.yaml         # Vision model architecture settings
│   ├── preprocessing_config.yaml # Data preprocessing settings
│   ├── training_config.yaml      # Training parameters for vision models
│   ├── lgbm_config.yaml          # LGBM training parameters
│   └── greatlakes_paths.yaml     # HPC-specific paths
├── embeddings/                   # Directory to store embedding CSVs
├── logs/                         # Log files saved here
├── scripts/
│   ├── train_model.py            # Vision model training script
│   ├── extract_embeddings.py     # Script to extract embeddings
│   ├── train_lgbm.py             # LGBM training script
│   ├── setup_greatlakes.sh       # HPC environment setup script
│   └── slurm/                    # SLURM job scripts
│       ├── train_vision.sh
│       ├── vision_array.sh
│       ├── extract_embeddings.sh
│       ├── train_lgbm.sh
│       └── run_pipeline.sh
└── src/
    ├── __init__.py
    ├── preprocessing/
    │   ├── __init__.py
    │   └── tumor_preprocessor.py # Data loading and preprocessing
    ├── modeling/
    │   ├── __init__.py
    │   ├── image_model.py        # Core vision model and evaluation
    │   ├── attention.py          # Mutual attention block implementation
    │   ├── lgbm_trainer.py       # LGBM model training pipeline
    |   └── attention_trainer.py  # Mutual attention block training
    ├── synthetic/
    │   ├── __init__.py
    │   └── smote.py              # SMOTE implementation for data generation
    └── utils/
        ├── __init__.py
        ├── utils.py              # General utility functions
        ├── embeddings.py         # Embedding extraction utilities
        ├── metrics.py            # Custom evaluation metrics (pAUC)
        └── slurm_utils.py        # HPC-specific utilities
``` -->

## Installation

```bash
# Clone the repository
git clone https://github.com/petepritch/ICIS-Cancer-Detection.git
cd skincare-ai

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Results

- True Positive Rate: Up to 82% for malignant lesions
- False Positive Rate: Maintained within clinically acceptable limits
- pAUC (TPR range 0.8-1.0): Competitive with top Kaggle competition entries

## Authors

- Neil Ash (nfash@umich.edu)
- Pete Pritchard (petep@umich.edu)
- Andrew Jones (drej@umich.edu)
- Zekai Xu (xuzekai@umich.edu)
- Rachel Gu (rachgu@umich.edu)