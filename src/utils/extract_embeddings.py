#!/usr/bin/env python3
"""
Extract embeddings from trained models for LGBM training
"""

import os
import sys
import argparse
import logging
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.utils import setup_logging, set_seeds, load_config
from src.utils.embeddings import load_model_for_embeddings, generate_embeddings_csv
from src.preprocessing.tumor_preprocessor import TumorPreprocessor, create_transforms


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from trained models"
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing configuration files",
    )

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help='Model architecture name (e.g., "vit_small_patch16_224")',
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="embeddings",
        help="Directory to save embeddings",
    )

    parser.add_argument(
        "--csv-file", type=str, required=True, help="Path to metadata CSV file"
    )

    parser.add_argument(
        "--img-dir", type=str, required=True, help="Directory containing images"
    )

    parser.add_argument(
        "--embedding-name",
        type=str,
        default=None,
        help="Name suffix for embedding files (default: model architecture name)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for embedding extraction"
    )

    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/extract_embeddings.log",
        help="Path to log file",
    )

    return parser.parse_args()


def main():
    """Main function for embedding extraction"""
    # Parse arguments
    args = parse_args()

    # Set up logging
    log_dir = os.path.dirname(args.log_file)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_file=args.log_file)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seeds for reproducibility
    set_seeds(args.seed)

    try:
        # Load configs
        model_config = os.path.join(args.config_dir, "model_config.yaml")
        preprocessing_config = os.path.join(
            args.config_dir, "preprocessing_config.yaml"
        )

        with open(model_config, "r") as f:
            model_cfg = yaml.safe_load(f)

        with open(preprocessing_config, "r") as f:
            preprocessing_cfg = yaml.safe_load(f)

        # Create transforms
        combined_cfg = {**model_cfg, **preprocessing_cfg}
        _, test_transform = create_transforms(combined_cfg)

        # Create preprocessor (for data loading)
        preprocessor = TumorPreprocessor(
            csv_file=args.csv_file,
            img_dir=args.img_dir,
            train_transform=test_transform,  # Use same transform for both
            test_transform=test_transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Get dataloaders
        train_loader, test_loader = preprocessor.get_dataloaders()

        # Get train and test dataframes to extract IDs
        train_df = preprocessor.train_df
        test_df = preprocessor.test_df

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model, _ = load_model_for_embeddings(
            model_path=args.model_path, model_name=args.model_name, device=device
        )

        # Set embedding name (for output files)
        if args.embedding_name is None:
            embedding_name = args.model_name.replace("/", "_")
        else:
            embedding_name = args.embedding_name

        # Generate train embeddings
        logger.info("Generating train embeddings...")
        train_output_path = os.path.join(
            args.output_dir, f"train_embeddings_{embedding_name}.csv"
        )
        generate_embeddings_csv(
            model=model,
            dataloader=train_loader,
            device=device,
            output_path=train_output_path,
            id_list=train_df["isic_id"].values,
            target_list=train_df["target"].values,
        )

        # Generate test embeddings
        logger.info("Generating test embeddings...")
        test_output_path = os.path.join(
            args.output_dir, f"test_embeddings_{embedding_name}.csv"
        )
        generate_embeddings_csv(
            model=model,
            dataloader=test_loader,
            device=device,
            output_path=test_output_path,
            id_list=test_df["isic_id"].values,
            target_list=test_df["target"].values,
        )

        logger.info("Embedding extraction completed successfully!")
        logger.info(f"Train embeddings saved to: {train_output_path}")
        logger.info(f"Test embeddings saved to: {test_output_path}")

    except Exception as e:
        logger.exception(f"Error during embedding extraction: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
