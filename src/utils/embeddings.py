import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import timm
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)


def extract_embeddings(model, dataloader, device, embedding_layer="pre_logits"):
    """
    Extract embeddings from a model for all images in a dataloader

    Args:
        model: PyTorch model
        dataloader: DataLoader containing images
        device: Device to run model on
        embedding_layer: Name of the layer to extract embeddings from

    Returns:
        numpy array of embeddings
    """
    model.eval()
    all_embeddings = []
    all_targets = []

    # Extract activations from the specified layer
    if embedding_layer not in ["pre_logits", "features"]:
        logger.warning(
            f"Unknown embedding layer: {embedding_layer}. Using 'pre_logits'."
        )
        embedding_layer = "pre_logits"

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)

            # Forward pass through the model
            if embedding_layer == "pre_logits":
                # Most timm models have a forward_features method
                if hasattr(model, "forward_features"):
                    features = model.forward_features(images)
                    # Different models handle pooling differently
                    if hasattr(model, "global_pool"):
                        embeddings = model.global_pool(features)
                    else:
                        # Use adaptive pooling as fallback
                        embeddings = torch.nn.functional.adaptive_avg_pool2d(
                            features, (1, 1)
                        )
                        embeddings = embeddings.view(embeddings.size(0), -1)
                else:
                    # Fallback to full forward pass and extract before classifier
                    modules = list(model.children())[:-1]
                    partial_model = torch.nn.Sequential(*modules)
                    embeddings = partial_model(images)
                    embeddings = embeddings.view(embeddings.size(0), -1)
            else:  # 'features'
                # Get intermediate feature maps (before pooling)
                if hasattr(model, "forward_features"):
                    embeddings = model.forward_features(images)
                    # Flatten features if they're not already
                    if len(embeddings.shape) > 2:
                        embeddings = embeddings.mean(
                            dim=[2, 3]
                        )  # Global average pooling
                else:
                    # Try to get features from custom model
                    # This is model-dependent and may need adaptation
                    modules = list(model.children())[
                        :-2
                    ]  # Typically before pooling and classifier
                    partial_model = torch.nn.Sequential(*modules)
                    embeddings = partial_model(images)
                    embeddings = embeddings.mean(dim=[2, 3])  # Global average pooling

            # Convert to numpy
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            all_targets.append(targets.numpy())

    # Concatenate all batches
    all_embeddings = np.vstack(all_embeddings)
    all_targets = np.concatenate(all_targets)

    return all_embeddings, all_targets


def generate_embeddings_csv(
    model, dataloader, device, output_path, id_list=None, target_list=None
):
    """
    Generate CSV file with embeddings

    Args:
        model: PyTorch model
        dataloader: DataLoader with images
        device: Device to run model on
        output_path: Path to save CSV
        id_list: List of IDs for each sample (optional)
        target_list: List of targets for each sample (optional)

    Returns:
        DataFrame with embeddings
    """
    # Extract embeddings
    embeddings, targets = extract_embeddings(model, dataloader, device)

    # Create DataFrame
    df_embeddings = pd.DataFrame(
        embeddings, columns=[f"embed_{i}" for i in range(embeddings.shape[1])]
    )

    # Add ID column if provided
    if id_list is not None:
        if len(id_list) != len(df_embeddings):
            raise ValueError(
                f"Length mismatch: id_list ({len(id_list)}) vs embeddings ({len(df_embeddings)})"
            )
        df_embeddings.insert(0, "isic_id", id_list)

    # Add target column if provided
    if target_list is not None:
        if len(target_list) != len(df_embeddings):
            raise ValueError(
                f"Length mismatch: target_list ({len(target_list)}) vs embeddings ({len(df_embeddings)})"
            )
        df_embeddings.insert(1, "target", target_list)
    elif targets is not None:
        df_embeddings.insert(1, "target", targets)

    # Save to CSV
    df_embeddings.to_csv(output_path, index=False)
    logger.info(f"Embeddings saved to: {output_path}")

    return df_embeddings


def load_model_for_embeddings(model_path, model_name, num_classes=1, device=None):
    """
    Load a model for embedding extraction

    Args:
        model_path: Path to model checkpoint
        model_name: Model architecture name (timm model name)
        num_classes: Number of classes (default: 1 for binary)
        device: Device to load model on (default: auto-detect)

    Returns:
        model, device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.to(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Check if checkpoint contains state_dict directly or in a nested format
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Try to load the checkpoint directly
            model.load_state_dict(checkpoint)

        logger.info(f"Model {model_name} loaded from {model_path}")
        return model, device

    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        raise
