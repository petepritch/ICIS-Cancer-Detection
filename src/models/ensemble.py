import torch
import torch.nn as nn


class FinalClassificationHead(nn.Module):
    """
    MLP head for final classification.
    """

    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.5):
        """
        Initialize classification head.

        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes (1 for binary)
            hidden_dim: Size of hidden layer
            dropout: Dropout rate
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Forward pass of classification head."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MetadataEncoderMLP(nn.Module):
    """
    MLP for encoding metadata features.
    """

    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout_rate=0.5):
        """
        Initialize metadata encoder MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for binary)
            dropout_rate: Dropout rate
        """
        super().__init__()
        layers = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of metadata encoder."""
        return self.mlp(x)


class ImageMetadataMutualAttentionModel(nn.Module):
    """
    Combined model using image encoder, metadata encoder, mutual attention, and classifier.
    """

    def __init__(
        self, image_encoder, metadata_encoder, attention_module, classifier_head
    ):
        """
        Initialize the combined model.

        Args:
            image_encoder: Instance of ImageEncoder
            metadata_encoder: Instance of MetadataEncoderMLP
            attention_module: Instance of MultiMutualAttention
            classifier_head: Instance of FinalClassificationHead
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.attention_block = attention_module
        self.classifier = classifier_head

    def forward(self, image_batch, metadata_batch):
        """
        Forward pass for combined model.

        Args:
            image_batch: Batch of input images
            metadata_batch: Batch of metadata features

        Returns:
            torch.Tensor: Output logits
        """
        # Extract features
        image_features = self.image_encoder(image_batch)
        metadata_features = self.metadata_encoder(metadata_batch)

        # Fuse features using attention
        fused_features = self.attention_block([image_features, metadata_features])

        # Classify
        logits = self.classifier(fused_features)

        return logits


class FullEnsembleMutualAttentionModel(nn.Module):
    """
    Ensemble model combining multiple image encoders with metadata.
    """

    def __init__(
        self, image_encoders, metadata_encoder, attention_module, classifier_head
    ):
        """
        Initialize ensemble model.

        Args:
            image_encoders: List of ImageEncoder instances
            metadata_encoder: Instance of MetadataEncoderMLP
            attention_module: Instance of MultiMutualAttention
            classifier_head: Instance of FinalClassificationHead
        """
        super().__init__()
        if not isinstance(image_encoders, list):
            raise TypeError("image_encoders must be a list of nn.Module instances.")

        self.image_encoders = nn.ModuleList(image_encoders)
        self.metadata_encoder = metadata_encoder
        self.attention_block = attention_module
        self.classifier = classifier_head

    def forward(self, image_batch, metadata_batch):
        """
        Forward pass for ensemble model.

        Args:
            image_batch: Batch of input images
            metadata_batch: Batch of metadata features

        Returns:
            torch.Tensor: Output logits
        """
        # Extract features from each image encoder
        image_features_list = [encoder(image_batch) for encoder in self.image_encoders]

        # Extract metadata features
        metadata_features = self.metadata_encoder(metadata_batch)

        # Combine all features for attention
        all_features = image_features_list + [metadata_features]

        # Fuse features using attention
        fused_features = self.attention_block(all_features)

        # Classify
        logits = self.classifier(fused_features)

        return logits
