import torch
import torch.nn as nn
from encoders import ImageEncoder
from attention import MultiMutualAttention

# --- Final Classification Head ---
class FinalClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.5):
        """
        MLP head to map fused features to class logits.

        Args:
            input_dim (int): Dimension of the fused feature vector from MultiMutualAttention.
            num_classes (int): Number of output classes (e.g., 1 for binary).
            hidden_dim (int): Size of the hidden layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        print(f"Initialized FinalClassificationHead: input={input_dim}, hidden={hidden_dim}, output={num_classes}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # Return logits directly, apply sigmoid/softmax in loss or post-processing
        return x

# --- Overall Model for Two Image Streams ---
class TwoImageMutualAttentionModel(nn.Module):
    def __init__(self, encoder1, encoder2, attention_module, classifier_head):
        """
        Combines two image encoders, a mutual attention block, and a classifier.

        Args:
            encoder1 (nn.Module): Instance of the first ImageEncoder.
            encoder2 (nn.Module): Instance of the second ImageEncoder.
            attention_module (nn.Module): Instance of MultiMutualAttention (K=2).
            classifier_head (nn.Module): Instance of FinalClassificationHead.
        """
        super().__init__()
        # Store the passed-in modules
        self.encoder_m1 = encoder1
        self.encoder_m2 = encoder2
        self.attention_block = attention_module
        self.classifier = classifier_head
        print("Initialized TwoImageMutualAttentionModel.")

    def forward(self, image_batch):
        """
        Defines the forward pass.

        Args:
            image_batch (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Output logits from the classifier.
        """
        # 1. Get features from each image encoder
        # Assuming the same image batch goes to both for now
        m1_features = self.encoder_m1(image_batch) # Shape: (batch, feature_dim1)
        m2_features = self.encoder_m2(image_batch) # Shape: (batch, feature_dim2)

        # 2. Fuse features using the attention block
        # Pass features as a list
        fused_features = self.attention_block([m1_features, m2_features]) # Shape: (batch, feature_dim1+feature_dim2)

        # 3. Classify the fused features
        logits = self.classifier(fused_features) # Shape: (batch, num_classes)

        return logits
    

# --- Metadata MLP Model Definition (Copied from train_metadata_mlp.py for completeness) ---
# It's better to have this in a separate file and import it.
class MetadataEncoderMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout_rate=0.5):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)) # Batch norm often helps MLPs
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim)) # Final output layer IS included here

        self.mlp = nn.Sequential(*layers) # All layers in a single sequential 'mlp'
        print(f"Initialized MetadataEncoderMLP (Original Structure for Loading):")
        # print(self.mlp) # Can print if needed

    def forward(self, x):
         # Default forward pass including the final classifier
        return self.mlp(x)

# --- Combined Image + Metadata Mutual Attention Model ---
class ImageMetadataMutualAttentionModel(nn.Module):
    def __init__(self, image_encoder, metadata_encoder, attention_module, classifier_head):
        """
        Combines one image encoder, one metadata encoder, mutual attention, and a classifier.

        Args:
            image_encoder (nn.Module): Instance of ImageEncoder.
            metadata_encoder (nn.Module): Instance of MetadataEncoderMLP (modified for features).
            attention_module (nn.Module): Instance of MultiMutualAttention (K=2).
            classifier_head (nn.Module): Instance of FinalClassificationHead.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.attention_block = attention_module
        self.classifier = classifier_head
        print("Initialized ImageMetadataMutualAttentionModel.")

    def forward(self, image_batch, metadata_batch):
        """
        Defines the forward pass for combined image and metadata input.

        Args:
            image_batch (torch.Tensor): Batch of input images.
            metadata_batch (torch.Tensor): Batch of input metadata features.

        Returns:
            torch.Tensor: Output logits from the classifier.
        """
        # 1. Get features from each encoder
        image_features = self.image_encoder(image_batch)       # Shape: (batch, image_feature_dim)
        metadata_features = self.metadata_encoder(metadata_batch) # Shape: (batch, metadata_feature_dim)

        # 2. Fuse features using the attention block
        # Pass features as a list: [image_features, metadata_features]
        fused_features = self.attention_block([image_features, metadata_features])
        # Output shape: (batch, image_feature_dim + metadata_feature_dim)

        # 3. Classify the fused features
        logits = self.classifier(fused_features) # Shape: (batch, num_classes)

        return logits

# --- Final Classification Head (Copied from model.py for completeness) ---
# Better to import this if model.py is accessible
class FinalClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        print(f"Initialized FinalClassificationHead: input={input_dim}, hidden={hidden_dim}, output={num_classes}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x # Return logits
    
class FullEnsembleMutualAttentionModel(nn.Module):
    def __init__(self, image_encoders, metadata_encoder, attention_module, classifier_head):
        """
        Combines multiple image encoders, one metadata encoder, mutual attention, and a classifier.

        Args:
            image_encoders (list[nn.Module]): List containing instances of ImageEncoder (e.g., 4 ViTs).
            metadata_encoder (nn.Module): Instance of MetadataEncoderMLP (adapted for features).
            attention_module (nn.Module): Instance of MultiMutualAttention (K=5).
            classifier_head (nn.Module): Instance of FinalClassificationHead.
        """
        super().__init__()
        if not isinstance(image_encoders, list):
             raise TypeError("image_encoders must be a list of nn.Module instances.")
        # Use ModuleList to correctly register the image encoders
        self.image_encoders = nn.ModuleList(image_encoders)
        self.metadata_encoder = metadata_encoder
        self.attention_block = attention_module
        self.classifier = classifier_head

    def forward(self, image_batch, metadata_batch):
        """
        Defines the forward pass for the full ensemble.

        Args:
            image_batch (torch.Tensor): Batch of input images (passed to all image encoders).
            metadata_batch (torch.Tensor): Batch of input metadata features.

        Returns:
            torch.Tensor: Output logits from the classifier.
        """
        # 1. Get features from each encoder
        image_features_list = [encoder(image_batch) for encoder in self.image_encoders]
        metadata_features = self.metadata_encoder(metadata_batch)


        # Combine features into a single list for the attention block
        all_features = image_features_list + [metadata_features] # List of 5 tensors

        # 2. Fuse features using the attention block
        # The error likely happens inside here if shapes are wrong
        fused_features = self.attention_block(all_features)
        # Output shape: (batch, sum_of_all_feature_dims)

        # 3. Classify the fused features
        logits = self.classifier(fused_features) # Shape: (batch, num_classes)

        return logits