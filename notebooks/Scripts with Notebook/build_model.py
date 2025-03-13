# -*- coding: utf-8 -*-
"""
This file contains a helper function that lets us easily build a base model. Either with pretrained weights by setting "pretrained=True",
or a base model, which can have custom weights loaded in by setting pretrained=False.
"""

import torch.nn as nn
from torchvision.models import (
    resnet18, resnet50, resnet101,
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
    efficientnet_b0, efficientnet_b3,
    EfficientNet_B0_Weights, EfficientNet_B3_Weights,
    alexnet, AlexNet_Weights,
    vgg16, VGG16_Weights,
)

def build_model(model_name="resnet18", pretrained=True):
    """
    Creates a torchvision model with a final layer for binary classification.
    model_name: str, e.g. "resnet18", "resnet50", "efficientnet_b3", "alexnet", ...
    pretrained: bool, load pretrained ImageNet weights if True.
    
    Returns: A torch.nn.Module with `model(...)=logits` for binary classification.
    """
    # A mapping of model_name -> (constructor_fn, default_weights, final_layer_name)
    model_specs = {
        "resnet18": {
            "builder": resnet18,
            "weights": ResNet18_Weights.IMAGENET1K_V1,
            "final_layer": "fc"
        },
        "resnet50": {
            "builder": resnet50,
            "weights": ResNet50_Weights.IMAGENET1K_V1,
            "final_layer": "fc"
        },
        "resnet101": {
            "builder": resnet101,
            "weights": ResNet101_Weights.IMAGENET1K_V1,
            "final_layer": "fc"
        },
        "efficientnet_b0": {
            "builder": efficientnet_b0,
            "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
            "final_layer": "classifier"
        },
        "efficientnet_b3": {
            "builder": efficientnet_b3,
            "weights": EfficientNet_B3_Weights.IMAGENET1K_V1,
            "final_layer": "classifier"
        },
        "alexnet": {
            "builder": alexnet,
            "weights": AlexNet_Weights.IMAGENET1K_V1,
            "final_layer": "classifier"
        },
        "vgg16": {
            "builder": vgg16,
            "weights": VGG16_Weights.IMAGENET1K_V1,
            "final_layer": "classifier"
        },
    }

    if model_name not in model_specs:
        raise ValueError(f"Model '{model_name}' not recognized. Add to model_specs if needed.")

    entry = model_specs[model_name]
    
    # Decide on weights or None depending on the pretrained flag
    weights_arg = entry["weights"] if pretrained else None
    
    # Build the base model
    base_model = entry["builder"](weights=weights_arg)
    
    # Replace the final classification layer with a single-logit output
    final_layer = entry["final_layer"]
    if final_layer == "fc":
        # ResNets
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, 1)
    elif final_layer == "classifier":
        # EfficientNet, AlexNet, VGG, etc. Typically you do something like:
        #   model.classifier = nn.Sequential(..., nn.Linear(..., 1))
        # but it varies slightly by architecture. We'll handle the known patterns:
        if "efficientnet" in model_name:
            # e.g. base_model.classifier[1] = nn.Linear(num_features, 1)
            in_features = base_model.classifier[1].in_features
            base_model.classifier[1] = nn.Linear(in_features, 1)
        elif "alexnet" in model_name:
            # AlexNet's classifier is a chain of layers
            # The last linear layer is at index 6
            in_features = base_model.classifier[6].in_features
            base_model.classifier[6] = nn.Linear(in_features, 1)
        elif "vgg" in model_name:
            # VGG16's classifier is also a chain of layers
            # The last linear layer is at index 6
            in_features = base_model.classifier[6].in_features
            base_model.classifier[6] = nn.Linear(in_features, 1)
        else:
            # If there's another model using .classifier, adapt similarly
            raise NotImplementedError(f"Final layer adaptation not implemented for {model_name}.")
    else:
        raise ValueError(f"Unknown final layer '{final_layer}' for {model_name}.")

    return base_model
