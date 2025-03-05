import torch
import torch.nn as nn
import torchvision.models as models

class ISICResNet(nn.Module):
    def __init__(self, config):
        """
        Initialize ResNet Model.
        """
        super(ISICResNet, self).__init__()

        arch = config['model']['arch']
        pretrained = config['model']['pretrained']
        num_classes = config['model']['num_classes']
        freeze_backbone = config['model']['freeze_backbone']
        dropout_rate = config['model']['dropout_rate']

        if arch == 'resnet18':
            self.resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        elif arch == 'resnet34':
            self.resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        elif arch == 'resnet50':
            self.resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif arch == 'resnet101':
            self.resnet = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f'Invalid ResNet architecture: {arch}')
        
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

def forward(self, x):
    return self.resnet(x)