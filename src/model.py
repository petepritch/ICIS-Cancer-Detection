import torch
import torch.nn
import torchvision.models as models

class SkinCancerModel(nn.Module):
    """
    Class description
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SkinCancerModel, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)  
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)  

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = SkinCancerModel(num_classes=2)
    print(model)  