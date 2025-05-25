import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights
import timm

class TripletResNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, embedding_size
        )
        self.normalize = nn.LayerNorm(embedding_size)  # Normalização

    def forward(self, x):
        x = self.backbone(x)
        return self.normalize(x)  # Embeddings normalizados

class ResNet50(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = torch.nn.functional.normalize(x, p=1, dim=1)
        return x.view(x.size(0), -1)
    
class ResNet(nn.Module):
    def __init__(self, model_name='vit_large_patch16_224', pretrained=True):
        """ """
        super(ResNet, self).__init__()

        # Load pretrained DINO model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

    def forward(self, x):
        return self.backbone(x)  # Already returns flattened embeddings
    
    
if __name__ == '__main__':
    model = ResNet50()
    out = model(torch.randn(2, 3, 224, 224))
    print(out)
    print(out.shape)
