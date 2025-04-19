import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


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


if __name__ == '__main__':
    model = ResNet50()
    out = model(torch.randn(2, 3, 224, 224))
    print(out)
    print(out.shape)
