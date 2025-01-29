import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class TripletResNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)
        self.normalize = nn.LayerNorm(embedding_size)  # Normalização

    def forward(self, x):
        x = self.backbone(x)
        return self.normalize(x)  # Embeddings normalizados
    
    
if __name__ == "__main__":
    model = TripletResNet()
    out = model(torch.randn(2, 3, 224, 224))
    print(out)
    print(out.shape)