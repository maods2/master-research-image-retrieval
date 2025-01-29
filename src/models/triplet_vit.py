from torchvision.models import ViT_B_16_Weights
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
    
class TripletViT(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.backbone.heads.head.in_features
        self.projection = nn.Linear(in_features, embedding_size)
        self._init_weights(self.projection)
        
        # Substitui a camada final original
        self.backbone.heads = nn.Identity()

    def _init_weights(self, module):
        nn.init.trunc_normal_(module.weight, std=0.02)
        nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.backbone(x) 
        x = self.projection(x)
        return F.normalize(x, p=2, dim=0)
    
if __name__ == "__main__":
    import torch
    model = TripletViT()
    out = model(torch.randn(2, 3, 224, 224))
    print(out)
    print(out.shape)