import torch
from torchvision.models import ViT_B_16_Weights
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
import timm

class ViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViT, self).__init__()
        # Creating the model without the classification layer
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.model.forward_features = self._forward_features  # Overriding the forward function
    
    def _forward_features(self, x):
        # Obtaining embeddings for all tokens
        out = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(out.shape[0], -1, -1)
        out = torch.cat((cls_token, out), dim=1)
        out = self.model.pos_drop(out)
        out = self.model.blocks(out)
        return out  # Returns all tokens
    
    def forward(self, x, cls_token_only=True):
        out = self.model.forward_features(x)
        if cls_token_only:
            return out[:, 0, :]
        return out
    
    
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
    model = ViT()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)  # Should print torch.Size([1, 768]) for ViT-B/16