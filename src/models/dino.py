import torch
import torch.nn as nn
import timm

class DINO(nn.Module):
    def __init__(self, model_name='vit_small_patch16_224_dino', pretrained=True):
        """
        'vit_small_patch16_224_dino' → 384 dims
        'vit_base_patch16_224_dino' → 768 dims
        'vit_base_patch8_224_dino'
        """
        super(DINO, self).__init__()

        # Load pretrained DINO model from timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # num_classes=0 returns features

    def forward(self, x):
        return self.backbone(x)  # Already returns flattened embeddings
    
    
    
if __name__ == "__main__":
    model = DINO()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)  
    print(output)  