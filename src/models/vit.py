import torch
import torch.nn as nn
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
    
    
    
if __name__ == "__main__":
    model = ViT()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)  # Should print torch.Size([1, 768]) for ViT-B/16