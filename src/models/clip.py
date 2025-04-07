import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class CLIP(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super(CLIP, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)

    def forward(self, x):
        return self.model(pixel_values=x).pooler_output
    
    
    
    
if __name__ == "__main__":
    model = CLIP()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)  
    print(output)