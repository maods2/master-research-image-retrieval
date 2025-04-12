import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image
from huggingface_hub import login, hf_hub_download


# need to specify MLP layer and activation function for proper init
model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model = model.eval()

with torch.no_grad():
    # Dummy input tensor
    x = torch.randn(32, 3, 224, 224).to('cuda')
    output = model(x)
    print(output.shape)  
    print(output)  