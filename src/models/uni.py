"""
https://huggingface.co/MahmoodLab/UNI
https://github.com/mahmoodlab/UNI
https://www.nature.com/articles/s41591-024-02857-3.epdf?sharing_token=CzM2TCW_6hilYJ6BCMgx5dRgN0jAjWel9jnR3ZoTv0PwDLGHgijc_t52lQyxVvw552KDCUhMbS4KuO_rvLnP6S1BpmIj9puojkF8lfR5R8uEX08B0FxePgIH0t7DovKvZF4NHQKlq4TZHGAA1wEIdkYKvcr8nUsaa-nNYbNw3JI%3D
"""
import sys
import os


sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
import torch
import torch.nn as nn
import timm
import os
from src.utils.checkpoint_utils import load_full_model


class UNI(nn.Module):
    def __init__(self, model_name='uni', artifact_dir=None):
        """ """
        super(UNI, self).__init__()

        # Load pretrained DINO model from timm
        self.backbone = load_full_model(
            model_name=model_name, 
            save_dir=model_name, 
            map_location='cpu',
            artifact_dir=artifact_dir
        )

    def forward(self, x):
        return self.backbone(x)  # Already returns flattened embeddings


if __name__ == '__main__':
    from huggingface_hub import login, hf_hub_download
    import torch
    import os

    # login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    # local_dir = "./assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    model = UNI('UNI2-h')
    model = model.to('cuda')
    model.eval()
    with torch.no_grad():
        # Dummy input tensor
        x = torch.randn(32, 3, 224, 224).to('cuda')
        output = model(x)
        print(model)
        # print(output.shape)
        # print(output)
