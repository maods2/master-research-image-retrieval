"""
https://huggingface.co/owkin/phikon-v2
https://huggingface.co/owkin/phikon
https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v3
https://arxiv.org/pdf/2409.09173
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


class Phikon(nn.Module):
    def __init__(self, model_name='phikon', pretrained=True):
        """ """
        super(Phikon, self).__init__()

        # Load pretrained DINO model from timm
        self.backbone = load_full_model(
            model_name=model_name, save_dir=model_name, map_location='cpu'
        )

    def forward(self, x):
        x = self.backbone(x)
        return x.last_hidden_state[:, 0, :]
