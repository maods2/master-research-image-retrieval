"""
https://arxiv.org/pdf/2408.00738
https://huggingface.co/paige-ai/Virchow2
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


class Virchow2(nn.Module):
    def __init__(self, model_name='Virchow2', pretrained=True):
        """ """
        super(Virchow2, self).__init__()

        self.backbone = load_full_model(
            model_name=model_name, save_dir=model_name, map_location='cpu'
        )

    def forward(self, x, only_cls_token=True):

        output = self.backbone(x)   # size: 1 x 261 x 1280
        class_token = output[:, 0]    # size: 1 x 1280

        if not only_cls_token:
            patch_tokens = output[
                :, 5:
            ]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
            embedding = torch.cat(
                [class_token, patch_tokens.mean(1)], dim=-1
            )  # size: 1 x 2560 -concatenate class token and average pool of patch tokens
            return embedding

        return class_token  # size: 1 x 1280


if __name__ == '__main__':
    from huggingface_hub import login, hf_hub_download

    # login()

    # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    # hf_hub_download("paige-ai/Virchow2", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    model = Virchow2()
    model = model.to('cuda')
    model.eval()
    with torch.no_grad():
        # Dummy input tensor
        x = torch.randn(32, 3, 224, 224).to('cuda')
        output = model(x)
        print(output.shape)
        print(output)
