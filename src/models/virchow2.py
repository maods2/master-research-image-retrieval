"""
https://arxiv.org/pdf/2408.00738
https://huggingface.co/paige-ai/Virchow2
"""

import torch
import torch.nn as nn
import timm
import os

local_dir = "./assets/ckpts/vit_Virchow2/"
class Virchow2(nn.Module):
    def __init__(self, model_name='hf-hub:paige-ai/Virchow2', pretrained=True):
        """ """
        super(Virchow2, self).__init__()

        # Load pretrained DINO model from timm
        self.backbone = model = timm.create_model(
            model_name,
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        # model.load_state_dict(
        #     torch.load(
        #         os.path.join(local_dir, 'pytorch_model.bin'),
        #         map_location='cpu',
        #     ),
        #     strict=True,
        # )

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
