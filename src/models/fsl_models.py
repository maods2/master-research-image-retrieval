"""
https://huggingface.co/MahmoodLab/UNI
https://github.com/mahmoodlab/UNI
https://www.nature.com/articles/s41591-024-02857-3.epdf?sharing_token=CzM2TCW_6hilYJ6BCMgx5dRgN0jAjWel9jnR3ZoTv0PwDLGHgijc_t52lQyxVvw552KDCUhMbS4KuO_rvLnP6S1BpmIj9puojkF8lfR5R8uEX08B0FxePgIH0t7DovKvZF4NHQKlq4TZHGAA1wEIdkYKvcr8nUsaa-nNYbNw3JI%3D
"""

import torch
import torch.nn as nn
import timm
import os

local_dir = './assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/'


class UNIFsl(nn.Module):
    def __init__(self, model_name='vit_large_patch16_224', pretrained=True):
        """ """
        super(UNIFsl, self).__init__()

        # Load pretrained DINO model from timm
        self.backbone = model = timm.create_model(
            model_name,
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(
            torch.load(
                os.path.join(local_dir, 'pytorch_model.bin'),
                map_location='cpu',
            ),
            strict=True,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        test_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out_dim = self.backbone(test_tensor).shape[-1]

        # Projection head (non-linear)
        self.projection = nn.Sequential(
            nn.Linear(out_dim, 512), nn.GELU(), nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x
    
    def compute_prototypes(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class prototypes from embeddings and labels.

        Args:
            embeddings: Feature embeddings.
            labels: Corresponding labels.

        Returns:
            Tensor of class prototypes.
        """
        prototypes = []
        for c in torch.unique(labels):
            class_mask = labels == c
            class_proto = embeddings[class_mask].mean(0)
            prototypes.append(class_proto)
        return torch.stack(prototypes)  # [n_classes, embedding_dim]

    def predict_with_prototypes(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Predicts the class of query embeddings based on the closest prototype.

        Args:
            query_embeddings: Tensor of shape [n_query, D], query feature embeddings.
            prototypes: Tensor of shape [n_way, D], class prototypes.

        Returns:
            Tensor of shape [n_query], predicted class indices.
        """
        dists = torch.cdist(query_embeddings, prototypes)  # [n_query, n_way]
        preds = torch.argmin(dists, dim=1)  # [n_query]
        return preds

if __name__ == '__main__':
    model = UNIFsl()
    model = model.to('cuda')
    model.eval()
    with torch.no_grad():
        # Dummy input tensor
        x = torch.randn(32, 3, 224, 224).to('cuda')
        output = model(x)
        print(output.shape)
        print(output)
