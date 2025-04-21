"""
https://huggingface.co/MahmoodLab/UNI
https://github.com/mahmoodlab/UNI
https://www.nature.com/articles/s41591-024-02857-3.epdf?sharing_token=CzM2TCW_6hilYJ6BCMgx5dRgN0jAjWel9jnR3ZoTv0PwDLGHgijc_t52lQyxVvw552KDCUhMbS4KuO_rvLnP6S1BpmIj9puojkF8lfR5R8uEX08B0FxePgIH0t7DovKvZF4NHQKlq4TZHGAA1wEIdkYKvcr8nUsaa-nNYbNw3JI%3D
"""

import torch.nn.functional as F
from torch import Tensor
import torch
import torch.nn as nn
import timm
import os

local_dir = './assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/'


class BaseFsl(nn.Module):
    def compute_prototypes(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        prototypes = []
        for c in torch.unique(labels):
            class_mask = labels == c
            class_proto = embeddings[class_mask].mean(0)
            prototypes.append(class_proto)
        return torch.stack(prototypes)

    def predict_with_prototypes(
        self, query_embeddings: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        dists = torch.cdist(query_embeddings, prototypes)
        return torch.argmin(dists, dim=1)

    def predict_probabilities(
        self, query_embeddings: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        dists = torch.cdist(query_embeddings, prototypes)
        return (-dists).softmax(dim=1)


class UNIFsl(BaseFsl):
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


class ResNetFsl(BaseFsl):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()

        # Initialize backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Calculate output dimension
        with torch.no_grad():
            test_tensor = torch.randn(1, 3, 224, 224)
            out_dim = self.backbone(test_tensor).shape[-1]

        # Create projection
        self.projection = nn.Sequential(
            nn.Linear(out_dim, 512), nn.GELU(), nn.Linear(512, 128)
        )

        # Freeze backbone if needed
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x


class SemanticAttributeFsl(nn.Module):
    def __init__(
        self, model_name='vit_large_patch16_224', pretrained=True, config=None
    ):
        super(SemanticAttributeFsl, self).__init__()

        # Backbone with projection head
        self.backbone = UNIFsl(model_name, pretrained)
        self.backbone.load_state_dict(
            torch.load(config['model_path'], map_location='cpu'),
            strict=True,
        )

        # Load support set (dict: class_name -> (images, labels))
        self.support_set = (
            {}
        )  # class_name -> (support_imgs: Tensor [N_shot, C, H, W], support_labels: Tensor [N_shot])
        self.device = config.get('device', 'cpu')
        self._load_support_set(config['support_set'])

    def _load_support_set(self, support_set_config):
        """
        support_set_config: dict { class_name: {"images": Tensor, "labels": Tensor} }
        """
        for class_name, data in support_set_config.items():
            support_imgs = data['images'].to(
                self.device
            )      # [N_shot, C, H, W]
            support_lbls = data['labels'].to(self.device)      # [N_shot]
            self.support_set[class_name] = (support_imgs, support_lbls)

    def _prototypical_scores(
        self, support_embeddings, support_labels, query_embeddings
    ):
        """
        support_embeddings: [N_shot, D]
        support_labels: [N_shot]  (all 1s ideally for the target class)
        query_embeddings: [B, D]

        Returns:
            probs: [B] probability for each query being in the same class as support
        """
        # Get prototype vector
        prototype = support_embeddings.mean(dim=0)  # [D]

        # Euclidean distance to prototype
        dists = torch.norm(
            query_embeddings - prototype.unsqueeze(0), dim=1
        )  # [B]

        # Convert to similarity (negative distance) and sigmoid
        scores = -dists
        probs = torch.sigmoid(scores)
        return probs  # [B]

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, C, H, W] - query images

        Returns:
            out: [B, N_classes] - probability per class
        """
        query_embeddings = self.backbone(x)  # [B, D]
        all_probs = []

        for class_name, (
            support_imgs,
            support_lbls,
        ) in self.support_set.items():
            support_embeddings = self.backbone(support_imgs)  # [N_shot, D]

            probs = self._prototypical_scores(
                support_embeddings=support_embeddings,
                support_labels=support_lbls,
                query_embeddings=query_embeddings,
            )  # [B]

            all_probs.append(probs.unsqueeze(1))  # [B, 1]

        # Concatenate along class dimension
        out = torch.cat(all_probs, dim=1)  # [B, N_classes]
        return out


if __name__ == '__main__':
    model = UNIFsl()
    model = model.to('cuda')
    model.eval()
    prototypes = torch.randn(5, 128).to('cuda')  # Dummy prototypes
    with torch.no_grad():
        # Dummy input tensor
        x = torch.randn(1, 3, 224, 224).to('cuda')
        output = model(x)
        prob = model.predict_probabilities(output, prototypes)
        print(output.shape)
        print(output)
        print(prob.shape)
        print(prob)
