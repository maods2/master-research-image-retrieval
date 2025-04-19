import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ---- Phase 3: Optimization & Fine-Tuning ---- #

# Hard Negative Mining
class HardNegativeMining:
    def __init__(self, margin=0.5):
        self.margin = margin

    def get_hard_negatives(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        dists = torch.cdist(
            embeddings, embeddings
        )  # Compute pairwise distances
        hard_negatives = []
        for i in range(batch_size):
            negatives = dists[i][
                labels != labels[i]
            ]  # Get different class samples
            if len(negatives) > 0:
                hardest_negative = torch.max(negatives)
                hard_negatives.append(hardest_negative)
            else:
                hard_negatives.append(torch.tensor(0.0))
        return torch.stack(hard_negatives)


# Learnable Temperature Scaling for Similarity Scores
class LearnableTemperature(nn.Module):
    def __init__(self, init_temp=0.07):
        super(LearnableTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))

    def forward(self, similarities):
        return similarities / self.temperature.exp()


# Adaptive Loss Weighting
class AdaptiveLossWeighting(nn.Module):
    def __init__(self, init_weight=1.0):
        super(AdaptiveLossWeighting, self).__init__()
        self.loss_weight = nn.Parameter(torch.tensor(init_weight))

    def forward(self, loss):
        return loss * self.loss_weight.exp()


# Fine-Tuned Feature Extractor
class FineTunedFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50', embedding_dim=256):
        super(FineTunedFeatureExtractor, self).__init__()

        # Load Backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            fpn_channels = [512, 1024, 2048]
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
            fpn_channels = [40, 112, 1280]
        else:
            raise ValueError('Unsupported backbone')

        # Feature Pyramid Network (FPN)
        self.fpn = FPN(fpn_channels, out_channels=256)

        # Squeeze-and-Excitation Block
        self.se_block = SEBlock(256)

        # Efficient Multi-Head Attention
        self.attention = EfficientMultiHeadAttention(256)

        # Learnable Temperature Scaling
        self.temperature = LearnableTemperature()

        # Embedding Layer
        self.embedding_layer = nn.Linear(256, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        fpn_features = sum(fpn_features)
        fpn_features = self.se_block(fpn_features)
        fpn_features = torch.flatten(fpn_features, start_dim=1)
        fpn_features = self.attention(fpn_features.unsqueeze(1)).squeeze(1)
        embedding = self.embedding_layer(fpn_features)
        return self.temperature(embedding)


# Multi-Task Loss with Adaptive Weighting
class AdaptiveMultiTaskLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(AdaptiveMultiTaskLoss, self).__init__()
        self.contrastive_loss = ContrastiveLoss(margin)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.loss_weight = AdaptiveLossWeighting()

    def forward(self, emb1, emb2, emb3, label):
        loss = self.contrastive_loss(emb1, emb2, label) + self.triplet_loss(
            emb1, emb2, emb3
        )
        return self.loss_weight(loss)


# Fine-Tuning with Data Augmentations
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Model Initialization
model = FineTunedFeatureExtractor(
    backbone='resnet50', embedding_dim=256
).cuda()
criterion = AdaptiveMultiTaskLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

print('Phase 3 Model Ready!')
