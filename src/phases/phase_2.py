import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ---- Phase 2: Enhanced Feature Representation ---- #

# Feature Pyramid Network (FPN) Module
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                for in_channels in in_channels_list
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, features):
        # Top-down pathway
        last_feat = self.lateral_convs[-1](features[-1])
        outputs = [last_feat]
        for i in range(len(features) - 2, -1, -1):
            feat = self.lateral_convs[i](features[i])
            last_feat = (
                nn.functional.interpolate(
                    last_feat, size=feat.shape[-2:], mode='nearest'
                )
                + feat
            )
            outputs.insert(0, last_feat)

        # Apply convolution
        outputs = [
            fpn_conv(feat) for feat, fpn_conv in zip(outputs, self.fpn_convs)
        ]
        return outputs


# Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.shape
        se = x.view(batch, channels, -1).mean(dim=2)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(batch, channels, 1, 1)
        return x * se


# Efficient Multi-Head Attention
class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(EfficientMultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        return self.attn(x, x, x)[0]


# Updated Feature Extractor with FPN and Attention
class AdvancedFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50', embedding_dim=256):
        super(AdvancedFeatureExtractor, self).__init__()

        # Load Backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(
                *list(self.backbone.children())[:-2]
            )  # Keep feature maps
            fpn_channels = [512, 1024, 2048]
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
            fpn_channels = [40, 112, 1280]
        else:
            raise ValueError('Unsupported backbone')

        # Feature Pyramid Network
        self.fpn = FPN(fpn_channels, out_channels=256)

        # Squeeze-and-Excitation Block
        self.se_block = SEBlock(256)

        # Efficient Multi-Head Attention
        self.attention = EfficientMultiHeadAttention(256)

        # Embedding Layer
        self.embedding_layer = nn.Linear(256, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        fpn_features = sum(fpn_features)  # Aggregate pyramid features
        fpn_features = self.se_block(fpn_features)
        fpn_features = torch.flatten(fpn_features, start_dim=1)  # Flatten
        fpn_features = self.attention(fpn_features.unsqueeze(1)).squeeze(
            1
        )  # Apply attention
        embedding = self.embedding_layer(fpn_features)
        return embedding


# Loss Function: Multi-Task Loss (Triplet + Contrastive Loss)
class MultiTaskLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MultiTaskLoss, self).__init__()
        self.contrastive_loss = ContrastiveLoss(margin)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, emb1, emb2, emb3, label):
        return self.contrastive_loss(emb1, emb2, label) + self.triplet_loss(
            emb1, emb2, emb3
        )


# Model Initialization
model = AdvancedFeatureExtractor(backbone='resnet50', embedding_dim=256).cuda()
criterion = MultiTaskLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print('Phase 2 Model Ready!')
