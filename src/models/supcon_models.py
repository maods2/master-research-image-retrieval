import torch
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, base_model, hidden_dim=512, out_dim=128):
        super().__init__()
        self.backbone = base_model
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.proj(features)
