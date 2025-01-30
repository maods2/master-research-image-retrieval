from torch import nn
import torch
import torch.nn.functional as F

class AdaptiveTripletLoss(nn.Module):
    def __init__(self, margin=0.5, mining="hard"):
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(self, anchor, positive, negative):
        # Já vem normalizado dos modelos
        pos_dist = 1 - torch.sum(anchor * positive, dim=1)  # Cosine distance
        neg_dist = 1 - torch.sum(anchor * negative, dim=1)
        
        if self.mining == "hard":
            loss = F.relu(pos_dist - neg_dist + self.margin)
        elif self.mining == "semi_hard":
            mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)
            loss = torch.where(mask, pos_dist - neg_dist + self.margin, torch.zeros_like(pos_dist))
        else:
            loss = pos_dist - neg_dist + self.margin
            
        return torch.mean(loss)