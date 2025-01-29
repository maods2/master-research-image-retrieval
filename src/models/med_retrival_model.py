"""
Final Architecture Overview
1. Backbone + Feature Pyramid Network (FPN) → Multi-scale features.
2. Squeeze-and-Excitation Blocks → Channel-wise feature recalibration.
3. Efficient Multi-Head Attention → Lightweight attention for feature fusion.
4. Multi-Task Loss → Jointly optimize classification and metric learning.
5. Triplet Loss + Contrastive Loss → Compare performance.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import PerformerLM  # Requires: pip install performer-pytorch
from torchvision import models
from torchvision.models import ResNet50_Weights


# Note: This uses a dummy dataset - replace with actual medical image loader
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=100, img_size=(3, 224, 224)):
        self.images = torch.randn(size, *img_size)
        self.labels = torch.randint(0, 2, (size, 14)).float()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {'image': self.images[idx], 'labels': self.labels[idx]}
    
# ====================== Model Components ======================
class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale features"""
    def __init__(self, backbone_channels=[256, 512, 1024], fpn_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, fpn_channels, 1) for ch in backbone_channels
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1) 
            for _ in backbone_channels
        ])

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        pyramid = [laterals[-1]]
        for i in range(len(laterals)-2, -1, -1):
            pyramid.append(F.interpolate(pyramid[-1], scale_factor=2) + laterals[i])
            
        # Smooth and reverse
        return [conv(f) for conv, f in zip(self.smooth_convs, pyramid[::-1])]

class MedicalRetrievalModel(nn.Module):
    """Complete medical image retrieval model"""
    def __init__(self, num_classes, embed_dim=512):
        super().__init__()
        
        # Backbone with SE blocks
        self.backbone = self._create_se_resnet()
        self.fpn = FPN()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, num_classes),  # ResNet50 final features
            nn.Sigmoid()
        )
        
        # Attention & projection
        self.embed_proj = nn.Linear(num_classes, embed_dim)
        self.attn = PerformerLM(
            dim=embed_dim,
            depth=1,
            heads=8,
            dim_head=64,
            causal=False
        )
        self.final_proj = nn.Linear(embed_dim, embed_dim)
        
    def _create_se_resnet(self):
        """Create ResNet-50 with SE blocks"""
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Add SE blocks to layers
        model.layer2.add_module("se", SqueezeExcitation(512))
        model.layer3.add_module("se", SqueezeExcitation(1024))
        model.layer4.add_module("se", SqueezeExcitation(2048))
        
        # Remove original FC layer
        model.fc = nn.Identity()
        return model

    def forward(self, x):
        # Backbone features
        with torch.no_grad():
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            c2 = self.backbone.layer1(x)
            c3 = self.backbone.layer2(c2)
            c4 = self.backbone.layer3(c3)
            c5 = self.backbone.layer4(c4)
        
        # FPN features
        pyramid = self.fpn([c3, c4, c5])
        features = torch.cat([p.flatten(2).mean(2) for p in pyramid], dim=1)
        
        # Classification
        logits = self.classifier(c5)
        
        # Embedding fusion
        class_emb = self.embed_proj(logits)
        attn_out = self.attn(class_emb.unsqueeze(1))
        final_emb = self.final_proj(attn_out.squeeze(1))
        
        return final_emb, logits

# ====================== Loss Functions ======================
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.cosine_similarity(anchor, positive)
        neg_dist = F.cosine_similarity(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb1, emb2, labels):
        sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
        sim_matrix = sim_matrix / self.temperature
        return F.cross_entropy(sim_matrix, labels)

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss()
        self.triplet = TripletLoss()

    def forward(self, pred_logits, embeddings, labels, positive, negative):
        cls_loss = self.bce(pred_logits, labels)
        metric_loss = self.triplet(embeddings, positive, negative)
        return self.alpha * cls_loss + self.beta * metric_loss

# ====================== Training Setup ======================
if __name__ == "__main__":
    # Example configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 14  # Example: CheXpert 14 pathologies
    
    # Model
    model = MedicalRetrievalModel(num_classes).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Loss
    criterion = MultiTaskLoss(alpha=1.0, beta=0.5)
    
    # Example training loop
    for epoch in range(10):
        for batch in DummyDataset():  # Replace with real dataloader
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            embeddings, logits = model(images)
            
            # Dummy positive/negative pairs - replace with real sampling
            positive = embeddings[torch.randperm(embeddings.size(0))]
            negative = embeddings[torch.randperm(embeddings.size(0))]
            
            # Loss calculation
            loss = criterion(logits, embeddings, labels, positive, negative)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

