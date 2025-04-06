import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()  # Remove classification head

    def forward(self, x):
        x = self.vit(x)  # (B, num_patches + 1, D)
        return x[:, 1:, :]  # Remove CLS token, keep patch embeddings

class PromptEncoder(nn.Module):
    def __init__(self, d_model=768, d_proj=768):
        super().__init__()
        self.projection = nn.Linear(d_model, d_proj)
        self.similarity_proj = nn.Linear(4, d_proj)  # For 2x2 similarity matrix

    def forward(self, pos_embeds, neg_embeds):
        B = pos_embeds.size(0)
        
        # Normalize embeddings
        pos_normalized = F.normalize(pos_embeds, p=2, dim=-1)
        neg_normalized = F.normalize(neg_embeds, p=2, dim=-1)
        
        # Compute similarity matrix between average prompt embeddings
        pos_avg = pos_normalized.mean(dim=2)  # (B, 2, D)
        neg_avg = neg_normalized.mean(dim=2)  # (B, 2, D)
        similarity = torch.einsum('bpd,bqd->bpq', pos_avg, neg_avg)  # (B, 2, 2)
        
        # Project embeddings and compute statistics
        pos_proj = self.projection(pos_embeds)  # (B, 2, 196, d_proj)
        neg_proj = self.projection(neg_embeds)
        
        pos_mean = pos_proj.mean(dim=1)  # (B, 196, d_proj)
        pos_std = pos_proj.std(dim=1)
        neg_mean = neg_proj.mean(dim=1)
        neg_std = neg_proj.std(dim=1)
        
        # Concatenate statistics along token dimension
        stats = torch.cat([pos_mean, pos_std, neg_mean, neg_std], dim=1)  # (B, 784, d_proj)
        
        # Project similarity matrix
        similarity_flat = similarity.view(B, -1)  # (B, 4)
        similarity_proj = self.similarity_proj(similarity_flat)  # (B, d_proj)
        
        return stats, similarity_proj

class ClassifierHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        return self.mlp(x.mean(dim=1))  # (B, 1)

class PromptableBinaryClassifier(nn.Module):
    def __init__(self, d_model=768, d_proj=768):
        super().__init__()
        self.vit_backbone = ViTBackbone()
        self.prompt_encoder = PromptEncoder(d_model, d_proj)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_proj,
                nhead=8,
                dim_feedforward=2048,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        self.classifier = ClassifierHead(d_proj)

    def forward(self, query_img, pos_prompts, neg_prompts):
        # Process query image
        query_embeds = self.vit_backbone(query_img)  # (B, 196, d_model)
        
        # Process positive prompts
        B, num_p = pos_prompts.shape[:2]
        pos_embeds = self.vit_backbone(pos_prompts.flatten(0, 1))  # (B*2, 196, d_model)
        pos_embeds = pos_embeds.view(B, num_p, *pos_embeds.shape[-2:])  # (B, 2, 196, d_model)
        
        # Process negative prompts
        neg_embeds = self.vit_backbone(neg_prompts.flatten(0, 1))
        neg_embeds = neg_embeds.view(B, num_p, *neg_embeds.shape[-2:])
        
        # Get prompt statistics and similarity projection
        stats, sim_proj = self.prompt_encoder(pos_embeds, neg_embeds)
        
        # Combine query embeddings with statistics
        combined = torch.cat([query_embeds, stats], dim=1)  # (B, 980, d_proj)
        
        # Transformer processing
        transformer_out = self.transformer(combined)  # (B, 980, d_proj)
        
        # Add similarity projection as new token
        final_tokens = torch.cat([
            transformer_out,
            sim_proj.unsqueeze(1)  # (B, 1, d_proj)
        ], dim=1)  # (B, 981, d_proj)
        
        # Classification
        logits = self.classifier(final_tokens)  # (B, 1)
        return logits.squeeze(-1)  # (B,)
    
    
# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    BATCH_SIZE = 12
    num_prompts = 2
    img_size = 224

    # Dummy data
    query_img = torch.randn(BATCH_SIZE, 3, img_size, img_size)
    pos_prompts = torch.randn(BATCH_SIZE, num_prompts, 3, img_size, img_size)
    neg_prompts = torch.randn(BATCH_SIZE, num_prompts, 3, img_size, img_size)

    # Initialize model with proper parameters
    model = PromptableBinaryClassifier(
         d_model=768,  # ViT model dimension
        d_proj=768  # Projection dimension  
        )

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    query_img = query_img.to(device)
    pos_prompts = pos_prompts.to(device)
    neg_prompts = neg_prompts.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(query_img, pos_prompts, neg_prompts)
    
    print("\nTest results:")
    print("Logits shape:", logits.shape)  # Should be (12,)
    print("Sample logits:", logits[:2].cpu().numpy())
    print("Device used:", device)