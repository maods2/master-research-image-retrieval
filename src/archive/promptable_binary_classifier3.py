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
import math

# ----------------------------
# ViT Backbone
# ----------------------------
class ViTBackbonePreTrained(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViTBackbonePreTrained, self).__init__()
        # Creating the model without the classification layer
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.model.forward_features = self._forward_features  # Overriding the forward function
    
    def _forward_features(self, x):
        # Obtaining embeddings for all tokens
        out = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(out.shape[0], -1, -1)
        out = torch.cat((cls_token, out), dim=1)
        out = self.model.pos_drop(out)
        out = self.model.blocks(out)
        return out  # Returns all tokens
    
    def forward(self, x, cls_token_only=False):
        out = self.model.forward_features(x)
        if cls_token_only:
            return out[:, 0, :]
        return out


class ViTBackbone(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=6, num_heads=12, dropout=0.1):
        super(ViTBackbone, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Create patch embeddings using a conv layer.
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings for each patch.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
    
    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: token embeddings [B, num_patches, embed_dim]
        """
        B = x.size(0)
        # patch embedding; result shape: [B, embed_dim, H/patch_size, W/patch_size]
        x = self.patch_embed(x)  
        x = x.flatten(2).transpose(1, 2)  # shape: [B, num_patches, embed_dim]
        x = x + self.pos_embed  # add positional embeddings
        
        # Transformer expects sequence first; here we use batch-first transformer encoder
        x = self.transformer(x)  # shape remains [B, num_patches, embed_dim]
        return x

# ----------------------------
# Prompt Encoder
# ----------------------------
class PromptEncoder(nn.Module):
    def __init__(self, in_dim, proj_dim=256):
        """
        in_dim: Dimension of prompt embeddings (from the backbone)
        proj_dim: Dimension for projected prompt tokens.
        
        Expects:
            pos_embeds: [B, num_tokens, embed_dim]
            neg_embeds: [B, num_tokens, embed_dim]
            
        Returns a dictionary with:
            'sim_matrix': cosine similarity matrix [B, num_tokens, num_tokens]
            'pos_mean', 'pos_std': statistical features from positive tokens [B, proj_dim]
            'neg_mean', 'neg_std': statistical features from negative tokens [B, proj_dim]
        """
        super(PromptEncoder, self).__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
    
    def forward(self, pos_embeds, neg_embeds):
        """
        pos_embeds: [B, num_tokens, embed_dim]
        neg_embeds: [B, num_tokens, embed_dim]
        
        Returns a dictionary containing:
            'sim_matrix': cosine similarity between normalized pos and neg embeddings, shape [B, num_tokens, num_tokens]
            'pos_mean', 'pos_std': statistics from projected positive embeddings, shape [B, proj_dim]
            'neg_mean', 'neg_std': statistics from projected negative embeddings, shape [B, proj_dim]
        """
        # Normalize along feature dim (L2 norm)
        pos_norm = F.normalize(pos_embeds, p=2, dim=-1)  # [B, num_tokens, embed_dim]
        neg_norm = F.normalize(neg_embeds, p=2, dim=-1)  # [B, num_tokens, embed_dim]
        
        # Compute cosine similarity matrix between all tokens.
        sim_matrix = torch.matmul(pos_norm, neg_norm.transpose(1, 2))  # [B, num_tokens, num_tokens]
        
        # Project the tokens into a new space.
        pos_proj = self.proj(pos_embeds)  # [B, num_tokens, proj_dim]
        neg_proj = self.proj(neg_embeds)  # [B, num_tokens, proj_dim]
        
        # Compute statistical features over the token dimension.
        pos_mean = pos_proj.mean(dim=1)  # [B, proj_dim]
        pos_std  = pos_proj.std(dim=1)   # [B, proj_dim]
        neg_mean = neg_proj.mean(dim=1)  # [B, proj_dim]
        neg_std  = neg_proj.std(dim=1)   # [B, proj_dim]
        
        return {
            'sim_matrix': sim_matrix,
            'pos_mean': pos_mean,
            'pos_std': pos_std,
            'neg_mean': neg_mean,
            'neg_std': neg_std
        }

# ----------------------------
# Classifier Head
# ----------------------------
class ClassifierHead(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super(ClassifierHead, self).__init__()
        hidden_features = hidden_features or in_features // 2
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)  # binary output (logit)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------
# Promptable Binary Classifier
# ----------------------------
class PromptableBinaryClassifier(nn.Module):
    def __init__(self, num_prompts=2, image_size=224, patch_size=16, backbone_embed_dim=768,
                 vit_depth=6, vit_num_heads=12, prompt_proj_dim=256,
                 transformer_depth=2, transformer_dropout=0.1,
                 num_prompt_images=2):
        """
        num_prompt_images: Number of prompt images per type (positive and negative).
        The backbone will process each prompt image and we will flatten the patch tokens.
        """
        super(PromptableBinaryClassifier, self).__init__()
        
        # Backbone for both query and prompt images.
        self.backbone = ViTBackbone(image_size=image_size, patch_size=patch_size,
                                    in_channels=3, embed_dim=backbone_embed_dim,
                                    depth=vit_depth, num_heads=vit_num_heads)
        
        # Prompt encoder to process the prompt token sequences.
        self.prompt_encoder = PromptEncoder(in_dim=backbone_embed_dim, proj_dim=prompt_proj_dim)
        
        # Project the prompt statistics (mean, std) into the backbone dimension.
        self.prompt_proj = nn.Linear(prompt_proj_dim, backbone_embed_dim)
        
        # A small transformer encoder to fuse the query tokens with prompt statistics.
        encoder_layer = nn.TransformerEncoderLayer(d_model=backbone_embed_dim, nhead=vit_num_heads,
                                                   dropout=transformer_dropout)
        self.combined_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        
        # To project the flattened similarity matrix into a fixed backbone dimension.
        # Note: num_tokens will be: num_prompt_images * num_patches (from backbone).
        # We require this to be fixed, so we compute it here.
        dummy_img = torch.zeros(1, 3, image_size, image_size)
        dummy_tokens = self.backbone(dummy_img)  # [1, num_patches, backbone_embed_dim]
        num_patches = dummy_tokens.size(1)
        self.num_prompt_tokens = num_prompt_images * num_patches
        self.sim_proj = nn.Linear(self.num_prompt_tokens * self.num_prompt_tokens, backbone_embed_dim)
        
        # Final classifier head combining transformer pooled tokens and sim features.
        self.classifier_head = ClassifierHead(in_features=backbone_embed_dim * 2)
        self.num_prompt_images = num_prompt_images
    
    def forward(self, query_img, pos_prompts, neg_prompts):
        """
        query_img: [B, 3, 224, 224]
        pos_prompts: [B, num_prompt_images, 3, 224, 224]
        neg_prompts: [B, num_prompt_images, 3, 224, 224]
        """
        B = query_img.size(0)
        
        # Process the query image through the backbone.
        query_tokens = self.backbone(query_img)  # [B, num_patches, backbone_embed_dim]
        
        # Process prompts:
        # Flatten the prompt images: combine batch and prompt image dims.
        B_np, C, H, W = pos_prompts.size(0) * pos_prompts.size(1), pos_prompts.size(2), pos_prompts.size(3), pos_prompts.size(4)
        pos_prompts_flat = pos_prompts.view(B_np, C, H, W)
        neg_prompts_flat = neg_prompts.view(B_np, C, H, W)
        
        # Get token embeddings for each prompt image.
        pos_tokens = self.backbone(pos_prompts_flat)  # [B*num_prompt_images, num_patches, backbone_embed_dim]
        neg_tokens = self.backbone(neg_prompts_flat)  # [B*num_prompt_images, num_patches, backbone_embed_dim]
        
        # Instead of averaging over tokens, we flatten all tokens for each sample.
        num_patches = pos_tokens.size(1)
        # Reshape to combine prompt images with their patch tokens.
        pos_prompt_embeds = pos_tokens.view(B, self.num_prompt_images * num_patches, -1)  # [B, num_tokens, backbone_embed_dim]
        neg_prompt_embeds = neg_tokens.view(B, self.num_prompt_images * num_patches, -1)  # [B, num_tokens, backbone_embed_dim]
        
        # Pass the full token sequences to the prompt encoder.
        prompt_stats = self.prompt_encoder(pos_prompt_embeds, neg_prompt_embeds)
        # prompt_stats['sim_matrix'] has shape [B, num_tokens, num_tokens]
        
        # Compute statistical tokens: one token each for positive mean, positive std, negative mean, negative std.
        prompt_tokens = torch.stack([
            prompt_stats['pos_mean'],
            prompt_stats['pos_std'],
            prompt_stats['neg_mean'],
            prompt_stats['neg_std']
        ], dim=1)  # [B, 4, prompt_proj_dim]
        
        # Project these statistical tokens to the backbone dimension.
        prompt_tokens = self.prompt_proj(prompt_tokens)  # [B, 4, backbone_embed_dim]
        
        # Concatenate query tokens with the projected prompt tokens.
        combined_tokens = torch.cat([query_tokens, prompt_tokens], dim=1)  # [B, num_query_tokens+4, backbone_embed_dim]
        
        # Fuse the tokens using a small transformer.
        combined_tokens = combined_tokens.transpose(0, 1)  # [S, B, E]
        fused_tokens = self.combined_transformer(combined_tokens)  # [S, B, E]
        fused_tokens = fused_tokens.transpose(0, 1)  # [B, S, E]
        
        # Pool over the sequence dimension (e.g. average pooling).
        fused_repr = fused_tokens.mean(dim=1)  # [B, backbone_embed_dim]
        
        # Flatten the similarity matrix from the prompt encoder.
        # sim_matrix shape: [B, num_tokens, num_tokens] where num_tokens = self.num_prompt_images * num_patches.
        sim_matrix_flat = prompt_stats['sim_matrix'].view(B, -1)
        sim_features = self.sim_proj(sim_matrix_flat)  # [B, backbone_embed_dim]
        
        # Concatenate the fused representation with the similarity features.
        final_features = torch.cat([fused_repr, sim_features], dim=1)  # [B, 2*backbone_embed_dim]
        
        # Obtain the final binary logit.
        logits = self.classifier_head(final_features)  # [B, 1]
        return logits
# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    
    from pbc_dataset import PromptDataset, custom_collate
    from torch.utils.data import DataLoader
    from promptable_binary_classifier2 import Trainer, test_overfitting
    
    # Assume BATCH_SIZE=12, num_prompts=2, and image size 224x224.
    BATCH_SIZE = 24
    EPOCHS = 1
    LR = 1e-4
    num_prompts = 2
    img_size = 224
    
    # Instantiate the model.
    model = PromptableBinaryClassifier(num_prompts=2, image_size=img_size, patch_size=16, backbone_embed_dim=768,
                                         vit_depth=6, vit_num_heads=12, prompt_proj_dim=256,
                                         transformer_depth=2, transformer_dropout=0.1)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # Dataset and DataLoader
    train_dataset = PromptDataset("datasets/final/terumo/train", num_prompts=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate,
        # num_workers=4
    )

    
    # Option 1: Normal training
    # print("Starting normal training...")
    normal_trainer = Trainer(model, train_loader)
    normal_trainer.train(epochs=EPOCHS, lr=LR)
    # normal_trainer.plot_training_history()
    
    # Option 2: Overfitting test (uncomment to run)
    # test_overfitting(model, train_loader, epochs=50, lr=1e-3)
    
    
    # # Create dummy data.
    # query_img = torch.randn(BATCH_SIZE, 3, img_size, img_size)
    # pos_prompts = torch.randn(BATCH_SIZE, num_prompts, 3, img_size, img_size)
    # neg_prompts = torch.randn(BATCH_SIZE, num_prompts, 3, img_size, img_size)
    # # Move data to device.
    # device = next(model.parameters()).device
    # query_img = query_img.to(device)
    # pos_prompts = pos_prompts.to(device)
    # neg_prompts = neg_prompts.to(device)
    
    # # Forward pass.
    # logits = model(query_img, pos_prompts, neg_prompts)
    # print("Logits shape:", logits.shape)

