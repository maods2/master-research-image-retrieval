import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torch.nn.functional as F

# ------------------
# Architecture
# ------------------


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Cross-attention between positive and negative embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # Similarity projection layers
        self.pos_proj = nn.Linear(embed_dim, embed_dim)
        self.neg_proj = nn.Linear(embed_dim, embed_dim)

        # Context transformer
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
            ),
            num_layers=num_layers,
        )

        # Similarity metrics
        self.sim_weight = nn.Parameter(torch.tensor(0.1))
        self.context_proj = nn.Linear(3 * embed_dim, embed_dim)
        self.combined_proj = nn.Linear(
            3 * embed_dim, embed_dim
        )  # New projection layer

    def forward(self, pos_emb, neg_emb):
        """
        pos_emb: [B, N, D] - positive prompt embeddings
        neg_emb: [B, M, D] - negative prompt embeddings
        Returns: [B, D] context embedding
        """
        B, N, D = pos_emb.shape
        M = neg_emb.shape[1]

        # Project embeddings
        pos_proj = self.pos_proj(pos_emb)  # [B, N, D]
        neg_proj = self.neg_proj(neg_emb)  # [B, M, D]

        # Compute cross-attention between positive and negative
        attn_output, _ = self.cross_attn(
            query=pos_proj, key=neg_proj, value=neg_proj, need_weights=False
        )  # [B, N, D]

        # Compute cosine similarities
        pos_neg_sim = F.cosine_similarity(
            pos_proj.unsqueeze(2),  # [B, N, 1, D]
            neg_proj.unsqueeze(1),  # [B, 1, M, D]
            dim=-1,
        )  # [B, N, M]
        sim_mask = (pos_neg_sim > 0.5).float()  # Learnable threshold
        sim_context = torch.einsum(
            'bnm,bmd->bnd', sim_mask, neg_proj
        )  # [B, N, D]

        # Combine features
        combined = torch.cat(
            [pos_emb, attn_output, self.sim_weight * sim_context], dim=-1
        )  # [B, N, 3D]

        # Project combined features back to embed_dim
        combined = self.combined_proj(combined)  # [B, N, D]

        # Process through transformer
        encoded = self.transformer(combined)  # [B, N, D]

        # Aggregate using attention pooling
        cls_token = encoded.mean(dim=1)  # [B, D]
        max_token = encoded.max(dim=1).values  # [B, D]
        std_token = encoded.std(dim=1)  # [B, D]

        context = torch.cat([cls_token, max_token, std_token], dim=-1)
        return self.context_proj(context)


class PromptableBinaryClassifier(nn.Module):
    def __init__(self, num_classes=1, num_prompts=10):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=True, num_classes=0
        )
        self.embed_dim = self.backbone.embed_dim
        self.num_prompts = num_prompts

        # Prompt processing
        self.prompt_encoder = PromptEncoder(embed_dim=self.embed_dim)

        # Dynamic attention fusion
        self.fusion_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads=8, batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

    def _get_cls_embeddings(self, x):
        features = self.backbone(x)
        return features

    def forward(self, query_img, pos_imgs, neg_imgs):
        B = query_img.shape[0]

        # Get embeddings
        query_emb = self._get_cls_embeddings(query_img)
        pos_flat = pos_imgs.view(B * self.num_prompts, *pos_imgs.shape[2:])
        pos_embs = self._get_cls_embeddings(pos_flat).view(
            B, self.num_prompts, -1
        )
        neg_flat = neg_imgs.view(B * self.num_prompts, *neg_imgs.shape[2:])
        neg_embs = self._get_cls_embeddings(neg_flat).view(
            B, self.num_prompts, -1
        )

        # Encode prompts with similarity
        prompt_context = self.prompt_encoder(pos_embs, neg_embs)

        # Attention-based fusion
        fused, _ = self.fusion_attn(
            query=query_emb.unsqueeze(1),
            key=prompt_context.unsqueeze(1),
            value=prompt_context.unsqueeze(1),
        )
        fused = fused.squeeze(1)

        return self.classifier(fused)


# ------------------
# Dataset
# ------------------


class PromptDataset(Dataset):
    def __init__(self, samples, num_prompts=10, transform=None):
        """
        samples: List of tuples (
            query_img,
            [pos_img1, pos_img2, ...],
            [neg_img1, neg_img2, ...],
            label
        )
        """
        self.samples = samples
        self.num_prompts = num_prompts
        self.transform = transform or self.default_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, pos_list, neg_list, label = self.samples[idx]

        # Process query
        query = self.transform(query)

        # Process positive prompts (pad/crop to num_prompts)
        pos_prompts = [
            self.transform(img) for img in pos_list[: self.num_prompts]
        ]
        while len(pos_prompts) < self.num_prompts:
            pos_prompts.append(torch.zeros_like(pos_prompts[0]))

        # Process negative prompts
        neg_prompts = [
            self.transform(img) for img in neg_list[: self.num_prompts]
        ]
        while len(neg_prompts) < self.num_prompts:
            neg_prompts.append(torch.zeros_like(neg_prompts[0]))

        return {
            'query': query,
            'pos_imgs': torch.stack(pos_prompts),
            'neg_imgs': torch.stack(neg_prompts),
            'label': torch.tensor(label, dtype=torch.float),
        }

    def default_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


# ------------------
# Training Pipeline
# ------------------


class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.loaders = {'train': train_loader, 'val': val_loader}
        self.device = device
        self.scaler = GradScaler()

        self.optimizer = torch.optim.AdamW(
            [
                {'params': model.backbone.parameters(), 'lr': 1e-5},
                {'params': model.prompt_encoder.parameters(), 'lr': 1e-4},
                {'params': model.classifier.parameters(), 'lr': 1e-4},
            ]
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.loaders['train']:
            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(
                    batch['query'].to(self.device),
                    batch['pos_imgs'].to(self.device),
                    batch['neg_imgs'].to(self.device),
                )
                loss = self.criterion(
                    outputs.squeeze(), batch['label'].to(self.device)
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.loaders['train'])

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.loaders['val']:
                outputs = self.model(
                    batch['query'].to(self.device),
                    batch['pos_imgs'].to(self.device),
                    batch['neg_imgs'].to(self.device),
                )
                loss = self.criterion(
                    outputs.squeeze(), batch['label'].to(self.device)
                )
                total_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                correct += (
                    (preds == batch['label'].to(self.device)).sum().item()
                )
                total += batch['label'].shape[0]

        return total_loss / len(self.loaders['val']), correct / total


# ------------------
# Usage Example
# ------------------

if __name__ == '__main__':
    # Example dummy data
    dummy_data = [
        (
            torch.rand(3, 224, 224),  # Query
            [torch.rand(3, 224, 224) for _ in range(10)],  # Positives
            [torch.rand(3, 224, 224) for _ in range(10)],  # Negatives
            1,
        )  # Label
    ] * 1000

    dataset = PromptDataset(dummy_data, num_prompts=10)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = PromptableBinaryClassifier(num_classes=1)
    trainer = Trainer(model, train_loader, train_loader)

    for epoch in range(10):
        train_loss = trainer.train_epoch()
        val_loss, val_acc = trainer.validate()
        print(
            f'Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}'
        )
