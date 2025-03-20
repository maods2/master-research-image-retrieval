import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import autocast, GradScaler

# ------------------
# Architecture
# ------------------

class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=4):
        super().__init__()
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4*embed_dim
            ),
            num_layers=num_layers
        )
        self.context_proj = nn.Linear(2*embed_dim, embed_dim)
        
    def forward(self, pos_emb, neg_emb):
        """
        pos_emb: [B, N, D] - positive prompt embeddings
        neg_emb: [B, M, D] - negative prompt embeddings
        Returns: [B, D] context embedding
        """
        # Concatenate positive and negative embeddings
        combined = torch.cat([pos_emb, neg_emb], dim=1)  # [B, N+M, D]
        
        # Process through transformer
        encoded = self.transformer(combined)  # [B, N+M, D]
        
        # Aggregate using CLS token (first position)
        context = encoded[:, 0, :]  # [B, D]
        return self.context_proj(context)

class PromptableBinaryClassifier(nn.Module):
    def __init__(self, num_classes=1, num_prompts=10):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.embed_dim = self.backbone.embed_dim
        self.num_prompts = num_prompts
        
        # Prompt processing
        self.prompt_encoder = PromptEncoder(embed_dim=self.embed_dim)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=12,
                    dim_feedforward=4*self.embed_dim
                ),
                num_layers=2
            )
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    def _get_cls_embeddings(self, x):
        """Extract CLS token embeddings from input images"""
        features = self.backbone(x)  # [B, num_tokens, D]
        return features[:, 0, :]  # CLS token [B, D]

    def forward(self, query_img, pos_imgs, neg_imgs):
        """
        query_img: [B, C, H, W]
        pos_imgs: [B, N, C, H, W] - positive prompt images
        neg_imgs: [B, M, C, H, W] - negative prompt images
        """
        B = query_img.shape[0]
        
        # Get query embedding
        query_emb = self._get_cls_embeddings(query_img)  # [B, D]
        
        # Process positive prompts
        pos_flat = pos_imgs.view(B*self.num_prompts, *pos_imgs.shape[2:])
        pos_embs = self._get_cls_embeddings(pos_flat)
        pos_embs = pos_embs.view(B, self.num_prompts, -1)  # [B, N, D]
        
        # Process negative prompts
        neg_flat = neg_imgs.view(B*self.num_prompts, *neg_imgs.shape[2:])
        neg_embs = self._get_cls_embeddings(neg_flat)
        neg_embs = neg_embs.view(B, self.num_prompts, -1)  # [B, M, D]
        
        # Encode prompts
        prompt_context = self.prompt_encoder(pos_embs, neg_embs)  # [B, D]
        
        # Combine with query embedding
        combined = query_emb + prompt_context
        
        # Final fusion
        fused = self.fusion(combined.unsqueeze(1)).squeeze(1)
        
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
        pos_prompts = [self.transform(img) for img in pos_list[:self.num_prompts]]
        while len(pos_prompts) < self.num_prompts:
            pos_prompts.append(torch.zeros_like(pos_prompts[0]))
            
        # Process negative prompts
        neg_prompts = [self.transform(img) for img in neg_list[:self.num_prompts]]
        while len(neg_prompts) < self.num_prompts:
            neg_prompts.append(torch.zeros_like(neg_prompts[0]))
            
        return {
            'query': query,
            'pos_imgs': torch.stack(pos_prompts),
            'neg_imgs': torch.stack(neg_prompts),
            'label': torch.tensor(label, dtype=torch.float)
        }
    
    def default_transform(self):
        return nn.Sequential(
            nn.Resize((224, 224)),
            nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        
        self.optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-5},
            {'params': model.prompt_encoder.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 1e-4}
        ])
        
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
                    batch['neg_imgs'].to(self.device)
                )
                loss = self.criterion(outputs.squeeze(), batch['label'].to(self.device))
            
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
                    batch['neg_imgs'].to(self.device)
                )
                loss = self.criterion(outputs.squeeze(), batch['label'].to(self.device))
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == batch['label'].to(self.device)).sum().item()
                total += batch['label'].shape[0]
                
        return total_loss / len(self.loaders['val']), correct / total

# ------------------
# Usage Example
# ------------------

if __name__ == "__main__":
    # Example dummy data
    dummy_data = [
        (torch.rand(3, 224, 224),  # Query
         [torch.rand(3, 224, 224) for _ in range(10)],  # Positives
         [torch.rand(3, 224, 224) for _ in range(10)],  # Negatives
         1)  # Label
    ] * 1000
    
    dataset = PromptDataset(dummy_data, num_prompts=10)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = PromptableBinaryClassifier(num_classes=1)
    trainer = Trainer(model, train_loader, train_loader)
    
    for epoch in range(10):
        train_loss = trainer.train_epoch()
        val_loss, val_acc = trainer.validate()
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")