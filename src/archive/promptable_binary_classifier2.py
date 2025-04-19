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

# -------------------------------
# 1. Shared ViT Backbone
# -------------------------------
class ViTBackbone(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViTBackbone, self).__init__()
        # Create the ViT model without the classification layer
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

    def forward(self, x):
        # Process x and return the CLS token (assuming the model returns [B, tokens, embed_dim])
        out = self.model(x)
        if out.dim() == 3:
            return out[:, 0, :]  # return the CLS token
        return out


# -------------------------------
# 2. Prompt Encoder
# -------------------------------
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, num_transformer_layers=2, num_heads=4):
        super(PromptEncoder, self).__init__()
        # Projection for prompt embeddings
        self.proj = nn.Linear(embed_dim, embed_dim)
        # Transformer layers to process the concatenation of embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        # Adjustment to concatenate positive and negative embeddings
        self.adjust = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, pos_embeds, neg_embeds):
        """
        pos_embeds, neg_embeds: tensors of shape [B, embed_dim]
        """
        # 1. Projection of embeddings
        pos_proj = self.proj(pos_embeds)  # [B, embed_dim]
        neg_proj = self.proj(neg_embeds)  # [B, embed_dim]

        # 2. (Optional) Can calculate cosine similarity, masking, etc.
        # Example: cos_sim = F.cosine_similarity(pos_proj.unsqueeze(1), neg_proj.unsqueeze(0), dim=-1)

        # 3. Concatenate and adjust dimensions
        combined = torch.cat([pos_proj, neg_proj], dim=-1)  # [B, 2*embed_dim]
        combined = self.adjust(combined).unsqueeze(
            1
        )        # [B, 1, embed_dim]

        # 4. Processing with Transformer
        transformer_out = self.transformer(
            combined
        )         # [B, 1, embed_dim]

        # 5. Aggregation (removing sequence dimension)
        prompt_context = transformer_out.squeeze(1)          # [B, embed_dim]
        return prompt_context


# -------------------------------
# 3. Attention Fusion
# -------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(AttentionFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

    def forward(self, query_embed, prompt_context):
        """
        query_embed: [B, embed_dim]
        prompt_context: [B, embed_dim]
        """
        # Expand to include sequence dimension (size 1)
        query = prompt_context.unsqueeze(1)      # [B, 1, embed_dim]
        key = prompt_context.unsqueeze(1)     # [B, 1, embed_dim]
        value = query_embed.unsqueeze(1)     # [B, 1, embed_dim]
        attn_output, _ = self.multihead_attn(query, key, value)
        fused = attn_output.squeeze(1)          # [B, embed_dim]
        return fused


# -------------------------------
# 4. Classifier Head
# -------------------------------
class ClassifierHead(nn.Module):
    def __init__(self, embed_dim):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(embed_dim, 1)  # Binary output

    def forward(self, x):
        logits = self.fc(x)
        return logits


# -------------------------------
# 5. Complete Model: Promptable Binary Classifier
# -------------------------------
class PromptableBinaryClassifier(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', embed_dim=768):
        super(PromptableBinaryClassifier, self).__init__()
        # Use a single shared ViT backbone for all inputs
        self.shared_backbone = ViTBackbone(vit_model_name)

        # Prompt Encoder to process prompt embeddings
        self.prompt_encoder = PromptEncoder(
            embed_dim, num_transformer_layers=1, num_heads=4
        )

        # Attention Fusion module to combine query and prompt context
        self.attn_fusion = AttentionFusion(embed_dim, num_heads=4)

        # Classifier head for the final binary output
        self.classifier = ClassifierHead(embed_dim)

    def forward(self, query_img, pos_prompts, neg_prompts):
        """
        query_img: tensor [B, 3, 224, 224]
        pos_prompts: tensor [B, N, 3, 224, 224]
        neg_prompts: tensor [B, N, 3, 224, 224]
        """
        B = query_img.size(0)

        # Process the query image
        q_embed = self.shared_backbone(query_img)  # [B, embed_dim]

        # Process positive prompts
        B, N, C, H, W = pos_prompts.shape
        pos_prompts = pos_prompts.view(B * N, C, H, W)
        pos_embeds = self.shared_backbone(pos_prompts)  # [B*N, embed_dim]
        pos_embeds = pos_embeds.view(B, N, -1)           # [B, N, embed_dim]
        pos_agg = pos_embeds.mean(
            dim=1
        )                 # Aggregate via mean -> [B, embed_dim]

        # Process negative prompts
        B, N, C, H, W = neg_prompts.shape
        neg_prompts = neg_prompts.view(B * N, C, H, W)
        neg_embeds = self.shared_backbone(neg_prompts)  # [B*N, embed_dim]
        neg_embeds = neg_embeds.view(B, N, -1)           # [B, N, embed_dim]
        neg_agg = neg_embeds.mean(dim=1)                 # [B, embed_dim]

        # Get prompt context via Prompt Encoder
        prompt_context = self.prompt_encoder(
            pos_agg, neg_agg
        )  # [B, embed_dim]

        # Fuse query embeddings with prompt context
        fused = self.attn_fusion(q_embed, prompt_context)  # [B, embed_dim]

        # Final classification
        logits = self.classifier(fused)  # [B, 1]
        return logits


def train_model(model, dataloader, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            # Move data to GPU
            query = batch['query'].to(device)
            pos = batch['pos_imgs'].to(device)
            neg = batch['neg_imgs'].to(device)
            labels = batch['query_label'].float().to('cuda')

            # Zero grad
            optimizer.zero_grad()

            # Forward pass
            logits = model(query, pos, neg).squeeze()

            # Calculate loss
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * query.size(0)
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix(
                {'loss': running_loss / total, 'acc': correct / total}
            )

        # Calculate epoch accuracy
        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model


class Trainer:
    def __init__(self, model, dataloader, device='cuda'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.history = {'train_loss': [], 'train_acc': []}

    def train(self, epochs=10, lr=1e-4, overfit_batch=False):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Overfit mode: use a single batch
        if overfit_batch:
            single_batch = next(iter(self.dataloader))
            print('\nOverfit mode activated - using a single batch')

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Progress bar
            if overfit_batch:
                progress_bar = tqdm(
                    [single_batch], desc=f'Epoch {epoch+1}/{epochs}'
                )
            else:
                progress_bar = tqdm(
                    self.dataloader, desc=f'Epoch {epoch+1}/{epochs}'
                )

            for batch in progress_bar:
                # Prepare data
                query = batch['query'].to(self.device)
                pos = batch['pos_imgs'].to(self.device)
                neg = batch['neg_imgs'].to(self.device)
                labels = batch['query_label'].float().to('cuda')

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                logits = self.model(query, pos, neg).squeeze()

                # Calculate loss
                loss = criterion(logits, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item() * query.size(0)
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Update progress bar
                progress_bar.set_postfix(
                    {'loss': running_loss / total, 'acc': correct / total}
                )

            # Store metrics
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)

            print(
                f'Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}'
            )

    def plot_training_history(self):
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


def test_overfitting(model, dataloader, epochs=50, lr=1e-3):
    # Get a single batch
    single_batch = next(iter(dataloader))

    # Create trainer with overfit mode
    trainer = Trainer(model, dataloader)

    print('Starting overfitting test on a single batch...')
    trainer.train(epochs=epochs, lr=lr, overfit_batch=True)

    # Plot results
    trainer.plot_training_history()

    # Check if overfitting was achieved (accuracy should approach 1.0)
    final_acc = trainer.history['train_acc'][-1]
    print(f'\nFinal accuracy on batch: {final_acc:.4f}')
    if final_acc > 0.95:
        print('✅ Model successfully overfitted on the batch (as expected)')
    else:
        print('❌ Model failed to overfit - check architecture/learning rate')


# -------------------------------
# Example usage
# -------------------------------
if __name__ == '__main__':

    from pbc_dataset import PromptDataset, custom_collate
    from torch.utils.data import DataLoader

    BATCH_SIZE = 12
    EPOCHS = 1
    LR = 1e-4

    # Dataset and DataLoader
    train_dataset = PromptDataset('datasets/final/terumo/train', num_prompts=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate,
        # num_workers=4
    )

    # Model
    model = PromptableBinaryClassifier()

    # Option 1: Normal training
    print('Starting normal training...')
    normal_trainer = Trainer(model, train_loader)
    normal_trainer.train(epochs=EPOCHS, lr=LR)
    normal_trainer.plot_training_history()

    # Option 2: Overfitting test (uncomment to run)
    # test_overfitting(model, train_loader, epochs=50, lr=1e-3)

    # model.to('cuda')
    # for batch in dataloader:

    #     query_img = batch['query'].to('cuda')
    #     pos_prompts = batch['pos_imgs'].to('cuda')
    #     neg_prompts = batch['neg_imgs'].to('cuda')
    #     labels = batch['query_label'].to('cuda')

    #     print("\nPrompt shapes:")
    #     print(f"Query: {query_img.shape}")
    #     print(f"Positives: {pos_prompts.shape}")
    #     print(f"Negatives: {neg_prompts.shape}")
    #     logits = model(query_img, pos_prompts, neg_prompts)
    #     print("Logits:", logits.shape)
    #     print("Logits:", logits)
    #     print("Labels:", labels)
    #     break
