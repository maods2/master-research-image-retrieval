import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import matplotlib.pyplot as plt
import numpy as np

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 50
mask_ratio = 0.75

# Transformações com normalização do ImageNet
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carregar dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Modelo MAE
class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder com ViT pré-treinada
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.enc_embed_dim = self.vit.hidden_dim
        self.patch_size = 16
        self.num_patches = (224 // self.patch_size) ** 2
        
        # Decoder
        self.decoder_embed = 512
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.decoder_embed))
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.decoder_embed,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=8
        )
        self.decoder_pos_embed = nn.Embedding(self.num_patches, self.decoder_embed)
        self.decoder_head = nn.Linear(self.decoder_embed, self.patch_size**2 * 3)

    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        return x_masked, ids_keep, ids_shuffle

    def forward_encoder(self, x):
        # Processar entrada mantendo o class token
        x = self.vit._process_input(x)  # (B, num_patches, hidden_dim)
        cls_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches+1, hidden_dim)
        
        # Adicionar positional embedding completo
        x = x + self.vit.encoder.pos_embedding
        
        # Separar class token e patches
        cls_token = x[:, :1]
        patches = x[:, 1:]
        
        # Aplicar masking apenas nos patches
        patches_masked, ids_keep, ids_shuffle = self.random_masking(patches)
        
        # Recombinar com class token
        x = torch.cat([cls_token, patches_masked], dim=1)
        
        # Passar pelas camadas do encoder
        for blk in self.vit.encoder.layers[:-1]:
            x = blk(x)
        x = self.vit.encoder.ln(x)
        
        return x, ids_shuffle

    def forward_decoder(self, x, ids_shuffle):
        B = x.shape[0]
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Adicionar mask tokens
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.decoder_embed))
        
        # Adicionar positional embedding
        x = x + self.decoder_pos_embed.weight.unsqueeze(0)
        
        # Passar pelo decoder
        x = self.decoder(x)
        
        # Prever pixels
        x = self.decoder_head(x)
        return x

    def forward_loss(self, imgs, pred):
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        
        mask = torch.zeros_like(loss)
        mask[:, :int(self.num_patches * mask_ratio)] = 1
        return (loss * mask).sum() / mask.sum()

    def patchify(self, imgs):
        p = self.patch_size
        x = imgs.unfold(2, p, p).unfold(3, p, p)
        x = x.reshape(imgs.shape[0], 3, -1).transpose(1, 2)
        return x.reshape(imgs.shape[0], -1, p*p*3)

    def forward(self, imgs):
        latent, ids_shuffle = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_shuffle)
        loss = self.forward_loss(imgs, pred)
        return loss

# Inicializar modelo e otimizador
model = MaskedAutoencoder().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# Treinamento
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        
        optimizer.zero_grad()
        loss = model(images)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')

# Visualização dos resultados
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return tensor * std + mean

def plot_reconstructions(model, loader, num_samples=5):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(loader))
        images = images[:num_samples].to(device)
        
        # Reconstrução
        loss = model(images)
        pred = model.forward_decoder(*model.forward_encoder(images))
        
        # Converter para imagens
        original = denormalize(images)
        reconstructed = model.patchify(original)
        reconstructed = reconstructed.reshape(-1, 3, 224, 224)
        
        # Plot
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))
        for i in range(num_samples):
            axes[0, i].imshow(original[i].cpu().permute(1, 2, 0))
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructed[i].cpu().permute(1, 2, 0))
            axes[1, i].axis('off')
        plt.show()

plot_reconstructions(model, train_loader)