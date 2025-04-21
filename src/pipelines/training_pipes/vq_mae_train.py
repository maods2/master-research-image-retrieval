import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import timm
import random


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0/self.num_embeddings, 1.0/self.num_embeddings)

    def forward(self, inputs):
        # Flatten input: (B, D, H, W) -> (B*H*W, D)
        input_shape = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Compute distances
        distances = (flat_input.pow(2).sum(1, keepdim=True)
                     - 2 * flat_input @ self.embedding.weight.t()
                     + self.embedding.weight.pow(2).sum(1, keepdim=True).t())

        # Get encoding indices
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)

        # Quantize and unflatten
        quantized = encodings @ self.embedding.weight
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])


class VQMAE(nn.Module):
    def __init__(self,
                 backbone='resnet18',
                 img_size=224,
                 patch_size=16,
                 num_embeddings=512,
                 embedding_dim=512,
                 mask_ratio=0.75):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Encoder
        if backbone.startswith('resnet'):
            res = getattr(models, backbone)(pretrained=False)
            self.encoder = nn.Sequential(*list(res.children())[:-2])  # remove avgpool & fc
            self.enc_out_dim = res.fc.in_features
            self.grid_size = (img_size // 32, img_size // 32)  # resnet downscale 32x
        elif 'vit' in backbone:
            vit = timm.create_model(backbone, pretrained=False)
            self.encoder = vit.patch_embed  # patch embedding
            self.pos_embed = vit.pos_embed
            self.cls_token = vit.cls_token
            self.encoder_blocks = vit.blocks
            self.enc_out_dim = vit.embed_dim
            self.grid_size = (img_size // patch_size, img_size // patch_size)
        else:
            raise ValueError('Unsupported backbone')

        # Vector Quantizer
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder (simple convtranspose)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, embedding_dim//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embedding_dim//2), nn.ReLU(True),
            nn.ConvTranspose2d(embedding_dim//2, embedding_dim//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embedding_dim//4), nn.ReLU(True),
            nn.ConvTranspose2d(embedding_dim//4, 3, kernel_size=4, stride=2, padding=1)
        )

    def random_mask(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ph, pw = self.patch_size, self.patch_size
        gh, gw = H // ph, W // pw
        mask = torch.rand(B, gh * gw, device=x.device) < self.mask_ratio
        mask = mask.view(B, 1, gh, gw).repeat(1, C, ph, pw).reshape(B, C, H, W)
        x_masked = x.clone()
        x_masked[mask] = 0
        return x_masked, mask

    def forward(self, x):
        x_masked, mask = self.random_mask(x)
        # Encode
        if hasattr(self, 'encoder_blocks'):
            x_p = self.encoder(x_masked)  # patch embeddings
            B, n, _ = x_p.size()
            x_p = x_p + self.pos_embed[:, 1:n+1, :]
            for blk in self.encoder_blocks:
                x_p = blk(x_p)
            feat = x_p.transpose(1, 2).reshape(B, self.enc_out_dim,
                                              self.grid_size[0], self.grid_size[1])
        else:
            feat = self.encoder(x_masked)
        # Quantize
        quantized, vq_loss, _ = self.quantizer(feat)
        # Decode
        recon = self.decoder(quantized)
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        return recon, loss

    def extract_embeddings(self, x):
        # Full forward without mask, return quantized embeddings and indices
        if hasattr(self, 'encoder_blocks'):
            x_p = self.encoder(x)
            B, n, _ = x_p.size()
            x_p = x_p + self.pos_embed[:, 1:n+1, :]
            for blk in self.encoder_blocks:
                x_p = blk(x_p)
            feat = x_p.transpose(1, 2).reshape(B, self.enc_out_dim,
                                              self.grid_size[0], self.grid_size[1])
        else:
            feat = self.encoder(x)
        quantized, _, indices = self.quantizer(feat)
        return quantized, indices


def get_data_loaders(data_dir, img_size=224, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader


def train(model, data_loader, epochs=10, lr=1e-4, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            recon, loss = model(imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")


# Example usage:
# loader = get_data_loaders('/path/to/images')
# model = VQMAE(backbone='resnet18')\# or backbone='vit_base_patch16_224'
# train(model, loader)
