import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from typing import Dict, Any, Callable, Optional, Union, List
import numpy as np

from pipelines.training_pipes.base_trainer import BaseTrainer
from metrics.metric_base import MetricLoggerBase
from utils.checkpoint_utils import save_model_and_log_artifact
from dataloaders.dataset import StandardImageDataset

# --------------------------
# Encoder Base Class
# --------------------------
class EncoderBase(nn.Module):
    """Base class for all encoders"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        raise NotImplementedError('Subclasses must implement forward method')


# --------------------------
# CNN Encoder
# --------------------------
class CNNEncoder(EncoderBase):
    def __init__(self, latent_dim: int, in_channels: int = 3):
        super().__init__(latent_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


# --------------------------
# ViT Encoder
# --------------------------
class ViTEncoder(EncoderBase):
    def __init__(
        self,
        latent_dim: int,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        in_channels: int = 3,
    ):
        super().__init__(latent_dim)

        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projection of flattened patches
        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        # Positional embedding and cls token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=depth,
        )

        # Final projection to latent space
        self.to_latent = nn.Linear(dim, latent_dim)

    def forward(self, x):
        b, c, h, w = x.shape

        # Adjust patch size based on actual input dimensions
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            # If image size is not divisible by patch size, resize the image
            new_h = (h // p) * p
            new_w = (w // p) * p
            x = F.interpolate(
                x, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
            h, w = new_h, new_w

        num_patches = (h // p) * (w // p)

        # Reshape to patches
        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b, num_patches, c * p * p)

        # Project patches to embedding dimension
        x = self.patch_embedding(x)

        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding (truncate or pad if necessary)
        if x.size(1) > self.pos_embedding.size(1):
            x = x[:, : self.pos_embedding.size(1)]
        elif x.size(1) < self.pos_embedding.size(1):
            pad_size = self.pos_embedding.size(1) - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_size))

        x = x + self.pos_embedding[:, : x.size(1)]

        # Apply transformer
        x = self.transformer(x)

        # Get cls token output
        x = x[:, 0]

        return self.to_latent(x)


# --------------------------
# VQ-VAE Model
# --------------------------
class VQVAE(nn.Module):
    def __init__(
        self,
        encoder: EncoderBase,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.encoder = encoder
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        z = self.encoder(x)

        # Reshape for quantization
        z = z.view(-1, self.embedding_dim, 1, 1)

        # Calculate distances to embeddings
        d = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.matmul(
                z.view(-1, self.embedding_dim), self.embedding.weight.t()
            )
        )

        # Get nearest embedding
        min_encoding_indices = torch.argmin(d, dim=1)

        # Get quantized latent
        z_q = self.embedding(min_encoding_indices).view(
            -1, self.embedding_dim, 1, 1
        )

        # Decode
        x_recon = self.decoder(z_q)

        return x_recon, z, z_q, min_encoding_indices

    def get_embeddings(self, x):
        """
        Get embeddings for input images without reconstruction.
        Useful for downstream tasks like retrieval or classification.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            embeddings: Tensor of shape [batch_size, embedding_dim]
            indices: Indices of the codebook entries
        """
        # Encode
        z = self.encoder(x)

        # Reshape for quantization
        z = z.view(-1, self.embedding_dim, 1, 1)

        # Calculate distances to embeddings
        d = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.matmul(
                z.view(-1, self.embedding_dim), self.embedding.weight.t()
            )
        )

        # Get nearest embedding
        min_encoding_indices = torch.argmin(d, dim=1)

        # Get quantized latent
        z_q = self.embedding(min_encoding_indices)

        return z_q, min_encoding_indices


# --------------------------
# VQ-VAE Loss
# --------------------------
class VQVAELoss(nn.Module):
    def __init__(self, commitment_cost: float = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.mse_loss = nn.MSELoss()

    def forward(self, x_recon, x, z, z_q):
        # Reconstruction loss
        recon_loss = self.mse_loss(x_recon, x)

        # VQ loss
        vq_loss = self.mse_loss(
            z_q.detach(), z
        ) + self.commitment_cost * self.mse_loss(z_q, z.detach())

        # Total loss
        loss = recon_loss + vq_loss

        return loss, recon_loss, vq_loss


# --------------------------
# Dataset
# --------------------------
class ImageDataset(Dataset):
    def __init__(
        self, root_dir: str, transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )

        self.image_files = [
            f
            for f in os.listdir(root_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


# --------------------------
# VQ-VAE Trainer
# --------------------------
class VQVAETrain(BaseTrainer):
    def __init__(self, config: dict):
        self.config = config
        super().__init__()

        # Initialize loss function
        self.loss_fn = VQVAELoss(
            commitment_cost=config['model'].get('commitment_cost', 0.25)
        )

    def train_one_epoch(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: str,
        epoch: int,
    ) -> float:
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Training Epoch {epoch + 1}',
        )

        for batch_idx, (x, y) in progress_bar:
            x = x.to(device)

            # Forward pass
            x_recon, z, z_q, indices = model(x)

            # Calculate loss
            loss, recon_loss, vq_loss = loss_fn(x_recon, x, z, z_q)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    'loss': running_loss / (batch_idx + 1),
                    'recon_loss': recon_loss.item(),
                    'vq_loss': vq_loss.item(),
                }
            )

        return running_loss / len(train_loader)

    def __call__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        logger: Callable,
        metric_logger: MetricLoggerBase,
    ) -> nn.Module:
        device = config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        epochs = config['training']['epochs']

        model.to(device)
        checkpoint_path = None
        min_val_loss = float('inf')
        training_loss = []

        for epoch in range(epochs):
            # Train one epoch
            epoch_loss = self.train_one_epoch(
                model,
                loss_fn,
                optimizer,
                train_loader,
                device,
                epoch,
            )

            # Log metrics
            logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

            training_loss.append(epoch_loss)

            # Save model if best loss
            if epoch_loss < min_val_loss:
                min_val_loss = epoch_loss
                checkpoint_path = save_model_and_log_artifact(
                    metric_logger, config, model, filepath=checkpoint_path
                )

        # Log final metrics
        metrics = {
            'epoch_loss': training_loss,
        }
        metric_logger.log_json(metrics, 'train_metrics')

        return model


# --------------------------
# Example Usage
# --------------------------
def create_vqvae_model(config: dict) -> VQVAE:
    """Create a VQ-VAE model with the specified encoder type"""
    encoder_type = config['model'].get('encoder_type', 'cnn')
    latent_dim = config['model'].get('latent_dim', 64)

    if encoder_type.lower() == 'cnn':
        encoder = CNNEncoder(latent_dim=latent_dim)
    elif encoder_type.lower() == 'vit':
        encoder = ViTEncoder(latent_dim=latent_dim)
    else:
        raise ValueError(f'Unknown encoder type: {encoder_type}')

    return VQVAE(
        encoder=encoder,
        num_embeddings=config['model'].get('num_embeddings', 512),
        embedding_dim=config['model'].get('embedding_dim', 64),
        commitment_cost=config['model'].get('commitment_cost', 0.25),
    )


def create_dataloader(config: dict) -> DataLoader:
    """Create a dataloader for the VQ-VAE training"""
    dataset = StandardImageDataset(
        root_dir=config['data']['root_dir'],
        transform=A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        ),
    )

    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
    )


if __name__ == '__main__':
    config = {
        'model': {
            'encoder_type': 'vit',  # or 'vit'
            'latent_dim': 64,
            'num_embeddings': 512,
            'embedding_dim': 64,
            'commitment_cost': 0.25,
        },
        'data': {'root_dir': 'datasets/final/glomerulo/train'},
        'training': {'epochs': 100, 'batch_size': 32, 'num_workers': 4},
    }

    model = create_vqvae_model(config)
    dataloader = create_dataloader(config)
    trainer = VQVAETrain(config)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create a simple metric logger for testing
    class SimpleMetricLogger:
        def log_metric(self, name, value, step=None):
            print(f'{name}: {value}')

        def log_metrics(self, metrics, step=None):
            for name, value in metrics.items():
                print(f'{name}: {value}')

        def log_params(self, params):
            print(f'Parameters: {params}')

        def log_artifact(self, path):
            print(f'Saved artifact to: {path}')

        def log_json(self, data, filename):
            print(f'Logged JSON data to: {filename}')

    metric_logger = SimpleMetricLogger()

    trainer(
        model=model,
        loss_fn=trainer.loss_fn,
        optimizer=optimizer,
        train_loader=dataloader,
        test_loader=dataloader,  # Using same loader for testing
        config=config,
        logger=print,
        metric_logger=metric_logger,
    )
