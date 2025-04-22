import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange
from timm.models.vision_transformer import vit_base_patch16_224

from tqdm import tqdm

# ----------------------------
# CONFIGURAÇÕES
# ----------------------------
BATCH_SIZE = 64
IMG_SIZE = 224
EMBED_DIM = 768  # compatível com ViT base
NUM_CODEBOOK_ENTRIES = 512
MASK_RATIO = 0.4

# ----------------------------
# DATASET (CIFAR10 como exemplo)
# ----------------------------
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

dataset = datasets.CIFAR10(
    root='./data', train=True, transform=transform, download=True
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# VQ-VAE CODEBOOK MODULE
# ----------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(
            -1 / num_embeddings, 1 / num_embeddings
        )
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.codebook.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(
            flat_inputs.dtype
        )
        quantized = torch.matmul(encodings, self.codebook.weight).view_as(
            inputs
        )
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices.view(inputs.shape[:-1])


# ----------------------------
# MAE-STYLE DECODER
# ----------------------------
class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim=768, out_channels=3, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.linear = nn.Linear(
            embed_dim, patch_size * patch_size * out_channels
        )

    def forward(self, x, h, w):
        x = self.linear(x)
        x = rearrange(
            x,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            p1=self.patch_size,
            p2=self.patch_size,
            h=h,
            w=w,
        )
        return x


# ----------------------------
# MODELO COMPLETO
# ----------------------------
class MAEVQModel(nn.Module):
    def __init__(self, vit, codebook, decoder):
        super().__init__()
        self.encoder = vit.patch_embed
        self.codebook = codebook
        self.decoder = decoder

    def forward(self, x, mask_ratio=0.4):
        x = self.encoder(x)
        B, D, H, W = x.shape
        x = rearrange(x, 'b d h w -> b (h w) d')
        N = x.shape[1]

        mask = torch.rand(B, N, device=x.device) < mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0

        quantized, vq_loss, _ = self.codebook(x_masked)
        recon = self.decoder(quantized, h=H, w=W)

        return recon, vq_loss

    def extract_embeddings(self, x):
        x = self.encoder(x)
        B, D, H, W = x.shape
        x = rearrange(x, 'b d h w -> b (h w) d')
        quantized, _, _ = self.codebook(x)
        return quantized.mean(dim=1)


# ----------------------------
# TREINAMENTO
# ----------------------------
vit = vit_base_patch16_224(pretrained=True)
codebook = VectorQuantizer(NUM_CODEBOOK_ENTRIES, EMBED_DIM)
decoder = SimpleDecoder(embed_dim=EMBED_DIM)

model = MAEVQModel(vit, codebook, decoder)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, _ in tqdm(dataloader, desc=f'Epoch {epoch + 1}'):
        imgs = imgs.to(device)
        recon, vq_loss = model(imgs, mask_ratio=MASK_RATIO)
        recon_loss = F.mse_loss(recon, imgs)
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}')

# ----------------------------
# EMBEDDING EXTRACTION (EXEMPLO PARA RETRIEVAL)
# ----------------------------
def generate_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            emb = model.extract_embeddings(imgs)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


# embeddings = generate_embeddings(model, dataloader, device)
# print("Shape dos embeddings:", embeddings.shape)
