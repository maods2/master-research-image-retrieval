import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
# 1. DATASET FEW‑SHOT A PARTIR DE PASTAS
class FewShotFolderDataset(Dataset):
    def __init__(self, root_dir, n_way=2, k_shot=5, q_queries=5, transform=None):
        """
        root_dir: pasta raiz com subpastas por classe
        n_way: número de classes por episódio
        k_shot: support shots por classe
        q_queries: query shots por classe
        """
        self.root_dir = Path(root_dir)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.transform = transform

        # Mapeia cada classe para lista de caminhos de imagem
        self.class_to_images = {}
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                images = []
                for ext in image_extensions:
                    images.extend(class_dir.rglob(ext))
                self.class_to_images[class_dir.name] = images

        self.classes = [(i, cls) for i, cls in enumerate(list(self.class_to_images.keys()))]

    def __len__(self):
        # define um tamanho arbitrário de episódios por época
        return 1000

    def __getitem__(self, idx):
        # Seleciona n_way classes
        selected = random.sample(self.classes, self.n_way)
        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []

        for i, cls in selected:
            imgs = random.sample(self.class_to_images[cls], self.k_shot + self.q_queries)
            support_paths = imgs[:self.k_shot]
            query_paths   = imgs[self.k_shot:]

            for p in support_paths:
                img = Image.open(p).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                support_imgs.append(img)
                support_lbls.append(i)

            for p in query_paths:
                img = Image.open(p).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                query_imgs.append(img)
                query_lbls.append(i)

        support = torch.stack(support_imgs)      # [n_way*k_shot, C, H, W]
        query   = torch.stack(query_imgs)        # [n_way*q_queries, C, H, W]
        return support, torch.tensor(support_lbls), query, torch.tensor(query_lbls)


# 2. TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# 3. EMBEDDING CNN
class ConvEmbedding(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 24 * 24, out_dim)
        )
    def forward(self, x):
        return self.encoder(x)

# 4. PROTOTYPICAL LOSS
def prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels, n_way):
    # calcula protótipos: média por classe
    prototypes = torch.stack([
        support_embeddings[support_labels == i].mean(0)
        for i in range(n_way)
    ])  # [n_way, dim]
    # distâncias query → protótipos
    dists = torch.cdist(query_embeddings, prototypes)  # [n_query, n_way]
    log_p = (-dists).log_softmax(dim=1)
    loss = nn.NLLLoss()(log_p, query_labels)
    acc  = (log_p.argmax(1) == query_labels).float().mean().item()
    return loss, acc

# 5. TREINO
def train_few_shot(model, dataloader, optimizer, device, n_way, epochs=5):
    history = {"loss": [], "acc": []}
    model.train()
    for epoch in range(epochs):
        total_loss, total_acc = 0, 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for support, s_lbls, query, q_lbls in progress_bar:
            # support: [1, n_way*k, C, H, W] → remover batch dim
            support = support.squeeze(0).to(device)
            s_lbls  = s_lbls.squeeze(0).to(device)
            query   = query.squeeze(0).to(device)
            q_lbls  = q_lbls.squeeze(0).to(device)

            optimizer.zero_grad()
            emb_s = model(support)
            emb_q = model(query)
            loss, acc = prototypical_loss(emb_s, s_lbls, emb_q, q_lbls, n_way)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc  += acc

            progress_bar.set_postfix(loss=loss.item(), acc=acc)
            # support: [1, n_way*k, C, H, W] → remover batch dim
            support = support.squeeze(0).to(device)
            s_lbls  = s_lbls.squeeze(0).to(device)
            query   = query.squeeze(0).to(device)
            q_lbls  = q_lbls.squeeze(0).to(device)

            optimizer.zero_grad()
            emb_s = model(support)
            emb_q = model(query)
            loss, acc = prototypical_loss(emb_s, s_lbls, emb_q, q_lbls, n_way)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc  += acc

        avg_loss = total_loss / len(dataloader)
        avg_acc  = total_acc  / len(dataloader)
        history["loss"].append(avg_loss)
        history["acc"].append(avg_acc)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    return history

# 6. FUNÇÃO DE PREDIÇÃO EM FEW‑SHOT
@torch.no_grad()
def predict_few_shot(model, support_imgs, support_lbls, query_imgs, device):
    """
    Dado:
      support_imgs: tensor [n_way*k_shot, C, H, W]
      support_lbls: tensor [n_way*k_shot] com labels de 0..n_way-1
      query_imgs:   tensor [n_query, C, H, W]
    Retorna:
      pred_labels: tensor [n_query] com predições
    """
    model.eval()
    emb_s = model(support_imgs.to(device))
    emb_q = model(query_imgs.to(device))
    n_way = len(torch.unique(support_lbls))
    # calcula protótipos
    prototypes = torch.stack([
        emb_s[support_lbls == i].mean(0)
        for i in range(n_way)
    ])
    dists = torch.cdist(emb_q, prototypes)
    preds = dists.argmin(dim=1)
    return preds.cpu()

# 7. MAIN: configurações, dataset, treino e salvamento
if __name__ == "__main__":
    # HYPERPARAMS
    ROOT = "datasets/final/terumo/train"  # raiz com 6 subpastas de classe
    N_WAY = 6
    K_SHOT = 5
    Q_QUERIES = 32
    BATCH = 1
    EPOCHS = 10
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset e DataLoader few‑shot
    fs_dataset = FewShotFolderDataset(
        ROOT, n_way=N_WAY, k_shot=K_SHOT,
        q_queries=Q_QUERIES, transform=transform
    )
    fs_loader = DataLoader(fs_dataset, batch_size=BATCH, shuffle=True)

    # Modelo e otimizador
    model = ConvEmbedding(out_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Treina
    history = train_few_shot(model, fs_loader, optimizer, device, N_WAY, epochs=EPOCHS)

    # Salva checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/fewshot_protonet.pth")
    print("Modelo salvo em checkpoints/fewshot_protonet.pth")

    # Exemplo de predição em um episódio
    support, s_lbls, query, q_lbls = next(iter(fs_loader))
    support = support.squeeze(0); s_lbls = s_lbls.squeeze(0)
    query   = query.squeeze(0)
    preds = predict_few_shot(model, support, s_lbls, query, device)
    print("Ground‑truth:", q_lbls.numpy())
    print("Predictions :", preds.numpy())
