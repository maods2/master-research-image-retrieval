import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import random

# 1. Carrega o dataset PCam da Hugging Face
pcam = load_dataset("zacharielegault/PatchCamelyon")

# 2. Transformacoes de imagem
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

# 3. Classe Few-Shot Episodica
class FewShotPCam(Dataset):
    def __init__(self, dataset, n_way=2, k_shot=5, q_queries=5, transform=None):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.transform = transform
        self.class_to_images = {}
        for item in dataset:
            label = item['label']
            if label not in self.class_to_images:
                self.class_to_images[label] = []
            self.class_to_images[label].append(item['image'])

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        selected_classes = random.sample(list(self.class_to_images.keys()), self.n_way)
        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for i, cls in enumerate(selected_classes):
            images = random.sample(self.class_to_images[cls], self.k_shot + self.q_queries)
            support = images[:self.k_shot]
            query = images[self.k_shot:]
            for img in support:
                img_tensor = self.transform(img)
                support_images.append(img_tensor)
                support_labels.append(i)
            for img in query:
                img_tensor = self.transform(img)
                query_images.append(img_tensor)
                query_labels.append(i)

        return (torch.stack(support_images), torch.tensor(support_labels),
                torch.stack(query_images), torch.tensor(query_labels))

# 4. Modelo de Embedding CNN
class ConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 24 * 24, 128)
        )

    def forward(self, x):
        return self.encoder(x)

# 5. Função de perda prototípica

def prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels, n_way):
    prototypes = torch.stack([support_embeddings[support_labels == i].mean(0) for i in range(n_way)])
    dists = torch.cdist(query_embeddings, prototypes)
    log_p_y = (-dists).log_softmax(dim=1)
    loss = nn.NLLLoss()(log_p_y, query_labels)
    acc = (log_p_y.argmax(1) == query_labels).float().mean().item()
    return loss, acc

# 6. Treinamento

def train_few_shot(model, dataloader, optimizer, device, n_way, k_shot, epochs=5):
    history = {"loss": [], "acc": []}
    model.train()
    for epoch in range(epochs):
        total_loss, total_acc = 0, 0
        for support_imgs, support_lbls, query_imgs, query_lbls in dataloader:
            support_imgs = support_imgs.squeeze(0).to(device)
            support_lbls = support_lbls.squeeze(0).to(device)
            query_imgs = query_imgs.squeeze(0).to(device)
            query_lbls = query_lbls.squeeze(0).to(device)

            optimizer.zero_grad()
            support_emb = model(support_imgs)
            query_emb = model(query_imgs)
            loss, acc = prototypical_loss(support_emb, support_lbls, query_emb, query_lbls, n_way)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        history["loss"].append(avg_loss)
        history["acc"].append(avg_acc)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    return history

# 7. Execução
n_way = 2
k_shot = 5
fewshot_dataset = FewShotPCam(pcam['train'], n_way=n_way, k_shot=k_shot, transform=transform)
dataloader = DataLoader(fewshot_dataset, batch_size=1, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvEmbedding().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Treinar
history = train_few_shot(model, dataloader, optimizer, device, n_way=n_way, k_shot=k_shot, epochs=5)
print(history)
