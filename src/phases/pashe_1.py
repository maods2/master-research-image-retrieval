import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ---- Phase 1: Baseline Model ---- #

# Backbone Feature Extractor (ViT, VGG, ResNet, EfficientNet as options)
class FeatureExtractor(nn.Module):
    def __init__(self, backbone="resnet50", embedding_dim=256):
        super(FeatureExtractor, self).__init__()
        
        # Load Pretrained Backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove FC layer
        elif backbone == "convnext_tiny":
            self.backbone = models.convnext_tiny(pretrained=True)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()  # Remove classifier layer
        elif backbone == "vit_b_16":
            self.backbone = models.vit_b_16(pretrained=True)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
        elif backbone == "vgg16":
            self.backbone = models.vgg16(pretrained=True)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=True)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError("Unsupported backbone")
        
        # Embedding Layer
        self.embedding_layer = nn.Linear(in_features, embedding_dim)
        
    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)  # Flatten output
        embedding = self.embedding_layer(features)
        return embedding

# Contrastive Loss (Optional: Can switch to Triplet Loss)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, emb1, emb2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(emb1, emb2)
        loss = (label * euclidean_distance ** 2) + (1 - label) * torch.clamp(self.margin - euclidean_distance, min=0) ** 2
        return loss.mean()

# Dummy Dataset (Replace with Medical Image Dataset Loader)
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.data = np.random.rand(num_samples, 3, 224, 224).astype(np.float32)
        self.labels = np.random.randint(0, 2, size=(num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(torch.tensor(img))
        return img, label

# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataloader
train_dataset = DummyDataset(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model Initialization
model = FeatureExtractor(backbone="resnet50", embedding_dim=256).cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.float().cuda()
        optimizer.zero_grad()
        
        # Forward Pass
        emb1 = model(images)
        emb2 = model(images)  # Simulating pairs (replace with actual pairs in real dataset)
        loss = criterion(emb1, emb2, labels)
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

print("Training Complete!")
