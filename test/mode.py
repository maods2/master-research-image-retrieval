import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, Any, Callable
from tqdm import tqdm

# -------------------------
# SupCon Loss
# -------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = labels.shape[0]
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        logits = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


# -------------------------
# Contrastive Dataset
# -------------------------
class ContrastiveDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1 = img2 = img

        return torch.stack([img1, img2]), label


# -------------------------
# SupCon Trainer
# -------------------------
class SupConTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.temperature = config['model'].get('temperature', 0.07)

    def train_one_epoch(self, model, optimizer, dataloader, device, epoch):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')

        criterion = SupConLoss(temperature=self.temperature)

        for images, labels in progress_bar:
            images = images.to(device)  # [B, 2, C, H, W]
            labels = labels.to(device)

            optimizer.zero_grad()
            bsz = images.shape[0]
            features = model(images.view(-1, *images.shape[2:]))  # [2*B, D]
            features = F.normalize(features, dim=1)
            features = features.view(bsz, 2, -1)  # [B, 2, D]

            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        return avg_loss, 0.0, None  # acc/supp set not used here

    def __call__(
        self,
        model,
        loss_fn,  # not used
        optimizer,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger,
    ):
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        epochs = config['training']['epochs']
        patience = config['training'].get('early_stopping_patience', 10)

        min_loss = float('inf')
        epochs_without_improvement = 0
        checkpoint_path = None
        train_history = {'loss': []}

        for epoch in range(epochs):
            avg_loss, _, _ = self.train_one_epoch(model, optimizer, train_loader, device, epoch)

            logger.info(f'[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.4f}')
            print(f'[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.4f}')
            train_history['loss'].append(avg_loss)

            (
                should_stop,
                min_loss,
                epochs_without_improvement,
                checkpoint_path,
            ) = self.save_model_if_best(
                model=model,
                metric=avg_loss,
                best_metric=min_loss,
                epochs_without_improvement=epochs_without_improvement,
                checkpoint_path=checkpoint_path,
                config=config,
                metric_logger=metric_logger,
                mode='loss',
            )

            if should_stop:
                logger.info(f'Early stopping triggered after {epochs_without_improvement} epochs.')
                print(f'Early stopping triggered after {epochs_without_improvement} epochs.')
                break

        train_history['last_epoch_metrics'] = {'loss': avg_loss}
        metric_logger.log_json(train_history, 'train_metrics')

        return model

    def save_model_if_best(self, model, metric, best_metric, epochs_without_improvement, checkpoint_path, config, metric_logger, mode='loss'):
        improved = metric < best_metric if mode == 'loss' else metric > best_metric
        if improved:
            best_metric = metric
            epochs_without_improvement = 0
            checkpoint_path = config['training'].get('checkpoint_path', 'best_model.pt')
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        should_stop = epochs_without_improvement >= config['training'].get('early_stopping_patience', 10)
        return should_stop, best_metric, epochs_without_improvement, checkpoint_path


# -------------------------
# Quick Test (Dummy)
# -------------------------
if __name__ == '__main__':
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10
    import torchvision.models as models

    transform = T.Compose([
        T.RandomResizedCrop(32),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])

    dataset = ContrastiveDataset(CIFAR10(root='./data', train=True, download=True, transform=transform))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    backbone = models.resnet18(pretrained=False)
    backbone.fc = nn.Identity()

    class ProjectionHead(nn.Module):
        def __init__(self, base_model, out_dim=128):
            super().__init__()
            self.backbone = base_model
            self.proj = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, out_dim)
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.proj(features)

    model = ProjectionHead(backbone)

    trainer = SupConTrainer(config={
        'model': {'temperature': 0.07},
        'training': {'epochs': 10, 'early_stopping_patience': 3}
    })

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    class DummyLogger:
        def info(self, msg): print(msg)

    class DummyMetricLogger:
        def log_json(self, d, name): print(f"Metrics logged: {d}")

    trainer(model, None, optimizer, dataloader, None, trainer.config, DummyLogger(), DummyMetricLogger())
