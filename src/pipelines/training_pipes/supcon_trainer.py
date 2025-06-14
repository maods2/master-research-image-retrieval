import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, Any, Callable
from tqdm import tqdm
from dataloaders.dataset_contrastive import ContrastiveDataset
from losses.loss_contrastive import SupConLoss
from pipelines.training_pipes.base_trainer import BaseTrainer
from metrics.metric_base import MetricLoggerBase
from utils.checkpoint_utils import save_model_and_log_artifact


class SupConTrainer(BaseTrainer):
    """
    https://arxiv.org/abs/2004.11362
    Supervised Contrastive Learning
    This trainer implements the supervised contrastive learning approach.
    It uses a contrastive loss function to learn representations by maximizing agreement
    between augmented views of the same instance while minimizing agreement with different instances.
    """

    def __init__(self, config: dict):
        self.config = config

    def train_one_epoch(
        self, model, loss_fn, optimizer, dataloader, device, epoch
    ):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')

        for images, labels in progress_bar:
            images = images.to(device)  # [B, 2, C, H, W]
            labels = labels.to(device)

            optimizer.zero_grad()
            bsz = images.shape[0]
            features = model(images.view(-1, *images.shape[2:]))  # [2*B, D]
            features = F.normalize(features, dim=1)
            features = features.view(bsz, 2, -1)  # [B, 2, D]

            loss = loss_fn(features, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        return avg_loss

    def __call__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger: MetricLoggerBase,
    ):
        device = config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.to(device)
        epochs = config['training']['epochs']
        patience = config['training'].get('early_stopping_patience', 10)

        min_loss = float('inf')
        epochs_without_improvement = 0
        checkpoint_path = None
        train_history = {'loss': []}

        for epoch in range(epochs):
            avg_loss = self.train_one_epoch(
                model, loss_fn, optimizer, train_loader, device, epoch
            )

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
                logger.info(
                    f'Early stopping triggered after {epochs_without_improvement} epochs.'
                )
                print(
                    f'Early stopping triggered after {epochs_without_improvement} epochs.'
                )
                break

        train_history['last_epoch_metrics'] = {'loss': avg_loss}
        metric_logger.log_json(train_history, 'train_metrics')

        return model


# -------------------------
# Quick Test (Dummy)
# -------------------------
if __name__ == '__main__':
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10
    import torchvision.models as models
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # transform = A.Compose(
    #     [T.RandomResizedCrop(32), T.RandomHorizontalFlip(), T.ToTensor()]
    # )
    transform = A.Compose(
        [
            A.RandomResizedCrop((224, 224)),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )
    root_dir = './datasets/final/glomerulo/train'
    dataset = ContrastiveDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=96, shuffle=True)

    backbone = models.resnet18(pretrained=False)
    backbone.fc = nn.Identity()

    class ProjectionHead(nn.Module):
        def __init__(self, base_model, out_dim=128):
            super().__init__()
            self.backbone = base_model
            self.proj = nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, out_dim)
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.proj(features)

    model = ProjectionHead(backbone)

    trainer = SupConTrainer(
        config={'training': {'epochs': 10, 'early_stopping_patience': 3}}
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    class DummyLogger:
        def info(self, msg):
            print(msg)

    class DummyMetricLogger:
        def log_json(self, d, name):
            print(f'Metrics logged: {d}')

        def log_artifact(self, filepath):
            pass

    criterion = SupConLoss(config={'temperature': 0.07})

    trainer(
        model,
        criterion,
        optimizer,
        dataloader,
        None,
        trainer.config,
        DummyLogger(),
        DummyMetricLogger(),
    )
