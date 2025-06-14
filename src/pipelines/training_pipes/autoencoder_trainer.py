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
from torch.utils.data import DataLoader
from tqdm import tqdm
from pipelines.training_pipes.base_trainer import BaseTrainer
from metrics.metric_base import MetricLoggerBase
from dataloaders.dataset import StandardImageDataset

class AutoencoderTrainer(BaseTrainer):
    def __init__(self, config: dict):
        self.config = config

    def train_one_epoch(self, model, loss_fn, optimizer, dataloader, device, epoch):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for images, _ in progress_bar:
            images = images.to(device)
            optimizer.zero_grad()

            latent = model.encode(images)
            outputs = model.decode(latent)
            
            loss = loss_fn(outputs, images)
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
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        epochs = config["training"]["epochs"]
        patience = config["training"].get("early_stopping_patience", 10)

        min_loss = float("inf")
        epochs_without_improvement = 0
        checkpoint_path = None
        train_history = {"loss": []}

        for epoch in range(epochs):
            avg_loss = self.train_one_epoch(
                model, loss_fn, optimizer, train_loader, device, epoch
            )

            logger.info(f"[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")
            print(f"[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")
            train_history["loss"].append(avg_loss)

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
                mode="loss",
            )

            if should_stop:
                logger.info(
                    f"Early stopping triggered after {epochs_without_improvement} epochs."
                )
                print(
                    f"Early stopping triggered after {epochs_without_improvement} epochs."
                )
                break

        train_history["last_epoch_metrics"] = {"loss": avg_loss}
        metric_logger.log_json(train_history, "train_metrics")

        return model


# -------------------------
# Network Definition
# -------------------------
from torchvision import models
from torchvision.models import ResNet18_Weights


class Autoencoder(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        # Encoder (ResNet18 without final FC)
        self.encoder = backbone
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        # Decoder (simple)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, output_padding=1),
            nn.Tanh(),
        )
    def encode(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        return x


# -------------------------
# Dummy Test
# -------------------------
if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader

    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )

    root_dir = './datasets/final/glomerulo/train'
    dataset = StandardImageDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=96, shuffle=True)


    backbone = models.resnet18(pretrained=False)
    backbone.fc = nn.Identity()
    
    model = Autoencoder(backbone)
    trainer = AutoencoderTrainer(config={"training": {"epochs": 10, "early_stopping_patience": 3}})
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    class DummyLogger:
        def info(self, msg): print(msg)

    class DummyMetricLogger:
        def log_json(self, d, name): print(f"Metrics logged: {d}")
        def log_artifact(self, filepath): pass

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
