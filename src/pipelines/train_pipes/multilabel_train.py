import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import torch
import torchvision.models as models
import torch.nn as nn

from metrics.metric_base import MetricLoggerBase
from utils.checkpoint_utils import save_model_and_log_artifact
from utils.metric_logger import TxtMetricLogger


# Load a pretrained ResNet-50 model
def load_pretrained_model(num_classes: int) -> torch.nn.Module:
    model = models.resnet50(pretrained=True)  # Load a pretrained ResNet-50
    # Modify the final layer to fit your number of classes (for multilabel classification)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(
    model: torch.nn.Module, 
    loss_fn: callable, 
    optimizer: torch.optim.Optimizer, 
    train_loader: torch.utils.data.DataLoader, 
    device: str, 
    log_interval: int, 
    epoch: int
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        loss_fn (callable): Loss function for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        device (str): Device to run the model on (e.g., "cuda" or "cpu").
        log_interval (int): Interval for logging batch-wise metrics.
    
    Returns:
        tuple: The average loss for the epoch, all predictions for the epoch, all targets for the epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1}")

    for batch_idx, (images, targets) in progress_bar:
        images = images.to(device)
        targets = targets.to(device).float()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Predictions and metrics
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.append(preds.cpu().detach())
        all_targets.append(targets.cpu().detach())

        # Logging batch-wise loss
        progress_bar.set_postfix(loss=running_loss/(batch_idx+1))

    # Return the average loss and all predictions/targets for the epoch
    return running_loss/len(train_loader), torch.cat(all_preds), torch.cat(all_targets)


def compute_metrics(all_preds: torch.Tensor, all_targets: torch.Tensor) -> tuple[float, float]:
    """
    Compute F1 score and accuracy for the epoch.

    Args:
        all_preds (torch.Tensor): All predictions for the epoch.
        all_targets (torch.Tensor): All targets for the epoch.

    Returns:
        tuple: F1 score, accuracy
    """
    all_preds = all_preds.numpy()
    all_targets = all_targets.numpy()

    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    epoch_accuracy = accuracy_score(all_targets, (all_preds > 0.5).astype(int))

    return epoch_f1, epoch_accuracy


def train_multilabel(
    model: torch.nn.Module, 
    loss_fn: callable, 
    optimizer: torch.optim.Optimizer, 
    train_loader: torch.utils.data.DataLoader, 
    config: dict, 
    logger: callable, 
    metric_logger: MetricLoggerBase
) -> torch.nn.Module:
    """
    Train the model for the specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        loss_fn (callable): Loss function for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        config (dict): Configuration dictionary (e.g., device, logging frequency).
        logger (callable): Function to log metrics.
        metric_logger (MetricLoggerBase): Logger for metrics.

    Returns:
        torch.nn.Module: The trained model.
    """
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    log_interval = config.get("log_interval", 10)
    epochs = config["training"]["epochs"]

    model.to(device)

    checkpoint_path = None
    max_val_f1score = 0
    
    for epoch in range(epochs):
        # Train the model for one epoch
        epoch_loss, all_preds, all_targets = train_one_epoch(
            model, loss_fn, optimizer, train_loader, device, log_interval, epoch
        )

        # Compute metrics (F1 score and accuracy)
        epoch_f1, epoch_accuracy = compute_metrics(all_preds, all_targets)

        # Log metrics for the epoch
        metric_logger.log_metric('epoch_loss', epoch_loss, step=epoch)
        metric_logger.log_metric('epoch_accuracy', epoch_accuracy, step=epoch)
        metric_logger.log_metric('epoch_f1', epoch_f1, step=epoch)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, F1 Score: {epoch_f1:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        if epoch_f1 > max_val_f1score:
            max_val_f1score = epoch_f1
            checkpoint_path = save_model_and_log_artifact(metric_logger, config, model, filepath=checkpoint_path)

    return model

# Example usage
if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader
    import sys
    sys.path.append("/workspaces/master-research-image-retrieval/")
    print(sys.path)
    from dataloaders.dataset_terumo import TerumoImageDataset
    # Assume model, loss_fn, optimizer, train_loader are defined
    config = {
        "device": "cuda",
        "log_interval": 20
    }

    def simple_logger(metrics):
        print(metrics)

    num_classes = 6
    model = load_pretrained_model(num_classes)
    
    root_dir = "datasets/final/terumo/train"
    custom_mapping = {"Crescent": 0, "Hypercellularity": 1, "Membranous": 2, "Normal": 3, "Podocytopathy": 4, "Sclerosis": 5}
    
    # Define transformations using Albumentations
    data_transforms = A.Compose([
        A.Resize(224, 224),                        # Resize the image
        A.HorizontalFlip(p=0.5),                  # Random horizontal flip
        A.RandomBrightnessContrast(p=0.2),        # Random brightness and contrast adjustments
        A.Normalize(mean=(0.5, 0.5, 0.5),         # Normalize to [-1, 1]
                    std=(0.5, 0.5, 0.5)),
        ToTensorV2()                              # Convert to PyTorch tensor
    ])

    # Create the dataset
    dataset = TerumoImageDataset(root_dir=root_dir, transform=data_transforms, class_mapping=custom_mapping)
    train_loader = DataLoader(
        dataset,                      # Dataset instance
        batch_size=32,                      # Set batch size as per your preference
        shuffle=True,                       # Shuffle dataset for better generalization
        num_workers=3,                      # Number of CPU cores to use for loading data in parallel
        pin_memory=True,                    # Pin memory to speed up data transfer to GPU
        )
    trained_model = train_multilabel(
        model,  # Your PyTorch model
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        train_loader=train_loader,  # Your DataLoader
        config=config,
        logger=simple_logger,
        epochs=10  # Set number of epochs
    )

    print("Training completed.")
