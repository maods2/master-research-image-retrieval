import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from metrics.metric_base import MetricLoggerBase
from metrics.metric_factory import get_metrics
from pipelines.training_pipes.base_trainer import BaseTrainer
from utils.checkpoint_utils import save_model_and_log_artifact


class TripletTrain(BaseTrainer):
    def __init__(self, config: dict):
        self.config = config
        super().__init__()

    def train_one_epoch(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        device,
        epoch,
    ):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Training Epoch {epoch + 1}',
        )

        for batch_idx, (anchor, positive, negative) in progress_bar:

            # Forward pass
            a_emb = model(anchor.to(device))
            p_emb = model(positive.to(device))
            n_emb = model(negative.to(device))

            loss = loss_fn(a_emb, p_emb, n_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Logging batch-wise loss
            progress_bar.set_postfix(loss=running_loss / (batch_idx + 1))

        # Calcula a média da perda para a época
        return running_loss / len(train_loader)

    def __call__(
        self,
        model: torch.nn.Module,
        loss_fn: callable,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        config: dict,
        logger: callable,
        metric_logger: MetricLoggerBase,
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
        device = config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.to(device)
        epochs = config['training']['epochs']

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


