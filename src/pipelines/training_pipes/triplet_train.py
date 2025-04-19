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
        self.retrieval_at_k_metrics = get_metrics(config['training'])

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

        epochs = config['training']['epochs']

        model.to(device)

        checkpoint_path = None
        min_val_loss = float('inf')

        training_loss = []
        training_mapatk = []

        for epoch in range(epochs):
            # Train the model for one epoch
            epoch_loss = self.train_one_epoch(
                model,
                loss_fn,
                optimizer,
                train_loader,
                device,
                epoch,
            )

            train_loader.dataset.switch_to_classifcation_dataset()   ## applied to MixTripletDataset
            retrieval_at_k_metrics = self.eval_retrieval_at_k(
                model, train_loader, config, logger
            )
            train_loader.dataset.switch_to_triplet_dataset()   ## applied to MixTripletDataset

            mapatk = retrieval_at_k_metrics['MapAtK']['map_at_k_results']
            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, MapAt10: {mapatk['mapAt10']:.4f}"
            )
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, MapAt10: {mapatk['mapAt10']:.4f}"
            )

            training_loss.append(epoch_loss)
            training_mapatk.append(mapatk['mapAt10'])

            if epoch_loss < min_val_loss:
                min_val_loss = epoch_loss
                checkpoint_path = save_model_and_log_artifact(
                    metric_logger, config, model, filepath=checkpoint_path
                )

        train_loader.dataset.switch_to_classifcation_dataset()
        test_loader.dataset.switch_to_classifcation_dataset()

        metrics = {
            'epoch_loss': training_loss,
            'epoch_mapAt10': training_mapatk,
        }
        metric_logger.log_json(metrics, 'train_metrics')

        return model
