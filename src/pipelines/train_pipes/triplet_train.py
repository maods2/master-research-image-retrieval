from abc import ABC, abstractmethod

import torch
from tqdm import tqdm
import copy

from metrics.metric_base import MetricLoggerBase
from metrics.precision_at_k import PrecisionAtK
from utils.checkpoint_utils import save_model_and_log_artifact


class BaseTrain(ABC):
    @abstractmethod
    def __call__(self, 
        model: torch.nn.Module, 
        loss_fn: callable, 
        optimizer: torch.optim.Optimizer, 
        train_loader: torch.utils.data.DataLoader, 
        config: dict, 
        logger: callable, 
        metric_logger: MetricLoggerBase
    ) -> torch.nn.Module:
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def train_one_epoch(self, model, loss_fn, optimizer, train_loader, device, log_interval, epoch):
        raise NotImplementedError("Subclasses must implement this method.")
    
    
class TripletTrain(BaseTrain):
    def __init__(self):
        self.precision_at_k = PrecisionAtK([1, 5, 10])
        self._dataset_already_cloned = False
    
    def compute_precisiont_at_k(self, model, train_loader, config, logger):
        if not self._dataset_already_cloned:
            copy_loader = copy.deepcopy(train_loader)
            copy_loader.dataset.switch_to_classifcation_dataset()
            self.precision_at_k(model, copy_loader, copy_loader, config, logger)
            self._dataset_already_cloned = True
            
        return self.precision_at_k(model, train_loader, train_loader, config, logger)
        
    def train_one_epoch(self, model, loss_fn, optimizer, train_loader, device, log_interval, epoch):
        model.train()  
        running_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1}")

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
            progress_bar.set_postfix(loss=running_loss/(batch_idx+1))
        
        # Calcula a média da perda para a época
        return running_loss / len(train_loader)

    def __call__(self, 
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
        min_val_loss = float("inf")

        for epoch in range(epochs):
            # Train the model for one epoch
            epoch_loss = self.train_one_epoch(
                model, loss_fn, optimizer, train_loader, device, log_interval, epoch
            )
            
            precision_at_k = self.compute_precisiont_at_k(model, train_loader, config, logger)

            # Log metrics
            metric_logger.log_metric('epoch_loss', epoch_loss, step=epoch)
            metric_logger.log_metrics(precision_at_k, step=epoch)
            # Save the model if the validation F1 score is the best so far
            if epoch_loss < min_val_loss:
                min_val_loss = epoch_loss
                checkpoint_path = save_model_and_log_artifact(metric_logger, config, model, filepath=checkpoint_path)

        return model
