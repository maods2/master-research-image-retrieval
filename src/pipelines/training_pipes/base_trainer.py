from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from metrics.metric_base import MetricLoggerBase
from utils.checkpoint_utils import save_model_and_log_artifact
from utils.dataloader_utils import create_balanced_db_and_query
from utils.embedding_utils import load_or_create_embeddings
from utils.metrics_utils import compute_metrics


class BaseTrainer(ABC):
    """Abstract base class for training PyTorch models.

    This class defines the common interface and shared functionality for all trainers,
    ensuring consistent behavior across different training pipelines.
    """

    def __init__(self) -> None:
        """ """
        self.sample_dataloader = None
        # Initialize retrieval metrics
        self.retrieval_at_k_metrics = []

    def _initialize_sample_dataloader(
        self,
        data_loader: DataLoader,
        total_db_samples=400,
        total_query_samples=60,
        seed=42,
    ) -> None:
        """Initialize sample dataloaders for database and query samples.

        Args:
            data_loader: The original dataloader to sample from
        """
        db_subset, query_subset = create_balanced_db_and_query(
            dataset=data_loader.dataset,
            total_db_samples=total_db_samples,
            total_query_samples=total_query_samples,
            seed=seed,
        )

        db_loader = torch.utils.data.DataLoader(
            dataset=db_subset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
        )

        query_loader = torch.utils.data.DataLoader(
            dataset=query_subset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
        )

        self.sample_dataloader = {'db': db_loader, 'query': query_loader}

    def eval_retrieval_at_k(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        config: Dict[str, Any],
        logger: Callable,
    ) -> Dict[str, Any]:
        """Evaluate retrieval metrics at different k values.

        Args:
            model: The model to evaluate
            train_loader: DataLoader for training data
            config: Configuration dictionary
            logger: Logging function or object

        Returns:
            Dictionary of metric names and their values
        """
        # Initialize sample dataloader if not already created
        if self.sample_dataloader is None:
            self._initialize_sample_dataloader(
                train_loader,
                total_db_samples=config['training']['val_retrieval'][
                    'total_db_samples'
                ],
                total_query_samples=config['training']['val_retrieval'][
                    'total_query_samples'
                ],
                seed=config['training']['val_retrieval']['seed'],
            )

        embeddings = load_or_create_embeddings(
            model,
            self.sample_dataloader['db'],
            self.sample_dataloader['query'],
            config,
            logger,
            device=None,
        )

        results = {}
        for metric in self.retrieval_at_k_metrics:
            res = metric(
                model=model,
                train_loader=train_loader,
                test_loader=train_loader,
                embeddings=embeddings,
                config=config,
                logger=logger,
            )
            results[metric.__class__.__name__] = res

        return results

    def eval_classification(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        config: Dict[str, Any],
        logger: Callable,
    ) -> Dict[str, Any]:
        """
        Evaluate traditional classification (direct prediction).

        Args:
            model: PyTorch model with classification head.
            test_loader: Dataloader providing (inputs, labels) pairs.
            config: Dictionary with configuration parameters. Expected: device.
            logger: Logging function.

        Returns:
            Dictionary with evaluation metrics.
        """

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating'):
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        return compute_metrics(y_true, y_pred, logger)

    def save_model_if_best(
        self,
        model,
        metric,
        best_metric,
        model_path,
        metric_logger,
        mode='loss',
    ):
        """
        Save the model if the current metric is better than the best metric.

        Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        metric (float): The current metric value (loss or accuracy).
        best_metric (float): The best metric value so far.
        model_path (str): Path to save the model.
        config (dict): Configuration dictionary.
        metric_logger (object): Logger to log the metrics.
        mode (str): Mode to determine whether to save based on 'loss' or 'accuracy'. Default is 'loss'.

        Returns:
        str: Path to the saved model checkpoint if the model is saved, otherwise None.
        """
        if (mode == 'loss' and metric < best_metric) or (
            mode == 'accuracy' and metric > best_metric
        ):
            best_metric = metric
            model_path = save_model_and_log_artifact(
                metric_logger, self.config, model, filepath=model_path
            )

    @abstractmethod
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
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def train_one_epoch(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        device,
        log_interval,
        epoch,
    ):
        raise NotImplementedError('Subclasses must implement this method.')
