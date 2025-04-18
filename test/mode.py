from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader



class BaseTrainer(ABC):
    """Abstract base class for training PyTorch models.
    
    This class defines the common interface and shared functionality for all trainers,
    ensuring consistent behavior across different training pipelines.
    """
    
    def __init__(self) -> None:
        """
        """
        self.sample_dataloader = None
        # Initialize retrieval metrics
        self.retrieval_at_k_metrics = []

    
    def _initialize_sample_dataloader(self, data_loader: DataLoader) -> None:
        """Initialize sample dataloaders for database and query samples.
        
        Args:
            data_loader: The original dataloader to sample from
        """
        db_subset, query_subset = create_balanced_db_and_query(
            dataset=data_loader.dataset,
            total_db_samples=400,
            total_query_samples=60,
            seed=42
        )
        
        db_loader = torch.utils.data.DataLoader(
            dataset=db_subset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory
        )

        query_loader = torch.utils.data.DataLoader(
            dataset=query_subset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory
        )
         
        self.sample_dataloader = {
            'db': db_loader,
            'query': query_loader
        }
    
    def eval_retrieval_at_k(
        self, 
        model: torch.nn.Module, 
        train_loader: DataLoader, 
        config: Dict[str, Any], 
        logger: Callable
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
            self._initialize_sample_dataloader(train_loader)
        
        embeddings = load_or_create_embeddings(
            model,
            self.sample_dataloader['db'],
            self.sample_dataloader['query'],
            config,
            logger,
            device=None
        )
        
        results = {}
        for metric in self.retrieval_at_k_metrics:
            res = metric(
                model=model, 
                train_loader=train_loader, 
                test_loader=train_loader, 
                embeddings=embeddings, 
                config=config, 
                logger=logger
            )
            results[metric.__class__.__name__] = res
            
        return results
    

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