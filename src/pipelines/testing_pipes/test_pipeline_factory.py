from typing import Dict

import numpy as np

from metrics.metric_factory import get_metrics
from metrics.metric_base import MetricBase, MetricLoggerBase
from typing import Any, List
import torch
from torch.utils.data import DataLoader

from pipelines.testing_pipes.fsl_testing import fsl_test_fn
from utils.embedding_utils import load_or_create_embeddings


def default_test_fn(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    logger: Any,
    metric_logger: MetricLoggerBase,
) -> None:
    """
    Run metrics on the test data and log the results.

    Args:
        model: PyTorch model being tested.
        train_loader: DataLoader for the training data.
        test_loader: DataLoader for the testing data.
        config: Configuration dictionary.
        logger: Logger object for logging.
        metric_logger: MetricLoggerBase instance for logging metrics.
    """
    metrics_list: List[MetricBase] = get_metrics(config['testing'])
    device = (
        config['device']
        if config.get('device')
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    embeddings = load_or_create_embeddings(
        model, train_loader, test_loader, config, logger, device
    )

    for metric in metrics_list:
        results = metric(
            model, train_loader, test_loader, embeddings, config, logger
        )

        logger.info(f'Results for {metric.__class__.__name__}: {results}')
        metric_logger.log_metrics(results)


def get_test_function(testing_config: Dict):

    if testing_config['pipeline'] == 'default':
        return default_test_fn
    
    if testing_config['pipeline'] == 'fsl':
        return fsl_test_fn
    
    else:
        raise ValueError(
            f"Testing pipeline {testing_config['pipeline']} is not supported"
        )
