from typing import Dict

import numpy as np

from metrics.factory import get_metrics
from metrics.metric_base import MetricBase, MetricLoggerBase
from typing import Any, List
import torch
from torch.utils.data import DataLoader

from utils.embedding_utils import create_embeddings_dict


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
    device = config['device'] if config.get('device') else ('cuda' if torch.cuda.is_available() else 'cpu')

    if config['testing'].get('load_embeddings', False):
        embeddings = np.load(config['testing']['embeddings_path'], allow_pickle=True)
    else:
        embeddings, file_path = create_embeddings_dict(
            model,
            train_loader,
            test_loader,
            device,
            logger,
            config
        )
        config['testing']['embeddings_path'] = file_path


    
    for metric in metrics_list:
        results = metric(
            model,
            train_loader,
            test_loader,
            embeddings,
            config,
            logger
         )
        
        logger.info(f'Results for {metric.__class__.__name__}: {results}')
        metric_logger.log_metrics(results)


def get_test_function(testing_config: Dict):

    if testing_config['pipeline'] == 'default':
        return default_test_fn
    else:
        raise ValueError(
            f"Testing pipeline {testing_config['pipeline']} is not supported"
        )
