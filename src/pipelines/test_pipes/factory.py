from typing import Dict

from metrics.factory import get_metrics
from metrics.metric_base import MetricBase, MetricLoggerBase
from typing import Any, List
import torch
from torch.utils.data import DataLoader


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

    for metric in metrics_list:
        results = metric(model, train_loader, test_loader, config, logger)
        logger.info(f'Results for {metric.__class__.__name__}: {results}')

        # Log metrics using the provided logger
        metric_logger.log_metrics(results)


def get_test_function(testing_config: Dict):

    if testing_config['pipeline'] == 'default':
        return default_test_fn
    else:
        raise ValueError(
            f"Testing pipeline {testing_config['pipeline']} is not supported"
        )
