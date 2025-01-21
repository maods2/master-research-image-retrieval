from typing import Dict

from pipelines.test_pipes.metrics.factory import get_metrics
from pipelines.test_pipes.metrics.metric_base import MetricLoggerBase



def default_test_fn(model, train_loader, test_loader, config, logger, metric_logger: MetricLoggerBase):
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
    metrics_list = get_test_function(config["testing"])
    
    for metric in metrics_list:
        results = metric(model, train_loader, test_loader, config, logger)
        logger.info(f"Results for {metric.__class__.__name__}: {results}")
        
        # Log metrics using the provided logger
        metric_logger.log_metrics(results)

        
        
def get_test_function(testing_config: Dict):
    
    if testing_config["pipeline"] == "default":
        return default_test_fn
    else:
        raise ValueError(f"Testing pipeline {testing_config["pipeline"]} is not supported")