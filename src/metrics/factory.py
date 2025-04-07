from typing import Dict, List
import importlib


def get_metrics(testing_config: Dict) -> List:
    """
    Constructs test pipelines based on the provided testing configuration.

    Args:
        testing_config (Dict): Configuration containing a list of metrics and their parameters.

    Returns:
        List: A list of metric pipeline instances.
    """
    test_pipelines = []

    for metric_dict in testing_config['list_of_metrics']:
        metric_type = metric_dict['type']
        metric_instance = create_metric_instance(metric_type, metric_dict)
        test_pipelines.append(metric_instance)

    return test_pipelines


def create_metric_instance(metric_type: str, metric_config: Dict):
    """
    Dynamically imports and creates an instance of the specified metric class.

    Args:
        metric_type (str): Type of the metric to be created.
        metric_config (Dict): Configuration parameters for the metric.

    Returns:
        object: An instance of the specified metric class.
    """
    # Define the mapping of metric types to their respective module paths
    metric_modules = {
        'accuracy': 'metrics.accuracy.Accuracy',
        'f1_score': 'metrics.f1_score.F1Score',
        "multilabel_accuracy": "metrics.accuracy.MultilabelAccuracy",
        'map@k': 'metrics.map_at_k.MapAtK',
        'map@k2': 'metrics.map_at_k.MapAtK2',
        'precision@k': 'metrics.precision_at_k.PrecisionAtK',
        'recall@k': 'metrics.recall_at_k.RecallAtK',
        'accuracy@k': 'metrics.accuracy_at_k.AccuracyAtK',
    }

    if metric_type not in metric_modules:
        raise ValueError(f"Testing metric '{metric_type}' is not supported.")

    # Dynamically import the metric class
    module_path, class_name = metric_modules[metric_type].rsplit('.', 1)
    MetricClass = getattr(importlib.import_module(module_path), class_name)

    # Initialize the metric instance, passing additional parameters if required
    return MetricClass(**metric_config)
