from abc import ABC, abstractmethod

class MetricBase(ABC):
    """
    Base interface for all metrics. All metrics must inherit from this class and implement the required methods.
    """

    @abstractmethod
    def __call__(self, model, train_loader, test_loader, embeddings, config, logger):
        """
        Method to compute the metric.

        Args:
            model: PyTorch model being tested.
            train_loader: DataLoader for the training data.
            test_loader: DataLoader for the testing data.
            embeddings: Dictionary containing embeddings.
            config: Configuration dictionary.
            logger: Logger object for logging.

        Returns:
            dict: Metric results in the form of a dictionary (e.g., {"metric_name": value}).
        """
        raise NotImplementedError('Subclasses must implement this method.')

    

class MetricLoggerBase(ABC):
    """
    Base interface for metric logging. Allows switching between different logging backends.
    """

    @abstractmethod
    def log_metric(self, metric_name: str, value: float, step: int = None):
        """
        Log a single metric.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log multiple metrics.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def log_params(self, params: dict):
        """
        Log parameters.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def log_artifact(self, artifact_path: str):
        """
        Log an artifact.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def log_json(self, artifact_path: str):
        """
        Log an artifact.
        """
        raise NotImplementedError('Subclasses must implement this method.')
