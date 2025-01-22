from metrics.metric_base import MetricLoggerBase
import mlflow
import os
from datetime import datetime

class MLFlowMetricLogger(MetricLoggerBase):
    def __init__(self, config: dict):
        """
        Initializes the MLFlowMetricLogger with experiment settings.

        :param config: Dictionary containing the experiment settings.
        """
        self.model_name = config.get('model', {}).get('name', 'default_model')
        self.experiment_name = config.get('model', {}).get('experiment_name', 'default_experiment')
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = None  # To store the unique identifier of the run

        self._setup_mlflow()

    def _setup_mlflow(self):
        """
        Configures MLflow with the experiment name and starts a new run.
        """
        mlflow.set_experiment(self.experiment_name)  # Sets the experiment name
        with mlflow.start_run(run_name=f"{self.model_name}_{self.timestamp}") as run:
            self.run_id = run.info.run_id  # Stores the run ID for future reference
            mlflow.log_param("model_name", self.model_name)  # Logs the model name
            mlflow.log_param("experiment_name", self.experiment_name)  # Logs the experiment name
            mlflow.log_param("timestamp", self.timestamp)  # Logs the timestamp

    def log_metric(self, metric_name: str, value: float):
        """
        Logs a single metric to MLflow.

        :param metric_name: Name of the metric.
        :param value: Value of the metric.
        """
        if not self.run_id:
            raise ValueError("MLflow is not properly configured, or the run was not started.")
        mlflow.log_metric(metric_name, value)

    def log_metrics(self, metrics: dict):
        """
        Logs multiple metrics to MLflow.

        :param metrics: Dictionary of metrics, where keys are metric names and values are their respective values.
        """
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value)
            
            
class TxtMetricLogger(MetricLoggerBase):
    def __init__(self, config: dict):
        workspace_dir = config.get('workspace_dir', './')
        model_name = config.get('model', {}).get('name', 'default_model')
        experiment_name = config.get('model', {}).get('experiment_name', 'default_experiment')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        file_name = f"{model_name}_{experiment_name}_{timestamp}_metrics.txt"
        self.file_path = os.path.join(workspace_dir, file_name)

    def log_metric(self, metric_name: str, value: float):
        with open(self.file_path, 'a') as f:
            f.write(f"{metric_name}: {value}\n")

    def log_metrics(self, metrics: dict):
        with open(self.file_path, 'a') as f:
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value}\n")
                

def setup_metric_logger(config):
    if config["metric_logging"]["tool"] == "txt":
        from utils.metric_logger import TxtMetricLogger
        return TxtMetricLogger(config)
    
    elif config["metric_logging"]["tool"] == "mlflow":
        from utils.metric_logger import MLFlowMetricLogger
        return MLFlowMetricLogger(config)
    
    else:
        raise ValueError(f"Unsupported metric logging tool: {config['metric_logging']['tool']}")
    
    return logger

if __name__ == "__main__":
    # Test the MLFlowMetricLogger
    config = {
        "metric_logging": {
            "tool": "mlflow"
        },
        "model": {
            "name": "resnet50",
            "experiment_name": "image_retrieval"
        }
    }   
    mlflow_logger = MLFlowMetricLogger(config)
    mlflow_logger.log_metric("accuracy", 0.95)
    mlflow_logger.log_metrics({"precision": 0.85, "recall": 0.90})