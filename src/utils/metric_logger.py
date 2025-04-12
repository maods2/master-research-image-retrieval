import json
from metrics.metric_base import MetricLoggerBase
import mlflow
import os
from datetime import datetime

def save_json_with_timestamp(data, base_filename, config, subfolder="results"):
    """
    Save JSON data to a file in a timestamped folder, and save config if not already saved.
    
    Parameters:
    -----------
    data : dict
        Data to save as JSON
    base_filename : str
        Base name for the file (without extension)
    config : dict
        Configuration dictionary containing model.experiment_name
    subfolder : str, optional
        Name of the subfolder to create within the results directory
        
    Returns:
    --------
    str
        Path to the saved file
    """

    result_dir = subfolder
    os.makedirs(result_dir, exist_ok=True)
    
    # Check if config has already been saved
    config_filepath = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_filepath):
        # Save config
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_filepath}")
    
    # Create full filepath for the main data
    filepath = os.path.join(result_dir, f"{base_filename}.json")
    
    # Save JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {filepath}")
    return filepath

class MLFlowMetricLogger(MetricLoggerBase):
    def __init__(self, config: dict):
        """
        Initializes the MLFlowMetricLogger with experiment settings.

        :param config: Dictionary containing the experiment settings.
        """
        self.model_name = config.get('model', {}).get('name', 'default_model')
        self.experiment_name = config.get('model', {}).get(
            'experiment_name', 'default_experiment'
        )
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_id = None  # To store the unique identifier of the run

        self._setup_mlflow()

    def _setup_mlflow(self):
        """
        Configures MLflow with the experiment name and starts a new run.
        """
        mlflow.set_experiment(self.experiment_name)  # Sets the experiment name
        with mlflow.start_run(
            run_name=f'{self.model_name}_{self.timestamp}'
        ) as run:
            self.run_id = (
                run.info.run_id
            )  # Stores the run ID for future reference
            mlflow.log_param(
                'model_name', self.model_name
            )  # Logs the model name
            mlflow.log_param(
                'experiment_name', self.experiment_name
            )  # Logs the experiment name
            mlflow.log_param('timestamp', self.timestamp)  # Logs the timestamp

    def log_metric(self, metric_name: str, value: float, step: int = None):
        """
        Logs a single metric to MLflow.

        :param metric_name: Name of the metric.
        :param value: Value of the metric.
        :param step: Optional step/epoch number to associate with the metric.
        """
        if not self.run_id:
            raise ValueError(
                'MLflow is not properly configured, or the run was not started.'
            )
        if step is not None:
            mlflow.log_metric(metric_name, value, step=step)
        else:
            mlflow.log_metric(metric_name, value)

    def log_metrics(self, metrics: dict, step: int = None):
        """
        Logs multiple metrics to MLflow.

        :param metrics: Dictionary of metrics, where keys are metric names and values are their respective values.
        :param step: Optional step/epoch number to associate with the metrics.
        """
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, step=step)

    def log_params(self, params: dict):
        """
        Logs parameters to MLflow.

        :param params: Dictionary of parameters, where keys are parameter names and values are their respective values.
        """
        if not self.run_id:
            raise ValueError(
                'MLflow is not properly configured, or the run was not started.'
            )
        mlflow.log_params(params)

    def log_artifact(self, artifact_path: str):
        """
        Logs an artifact to MLflow.

        :param artifact_path: Path to the artifact file (e.g., model weights or configuration file).
        """
        if not self.run_id:
            raise ValueError(
                'MLflow is not properly configured, or the run was not started.'
            )
        mlflow.log_artifact(artifact_path)
        
    def log_json(self, params: dict):
        """
        Logs a JSON file to MLflow.

        :param params: Dictionary of parameters to log as JSON.
        """
        if not self.run_id:
            raise ValueError(
                'MLflow is not properly configured, or the run was not started.'
            )

        mlflow.log_dict(params)


class TxtMetricLogger(MetricLoggerBase):
    def __init__(self, config: dict):
        self.config = config
        workspace_dir = config.get('workspace_dir', './')
        model_name = config.get('model', {}).get('name', 'default_model')
        experiment_name = config.get('model', {}).get(
            'experiment_name', 'default_experiment'
        )
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.folder_name = f'{experiment_name}_{timestamp}'
        file_name = f'{self.folder_name}_metrics.txt'
        
        self.file_path = os.path.join(workspace_dir, file_name)

    def log_metric(self, metric_name: str, value: float, step: int = None):
        with open(self.file_path, 'a') as f:
            f.write(f'{metric_name}: {value}\n')

    def log_metrics(self, metrics: dict, step: int = None):
        with open(self.file_path, 'a') as f:
            for metric_name, value in metrics.items():
                self.log_json(value, metric_name)
   
    def log_json(self, params: dict, base_filename: str):
        save_json_with_timestamp(
            params,
            base_filename,
            config=self.config,
            subfolder=f"local_experiments/{self.folder_name}"
        )
        
    def log_params(self, params: dict):
        with open(self.file_path, 'a') as f:
            for param_name, value in params.items():
                f.write(f'{param_name}: {value}\n')

    def log_artifact(self, artifact_path: str):
        with open(self.file_path, 'a') as f:
            f.write(f'Artifact: {artifact_path}\n')

########## Factory Function ##########


def setup_metric_logger(config):
    if config['metric_logging']['tool'] == 'txt':
        return TxtMetricLogger(config)

    elif config['metric_logging']['tool'] == 'mlflow':
        return MLFlowMetricLogger(config)

    else:
        raise ValueError(
            f"Unsupported metric logging tool: {config['metric_logging']['tool']}"
        )
