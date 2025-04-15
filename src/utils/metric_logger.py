import json
from metrics.metric_base import MetricLoggerBase
import mlflow
import os
from datetime import datetime
from typing import Dict, Optional

class MLFlowMetricLogger(MetricLoggerBase):
    def __init__(self, config: Dict):
        self.model_name = config.get('model', {}).get('name', 'default_model')
        self.experiment_name = config.get('model', {}).get('experiment_name', 'default_experiment')
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_id: Optional[str] = None

        self._initialize_mlflow()

    def _initialize_mlflow(self):
        """Sets MLflow experiment and starts a new run with basic parameters."""
        mlflow.set_experiment(self.experiment_name)
        run_name = f'{self.model_name}_{self.timestamp}'
        run = mlflow.start_run(run_name=run_name)
        self.run_id = run.info.run_id

        self._log_basic_params()

    def _log_basic_params(self):
        """Logs initial experiment parameters."""
        mlflow.log_params({
            'model_name': self.model_name,
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp
        })

    def _ensure_run_started(self):
        """Raises an error if no MLflow run is active."""
        if not self.run_id:
            raise RuntimeError("MLflow run is not active or was not initialized properly.")

    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        self._ensure_run_started()
        mlflow.log_metric(metric_name, value, step=step) if step is not None else mlflow.log_metric(metric_name, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_params(self, params: Dict):
        self._ensure_run_started()
        mlflow.log_params(params)

    def log_artifact(self, artifact_path: str):
        self._ensure_run_started()
        mlflow.log_artifact(artifact_path)

    def log_json(self, params: dict, base_filename: str = "params.json"):
        """
        Logs a dictionary as a JSON artifact to MLflow.
        
        :param params: Dictionary to log.
        :param artifact_file_name: Name to give to the JSON file in MLflow.
        """
        self._ensure_run_started()
        mlflow.log_dict(params, base_filename)

class FileUtils:
    @staticmethod
    def ensure_dir_exists(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_json(data: dict, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def save_config_once(config: dict, save_dir: str, filename: str = "config.json"):
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            FileUtils.save_json(config, filepath)
            print(f"Config saved to {filepath}")

    @staticmethod
    def save_json_with_timestamp(data: dict, base_filename: str, config: dict, subfolder: str = "results") -> str:
        FileUtils.ensure_dir_exists(subfolder)
        FileUtils.save_config_once(config, subfolder)

        filepath = os.path.join(subfolder, f"{base_filename}.json")
        FileUtils.save_json(data, filepath)
        print(f"Results saved to {filepath}")
        return filepath


class TxtMetricLogger(MetricLoggerBase):
    def __init__(self, config: Dict):
        self.config = config
        self.folder_name = self._generate_experiment_folder()
        self.workspace_dir = config.get('workspace_dir', f"{config.get('output').get('results_dir')}/{self.folder_name}")
        FileUtils.ensure_dir_exists(self.workspace_dir)

        file_name = f'{self.folder_name}_metrics.txt'
        self.metric_file_path = os.path.join(self.workspace_dir, file_name)

    def _generate_experiment_folder(self) -> str:
        # model_name = self.config.get('model', {}).get('name', 'default_model')
        experiment_name = self.config.get('model', {}).get('experiment_name', 'default_experiment')
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return f"{experiment_name}_{timestamp}"

    def _append_line(self, line: str):
        with open(self.metric_file_path, 'a') as f:
            f.write(f"{line}\n")

    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        self._append_line(f"{metric_name}: {value}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for metric_name, value in metrics.items():
            self.log_json(value, metric_name)

    def log_json(self, params: dict, base_filename: str):
        FileUtils.save_json_with_timestamp(
            data=params,
            base_filename=base_filename,
            config=self.config,
            subfolder=self.workspace_dir
        )

    def log_params(self, params: Dict[str, str]):
        for param_name, value in params.items():
            self._append_line(f"{param_name}: {value}")

    def log_artifact(self, artifact_path: str):
        artifact_save_path = os.path.join(self.workspace_dir, os.path.basename("artifact_path.txt" ))
        with open(artifact_save_path, 'w') as f:
            f.write(f"Artifact path: {artifact_path}\n")



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
