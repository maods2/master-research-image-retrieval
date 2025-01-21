from pipelines.test_pipes.metrics.metric_base import MetricLoggerBase


class MLFlowMetricLogger(MetricLoggerBase):
    def log_metric(self, metric_name: str, value: float):
        mlflow.log_metric(metric_name, value)

    def log_metrics(self, metrics: dict):
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value)