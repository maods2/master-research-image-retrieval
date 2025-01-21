from pipelines.test_pipes.metrics.metric_base import MetricBase


class MapAtK(MetricBase):
    def __init__(self, k: int):
        self.k = k

    def __call__(self, model, train_loader, test_loader, config, logger):
        # Dummy implementation for MAP@K computation
        logger.info(f"Computing MAP@{self.k}...")
        map_k = 0.85  # Placeholder computation
        return {f"map@{self.k}": map_k}