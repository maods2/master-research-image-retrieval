from abc import ABC, abstractmethod

from utils.checkpoint_utils import save_model_and_log_artifact

class BaseTrainer(ABC):
        def __init__(self, config):
            self.config = config

        @abstractmethod
        def train_epoch(self):
            raise NotImplementedError("Method not implemented")

        @abstractmethod
        def train(self, num_epochs):
            raise NotImplementedError("Method not implemented")
            
        def save_model_if_best(self, model, metric, best_metric, model_path, metric_logger, mode='loss'):
            """
            Save the model if the current metric is better than the best metric.

            Args:
            epoch (int): Current epoch number.
            model (torch.nn.Module): The model to be saved.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            metric (float): The current metric value (loss or accuracy).
            best_metric (float): The best metric value so far.
            model_path (str): Path to save the model.
            config (dict): Configuration dictionary.
            metric_logger (object): Logger to log the metrics.
            mode (str): Mode to determine whether to save based on 'loss' or 'accuracy'. Default is 'loss'.

            Returns:
            str: Path to the saved model checkpoint if the model is saved, otherwise None.
            """
            if (mode == 'loss' and metric < best_metric) or (mode == 'accuracy' and metric > best_metric):
                best_metric = metric
                checkpoint_path = save_model_and_log_artifact(
                    metric_logger, self.config , model, filepath=model_path
                )
                return checkpoint_path
            return None
            