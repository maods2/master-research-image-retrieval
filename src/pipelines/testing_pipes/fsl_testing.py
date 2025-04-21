from typing import Dict

import numpy as np

from dataloaders.dataset_fewshot import SupportSetDataset
from metrics.metric_factory import get_metrics
from metrics.metric_base import MetricBase, MetricLoggerBase
from typing import Any, List
import torch
from torch.utils.data import DataLoader

from pipelines.training_pipes.few_shot_train import FewShotTrain
from utils.embedding_utils import load_or_create_embeddings


def load_support_set_from_loader(config, device='cpu'):
    """
    Carrega todas as imagens e labels do support_loader e empilha em dois tensores únicos.

    Returns:
        support_set: Tensor [N_total, C, H, W]
        support_labels: Tensor [N_total]
    """
    support_loader = DataLoader(
        dataset=SupportSetDataset(
            config['data']['train_dir'],
            transform=None,
            class_mapping=config['data']['class_mapping'],
            config=config,
            n_per_class=config['model']['k_shot'],
        ),
        batch_size=10,
        shuffle=False,
    )

    all_images = []
    all_labels = []

    for images, labels in support_loader:
        all_images.append(images)
        all_labels.append(labels)

    support_set = torch.cat(all_images, dim=0).to(device)
    support_labels = torch.cat(all_labels, dim=0).to(device)

    return support_set, support_labels


def fsl_test_fn(
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

    device = (
        config['device']
        if config.get('device')
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    fslt = FewShotTrain(config={})
    model.to(device)
    support_set = load_support_set_from_loader(config, device=device)
    train_val = fslt.eval_few_shot_classification(
        model,
        train_loader,
        support_set,
        device=device,
        config=config,
        logger=logger,
    )

    test_val = fslt.eval_few_shot_classification(
        model,
        test_loader,
        support_set,
        device=device,
        config=config,
        logger=logger,
    )

    results = {'train': train_val, 'test': test_val}

    logger.info(f'Results for fsl classification: {results}')
    metric_logger.log_metrics(results)
