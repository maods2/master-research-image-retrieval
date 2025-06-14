from dataloaders.transform_factory import get_transforms
from dataloaders.dataset_factory import get_dataloader
from optimizers.optimizer_factory import get_optimizer
from pipelines.testing_pipes.test_pipeline_factory import get_test_function
from pipelines.training_pipes.train_factory import get_train_function
from models.model_factory import get_model
from losses.loss_factory import get_loss
from utils.logger import setup_logger
from utils.metric_logger import setup_metric_logger

# from metrics.retrieval_metrics import evaluate


def train_wrapper(config):
    logger = setup_logger(config)
    metric_logger = setup_metric_logger(config)

    # Load model, loss, and optimizer
    model = get_model(config['model'])
    loss_fn = get_loss(config['loss'])
    optimizer = get_optimizer(config['optimizer'], model)

    # Load transformations
    transforms_train = get_transforms(config['transform'].get('train', None))
    transforms_test = get_transforms(config['transform'].get('test', None))

    # Load data loaders
    train_loader, test_loader = get_dataloader(
        config, transforms_train, transforms_test
    )

    # Custom training function
    train_fn = get_train_function(config)
    test_fn = get_test_function(config['testing'])

    # Training
    logger.info('Starting training...')
    train_fn(
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger,
    )

    # Teste padrão
    if config['testing']['enabled']:
        logger.info('Running testing...')
        test_fn(
            model, train_loader, test_loader, config, logger, metric_logger
        )
