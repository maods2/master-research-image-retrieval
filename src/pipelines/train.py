from src.dataloaders.factory_transforms import get_transforms
from src.dataloaders.factory_loaders import get_dataloader
from src.optimizers.factory import get_optimizer
from src.pipelines.train_pipes.factory import get_train_function
from src.models.factory import get_model
from src.losses.factory import get_loss
from src.utils.logger import setup_logger
from src.utils.catalog import save_artifacts
# from src.metrics.retrieval_metrics import evaluate

def train_wrapper(config):
    logger = setup_logger(config["logging"])

    # Load model, loss, and optimizer
    model = get_model(config["model"])
    loss_fn = get_loss(config["loss"])
    optimizer = get_optimizer(config["optimizer"], model)
    
    # Load transformations
    transforms = get_transforms(config['transform'])

    # Load data loaders
    train_loader, test_loader = get_dataloader(config["data"], transforms)


    # Custom training function
    train_fn = get_train_function(config["training"]["pipeline"])
    # test_fn = get_test_function(config["testing"]["pipeline"])
    # retrieval_fn = get_retrieval_function(config["retrieval"]["pipeline"])

    # Training
    logger.info("Starting training...")
    train_fn(model, loss_fn, optimizer, train_loader, config, logger)

    # Teste padrão
    logger.info("Running standard testing...")
    # test_fn(model, test_loader, config, logger)

    # Retrieval testing
    logger.info("Running retrieval testing...")
    # results = retrieval_fn(model, test_loader, config, logger)

    # Saving artifacts
    # save_artifacts(model, results, config["output"])
