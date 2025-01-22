from dataloaders.factory_loaders import get_dataloader
from dataloaders.factory_transforms import get_transforms
from models.factory import get_model
from pipelines.test_pipes.factory import get_test_function
from utils.logger import setup_logger
from utils.metric_logger import setup_metric_logger


def test_wrapper(config):
    logger = setup_logger(config["logging"])
    metric_logger = setup_metric_logger(config)

    # load model
    transforms = get_transforms(config['transform'])
    model = get_model(config["model"])

    train_loader, test_loader = get_dataloader(config["data"], transforms)

    # Função de teste
    test_fn = get_test_function(config["testing"])

    # Teste
    logger.info("Running test...")
    test_fn(model, train_loader, test_loader, config, logger, metric_logger)
