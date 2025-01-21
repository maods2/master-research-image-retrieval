from dataloaders.factory_loaders import get_dataloader
from models.factory import get_model
from pipelines.test_pipes.factory import get_test_function
from utils.logger import setup_logger


def test_wrapper(config):
    logger = setup_logger(config["logging"])

    # Carregar modelo e data loaders
    model = get_model(config["model"])

    test_loader = get_dataloader(config["data"]["test"])

    # Função de teste
    test_fn = get_test_function(config["testing"])

    # Teste
    logger.info("Running test...")
    test_fn(model, test_loader, config, logger)
