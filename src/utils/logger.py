import logging
import os
from utils.checkpoint_utils import generate_experiment_folder


def setup_logger(config):
    logging_config = config['logging']
    workspace_dir, _ = generate_experiment_folder(config)
    
    log_file = os.path.join(workspace_dir, logging_config['log_file'])
    logger = logging.getLogger()
    logger.setLevel(logging_config['log_level'])
    handler = logging.FileHandler(log_file)
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    logger.addHandler(handler)

    return logger
