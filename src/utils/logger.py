import logging


def setup_logger(logging_config):
    logger = logging.getLogger()
    logger.setLevel(logging_config['log_level'])
    handler = logging.FileHandler(logging_config['log_file'])
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    logger.addHandler(handler)

    return logger
