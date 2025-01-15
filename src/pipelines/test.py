def test_wrapper(config):
    logger = setup_logger(config["logging"])

    # Carregar modelo e data loaders
    model = get_model(config["model"])
    model.load_state_dict(load_weights(config["model"]["weights"]))
    test_loader = get_dataloader(config["data"]["test"])

    # Função de teste
    test_fn = get_test_function(config["testing"]["pipeline"])

    # Teste
    logger.info("Running test...")
    test_fn(model, test_loader, config, logger)
