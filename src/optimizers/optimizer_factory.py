import torch.optim as optim


def get_optimizer(optimizer_config, model):
    optimizer_name = optimizer_config['name']
    lr = float(optimizer_config['lr'])
    weight_decay = float(optimizer_config.get('weight_decay', 0.0))

    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f'Optimizer {optimizer_name} is not supported')

    return optimizer
