import os
import torch
from datetime import datetime


def save_checkpoint(
    model,
    filepath=None,
    optimizer=None,
    epoch=None,
    loss=None,
    scheduler=None,
    config=None,
):
    """
    Saves a checkpoint with the model's state and optionally optimizer, epoch, loss, and scheduler states.

    Args:
        model (torch.nn.Module): The PyTorch model.
        filepath (str, optional): Path to save the checkpoint file. If None, a file name will be auto-generated.
        optimizer (torch.optim.Optimizer, optional): Optimizer used during training. Default is None.
        epoch (int, optional): Current training epoch. Default is None.
        loss (float, optional): Last recorded loss value. Default is None.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.
        model_id (str): Identifier for the model, included in the generated file name.

    Returns:
        str: The file path of the saved checkpoint.
    """
    checkpoint = {'model_state_dict': model.state_dict()}

    # Optionally include additional parameters in the checkpoint
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Generate file name if not provided config['workspace_dir']
    if config and not filepath:
        workspace_dir = config.get('workspace_dir', './')
        model_name = config.get('model', {}).get('name', 'default_model')
        experiment_name = config.get('model', {}).get(
            'experiment_name', 'default_experiment'
        )
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = (
            f'{experiment_name}_{model_name}_{timestamp}_checkpoint.pth'
        )
        filepath = os.path.join(workspace_dir, file_name)

    elif not filepath:
        date_str = datetime.now().strftime('%Y-%m-%d')
        filepath = f'checkpoint_{date_str}.pth'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved at {filepath}')
    return filepath


# Saving artifacts
def save_model_and_log_artifact(metric_logger, config, model, filepath=None):
    filepath = save_checkpoint(model, filepath=filepath, config=config)
    metric_logger.log_artifact(filepath)
    return filepath


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Loads a checkpoint and restores the state of the model, optimizer, and scheduler if available.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The PyTorch model.
        optimizer (torch.optim.Optimizer, optional): Optimizer used during training. Default is None.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.

    Returns:
        dict: A dictionary with optional keys 'epoch' and 'loss' from the checkpoint.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model state loaded from {filepath}')

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Optimizer state restored.')

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('Scheduler state restored.')

    return {
        'epoch': checkpoint.get('epoch', None),
        'loss': checkpoint.get('loss', None),
    }


def load_full_model(model_name: str, save_dir: str, map_location=None):
    """
    Loads a previously saved model using torch.save.

    Args:
        model_name: Name of the saved model (without extension)
        save_dir: Directory where the model is saved
        map_location: 'cpu' or 'cuda', defines where the model will be loaded

    Returns:
        model: Loaded PyTorch model
    """
    load_path = os.path.join(f'./assets/{model_name}', f'{model_name}.pt')
    model = torch.load(
        load_path, map_location=map_location, weights_only=False
    )
    print(f'Model loaded from: {load_path}')
    return model
