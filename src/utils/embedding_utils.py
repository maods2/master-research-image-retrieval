import time
from typing import Dict, Tuple, Any
from dataloaders.transform_factory import get_transforms
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
from tqdm import tqdm
import os


def invert_dict(d):
    """Convert a dictionary to an inverted dictionary where keys become values and values become keys."""
    if isinstance(d, np.ndarray):
        d = d.item()  # Convert numpy array to dictionary
    return {v: k for k, v in d.items()}


def create_embeddings(
    model,
    data_loader,
    normalize_embeddings,
    device,
    logger,
    desc='Extracting features',
):
    """
    Generate embeddings and labels from a given data loader.

    Args:
        model: The model used for generating embeddings.
        data_loader: The data loader containing the data.
        device: The device to perform computations (e.g., 'cuda' or 'cpu').
        logger: Logger object for logging information.
        desc: Description for the progress bar.

    Returns:
        tuple: A tuple containing embeddings (np.ndarray) and labels (np.ndarray).
    """
    embeddings = []
    labels = []
    model.eval()
    model.to(device)
    data_loader = tqdm(data_loader, desc=desc)

    for img, label in data_loader:
        with torch.no_grad():
            embedding = model(img.to(device))
            if normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, dim=1)

        embeddings.append(embedding.cpu().numpy())
        # Handle both one-hot and standard labels
        if len(label.shape) > 1:  # one-hot encoded
            label = label.argmax(dim=1)
        labels.append(label.cpu().numpy())
        # labels.append(label.argmax(dim=1).cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    logger.info(
        f'Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}'
    )
    return embeddings, labels


def get_dataset_attribute(dataset, attribute_name: str):
    """
    Helper function to get attributes from either regular or subset dataset.

    Args:
        dataset: Dataset object (can be either regular dataset or subset)
        attribute_name: Name of the attribute to retrieve (e.g., 'labels', 'image_paths', 'class_mapping')

    Returns:
        The requested attribute value
    """
    if hasattr(dataset, 'dataset') and isinstance(
        dataset, Subset
    ):  # Subset dataset
        indices = dataset.indices
        original_dataset = dataset.dataset

        if attribute_name in ['image_paths', 'labels', 'labels_str']:
            # Handle list-type attributes that need to be subset
            original_attr = getattr(original_dataset, attribute_name)
            return [original_attr[i] for i in indices]
        else:
            # Handle other attributes (like class_mapping) that should be returned as-is
            return getattr(original_dataset, attribute_name)
    else:
        # Regular dataset - return attribute directly
        return getattr(dataset, attribute_name)


def create_embeddings_dict(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    logger: Any,
    config: Dict[str, Any],
) -> Dict[str, Tuple]:
    """
    Create a dictionary containing embeddings and labels for both training and test data.

    Args:
        model: PyTorch model used for generating embeddings.
        train_loader: DataLoader for the training data.
        test_loader: DataLoader for the testing data.
        device: Device to perform computations ('cuda' or 'cpu').
        logger: Logger object for logging information.

    Returns:
        Dict[str, Tuple]: A dictionary with keys 'db_embeddings', 'db_labels',
                          'query_embeddings', and 'query_labels'.
    """
    if hasattr(train_loader.dataset, 'validation_dataset'):
        test_loader.dataset.validation_dataset = True
        train_loader.dataset.validation_dataset = True

    if hasattr(train_loader.dataset, 'k_shot') and hasattr(
        test_loader.dataset, 'k_shot'
    ):
        train_loader.dataset.k_shot = 1
        test_loader.dataset.k_shot = 1

    train_loader.dataset.set_transform(
        get_transforms(config['transform'].get('test', None))
    )

    logger.info('Creating embeddings database from training data...')
    normalize_embeddings = config['testing'].get('normalize_embeddings', False)
    db_embeddings, db_labels = create_embeddings(
        model,
        train_loader,
        normalize_embeddings,
        device,
        logger,
        desc='Creating database',
    )

    logger.info('Generating query embeddings from test data...')
    query_embeddings, query_labels = create_embeddings(
        model,
        test_loader,
        normalize_embeddings,
        device,
        logger,
        desc='Generating queries',
    )

    # Use the generalist function to get attributes
    embeddings = {
        'db_embeddings': db_embeddings,
        'db_labels': db_labels,
        'db_path': get_dataset_attribute(train_loader.dataset, 'image_paths'),
        'query_embeddings': query_embeddings,
        'query_labels': query_labels,
        'query_classes': get_dataset_attribute(
            test_loader.dataset, 'labels_str'
        ),
        'query_paths': get_dataset_attribute(
            test_loader.dataset, 'image_paths'
        ),
        'class_mapping': invert_dict(
            get_dataset_attribute(train_loader.dataset, 'class_mapping')
        ),
    }

    if config['testing']['save_embeddings']:
        # Ensure the directory exists
        os.makedirs(config['workspace_dir'], exist_ok=True)

        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(
            config['workspace_dir'],
            f'embeddings_{timestamp}.npz',
        )
        np.savez(path, **embeddings)
        logger.info(f'Embeddings saved to {path}')
        return embeddings, path

    return embeddings, None


def load_or_create_embeddings(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    logger: Any,
    device: str = None,
) -> Tuple[Dict, str]:
    """
    Load existing embeddings or create new ones based on configuration.

    Args:
        model: PyTorch model to use for creating embeddings
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        config: Configuration dictionary containing testing settings
        logger: Logger instance for logging information
        device: Device to use for computation (default: None, will use cuda if available)

    Returns:
        Tuple containing:
            - Dictionary of embeddings
            - String path where embeddings were saved (if created)
    """
    if device is None:
        device = config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu'
        )
    logger.info(f'Using device: {device}')

    if config['testing'].get('load_embeddings', False):
        logger.info(
            f"Loading embeddings from {config['testing']['embeddings_path']}"
        )
        embeddings = np.load(
            config['testing']['embeddings_path'], allow_pickle=True
        )
        return embeddings, config['testing']['embeddings_path']

    logger.info('Creating new embeddings...')
    embeddings, file_path = create_embeddings_dict(
        model, train_loader, test_loader, device, logger, config
    )
    config['testing']['embeddings_path'] = file_path
    return embeddings
