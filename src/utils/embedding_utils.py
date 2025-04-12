import time
from typing import Dict, Tuple, Any
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm

def invert_dict(d):
    """Convert a dictionary to an inverted dictionary where keys become values and values become keys."""
    if isinstance(d, np.ndarray):
        d = d.item()  # Convert numpy array to dictionary
    return {v: k for k, v in d.items()}

def create_embeddings(
    model, data_loader, device, logger, desc='Extracting features'
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
        embeddings.append(embedding.cpu().numpy())
        labels.append(label.argmax(dim=1).cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    logger.info(
        f'Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}'
    )
    return embeddings, labels
    
    
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
    logger.info('Creating embeddings database from training data...')
    db_embeddings, db_labels = create_embeddings(
        model, train_loader, device, logger, desc='Creating database'
    )

    logger.info('Generating query embeddings from test data...')
    query_embeddings, query_labels = create_embeddings(
        model, test_loader, device, logger, desc='Generating queries'
    )

    embeddings = {
        'db_embeddings': db_embeddings,
        'db_labels': db_labels,
        'db_path': train_loader.dataset.image_paths,
        'query_embeddings': query_embeddings,
        'query_labels': query_labels,
        'query_classes': test_loader.dataset.labels,
        'query_paths': test_loader.dataset.image_paths,
        'class_mapping': invert_dict(train_loader.dataset.class_mapping),
    }
    
    if config['testing']['save_embeddings']:
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        path = config['testing']['embeddings_save_path'] + f'_{timestamp}.npz'
        np.savez(
           path,
           **embeddings
        )
        logger.info(f'Embeddings saved to {path}')

        
    return embeddings, path