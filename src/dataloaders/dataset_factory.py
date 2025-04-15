from torch.utils.data import DataLoader
from dataloaders.dataset import StandardImageDataset
from dataloaders.dataset_terumo import TerumoImageDataset
from dataloaders.dataset_triplet import MixedTripletDataset, TripletDataset


def get_dataloader(data_config, transform):
    """
    Factory function to get data loaders for train and test sets.

    Args:
        data_config (dict): Configuration dict containing information on data directories,
                             transformations, and other parameters.
        transform (A.Compose): Composed Albumentations transformation pipeline.

    Returns:
        train_loader (DataLoader): PyTorch DataLoader for the training set
        test_loader (DataLoader): PyTorch DataLoader for the test set
    """
    # Get transformations based on config
    dataset_name = data_config.get(
        'dataset_type', 'TerumoImageDataset'
    )  # Default to TerumoImageDataset

    # Select dataset class dynamically based on config
    if dataset_name == 'StandardImageDataset':
        dataset_class = StandardImageDataset
    elif dataset_name == 'TerumoImageDataset':
        dataset_class = TerumoImageDataset
    elif dataset_name == 'TripletDataset':
        dataset_class = TripletDataset
    elif dataset_name == 'MixedTripletDataset':
        dataset_class = MixedTripletDataset
    else:
        raise ValueError(f'Dataset {dataset_name} is not supported.')
    # Create dataset instances for training and testing
    train_dataset = dataset_class(
        root_dir=data_config['train_dir'],  # Directory for training data
        transform=transform,  # Transformations to apply
        class_mapping=data_config['class_mapping'],  # Custom class mappings
    )

    # TODO: add support for different transformations for test set
    test_dataset = dataset_class(
        root_dir=data_config['test_dir'],  # Directory for testing data
        transform=transform,  # Transformations to apply
        class_mapping=data_config['class_mapping'],  # Custom class mappings
    )

    # Create DataLoader instances for both datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],  # Define the batch size
        shuffle=data_config.get("suffle_train", True),  # Shuffle for training
        num_workers=data_config[
            'num_workers'
        ],  # Number of workers for data loading
        pin_memory=True,  # Pin memory for faster data transfer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],  # Define the batch size
        shuffle=data_config.get("suffle_test", False),  # No need to shuffle test set
        num_workers=data_config[
            'num_workers'
        ],  # Number of workers for data loading
        pin_memory=True,  # Pin memory for faster data transfer
    )

    return train_loader, test_loader
