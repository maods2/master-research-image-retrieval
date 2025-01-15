from torch.utils.data import DataLoader
from src.dataloaders.dataset_terumo import TerumoImageDataset

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

    # Create dataset instances for training and testing
    train_dataset = TerumoImageDataset(
        root_dir=data_config["train_dir"],  # Directory for training data
        transform=transform,                # Transformations to apply
        class_mapping=data_config["class_mapping"]  # Custom class mappings
    )

    test_dataset = TerumoImageDataset(
        root_dir=data_config["test_dir"],   # Directory for testing data
        transform=transform,                # Transformations to apply
        class_mapping=data_config["class_mapping"]  # Custom class mappings
    )
    
    # Create DataLoader instances for both datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],  # Define the batch size
        shuffle=True,                          # Shuffle for training
        num_workers=data_config["num_workers"],  # Number of workers for data loading
        pin_memory=True                        # Pin memory for faster data transfer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],  # Define the batch size
        shuffle=False,                         # No need to shuffle test set
        num_workers=data_config["num_workers"],  # Number of workers for data loading
        pin_memory=True                        # Pin memory for faster data transfer
    )

    return train_loader, test_loader