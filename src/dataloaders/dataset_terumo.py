from pathlib import Path
import time
import cv2
from src.dataloaders.dataset import StandardImageDataset
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

class TerumoImageDataset(StandardImageDataset):
    def __init__(self, root_dir, transform=None, class_mapping=None):
        """
        Initializes the dataset.

        Args:
            root_dir (str): Root directory containing class subdirectories.
            transform (callable, optional): Transformations to be applied to the images.
            class_mapping (dict, optional): Custom mapping for classes.
                                             If None, it will be automatically created based on folder names.
        """
        super().__init__(root_dir, transform, class_mapping)


# Example usage
if __name__ == "__main__":
    start = time.time()
    
    root_dir = "datasets/final/terumo/train"
    custom_mapping = {"Crescent": 0, "Hypercellularity": 1, "Membranous": 2, "Normal": 3, "Podocytopathy": 4, "Sclerosis": 5}
    
    # Define transformations using Albumentations
    data_transforms = A.Compose([
        A.Resize(224, 224),                        # Resize the image
        A.HorizontalFlip(p=0.5),                  # Random horizontal flip
        A.RandomBrightnessContrast(p=0.2),        # Random brightness and contrast adjustments
        A.Normalize(mean=(0.5, 0.5, 0.5),         # Normalize to [-1, 1]
                    std=(0.5, 0.5, 0.5)),
        ToTensorV2()                              # Convert to PyTorch tensor
    ])

    # Create the dataset
    dataset = TerumoImageDataset(root_dir=root_dir, transform=data_transforms, class_mapping=custom_mapping)

    # Example of access
    print("Dataset size:", len(dataset))
    print("Classes:", dataset.class_mapping)
    img, one_hot_label = dataset[0]
    print("First image shape:", img.shape, "One-hot label:", one_hot_label)
    
    # Create DataLoader
    train_loader = DataLoader(
        dataset,                      # Dataset instance
        batch_size=32,                      # Set batch size as per your preference
        shuffle=True,                       # Shuffle dataset for better generalization
        num_workers=3,                      # Number of CPU cores to use for loading data in parallel
        pin_memory=True,                    # Pin memory to speed up data transfer to GPU
    )
    i=0
    # Example usage: iterate over the DataLoader
    for images, one_hot_labels in train_loader:
        # print(images.shape, one_hot_labels)
        if i == 5:
            break
        i+=1
        
    end = time.time()
    print("Time taken:", end - start)