from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

class TerumoImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_mapping=None):
        """
        Initializes the dataset.

        Args:
            root_dir (str): Root directory containing class subdirectories.
            transform (callable, optional): Transformations to be applied to the images.
            class_mapping (dict, optional): Custom mapping for classes.
                                             If None, it will be automatically created based on folder names.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []  # List of full paths to images
        self.labels = []       # List of text labels
        self.one_hot_labels = [] # List of one-hot encoded labels

        # Get class names from folder names
        classes = sorted([folder.name for folder in self.root_dir.iterdir() if folder.is_dir()])

        # Create automatic mapping if none is provided
        if class_mapping is None:
            class_mapping = {class_name: idx for idx, class_name in enumerate(classes)}
            
        for class_name in classes:
            if class_name not in class_mapping:
                raise ValueError(f"Class {class_name} not found in class mapping")

        self.class_mapping = class_mapping

        for class_name in classes:
            class_dir = self.root_dir / class_name
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
            
            # Collect all matching image files for each extension
            images = []
            for ext in image_extensions:
                images.extend(class_dir.rglob(ext))
            
            # Register images and labels
            for file_path in images:

                # Create one-hot encoding for the class
                one_hot_label = np.zeros(len(self.class_mapping), dtype=np.float32)
                one_hot_label[self.class_mapping[class_name]] = 1.0
                
                self.image_paths.append(file_path)
                self.labels.append(class_name)
                self.one_hot_labels.append(one_hot_label)
                
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns an image and its corresponding label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (transformed image, one-hot encoded label)
        """
        img_path = str(self.image_paths[idx])
        one_hot_label = torch.tensor(self.one_hot_labels[idx], dtype=torch.float32)

        # Load the image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations, if provided
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, one_hot_label

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