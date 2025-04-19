from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class StandardImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        class_mapping=None,
        config=None,
        return_one_hot=False,
    ):
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
        self.image_paths = []       # List of full paths to images
        self.labels = []            # List of integer labels
        self.one_hot_labels = []    # List of one-hot encoded labels
        self.labels_str = []        # List of text labels
        self.return_one_hot = return_one_hot

        # Get class names from folder names
        classes = sorted(
            [
                folder.name
                for folder in self.root_dir.iterdir()
                if folder.is_dir()
            ]
        )

        # Create automatic mapping if none is provided
        if class_mapping is None:
            class_mapping = {
                class_name: idx for idx, class_name in enumerate(classes)
            }

        for class_name in classes:
            if class_name not in class_mapping:
                raise ValueError(
                    f'Class {class_name} not found in class mapping'
                )

        self.class_mapping = class_mapping
        self.image_dict = {
            self.class_mapping[class_name]: [] for class_name in classes
        }

        for class_name in classes:
            class_dir = self.root_dir / class_name
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']

            # Collect all matching image files for each extension
            images = []
            for ext in image_extensions:
                images.extend(class_dir.rglob(ext))

            #
            self.image_dict[self.class_mapping[class_name]].extend(images)

            # Register images and labels
            for file_path in images:

                # Create one-hot encoding for the class
                one_hot_label = np.zeros(
                    len(self.class_mapping), dtype=np.float32
                )
                one_hot_label[self.class_mapping[class_name]] = 1.0

                self.image_paths.append(file_path)
                self.labels.append(self.class_mapping[class_name])
                self.one_hot_labels.append(one_hot_label)
                self.labels_str.append(class_name)

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

        if self.return_one_hot:
            label = torch.tensor(self.one_hot_labels[idx], dtype=torch.float32)
        else:
            label = self.labels[idx]

        # Load the image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f'Failed to load image at {img_path}')

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations, if provided
        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader

    root_dir = './datasets/final/glomerulo/train'
    custom_mapping = {
        'Crescent': 0,
        'Hypercellularity': 1,
        'Membranous': 2,
        'Normal': 3,
        'Podocytopathy': 4,
        'Sclerosis': 5,
    }

    # Define transformations using Albumentations
    data_transforms = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )

    # Create the dataset
    dataset = StandardImageDataset(
        root_dir=root_dir,
        transform=data_transforms,
        class_mapping=custom_mapping,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    # Test the dataset
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break
