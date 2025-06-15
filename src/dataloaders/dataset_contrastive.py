import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataloaders.dataset import StandardImageDataset
import torch
import cv2


class ContrastiveDataset(StandardImageDataset):
    """
    Returns two randomly augmented views per sample, using an optional base transform
    """

    def __init__(
        self, root_dir, transform=None, class_mapping=None, config=None
    ):
        super().__init__(root_dir, transform=None, class_mapping=class_mapping)
        self._transform = transform
        self.validation_dataset = None

    def _open_image(self, path):
        """
        Open an image file and convert it to RGB format.
        """
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f'Failed to load image at {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _validation__getitem__(self, idx):
        """
        Fetch a single image and its label for validation purposes.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open and transform the image
        image = self._open_image(image_path)
        if self._transform:
            image = self._transform(image=image)['image']

        return image, label

    def __getitem__(self, idx):
        if self.validation_dataset is not None:
            return self._validation__getitem__(idx)

        image, label = super().__getitem__(idx)

        if self._transform:
            img1 = self._transform(image=image)['image']
            img2 = self._transform(image=image)['image']
        else:
            img1 = img2 = image

        return torch.stack([img1, img2]), label

    def set_transform(self, transform):
        """
        ovwerride the base class method to set a new transform.
        """
        self._transform = transform
