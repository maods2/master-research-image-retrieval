import random
import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A

from dataloaders.dataset import StandardImageDataset
from dataloaders.dataset_terumo import TerumoImageDataset


class TripletDataset(StandardImageDataset):
    def __init__(
        self, root_dir, transform=None, class_mapping=None, config=None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform if transform else A.Compose([A.ToFloat()])
        super().__init__(root_dir, transform, class_mapping)

        self.validation_dataset = None
        self.num_classes = len(self.class_mapping)

    def __len__(self):
        return len(self.image_paths)

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
        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.validation_dataset is not None:
            return self._validation__getitem__(idx)

        anchor_path = self.image_paths[idx]
        anchor_class = self.labels[idx]

        # Choose a positive from the same class (different from the anchor)
        positive_path = random.choice(
            [p for p in self.image_dict[anchor_class] if p != anchor_path]
        )

        # Choose a negative from a different class
        negative_class = random.choice(
            [cls for cls in self.image_dict.keys() if cls != anchor_class]
        )
        negative_path = random.choice(self.image_dict[negative_class])

        # Load images using OpenCV
        anchor = cv2.cvtColor(cv2.imread(str(anchor_path)), cv2.COLOR_BGR2RGB)
        positive = cv2.cvtColor(
            cv2.imread(str(positive_path)), cv2.COLOR_BGR2RGB
        )
        negative = cv2.cvtColor(
            cv2.imread(str(negative_path)), cv2.COLOR_BGR2RGB
        )

        # Apply Albumentations transformations
        if self.transform:
            anchor = self.transform(image=anchor)['image']
            positive = self.transform(image=positive)['image']
            negative = self.transform(image=negative)['image']

        return anchor, positive, negative

    def set_transform(self, transform):
        self.transform = transform


class MixedTripletDataset(Dataset):
    def __init__(
        self, root_dir, transform=None, class_mapping=None, config=None
    ):
        self.triplet_active = True
        self._multiclass_dataset = TerumoImageDataset(
            root_dir, transform, class_mapping
        )
        self._triplet_dataset = TripletDataset(
            root_dir, transform, class_mapping
        )
        self.labels = self._multiclass_dataset.labels
        self.image_paths = self._multiclass_dataset.image_paths
        self.class_mapping = self._multiclass_dataset.class_mapping
        self.labels_str = self._multiclass_dataset.labels_str

    def __len__(self):
        if self.triplet_active:
            return self._triplet_dataset.__len__()
        return self._multiclass_dataset.__len__()

    def __getitem__(self, idx: int):
        if self.triplet_active:
            return self._triplet_dataset.__getitem__(idx)
        return self._multiclass_dataset.__getitem__(idx)

    def switch_to_classifcation_dataset(self):
        self.triplet_active = False

    def switch_to_triplet_dataset(self):
        self.triplet_active = True


# Example usage
if __name__ == '__main__':
    from albumentations.pytorch import ToTensorV2

    IMAGE_DIR = 'datasets/final/terumo/train'

    data_transforms = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )
    data = TripletDataset(IMAGE_DIR, data_transforms)
    print('Number of samples:', len(data))
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=True
    )
    for batch in data_loader:
        anchor, positive, negative = batch
        print(
            'Anchor shape:',
            anchor.shape,
            'Positive shape:',
            positive.shape,
            'Negative shape:',
            negative.shape,
        )
        break
