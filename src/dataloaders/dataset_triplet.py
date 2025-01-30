import random
import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A

from dataloaders.dataset_terumo import TerumoImageDataset


class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_mapping=None):
        self.root_dir = Path(root_dir)
        self.transform = transform if transform else A.Compose([A.ToFloat()])
        self.image_paths = []
        self.labels = []

        # Identify classes from directories
        classes = sorted(
            [
                folder.name
                for folder in self.root_dir.iterdir()
                if folder.is_dir()
            ]
        )
        self.class_mapping = class_mapping or {
            class_name: idx for idx, class_name in enumerate(classes)
        }

        if any(class_name not in self.class_mapping for class_name in classes):
            raise ValueError(
                'Some classes in the directory are not in the provided mapping.'
            )

        # Organize image paths by class
        self.image_dict = {class_name: [] for class_name in classes}
        for class_name in classes:
            class_dir = self.root_dir / class_name
            image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

            images = [
                file
                for file in class_dir.rglob('*')
                if file.suffix.lower() in image_extensions
            ]
            if len(images) < 2:
                raise ValueError(
                    f'The class {class_name} has less than two images, which makes it impossible to form positive pairs.'
                )

            self.image_dict[class_name].extend(images)
            self.image_paths.extend(images)
            self.labels.extend([class_name] * len(images))

        self.num_classes = len(classes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class MixedTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_mapping=None):
        self.triplet_active = True
        self._multiclass_dataset = TerumoImageDataset(
            root_dir, transform, class_mapping
        )
        self._triplet_dataset = TripletDataset(
            root_dir, transform, class_mapping
        )

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
