import os
import random
from pathlib import Path
import cv2
import torch

from torch.utils.data import Dataset

from PIL import Image
from tqdm import tqdm

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataloaders.dataset import StandardImageDataset


class FewShotFolderDataset(StandardImageDataset):
    def __init__(
        self, root_dir, transform=None, class_mapping=None, config=None
    ):
        """ """
        super(FewShotFolderDataset, self).__init__(
            root_dir, transform, class_mapping, config
        )

        self.root_dir = Path(root_dir)
        self.n_way = config['model'].get(
            'n_way', 2
        )   # number of classes per episode
        self.k_shot = config['model'].get(
            'k_shot', 5
        )   # support shots per class
        self.q_queries = config['model'].get(
            'q_queries', 6
        )   #  query shots per class
        self.transform = transform

        self.classes = [
            (i, cls) for i, cls in enumerate(list(self.class_mapping.keys()))
        ]

    def __len__(self):
        # Return the total number of samples divided by the number of queries per index
        return len(self.image_paths) // self.q_queries

    def _open_image(self, path):
        """
        Open an image file and convert it to RGB format.
        """
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f'Failed to load image at {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        # Seleciona n_way classes
        selected = random.sample(self.classes, self.n_way)
        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []

        for i, cls in selected:
            imgs = random.sample(
                self.image_dict[self.class_mapping[cls]],
                self.k_shot + self.q_queries,
            )
            support_paths = imgs[: self.k_shot]
            query_paths = imgs[self.k_shot :]

            for p in support_paths:
                img = self._open_image(p)
                if self.transform:
                    img = self.transform(image=img)['image']
                support_imgs.append(img)
                support_lbls.append(i)

            for p in query_paths:
                img = self._open_image(p)
                if self.transform:
                    img = self.transform(image=img)['image']
                query_imgs.append(img)
                query_lbls.append(i)

        support = torch.stack(support_imgs)      # [n_way*k_shot, C, H, W]
        query = torch.stack(query_imgs)        # [n_way*q_queries, C, H, W]
        return (
            support,
            torch.tensor(support_lbls),
            query,
            torch.tensor(query_lbls),
        )


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
    config = {'model': {'n_way': 2, 'k_shot': 5, 'q_queries': 6}}

    # Define transformations using Albumentations
    data_transforms = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )

    # Create the dataset
    dataset = FewShotFolderDataset(
        root_dir=root_dir,
        transform=data_transforms,
        class_mapping=custom_mapping,
        config=config,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    # Test the dataset
    support, s_lbls, query, q_lbls = next(iter(train_loader))
    print(f'Support shape: {support.shape}, Labels: {s_lbls}')
    print(f'Query shape: {query.shape}, Labels: {q_lbls}')
