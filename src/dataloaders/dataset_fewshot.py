import os
import random
from pathlib import Path
import cv2
import torch
import numpy as np
from typing import List, Tuple
from typing import Dict, Any
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
        self.validation_dataset = None
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
        """
        This will return the number of samples in the dataset, and will be used
        to determine the number of batches per epoch.

        Specifically for few-shot learning training phase, the batch size defined
        in the dataloader will be equal 1, due the __getitem__(self, idx) method will return
        a support set and a query set of size n_way*k_shot and n_way*q_queries
        respectively.

        The method __getitem__(self, idx) will return a tuple of 4 elements with the following shapes:
        - support: [n_way*k_shot, C, H, W]
        - support_labels: [n_way*k_shot]
        - query: [n_way*q_queries, C, H, W]
        - query_labels: [n_way*q_queries]
        For that reason, the training might be slow, because of the number of images processed in each batch.

        __getitem__(self, idx) will return randomly selected iamges from the dataset.
        To return the number of batches per epoch, we need to return the number of images in the dataset divided by the number of queries.
        len(self.image_paths) // self.q_queries

        As the __getitem__(self, idx) randomly selects images from the dataset, for huge datasets and havy we might benefit from
        returning the number of images less than (len(self.image_paths) // self.q_queries) during the training phase, since
        the images are randomly selected.
        e.g. if we have 10000 images in the dataset and we are using 5 queries, we will have 2000 batches per epoch.
        we can set the number of images to 250, so we will have 250 batches per epoch for heavy models/datasets.

        For the validation phase, we will return the number of images in the dataset, since we want to validate the model
        and the __getitem__(self, idx) will work as a normal classification dataset.


        """
        if self.validation_dataset is not None:
            return len(self.labels)
        return 250
        return len(self.image_paths) // self.q_queries

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

    def __getitem__(self, idx):
        if self.validation_dataset is not None:
            return self._validation__getitem__(idx)

        # Randomly select n_way classes
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


class SupportSetDataset(StandardImageDataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        class_mapping=None,
        config=None,
        n_per_class=None,
    ):
        """
        Dataset para carregar conjuntos de suporte para few-shot learning.

        Args:
            root_dir (str): Caminho base onde estão as imagens.
            transform (callable, optional): Transformações para aplicar nas imagens.
            class_mapping (dict): Mapeamento de nomes de classe para índices.
            config (dict): Dicionário com configurações, incluindo paths e transformações.
            n_per_class (int): Número de imagens por classe a serem amostradas.
        """
        self.root_dir = root_dir
        self.config = config
        self.samples = []
        self.n_per_class = n_per_class
        self.class_mapping = class_mapping or {}

        # Define transformações com Albumentations
        self.transform = transform or self._build_transforms(config)

        # Carrega dados se for um support set direto ou herda StandardImageDataset
        if 'support_set' in self.config['data']:
            self._load_support_set(self.config['data']['support_set'])
        else:
            super().__init__(
                root_dir=self.config['data']['train_dir'],
                transform=self.transform,
                class_mapping=self.class_mapping,
            )
            class_to_path = {
                cls: paths for cls, paths in self.image_dict.items()
            }
            self._load_support_set(class_to_path)

    def _build_transforms(self, config):
        resize_height, resize_width = tuple(config['transform'].get('resize'))
        normalize_mean = tuple(config['transform']['normalize'].get('mean'))
        normalize_std = tuple(config['transform']['normalize'].get('std'))

        return A.Compose(
            [
                A.Resize(resize_height, resize_width),
                A.Normalize(mean=normalize_mean, std=normalize_std),
                ToTensorV2(),
            ]
        )

    def _load_support_set(self, support_set_config):
        """
        Lê e armazena os paths das imagens do support set.
        """
        for class_name, paths in support_set_config.items():
            label = (
                self.class_mapping[class_name]
                if isinstance(class_name, str)
                else class_name
            )

            # Seleciona N imagens aleatórias, se necessário
            if self.n_per_class:
                paths = random.sample(paths, min(self.n_per_class, len(paths)))

            for img_path in paths:
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f'Image not found at {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    config = {'model': {'n_way': 6, 'k_shot': 5, 'q_queries': 6}}

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
    test_loader = DataLoader(
        dataset,
        batch_size=36,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    # Test the train loader
    support, s_lbls, query, q_lbls = next(iter(train_loader))
    print(f'Support shape: {support.shape}, Labels: {s_lbls.shape}')
    print(f'Query shape: {query.shape}, Labels: {q_lbls.shape}')

    # test the test loader
    test_loader.dataset.k_shot = 1
    test_loader.dataset.validation_dataset = True
    query, q_lbls = next(iter(test_loader))
    print(f'Query shape: {query.shape}, Labels: {q_lbls.shape}')
    print()
