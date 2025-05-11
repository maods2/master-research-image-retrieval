import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataloaders.dataset import StandardImageDataset


class ContrastiveDataset(StandardImageDataset):
    """
    Returns two randomly augmented views per sample, using an optional base transform
    """
    def __init__(self, root_dir, transform=None, class_mapping=None):
        super().__init__(root_dir, transform=None, class_mapping=class_mapping)
        self.aug_transform = transform

    def __getitem__(self, idx):
        # load raw image (no transform in parent)
        img, lbl = super().__getitem__(idx)

        # create two augmented views
        v1 = self.aug_transform(image=img)['image'] if self.aug_transform is not None else img
        v2 = self.aug_transform(image=img)['image'] if self.aug_transform is not None else img
        return (v1, v2), lbl
