import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Type
import random
import shutil


def get_dataset(
        dataset: Type[Dataset], 
        train_transforms: transforms.Compose, 
        val_transforms: transforms.Compose, 
        root: str, 
        tta: bool = False,
        ) -> Tuple[Dataset, Dataset]:
    """Get train and test dataset with transforms"""

    download = False
    if not os.path.exists(root):
        download = True
    
    test_transforms = train_transforms if tta else val_transforms

    train_dataset = dataset(
        root, 
        transform=train_transforms, 
        target_transform=test_transforms,
        target_types='segmentation', 
        download=download
    )
    test_dataset = dataset(
        root, 
        split='test', 
        target_types='segmentation',
        target_transform=test_transforms
    )

    return train_dataset, test_dataset


def get_dataloaders(train_set: Dataset, test_set: Dataset, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Get train and test data loaders."""
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return train_loader, test_loader

