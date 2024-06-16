import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Type


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

    train_dataset = dataset(root, train=True, transform=train_transforms, download=download)
    test_dataset = dataset(root, train=False, transform=test_transforms)

    return train_dataset, test_dataset


def get_dataloaders(train_set: Dataset, test_set: Dataset, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Get train and test data loaders."""
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
