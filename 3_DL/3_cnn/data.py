import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


def get_mnist(root: str) -> tuple[datasets.mnist.MNIST, datasets.mnist.MNIST]:
    """Get train and test MNIST dataset with normalize transform"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    download = False
    if not os.path.exists(root):
        download = True

    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform)

    return train_dataset, test_dataset


def get_dataloaders(train_set: Dataset, test_set: Dataset, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
