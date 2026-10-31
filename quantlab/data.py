"""Caricamento CIFAR-10. [INFRASTRUTTURA — già pronta]"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _transforms(train: bool):
    t = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose(t + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def cifar10_loaders(root: str = "data", batch_size: int = 128, num_workers: int = 2):
    """Restituisce (train_loader, test_loader). Scarica il dataset al primo uso."""
    train = datasets.CIFAR10(root, train=True, download=True, transform=_transforms(True))
    test = datasets.CIFAR10(root, train=False, download=True, transform=_transforms(False))
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test, 512, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


def calibration_loader(n: int = 512, root: str = "data", batch_size: int = 128, seed: int = 0):
    """n immagini di training SENZA augmentation, per stimare le statistiche delle attivazioni.

    Il set di calibrazione viene dal train set: usare il test set sarebbe un leak.
    """
    ds = datasets.CIFAR10(root, train=True, download=True, transform=_transforms(False))
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n]
    return DataLoader(Subset(ds, idx.tolist()), batch_size, shuffle=False)
