"""Modelli usati come banco di prova. [INFRASTRUTTURA — già pronta]

SmallCNN è volutamente senza BatchNorm: ogni layer con pesi è un Conv2d o un
Linear "puro", quindi la quantizzazione si applica direttamente senza dover
prima fondere la normalizzazione nei pesi.
"""
import torch
from torch import nn


class SmallCNN(nn.Module):
    """CNN da ~0.67M parametri per CIFAR-10 (32x32x3). Accuratezza attesa intorno all'80% (misurala al primo training)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def quantizable_layers(model: nn.Module):
    """Restituisce [(nome, layer)] dei Conv2d e Linear, nell'ordine del forward."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
