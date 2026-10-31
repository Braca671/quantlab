"""Metriche e utilità. [INFRASTRUTTURA — già pronta]"""
import copy

import torch
from torch import nn


@torch.no_grad()
def accuracy(model: nn.Module, loader, device: str = "cpu") -> float:
    model.eval().to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def clone(model: nn.Module) -> nn.Module:
    """Copia profonda: quantizza sempre una copia, mai il modello di riferimento."""
    return copy.deepcopy(model)


def model_size_bytes(model: nn.Module, bits: int | None = None) -> int:
    """Dimensione dei pesi dei layer Conv/Linear: float32, oppure `bits` per peso se indicato.

    Stima semplificata: ignora bias e parametri di quantizzazione (scale, zero).
    """
    n = sum(m.weight.numel() for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    return n * (bits if bits else 32) // 8
