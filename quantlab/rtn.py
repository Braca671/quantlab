"""PASSO 2b — Baseline: round-to-nearest (RTN).  [DA IMPLEMENTARE]

Ogni peso viene arrotondato al punto più vicino della griglia, indipendentemente
dagli altri. È la baseline che GPTQ deve battere: semplice, veloce, nessun dato
di calibrazione, ma ignora le correlazioni tra gli ingressi del layer.

Verifica: pytest tests/test_step2_layers_rtn.py
"""
from torch import nn


def rtn_quantize_layer_(layer: nn.Module, bits: int, per_channel: bool = True) -> nn.Module:
    """Quantizza IN PLACE i pesi di un Conv2d o Linear (il bias resta in float). Restituisce il layer."""
    raise NotImplementedError("Passo 2b")


def rtn_quantize_model_(model: nn.Module, bits: int, per_channel: bool = True, skip=()) -> nn.Module:
    """Applica RTN a tutti i Conv2d/Linear del modello, tranne i nomi in `skip`. IN PLACE."""
    raise NotImplementedError("Passo 2b")
