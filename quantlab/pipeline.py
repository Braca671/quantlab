"""PASSO 6 — Quantizzare un modello intero, layer dopo layer.  [DA IMPLEMENTARE]

GPTQ lavora su un layer alla volta. Per il modello intero:
1. si fa passare il set di calibrazione nel modello e, con un forward hook, si accumula
   la H di ogni layer (HessianAccumulator + inputs_as_matrix);
2. si quantizza il layer con `method` ("gptq" oppure "obs");
3. si passa al layer successivo.

Scelta progettuale da discutere (e da misurare): gli ingressi del layer k vanno raccolti
dal modello ORIGINALE o da quello in cui i layer 1..k-1 sono GIÀ quantizzati?
La seconda opzione permette a ogni layer di compensare l'errore dei precedenti.

Verifica: pytest tests/test_step6_pipeline.py
"""
from torch import nn


def quantize_model_(model: nn.Module, calib_loader, bits: int, method: str = "gptq",
                    damp: float = 0.01, sequential: bool = True, skip=(), device: str = "cpu") -> nn.Module:
    """Quantizza IN PLACE tutti i Conv2d/Linear del modello (tranne `skip`). Restituisce il modello.

    sequential=True: gli ingressi del layer k vengono dal modello con i layer precedenti
    già quantizzati. sequential=False: vengono tutti dal modello originale.
    """
    raise NotImplementedError("Passo 6")
