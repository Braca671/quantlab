"""PASSO 1 — Quantizzazione uniforme asimmetrica.  [DA IMPLEMENTARE]

Un peso reale w viene rappresentato con un intero q a `bits` bit:

    q = clamp(round(w / scale) + zero, 0, 2**bits - 1)      (quantize)
    w_hat = scale * (q - zero)                              (dequantize)

`scale` e `zero` (lo "zero-point") si scelgono dal range [min, max] dei pesi.
Convenzioni di questo repo (i test le assumono):
- i pesi sono una matrice W di forma (out, in): una riga = un canale di uscita;
- per canale: un (scale, zero) per riga, forma (out, 1); per tensore: forma (1, 1);
- il range viene sempre esteso per includere lo 0: lo zero reale deve essere
  rappresentabile esattamente (pensa al padding di una convoluzione);
- `zero` è un intero (memorizzato come float);
- una riga tutta nulla non deve produrre divisioni per zero.

Studio: Nagel et al. (2021), "A White Paper on Neural Network Quantization", sez. 2.1–2.3.
Verifica: pytest tests/test_step1_quant.py
"""
import torch


def compute_qparams(W: torch.Tensor, bits: int, per_channel: bool = True):
    """Calcola (scale, zero) con il metodo min-max.

    Args:
        W: pesi, forma (out, in).
        bits: numero di bit (es. 8, 4, 3).
        per_channel: True -> un (scale, zero) per riga; False -> uno per tutto il tensore.

    Returns:
        (scale, zero), entrambi di forma (out, 1) oppure (1, 1).
    """
    raise NotImplementedError("Passo 1")


def quantize(W: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int) -> torch.Tensor:
    """Pesi reali -> livelli interi in [0, 2**bits - 1] (restituiti come float)."""
    raise NotImplementedError("Passo 1")


def dequantize(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Livelli interi -> valori reali rappresentabili."""
    raise NotImplementedError("Passo 1")


def fake_quant(W: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantizza e dequantizza: restituisce pesi float che assumono solo valori della griglia.

    È la "simulazione" della quantizzazione usata in tutto il repo per misurare l'errore.
    """
    raise NotImplementedError("Passo 1")
