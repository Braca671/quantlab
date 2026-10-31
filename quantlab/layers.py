"""PASSO 2a — Vedere ogni layer come un prodotto matrice-matrice.  [DA IMPLEMENTARE]

Tutti gli algoritmi del repo lavorano su una matrice di pesi W (out, d) e su una
matrice di ingressi X (d, n), con uscita Y = W @ X (bias escluso).

- Linear:  W è già (out, in); X sono i vettori di ingresso messi in colonna.
- Conv2d:  la convoluzione è un prodotto matriciale dopo "im2col":
           ogni colonna di X è una patch (in_ch * kh * kw) dell'immagine, e W è il
           kernel appiattito a (out_ch, in_ch * kh * kw). Una patch per ogni posizione
           di uscita e per ogni immagine del batch: n = batch * H_out * W_out.

L'ordine di appiattimento di W e quello delle patch in X devono coincidere: il test
verifica proprio che W @ X riproduca l'uscita del layer.
Limite accettato: solo Conv2d con groups=1.

Studio: documentazione di torch.nn.functional.unfold; cerca "im2col convolution".
Verifica: pytest tests/test_step2_layers_rtn.py
"""
import torch
from torch import nn


def weight_as_matrix(layer: nn.Module) -> torch.Tensor:
    """Restituisce una COPIA dei pesi come matrice (out, d)."""
    raise NotImplementedError("Passo 2a")


def set_weight_from_matrix(layer: nn.Module, W: torch.Tensor) -> None:
    """Scrive la matrice W (out, d) nei pesi del layer, ripristinandone la forma originale."""
    raise NotImplementedError("Passo 2a")


def inputs_as_matrix(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Trasforma l'ingresso del layer nella matrice X (d, n) tale che W @ X = uscita senza bias.

    Args:
        x: per Linear forma (..., in); per Conv2d forma (B, C, H, W).
    """
    raise NotImplementedError("Passo 2a")
