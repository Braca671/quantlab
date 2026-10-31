"""PASSO 3 — Il problema di ricostruzione del layer e la sua Hessiana.  [DA IMPLEMENTARE]

GPTQ non minimizza ||W - Q|| (questo lo fa già RTN), ma l'errore sulle USCITE
del layer, misurato sui dati di calibrazione:

    E(Q) = || W X - Q X ||_F^2  =  tr( (W - Q) H (W - Q)^T ) / 2,    H = 2 X X^T

H (d x d) è l'Hessiana di E rispetto a ogni riga di Q, ed è la stessa per tutte le righe.
In questo repo H è normalizzata per il numero di colonne di X:  H = 2/n * X X^T.

Studio: Frantar & Alistarh (2022), "Optimal Brain Compression", sez. 3;
        Frantar et al. (2023), "GPTQ", sez. 3.
Verifica: pytest tests/test_step3_hessian.py
"""
import torch


class HessianAccumulator:
    """Accumula H = 2/n * X X^T su più batch, senza tenere in memoria tutta X.

    Dopo aver aggiunto i batch X_1 (d, n_1), X_2 (d, n_2), ... l'attributo H deve
    valere 2/(n_1 + n_2 + ...) * [X_1 X_1^T + X_2 X_2^T + ...], esattamente come se
    le colonne fossero state concatenate.
    """

    def __init__(self, d: int, device=None, dtype=torch.float64):
        raise NotImplementedError("Passo 3")

    def add_batch(self, X: torch.Tensor) -> None:
        """X: matrice di ingressi (d, n_batch)."""
        raise NotImplementedError("Passo 3")


def reconstruction_error(W: torch.Tensor, Q: torch.Tensor, H: torch.Tensor) -> float:
    """tr((W - Q) H (W - Q)^T): la quantità che GPTQ minimizza (a meno di costanti)."""
    raise NotImplementedError("Passo 3")


def prepare_hessian(H: torch.Tensor, W: torch.Tensor, damp: float = 0.01):
    """Rende H invertibile e numericamente stabile. Restituisce COPIE (H_damped, W_clean).

    1. Colonne "morte": se H[i, i] == 0 l'ingresso i è sempre nullo nei dati di calibrazione.
       Il peso corrispondente non influenza l'uscita: si pone W[:, i] = 0 e H[i, i] = 1.
    2. Damping: si somma damp * media(diag(H)) alla diagonale (dopo il passo 1).
    """
    raise NotImplementedError("Passo 3")
