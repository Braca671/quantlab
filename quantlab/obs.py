"""PASSO 4 — Optimal Brain Surgeon a ordine fisso: la versione lenta ed esatta.  [DA IMPLEMENTARE]

Si quantizzano le colonne di W una alla volta, da sinistra a destra. Quando la colonna i
viene fissata al valore quantizzato q, l'errore commesso si "spalma" in modo ottimo sulle
colonne non ancora quantizzate, così da minimizzare E(Q) (vedi hessian.py).

Derivazione da fare a mano (è il cuore del progetto): per una riga w, con F l'insieme
delle colonne ancora libere e H_F^{-1} l'inversa dell'Hessiana ristretta a F, fissare la
coordinata i a q produce l'aggiornamento ottimo

    delta = - (w_i - q) / [H_F^{-1}]_ii  *  [H_F^{-1}]_{:, i}

e poi la colonna i va rimossa da F. Ricavalo con un moltiplicatore di Lagrange.

Qui l'inversa ristretta si aggiorna in modo esplicito a ogni passo: costa O(d^3) per riga
in totale ed è lenta, ma è il riferimento corretto contro cui testare GPTQ.

Convenzioni (i test le assumono):
- H e W si preparano con prepare_hessian(H, W, damp);
- i parametri di quantizzazione si calcolano UNA volta, per canale, da W_clean
  prima di iniziare, e restano fissi durante tutto il ciclo.

Studio: Hassibi & Stork (1993), "Optimal Brain Surgeon"; Frantar & Alistarh (2022), sez. 3–4.
Verifica: pytest tests/test_step4_obs.py
"""
import torch


def obs_quantize(W: torch.Tensor, H: torch.Tensor, bits: int, damp: float = 0.01) -> torch.Tensor:
    """Restituisce Q (stessa forma di W) quantizzata con OBS a ordine fisso."""
    raise NotImplementedError("Passo 4")
