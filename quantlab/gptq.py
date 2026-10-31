"""PASSO 5 — GPTQ: lo stesso risultato di OBS, ma veloce.  [DA IMPLEMENTARE]

Due idee trasformano obs_quantize in un algoritmo che scala a miliardi di parametri:

1. Cholesky. Con l'ordine fisso, le righe di H_F^{-1} che servono a ogni passo sono
   esattamente le righe del fattore di Cholesky (superiore) di H^{-1}, a meno di una
   normalizzazione. Si calcola una sola fattorizzazione all'inizio, invece di aggiornare
   l'inversa a ogni colonna. Capire PERCHÉ è vero è l'esercizio principale di questo passo.

2. Aggiornamenti a blocchi ("lazy batch"). Si elaborano `block_size` colonne alla volta
   e si propaga l'errore accumulato al resto della matrice con un'unica moltiplicazione:
   stesso risultato matematico, molta meno memoria letta e scritta.

Il risultato deve coincidere con obs_quantize (a meno dell'errore di arrotondamento)
per QUALSIASI block_size. Stesse convenzioni di obs.py su damping e qparams.

Studio: Frantar et al. (2023), "GPTQ", sez. 4; Trefethen & Bau, "Numerical Linear Algebra",
        lezione 23 (Cholesky).
Verifica: pytest tests/test_step5_gptq.py
"""
import torch


def gptq_quantize(W: torch.Tensor, H: torch.Tensor, bits: int, damp: float = 0.01,
                  block_size: int = 128) -> torch.Tensor:
    """Restituisce Q (stessa forma di W) quantizzata con GPTQ."""
    raise NotImplementedError("Passo 5")
