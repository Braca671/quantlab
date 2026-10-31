# quantlab — Quantizzazione post-training da zero

Implementazione da zero di **RTN**, **Optimal Brain Surgeon** e **GPTQ** per quantizzare reti
neurali a 8, 4, 3 e 2 bit, con un'analisi di quando e perché la compensazione dell'errore
basata sull'Hessiana funziona. Banco di prova: una CNN su CIFAR-10, poi GPT-2 small, poi il
deploy su microcontrollore STM32.

> 🚧 Progetto in corso. Lo stato di avanzamento è nella tabella qui sotto.

## Stato

| Passo | Modulo | Stato |
|---|---|---|
| 0. Setup e modello float | `notebooks/00_setup.ipynb` | ⬜ |
| 1. Quantizzazione uniforme | `quantlab/quant.py` | ⬜ |
| 2. Layer come matrici + baseline RTN | `quantlab/layers.py`, `rtn.py` | ⬜ |
| 3. Hessiana della ricostruzione | `quantlab/hessian.py` | ⬜ |
| 4. OBS a ordine fisso (riferimento esatto) | `quantlab/obs.py` | ⬜ |
| 5. GPTQ (Cholesky + blocchi) | `quantlab/gptq.py` | ⬜ |
| 6. Modello intero | `quantlab/pipeline.py` | ⬜ |
| 7. GPT-2 small, perplexity WikiText-2 | — | ⬜ |
| 8. Deploy su STM32G491RE | — | ⬜ |

Il piano dettagliato, con letture e domande per ogni passo, è in [ROADMAP.md](ROADMAP.md).

## Struttura

```
quantlab/
  models.py, data.py, evaluate.py   infrastruttura (pronta)
  quant.py      passo 1   quantizzazione uniforme asimmetrica
  layers.py     passo 2a  Conv2d/Linear come prodotto W @ X (im2col)
  rtn.py        passo 2b  baseline round-to-nearest
  hessian.py    passo 3   H = 2/n X Xᵀ, errore di ricostruzione, damping
  obs.py        passo 4   OBS a ordine fisso, versione lenta ed esatta
  gptq.py       passo 5   GPTQ
  pipeline.py   passo 6   quantizzazione del modello layer per layer
tests/          un file per passo: sono la specifica di ogni modulo
scripts/        train_cnn.py
notebooks/      00_setup.ipynb, poi un notebook per ogni esperimento
```

## Uso

```bash
pip install -r requirements.txt
pytest tests/test_step1_quant.py      # i test del passo su cui stai lavorando
python scripts/train_cnn.py --epochs 20
```

## Riferimenti

Frantar et al. (2023) *GPTQ* · Frantar & Alistarh (2022) *Optimal Brain Compression* ·
Hassibi & Stork (1993) *Optimal Brain Surgeon* · Nagel et al. (2021) *A White Paper on Neural Network Quantization*.
