# Roadmap — un passo alla volta

Ogni passo ha la stessa struttura: **studia → implementa → verifica con i test → rispondi alle domande**.
Le domande non sono decorative: le risposte scritte (anche brevi) diventano il materiale del README finale
e la parte che distingue il progetto da un'implementazione copiata.

Regola di lavoro: prima di scrivere codice, ricava le formule **a mano**, su carta.
Il codice di riferimento ufficiale (IST-DASLab/gptq) si guarda solo **dopo** aver fatto passare i test.

---

## Passo 0 — Setup e modello di riferimento  ·  ~1 ora
- Apri `notebooks/00_setup.ipynb` su Colab (GPU T4), addestra SmallCNN (~20 epoche).
- Annota l'accuratezza float32: è il riferimento di tutto il progetto.
- **Domanda:** quanti byte occupano i pesi in float32? E a 8, 4, 3 bit? (`model_size_bytes`)

## Passo 1 — Quantizzazione uniforme  ·  `quant.py`  ·  ~2–3 ore
**Studia**
- Nagel et al. (2021), *A White Paper on Neural Network Quantization*, arXiv:2106.08295 — sezioni 2.1–2.3
  (quantizzazione uniforme affine, simmetrica vs asimmetrica, per tensore vs per canale).
- Jacob et al. (2018), *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only
  Inference* (CVPR) — sezione 2: perché lo zero deve essere rappresentabile esattamente.

**Verifica:** `pytest tests/test_step1_quant.py`

**Domande**
1. Perché l'errore di arrotondamento è al massimo `scale/2`, e quando questa garanzia salta?
2. Cosa succede alla griglia se un solo peso della riga è un outlier enorme?
3. Perché per canale e non per singolo peso? Qual è il costo in memoria dei parametri (scale, zero)?

## Passo 2 — Layer come matrici e baseline RTN  ·  `layers.py`, `rtn.py`  ·  ~3 ore
**Studia**
- Documentazione di `torch.nn.functional.unfold`; un qualsiasi tutorial su "im2col".
- Ripassa come CMSIS-NN implementa la convoluzione sulla STM32: è lo stesso trucco.

**Verifica:** `pytest tests/test_step2_layers_rtn.py`

**Esperimento (notebook 01):** accuratezza di SmallCNN con RTN a 8/4/3/2 bit, per tensore e per canale.
Prima tabella del progetto.

**Domande**
1. A quanti bit l'accuratezza crolla? Il crollo è graduale o improvviso?
2. Quale layer, quantizzato da solo, fa più danni? (quantizza un layer alla volta)
3. Perché il primo e l'ultimo layer vengono spesso lasciati a precisione più alta?

## Passo 3 — Il problema di ricostruzione e la sua Hessiana  ·  `hessian.py`  ·  ~2–3 ore
**Studia**
- Frantar & Alistarh (2022), *Optimal Brain Compression*, arXiv:2208.11580 — sezione 3
  (riformulazione layer-wise, perché H è uguale per tutte le righe).
- Frantar, Ashkboos, Hoefler, Alistarh (2023), *GPTQ*, arXiv:2210.17323 — sezione 3.

**Verifica:** `pytest tests/test_step3_hessian.py`

**Domande**
1. Dimostra che `||WX − QX||²_F = tr((W−Q) H (W−Q)ᵀ)/2` con `H = 2XXᵀ`.
2. Perché minimizzare `||W − Q||` (RTN) non basta? Costruisci un esempio a 2 dimensioni in cui RTN
   è ottimo per i pesi ma pessimo per le uscite.
3. Calcola lo spettro di H per ogni layer di SmallCNN: quanto è mal condizionata? Le colonne morte da dove vengono (pensa alla ReLU)?

## Passo 4 — OBS a ordine fisso, versione lenta  ·  `obs.py`  ·  ~4–6 ore (il passo più importante)
**Studia**
- Hassibi & Stork (1993), *Second order derivatives for network pruning: Optimal Brain Surgeon* (NeurIPS).
  Segui la derivazione con il moltiplicatore di Lagrange.
- LeCun, Denker, Solla (1990), *Optimal Brain Damage* — il predecessore, utile per capire cosa aggiunge OBS.
- Frantar & Alistarh (2022), sezione 4 — come OBS diventa un algoritmo di quantizzazione (OBQ).

**Esercizio su carta (prima del codice):** ricava la formula dell'aggiornamento ottimo quando fissi
una coordinata, e la formula per rimuovere una riga/colonna dall'inversa (complemento di Schur).

**Verifica:** `pytest tests/test_step4_obs.py`

**Domande**
1. Perché con H diagonale OBS coincide con RTN? Cosa significa in termini di dati?
2. OBQ originale sceglie a ogni passo il peso "più facile"; qui l'ordine è fisso. Cosa si perde?

## Passo 5 — GPTQ veloce  ·  `gptq.py`  ·  ~4–6 ore
**Studia**
- GPTQ, sezione 4 (ordine arbitrario, Cholesky, lazy batch updates).
- Trefethen & Bau, *Numerical Linear Algebra*, lezione 23 (fattorizzazione di Cholesky).
  In alternativa: Golub & Van Loan, *Matrix Computations*, sezione 4.2.

**Esercizio su carta:** dimostra che le righe di H_F⁻¹ usate a ogni passo di OBS coincidono, a meno di
un fattore, con le righe del fattore di Cholesky superiore di H⁻¹. Suggerimento: confronta gli
aggiornamenti successivi dell'inversa con i passi dell'eliminazione gaussiana.

**Verifica:** `pytest tests/test_step5_gptq.py` (deve coincidere con OBS per ogni `block_size`)

**Domande**
1. Perché senza damping la Cholesky fallisce sui layer reali?
2. Complessità di OBS e di GPTQ in funzione di `d` e `out`: misurala e confrontala con la teoria.
3. Il blocco non cambia il risultato: allora cosa cambia? (pensa alla gerarchia di memoria della GPU)

## Passo 6 — Modello intero  ·  `pipeline.py`  ·  ~3 ore
**Verifica:** `pytest tests/test_step6_pipeline.py`

**Esperimento (notebook 02):** GPTQ vs RTN su SmallCNN a 4/3/2 bit; `sequential=True` vs `False`;
dimensione del set di calibrazione (32, 128, 512, 2048 immagini); damping.

**Qui il codice ufficiale si può finalmente aprire** (github.com/IST-DASLab/gptq) per confrontare le scelte.

## Passo 7 — Transformer: GPT-2 small  ·  ~1 giornata, GPU Colab
- Stessa pipeline sui layer Linear di GPT-2 small (HuggingFace `transformers`), perplexity su WikiText-2.
- Confronta i tuoi numeri RTN e GPTQ con la tabella del paper GPTQ: è la prova che l'implementazione è giusta.
- **Domanda:** perché gli LLM sono molto più sensibili agli outlier nelle attivazioni di una CNN?

## Passo 8 — Deploy su STM32G491RE  ·  il ponte con il progetto MCU
- SmallCNN a int8 (e int4 per i pesi, se il runtime lo supporta) con X-CUBE-AI o TFLite Micro.
- Misure on-device: accuratezza, latenza (contatore DWT), flash, RAM di picco.
- **Domanda:** il guadagno di GPTQ a 8 bit è misurabile? Dove sta il vero collo di bottiglia su MCU?

---

## Riferimenti principali

| Lavoro | Perché leggerlo |
|---|---|
| Nagel et al. 2021, arXiv:2106.08295 | basi della quantizzazione, terminologia |
| Jacob et al. 2018, CVPR | aritmetica solo intera, il modello usato sui microcontrollori |
| LeCun et al. 1990 (OBD), Hassibi & Stork 1993 (OBS) | l'idea di usare l'Hessiana per comprimere |
| Frantar & Alistarh 2022, arXiv:2208.11580 (OBC/OBQ) | OBS applicato alla quantizzazione layer-wise |
| Frantar et al. 2023, arXiv:2210.17323 (GPTQ) | l'algoritmo che implementi |
| Trefethen & Bau 1997, lezione 23 | Cholesky, il nucleo numerico di GPTQ |
| Nagel et al. 2020, *Up or Down? Adaptive Rounding* (AdaRound) | alternativa basata su ottimizzazione, per il confronto finale |
