"""Passo 4 — OBS a ordine fisso. Esegui: pytest tests/test_step4_obs.py"""
import torch

from quantlab.hessian import reconstruction_error, prepare_hessian
from quantlab.obs import obs_quantize
from quantlab.quant import compute_qparams, fake_quant


def correlated_problem(out=8, d=24, n=400, seed=0):
    """Ingressi fortemente correlati: è il caso in cui compensare l'errore paga."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    X = A @ torch.randn(d, n, generator=g, dtype=torch.float64)
    return torch.randn(out, d, generator=g, dtype=torch.float64), 2 / n * X @ X.T


def rtn(W, bits):
    s, z = compute_qparams(W, bits)
    return fake_quant(W, s, z, bits)


def test_output_lies_on_the_grid_of_the_original_weights():
    W, H = correlated_problem()
    Q = obs_quantize(W, H, bits=3)
    _, Wc = prepare_hessian(H, W, 0.01)
    s, z = compute_qparams(Wc, 3)
    assert torch.allclose(fake_quant(Q, s, z, 3), Q), "ogni peso deve stare sulla griglia fissata all'inizio"


def test_obs_beats_rtn_on_reconstruction_error():
    for bits in [2, 3, 4]:
        ratios = []
        for seed in range(3):
            W, H = correlated_problem(seed=10 * bits + seed)
            e_obs = reconstruction_error(W, obs_quantize(W, H, bits), H)
            e_rtn = reconstruction_error(W, rtn(W, bits), H)
            ratios.append(e_obs / e_rtn)
        mean = sum(ratios) / len(ratios)
        assert mean < 0.8, f"{bits} bit: errore OBS/RTN medio = {mean:.2f}, atteso < 0.8"


def test_first_column_equals_rtn():
    """La prima colonna non riceve compensazioni: deve coincidere con RTN."""
    W, H = correlated_problem()
    Q = obs_quantize(W, H, bits=3)
    _, Wc = prepare_hessian(H, W, 0.01)
    assert torch.allclose(Q[:, 0], rtn(Wc, 3)[:, 0])


def test_diagonal_hessian_reduces_to_rtn():
    """Con ingressi scorrelati non c'è nulla da compensare: OBS = RTN."""
    W = torch.randn(6, 10, dtype=torch.float64)
    H = torch.diag(torch.rand(10, dtype=torch.float64) + 0.5)
    assert torch.allclose(obs_quantize(W, H, 3), rtn(W, 3))
