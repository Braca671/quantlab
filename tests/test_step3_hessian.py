"""Passo 3 — Hessiana del problema di ricostruzione. Esegui: pytest tests/test_step3_hessian.py"""
import torch

from quantlab.hessian import HessianAccumulator, reconstruction_error, prepare_hessian


def test_accumulator_equals_full_batch():
    torch.manual_seed(0)
    X = torch.randn(12, 300, dtype=torch.float64)
    acc = HessianAccumulator(12)
    for chunk in [X[:, :17], X[:, 17:150], X[:, 150:]]:          # batch di dimensioni diverse
        acc.add_batch(chunk)
    assert torch.allclose(acc.H, 2 / 300 * X @ X.T)


def test_accumulator_accepts_float32_and_is_symmetric_psd():
    torch.manual_seed(0)
    acc = HessianAccumulator(10)
    acc.add_batch(torch.randn(10, 50))
    H = acc.H
    assert torch.allclose(H, H.T)
    assert torch.linalg.eigvalsh(H.double()).min() > -1e-9


def test_reconstruction_error_matches_output_error():
    torch.manual_seed(0)
    W, Q = torch.randn(5, 12, dtype=torch.float64), torch.randn(5, 12, dtype=torch.float64)
    X = torch.randn(12, 40, dtype=torch.float64)
    H = 2 / 40 * X @ X.T
    expected = 2 / 40 * ((W - Q) @ X).pow(2).sum()
    assert abs(reconstruction_error(W, Q, H) - expected.item()) < 1e-9


def test_prepare_hessian_dead_columns_and_damping():
    torch.manual_seed(0)
    X = torch.randn(6, 30, dtype=torch.float64)
    X[2] = 0                                                       # ingresso sempre nullo
    H = 2 / 30 * X @ X.T
    W = torch.randn(4, 6, dtype=torch.float64)
    H0, W0 = H.clone(), W.clone()
    Hd, Wc = prepare_hessian(H, W, damp=0.1)
    assert torch.equal(H, H0) and torch.equal(W, W0), "deve restituire copie"
    assert torch.all(Wc[:, 2] == 0) and torch.equal(Wc[:, [0, 1, 3, 4, 5]], W[:, [0, 1, 3, 4, 5]])
    diag = H.diag().clone(); diag[2] = 1
    expected_diag = diag + 0.1 * diag.mean()
    assert torch.allclose(Hd.diag(), expected_diag)
    torch.linalg.cholesky(Hd)                                      # deve essere definita positiva
