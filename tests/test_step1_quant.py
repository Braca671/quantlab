"""Passo 1 — quantizzazione uniforme. Esegui: pytest tests/test_step1_quant.py"""
import pytest
import torch

from quantlab.quant import compute_qparams, quantize, dequantize, fake_quant


@pytest.fixture
def W():
    torch.manual_seed(0)
    return torch.randn(16, 64) * torch.logspace(-2, 0, 16)[:, None]   # righe con scale diverse


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_shapes_and_integer_zero(W, bits):
    s, z = compute_qparams(W, bits, per_channel=True)
    assert s.shape == (16, 1) and z.shape == (16, 1)
    assert torch.all(s > 0)
    assert torch.equal(z, z.round()), "lo zero-point deve essere intero"
    assert torch.all((z >= 0) & (z <= 2**bits - 1))
    s, z = compute_qparams(W, bits, per_channel=False)
    assert s.shape == (1, 1) and z.shape == (1, 1)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_levels_are_valid_integers(W, bits):
    s, z = compute_qparams(W, bits)
    q = quantize(W, s, z, bits)
    assert torch.equal(q, q.round())
    assert q.min() >= 0 and q.max() <= 2**bits - 1


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_rounding_error_at_most_half_step(W, bits):
    s, z = compute_qparams(W, bits)
    err = (fake_quant(W, s, z, bits) - W).abs()
    assert torch.all(err <= s / 2 + 1e-6), "dentro il range l'errore è al massimo mezzo passo"


def test_zero_is_exactly_representable(W):
    for bits in [2, 3, 4, 8]:
        for pc in [True, False]:
            s, z = compute_qparams(W, bits, per_channel=pc)
            zeros = torch.zeros_like(W)
            assert torch.equal(fake_quant(zeros, s, z, bits), zeros)


def test_fake_quant_is_idempotent(W):
    s, z = compute_qparams(W, 4)
    once = fake_quant(W, s, z, 4)
    assert torch.allclose(fake_quant(once, s, z, 4), once)


def test_dequantize_inverts_quantize_on_grid(W):
    s, z = compute_qparams(W, 4)
    q = quantize(W, s, z, 4)
    assert torch.equal(quantize(dequantize(q, s, z), s, z, 4), q)


def test_all_zero_row_is_safe():
    W = torch.randn(4, 8); W[2] = 0
    s, z = compute_qparams(W, 4)
    assert torch.isfinite(s).all() and torch.isfinite(z).all() and torch.all(s > 0)
    assert torch.equal(fake_quant(W, s, z, 4)[2], torch.zeros(8))


def test_per_channel_beats_per_tensor(W):
    """Con righe a scale molto diverse, una griglia per riga è molto più precisa."""
    errs = {}
    for pc in [True, False]:
        s, z = compute_qparams(W, 4, per_channel=pc)
        errs[pc] = (fake_quant(W, s, z, 4) - W).pow(2).sum()
    assert errs[True] < 0.5 * errs[False]
