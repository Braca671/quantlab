"""Passo 5 — GPTQ veloce. Esegui: pytest tests/test_step5_gptq.py"""
import time

import pytest
import torch

from quantlab.gptq import gptq_quantize
from quantlab.obs import obs_quantize
from tests.test_step4_obs import correlated_problem


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_gptq_equals_obs(bits):
    W, H = correlated_problem(out=10, d=40, seed=bits)
    assert torch.allclose(gptq_quantize(W, H, bits), obs_quantize(W, H, bits), atol=1e-6)


@pytest.mark.parametrize("block_size", [1, 3, 16, 40, 128])
def test_block_size_does_not_change_result(block_size):
    W, H = correlated_problem(out=10, d=40)
    ref = gptq_quantize(W, H, 3, block_size=40)
    assert torch.allclose(gptq_quantize(W, H, 3, block_size=block_size), ref, atol=1e-8)


def test_dead_columns_and_float32():
    W, H = correlated_problem(out=10, d=40)
    H[5, :] = 0; H[:, 5] = 0
    Q = gptq_quantize(W.float(), H.float(), 4)
    assert Q.dtype == torch.float32 and torch.isfinite(Q).all()
    assert torch.all(Q[:, 5] == 0)


def _best_time(fn, repeats=2):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def test_gptq_is_much_faster_than_obs():
    W, H = correlated_problem(out=64, d=512, n=2000)
    t_obs = _best_time(lambda: obs_quantize(W, H, 4))
    t_gptq = _best_time(lambda: gptq_quantize(W, H, 4))
    assert t_gptq < t_obs / 2, f"GPTQ {t_gptq:.3f}s vs OBS {t_obs:.3f}s"
