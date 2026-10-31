"""Passo 6 — modello intero. Esegui: pytest tests/test_step6_pipeline.py"""
import pytest
import torch

from quantlab.layers import weight_as_matrix
from quantlab.models import SmallCNN, quantizable_layers
from quantlab.pipeline import quantize_model_
from quantlab.rtn import rtn_quantize_model_


def fake_calibration(n=64, batch=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    return [(x[i:i + batch], torch.zeros(batch, dtype=torch.long)) for i in range(0, n, batch)]


@pytest.mark.parametrize("sequential", [True, False])
def test_all_layers_quantized(sequential):
    torch.manual_seed(0)
    model = SmallCNN()
    quantize_model_(model, fake_calibration(), bits=3, sequential=sequential)
    for _, layer in quantizable_layers(model):
        assert all(r.unique().numel() <= 8 for r in weight_as_matrix(layer))


def test_gptq_model_closer_to_float_than_rtn():
    torch.manual_seed(0)
    ref = SmallCNN().eval()
    calib = fake_calibration()
    x = torch.cat([b for b, _ in fake_calibration(n=128, seed=1)])
    gptq, rtn = SmallCNN().eval(), SmallCNN().eval()
    gptq.load_state_dict(ref.state_dict()); rtn.load_state_dict(ref.state_dict())
    quantize_model_(gptq, calib, bits=3)
    rtn_quantize_model_(rtn, bits=3)
    with torch.no_grad():
        y = ref(x)
        e_gptq, e_rtn = (gptq(x) - y).pow(2).mean(), (rtn(x) - y).pow(2).mean()
    assert e_gptq < e_rtn, f"GPTQ {e_gptq:.3g} vs RTN {e_rtn:.3g}"
