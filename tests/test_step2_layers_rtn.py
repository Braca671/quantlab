"""Passo 2 — layer come matrici e baseline RTN. Esegui: pytest tests/test_step2_layers_rtn.py"""
import pytest
import torch
from torch import nn

from quantlab.layers import weight_as_matrix, set_weight_from_matrix, inputs_as_matrix
from quantlab.models import SmallCNN, quantizable_layers
from quantlab.rtn import rtn_quantize_layer_, rtn_quantize_model_

LAYERS = [
    (nn.Linear(20, 7), (5, 20)),
    (nn.Linear(20, 7), (2, 3, 20)),                                   # dimensioni extra di batch
    (nn.Conv2d(3, 8, 3, padding=1), (2, 3, 10, 10)),
    (nn.Conv2d(4, 6, (3, 5), stride=2, padding=(1, 2), dilation=1), (2, 4, 11, 9)),
    (nn.Conv2d(4, 6, 3, dilation=2, bias=False), (1, 4, 12, 12)),
]


@pytest.mark.parametrize("layer,shape", LAYERS)
def test_layer_is_a_matrix_product(layer, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    W, X = weight_as_matrix(layer), inputs_as_matrix(layer, x)
    out = layer(x)
    bias = layer.bias if layer.bias is not None else torch.zeros(W.shape[0])
    if isinstance(layer, nn.Linear):
        expected = out.reshape(-1, W.shape[0]).T - bias[:, None]
    else:
        expected = out.transpose(0, 1).reshape(W.shape[0], -1) - bias[:, None]
    assert W.shape[1] == X.shape[0]
    assert torch.allclose(W @ X, expected, atol=1e-5), "W @ X deve riprodurre l'uscita del layer (senza bias)"


@pytest.mark.parametrize("layer,shape", LAYERS)
def test_weight_roundtrip_and_copy(layer, shape):
    before = layer.weight.detach().clone()
    W = weight_as_matrix(layer)
    W.mul_(0)                                                          # deve essere una copia
    assert torch.equal(layer.weight, before)
    set_weight_from_matrix(layer, weight_as_matrix(layer) * 2)
    assert layer.weight.shape == before.shape
    assert torch.allclose(layer.weight, before * 2)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_rtn_layer_uses_at_most_2_to_bits_values_per_channel(bits):
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 16, 3)
    bias = conv.bias.detach().clone()
    rtn_quantize_layer_(conv, bits)
    W = weight_as_matrix(conv)
    for row in W:
        assert row.unique().numel() <= 2**bits
    assert torch.equal(conv.bias, bias), "il bias resta in float"


def test_rtn_model_quantizes_all_but_skipped():
    torch.manual_seed(0)
    model = SmallCNN()
    names = [n for n, _ in quantizable_layers(model)]
    last = names[-1]
    last_before = dict(model.named_modules())[last].weight.detach().clone()
    rtn_quantize_model_(model, bits=3, skip=(last,))
    for name, layer in quantizable_layers(model):
        if name == last:
            assert torch.equal(layer.weight, last_before)
        else:
            assert all(r.unique().numel() <= 8 for r in weight_as_matrix(layer))
