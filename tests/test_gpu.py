"""Pruebas que solo corren si hay GPU (se omiten en CPU)."""
import pytest
import torch

from eho.data.synthetic import make_dataset
from eho.model import EHO
from eho.train import fit, evaluate, calibrate_memory

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sin GPU")


@cuda
def test_entrenamiento_completo_en_gpu():
    torch.manual_seed(0)
    tr, va = make_dataset(96, "sano", n_points=64, seed=1), make_dataset(32, "dt2", n_points=64, seed=2)
    m = EHO().to("cuda:0")
    log = fit(m, tr, va, epochs=6, eco_warmup=2, invade_every=2, verbose=False)
    calibrate_memory(m, make_dataset(32, "sano", n_points=64, seed=3), va)
    ev = evaluate(m, va)
    assert all(torch.isfinite(torch.tensor(v)) for v in ev.values())
    assert m.moe.n.device.type == "cuda" and m.memory.keys.device.type == "cuda"


@cuda
def test_garantias_en_gpu():
    from eho.thermo import DissipationPotential
    dp = DissipationPotential(8).cuda()
    g = torch.logspace(-1, 3, 64, device="cuda")
    assert (dp.relative_viscosity(g, torch.randn(64, 8, device="cuda")) >= 1).all()
