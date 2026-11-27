"""Pruebas de la v0.9: eficiencia (compactar, LOO en un paso, colmena que converge) y regla de parada calibrada."""
import time

import torch

from eho.colony import BeeColony
from eho.forrajeo import ReglaParada
from eho.centinela import loo_predictions
from eho.dominios import AdveccionDifusion1D, concat
from eho.central import EHOCentral
from eho.train_central import fit_central


def _modelo(mem="completa"):
    dom = AdveccionDifusion1D()
    tr = concat(*[dom.generate(6, 6, r, i) for i, r in enumerate(["difusivo", "advectivo"])])
    torch.manual_seed(0)
    m = EHOCentral(dom, memory=mem, d=24, n_slices=4, d_val=dom.memory_values({k: v[:2] for k, v in tr.items()}).shape[-1])
    fit_central(m, tr, tr, epochs=4, eco_warmup=1)
    return m.eval(), tr


def test_compactar_no_cambia_predicciones():
    m, tr = _modelo()
    with torch.no_grad():
        a = m(tr)
        antes = sum(p.numel() for p in m.parameters())
        ahorro = m.compact()
        b = m(tr)
    assert ahorro > 0 and sum(p.numel() for p in m.parameters()) == antes - ahorro
    assert torch.allclose(a["scalars"], b["scalars"], atol=1e-5) and torch.allclose(a["fields"], b["fields"], atol=1e-5)
    assert m.compact() == 0


def test_loo_en_un_paso_coincide_con_el_lento():
    m, tr = _modelo("episodica")
    ds = {k: v[:12] for k, v in tr.items()}                      # 2 episodios de 6
    lento, _ = loo_predictions([m], ds, k=5, seed=0)             # k = todas las demás (5)
    rapido, _ = loo_predictions([m], ds, k=None)
    assert torch.allclose(lento, rapido, atol=1e-4)
    idx = torch.tensor([0, 2, 4, 6, 7])
    p, _ = loo_predictions([m], ds, k=None, idx=idx)
    assert torch.isnan(p[0, 1]).all() and not torch.isnan(p[0, idx]).any()


def test_colmena_en_abstencion_termina_pronto():
    V = torch.full((4, 5), 0.02)
    t0 = time.perf_counter()
    d = BeeColony(quorum=0.6, t_max=60.0, seed=0).decide(V)
    assert d.choice is None and d.time < 30.0                     # antes agotaba t_max
    fuerte = BeeColony(quorum=0.6, seed=0).decide(torch.tensor([[0.95, 0.1]] * 4))
    assert fuerte.choice == 0


def test_regla_parada_con_referencia_calibrada():
    r = ReglaParada(tau=4, rendicion=2, n_min=4, referencia=2.0)
    r.registrar([6.0, 5.0, 4.0, 5.0])                              # casos nuevos: enseñan
    assert r.continuar()
    r.registrar([1.3, 1.6, 0.9, 1.8]); assert r.continuar()        # 1.ª ronda sin nada útil: aún no se rinde
    r.registrar([1.4, 1.2, 1.9, 1.0])                              # conocidos difíciles ya no cuentan
    assert not r.continuar()


def test_alarma_por_paciente_respeta_alfa():
    """p-valor conformal entre pacientes: la tasa de falsas alarmas no supera α (más holgura muestral)."""
    from eho.centinela import Centinela
    torch.manual_seed(0)
    c = Centinela([], torch.ones(1), alpha=0.05)
    falsas, n = 0, 0
    for _ in range(200):
        c.cal_ep = torch.randn(59).sort().values             # 59 pacientes de calibración
        t = torch.randn(20)                                   # 20 pacientes conocidos nuevos
        ge = 59 - torch.searchsorted(c.cal_ep, t, right=False)
        falsas += int((((ge.float() + 1) / 60) <= c.alpha).sum()); n += 20
    assert falsas / n <= 0.05 + 0.01
