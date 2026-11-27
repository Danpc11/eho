"""Pruebas de regresión de la auditoría externa (v0.9.3)."""
import pytest
import torch

from eho.forrajeo import ReglaParada
from eho.colony import ColonyPredictor, ltt_threshold, ltt_grid
from eho.network import MicrovascularNetwork, PhaseSeparation
from eho.train_central import load_flexible, fit_central
from eho.dominios import AdveccionDifusion1D, concat
from eho.central import EHOCentral

CFG = {"nombre": "t", "dominio": {"nombre": "edp1d", "regimenes": ["difusivo"], "episodios_por_regimen": 8,
                                  "muestras_por_episodio": 4, "episodios_validacion": 4, "episodios_calibracion": 30},
       "modelo": {"d": 24, "n_slices": 4}, "colmena": {"n_agentes": 2},
       "entrenamiento": {"epocas": 2, "dispositivo": "cpu"},
       "activo": {"presupuesto": 2, "epocas_ajuste": 1, "repeticion": 4, "presupuesto_total": 3}}


@pytest.fixture(scope="module")
def gemelo():
    from eho import Gemelo
    return Gemelo.desde_config(CFG).entrenar(verbose=False)


def test_stop_zero_gain():
    r = ReglaParada(referencia=1.0, n_min=4, rendicion=2)
    r.registrar([0.5, 0.5])
    r.registrar([0.5, 0.5])
    assert r.continuar() is False


def test_stop_batch_permutation():
    a, b = ReglaParada(beta=0.5), ReglaParada(beta=0.5)
    a.registrar([2.0, 2.0, 1.0, 1.0])
    b.registrar([1.0, 1.0, 2.0, 2.0])
    assert a.marginal == b.marginal and a.continuar() == b.continuar()


def test_stop_validates_inputs():
    with pytest.raises(ValueError):
        ReglaParada(referencia=0.0)
    with pytest.raises(ValueError):
        ReglaParada().registrar([float("nan")])


def test_zero_and_total_budget(gemelo):
    import copy
    g = copy.deepcopy(gemelo)
    cand = g.dominio_variante(0).generate(6, 4, "difusivo", seed=3)
    assert g.proponer(cand, presupuesto=0) == []
    with pytest.raises(ValueError):
        g.proponer(cand, presupuesto=-1)
    g._simulaciones_usadas = 2                         # presupuesto_total = 3: solo queda 1
    assert len(g.proponer(cand, presupuesto=2)) <= 1
    g._simulaciones_usadas = 3
    assert g.proponer(cand, presupuesto=2) == []


def test_resume_campaign_and_recalibrate(gemelo, tmp_path):
    import copy
    from eho import Gemelo
    g = copy.deepcopy(gemelo)
    cand = g.dominio_variante(0).generate(4, 4, "difusivo", seed=3)
    umbral0 = g.predictor.threshold
    viejo = g.predictor
    g.incorporar({k: v[cand["episode"] < 2] for k, v in cand.items()})
    assert g.predictor is not viejo and g.calibracion_vigente      # se recalibró con los agentes nuevos
    g.guardar(tmp_path / "m.eho")
    h = Gemelo.cargar(tmp_path / "m.eho")
    assert h._simulaciones_usadas == g._simulaciones_usadas == 2
    assert h.parada is not None and h.parada.ganancias == g.parada.ganancias
    assert h.parada.rondas_sin_util == g.parada.rondas_sin_util


def test_field_weights_follow_scalar_decision(gemelo):
    q = gemelo.dominio_variante(0).generate(2, 4, "difusivo", seed=9)
    r = gemelo.predecir(q)
    W = r["pesos_campos"]
    assert torch.allclose(W.sum(0), torch.ones(W.shape[1]), atol=1e-5)
    assert torch.allclose(W[:, ~r["aceptado"]], torch.full_like(W[:, ~r["aceptado"]], 1.0 / W.shape[0]))
    assert r["campos"].shape[0] == q["episode"].numel()


def test_ltt_requires_independent_calibration():
    P, Y = torch.randn(3, 40, 2), torch.randn(40, 2)
    with pytest.raises(ValueError):
        ColonyPredictor(torch.ones(2)).fit(P, Y, method="ltt")


def test_ltt_no_certified_threshold_abstains():
    s = torch.rand(100)
    thr, info = ltt_threshold(s, torch.ones(100, dtype=torch.bool), 0.1, 0.1, grid=ltt_grid(torch.rand(100), 100, 0.1, 0.1))
    assert not info["certificado"] and (s >= thr).sum() == 0
    assert ltt_grid(torch.rand(50), 10, 0.1, 0.1) == []             # muy pocos pacientes: nada certificable


def test_ltt_grid_controls_risk():
    g, viol = torch.Generator().manual_seed(0), 0
    for _ in range(100):
        ref, s = torch.rand(300, generator=g), torch.rand(400, generator=g)
        fail = torch.rand(400, generator=g) < 0.4 * (1 - s) ** 2
        thr, info = ltt_threshold(s, fail, 0.1, 0.1, grid=ltt_grid(ref, 400, 0.1, 0.1))
        s2 = torch.rand(20000, generator=g)
        f2 = torch.rand(20000, generator=g) < 0.4 * (1 - s2) ** 2
        acc = s2 >= thr
        viol += int(acc.any() and f2[acc].float().mean() > 0.1)
    assert viol / 100 <= 0.1 + 0.05 and info["certificado"]


def _red(q_ratio_diam=(8.0, 3.0), H_in=0.6):
    # bifurcación: 0 -> 1, 1 -> 2 (ancha), 1 -> 3 (estrecha)
    return MicrovascularNetwork(4, [(0, 1), (1, 2), (1, 3)], [500.0, 500.0, 500.0], [10.0, *q_ratio_diam],
                        {0: 30.0, 2: 0.0, 3: 0.0}, {0: H_in})


def test_rbc_flux_consistency_and_bound():
    class Extrema(torch.nn.Module):                      # manda casi todos los glóbulos a la rama estrecha
        def forward(self, FQ, Da, Db, Df, Hf, theta=None):
            return torch.full_like(FQ, 0.001)
    res = _red().solve(lambda D, H: 1.0 + 2.5 * H, Extrema())
    assert res["identidad_rbc"] < 1e-12                  # Q_rbc = H Q exactamente
    assert float(res["H"].max()) <= 0.95 + 1e-9          # sin recortar después: la fracción ya lo respeta
    assert res["rbc_residual"].abs().max() < 1e-9 * res["rbc_flux"].abs().max()


def test_zero_flow_and_convergence_flags():
    net = MicrovascularNetwork(4, [(0, 1), (1, 2), (1, 3)], [500.0] * 3, [10.0, 8.0, 8.0], {0: 0.0, 2: 0.0, 3: 0.0}, {0: 0.45})
    res = net.solve(lambda D, H: 1.0 + 2.5 * H, PhaseSeparation())       # sin gradiente de presión
    assert torch.isfinite(res["H"]).all() and torch.isfinite(res["rbc_flux"]).all()
    res = _red().solve(lambda D, H: 1.0 + 2.5 * H, PhaseSeparation(), iters=1)
    assert res["convergio"] is False and res["iteraciones"] == 1
    assert _red().solve(lambda D, H: 1.0 + 2.5 * H, PhaseSeparation())["convergio"] is True


def test_flexible_load_equivalence_and_strictness():
    dom = AdveccionDifusion1D()
    tr = concat(*[dom.generate(6, 6, r, i) for i, r in enumerate(["difusivo", "advectivo"])])
    torch.manual_seed(0)
    m = EHOCentral(dom, memory="ninguna", d=24, n_slices=4)
    fit_central(m, tr, tr, epochs=2, eco_warmup=1)
    m.eval()
    n = load_flexible(EHOCentral(dom, memory="ninguna", d=24, n_slices=4), m.state_dict()).eval()
    assert all(k.startswith(("novelty_map.", "memory.mpc.")) for k in n.claves_omitidas)
    with torch.no_grad():
        assert torch.allclose(m(tr)["scalars"], n(tr)["scalars"], atol=1e-5)   # misma predicción
    with pytest.raises(RuntimeError):                                          # otra arquitectura: no en silencio
        load_flexible(EHOCentral(dom, memory="ninguna", d=32, n_slices=4), m.state_dict())
