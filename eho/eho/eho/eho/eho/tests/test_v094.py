"""Pruebas de regresión de la segunda auditoría externa (v0.9.4)."""
import copy

import pytest
import torch

from eho.colony import BeeColony, ColonyPredictor, colony_predict, ltt_threshold, ltt_grid
from eho.centinela import Centinela
from eho.dominios import assert_disjoint, concat
from eho.network import MicrovascularNetwork, PhaseSeparation
from eho.train_central import load_flexible

CFG = {"nombre": "t", "dominio": {"nombre": "edp1d", "regimenes": ["difusivo"], "episodios_por_regimen": 8,
                                  "muestras_por_episodio": 4, "episodios_validacion": 4, "episodios_calibracion": 30},
       "modelo": {"d": 24, "n_slices": 4}, "colmena": {"n_agentes": 2},
       "entrenamiento": {"epocas": 2, "dispositivo": "cpu"},
       "activo": {"presupuesto": 2, "epocas_ajuste": 1, "repeticion": 4, "presupuesto_total": 6}}


@pytest.fixture(scope="module")
def gemelo():
    from eho import Gemelo
    return Gemelo.desde_config(CFG).entrenar(verbose=False)


def test_calibrated_colony_selects_winner():
    # 3 agentes de acuerdo en 0 y uno discrepante en 10: la media es 2.5, el ganador es ~0
    P = torch.tensor([[[0.0]], [[0.1]], [[-0.1]], [[10.0]]])
    scale = torch.ones(1)
    media, _, _ = colony_predict(P, scale, BeeColony(quorum=1.01), mode="media")
    gan, acc, sup = colony_predict(P, scale, BeeColony(quorum=1.01), mode="ganador")
    assert abs(float(media) - 2.5) < 1e-6 and abs(float(gan)) < 0.2
    assert not bool(acc[0]) and float(sup) > 0                      # ganador y quórum son cosas distintas
    pr = ColonyPredictor(scale, mode="ganador").fit(P.expand(-1, 5, -1), torch.zeros(5, 1), target_coverage=0.8)
    assert abs(float(pr.predict(P)[0])) < 0.2


def test_uncertified_ltt_rejects_unseen_support():
    thr, info = ltt_threshold(torch.tensor([0.2, 0.3]), torch.tensor([True, True]), 0.1, 0.1,
                              grid=ltt_grid(torch.rand(50), 100, 0.1, 0.1))
    assert thr == float("inf") and not info["certificado"]
    pr = ColonyPredictor(torch.ones(1))
    pr.method, pr.ltt_info, pr.threshold = "ltt", info, 0.0          # aunque el umbral fuera 0
    _, ok, _ = pr.predict(torch.zeros(3, 4, 1))
    assert not ok.any()
    with pytest.raises(ValueError):
        ltt_threshold(torch.rand(10), torch.zeros(10, dtype=torch.bool))   # sin rejilla ya no hay "LTT"


def test_query_order_invariance():
    V = torch.tensor([[0.9, 0.2], [0.8, 0.3], [0.7, 0.4]])
    col = BeeColony(quorum=1.01, seed=3)
    a = col.decide(V).support.clone()
    col.decide(torch.rand(3, 4)); col.decide(torch.rand(3, 2))       # otras consultas en medio
    assert torch.equal(a, col.decide(V).support)


def test_stale_calibration_rejects(gemelo):
    g = copy.deepcopy(gemelo)
    q = g.dominio_variante(0).generate(2, 4, "difusivo", seed=9)
    g.calibracion_vigente = False
    assert not g.predecir(q)["aceptado"].any()
    assert g.evaluar_colectivo(q)["cobertura"] == 0.0


def test_evaluate_collective_prediction(gemelo):
    q = gemelo.dominio_variante(0).generate(4, 4, "difusivo", seed=11)
    r = gemelo.evaluar_colectivo(q)
    assert set(r) >= {"MAE_agentes", "MAE_ensemble", "MAE_colmena", "MAE_aceptados", "cobertura", "riesgo_selectivo"}
    assert r["MAE_ensemble"] <= r["MAE_agentes"] + 1e-9             # desigualdad triangular


def test_budget_all_modes(gemelo):
    for modo in ("mvt", "quorum"):
        g = copy.deepcopy(gemelo)
        g.cfg.activo.parada = modo
        cand = g.dominio_variante(0).generate(8, 4, "difusivo", seed=3)
        g._simulaciones_usadas = 5                                    # total = 6: queda 1
        assert len(g.proponer(cand, presupuesto=4)) <= 1
        with pytest.raises(ValueError):
            g.incorporar({k: v[cand["episode"] < 3] for k, v in cand.items()})
    g = copy.deepcopy(gemelo)
    g.cfg.activo.usar_error, g.parada = False, None
    with pytest.raises(ValueError):
        g._regla()                                                    # mvt sin error de inspección


def test_resume_training_equivalence(gemelo, tmp_path):
    from eho import Gemelo
    cand = gemelo.dominio_variante(0).generate(4, 4, "difusivo", seed=3)
    A = {k: v[cand["episode"] < 2] for k, v in cand.items()}
    B = {k: v[cand["episode"] >= 2] for k, v in cand.items()}
    q = gemelo.dominio_variante(0).generate(2, 4, "difusivo", seed=21)
    g1 = copy.deepcopy(gemelo); g1.incorporar(A); g1.incorporar(B)
    g2 = copy.deepcopy(gemelo); g2.incorporar(A)
    g2.guardar(tmp_path / "c.eho", modo="reanudar")
    g2 = Gemelo.cargar(tmp_path / "c.eho", permitir_pickle=True); g2.incorporar(B)
    assert torch.allclose(g1.predecir(q)["por_agente"], g2.predecir(q)["por_agente"], atol=1e-5)
    assert g1._ejecuciones == g2._ejecuciones == 4
    g3 = copy.deepcopy(gemelo); g3.guardar(tmp_path / "d.eho")        # despliegue
    with pytest.raises(RuntimeError):
        Gemelo.cargar(tmp_path / "d.eho").incorporar(A)


def test_config_architecture_respected(gemelo):
    ds = [a.hive_info["d"] for a in gemelo.agentes]
    assert all(16 <= d <= 32 for d in ds)                              # base 24 ± 8, no los 40-56 por defecto


def test_frozen_ecology(gemelo):
    g = copy.deepcopy(gemelo)
    g.cfg.activo.ecologia_en_ajuste = False
    antes = [a.moe.n.clone() for a in g.agentes]
    cand = g.dominio_variante(0).generate(2, 4, "difusivo", seed=5)
    g.incorporar(cand)
    assert all(torch.equal(x, a.moe.n) for x, a in zip(antes, g.agentes))


def test_missing_checkpoint_parameter(gemelo):
    m = gemelo.agentes[0]
    st = {k: v for k, v in m.state_dict().items()}
    quitar = next(k for k in st if k.startswith("moe.") and st[k].dim() > 1)
    st.pop(quitar)
    with pytest.raises(RuntimeError):
        load_flexible(copy.deepcopy(m), st)


def test_conformal_fixed_score_and_protocol(gemelo):
    d0 = gemelo.dominio_variante(0)
    calib = concat(*[d0.generate(1, 4, "difusivo", 50_000 + i) for i in range(30)])
    cen = Centinela(gemelo.agentes, gemelo.escala, alpha=0.1).fit(calib, sondeo={"n": 3})
    assert cen.cal_ep.numel() == 20                                    # 10 pacientes fijan s0, 20 calibran
    nuevo = concat(*[d0.generate(1, 4, "difusivo", 60_000 + i) for i in range(6)])
    p, _ = cen.patient_pvalues(nuevo)
    assert ((p > 0) & (p <= 1)).all()
    g = copy.deepcopy(gemelo)                                          # agentes distintos: calibración caducada
    cen.agents = g.agentes
    with torch.no_grad():
        next(g.agentes[0].parameters()).add_(0.01)
    with pytest.raises(RuntimeError):
        cen.patient_pvalues(nuevo)


def test_seed_dataset_overlap(gemelo):
    d0 = gemelo.dominio_variante(0)
    a = d0.generate(4, 4, "difusivo", seed=1)
    with pytest.raises(ValueError):
        assert_disjoint(a, d0.generate(4, 4, "difusivo", seed=1))
    assert_disjoint(a, d0.generate(4, 4, "difusivo", seed=2))
    assert_disjoint(gemelo.validacion, gemelo.evolucion, "validación y evolución")


def test_network_residuals_and_validation():
    net = MicrovascularNetwork(4, [(0, 1), (1, 2), (1, 3)], [500.0] * 3, [10.0, 8.0, 3.0], {0: 30.0, 2: 0.0, 3: 0.0}, {0: 0.6})
    r = net.solve(lambda D, H: 1.0 + 2.5 * H, PhaseSeparation())
    assert r["valida"] and r["residuo_constitutivo"] < 1e-5 and r["identidad_rbc"] < 1e-10
    assert not net.solve(lambda D, H: 1.0 + 2.5 * H, PhaseSeparation(), iters=1)["valida"]
    with pytest.raises(ValueError):
        MicrovascularNetwork(2, [(0, 1)], [100.0], [-5.0], {0: 1.0, 1: 0.0}, {0: 0.4})
    with pytest.raises(ValueError):
        MicrovascularNetwork(2, [(0, 1)], [100.0], [5.0], {0: 1.0, 1: 0.0}, {0: 0.99})


def test_network_phase_gradient():
    """Con la restricción de hematocrito activa, el gradiente por diferenciación automática coincide
    con diferencias finitas (los límites ya no se desprenden del grafo)."""
    class Extrema(torch.nn.Module):
        def forward(self, FQ, Da, Db, Df, Hf, theta=None):
            return torch.full_like(FQ, 0.001)

    def run(D):
        net = MicrovascularNetwork(4, [(0, 1), (1, 2), (1, 3)], [500.0] * 3, [10.0, 8.0, 3.0],
                                   {0: 30.0, 2: 0.0, 3: 0.0}, {0: 0.6})
        net.D = D
        return net.solve(lambda D_, H: 1.0 + 2.5 * H, Extrema(), tol=1e-12, iters=300)["H"]

    D = torch.tensor([10.0, 8.0, 3.0], dtype=torch.float64, requires_grad=True)
    H = run(D)
    assert abs(float(H[2]) - 0.95) < 1e-6                                # la restricción está activa
    g, = torch.autograd.grad(H[1], D)
    for i in (1, 2):
        e = torch.zeros(3, dtype=torch.float64); e[i] = 1e-4
        fd = float((run((D + e).detach())[1] - run((D - e).detach())[1]) / 2e-4)
        assert abs(float(g[i]) - fd) < 1e-6
