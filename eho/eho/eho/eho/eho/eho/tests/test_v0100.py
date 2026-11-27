"""Pruebas de la v0.10: memoria bayesiana, diseño de mediciones, ganancia medida y prerregistro."""
import copy

import pytest
import torch

from eho.forrajeo import ReglaParada
from eho.memoria_bayes import MemoriaBayesiana, choose_measurements

CFG = {"nombre": "t", "dominio": {"nombre": "edp1d", "regimenes": ["difusivo", "advectivo"], "episodios_por_regimen": 10,
                                  "muestras_por_episodio": 8, "episodios_validacion": 4, "episodios_calibracion": 40},
       "modelo": {"d": 24, "n_slices": 4}, "colmena": {"n_agentes": 2},
       "entrenamiento": {"epocas": 3, "dispositivo": "cpu"},
       "activo": {"presupuesto": 4, "epocas_ajuste": 1, "repeticion": 4, "ganancia": "medida"}}


@pytest.fixture(scope="module")
def gemelo():
    from eho import Gemelo
    g = Gemelo.desde_config(CFG).entrenar(verbose=False)
    g.ajustar_memoria_bayesiana(r=3, pasos=60)
    return g


def _mb(seed=0, d=6, D=2, r=3):
    torch.manual_seed(seed)
    mb = MemoriaBayesiana(d, D, r).double()
    return mb, mb.J(torch.randn(7, d, dtype=torch.float64)).detach(), torch.randn(7, D, dtype=torch.float64)


def test_posterior_uncertainty_never_increases():
    """Proposición: Σ_k ⪯ Σ_{k-1} (ninguna medición aumenta la incertidumbre posterior)."""
    mb, J, e = _mb()
    with torch.no_grad():
        prev = mb.posterior(J[:0], e[:0])[1]
        for k in range(1, 8):
            S = mb.posterior(J[:k], e[:k])[1]
            assert torch.linalg.eigvalsh(prev - S).min() > -1e-10
            prev = S


def test_kalman_equals_batch_and_is_order_invariant():
    mb, J, e = _mb(1)
    with torch.no_grad():
        mB, SB = mb.posterior(J, e)
        for orden in (range(7), reversed(range(7))):
            m, S = mb.posterior(J[:0], e[:0])
            for i in orden:
                m, S = mb.kalman_update(m, S, J[i], e[i])
            assert torch.allclose(m, mB, atol=1e-9) and torch.allclose(S, SB, atol=1e-9)
        assert torch.allclose(mb.posterior_after(mb.posterior(J[:3], e[:3])[1], J[3]), mb.posterior(J[:4], e[:4])[1], atol=1e-9)


def test_design_criteria_are_consistent():
    mb, J, e = _mb(2)
    with torch.no_grad():
        S0 = torch.eye(3, dtype=torch.float64)
        assert all(float(mb.eig(S0, J[i])) >= -1e-12 and float(mb.delta_G(S0, J[i], J[5:])) >= -1e-12 for i in range(5))
        var = lambda S: float(torch.einsum("tdr,rs,tds->", J[5:], S, J[5:]))
        # Δ_G es exactamente la reducción de varianza del objetivo
        assert abs(float(mb.delta_G(S0, J[0], J[5:])) - (var(S0) - var(mb.posterior_after(S0, J[0])))) < 1e-9
        idx, S = choose_measurements(mb, J[:5], J[5:], n=2, criterion="objetivo")
        mejor = min(var(mb.posterior_after(S0, J[i])) for i in range(5))
        assert abs(var(mb.posterior_after(S0, J[idx[0]])) - mejor) < 1e-12          # el primer paso es el óptimo
        caro = torch.tensor([1.0] * 5); caro[idx[0]] = 1e6                          # si esa medición cuesta mucho, no se elige
        assert choose_measurements(mb, J[:5], J[5:], n=1, cost=caro)[0][0] != idx[0]
        with pytest.raises(ValueError):
            choose_measurements(mb, J[:5], None, n=1, criterion="objetivo")


def test_gemelo_bayes_prediction_and_design(gemelo):
    ep = gemelo.dominio_variante(0).generate(1, 12, "mixto", seed=5)
    obj, pool = {k: v[:4] for k, v in ep.items()}, {k: v[4:] for k, v in ep.items()}
    r0 = gemelo.predecir_bayes(obj)
    assert torch.allclose(r0["escalares"], r0["base"]) and abs(r0["incertidumbre_latente"] - 3.0) < 1e-6   # prior: traza = r
    r2 = gemelo.predecir_bayes(obj, {k: v[:2] for k, v in pool.items()})
    assert r2["incertidumbre_latente"] < r0["incertidumbre_latente"] and (r2["desviacion"] <= r0["desviacion"] + 1e-6).all()
    d_obj = gemelo.disenar_mediciones(pool, obj, n=2, criterio="objetivo")
    d_azar = gemelo.disenar_mediciones(pool, obj, n=2, criterio="aleatorio")
    assert len(set(d_obj["indices"])) == 2 and d_obj["varianza_objetivo"] <= d_azar["varianza_objetivo"] + 1e-6
    g = copy.deepcopy(gemelo)
    with torch.no_grad():
        next(g.agentes[0].parameters()).add_(1e-3)
    with pytest.raises(RuntimeError):
        g.predecir_bayes(obj)                                                       # agentes distintos: reajustar


def test_measured_gain_rule():
    r = ReglaParada(tau=4, rendicion=2, n_min=4)
    r.registrar_medida(0.20, 4, riesgo_antes=1.0); assert r.continuar()            # aprende
    r.registrar_medida(0.10, 4, riesgo_antes=0.8); assert r.continuar()
    r.registrar_medida(0.001, 4, riesgo_antes=0.7); assert r.continuar()           # 1.ª ronda sin ganancia
    r.registrar_medida(-0.02, 4, riesgo_antes=0.7)                                 # olvido: tampoco es útil
    assert not r.continuar() and abs(r.tasa_media - 0.281 / 20) < 1e-9
    with pytest.raises(ValueError):
        ReglaParada().registrar_medida(float("nan"), 4)


def test_gemelo_measured_gain_flow(gemelo):
    g = copy.deepcopy(gemelo)
    cand = g.dominio_variante(0).generate(4, 8, "mixto", seed=3)
    n_tr0 = sum(d["episode"].unique().numel() for d in (getattr(g, "_acumulado", None) or [[]])[0])
    g.incorporar(cand)
    h = g.historial[-1]
    assert h["ganancia_medida"] is not None and h["riesgo_referencia_antes"] > 0
    assert len(g._referencia) == 1 and g._referencia[0]["episode"].unique().numel() == 1   # 1 de 4 apartado
    assert sum(d["episode"].unique().numel() for d in g._acumulado[0]) - n_tr0 == 3        # se aprende de 3
    assert g.parada.n == 4 and g._simulaciones_usadas == 4                                 # pero se pagaron 4


def test_preregistration_and_absolute_tolerance():
    from eho.preregistro import cargar, para_dominio, exigir_confirmado
    from eho import Gemelo
    p = cargar()
    assert len(p["huella"]) == 64 and p["semillas"] == list(range(10))
    pre = para_dominio("edp1d")
    assert pre["tolerancia_absoluta"] == [0.25, 0.25] and 0 < pre["riesgo"] < 1
    if not p["confirmado"]:
        with pytest.raises(RuntimeError):
            exigir_confirmado()
    cfg = copy.deepcopy(CFG)
    cfg["colmena"] = {"n_agentes": 2, "calibracion": "ltt", "riesgo": 0.3, "delta": 0.2,
                      "tolerancia_absoluta": pre["tolerancia_absoluta"]}
    cfg["activo"]["ganancia"] = "sorpresa"
    g = Gemelo.desde_config(cfg).entrenar(verbose=False)
    assert torch.allclose(g.predictor.fail_scale, torch.tensor([0.25, 0.25]))
    assert g.protocolo["certificado"]["tolerancia_absoluta"] and g.protocolo["certificado"]["tolerancia"] == [0.25, 0.25]
    ds = g.dominio_variante(0).generate(4, 8, "difusivo", seed=11)
    assert 0.0 <= g.evaluar_colectivo(ds)["cobertura"] <= 1.0
