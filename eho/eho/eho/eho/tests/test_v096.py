"""Pruebas de la v0.9.6: errores que persistían tras la tercera auditoría."""
import copy
import sys

import pytest
import torch

from eho.centinela import Centinela
from eho.network import MicrovascularNetwork

CFG = {"nombre": "t", "dominio": {"nombre": "edp1d", "regimenes": ["difusivo"], "episodios_por_regimen": 8,
                                  "muestras_por_episodio": 6, "episodios_validacion": 4, "episodios_calibracion": 30},
       "modelo": {"d": 24, "n_slices": 4},
       "colmena": {"n_agentes": 2, "calibracion": "ltt", "riesgo": 0.3, "delta": 0.2, "protocolos": [0, 2]},
       "entrenamiento": {"epocas": 2, "dispositivo": "cpu"},
       "activo": {"presupuesto": 2, "epocas_ajuste": 1, "repeticion": 4, "presupuesto_total": 8}}


@pytest.fixture(scope="module")
def gemelo():
    from eho import Gemelo
    return Gemelo.desde_config(CFG).entrenar(verbose=False)


def test_safe_deployment_roundtrip(gemelo, tmp_path):
    from eho import Gemelo
    ep = gemelo.dominio_variante(0).generate(1, 6, "difusivo", seed=31)
    q, c = {k: v[2:] for k, v in ep.items()}, {k: v[:2] for k, v in ep.items()}
    r = gemelo.predecir(q, contexto=c)
    gemelo.guardar(tmp_path / "s.eho")
    torch.load(tmp_path / "s.eho", weights_only=True)                    # legible sin ejecutar código
    h = Gemelo.cargar(tmp_path / "s.eho", permitir_pickle=False)
    r2 = h.predecir(q, contexto=c)
    assert torch.equal(r["escalares"], r2["escalares"]) and torch.equal(r["campos"], r2["campos"])
    assert torch.equal(r["aceptado"], r2["aceptado"]) and h.protocolo == gemelo.protocolo
    with pytest.raises(RuntimeError):
        h.proponer(ep)                                                   # despliegue: solo inferencia
    with pytest.raises(RuntimeError):
        h.incorporar(ep)
    gemelo.guardar(tmp_path / "p.eho", modo="reanudar")
    with pytest.raises(RuntimeError):
        Gemelo.cargar(tmp_path / "p.eho", permitir_pickle=False)         # un pickle no entra por la vía segura
    g2 = copy.deepcopy(gemelo)
    for a in g2.agentes:
        a.compact()
    g2.guardar(tmp_path / "c.eho")
    assert torch.equal(g2.predecir(q, contexto=c)["escalares"],
                       Gemelo.cargar(tmp_path / "c.eho", permitir_pickle=False).predecir(q, contexto=c)["escalares"])


def test_certificate_per_context_protocol(gemelo):
    assert set(gemelo.predictores) == {0, 2} and gemelo.protocolo["protocolos"] == [0, 2]
    ep = gemelo.dominio_variante(0).generate(1, 6, "difusivo", seed=31)
    q = {k: v[3:] for k, v in ep.items()}
    assert gemelo.predecir(q)["protocolo_coincide"]                                          # k = 0 certificado
    assert gemelo.predecir(q, contexto={k: v[:2] for k, v in ep.items()})["protocolo_coincide"]
    r1 = gemelo.predecir(q, contexto={k: v[:1] for k, v in ep.items()})                      # k = 1 no calibrado
    assert not r1["protocolo_coincide"] and not r1["aceptado"].any()


def test_unique_simulation_ids_and_manifest(gemelo):
    g = copy.deepcopy(gemelo)
    cand = g.dominio_variante(0).generate(2, 6, "difusivo", seed=3)
    g.incorporar(cand, costo=3.0, manifiesto={"version_simulador": "x1", "semilla": 3})
    assert len(g.registro) == 2 and g.registro[0]["version_simulador"] == "x1" and g.registro[0]["costo"] == 1.5
    assert "segundos_ajuste" in g.historial[-1]
    usadas = g._simulaciones_usadas
    with pytest.raises(ValueError):
        g.incorporar(cand)                                               # la misma simulación dos veces
    assert g._simulaciones_usadas == usadas and len(g.registro) == 2     # y no se contó


def test_disconnected_network_rejected():
    with pytest.raises(ValueError):
        MicrovascularNetwork(5, [(0, 1), (1, 2), (3, 4)], [100.0] * 3, [8.0] * 3, {0: 10.0, 2: 0.0}, {0: 0.45})
    MicrovascularNetwork(3, [(0, 1), (1, 2)], [100.0] * 2, [8.0] * 2, {0: 10.0, 2: 0.0}, {0: 0.45})


def test_energy_trace_reports_discrete_violations(gemelo):
    ep = gemelo.dominio_variante(0).generate(2, 6, "difusivo", seed=5)
    t = gemelo.agentes[0].energy_trace(ep)
    assert t["energias"].shape == (4, 12) and 0.0 <= t["fraccion_violaciones"] <= 1.0   # 3 pasos + inicial, 12 muestras


def test_cusum_baseline():
    c = Centinela([], torch.ones(1))
    c.cal = torch.randn(500).abs().clamp(min=1e-3).sort().values
    torch.manual_seed(0)
    normal = c.cusum(torch.randn(30).abs().clamp(min=1e-3))
    cambio = c.cusum(8 * torch.randn(30).abs().clamp(min=1e-3))
    assert cambio > normal + 5


def test_non_inferiority():
    sys.path.insert(0, "analisis")
    from estadistica import no_inferioridad
    ref = [0.50, 0.52, 0.49, 0.51, 0.50, 0.53, 0.48, 0.50]
    assert no_inferioridad([x + 0.005 for x in ref], ref, margen=0.02)["no_inferior"] is True
    assert no_inferioridad([x + 0.05 for x in ref], ref, margen=0.02)["no_inferior"] is False
    assert no_inferioridad(ref[:3], ref[:3], margen=0.02)["no_inferior"] is None       # muy pocas semillas


def test_informed_ltt_grid_is_valid_and_more_powerful():
    """La rejilla informada por datos de referencia mantiene el control del riesgo y certifica más."""
    from eho.colony import ltt_grid, ltt_threshold
    g = torch.Generator().manual_seed(0)
    viol, cert_inf, cert_ciega = 0, 0, 0
    for _ in range(150):
        ref, s = torch.rand(200, generator=g), torch.rand(90, generator=g)
        pf = lambda x: 0.55 * (1 - x) ** 1.5                           # ~22 % de fallos en total
        fref, fail = torch.rand(200, generator=g) < pf(ref), torch.rand(90, generator=g) < pf(s)
        thr, info = ltt_threshold(s, fail, 0.2, 0.1, grid=ltt_grid(ref, 90, 0.2, 0.1, reference_fail=fref))
        cert_inf += int(info["certificado"])
        cert_ciega += int(ltt_threshold(s, fail, 0.2, 0.1, grid=ltt_grid(ref, 90, 0.2, 0.1))[1]["certificado"])
        s2 = torch.rand(20000, generator=g)
        acc = s2 >= thr
        viol += int(acc.any() and (torch.rand(20000, generator=g) < pf(s2))[acc].float().mean() > 0.2)
    assert viol / 150 <= 0.1 + 0.04
    assert cert_inf > 0 and cert_inf >= cert_ciega
