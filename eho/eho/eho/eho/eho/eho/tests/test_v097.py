"""Pruebas de regresión de la cuarta auditoría externa (v0.9.7)."""
import copy
import math
import sys

import pytest
import torch

from eho.centinela import Centinela
from eho.colony import ltt_grid
from eho.network import MicrovascularNetwork, PhaseSeparation

CFG = {"nombre": "t", "dominio": {"nombre": "edp1d", "regimenes": ["difusivo"], "episodios_por_regimen": 8,
                                  "muestras_por_episodio": 8, "episodios_validacion": 4, "episodios_calibracion": 40},
       "modelo": {"d": 24, "n_slices": 4},
       "colmena": {"n_agentes": 2, "calibracion": "ltt", "riesgo": 0.3, "delta": 0.2, "protocolos": [2, 5]},
       "entrenamiento": {"epocas": 2, "dispositivo": "cpu"},
       "activo": {"presupuesto": 2, "epocas_ajuste": 1, "repeticion": 4, "presupuesto_total": 8}}


@pytest.fixture(scope="module")
def gemelo():
    from eho import Gemelo
    return Gemelo.desde_config(CFG).entrenar(verbose=False)


def test_evaluar_uses_certificate_of_its_protocol(gemelo):
    ds = gemelo.dominio_variante(0).generate(4, 8, "difusivo", seed=11)
    gemelo.evaluar_colectivo(ds, k=5)
    with pytest.raises(ValueError):
        gemelo.evaluar_colectivo(ds, k=3)                                   # sin calibración para k = 3
    g = copy.deepcopy(gemelo)
    g.predictores[5].threshold, g.predictores[2].threshold = float("inf"), -float("inf")
    for p in g.predictores.values():
        p.ltt_info["certificado"] = True
    assert g.evaluar_colectivo(ds, k=5)["cobertura"] == 0.0 and g.evaluar_colectivo(ds, k=2)["cobertura"] == 1.0


def test_same_inspection_in_calibration_and_deployment(gemelo):
    """Con 5 mediciones, predecir() y la calibración calculan el MISMO error de inspección."""
    from eho.hive import predict_with_inspection, context_inspection
    from eho.dominios.base import split_episodes, subset
    ep = gemelo.dominio_variante(0).generate(1, 8, "difusivo", seed=31)
    P, Y, CE = predict_with_inspection(gemelo.agentes, ep, k=5)
    ctx, qry = split_episodes(ep, 5, 0)
    assert torch.allclose(CE[:, 0], context_inspection(gemelo.agentes, subset(ep, ctx)), atol=1e-6)
    r = gemelo.predecir(subset(ep, qry), contexto=subset(ep, ctx))
    assert torch.allclose(r["por_agente"], P, atol=1e-5) and r["protocolo_coincide"] and r["k_contexto_usado"] == 5
    pred, ok, apoyo, _ = gemelo._certificar(P, CE, 5)
    assert torch.allclose(r["apoyo"], apoyo, atol=1e-6) and torch.equal(r["aceptado"], ok)


def test_context_must_belong_to_query_patient(gemelo):
    d0 = gemelo.dominio_variante(0)
    a, b = d0.generate(1, 8, "difusivo", seed=31), d0.generate(1, 8, "difusivo", seed=32)
    b["episode"] = b["episode"] + 7
    with pytest.raises(ValueError):
        gemelo.predecir({k: v[2:] for k, v in a.items()}, contexto={k: v[:2] for k, v in b.items()})   # otro paciente
    with pytest.raises(ValueError):
        gemelo.predecir({k: v[:4] for k, v in a.items()}, contexto={k: v[:2] for k, v in a.items()})   # muestras repetidas


def test_certificate_is_bound_to_model_and_described(gemelo):
    c = gemelo.protocolo["certificado"]
    assert c["delta_por_protocolo"] == pytest.approx(0.1) and c["delta_conjunta"] == pytest.approx(0.2)   # δ repartida
    assert {"seleccion_contexto", "unidad_estadistica", "error", "tolerancia", "riesgo", "distribucion"} <= set(c)
    g = copy.deepcopy(gemelo)
    ep = g.dominio_variante(0).generate(1, 8, "difusivo", seed=31)
    q, ctx = {k: v[2:] for k, v in ep.items()}, {k: v[:2] for k, v in ep.items()}
    assert g.predecir(q, contexto=ctx)["modelo_coincide"]
    with torch.no_grad():
        next(g.agentes[0].parameters()).add_(1e-3)                          # el modelo ya no es el certificado
    r = g.predecir(q, contexto=ctx)
    assert not r["modelo_coincide"] and not r["aceptado"].any()


def test_audit_covers_incorporated_data_and_current_validation(gemelo):
    g = copy.deepcopy(gemelo)
    calib = g.datos_calibracion()
    robado = {k: v[calib["episode"] < 2] for k, v in calib.items()}        # 2 pacientes de calibración
    g.cfg.activo.recalibrar = False
    g.incorporar(robado)
    with pytest.raises(ValueError):
        g.auditar_particiones(calib)                                        # ya influyeron en el modelo
    with pytest.raises(ValueError):
        gemelo.auditar_particiones(calib, val=calib)                        # se audita la validación que se pasa


def test_manifest_cannot_override_and_costs_validated(gemelo):
    g = copy.deepcopy(gemelo)
    cand = g.dominio_variante(0).generate(4, 8, "difusivo", seed=3)
    A = {k: v[cand["episode"] < 2] for k, v in cand.items()}
    for malo in (-10.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            g.incorporar(A, costo=malo)
    g.cfg.activo.presupuesto_costo = 100.0
    with pytest.raises(ValueError):
        g.incorporar(A)                                                     # costo obligatorio
    g.incorporar(A, costo=4.0, manifiesto={"id": "otro", "ronda": 99}, ids=["mirheo-v1-s3-a", "mirheo-v1-s3-b"])
    r = g.registro[0]
    assert r["id"] == "mirheo-v1-s3-a" and r["ronda"] != 99 and r["metadatos"]["id"] == "otro" and len(r["huella"]) == 40
    with pytest.raises(ValueError):
        g.incorporar(A, costo=1.0, ids=["x", "y"])                          # mismos datos con otro id: la huella lo detecta


def test_reserve_before_running(gemelo):
    g = copy.deepcopy(gemelo)
    g.cfg.activo.presupuesto_costo = 10.0
    assert g.reservar(6, costo=8.0)["condiciones_reservadas"] == 6 and g.presupuesto_restante() == 2
    with pytest.raises(ValueError):
        g.reservar(3, costo=1.0)                                            # no caben 3 más
    with pytest.raises(ValueError):
        g.reservar(1, costo=5.0)                                            # ni ese costo
    cand = g.dominio_variante(0).generate(2, 8, "difusivo", seed=3)
    g.incorporar(cand, costo=3.0)
    assert g._reservado == 4 and g._simulaciones_usadas == 2
    g.liberar()
    assert g.presupuesto_restante() == 6


def test_shear_coupled_convergence_checks_flow():
    """Un solo vaso: H no cambia, pero el caudal sí hasta que viscosidad y cizalla se ponen de acuerdo."""
    net = MicrovascularNetwork(2, [(0, 1)], [500.0], [10.0], {0: 30.0, 1: 0.0}, {0: 0.4})
    visc = lambda D, H, g: 1.0 + 3.0 / (1.0 + 0.05 * g)
    r = net.solve(visc, PhaseSeparation(), shear_coupled=True, shear0=100.0)
    assert r["valida"] and r["iteraciones"] > 1
    q = float(r["Q"].abs()[0])
    eta = float(visc(net.D, r["H"], net.apparent_shear(r["Q"]))[0])
    q_recalc = math.pi * 10.0 ** 4 * 30.0 / (128.0 * eta * 500.0)
    assert abs(q - q_recalc) / q < 1e-4                                     # antes: 10.8 % de discrepancia
    assert not net.solve(visc, PhaseSeparation(), shear_coupled=True, iters=1)["valida"]


def test_negative_viscosity_rejected_and_dissipation_reported():
    net = MicrovascularNetwork(2, [(0, 1)], [500.0], [10.0], {0: 30.0, 1: 0.0}, {0: 0.4})
    with pytest.raises(ValueError):
        net.solve(lambda D, H: -torch.ones_like(D), PhaseSeparation())
    with pytest.raises(ValueError):
        net.solve(lambda D, H: torch.zeros_like(D), PhaseSeparation())
    r = net.solve(lambda D, H: 1.0 + 2.5 * H, PhaseSeparation())
    assert r["disipacion"] > 0 and r["comprobaciones"]["disipacion_no_negativa"] and r["comprobaciones"]["viscosidad_positiva"]


def test_non_inferiority_requires_real_pairs():
    sys.path.insert(0, "analisis")
    from estadistica import no_inferioridad
    with pytest.raises(ValueError):
        no_inferioridad([0.1], [0.2] * 5, margen=0.01)                      # antes: difusión de NumPy y "no inferior"
    with pytest.raises(ValueError):
        no_inferioridad([0.1] * 5, [float("nan")] * 5, margen=0.01)
    with pytest.raises(ValueError):
        no_inferioridad([0.1] * 5, [0.1] * 5, margen=-1)
    ref = {s: 0.50 + 0.01 * (s % 3) for s in range(8)}
    r = no_inferioridad({s: v + 0.004 + 0.001 * (s % 2) for s, v in ref.items()}, ref, margen=0.02)
    assert r["no_inferior"] is True and r["n"] == 8 and r["semillas"] == list(range(8))
    assert no_inferioridad({0: 0.1, 1: 0.1}, {1: 0.1, 2: 0.1}, margen=0.02)["n"] == 1     # solo la semilla común


def test_pickle_disabled_by_default(gemelo, tmp_path):
    from eho import Gemelo
    gemelo.guardar(tmp_path / "r.eho", modo="reanudar")
    with pytest.raises(RuntimeError):
        Gemelo.cargar(tmp_path / "r.eho")
    assert Gemelo.cargar(tmp_path / "r.eho", permitir_pickle=True).datos_entrenamiento is not None


def test_export_integrity_hash(gemelo, tmp_path):
    from eho import Gemelo
    gemelo.guardar(tmp_path / "s.eho")
    estado = torch.load(tmp_path / "s.eho", weights_only=True)
    k = next(k for k, v in estado["agentes"][0]["state"].items() if v.dtype.is_floating_point and v.numel() > 4)
    estado["agentes"][0]["state"][k] = estado["agentes"][0]["state"][k] + 1e-3
    torch.save(estado, tmp_path / "t.eho")
    with pytest.raises(RuntimeError):
        Gemelo.cargar(tmp_path / "t.eho")                                   # pesos alterados: no pasa
    assert Gemelo.cargar(tmp_path / "s.eho").protocolo["huella"] == gemelo.protocolo["huella"]


def test_ltt_grid_falls_back_instead_of_abstaining():
    ref = torch.rand(100)
    assert ltt_grid(ref, 90, 0.1, 0.1, reference_fail=torch.ones(100, dtype=torch.bool)) != []


def test_sentinel_insufficient_evidence_and_equal_false_alarm():
    c = Centinela([], torch.ones(1), alpha=0.1)
    c.cal_ep = torch.linspace(0.5, 1.5, 49)
    c.huella, c.vigente = None, lambda: True
    c.episode_scores = lambda ds, idx=None, seed=0: (torch.tensor([1.0, 9.0, float("nan")]), None, torch.arange(3))
    estados, _ = c.patient_states(None)
    assert estados == ["compatible", "fuera_de_dominio", "evidencia_insuficiente"]
    assert not c.patient_alarms(None)[0][2]
    known = torch.randn(999)
    thr = Centinela.threshold_at_false_alarm(known, 0.05)
    assert abs(float((torch.randn(200000) > thr).float().mean()) - 0.05) < 0.02
    assert Centinela.threshold_at_false_alarm(torch.randn(5), 0.05) == float("inf")      # muy pocos: no alarma


def test_armijo_guarantees_energy_descent(gemelo):
    m = copy.deepcopy(gemelo.agentes[0]).eval()
    ep = gemelo.dominio_variante(0).generate(3, 8, "difusivo", seed=5)
    assert m.energy_trace(ep)["fraccion_violaciones"] == 0.0               # paso por defecto: sin violaciones
    m.core.alpha = 20.0                                                     # paso absurdo
    assert m.energy_trace(ep)["fraccion_violaciones"] > 0.0                # el paso fijo sí viola
    m.core.step_mode = "armijo"
    t = m.energy_trace(ep)
    assert t["fraccion_violaciones"] == 0.0 and t["aumento_maximo"] <= 1e-6   # garantizado por construcción
