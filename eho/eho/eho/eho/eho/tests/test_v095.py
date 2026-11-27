"""Pruebas de regresión de la tercera auditoría externa (v0.9.5)."""
import copy

import pytest
import torch

from eho.centinela import huella_agentes
from eho.dominios import AdveccionDifusion1D, assert_disjoint, episode_hashes
from eho.network import MicrovascularNetwork, PhaseSeparation

CFG = {"nombre": "t", "dominio": {"nombre": "edp1d", "regimenes": ["difusivo"], "episodios_por_regimen": 8,
                                  "muestras_por_episodio": 4, "episodios_validacion": 4, "episodios_calibracion": 30},
       "modelo": {"d": 24, "n_slices": 4}, "colmena": {"n_agentes": 2, "calibracion": "ltt", "riesgo": 0.3, "delta": 0.2},
       "entrenamiento": {"epocas": 2, "dispositivo": "cpu"},
       "activo": {"presupuesto": 2, "epocas_ajuste": 1, "repeticion": 4, "presupuesto_total": 8}}
VISC = lambda D, H: 1.0 + 2.5 * H


@pytest.fixture(scope="module")
def gemelo():
    from eho import Gemelo
    return Gemelo.desde_config(CFG).entrenar(verbose=False)


def test_boundary_node_with_throughflow():
    # tres nodos con presión impuesta y fuerte reducción de diámetro: en el nodo 1 SALE sangre
    net = MicrovascularNetwork(3, [(0, 1), (1, 2)], [500.0, 500.0], [20.0, 5.0], {0: 30.0, 1: 10.0, 2: 0.0}, {0: 0.45})
    r = net.solve(VISC, PhaseSeparation())
    assert r["valida"] and abs(r["H_maximo"] - 0.45) < 1e-9           # antes: H ~ 4000 y valida = True
    # al revés ENTRA sangre en el nodo 1: sin hematocrito de entrada no se puede resolver
    net = MicrovascularNetwork(3, [(0, 1), (1, 2)], [500.0, 500.0], [5.0, 20.0], {0: 30.0, 1: 25.0, 2: 0.0}, {0: 0.45})
    r = net.solve(VISC, PhaseSeparation())
    assert not r["valida"] and not r["comprobaciones"]["resuelta"]
    net = MicrovascularNetwork(3, [(0, 1), (1, 2)], [500.0, 500.0], [5.0, 20.0], {0: 30.0, 1: 25.0, 2: 0.0}, {0: 0.45, 1: 0.30})
    r = net.solve(VISC, PhaseSeparation())
    assert r["valida"] and 0.30 < float(r["H"][1]) < 0.45             # mezcla de lo que llega y lo que entra


def test_valid_flag_requires_physical_range():
    net = MicrovascularNetwork(3, [(0, 1), (1, 2)], [500.0, 500.0], [20.0, 5.0], {0: 30.0, 1: 10.0, 2: 0.0}, {0: 0.45})
    r = net.solve(VISC, PhaseSeparation())
    H = r["H"].clone(); H[1] = 4000.0
    v = net._validez(r["p"], r["Q"], H, r["rbc_flux"], True, 0.0, 1e-6)
    assert not v["valida"] and not v["comprobaciones"]["H_en_rango"] and not v["comprobaciones"]["identidad"]
    H[1] = float("nan")
    assert not net._validez(r["p"], r["Q"], H, r["rbc_flux"], True, 0.0, 1e-6)["comprobaciones"]["finita"]


def test_parent_diameter_gradient():
    def run(D):
        net = MicrovascularNetwork(4, [(0, 1), (1, 2), (1, 3)], [500.0] * 3, [10.0, 8.0, 5.0],
                                   {0: 30.0, 2: 0.0, 3: 0.0}, {0: 0.45})
        net.D = D
        return net.solve(lambda D_, H: 1.0 + 2.5 * H, PhaseSeparation(), tol=1e-12, iters=400)["H"]
    D = torch.tensor([10.0, 8.0, 5.0], dtype=torch.float64, requires_grad=True)
    g, = torch.autograd.grad(run(D)[2], D)
    e = torch.zeros(3, dtype=torch.float64); e[0] = 1e-3
    fd = float((run((D + e).detach())[2] - run((D - e).detach())[2]) / 2e-3)
    assert abs(fd) > 1e-6 and abs(float(g[0]) - fd) < 2e-2 * abs(fd)     # el diámetro del vaso padre cuenta


def test_shear_coupled_solve():
    net = MicrovascularNetwork(4, [(0, 1), (1, 2), (1, 3)], [500.0] * 3, [10.0, 8.0, 4.0], {0: 30.0, 2: 0.0, 3: 0.0}, {0: 0.45})
    visto = {}
    def visc(D, H, g):                                                  # adelgazamiento por cizalla
        visto["g"] = g.detach().clone()
        return (1.0 + 2.5 * H) * (1.0 + 1.0 / (1.0 + g))
    r = net.solve(visc, PhaseSeparation(), shear_coupled=True)
    assert r["valida"] and float(visto["g"].max() - visto["g"].min()) > 0   # cizalla distinta en cada vaso
    assert torch.allclose(visto["g"], net.apparent_shear(r["Q"]), rtol=1e-3)


def test_fingerprint_changes_on_permutation_and_buffers(gemelo):
    ag = copy.deepcopy(gemelo.agentes)
    h0 = huella_agentes(ag)
    w = next(p for p in ag[0].parameters() if p.dim() == 2 and p.shape[0] > 1)
    with torch.no_grad():
        w.copy_(w.flip(0))                                              # misma suma y misma norma
    assert huella_agentes(ag) != h0
    ag = copy.deepcopy(gemelo.agentes)
    ag[0].moe.n.add_(0.01)                                              # un buffer (abundancias)
    assert huella_agentes(ag) != h0
    assert huella_agentes(gemelo.agentes, {"k": 2}) != huella_agentes(gemelo.agentes, {"k": 5})


def test_episode_hash_order_invariant_and_partial_overlap():
    d = AdveccionDifusion1D()
    a = d.generate(3, 4, "difusivo", seed=1)
    perm = torch.arange(a["episode"].numel()).flip(0)
    b = {k: v[perm] for k, v in a.items()}
    assert episode_hashes(a) == episode_hashes(b)
    with pytest.raises(ValueError):
        assert_disjoint(a, b)
    parcial = {k: v[:2] for k, v in a.items()}                           # mismo paciente, menos muestras
    with pytest.raises(ValueError):
        assert_disjoint(a, parcial)


def test_ltt_certificate_bound_to_context_protocol(gemelo):
    d0 = gemelo.dominio_variante(0)
    ep = d0.generate(1, 6, "difusivo", seed=31)
    ctx2 = {k: v[:2] for k, v in ep.items()}
    q = {k: v[2:] for k, v in ep.items()}
    r0 = gemelo.predecir(q)                                              # sin mediciones: protocolo distinto
    assert not r0["protocolo_coincide"] and not r0["aceptado"].any()
    assert gemelo.predecir(q, contexto=ctx2)["protocolo_coincide"]       # k = 2, como en la calibración
    assert not gemelo.predecir({k: v[3:] for k, v in ep.items()},
                               contexto={k: v[:3] for k, v in ep.items()})["protocolo_coincide"]
    otro = d0.generate(2, 2, "difusivo", seed=32)                        # contexto de dos pacientes
    with pytest.raises(ValueError):
        gemelo.predecir(q, contexto=otro)


def test_provenance_audit_is_mandatory(gemelo):
    with pytest.raises(ValueError):
        gemelo.auditar_particiones(calib=gemelo.datos_entrenamiento[0])  # mismo archivo para entrenar y calibrar
    gemelo.auditar_particiones(calib=gemelo.datos_calibracion())


def test_incorporar_validates_and_is_transactional(gemelo, monkeypatch):
    import eho.gemelo as G
    g = copy.deepcopy(gemelo)
    cand = g.dominio_variante(0).generate(4, 4, "difusivo", seed=3)
    A = {k: v[cand["episode"] < 2] for k, v in cand.items()}
    B = {k: v[cand["episode"] >= 2] for k, v in cand.items()}
    with pytest.raises(ValueError):
        g.incorporar([A])                                                # un conjunto para dos agentes
    with pytest.raises(ValueError):
        g.incorporar([A, B])                                             # condiciones distintas por agente
    antes = [p.clone() for p in g.agentes[0].parameters()]
    real, llamadas = G.fit_central, {"n": 0}

    def falla(*a, **k):
        llamadas["n"] += 1
        if llamadas["n"] == 2:
            raise RuntimeError("corte a mitad de la incorporación")
        return real(*a, **k)
    monkeypatch.setattr(G, "fit_central", falla)
    with pytest.raises(RuntimeError):
        g.incorporar(A)
    assert getattr(g, "_simulaciones_usadas", 0) == 0 and (getattr(g, "parada", None) is None or g.parada.n == 0)
    assert all(torch.equal(x, y) for x, y in zip(antes, g.agentes[0].parameters()))   # agente 0 restaurado
    assert g.historial == gemelo.historial and g.calibracion_vigente


def test_execution_and_cost_budgets(gemelo):
    g = copy.deepcopy(gemelo)
    g.cfg.activo.presupuesto_ejecuciones = 3
    cand = g.dominio_variante(0).generate(4, 4, "difusivo", seed=3)
    with pytest.raises(ValueError):
        g.incorporar(cand)                                               # 4 ejecuciones > 3
    g.cfg.activo.presupuesto_ejecuciones, g.cfg.activo.presupuesto_costo = None, 10.0
    g.incorporar({k: v[cand["episode"] < 2] for k, v in cand.items()}, costo=10.0)
    assert g.presupuesto_restante() == 0 and g.proponer(cand) == []


def test_deployment_file_has_no_validation_data(gemelo, tmp_path):
    from eho import Gemelo
    gemelo.guardar(tmp_path / "d.eho")
    estado = torch.load(tmp_path / "d.eho", weights_only=True)             # v0.9.6: se lee sin pickle
    assert estado["formato"].startswith("eho-seguro")
    assert not ({"validacion", "datos_entrenamiento", "acumulado", "valor"} & set(estado))
    h = Gemelo.cargar(tmp_path / "d.eho")
    ep = h.dominio_variante(0).generate(1, 6, "difusivo", seed=31)
    r = h.predecir({k: v[2:] for k, v in ep.items()}, contexto={k: v[:2] for k, v in ep.items()})
    assert r["protocolo_coincide"] and r["escalares"].shape[0] == 4


def test_cli_train_then_incorporate(tmp_path):
    """Flujo de línea de comandos: entrenar -> incorporar. Roto en v0.9.4 (se guardaba despliegue)."""
    import yaml
    from eho.__main__ import main
    from eho import Gemelo
    cfg = copy.deepcopy(CFG); cfg["colmena"] = {"n_agentes": 2}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    main(["entrenar", str(tmp_path / "c.yaml"), "-o", str(tmp_path / "m.eho")])
    nuevos = AdveccionDifusion1D().generate(2, 4, "difusivo", seed=77)
    torch.save(nuevos, tmp_path / "n.pt")
    with pytest.raises(RuntimeError):
        main(["incorporar", str(tmp_path / "m.eho"), str(tmp_path / "n.pt")])          # pickle sin autorizar
    main(["incorporar", str(tmp_path / "m.eho"), str(tmp_path / "n.pt"), "-o", str(tmp_path / "m2.eho"), "--despliegue",
          "--permitir-pickle"])
    g = Gemelo.cargar(tmp_path / "m2.eho")
    assert g._simulaciones_usadas == 2 and g.solo_despliegue and g.validacion is None
