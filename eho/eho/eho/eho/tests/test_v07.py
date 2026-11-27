"""Pruebas de la v0.7: novedad por resistencia, posición sin contexto, Learn-then-Test, Burgers,
ley de Pries, PDEBench, dominio de archivo y líneas base."""
import math

import pytest
import torch

from eho.memory.plasmodial import PlasmodialMemory
from eho.colony import heuristic_threshold as ltt_threshold, binom_pvalue
from eho.dominios.burgers1d import solve_burgers, Burgers1D
from eho.data.synthetic import pries_in_vitro, ground_truth, sample_theta
from eho.baselines import auroc, Mahalanobis, knn_score


def _mem(n=40, d=8, seed=0):
    torch.manual_seed(seed)
    m = PlasmodialMemory(d, 2, novelty_new=0.0, k_nn=4)
    k = torch.randn(n, d)
    m.write(k, torch.randn(n, 2))
    for _ in range(5):
        m.write(k[:10] + 0.01 * torch.randn(10, d), torch.randn(10, 2))   # uso sostenido de 10 nodos
    return m, k


def _brute_resistance(m, q):
    """Laplaciano aumentado explícito: nodo consulta + red + tierra; R = potencial de la consulta."""
    n = len(m)
    kn = torch.nn.functional.normalize(m._pos(), dim=-1).double()
    qn = torch.nn.functional.normalize(q, dim=-1).double()
    G = (m.D / m._lengths()).double()
    c = m.cfg
    out = []
    for b in range(q.shape[0]):
        L = torch.zeros(n + 1, n + 1, dtype=torch.float64)
        for (i, j), g in zip(m.edges.tolist(), G.tolist()):
            L[i, j] -= g; L[j, i] -= g; L[i, i] += g; L[j, j] += g
        L[:n, :n] += torch.diag(c["ground"] * (c["ground_floor"] + m.hab.double()))
        d = (kn - qn[b]).norm(dim=-1)
        dJ, J = d.topk(min(c["k_nn"], n), largest=False)
        for j, dj in zip(J.tolist(), dJ.tolist()):
            g = 1.0 / max(dj, 1e-3)
            L[n, j] -= g; L[j, n] -= g; L[n, n] += g; L[j, j] += g
        e = torch.zeros(n + 1, dtype=torch.float64); e[n] = 1.0
        out.append(torch.linalg.solve(L, e)[n].log())
    return torch.stack(out).float()


def test_resistencia_igual_a_fuerza_bruta():
    m, k = _mem()
    q = torch.randn(5, k.shape[1])
    assert torch.allclose(m.resistance_novelty(q), _brute_resistance(m, q), atol=1e-4)


def test_resistencia_lejos_mayor_y_habituado_menor():
    m, k = _mem()
    lejos = 5.0 * torch.randn(20, k.shape[1])
    cerca = k[:20] + 0.01 * torch.randn(20, k.shape[1])
    assert m.resistance_novelty(lejos).mean() > m.resistance_novelty(cerca).mean()
    # misma consulta exacta sobre un nodo usado de forma sostenida frente a uno poco usado
    r_usado = m.resistance_novelty(m._pos()[:10]).mean()
    r_poco = m.resistance_novelty(m._pos()[20:30]).mean()
    assert r_usado < r_poco


def test_posicion_separada_de_la_clave():
    torch.manual_seed(1)
    m = PlasmodialMemory(6, 1, novelty_new=0.0)
    k, p = torch.randn(30, 6), torch.randn(30, 6)
    m.write(k, torch.randn(30, 1), pos=p)
    assert m.pos.shape == m.keys.shape
    # la novedad de un candidato se mide contra la posición, no contra la clave
    assert m.cosine_novelty(p[:5]).max() < 0.15        # 0.1 = término de habituación (nodos nuevos)
    assert m.cosine_novelty(p[:5], space="key").min() > 0.15
    m._merge_closest(5)
    assert m.pos.shape == m.keys.shape


def test_ltt_controla_el_riesgo():
    """Apoyo informativo: la tasa real de fallo entre los aceptados no supera el riesgo
    en más de una fracción δ de las repeticiones (más holgura por el número finito de repeticiones)."""
    risk, delta, viol = 0.1, 0.1, 0
    g = torch.Generator().manual_seed(0)
    for rep in range(100):
        s = torch.rand(400, generator=g)
        p_fail = 0.4 * (1 - s) ** 2                                         # más apoyo, menos fallo
        fail = torch.rand(400, generator=g) < p_fail
        thr, info = ltt_threshold(s, fail, risk, delta)
        s2 = torch.rand(20000, generator=g)
        f2 = torch.rand(20000, generator=g) < 0.4 * (1 - s2) ** 2
        acc = s2 >= thr
        if acc.any() and f2[acc].float().mean() > risk:
            viol += 1
    assert viol / 100 <= delta + 0.05
    assert info["cobertura_calibracion"] > 0.3


def test_ltt_sin_evidencia_se_abstiene():
    s = torch.rand(10)
    thr, info = ltt_threshold(s, torch.zeros(10, dtype=torch.bool), 0.1, 0.1)
    assert thr > s.max() and info["aceptados_calibracion"] == 0      # 0.9^10 = 0.35 > δ
    assert binom_pvalue(0, 22, 0.1) < 0.1


def test_burgers_cole_hopf():
    N, nu, t = 256, 0.02, 0.2
    x = torch.arange(N, dtype=torch.float64) / N
    k = 2 * math.pi

    def exacta(tt):
        e = math.exp(-nu * k * k * tt)
        return -2 * nu * (-0.8 * e * k * torch.sin(k * x)) / (1 + 0.8 * e * torch.cos(k * x))

    uT = solve_burgers(exacta(0).unsqueeze(0), torch.tensor([nu]), torch.tensor([t]))[0]
    assert float((uT - exacta(t)).norm() / exacta(t).norm()) < 1e-6


def test_burgers_dominio_y_energia():
    ds = Burgers1D(n_solve=256).generate(4, 4, "intermedio", seed=0)
    assert torch.isfinite(ds["fields"]).all() and ds["fields"].shape == (16, 64, 1)
    assert (ds["scalars"][:, 0] < 0).all()                            # la energía decrece


def test_pries_minimo_entre_5_y_10_um():
    D = torch.linspace(3, 40, 400)
    eta = pries_in_vitro(D, torch.full_like(D, 0.45))
    assert 5.0 < float(D[eta.argmin()]) < 10.0
    assert 1.15 < float(eta.min()) < 1.4
    _, _, s = ground_truth(sample_theta(8, "dt2"), torch.full((8,), 0.4), torch.full((8,), 20.0),
                           torch.full((8,), 50.0), 16, simulador="pries")
    assert torch.isfinite(s).all() and (s[:, 0] > 0).all()


def test_pdebench_lector(tmp_path):
    h5py = pytest.importorskip("h5py")
    for beta in ("0.4", "1.0"):
        with h5py.File(tmp_path / f"adv_{beta}.hdf5", "w") as f:
            f["tensor"] = torch.randn(50, 30, 128).numpy()
            f["t-coordinate"] = torch.linspace(0, 2, 31).numpy()
    from eho.dominios import PDEBench1D
    d = PDEBench1D({"b0.4": str(tmp_path / "adv_0.4.hdf5"), "b1.0": str(tmp_path / "adv_1.0.hdf5")},
                   n_grid=64, t_index_range=(2, 20), traj_range=(0.0, 0.8))
    ds = d.generate(4, 5, "b0.4", seed=0)
    assert ds["points"].shape == (20, 64, 3) and ds["fields"].shape == (20, 64, 1)
    assert torch.isfinite(ds["scalars"]).all()
    with pytest.raises(ValueError):
        d.generate(10, 5, "b1.0")                                    # 50 > 40 trayectorias del rango


def test_dominio_archivo(tmp_path):
    from eho.dominios import DominioArchivo, AdveccionDifusion1D
    src = AdveccionDifusion1D().generate(10, 4, "difusivo", seed=0)
    torch.save(src, tmp_path / "res.pt")
    d = DominioArchivo({"difusivo": str(tmp_path / "res.pt")}, episodios_excluidos={"difusivo": [0, 1]})
    assert (d.d_point, d.n_fields, d.n_scalars) == (3, 1, 2)
    ds = d.generate(5, 4, "difusivo", seed=1)
    assert ds["episode"].unique().numel() == 5
    assert not torch.isin(ds["episodio_original"], torch.tensor([0, 1])).any()


def test_lineas_base_basicas():
    a, b = torch.zeros(50), torch.ones(50)
    assert auroc(a, b) == 1.0 and auroc(b, a) == 0.0 and auroc(a, a) == 0.5
    X = torch.randn(500, 4)
    m = Mahalanobis().fit(X)
    assert m.score(X + 10).mean() > m.score(X).mean()
    assert knn_score(X, X[:5] + 10).mean() > knn_score(X, X[:5] + 0.01).mean()


def test_gemelo_ltt_y_resistencia():
    from eho import Gemelo
    cfg = {"nombre": "t", "dominio": {"nombre": "edp1d", "regimenes": ["difusivo"], "episodios_por_regimen": 8,
                                      "muestras_por_episodio": 4, "episodios_validacion": 4,
                                      "episodios_calibracion": 30},
           "modelo": {"d": 24, "n_slices": 4, "novedad": "resistencia", "ecologia": False},
           "colmena": {"n_agentes": 2, "calibracion": "ltt", "riesgo": 0.3, "delta": 0.2},
           "entrenamiento": {"epocas": 2, "dispositivo": "cpu"}}
    g = Gemelo.desde_config(cfg).entrenar(verbose=False)
    assert g.predictor.method == "ltt" and "aceptados_calibracion" in g.predictor.ltt_info
    assert not g.agentes[0].use_ecology
    assert g.agentes[0].memory.mpc.cfg["novelty_mode"] == "resistencia"
    nuevo = g.dominio_variante(0).generate(2, 4, "difusivo", seed=5)
    r = g.predecir(nuevo)
    assert r["aceptado"].dtype == torch.bool
