"""Pruebas del sistema de memoria, la colmena y el núcleo general."""
import torch

from eho.memory import NeuralLTM, PlasmodialMemory, MemorySystem
from eho.colony import BeeColony, colony_predict, colony_select
from eho.dominios import AdveccionDifusion1D, Sangre, split_episodes
from eho.central import EHOCentral
from eho.train_central import fit_central, evaluate_fewshot


# ---------------- memoria neuronal (Titans) ----------------
def test_ltm_aprende_y_la_sorpresa_baja():
    torch.manual_seed(0)
    m = NeuralLTM(8, 2)
    k, v = torch.randn(16, 8), torch.randn(16, 2)
    e0 = float(m.surprise(k, v).mean())
    for _ in range(100):
        m.write(k, v)
    assert float(m.surprise(k, v).mean()) < 0.1 * e0


def test_ltm_olvida_sin_repaso():
    torch.manual_seed(0)
    m = NeuralLTM(8, 2, alpha=0.05)
    k, v = torch.randn(8, 8), torch.randn(8, 2)
    for _ in range(100):
        m.write(k, v)
    e_learned = float(m.surprise(k, v).mean())
    k2, v2 = torch.randn(8, 8), torch.randn(8, 2)
    for _ in range(200):
        m.write(k2, v2)                       # escribir otra cosa con olvido alto
    assert float(m.surprise(k, v).mean()) > e_learned


# ---------------- memoria plasmodial ----------------
def _mpc_con_grupos():
    torch.manual_seed(0)
    p = PlasmodialMemory(8, 2, novelty_new=0.2)
    centers = torch.randn(6, 8) * 3
    for c in centers:
        p.write(c[None], torch.randn(1, 2))
    return p, centers


def test_mpc_kirchhoff_conserva_flujo():
    p, centers = _mpc_con_grupos()
    b, _ = p._injection(torch.randn(4, 8))
    _, Q = p._solve(b)
    div = torch.zeros(4, len(p))
    div.index_add_(1, p.edges[:, 0], Q)
    div.index_add_(1, p.edges[:, 1], -Q)
    assert torch.allclose(div, b, atol=1e-4)          # lo que entra por cada nodo sale por sus aristas
    assert torch.allclose(b.sum(-1), torch.zeros(4), atol=1e-6)


def test_mpc_refuerza_lo_usado_y_poda_lo_demas():
    p, centers = _mpc_con_grupos()
    D0 = p.D.clone()
    for _ in range(40):
        p.write(centers[0:1].repeat(8, 1) + 0.05 * torch.randn(8, 8), torch.zeros(8, 2))
    near0 = (p.edges == 0).any(-1)
    grow_used, grow_other = (p.D[near0] - D0[near0]).mean(), (p.D[~near0] - D0[~near0]).mean()
    assert grow_used > 0 and grow_other < 0                      # lo usado se ensancha, lo demás se encoge
    assert (p.D[near0] - D0[near0]).min() > (p.D[~near0] - D0[~near0]).max() * 0 + 0  # toda ruta usada crece
    assert (p.D >= p.cfg["d_min"] - 1e-9).all()                 # plasticidad mínima conservada
    assert p.z.any()                                            # hubo consolidación


def test_mpc_lectura_es_distribucion():
    p, _ = _mpc_con_grupos()
    _, pi, _ = p.read(torch.randn(5, 8))
    assert torch.allclose(pi.sum(-1), torch.ones(5), atol=1e-5) and (pi >= 0).all()


def test_mpc_novedad_menor_en_lo_visto():
    p, centers = _mpc_con_grupos()
    assert (p.novelty(centers) < p.novelty(-centers)).all()


def test_sistema_memoria_neutro_al_inicio():
    ms = MemorySystem(16, 2)
    x, y = torch.randn(6, 16), torch.randn(6, 2)
    assert ms.read(x, x, y).abs().max() == 0                     # capa final en cero
    ms.write(x, y)
    assert ms.read(x).shape == (6, 16)


# ---------------- colmena ----------------
def test_colmena_elige_la_mejor():
    d = BeeColony().decide(torch.tensor([[0.9, 0.5, 0.4]] * 4))
    assert d.choice == 0


def test_colmena_se_abstiene_si_todo_es_malo():
    assert BeeColony().decide(torch.tensor([[0.1, 0.12, 0.08]] * 4)).choice is None


def test_colmena_se_abstiene_si_hay_division():
    assert BeeColony().decide(torch.tensor([[0.9, 0.2], [0.9, 0.2], [0.2, 0.9], [0.2, 0.9]])).choice is None


def test_colmena_rompe_empates_entre_opciones_buenas():
    assert BeeColony().decide(torch.tensor([[0.8, 0.8, 0.2]] * 4)).choice in (0, 1)


def test_colmena_mayoria():
    assert BeeColony().decide(torch.tensor([[0.9, 0.3], [0.85, 0.3], [0.8, 0.35], [0.3, 0.9]])).choice == 0


def test_prediccion_colectiva():
    preds = torch.tensor([[[1.0], [0.0]], [[1.05], [3.0]], [[0.95], [-2.0]], [[1.02], [6.0]]])
    _, acc, _ = colony_predict(preds, 0.2)
    assert acc.tolist() == [True, False]


def test_seleccion_de_experimento():
    scores = torch.tensor([[0.1, 0.2, 0.9, 0.3], [0.2, 0.1, 0.8, 0.2], [0.1, 0.3, 0.95, 0.1]])
    choice, _ = colony_select(scores)
    assert choice == 2


# ---------------- núcleo general ----------------
def test_nucleo_general_en_dos_dominios():
    torch.manual_seed(0)
    for dom, dv in [(AdveccionDifusion1D(n_grid=32), 5), (Sangre(n_points=32), 2)]:
        tr = dom.generate(6, 4, dom.regimes[0], 1)
        te = dom.generate(3, 5, dom.regimes[1], 2)
        m = EHOCentral(dom, memory="completa", d_val=dv)
        fit_central(m, tr, tr, epochs=2, eco_warmup=1)
        r0, r2 = evaluate_fewshot(m, te, 0), evaluate_fewshot(m, te, 2)
        assert all(torch.isfinite(torch.tensor(list(r.values()))).all() for r in (r0, r2))


def test_contexto_no_filtra_otros_episodios():
    """En entrenamiento, una muestra solo lee de su propio episodio y nunca de sí misma."""
    dom = AdveccionDifusion1D(n_grid=16)
    b = dom.generate(2, 3, "mixto", 0)
    m = EHOCentral(dom, memory="episodica", d_val=5).eval()       # en evaluación: sin estadísticas de lote
    torch.nn.init.normal_(m.memory.fuse[-1].weight)               # que la memoria influya
    out1 = m(b, episodic_self=True)["scalars"]
    b2 = {k: v.clone() for k, v in b.items()}
    b2["scalars"][3:] += 10.0                                       # cambiar SOLO el episodio 1
    b2["fields"][3:] += 10.0
    out2 = m(b2, episodic_self=True)["scalars"]
    assert torch.allclose(out1[:3], out2[:3], atol=1e-5)            # el episodio 0 no se entera


def test_division_de_episodios():
    dom = AdveccionDifusion1D(n_grid=16)
    ds = dom.generate(3, 5, "mixto", 0)
    c, q = split_episodes(ds, 2)
    assert c.numel() == 6 and q.numel() == 9 and not set(c.tolist()) & set(q.tolist())


def test_quorum_calibrado_logra_cobertura():
    from eho.colony import ColonyPredictor
    torch.manual_seed(0)
    A, B = 4, 60
    truth = torch.randn(B, 1)
    spread = torch.rand(B, 1, 1).expand(B, A, 1) * 2              # desacuerdo variable por muestra
    preds = (truth.unsqueeze(1) + spread * torch.randn(B, A, 1)).permute(1, 0, 2)
    cp = ColonyPredictor(torch.tensor([0.5])).fit(preds, truth, None, target_coverage=0.6)
    _, acc, _ = cp.predict(preds)
    assert abs(float(acc.float().mean()) - 0.6) < 0.15


def test_memoria_indexada_por_episodio():
    dom = AdveccionDifusion1D(n_grid=16)
    b = dom.generate(3, 4, "mixto", 0)
    m = EHOCentral(dom, memory="completa", d_val=5).eval()
    info = m.consolidate(b)
    assert "sorpresa" in info and m.memory.ltm_key is not None
    assert len(m.memory.mpc) > 0


def test_segundo_simulador_y_geometria_nueva():
    from eho.dominios import Sangre
    a = Sangre(simulador="carreau", n_points=16).generate(3, 2, "dt2", 0)
    b = Sangre(simulador="real", n_points=16, D_range=(3.0, 5.5)).generate(3, 2, "dt2", 0)
    assert torch.isfinite(a["scalars"]).all() and torch.isfinite(b["scalars"]).all()


def test_seleccion_con_muchas_opciones_buenas_no_se_traba():
    torch.manual_seed(0)
    V = 0.9 + 0.08 * torch.rand(4, 8)
    choice, _ = colony_select(V, normalize=False)
    assert choice is not None


def test_seleccion_se_abstiene_si_nada_vale_la_pena():
    V = 0.05 * torch.rand(4, 8)
    choice, _ = colony_select(V, normalize=False)
    assert choice is None


def test_valor_de_novedad_absoluto():
    from eho.active import NoveltyValue
    dom = AdveccionDifusion1D(n_grid=32)
    tr = dom.generate(8, 4, "difusivo", 1)
    m = EHOCentral(dom, memory="completa", d_val=5)
    fit_central(m, tr, tr, epochs=2, eco_warmup=0)
    m.consolidate(tr)
    nv = NoveltyValue([m.eval()]).fit(tr)
    v = nv.values(AdveccionDifusion1D(n_grid=32, k_min=4).generate(4, 2, "difusivo", 2))
    assert ((v >= 0) & (v <= 1)).all()


def test_mpc_llena_sigue_creciendo():
    torch.manual_seed(0)
    p = PlasmodialMemory(8, 2, max_nodes=16, novelty_new=0.05)
    for _ in range(10):
        p.write(torch.randn(8, 8) * 3, torch.randn(8, 2))
    assert len(p) <= 16
    far = torch.randn(1, 8) * 3
    nov_before = float(p.novelty(far))
    p.write(far, torch.zeros(1, 2))
    assert float(p.novelty(far)) < nov_before            # lo nuevo entra aunque la memoria esté llena
    assert int(p.edges.max()) < len(p)                   # aristas consistentes tras fusionar
