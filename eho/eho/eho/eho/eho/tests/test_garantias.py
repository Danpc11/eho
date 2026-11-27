"""Verificación numérica de las Proposiciones 1-3 y de la conservación exacta."""
import torch
import pytest

from eho.hopfield import EnergyHopfieldCore, EnergyLayerNorm, AssociativeMemory
from eho.ecology import EcoMoE
from eho.thermo import DissipationPotential, GenericCell, simple_shear
from eho.network import MicrovascularNetwork, PhaseSeparation, pries_in_vitro_relative_viscosity
from eho.panel import PlasmaViscosity, PanelEncoder, VAR_INDEX
from eho.model import EHO
from eho.data.synthetic import make_dataset

torch.manual_seed(0)


def test_layernorm_jacobiano_psd():
    """g = ∇L con L convexo => jacobiano simétrico y semidefinido positivo."""
    ln = EnergyLayerNorm(16)
    for _ in range(20):
        z = torch.randn(16)
        J = torch.autograd.functional.jacobian(ln, z)
        assert torch.allclose(J, J.T, atol=1e-5)
        assert torch.linalg.eigvalsh(0.5 * (J + J.T)).min() > -1e-5


@pytest.mark.parametrize("seed", range(5))
def test_proposicion1_energia_decrece(seed):
    torch.manual_seed(seed)
    core = EnergyHopfieldCore(32, steps=30, alpha=0.02).eval()
    _, E = core(torch.randn(4, 10, 32), return_energy=True)
    dE = E[:, 1:] - E[:, :-1]
    assert (dE <= 1e-4).all(), dE.max()


def test_lectura_memoria_es_atencion():
    mem = AssociativeMemory(8, 2, beta=50.0)
    k = torch.eye(8)[:3]
    v = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    mem.write(k, v)
    out, _ = mem.read(k[1:2] + 0.01 * torch.randn(1, 8))
    assert torch.allclose(out, v[1:2], atol=1e-2)
    assert mem.novelty(k[:1]) < mem.novelty(-k[:1])          # lo visto es menos novedoso


def test_proposicion2_lyapunov_lotka_volterra():
    moe = EcoMoE(8, 3, j_init=6, j_max=6, dt=0.05)
    for _ in range(10):
        pi = torch.softmax(torch.randn(200, 6) * 2, -1)
        r = 0.5 + torch.rand(6)
        A = (pi.T @ pi) / 200
        m2 = (pi ** 2).mean(0)
        A = A / torch.sqrt(m2[:, None] * m2[None, :])
        A.fill_diagonal_(1.0)
        moe.n = torch.rand(6) + 0.1
        V = moe.lotka_volterra(r, A, steps=200)
        assert all(b <= a + 1e-6 for a, b in zip(V, V[1:])), "V(n) debe decrecer"


def test_proposicion3_disipacion_no_negativa():
    dp = DissipationPotential(8)
    for _ in range(10):
        D = torch.randn(64, 3, 3)
        D = 0.5 * (D + D.transpose(-1, -2)) * 50
        c = torch.randn(64, 8)
        tau = dp.stress(D, c)
        diss = (tau * D).sum(dim=(-1, -2))
        assert (diss >= -1e-6).all()
        assert (dp.potential(D, c) >= -1e-6).all()
        assert torch.allclose(dp.potential(torch.zeros(1, 3, 3), c[:1]), torch.zeros(1), atol=1e-6)


def test_viscosidad_relativa_mayor_que_uno():
    dp = DissipationPotential(8)
    g = torch.logspace(-1, 3, 100)
    eta = dp.relative_viscosity(g, torch.randn(100, 8))
    assert (eta >= 1.0).all()


def test_generic_degeneracion():
    cell = GenericCell(3)
    s = torch.randn(16, 3)
    L, M, gE, gS = cell.operators(s)
    assert torch.allclose(L, -L.transpose(-1, -2), atol=1e-5)
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-5)
    assert torch.linalg.eigvalsh(M).min() > -1e-5
    assert (L @ gS.unsqueeze(-1)).abs().max() < 1e-4             # L∇S = 0
    assert (M @ gE.unsqueeze(-1)).abs().max() < 1e-4             # M∇E = 0
    ds = cell(s)
    dEdt = (gE * ds).sum(-1)
    dSdt = (gS * ds).sum(-1)
    assert dEdt.abs().max() < 1e-4 and (dSdt >= -1e-5).all()


def arbol_binario(niveles=4, D0=30.0):
    """Árbol divergente que converge de nuevo (red cerrada entrada -> salida)."""
    edges, L, D = [], [], []
    nodes = 1
    frontier = [0]
    for lvl in range(niveles):
        nf = []
        for n in frontier:
            for k in range(2):
                edges.append((n, nodes)); L.append(200.0); D.append(D0 * (0.79 ** (lvl + 1)) * (1.0 + 0.15 * (k - 0.5)))
                nf.append(nodes); nodes += 1
        frontier = nf
    sink = nodes; nodes += 1
    for n in frontier:
        edges.append((n, sink)); L.append(200.0); D.append(D0 * 0.79 ** niveles)
    inlet = nodes; nodes += 1
    edges.insert(0, (inlet, 0)); L.insert(0, 300.0); D.insert(0, D0)
    return MicrovascularNetwork(nodes, edges, L, D, {inlet: 100.0, sink: 0.0}, {inlet: 0.45})


def test_red_conserva_masa_y_globulos():
    net = arbol_binario()
    res = net.solve(lambda D, H: pries_in_vitro_relative_viscosity(D, H), PhaseSeparation())
    assert res["mass_residual"].abs().max() < 1e-9 * res["Q"].abs().max()
    assert res["rbc_residual"].abs().max() < 1e-9 * res["rbc_flux"].abs().max()


def test_plasma_monotono_y_faltantes():
    pv = PlasmaViscosity()
    x = torch.zeros(1, len(pv.variables))
    obs = torch.ones_like(x, dtype=torch.bool)
    base = pv(x, obs)
    assert 1.0 < float(base.detach()) < 1.4
    for i, v in enumerate(pv.variables):
        up = x.clone(); up[0, i] = 2.0
        if v == "hdl":
            assert pv(up, obs) <= base
        else:
            assert pv(up, obs) >= base
    none = torch.zeros_like(obs)
    assert torch.allclose(pv(x + 5, none), base)                  # sin datos => valor base


def test_panel_tolera_faltantes():
    enc = PanelEncoder(32)
    ids = torch.tensor([[VAR_INDEX["hematocrit"], VAR_INDEX["fibrinogen"], VAR_INDEX["hba1c"]]] * 2)
    vals, dt = torch.randn(2, 3), torch.zeros(2, 3)
    mask = torch.tensor([[True, True, True], [False, False, False]])
    out = enc(ids, vals, dt, mask)
    assert torch.isfinite(out).all()


def test_modelo_extremo_a_extremo():
    ds = make_dataset(8, "dt2", n_points=64)
    m = EHO()
    out = m(ds, return_energy=True)
    assert out["fields"].shape == (8, 64, 2)
    assert (out["scalars"][:, 0] >= 0).all()                      # η_rel ≥ 1
    loss = out["fields"].pow(2).mean() + out["scalars"].pow(2).mean()
    loss.backward()


def test_curva_de_flujo_monotona_y_convexidad():
    dp = DissipationPotential(8)
    c = torch.randn(1, 8).repeat(200, 1)
    g = torch.logspace(-2, 3, 200).unsqueeze(-1)
    tau = dp.f(g, c)
    assert (tau.diff() >= -1e-6 * tau[1:].abs().max()).all()     # τ(γ̇) no decreciente
    # convexidad de Φ a lo largo de una recta en el espacio de D
    D1 = torch.randn(1, 3, 3); D1 = D1 + D1.transpose(-1, -2)
    D2 = torch.randn(1, 3, 3); D2 = D2 + D2.transpose(-1, -2)
    cc = torch.randn(1, 8)
    for t in torch.linspace(0.05, 0.95, 10):
        mix = dp.potential(t * D1 + (1 - t) * D2, cc)
        assert mix <= t * dp.potential(D1, cc) + (1 - t) * dp.potential(D2, cc) + 1e-5


def test_lyapunov_con_paso_de_entrenamiento():
    """Con el paso dt = 0.2 usado en entrenamiento, V(n) también decrece."""
    torch.manual_seed(1)
    moe = EcoMoE(8, 3, j_init=6, j_max=6)          # dt por defecto = 0.2
    viol = 0
    for _ in range(50):
        pi = torch.softmax(torch.randn(200, 6) * 3, -1)
        m2 = (pi ** 2).mean(0)
        A = (pi.T @ pi) / 200 / torch.sqrt(m2[:, None] * m2[None, :])
        A.fill_diagonal_(1.0)
        r = 1.0 + (torch.rand(6) * 2 - 1)
        moe.n = torch.rand(6) + 0.1
        V = moe.lotka_volterra(r, A, steps=20)
        viol += sum(b > a + 1e-6 for a, b in zip(V, V[1:]))
    assert viol == 0, f"{viol} pasos con V creciente"


def test_extincion_conserva_dos_especies():
    moe = EcoMoE(8, 3, j_init=4, j_max=4, patience=1)
    moe.n = torch.tensor([1.0, 1e-6, 1e-6, 1e-6])
    dead = moe.extinctions()
    assert int(moe.active.sum()) == 2 and len(dead) == 2
