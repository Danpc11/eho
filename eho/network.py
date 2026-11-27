"""Red microvascular con conservación exacta (sección 6).

- Kirchhoff: (B R⁻¹ Bᵀ) p = b, resuelto con un solver lineal diferenciable.
  La masa de plasma se conserva a precisión de máquina.
- Separación de fases en bifurcaciones divergentes: ley logística de Pries
  + corrección neuronal ψ_net (inicializada en cero); F_β = 1 - F_α,
  así el número de glóbulos rojos también se conserva exactamente.
- Acoplamiento hematocrito-resistencia por punto fijo.

Unidades: diámetros en μm, longitudes en μm, presiones arbitrarias.
"""
import math
import torch
import torch.nn as nn


def pries_in_vitro_relative_viscosity(D_um: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Efecto Fåhræus-Lindqvist, ajuste in vitro de Pries et al. (1992)."""
    eta45 = 220.0 * torch.exp(-1.3 * D_um) + 3.2 - 2.44 * torch.exp(-0.06 * D_um ** 0.645)
    sig = 1.0 / (1.0 + 1e-11 * D_um ** 12)
    C = (0.8 + torch.exp(-0.075 * D_um)) * (-1.0 + sig) + sig
    return 1.0 + (eta45 - 1.0) * ((1.0 - H) ** C - 1.0) / ((1.0 - 0.45) ** C - 1.0)


class PhaseSeparation(nn.Module):
    """Fracción de glóbulos que va a la rama α en una bifurcación divergente."""

    def __init__(self, d_theta: int = 0, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 + d_theta, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, FQ_a, D_a, D_b, D_f, H_f, theta=None):
        Df = D_f.clamp(min=1.0)
        X0 = (0.964 * (1 - H_f) / Df).clamp(max=0.45)
        A = -13.29 * ((D_a ** 2 / D_b ** 2 - 1) / (D_a ** 2 / D_b ** 2 + 1)) * (1 - H_f) / Df
        Bp = 1 + 6.98 * (1 - H_f) / Df
        x = ((FQ_a - X0) / (1 - 2 * X0)).clamp(1e-4, 1 - 1e-4)
        feats = torch.stack([FQ_a, D_a / Df, D_b / Df, H_f], -1)
        if theta is not None:
            feats = torch.cat([feats, theta.expand(feats.shape[0], -1)], -1)
        logit = A + Bp * torch.log(x / (1 - x)) + self.net(feats).squeeze(-1)
        F_a = torch.sigmoid(logit)
        # zonas de desnatado de plasma: sin glóbulos si FQ < X0, todos si FQ > 1 - X0
        F_a = torch.where(FQ_a <= X0, torch.zeros_like(F_a), F_a)
        F_a = torch.where(FQ_a >= 1 - X0, torch.ones_like(F_a), F_a)
        return F_a


class MicrovascularNetwork:
    """Grafo dirigido por construcción: edges[e] = (i, j)."""

    def __init__(self, n_nodes, edges, lengths_um, diams_um, p_boundary: dict, H_inlet: dict):
        self.n_nodes = n_nodes
        self.edges = torch.as_tensor(edges, dtype=torch.long)
        self.L = torch.as_tensor(lengths_um, dtype=torch.float64)
        self.D = torch.as_tensor(diams_um, dtype=torch.float64)
        self.pb = p_boundary                      # {nodo: presión}
        self.H_in = H_inlet                       # {nodo de entrada: hematocrito}
        E = self.edges.shape[0]
        self.Bmat = torch.zeros(n_nodes, E, dtype=torch.float64)
        self.Bmat[self.edges[:, 0], torch.arange(E)] = 1.0
        self.Bmat[self.edges[:, 1], torch.arange(E)] = -1.0

    def solve_flow(self, eta_rel: torch.Tensor):
        """Q_e de Poiseuille con Kirchhoff exacto. eta_rel [E]."""
        G = math.pi * self.D ** 4 / (128.0 * eta_rel * self.L)          # conductancia
        K = self.Bmat @ torch.diag(G) @ self.Bmat.T                       # laplaciano ponderado
        n = self.n_nodes
        bnodes = torch.tensor(sorted(self.pb), dtype=torch.long)
        inner = torch.tensor([i for i in range(n) if i not in self.pb], dtype=torch.long)
        p = torch.zeros(n, dtype=torch.float64)
        p[bnodes] = torch.tensor([self.pb[i] for i in sorted(self.pb)], dtype=torch.float64)
        rhs = -K[inner][:, bnodes] @ p[bnodes]
        p_in = torch.linalg.solve(K[inner][:, inner], rhs)
        p = p.index_put((inner,), p_in)
        Q = G * (self.Bmat.T @ p)                                         # Q_e = G_e (p_i - p_j)
        return p, Q

    def solve(self, viscosity_fn, phase_sep: PhaseSeparation, iters: int = 30, theta=None):
        """Punto fijo hematocrito ↔ resistencia. Devuelve dict con Q, H, p y residuos."""
        E = self.edges.shape[0]
        H = torch.full((E,), float(next(iter(self.H_in.values()))), dtype=torch.float64)
        for _ in range(iters):
            p, Q = self.solve_flow(viscosity_fn(self.D, H))
            H_new, rbc = self._propagate_hematocrit(Q, phase_sep, theta)
            if torch.max(torch.abs(H_new - H)) < 1e-10:
                H = H_new
                break
            H = 0.5 * H + 0.5 * H_new
        p, Q = self.solve_flow(viscosity_fn(self.D, H))
        H, rbc = self._propagate_hematocrit(Q, phase_sep, theta)
        return {"p": p, "Q": Q, "H": H, "rbc_flux": rbc,
                "mass_residual": self.node_residual(Q), "rbc_residual": self.node_residual(rbc)}

    def node_residual(self, flux):
        """Σ entradas - Σ salidas en nodos internos (debe ser ~0)."""
        inner = [i for i in range(self.n_nodes) if i not in self.pb]
        return (self.Bmat @ flux)[inner]

    def _propagate_hematocrit(self, Q, phase_sep, theta):
        E = self.edges.shape[0]
        src = torch.where(Q >= 0, self.edges[:, 0], self.edges[:, 1])
        dst = torch.where(Q >= 0, self.edges[:, 1], self.edges[:, 0])
        q = Q.abs()
        rbc = torch.zeros(E, dtype=torch.float64)
        done = torch.zeros(E, dtype=torch.bool)
        node_in = {i: [] for i in range(self.n_nodes)}
        node_out = {i: [] for i in range(self.n_nodes)}
        for e in range(E):
            node_out[int(src[e])].append(e)
            node_in[int(dst[e])].append(e)
        for _ in range(E + 1):
            progressed = False
            for node in range(self.n_nodes):
                outs = [e for e in node_out[node] if not done[e]]
                if not outs:
                    continue
                if node in self.H_in and not node_in[node]:
                    for e in outs:
                        rbc[e] = self.H_in[node] * q[e]
                        done[e] = True
                    progressed = True
                    continue
                if not all(done[e] for e in node_in[node]):
                    continue
                inflow_rbc = sum(rbc[e] for e in node_in[node])
                inflow_q = sum(q[e] for e in node_in[node])
                H_f = inflow_rbc / inflow_q.clamp(min=1e-30)
                if len(outs) == 1:
                    rbc[outs[0]] = inflow_rbc
                elif len(outs) == 2:
                    a, b = outs
                    FQ = q[a] / (q[a] + q[b])
                    D_f = max(float(self.D[e]) for e in node_in[node])
                    Fa = phase_sep(FQ.view(1).float(), self.D[a].view(1).float(), self.D[b].view(1).float(),
                                   torch.tensor([D_f]), H_f.view(1).float(), theta).double().squeeze()
                    rbc[a] = Fa * inflow_rbc
                    rbc[b] = (1 - Fa) * inflow_rbc                       # conservación exacta
                else:                                                     # >2 salidas: reparto por flujo
                    for e in outs:
                        rbc[e] = inflow_rbc * q[e] / sum(q[k] for k in outs)
                for e in outs:
                    done[e] = True
                progressed = True
            if bool(done.all()) or not progressed:
                break
        H = (rbc / q.clamp(min=1e-30)).clamp(0.0, 0.95)
        return H, rbc * torch.sign(Q + (Q == 0).double())
