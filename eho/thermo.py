"""Cierre termodinámico (sección 5).

1. DissipationPotential: Φ(D; c) = ψ(s(D); c) con ψ' = f ≥ 0 no decreciente
   (curva de flujo monótona aprendida). => Φ convexa en D, Φ(0) = 0 = min Φ,
   y D:∂Φ/∂D ≥ 0 (Proposición 3). En corte simple η_rel = 1 + f(γ̇)/γ̇ ≥ 1.

2. GenericCell: ṡ = L∇E + M∇S con L = -Lᵀ, M = Mᵀ ⪰ 0, L∇S = 0, M∇E = 0
   impuestos por construcción (proyectores), como en GFINN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneFlowCurve(nn.Module):
    """h(u; c) no decreciente en u = log γ̇, para cada contexto c.

    Pesos sobre la ruta de u no negativos y activaciones monótonas => monótona en u.
    El contexto solo entra en los sesgos, así que la monotonía vale para todo c.
    """

    def __init__(self, d_ctx: int, width: int = 32, depth: int = 2):
        super().__init__()
        self.wu = nn.Parameter(-1.0 + 0.5 * torch.rand(width))
        self.wz = nn.ParameterList([nn.Parameter(-3.0 + 0.5 * torch.rand(width, width)) for _ in range(depth - 1)])
        self.bias = nn.ModuleList([nn.Linear(d_ctx, width) for _ in range(depth)])
        self.out = nn.Parameter(-2.0 + 0.5 * torch.rand(width))
        self.out_u = nn.Parameter(torch.tensor(0.0))
        self.out_b = nn.Linear(d_ctx, 1)

    def forward(self, u, c):
        z = torch.tanh(F.softplus(self.wu) * u + self.bias[0](c))
        for k, W in enumerate(self.wz):
            z = torch.tanh(z @ F.softplus(W).T + self.bias[k + 1](c))
        return z @ F.softplus(self.out) + F.softplus(self.out_u) * u.squeeze(-1) + self.out_b(c).squeeze(-1)


class DissipationPotential(nn.Module):
    """Potencial de disipación Φ(D; c) = ∫_0^{s(D)} f(σ; c) dσ.

    f = exp(h(log σ; c)) ≥ 0 es el esfuerzo extra en función de la tasa de corte,
    y es no decreciente porque h es monótona. Por tanto:
      - ψ(s) = ∫_0^s f es convexa y no decreciente, ψ(0) = 0;
      - s(D) = sqrt(2 tr D² + ε²) - ε es convexa  =>  Φ = ψ∘s es convexa en D;
      - D : ∂Φ/∂D = f(s) · 2 tr(D²)/sqrt(2 tr D² + ε²) ≥ 0  (Proposición 3).
    En corte simple: η_rel = 1 + f(γ̇)/γ̇ ≥ 1, curva de flujo τ(γ̇) monótona.
    """

    def __init__(self, d_ctx: int, width: int = 32, depth: int = 2, eps: float = 1e-3, gamma_ref: float = 10.0):
        super().__init__()
        self.h = MonotoneFlowCurve(d_ctx, width, depth)
        self.eps, self.gamma_ref = eps, gamma_ref

    def f(self, s, c):
        """Esfuerzo extra f(s) ≥ 0, no decreciente; s: [B,1] en 1/s."""
        u = torch.log(s.clamp(min=1e-8) / self.gamma_ref)
        return torch.exp(self.h(u, c)) * (s.squeeze(-1) > 0).float()

    def norm_D(self, D):
        I2 = 2.0 * (D * D).sum(dim=(-1, -2))
        return (torch.sqrt(I2 + self.eps ** 2) - self.eps).unsqueeze(-1)

    def stress(self, D, c):
        """τ = f(s) ∂s/∂D  (analítico)."""
        I2 = 2.0 * (D * D).sum(dim=(-1, -2))
        ds = 2.0 * D / torch.sqrt(I2 + self.eps ** 2).view(-1, 1, 1)
        return self.f(self.norm_D(D), c).view(-1, 1, 1) * ds

    def potential(self, D, c, n_quad: int = 32):
        """Φ(D) por cuadratura de Gauss-Legendre en [0, s(D)] (para verificación)."""
        s = self.norm_D(D)                                                   # [B,1]
        x, w = torch.tensor(__import__("numpy").polynomial.legendre.leggauss(n_quad), dtype=D.dtype, device=D.device)
        sig = 0.5 * s * (x.view(1, -1) + 1)                                  # [B,Q]
        B, Q = sig.shape
        fv = self.f(sig.reshape(-1, 1), c.repeat_interleave(Q, 0)).view(B, Q)
        return 0.5 * s.squeeze(-1) * (fv * w.view(1, -1)).sum(-1)

    def relative_viscosity(self, gamma_dot, c):
        """η_rel = 1 + f(γ̇)/γ̇ ≥ 1; gamma_dot: [B] en 1/s."""
        g = gamma_dot.clamp(min=1e-6)
        return 1.0 + self.f(g.unsqueeze(-1), c) / g


def simple_shear(gamma_dot: torch.Tensor) -> torch.Tensor:
    D = torch.zeros(gamma_dot.shape[0], 3, 3, device=gamma_dot.device, dtype=gamma_dot.dtype)
    D[:, 0, 1] = D[:, 1, 0] = gamma_dot / 2
    return D


class GenericCell(nn.Module):
    """Evolución de estados internos s (p. ej. agregación λ_s) con estructura GENERIC."""

    def __init__(self, dim: int, d_ctx: int = 0, width: int = 32):
        super().__init__()
        self.E = nn.Sequential(nn.Linear(dim + d_ctx, width), nn.Tanh(), nn.Linear(width, 1))
        self.S = nn.Sequential(nn.Linear(dim + d_ctx, width), nn.Tanh(), nn.Linear(width, 1))
        self.A = nn.Sequential(nn.Linear(dim + d_ctx, width), nn.Tanh(), nn.Linear(width, dim * dim))
        self.Bm = nn.Sequential(nn.Linear(dim + d_ctx, width), nn.Tanh(), nn.Linear(width, dim * dim))
        self.dim = dim

    @staticmethod
    def _proj(v):  # I - v vᵀ/|v|²
        n = v.shape[-1]
        vv = v.unsqueeze(-1) @ v.unsqueeze(-2)
        return torch.eye(n, device=v.device) - vv / (v.pow(2).sum(-1, keepdim=True).unsqueeze(-1) + 1e-12)

    def operators(self, s, c=None):
        with torch.enable_grad():
            s = s if s.requires_grad else s.detach().requires_grad_(True)
            x = s if c is None else torch.cat([s, c], -1)
            E, S = self.E(x).sum(), self.S(x).sum()
            gE = torch.autograd.grad(E, s, create_graph=True)[0]
            gS = torch.autograd.grad(S, s, create_graph=True)[0]
        n = self.dim
        A = self.A(x).view(-1, n, n)
        Bm = self.Bm(x).view(-1, n, n)
        PS, PE = self._proj(gS), self._proj(gE)
        L = PS @ (A - A.transpose(-1, -2)) @ PS               # antisimétrica, L∇S = 0
        M = PE @ (Bm @ Bm.transpose(-1, -2)) @ PE             # simétrica PSD, M∇E = 0
        return L, M, gE, gS

    def forward(self, s, c=None):
        L, M, gE, gS = self.operators(s, c)
        return (L @ gE.unsqueeze(-1) + M @ gS.unsqueeze(-1)).squeeze(-1)
