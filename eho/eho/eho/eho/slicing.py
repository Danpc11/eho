"""Rebanado físico (sección 2 del documento).

Comprime N partículas en M "estados físicos" con costo O(N·M·d).
Sigue la idea de Transolver, pero sin atención entre rebanadas:
la interacción entre estados la hace el núcleo de Hopfield.
"""
import torch
import torch.nn as nn


class PhysicsSlicing(nn.Module):
    def __init__(self, d: int, n_slices: int, tau0: float = 0.5):
        super().__init__()
        self.q = nn.Linear(d, n_slices)                 # <q_m, h_i>
        self.log_tau = nn.Parameter(torch.tensor(float(tau0)).log())

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None):
        """h: [B, N, d]; mask: [B, N] (1 = partícula válida).

        Devuelve tokens z [B, M, d] y pesos w [B, N, M].
        """
        tau = self.log_tau.exp().clamp(min=1e-2)
        w = torch.softmax(self.q(h) / tau, dim=-1)       # w_im, suma 1 sobre m
        if mask is not None:
            w = w * mask.unsqueeze(-1)
        mass = w.sum(dim=1).clamp(min=1e-6)              # [B, M]
        z = torch.einsum("bnm,bnd->bmd", w, h) / mass.unsqueeze(-1)
        return z, w

    @staticmethod
    def deslice(z_hat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """ĥ_i = Σ_m w_im ẑ_m  ->  [B, N, d]."""
        return torch.einsum("bnm,bmd->bnd", w, z_hat)
