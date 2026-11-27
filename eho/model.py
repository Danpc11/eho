"""EcoHopfield Operator (EHO): ensamblaje de los módulos.

partículas -> codificador -> rebanado -> núcleo de Hopfield (energía) ->
des-rebanado -> ecosistema de expertos (cierre termodinámico) -> campos y observables
                       ↘ memoria asociativa: corrección no paramétrica y novedad ν
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .slicing import PhysicsSlicing
from .hopfield import EnergyHopfieldCore, AssociativeMemory
from .ecology import EcoMoE
from . import to_device


class EHO(nn.Module):
    def __init__(self, d_point: int = 3, d_cond: int = 7, d: int = 48, n_slices: int = 12,
                 hopfield_steps: int = 3, j_init: int = 4, j_max: int = 12, top_k: int = 2,
                 n_fields: int = 2, n_scalars: int = 2, use_memory: bool = True,
                 use_ecology: bool = True, memory_capacity: int = 2048):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_point + d_cond, d), nn.GELU(), nn.Linear(d, d))
        self.slicing = PhysicsSlicing(d, n_slices)
        self.core = EnergyHopfieldCore(d, n_heads=4, d_head=12, n_patterns=32, steps=hopfield_steps, alpha=0.1)
        self.pool = nn.Sequential(nn.Linear(d + d_cond, d), nn.GELU(), nn.Linear(d, d))
        self.moe = EcoMoE(d, d_cond, n_fields=n_fields, j_init=j_init, j_max=j_max, top_k=top_k)
        self.memory = AssociativeMemory(d, n_scalars, capacity=memory_capacity, beta=8.0)
        self.mem_gate = nn.Parameter(torch.tensor(-1.0))
        self.use_memory, self.use_ecology = use_memory, use_ecology
        self.memory_correction = False      # se activa solo tras calibrar (ver train.calibrate_memory)

    def embed(self, points, cond, return_energy=False):
        B, N, _ = points.shape
        h = self.encoder(torch.cat([points, cond.unsqueeze(1).expand(B, N, -1)], -1))
        z, w = self.slicing(h)
        out = self.core(z, return_energy=return_energy)
        zT, energies = out if return_energy else (out, None)
        h_hat = self.slicing.deslice(zT, w) + h
        pooled = self.pool(torch.cat([self.core.norm(zT).mean(1), cond], -1))
        return h_hat, pooled, energies

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch, dense: bool = False, return_energy: bool = False):
        batch = to_device(batch, self.device)
        h_hat, pooled, energies = self.embed(batch["points"], batch["cond"], return_energy)
        fields, scal, pi, per_expert = self.moe(h_hat, pooled, batch["cond"], batch["gdot"], dense=dense)
        if self.use_memory and self.memory_correction and len(self.memory) > 0:
            corr, _ = self.memory.read(pooled)
            scal = scal + torch.sigmoid(self.mem_gate) * corr
        # log η_rel ≥ 0 se mantiene tras la corrección de memoria
        scal = torch.stack([scal[:, 0].clamp(min=0.0), scal[:, 1].clamp(0.0, 1.0)], -1)
        return {"fields": fields, "scalars": scal, "pi": pi, "pooled": pooled,
                "per_expert": per_expert, "energies": energies}

    @torch.no_grad()
    def novelty(self, batch):
        batch = to_device(batch, self.device)
        _, pooled, _ = self.embed(batch["points"], batch["cond"])
        return self.memory.novelty(pooled)


def sample_losses(out_fields, out_scal, batch):
    """Pérdida por muestra: L2 relativo de campos + error absoluto de escalares."""
    f, t = out_fields, batch["fields"].to(out_fields.device)
    rel = torch.linalg.vector_norm(f - t, dim=(1, 2)) / torch.linalg.vector_norm(t, dim=(1, 2)).clamp(min=1e-6)
    sc = (out_scal - batch["scalars"].to(out_scal.device)).abs().sum(-1)
    return rel + sc


def load_balance_loss(pi, top_k=2):
    """Pérdida auxiliar estándar (Switch Transformer), solo para la ablación sin ecología."""
    J = pi.shape[1]
    idx = pi.topk(top_k, -1).indices
    frac = F.one_hot(idx, J).float().sum(1).mean(0) / top_k
    return J * (frac * pi.mean(0)).sum()
