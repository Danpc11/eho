"""Dominio general de prueba: advección-difusión periódica en 1D.

    u_t + c u_x = κ u_xx,  x ∈ [0, 1) periódico.  Solución exacta por Fourier:
    û_k(T) = û_k(0) · exp(-κ (2πk)² T) · exp(-i 2πk c T)

Cada episodio es un FLUIDO con (κ, c) ocultos; sus muestras difieren en la condición
inicial u0 (visible en los tokens) y en el tiempo T (visible). Regímenes:
    difusivo  κ alto, c bajo      advectivo  κ bajo, c alto
    mixto     ambos intermedios   extremo    fuera de rango (para probar abstención)
"""
import math
import torch

from .base import Dominio, registrar_dominio

RANGES = {  # (κ_min, κ_max, c_min, c_max)
    "difusivo": (0.020, 0.040, 0.0, 0.5),
    "advectivo": (0.002, 0.008, 1.5, 3.0),
    "mixto": (0.008, 0.020, 0.5, 1.5),
    "extremo": (0.060, 0.090, 4.0, 5.0),
}


@registrar_dominio("edp1d")
class AdveccionDifusion1D(Dominio):
    name = "edp1d"
    d_point = 3
    d_cond = 1
    n_fields = 1
    n_scalars = 2
    expert = "generico"
    regimes = ["difusivo", "mixto", "advectivo"]

    def __init__(self, n_grid: int = 64, n_modes: int = 5, params_visible: bool = False, k_min: int = 1):
        """k_min: primer modo de Fourier de la condición inicial (modos k_min ... k_min + n_modes - 1).
        Con k_min > 1 las condiciones iniciales tienen frecuencias más altas: novedad VISIBLE en las
        entradas (útil para probar la elección de qué simular)."""
        self.n_grid, self.n_modes, self.params_visible, self.k_min = n_grid, n_modes, params_visible, k_min
        if params_visible:
            self.d_cond = 3

    def generate(self, n_episodes, per_episode, regime, seed=0):
        g = torch.Generator().manual_seed(seed)
        k0, k1, c0, c1 = RANGES[regime]
        n = n_episodes * per_episode
        kappa = (k0 + (k1 - k0) * torch.rand(n_episodes, generator=g)).repeat_interleave(per_episode)
        c = (c0 + (c1 - c0) * torch.rand(n_episodes, generator=g)).repeat_interleave(per_episode)
        T = 0.05 + 0.10 * torch.rand(n, generator=g)
        x = torch.arange(self.n_grid) / self.n_grid
        ks = torch.arange(self.k_min, self.k_min + self.n_modes).float()
        a = torch.randn(n, self.n_modes, generator=g) / ks
        b = torch.randn(n, self.n_modes, generator=g) / ks
        ph = 2 * math.pi * ks.view(1, -1, 1) * x.view(1, 1, -1)              # [1,K,N]
        u0 = (a.unsqueeze(-1) * torch.sin(ph) + b.unsqueeze(-1) * torch.cos(ph)).sum(1)
        decay = torch.exp(-kappa.view(-1, 1) * (2 * math.pi * ks) ** 2 * T.view(-1, 1))   # [n,K]
        shift = 2 * math.pi * ks * (c * T).view(-1, 1)                                     # [n,K]
        phs = ph - shift.unsqueeze(-1)
        uT = (decay.unsqueeze(-1) * (a.unsqueeze(-1) * torch.sin(phs) + b.unsqueeze(-1) * torch.cos(phs))).sum(1)
        pts = torch.stack([torch.cos(2 * math.pi * x).expand(n, -1), torch.sin(2 * math.pi * x).expand(n, -1), u0], -1)
        e0, eT = u0.pow(2).mean(-1).sqrt(), uT.pow(2).mean(-1).sqrt()
        scalars = torch.stack([(eT / e0).log(), uT.abs().max(-1).values], -1)
        vis = ((T - 0.1) / 0.03).unsqueeze(-1)
        cond = torch.cat([vis, ((kappa - 0.02) / 0.02).unsqueeze(-1), ((c - 1.5) / 1.0).unsqueeze(-1)], -1) \
            if self.params_visible else vis
        return {"points": pts, "cond": cond, "fields": uT.unsqueeze(-1), "scalars": scalars,
                "episode": torch.arange(n_episodes).repeat_interleave(per_episode),
                "kappa": kappa, "c": c, "T": T}

    def memory_values(self, batch):
        """Observables informativos: decaimiento (normalizado por k²) y desfase del modo dominante, más los escalares.
        Se calculan de las salidas medidas (u0 en los tokens, u(T) observado)."""
        u0 = batch["points"][..., 2]
        uT = batch["fields"][..., 0]
        F0, FT = torch.fft.rfft(u0, dim=-1), torch.fft.rfft(uT, dim=-1)
        k = F0[:, 1:].abs().argmax(-1) + 1                                  # modo dominante de u0
        f0 = F0.gather(1, k.unsqueeze(-1)).squeeze(-1)
        fT = FT.gather(1, k.unsqueeze(-1)).squeeze(-1)
        # log(|û_k(T)|/|û_k(0)|) / k² = -κ (2π)² T : no depende del modo elegido
        ratio = (fT.abs() / f0.abs().clamp(min=1e-6)).clamp(min=1e-6).log() / k.float() ** 2
        dphi = torch.angle(fT / f0)
        return torch.cat([batch["scalars"], torch.stack([ratio, torch.sin(dphi), torch.cos(dphi)], -1)], -1)
