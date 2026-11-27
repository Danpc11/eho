"""Dominio Burgers viscoso 1D con un resolvedor numérico real (v0.7).

    u_t + u u_x = ν u_xx,   x ∈ [0, 1) periódico

A diferencia de edp1d (solución exacta por Fourier, lineal), aquí la física es no lineal: se forman
frentes empinados cuyo ancho depende de ν, y el operador no tiene forma cerrada. Se resuelve con un
método pseudoespectral con factor integrante y Runge-Kutta 4 (regla de 2/3 contra el aliasing),
en lotes, en CPU o GPU. Las condiciones iniciales siguen la convención de PDEBench (suma de senos con
amplitudes y fases aleatorias), para poder usar luego los archivos de PDEBench como prueba externa
con eho/dominios/pdebench.py.

Cada episodio es un FLUIDO con viscosidad ν oculta; sus muestras difieren en la condición inicial
(visible en los tokens) y en el tiempo T (visible). Regímenes:
    viscoso      ν ∈ [0.020, 0.050]
    intermedio   ν ∈ [0.008, 0.020]
    poco_viscoso ν ∈ [0.004, 0.008]
    extremo      ν ∈ [0.0015, 0.003]   cambio OCULTO: frentes casi de choque (prueba de abstención)
Cambio VISIBLE: k_min > 1 (condiciones iniciales de mayor frecuencia) o amplitud mayor.
"""
import math

import torch

from .base import Dominio, registrar_dominio

RANGES = {"viscoso": (0.020, 0.050), "intermedio": (0.008, 0.020), "poco_viscoso": (0.004, 0.008),
          "extremo": (0.0015, 0.003)}


def solve_burgers(u0: torch.Tensor, nu: torch.Tensor, T: torch.Tensor, dt: float | None = None) -> torch.Tensor:
    """u0 [B, N] en malla uniforme de [0,1); nu, T [B]. Devuelve u(T) [B, N].

    Factor integrante: v̂ = e^{ν k² t} û, de modo que la parte difusiva (rígida) se integra exacta y
    RK4 solo integra el término no lineal -½ ∂x(u²). Cada muestra se detiene en su propio T
    (el último paso se recorta para caer exactamente en T)."""
    B, N = u0.shape
    dev, dt_ = u0.device, torch.float64
    k = 2 * math.pi * torch.fft.rfftfreq(N, d=1.0 / N).to(dev, dt_)              # números de onda
    dealias = (torch.arange(k.numel(), device=dev) < (N // 3)).to(dt_)
    umax = u0.abs().max().item()
    h = dt or min(0.4 / (N * max(umax, 1e-3)), 2e-3)
    nu, T = nu.to(dev, dt_).view(B, 1), T.to(dev, dt_).view(B, 1)
    uh = torch.fft.rfft(u0.to(dt_), dim=-1)
    lam = -nu * k ** 2                                                           # [B, K]

    def nonlin(vh):
        u = torch.fft.irfft(vh * dealias, n=N, dim=-1)
        return -0.5j * k * torch.fft.rfft(u * u, dim=-1)

    t = torch.zeros(B, 1, device=dev, dtype=dt_)
    while bool((t < T - 1e-12).any()):
        step = torch.clamp(T - t, min=0.0, max=h)                                  # [B,1], 0 si ya terminó
        E, E2 = torch.exp(lam * step), torch.exp(lam * step / 2)
        k1 = nonlin(uh)
        k2 = nonlin(E2 * (uh + 0.5 * step * k1))
        k3 = nonlin(E2 * uh + 0.5 * step * k2)
        k4 = nonlin(E * uh + step * E2 * k3)
        uh = E * uh + step / 6 * (E * k1 + 2 * E2 * (k2 + k3) + k4)
        t = t + step
    return torch.fft.irfft(uh, n=N, dim=-1).to(u0.dtype)


def random_ic(n, n_grid, n_modes=4, k_min=1, amp=1.0, generator=None):
    """Suma de senos (convención de PDEBench), normalizada a máx |u0| = amp."""
    x = torch.arange(n_grid) / n_grid
    ks = torch.randint(k_min, k_min + n_modes, (n, n_modes), generator=generator).float()
    a = torch.rand(n, n_modes, generator=generator) * 2 - 1
    ph = 2 * math.pi * torch.rand(n, n_modes, generator=generator)
    u0 = (a.unsqueeze(-1) * torch.sin(2 * math.pi * ks.unsqueeze(-1) * x + ph.unsqueeze(-1))).sum(1)
    return amp * u0 / u0.abs().max(-1, keepdim=True).values.clamp(min=1e-6)


@registrar_dominio("burgers1d")
class Burgers1D(Dominio):
    name = "burgers1d"
    d_point = 3
    d_cond = 1
    n_fields = 1
    n_scalars = 2
    expert = "generico"
    regimes = ["viscoso", "intermedio", "poco_viscoso"]

    def __init__(self, n_grid: int = 64, n_solve: int = 512, n_modes: int = 4, k_min: int = 1, amp: float = 1.0,
                 T_range=(0.05, 0.30), params_visible: bool = False):
        """n_grid: tokens que ve el modelo; n_solve: malla del resolvedor (múltiplo de n_grid)."""
        self.n_grid, self.n_solve, self.n_modes, self.k_min, self.amp = n_grid, n_solve, n_modes, k_min, amp
        self.T_range, self.params_visible = T_range, params_visible
        if params_visible:
            self.d_cond = 2

    def generate(self, n_episodes, per_episode, regime, seed=0):
        g = torch.Generator().manual_seed(seed)
        n0, n1 = RANGES[regime]
        n = n_episodes * per_episode
        lognu = math.log(n0) + (math.log(n1) - math.log(n0)) * torch.rand(n_episodes, generator=g)
        nu = lognu.exp().repeat_interleave(per_episode)
        T = self.T_range[0] + (self.T_range[1] - self.T_range[0]) * torch.rand(n, generator=g)
        u0f = random_ic(n, self.n_solve, self.n_modes, self.k_min, self.amp, g)
        uTf = solve_burgers(u0f, nu, T)
        s = self.n_solve // self.n_grid
        u0, uT = u0f[:, ::s], uTf[:, ::s]
        x = torch.arange(self.n_grid) / self.n_grid
        pts = torch.stack([torch.cos(2 * math.pi * x).expand(n, -1), torch.sin(2 * math.pi * x).expand(n, -1), u0], -1)
        e0, eT = u0f.pow(2).mean(-1).sqrt(), uTf.pow(2).mean(-1).sqrt()
        grad = (torch.roll(uTf, -1, -1) - uTf).abs().max(-1).values * self.n_solve   # pendiente máxima
        scalars = torch.stack([(eT / e0).clamp(min=1e-6).log(), grad.clamp(min=1e-3).log()], -1)
        vis = ((T - 0.175) / 0.07).unsqueeze(-1)
        cond = torch.cat([vis, ((nu.log() - math.log(0.01)) / 1.0).unsqueeze(-1)], -1) if self.params_visible else vis
        return {"points": pts, "cond": cond, "fields": uT.unsqueeze(-1), "scalars": scalars,
                "episode": torch.arange(n_episodes).repeat_interleave(per_episode), "nu": nu, "T": T}

    def memory_values(self, batch):
        """Escalares + tasa de disipación observada (log E_T/E_0 dividido entre T), que informa de ν."""
        T = batch["cond"][:, 0] * 0.07 + 0.175
        rate = batch["scalars"][:, 0] / T.clamp(min=1e-3)
        return torch.cat([batch["scalars"], rate.unsqueeze(-1)], -1)
