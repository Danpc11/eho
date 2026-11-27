"""Inferencia inversa por flow matching (sección 7).

θ_t = (1-t) θ_0 + t θ_1,  θ_0 ~ N(0, I),  L = E || v(θ_t, t, x) - (θ_1 - θ_0) ||².
Muestreo integrando dθ/dt = v(θ, t, x) de t=0 a 1. Calibración con SBC.
"""
import torch
import torch.nn as nn


class FlowMatchingPosterior(nn.Module):
    def __init__(self, d_theta: int, d_x: int, width: int = 128, depth: int = 4):
        super().__init__()
        layers, d_in = [], d_theta + d_x + 1
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.SiLU()]
            d_in = width
        layers.append(nn.Linear(width, d_theta))
        self.v = nn.Sequential(*layers)
        self.d_theta = d_theta

    def loss(self, theta1, x):
        t = torch.rand(theta1.shape[0], 1, device=theta1.device)
        theta0 = torch.randn_like(theta1)
        theta_t = (1 - t) * theta0 + t * theta1
        target = theta1 - theta0
        return ((self.v(torch.cat([theta_t, t, x], -1)) - target) ** 2).mean()

    @torch.no_grad()
    def sample(self, x, n_samples: int = 256, steps: int = 64):
        """x: [d_x] o [1, d_x] -> muestras [n_samples, d_theta] (RK2/Heun)."""
        x = x.view(1, -1).expand(n_samples, -1)
        th = torch.randn(n_samples, self.d_theta, device=x.device)
        h = 1.0 / steps
        for i in range(steps):
            t = torch.full((n_samples, 1), i * h, device=x.device)
            k1 = self.v(torch.cat([th, t, x], -1))
            k2 = self.v(torch.cat([th + h * k1, t + h, x], -1))
            th = th + 0.5 * h * (k1 + k2)
        return th


@torch.no_grad()
def sbc_ranks(posterior, prior_sampler, simulator, n_trials: int = 200, n_samples: int = 99):
    """Calibración basada en simulación: rangos uniformes <=> posterior calibrada."""
    ranks = []
    for _ in range(n_trials):
        th = prior_sampler(1)
        x = simulator(th)
        s = posterior.sample(x, n_samples)
        ranks.append((s < th).sum(0))
    return torch.stack(ranks)            # [n_trials, d_theta], valores en 0..n_samples
