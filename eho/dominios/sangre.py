"""Dominio sangre: cada episodio es un PACIENTE con mecánica de glóbulos rojos θ desconocida.

Visibles: hematocrito H (del paciente), diámetro D y tasa de corte γ̇ (de la medición).
Ocultos: θ = (rigidez, agregación, volumen, plasma). El modelo debe inferirlos con pocos
ejemplos del mismo paciente, como ocurriría con unas pocas mediciones de laboratorio.
"""
import math
import torch

from .base import Dominio, registrar_dominio
from ..data.synthetic import sample_theta, ground_truth


@registrar_dominio("sangre")
class Sangre(Dominio):
    name = "sangre"
    d_point = 3
    d_cond = 3
    n_fields = 2
    n_scalars = 2
    expert = "sangre"
    extra_key = "gdot"
    regimes = ["sano", "prediabetes", "dt2"]

    def __init__(self, theta_visible: bool = False, n_points: int = 96, simulador: str = "quemada",
                 D_range=(6.0, 60.0), H_range=(0.25, 0.50)):
        """simulador: "quemada" | "carreau" | "real" (ver data/synthetic.py).
        D_range: rango de diámetros [μm]; fuera de (6, 60) es una geometría nueva.
        H_range: rango de hematocrito; (0.55, 0.70) representa policitemia (hiperviscosidad)."""
        self.theta_visible = theta_visible
        self.n_points = n_points
        self.simulador = simulador
        self.D_range = D_range
        self.H_range = H_range
        if theta_visible:
            self.d_cond = 7

    def generate(self, n_episodes, per_episode, regime, seed=0):
        torch.manual_seed(seed)
        theta = sample_theta(n_episodes, regime).repeat_interleave(per_episode, 0)
        h0, h1 = self.H_range
        H = (h0 + (h1 - h0) * torch.rand(n_episodes)).repeat_interleave(per_episode)
        n = n_episodes * per_episode
        d0, d1 = self.D_range
        D = torch.exp(math.log(d0) + (math.log(d1) - math.log(d0)) * torch.rand(n))
        g = torch.exp(math.log(500.0) * torch.rand(n))
        pts, f, s = ground_truth(theta, H, D, g, self.n_points, simulador=self.simulador)
        vis = torch.stack([(H - 0.375) / 0.07, (D.log() - math.log(19.0)) / 0.7, (g.log() - math.log(22.0)) / 1.8], -1)
        cond = torch.cat([theta, vis], -1) if self.theta_visible else vis
        return {"points": pts, "cond": cond, "fields": f, "scalars": s, "gdot": g,
                "episode": torch.arange(n_episodes).repeat_interleave(per_episode), "theta": theta}

    def postprocess(self, scalars):
        return torch.stack([scalars[:, 0].clamp(min=0.0), scalars[:, 1].clamp(0.0, 1.0)], -1)
