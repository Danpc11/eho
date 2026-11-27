"""Memoria neuronal de largo plazo al estilo Titans (Behrouz et al., NeurIPS 2025).

La memoria es una red pequeña M cuyos parámetros se actualizan EN TIEMPO DE USO,
sin el optimizador del modelo, minimizando la pérdida asociativa ℓ = ||M(k) - v||².

    sorpresa      g_t = ∇_M ℓ(M_{t-1}; k_t, v_t)
    momento       S_t = η S_{t-1} - θ g_t          (sorpresa pasada + momentánea)
    olvido        M_t = (1 - α) M_{t-1} + S_t      (decaimiento = olvido adaptable)

Lo sorprendente (gradiente grande) se graba más. Las proyecciones de clave y valor
las aprende el modelo externo por gradiente; aquí se usa la aproximación de primer
orden (no se retropropaga a través de las actualizaciones internas de M).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralLTM(nn.Module):
    def __init__(self, d_key: int, d_val: int, width: int = 64, eta: float = 0.9,
                 theta: float = 0.05, alpha: float = 0.001, max_grad: float = 10.0):
        super().__init__()
        self.eta, self.theta, self.alpha, self.max_grad = eta, theta, alpha, max_grad
        shapes = [(width, d_key), (width,), (d_val, width), (d_val,)]
        init = [torch.randn(s) * (0.5 / s[-1] ** 0.5) if len(s) == 2 else torch.zeros(s) for s in shapes]
        for i, t in enumerate(init):
            self.register_buffer(f"w{i}", t)
            self.register_buffer(f"s{i}", torch.zeros_like(t))
        self.register_buffer("n_writes", torch.zeros((), dtype=torch.long))
        self.last_surprise = 0.0

    def _params(self):
        return [getattr(self, f"w{i}") for i in range(4)]

    @staticmethod
    def _fwd(params, k):
        w0, b0, w1, b1 = params
        return F.silu(k @ w0.T + b0) @ w1.T + b1

    def read(self, q: torch.Tensor) -> torch.Tensor:
        """Lectura sin modificar la memoria. q: [B, d_key] -> [B, d_val]."""
        return self._fwd([p.detach() for p in self._params()], q)

    @torch.no_grad()
    def write(self, k: torch.Tensor, v: torch.Tensor, steps: int = 1) -> float:
        """Graba pares (k, v). Devuelve la sorpresa media (norma del gradiente)."""
        k, v = k.detach(), v.detach()
        surprise = 0.0
        for _ in range(steps):
            with torch.enable_grad():
                params = [p.clone().requires_grad_(True) for p in self._params()]
                loss = ((self._fwd(params, k) - v) ** 2).sum(-1).mean()
                grads = torch.autograd.grad(loss, params)
            surprise = float(sum(g.pow(2).sum() for g in grads).sqrt())
            if not torch.isfinite(torch.tensor(surprise)):
                break
            scale = min(1.0, self.max_grad / max(surprise, 1e-12))   # recorte: sorpresa acotada
            grads = [g * scale for g in grads]
            for i, g in enumerate(grads):
                S = getattr(self, f"s{i}")
                S.mul_(self.eta).add_(g, alpha=-self.theta)
                W = getattr(self, f"w{i}")
                W.mul_(1 - self.alpha).add_(S)
        self.n_writes += 1
        self.last_surprise = surprise
        return surprise

    @torch.no_grad()
    def surprise(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Error asociativo por muestra (sorpresa sin escribir): [B]."""
        return ((self.read(k) - v) ** 2).sum(-1)

    @torch.no_grad()
    def reset(self):
        for i in range(4):
            getattr(self, f"s{i}").zero_()
