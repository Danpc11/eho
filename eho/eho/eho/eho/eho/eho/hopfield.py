"""Núcleo de Hopfield con energía y memoria asociativa (sección 3).

- g = ∇L(z) con L convexo (normalización de capa con γ escalar > 0),
  así H_L ⪰ 0 y la Proposición 1 (dE/dt ≤ 0) se cumple.
- E = E_att (atención entre tokens) + E_mem (memoria asociativa).
- Dinámica: z <- z - α ∂E/∂g, T pasos con pesos compartidos.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyLayerNorm(nn.Module):
    """g = γ (z - z̄)/sqrt(var + ε) + δ.

    Es el gradiente de L(z) = γ·sqrt(d)·sqrt(Σ_i (z_i - z̄)² + d·ε) + δ·z,
    que es convexo (norma de un mapa afín más un término lineal),
    por eso su jacobiano H_L es simétrico y PSD.
    γ es escalar y positivo (softplus) para preservar la convexidad.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.raw_gamma = nn.Parameter(torch.tensor(0.5413))  # softplus ≈ 1
        self.delta = nn.Parameter(torch.zeros(d))
        self.eps = eps

    @property
    def gamma(self):
        return F.softplus(self.raw_gamma)

    def forward(self, z):
        zc = z - z.mean(dim=-1, keepdim=True)
        var = (zc ** 2).mean(dim=-1, keepdim=True)
        return self.gamma * zc / torch.sqrt(var + self.eps) + self.delta


class AssociativeMemory(nn.Module):
    """Memoria clave-valor de simulaciones y fósiles.

    - read(q): regresión de Nadaraya-Watson V softmax(β K q).
    - novelty(q): ν = -(1/β) log Σ_k exp(β ξ_k·q)  (alta = no visto).
    - write / consolidate: escritura selectiva y fusión por k-medias.
    """

    def __init__(self, d_key: int, d_val: int, capacity: int = 4096, beta: float = 4.0):
        super().__init__()
        self.capacity, self.beta = capacity, beta
        self.register_buffer("keys", torch.zeros(0, d_key))
        self.register_buffer("vals", torch.zeros(0, d_val))
        self.register_buffer("tags", torch.zeros(0, dtype=torch.long))  # 0 = simulación, 1 = fósil

    def __len__(self):
        return self.keys.shape[0]

    def _norm(self, x):
        return F.normalize(x, dim=-1)

    def read(self, q):
        """q: [B, d] -> (valores [B, d_val], pesos [B, K]) ; ceros si vacía."""
        if len(self) == 0:
            return q.new_zeros(q.shape[0], self.vals.shape[1]), None
        a = torch.softmax(self.beta * self._norm(q) @ self._norm(self.keys).T, dim=-1)
        return a @ self.vals, a

    def novelty(self, q):
        if len(self) == 0:
            return q.new_full((q.shape[0],), float("inf"))
        s = self.beta * self._norm(q) @ self._norm(self.keys).T
        # normalizado para que ν ∈ [-1, 1 + log(K)/β]: 1 - max-sim aprox.
        return -(torch.logsumexp(s, dim=-1) - math.log(len(self))) / self.beta

    @torch.no_grad()
    def write(self, k, v, tag: int = 0):
        k, v = k.detach(), v.detach()
        self.keys = torch.cat([self.keys, k])
        self.vals = torch.cat([self.vals, v])
        self.tags = torch.cat([self.tags, torch.full((k.shape[0],), tag, dtype=torch.long, device=k.device)])
        if len(self) > self.capacity:
            self.consolidate(self.capacity // 2)

    @torch.no_grad()
    def consolidate(self, n_keep: int, iters: int = 10):
        """Fusiona claves cercanas (k-medias); los fósiles se conservan intactos."""
        fossil = self.tags == 1
        k, v = self.keys[~fossil], self.vals[~fossil]
        n_keep = max(1, min(n_keep - int(fossil.sum()), k.shape[0]))
        c_idx = torch.randperm(k.shape[0], device=k.device)[:n_keep]
        ck, cv = k[c_idx].clone(), v[c_idx].clone()
        for _ in range(iters):
            assign = torch.cdist(k, ck).argmin(dim=1)
            for j in range(n_keep):
                sel = assign == j
                if sel.any():
                    ck[j], cv[j] = k[sel].mean(0), v[sel].mean(0)
        self.keys = torch.cat([ck, self.keys[fossil]])
        self.vals = torch.cat([cv, self.vals[fossil]])
        self.tags = torch.cat([torch.zeros(n_keep, dtype=torch.long, device=k.device), self.tags[fossil]])


class EnergyHopfieldCore(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, d_head: int = 16, n_patterns: int = 64,
                 steps: int = 4, alpha: float = 0.1, beta: float | None = None, beta_xi: float = 1.0):
        super().__init__()
        self.norm = EnergyLayerNorm(d)
        self.WQ = nn.Parameter(torch.randn(n_heads, d_head, d) / math.sqrt(d))
        self.WK = nn.Parameter(torch.randn(n_heads, d_head, d) / math.sqrt(d))
        self.Xi = nn.Parameter(torch.randn(n_patterns, d) / math.sqrt(d))  # patrones aprendidos
        self.steps, self.alpha = steps, alpha
        self.beta = beta if beta is not None else 1.0 / math.sqrt(d_head)
        self.beta_xi = beta_xi
        self.extra_keys: torch.Tensor | None = None  # claves de la memoria externa (opcional)

    def energy(self, g: torch.Tensor) -> torch.Tensor:
        """E(g) por muestra: [B]."""
        B, M, _ = g.shape
        q = torch.einsum("hkd,bmd->bhmk", self.WQ, g)
        k = torch.einsum("hkd,bmd->bhmk", self.WK, g)
        s = self.beta * q @ k.transpose(-1, -2)                       # [B,H,M,M]
        eye = torch.eye(M, dtype=torch.bool, device=g.device)
        s = s.masked_fill(eye, float("-inf"))                         # m' ≠ m
        e_att = -(1.0 / self.beta) * torch.logsumexp(s, dim=-1).sum(dim=(1, 2))
        xi = self.Xi if self.extra_keys is None else torch.cat([self.Xi, self.extra_keys])
        e_mem = -(1.0 / self.beta_xi) * torch.logsumexp(self.beta_xi * g @ xi.T, dim=-1).sum(dim=1)
        return e_att + e_mem

    step_mode = "fixed"          # "fixed" (v0.1-v0.9.6) | "armijo" (v0.9.7, solo en evaluación)

    @torch.no_grad()
    def _armijo_step(self, z, v, E0, rho: float = 0.5, max_halvings: int = 20):
        """Paso con descenso de energía GARANTIZADO por muestra (búsqueda por retroceso).

        La energía decrece en tiempo continuo; con paso fijo α no hay cota. Aquí, para cada muestra, se
        prueba α, α ρ, α ρ², ... y se acepta el primer paso con E(z - α_j v) ≤ E(z). Si ninguno lo cumple
        tras max_halvings reducciones, la muestra no se mueve. Así E_{t+1} ≤ E_t se cumple SIEMPRE por
        construcción (propiedad verificable), a costa de evaluaciones extra de la energía. Solo en
        evaluación: en entrenamiento se usa el paso fijo, que es diferenciable de extremo a extremo."""
        alpha = torch.full((z.shape[0],) + (1,) * (z.dim() - 1), float(self.alpha), device=z.device, dtype=z.dtype)
        z_new = z - alpha * v
        hecho = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
        for _ in range(max_halvings + 1):
            E1 = self.energy(self.norm(z_new))
            ok = E1 <= E0
            hecho = hecho | ok
            if bool(hecho.all()):
                break
            alpha = torch.where(hecho.view(alpha.shape), alpha, alpha * rho)
            z_new = torch.where(hecho.view(alpha.shape), z_new, z - alpha * v)
        z_new = torch.where(hecho.view(alpha.shape), z_new, z)             # sin descenso posible: no moverse
        return z_new.detach().requires_grad_(True)

    def forward(self, z: torch.Tensor, return_energy: bool = False):
        energies = []
        create = self.training
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            for _ in range(self.steps):
                g = self.norm(z)
                E = self.energy(g)
                (dEdg,) = torch.autograd.grad(E.sum(), g, create_graph=create)
                if return_energy:
                    energies.append(E.detach())
                if getattr(self, "step_mode", "fixed") == "armijo" and not create:
                    z = self._armijo_step(z, dEdg, E)
                else:
                    z = z - self.alpha * dEdg
            if return_energy:
                energies.append(self.energy(self.norm(z)).detach())
        if not create:
            z = z.detach()
        return (z, torch.stack(energies, dim=1)) if return_energy else z
