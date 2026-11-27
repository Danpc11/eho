"""Memoria bayesiana de bajo rango (v0.10): lo que la memoria SABE de un paciente, con su incertidumbre.

Modelo. Para un paciente p con estado latente z_p ∈ R^r (lo que no se ve: parámetros celulares ocultos),

        y(x) = μ(x) + J(x) z_p + ε,        z_p ~ N(0, I),   ε ~ N(0, R),

donde μ(x) es la predicción del ensamble sin mediciones del paciente, J(x) ∈ R^{D×r} una matriz de
sensibilidad aprendida (función de los rasgos congelados de los agentes) y R = diag(σ²) el ruido.
Todo se expresa en unidades de la escala de los residuos, así que el prior N(0, I) no pierde generalidad.

Con k mediciones {(x_i, y_i)} y residuos e_i = y_i - μ(x_i), la posterior es gaussiana y cerrada:

        Σ_k = ( I + Σ_i J_iᵀ R⁻¹ J_i )⁻¹,        m_k = Σ_k Σ_i J_iᵀ R⁻¹ e_i,
        ŷ(x*) = μ(x*) + J* m_k,                  Cov[y(x*)] = J* Σ_k J*ᵀ + R.

Propiedades que se prueban (tests/test_v0100.py):
  1. Σ_k ⪯ Σ_{k-1}: ninguna medición aumenta la incertidumbre posterior (dentro del modelo).
  2. La actualización recursiva tipo Kalman coincide con la fórmula por lotes y no depende del orden.
  3. Sin mediciones se recupera la predicción base: no hay reentrenamiento ni pasos de gradiente.

Diseño de mediciones (qué medirle a un paciente):
  - D-óptimo: ganancia de información sobre z,  EIG(x) = ½ log det( I + Σ^{1/2} J_xᵀ R⁻¹ J_x Σ^{1/2} ).
  - Orientado a un objetivo G (p. ej. la predicción en las condiciones de interés x*), con g = J*:
        Δ_G(x) = tr[ J* Σ J_xᵀ (J_x Σ J_xᵀ + R)⁻¹ J_x Σ J*ᵀ ]      (reducción de varianza de G)
    y adquisición  a(x) = Δ_G(x) / C(x)^λ,  con C el costo de la medición.

Límites, dichos de entrada: es un modelo lineal-gaussiano sobre rasgos aprendidos. Σ mide la incertidumbre
DENTRO del modelo; si el modelo está mal especificado, medir más no garantiza menos error. Las coordenadas
de z no son rigidez ni agregación: no tienen significado físico sin supervisión que las identifique.
"""
import math

import torch
import torch.nn as nn

from .dominios.base import subset


class MemoriaBayesiana(nn.Module):
    def __init__(self, d_feat: int, n_out: int, r: int = 4, hidden: int = 64):
        super().__init__()
        self.r, self.n_out = r, n_out
        self.jnet = nn.Sequential(nn.Linear(d_feat, hidden), nn.GELU(), nn.Linear(hidden, n_out * r))
        self.log_sigma = nn.Parameter(torch.zeros(n_out))            # ruido por salida, en unidades escaladas
        self.register_buffer("f_mu", torch.zeros(d_feat))
        self.register_buffer("f_sd", torch.ones(d_feat))
        self.register_buffer("y_scale", torch.ones(n_out))

    # ------------------------------------------------------------------ piezas del modelo
    def J(self, feats):
        return self.jnet((feats - self.f_mu) / self.f_sd).view(-1, self.n_out, self.r)

    @property
    def R(self):
        return torch.exp(2 * self.log_sigma).clamp(min=1e-6)        # varianzas [D]

    def posterior(self, Jc, ec):
        """Jc [k, D, r], ec [k, D] (residuos escalados). Devuelve m [r], Σ [r, r]."""
        I = torch.eye(self.r, dtype=Jc.dtype, device=Jc.device)
        if Jc.shape[0] == 0:
            return torch.zeros(self.r, dtype=Jc.dtype, device=Jc.device), I
        Ri = 1.0 / self.R.to(Jc.dtype)
        A = I + torch.einsum("kdr,d,kds->rs", Jc, Ri, Jc)
        Sigma = torch.linalg.inv(A)
        m = Sigma @ torch.einsum("kdr,d,kd->r", Jc, Ri, ec)
        return m, 0.5 * (Sigma + Sigma.T)

    def kalman_update(self, m, Sigma, Jx, ex):
        """Una medición más (Jx [D, r], ex [D]): actualización recursiva, sin volver a los datos anteriores."""
        S = Jx @ Sigma @ Jx.T + torch.diag(self.R.to(Jx.dtype))
        K = Sigma @ Jx.T @ torch.linalg.inv(S)
        m2 = m + K @ (ex - Jx @ m)
        S2 = Sigma - K @ Jx @ Sigma
        return m2, 0.5 * (S2 + S2.T)

    def predictive(self, Jq, m, Sigma):
        """Media [B, D] y varianza marginal [B, D] de los residuos escalados en las consultas."""
        mean = torch.einsum("bdr,r->bd", Jq, m)
        var = torch.einsum("bdr,rs,bds->bd", Jq, Sigma, Jq) + self.R.to(Jq.dtype)
        return mean, var

    # ------------------------------------------------------------------ diseño de mediciones
    def eig(self, Sigma, Jx):
        """Ganancia de información esperada sobre z al medir en x (Jx [D, r]) dado Σ."""
        L = torch.linalg.cholesky(Sigma + 1e-9 * torch.eye(self.r, dtype=Sigma.dtype))
        M = L.T @ Jx.T @ torch.diag(1.0 / self.R.to(Jx.dtype)) @ Jx @ L
        return 0.5 * torch.logdet(torch.eye(self.r, dtype=Sigma.dtype) + M)

    def delta_G(self, Sigma, Jx, Jt):
        """Reducción de la varianza total del objetivo (Jt [T, D, r]) al medir en x (Jx [D, r])."""
        S = Jx @ Sigma @ Jx.T + torch.diag(self.R.to(Jx.dtype))
        C = torch.einsum("tdr,rs->tds", Jt, Sigma) @ Jx.T                     # [T, D, D]
        return torch.einsum("tde,ef,tdf->", C, torch.linalg.inv(S), C)

    def posterior_after(self, Sigma, Jx):
        """Σ tras medir en x: no depende del valor que se mida (modelo lineal-gaussiano)."""
        S = Jx @ Sigma @ Jx.T + torch.diag(self.R.to(Jx.dtype))
        S2 = Sigma - Sigma @ Jx.T @ torch.linalg.inv(S) @ Jx @ Sigma
        return 0.5 * (S2 + S2.T)


# ---------------------------------------------------------------------------------------------- rasgos
@torch.no_grad()
def base_and_features(agents, ds, chunk: int = 256):
    """μ(x): media de los agentes SIN contexto [B, D]; rasgos: estados globales concatenados [B, Σ d_a]."""
    mus, feats = [], []
    B = ds["episode"].numel()
    for i in range(0, B, chunk):
        part = subset(ds, torch.arange(i, min(i + chunk, B)))
        P, F = [], []
        for a in agents:
            a.eval()
            P.append(a(part)["scalars"].cpu())
            F.append(a.representations(part)["pooled"].cpu())
        mus.append(torch.stack(P).mean(0))
        feats.append(torch.cat(F, -1))
    return torch.cat(mus), torch.cat(feats)


def fit_memoria_bayes(agents, train, r: int = 4, steps: int = 400, lr: float = 3e-3, seed: int = 0,
                      k_max: int = 5, verbose: bool = False) -> MemoriaBayesiana:
    """Ajusta J y el ruido con los episodios de entrenamiento, sin tocar los agentes.
    Pérdida: verosimilitud predictiva gaussiana de las consultas de cada episodio dadas k mediciones
    al azar del mismo episodio (k entre 1 y k_max): se entrena exactamente lo que luego se usa."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    mu, feats = base_and_features(agents, train)
    y = train["scalars"].float()
    res = y - mu
    mb = MemoriaBayesiana(feats.shape[1], y.shape[1], r)
    mb.f_mu.copy_(feats.mean(0)); mb.f_sd.copy_(feats.std(0).clamp(min=1e-6))
    mb.y_scale.copy_(res.std(0).clamp(min=1e-6))
    e = res / mb.y_scale
    eps = train["episode"].unique()
    idx = {int(ep): torch.nonzero(train["episode"] == ep).flatten() for ep in eps}
    opt = torch.optim.Adam(mb.parameters(), lr=lr)
    for step in range(steps):
        loss, n = 0.0, 0
        for ep in eps[torch.randperm(eps.numel(), generator=g)[:16]].tolist():
            ids = idx[ep][torch.randperm(idx[ep].numel(), generator=g)]
            if ids.numel() < 2:
                continue
            k = int(torch.randint(1, min(k_max, ids.numel() - 1) + 1, (1,), generator=g))
            Jall = mb.J(feats[ids])
            m, S = mb.posterior(Jall[:k], e[ids[:k]])
            mean, var = mb.predictive(Jall[k:], m, S)
            loss = loss + (0.5 * ((e[ids[k:]] - mean) ** 2 / var + var.log())).mean()
            n += 1
        loss = loss / max(n, 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % 100 == 0:
            print(f"  memoria bayesiana, paso {step}: NLL {float(loss):.4f}", flush=True)
    return mb.eval()


@torch.no_grad()
def choose_measurements(mb: MemoriaBayesiana, J_pool, J_target=None, n: int = 2, cost=None, lam: float = 1.0,
                        criterion: str = "objetivo", Sigma=None, generator=None):
    """Elige n mediciones del conjunto candidato, una a una (voraz), sin necesitar sus valores.
    criterion: "objetivo" (Δ_G / C^λ; necesita J_target), "d_optimo" (EIG / C^λ) o "aleatorio".
    Devuelve los índices elegidos y la Σ resultante."""
    Sigma = torch.eye(mb.r, dtype=J_pool.dtype) if Sigma is None else Sigma
    cost = torch.ones(J_pool.shape[0]) if cost is None else torch.as_tensor(cost, dtype=torch.float32)
    libres, elegidos = list(range(J_pool.shape[0])), []
    for _ in range(min(n, len(libres))):
        if criterion == "aleatorio":
            j = libres[int(torch.randint(len(libres), (1,), generator=generator))]
        else:
            if criterion == "objetivo":
                if J_target is None:
                    raise ValueError("el criterio 'objetivo' necesita J_target")
                val = [float(mb.delta_G(Sigma, J_pool[i], J_target)) for i in libres]
            elif criterion == "d_optimo":
                val = [float(mb.eig(Sigma, J_pool[i])) for i in libres]
            else:
                raise ValueError(f"criterio desconocido: {criterion}")
            j = libres[max(range(len(libres)), key=lambda t: val[t] / float(cost[libres[t]]) ** lam)]
        elegidos.append(j)
        libres.remove(j)
        Sigma = mb.posterior_after(Sigma, J_pool[j])
    return elegidos, Sigma
