"""Ecosistema de expertos (secciones 4 y "Cómo evoluciona").

- Nicho (qué datos atiende y con qué se entrena cada experto): solo afinidad,
  π^aff_j ∝ exp(β_r <k_j, ḡ>). Si la abundancia decidiera el nicho, el dominante
  acapararía los datos y habría exclusión competitiva (Gause): lo observamos.
- Mezcla de salidas entre los top-k del nicho: ∝ n_j^λ π^aff_j (λ = 0.5).
- Escala lenta: Lotka-Volterra  dn_j/dt = n_j (r_j - Σ_l α_jl n_l)
  con aptitud r_j (ventaja donde es responsable) y solapamiento de nicho α
  simétrico => V(n) = -r·n + ½ nᵀAn es función de Lyapunov (Proposición 2).
- Extinción (n_j < ε durante varias épocas) -> fósil en la memoria.
- Especiación: mutante admitido si su aptitud de invasión Λ = r' - Σ α n* > 0.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .thermo import DissipationPotential


class Expert(nn.Module):
    """Cabeza de operador: campos por punto + viscosidad termodinámicamente consistente."""

    def __init__(self, d: int, d_cond: int, n_fields: int = 2, width: int = 64):
        super().__init__()
        self.field = nn.Sequential(nn.Linear(d + d_cond, width), nn.GELU(),
                                   nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_fields))
        self.ctx = nn.Sequential(nn.Linear(d + d_cond, width), nn.GELU(), nn.Linear(width, 16))
        self.visc = DissipationPotential(d_ctx=16)
        self.cfl = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, h_points, pooled, cond, gamma_dot):
        """h_points [B,N,d]; pooled [B,d]; cond [B,dc]; gamma_dot [B] en 1/s."""
        B, N, _ = h_points.shape
        fields = self.field(torch.cat([h_points, cond.unsqueeze(1).expand(B, N, -1)], -1))
        c = self.ctx(torch.cat([pooled, cond], -1))
        eta_rel = self.visc.relative_viscosity(gamma_dot, c)           # ≥ 1
        cfl = torch.sigmoid(self.cfl(c)).squeeze(-1)                   # δ/R ∈ (0,1)
        return fields, torch.stack([eta_rel.log(), cfl], -1)


class GenericExpert(nn.Module):
    """Experto de propósito general: salida por token (campos) y salida global (escalares)."""

    def __init__(self, d: int, d_cond: int, n_fields: int, n_scalars: int, width: int = 64):
        super().__init__()
        self.field = nn.Sequential(nn.Linear(2 * d + d_cond, width), nn.GELU(),
                                   nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_fields))
        self.scalar = nn.Sequential(nn.Linear(d + d_cond, width), nn.GELU(), nn.Linear(width, n_scalars))

    def forward(self, h_points, pooled, cond, extra=None):
        B, N, _ = h_points.shape
        ctx = torch.cat([pooled, cond], -1)
        fields = self.field(torch.cat([h_points, ctx.unsqueeze(1).expand(B, N, -1)], -1))
        return fields, self.scalar(ctx)


class EcoMoE(nn.Module):
    def __init__(self, d: int, d_cond: int, n_fields: int = 2, j_init: int = 4, j_max: int = 16,
                 top_k: int = 2, beta_r: float = 5.0, r0: float = 1.0, eta: float = 1.0,
                 eps_ext: float = 0.05, patience: int = 3, dt: float = 0.2, expert_factory=None):
        """expert_factory(): crea un experto con firma f(h_points, pooled, cond, extra) -> (campos, escalares).
        Por defecto, el experto de sangre con viscosidad termodinámica (extra = γ̇)."""
        super().__init__()
        self.d, self.j_max, self.top_k, self.beta_r = d, j_max, top_k, beta_r
        self.r0, self.eta, self.eps_ext, self.patience, self.dt = r0, eta, eps_ext, patience, dt
        factory = expert_factory or (lambda: Expert(d, d_cond, n_fields))
        self.experts = nn.ModuleList([factory() for _ in range(j_max)])
        self.keys = nn.Parameter(F.normalize(torch.randn(j_max, d), dim=-1))
        self.register_buffer("center", torch.zeros(d))                 # media móvil de ḡ
        self.register_buffer("n", torch.zeros(j_max))
        self.register_buffer("active", torch.zeros(j_max, dtype=torch.bool))
        self.register_buffer("low_count", torch.zeros(j_max, dtype=torch.long))
        self.n[:j_init] = 1.0 / j_init
        self.active[:j_init] = True
        self.history: list[dict] = []

    # ---------- enrutamiento ----------
    def routing(self, pooled):
        """Responsabilidad por nicho (solo afinidad): π^aff_j(x), [B,J].

        Se compara ḡ - media móvil con las claves: sin centrar, todas las muestras
        apuntan casi en la misma dirección, el nicho no depende de la muestra y la
        competencia es total (α ≈ 1), lo que lleva a exclusión competitiva.
        """
        if self.training:
            with torch.no_grad():
                self.center.mul_(0.99).add_(0.01 * pooled.mean(0))
        logits = self.beta_r * F.normalize(pooled - self.center, dim=-1) @ F.normalize(self.keys, dim=-1).T
        logits = logits.masked_fill(~self.active, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def forward(self, h_points, pooled, cond, extra=None, dense: bool = False, lam: float = 0.5):
        pi = self.routing(pooled)
        k = min(self.top_k, int(self.active.sum()))
        top_v, top_i = pi.topk(k, dim=-1)
        w = top_v * self.n.clamp(min=1e-6)[top_i] ** lam                # abundancia solo en la mezcla
        w = w / w.sum(-1, keepdim=True)
        B = pooled.shape[0]
        fields, scal = 0.0, 0.0
        per_expert = {}
        idx_set = torch.nonzero(self.active).flatten().tolist() if dense else sorted(set(top_i.flatten().tolist()))
        for j in idx_set:
            f_j, s_j = self.experts[j](h_points, pooled, cond, extra)
            per_expert[j] = (f_j, s_j)
            wj = (w * (top_i == j)).sum(-1)                            # [B]
            fields = fields + wj.view(B, 1, 1) * f_j
            scal = scal + wj.view(B, 1) * s_j
        return fields, scal, pi, per_expert

    # ---------- ecología (escala lenta) ----------
    @torch.no_grad()
    def ecology_stats(self, pi, loss_mix, loss_per_expert):
        """pi [B,J] (responsabilidad de nicho); loss_per_expert {j: [B]} -> r [J], A [J,J].

        Aptitud en el propio nicho: L_j = E[π_j ℓ_j]/E[π_j];
        r_j = r0 + η (L_ref - L_j)/L_ref,  L_ref = promedio de L_j entre especies activas.
        No depende de la abundancia (evita el círculo "el dominante define la referencia").
        La competencia entra solo por el solapamiento de nicho α.
        """
        J = self.j_max
        r = torch.zeros(J, device=pi.device)
        Lj = {}
        for j, lj in loss_per_expert.items():
            mass = pi[:, j].mean().clamp(min=1e-8)
            Lj[j] = (pi[:, j] * lj).mean() / mass
        ref = torch.stack(list(Lj.values())).mean().clamp(min=1e-8)
        for j, L in Lj.items():
            r[j] = self.r0 + self.eta * ((ref - L) / ref).clamp(-1.0, 1.0)
        m2 = (pi ** 2).mean(0).clamp(min=1e-12)
        A = (pi.T @ pi) / pi.shape[0] / torch.sqrt(m2.unsqueeze(0) * m2.unsqueeze(1))
        A = 0.5 * (A + A.T)                                           # simetría garantizada
        A.fill_diagonal_(1.0)
        act = self.active.float()
        A = A * act.unsqueeze(0) * act.unsqueeze(1)
        return r * act, A

    @staticmethod
    def lyapunov(n, r, A):
        return -(r * n).sum() + 0.5 * n @ A @ n

    @torch.no_grad()
    def lotka_volterra(self, r, A, steps: int = 20):
        """Integra dn/dt = n(r - An) con Euler en log n (positividad garantizada).

        El piso 1e-6 evita el subdesbordamiento; la extinción la decide la cuota
        relativa (extinctions), no el valor absoluto.
        """
        V = []
        for _ in range(steps):
            V.append(float(self.lyapunov(self.n, r, A)))
            growth = r - A @ self.n
            self.n = torch.where(self.active, (self.n * torch.exp(self.dt * growth)).clamp(min=1e-6), self.n)
        V.append(float(self.lyapunov(self.n, r, A)))
        return V

    @torch.no_grad()
    def extinctions(self, memory=None):
        """Elimina especies persistentemente raras; guarda su clave como fósil."""
        share = self.n / self.n[self.active].sum().clamp(min=1e-12)
        low = self.active & (share < self.eps_ext)
        self.low_count = torch.where(low, self.low_count + 1, torch.zeros_like(self.low_count))
        dead = (self.low_count >= self.patience) & self.active
        if int(self.active.sum()) - int(dead.sum()) < 2:               # mínimo 2 especies:
            keep = torch.where(self.active, self.n, torch.full_like(self.n, -1.0)).topk(2).indices
            dead[keep] = False                                         # sobreviven las 2 más abundantes
        out = []
        for j in torch.nonzero(dead).flatten().tolist():
            self.active[j] = False
            self.n[j] = 0.0
            self.low_count[j] = 0
            if memory is not None:
                memory.write(self.keys[j:j + 1], torch.zeros(1, memory.vals.shape[1], device=self.keys.device), tag=1)
            out.append(j)
        return out

    @torch.no_grad()
    def spawn_mutant(self, parent: int, direction: torch.Tensor | None = None, noise: float = 0.05):
        """Copia perturbada del padre en una ranura libre. Devuelve índice o None."""
        free = torch.nonzero(~self.active).flatten().tolist()
        if not free:
            return None
        j = free[0]
        self.experts[j].load_state_dict(copy.deepcopy(self.experts[parent].state_dict()))
        for p in self.experts[j].parameters():
            p.add_(noise * p.abs().mean() * torch.randn_like(p))
        new_key = self.keys[parent] + (direction if direction is not None else noise * torch.randn_like(self.keys[parent]))
        self.keys[j] = F.normalize(new_key, dim=-1)
        return j

    @torch.no_grad()
    def invasion_fitness(self, j_mut, r, A):
        """Λ = r' - Σ_l α_{j' l} n*_l  con la comunidad residente en equilibrio."""
        residents = self.active.clone()
        residents[j_mut] = False
        return float(r[j_mut] - (A[j_mut] * self.n * residents.float()).sum())

    @torch.no_grad()
    def admit(self, j, n0: float | None = None):
        self.active[j] = True
        self.n[j] = n0 if n0 is not None else 0.1 * float(self.n[self.active].mean())
        self.low_count[j] = 0


@torch.no_grad()
def chesson_diagnostic(model, val):
    """Diagnóstico de coexistencia de Chesson (Annu. Rev. Ecol. Syst. 2000) para los expertos activos (v0.8).

    Con el Lotka-Volterra del ecosistema, dn_i/dt = n_i (r_i - Σ_j A_ij n_j):
        solapamiento de nicho   ρ_ij = sqrt( A_ij A_ji / (A_ii A_jj) )
        razón de aptitud        κ_i/κ_j = (r_i / r_j) · sqrt( A_jj A_ji / (A_ii A_ij) )
        coexisten si            ρ_ij < κ_i/κ_j < 1/ρ_ij
    Si casi ningún par cumple la desigualdad, la ecología solo puede llevar a exclusión: es la
    explicación cuantitativa de por qué no ayuda en tareas estáticas. Devuelve un resumen."""
    from .model import sample_losses
    from . import to_device
    was = model.training
    model.eval()
    v = to_device(val, model.device)
    kw = {"episodic_self": model.memory is not None} if hasattr(model, "memory") and hasattr(model, "novelty_map") else {}
    out = model(v, dense=True, **kw)
    loss_mix = sample_losses(out["fields"], out["scalars"], v)
    loss_j = {j: sample_losses(f, s, v) for j, (f, s) in out["per_expert"].items()}
    r, A = model.moe.ecology_stats(out["pi"], loss_mix, loss_j)
    act = torch.nonzero(model.moe.active).flatten().tolist()
    pares, ok, rhos = 0, 0, []
    for a in act:
        for b in act:
            if b <= a:
                continue
            if min(float(A[a, a]), float(A[b, b]), float(A[a, b]), float(A[b, a])) <= 0 or r[a] <= 0 or r[b] <= 0:
                continue
            rho = float((A[a, b] * A[b, a] / (A[a, a] * A[b, b])).sqrt())
            kappa = float(r[a] / r[b] * (A[b, b] * A[b, a] / (A[a, a] * A[a, b])).sqrt())
            pares += 1
            rhos.append(rho)
            ok += int(rho < kappa < 1.0 / max(rho, 1e-9))
    model.train(was)
    return {"especies": len(act), "pares": pares, "pares_que_coexisten": ok,
            "fraccion_coexistencia": ok / pares if pares else float("nan"),
            "solapamiento_medio": sum(rhos) / len(rhos) if rhos else float("nan")}
