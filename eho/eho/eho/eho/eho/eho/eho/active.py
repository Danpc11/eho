"""Aprendizaje activo: qué simular después.

1. acquisition(): a(θ,c) = [ν(g) + κ·Var_j f_j] / costo   (regla fija).
2. DesignPolicy: política gaussiana entrenada con REINFORCE contra el propio EHO
   (el simulador hace de entorno). Recompensa = ganancia de información aproximada
   (novedad + desacuerdo) menos costo. Es una versión simplificada de los métodos
   de diseño experimental bayesiano con refuerzo (DAD, RL-DAD); sustituible por ellos.
"""
import math
import torch
import torch.nn as nn


@torch.no_grad()
def disagreement(model, batch):
    out = model(batch, dense=True)
    preds = torch.stack([s for (_, s) in out["per_expert"].values()], 0)   # [J,B,2]
    return preds.var(0).sum(-1) if preds.shape[0] > 1 else torch.zeros(preds.shape[1], device=preds.device)


@torch.no_grad()
def acquisition(model, candidates, kappa: float = 5.0, cost=None):
    nu = model.novelty(candidates).cpu()
    nu = torch.where(torch.isinf(nu), torch.ones_like(nu), nu)
    nu = nu - nu.min() + 1e-3                      # ν puede ser negativa: se desplaza a > 0
    score = nu + kappa * disagreement(model, candidates).cpu()
    if cost is not None:
        score = score / cost                       # así dividir entre el costo favorece lo barato
    return score


def dpd_cost_proxy(H, D, gdot):
    """Costo relativo de una simulación DPD: crece con el volumen (D²·L) y con H."""
    return (D / 20.0) ** 2 * (0.5 + H) * (1.0 + 0.1 * torch.log1p(gdot))


class DesignPolicy(nn.Module):
    """π(ξ | contexto): propone (θ, H, D, γ̇) normalizados en [0,1]^7."""

    def __init__(self, d_ctx: int = 4, d_design: int = 7, width: int = 64):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(d_ctx, width), nn.Tanh(), nn.Linear(width, d_design))
        self.log_std = nn.Parameter(torch.full((d_design,), -1.0))

    def dist(self, ctx):
        return torch.distributions.Normal(self.mu(ctx), self.log_std.exp())

    def sample(self, ctx):
        d = self.dist(ctx)
        x = d.sample()
        return torch.sigmoid(x), d.log_prob(x).sum(-1)


def train_design_policy(policy, reward_fn, ctx, iters: int = 50, batch: int = 64, lr: float = 3e-3):
    """REINFORCE con línea base media. reward_fn(diseños [B,7]) -> [B]."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    hist = []
    for _ in range(iters):
        c = ctx.expand(batch, -1)
        xi, logp = policy.sample(c)
        r = reward_fn(xi).detach()
        loss = -((r - r.mean()) * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(r.mean()))
    return hist


# ---------------------------------------------------------------------------
# Valor común para la colmena: ganancia de información (v0.4)
# ---------------------------------------------------------------------------

@torch.no_grad()
def expert_disagreement(agent, candidates):
    """Desacuerdo entre los expertos de UN agente (su comité interno): [B].

    Varianza entre expertos de los escalares más la de los campos. Es una estimación de la
    incertidumbre epistémica del agente, que el ecosistema de expertos da gratis.
    """
    agent.eval()
    out = agent(candidates, dense=True)
    S = torch.stack([s for (_, s) in out["per_expert"].values()])          # [J,B,D]
    if S.shape[0] < 2:
        return torch.zeros(S.shape[1])
    Fd = torch.stack([f for (f, _) in out["per_expert"].values()])         # [J,B,N,F]
    return (S.var(0).mean(-1) + Fd.var(0).mean((-1, -2))).cpu()


class InformationValue:
    """Valor de simular un candidato, para cada agente, en una escala ABSOLUTA [0, 1].

    EIG_a(x) = ½ log(1 + Var_a(x) / σ²_a)  (ganancia de información gaussiana), con σ²_a la
    mediana del desacuerdo del agente en datos ya conocidos (referencia). Luego
    v_a(x) = 1 - exp(-EIG_a(x) / (2 · ½ log 2)): un caso típico ya conocido vale ≈ 0.39 y
    uno 10 veces más incierto ≈ 0.82. Como la escala es absoluta, la colmena puede
    decidir que NO vale la pena simular nada (regla sensible al valor).
    """

    def __init__(self, agents):
        self.agents = agents
        self.ref = None

    def fit(self, reference):
        self.ref = torch.stack([expert_disagreement(a, reference).median().clamp(min=1e-8) for a in self.agents])
        return self

    def eig(self, candidates):
        var = torch.stack([expert_disagreement(a, candidates) for a in self.agents])   # [A,B]
        return 0.5 * torch.log1p(var / self.ref.unsqueeze(-1))

    def values(self, candidates):
        return 1.0 - torch.exp(-self.eig(candidates) / (2 * 0.5 * math.log(2.0)))


@torch.no_grad()
def _embed(agent, ds):
    _, pooled, _ = agent.embed({k: v.to(agent.device) for k, v in ds.items()})
    return pooled


class NoveltyValue:
    """Valor común para elegir qué simular: NOVEDAD calibrada de cada agente (v0.4).

    Hallazgo: el desacuerdo entre expertos NO sirve (los expertos coinciden con exceso de confianza
    fuera de distribución: AUROC 0.0-0.46 en advección-difusión). La novedad de la memoria
    plasmodial sí detecta lo nuevo (AUROC 1.00 en advección-difusión, 0.80 en sangre).

    Cada agente mide la novedad con su memoria plasmodial (o, si no la tiene, con la distancia a
    sus 5 vecinos más cercanos en su espacio de representación) y la lleva a una escala absoluta:
        v_a(x) = σ( 4 · (n_a(x) - q90_a) / (q90_a - q50_a) )
    donde q50 y q90 son cuantiles de la novedad en datos ya conocidos. Lo típico vale ≈ 0.02,
    el percentil 90 de lo conocido vale 0.5 y lo realmente nuevo se acerca a 1. Así la colmena
    puede decidir que ningún candidato merece una simulación.
    """

    def __init__(self, agents, k: int = 5):
        self.agents, self.k = agents, k
        self.refs, self.q50, self.q90 = [], None, None

    def _novelty(self, i, ds):
        a = self.agents[i]
        if a.memory is not None and a.memory.mpc is not None and len(a.memory.mpc) > 0:
            return a.novelty(ds).cpu()                             # memoria plasmodial (la mejor señal)
        if hasattr(a, "novelty_map") and len(a.novelty_map) > 0:
            return a.map_novelty(ds).cpu()                         # respaldo: mapa de novedad
        R = self.refs[i]
        X = _embed(a, ds).cpu()
        mu, sd = R.mean(0), R.std(0) + 1e-6
        return torch.cdist((X - mu) / sd, (R - mu) / sd).topk(min(self.k, R.shape[0]), largest=False).values.mean(-1)

    def fit(self, reference):
        self.refs = [_embed(a, reference).cpu() for a in self.agents]
        n = torch.stack([self._novelty(i, reference) for i in range(len(self.agents))])
        self.q50, self.q90 = n.quantile(0.5, dim=1), n.quantile(0.9, dim=1)
        return self

    def novelty(self, candidates):
        return torch.stack([self._novelty(i, candidates) for i in range(len(self.agents))])   # [A,B]

    def values(self, candidates):
        n = self.novelty(candidates)
        scale = (self.q90 - self.q50).clamp(min=1e-6).unsqueeze(-1)
        return torch.sigmoid(4.0 * (n - self.q90.unsqueeze(-1)) / scale)

    def episode_values(self, candidates):
        """Valor por EPISODIO candidato (media de sus muestras): [A, n_episodios], y los ids."""
        v = self.values(candidates)
        eps = candidates["episode"].unique()
        return torch.stack([v[:, candidates["episode"] == e].mean(1) for e in eps], 1), eps


# ---------------------------------------------------------------------------------------------
# v0.6: valor por INSPECCIÓN — "¿todavía me equivoco por aquí?"
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def inspection_errors(agent, ds, k: int = 2, seed: int = 0):
    """Error de inspección por muestra [B, 2] = (MAE de escalares, L2 relativo de campos).
    Cada muestra se predice usando k OTRAS muestras etiquetadas del mismo episodio como contexto,
    igual que en uso. Se mide ANTES de reentrenar con esos datos: es un error de prueba honesto."""
    agent.eval()
    g = torch.Generator().manual_seed(seed)
    B = ds["episode"].numel()
    err = torch.zeros(B, 2)
    use_ctx = k > 0 and agent.memory is not None
    for e in ds["episode"].unique():
        idx = torch.nonzero(ds["episode"] == e).flatten()
        for i in idx:
            others = idx[idx != i]
            c = others[torch.randperm(others.numel(), generator=g)[:k]]
            one = {kk: v[i.view(1)] for kk, v in ds.items()}
            ctx = {kk: v[c] for kk, v in ds.items()} if (use_ctx and c.numel()) else None
            out = agent(one, context=ctx)
            err[i, 0] = (out["scalars"].cpu() - one["scalars"]).abs().mean()
            if one["fields"].numel():
                t = one["fields"]
                err[i, 1] = torch.linalg.vector_norm(out["fields"].cpu() - t) / torch.linalg.vector_norm(t).clamp(min=1e-6)
    return err


class InspectionValue:
    """Valor de simular por ERROR RECIENTE en la vecindad (v0.6).

    Problema de v0.5: en cuanto la memoria registra los primeros casos nuevos, la región deja de
    parecer nueva y la colmena se detiene, aunque el modelo todavía se equivoque ahí. Novedad ≠ error.

    Solución, como la abeja que regresa a inspeccionar el sitio: cada caso simulado se compara con
    lo que el agente predijo ANTES de aprenderlo. Ese error se guarda con el caso. El valor de un
    candidato x es el error reciente de los casos simulados cercanos (núcleo gaussiano en el espacio
    de claves de la memoria, recalculado con el agente actual):

        ê_a(x) = Σ_r w_r K(x, x_r) s_r / Σ_r w_r K(x, x_r),    c_a(x) = Σ w K / (Σ w K + 1)
        v_a^err(x) = c_a(x) · σ( 4 (ê_a(x) - q90_a) / (q90_a - q50_a) )

    s es el error normalizado por la mediana de lo conocido (escalares y campos), q50 y q90 son
    cuantiles del error de inspección en datos conocidos y w_r decae por ronda (olvido) para que
    pese más lo que se midió después de aprender. Con la novedad se combina como un "o" suave:

        v = 1 - (1 - v^nov)(1 - v^err)

    Así la colmena sigue simulando una región mientras los casos que acaba de simular ahí
    sigan saliendo mal, y se detiene cuando ya salen bien aunque la región sea "conocida".

    c_a(x) dice además si la vecindad de x YA se inspeccionó. Las exploradoras (eho/gemelo.py)
    usan eso: si la colmena se abstiene, se simula el mejor candidato aún no inspeccionado.
    Detenerse exige evidencia de que ya se predice bien, no falta de evidencia.
    """

    def __init__(self, agents, k: int = 2, decay: float = 0.5, width: float = 2.0):
        self.agents, self.k, self.decay, self.width = agents, k, decay, width
        self.records = []           # [(ds, s [A,B], peso)]

    def _score(self, E):
        """E [A,B,2] -> error normalizado [A,B]."""
        has_f = bool((self.med[:, 1] > 0).all())
        s = E[..., 0] / self.med[:, :1]
        if has_f:
            s = 0.5 * (s + E[..., 1] / self.med[:, 1:])
        return s

    def _keys(self, a, ds):
        return self.agents[a]._map_key(_embed(self.agents[a], ds)).cpu()

    @torch.no_grad()
    def fit(self, reference):
        E = torch.stack([inspection_errors(m, reference, self.k) for m in self.agents])     # [A,B,2]
        self.med = E.median(1).values.clamp(min=1e-8)
        s = self._score(E)
        self.q50, self.q90 = s.quantile(0.5, dim=1), s.quantile(0.9, dim=1)
        self.h = []
        for a in range(len(self.agents)):
            K = self._keys(a, reference)
            D = torch.cdist(K, K)
            D.fill_diagonal_(float("inf"))
            self.h.append(float(D.topk(min(5, K.shape[0] - 1), largest=False).values.mean(-1).median()) * self.width)
        return self

    def register(self, ds):
        """Llamar con los casos recién simulados ANTES de reentrenar. ds: un conjunto, o una lista con
        uno por agente (cada uno simulado con su fuente). Devuelve el error normalizado [A,B]."""
        self.records = [(d, s, w * self.decay) for d, s, w in self.records]
        ds = list(ds) if isinstance(ds, (list, tuple)) else [ds] * len(self.agents)
        E = torch.stack([inspection_errors(m, d, self.k) for m, d in zip(self.agents, ds)])
        s = self._score(E)
        self.records.append((ds, s, 1.0))
        return s

    @torch.no_grad()
    def values(self, candidates):
        A = len(self.agents)
        B = candidates["episode"].numel()
        if not self.records:
            self.last_conf = torch.zeros(A, B)
            return torch.zeros(A, B)
        out, confs = [], []
        for a in range(A):
            Kc = self._keys(a, candidates)
            num, den = torch.zeros(B), torch.zeros(B)
            for ds, s, w in self.records:
                Kr = self._keys(a, ds[a])
                W = w * torch.exp(-torch.cdist(Kc, Kr) ** 2 / (2 * self.h[a] ** 2))        # [B,R]
                num += W @ s[a]
                den += W.sum(1)
            e_hat = num / den.clamp(min=1e-12)
            conf = den / (den + 1.0)
            scale = (self.q90[a] - self.q50[a]).clamp(min=1e-6)
            out.append(conf * torch.sigmoid(4.0 * (e_hat - self.q90[a]) / scale))
            confs.append(conf)
        self.last_conf = torch.stack(confs)
        return torch.stack(out)

    def episode_values(self, candidates):
        v = self.values(candidates)
        eps = candidates["episode"].unique()
        ep = lambda x: torch.stack([x[:, candidates["episode"] == e].mean(1) for e in eps], 1)
        self.last_episode_conf = ep(self.last_conf)
        return ep(v), eps


class SimulationValue:
    """Valor total de simular: novedad (memoria plasmodial) O error reciente (inspección)."""

    def __init__(self, agents, k: int = 2, use_error: bool = True, decay: float = 0.5):
        self.nov = NoveltyValue(agents)
        self.err = InspectionValue(agents, k=k, decay=decay) if use_error else None

    def fit(self, reference):
        self.nov.fit(reference)
        if self.err is not None:
            self.err.fit(reference)
        return self

    def refit_novelty(self, reference):
        """Tras reentrenar: recalibrar la escala de novedad (los errores registrados se conservan)."""
        self.nov.fit(reference)
        return self

    def register(self, ds):
        return self.err.register(ds) if self.err is not None else None

    def episode_values(self, candidates):
        vn, eps = self.nov.episode_values(candidates)
        if self.err is None:
            return vn, eps, {"novedad": vn}
        ve, _ = self.err.episode_values(candidates)
        return 1 - (1 - vn) * (1 - ve), eps, {"novedad": vn, "error": ve,
                                              "inspeccionado": self.err.last_episode_conf}
