"""Líneas base del artículo (v0.7), con el MISMO presupuesto y los mismos datos que el EHO.

Tres familias, una por afirmación:

1. Novedad sin etiquetas (afirmación 2). Detectores con puntuación "mayor = más nuevo":
       entradas      Mahalanobis y kNN sobre las entradas crudas (condiciones + resumen de tokens).
                     Si estos ya detectan lo nuevo, el cambio es visible y no hace falta memoria.
       rasgos        Mahalanobis y kNN sobre el estado global del modelo (pooled).
       clave         kNN coseno y Mahalanobis en el espacio de claves de la memoria plasmodial:
                     el MISMO espacio que la memoria. Separa el efecto del espacio del efecto del
                     mecanismo (la comparación de v0.4 los mezclaba).
       plasmodial    novedad coseno (v0.6) y por resistencia efectiva de Kirchhoff (v0.7).
       ensamble      varianza de las predicciones entre agentes (ensamble profundo).
       inspeccion    error al predecir el contexto etiquetado (usa etiquetas; se reporta aparte).
2. Qué simular (afirmación 3): aleatoria, novedad codiciosa, valor codicioso (la misma señal que
   la colmena, sin la dinámica de abejas), varianza del ensamble (aproximación de BALD para
   regresión con ensamble gaussiano: misma ordenación que la varianza epistémica) y core-set
   (k-centros en el espacio de claves).
3. Pocos ejemplos (afirmación 1): ajuste fino de todo el modelo o solo de la cabeza de expertos,
   con pasos y tasa elegidos en episodios de validación, y con su costo medido en segundos.
   El proceso neuronal atentivo corresponde a la variante memoria="episodica" del mismo núcleo.

Pendiente (no implementado): MC dropout (el núcleo no tiene dropout) y MAML.
"""
from __future__ import annotations

import copy
import time

import torch
import torch.nn.functional as F

from .dominios.base import subset, split_episodes
from .model import sample_losses
from . import to_device


# --------------------------------------------------------------------------------------------- métricas
def auroc(score_known: torch.Tensor, score_new: torch.Tensor) -> float:
    """AUROC = P(puntuación de un caso nuevo > la de uno conocido); empates cuentan 1/2."""
    a, b = score_known.double().flatten(), score_new.double().flatten()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    gt = (b.unsqueeze(1) > a.unsqueeze(0)).double().mean()
    eq = (b.unsqueeze(1) == a.unsqueeze(0)).double().mean()
    return float(gt + 0.5 * eq)


def auroc_ic(score_known, score_new, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """AUROC con intervalo bootstrap percentil (remuestreo independiente de ambos grupos)."""
    g = torch.Generator().manual_seed(seed)
    a, b = score_known.flatten(), score_new.flatten()
    vals = []
    for _ in range(n_boot):
        ia = torch.randint(a.numel(), (a.numel(),), generator=g)
        ib = torch.randint(b.numel(), (b.numel(),), generator=g)
        vals.append(auroc(a[ia], b[ib]))
    v = torch.tensor(vals)
    lo, hi = v.quantile((1 - level) / 2), v.quantile(1 - (1 - level) / 2)
    return auroc(a, b), float(lo), float(hi)


# --------------------------------------------------------------------------------------------- novedad
def raw_inputs(ds) -> torch.Tensor:
    """Entradas crudas: condiciones visibles + media y desviación de cada rasgo de los tokens."""
    pts = ds["points"].float()
    return torch.cat([ds["cond"].float(), pts.mean(1), pts.std(1)], -1)


class Mahalanobis:
    """Distancia de Mahalanobis con covarianza encogida (Ledoit-Wolf simple, encogimiento fijo)."""

    def __init__(self, shrink: float = 0.1):
        self.shrink = shrink

    def fit(self, X):
        X = X.double()
        self.mu = X.mean(0)
        C = torch.cov((X - self.mu).T).reshape(X.shape[1], X.shape[1])
        tr = torch.trace(C) / C.shape[0]
        C = (1 - self.shrink) * C + self.shrink * tr * torch.eye(C.shape[0], dtype=C.dtype)
        self.P = torch.linalg.inv(C + 1e-9 * torch.eye(C.shape[0], dtype=C.dtype))
        return self

    def score(self, X):
        d = X.double() - self.mu
        return ((d @ self.P) * d).sum(-1).clamp(min=0).sqrt().float()


def knn_score(R: torch.Tensor, X: torch.Tensor, k: int = 5, metric: str = "euclidea") -> torch.Tensor:
    """Distancia media a los k vecinos más cercanos de la referencia R."""
    if metric == "coseno":
        D = 1.0 - F.normalize(X, dim=-1) @ F.normalize(R, dim=-1).T
    else:
        mu, sd = R.mean(0), R.std(0) + 1e-6
        D = torch.cdist((X - mu) / sd, (R - mu) / sd)
    return D.topk(min(k, R.shape[0]), largest=False).values.mean(-1)


@torch.no_grad()
def _reps(agent, ds, chunk: int = 256):
    out = {}
    for i in range(0, ds["episode"].numel(), chunk):
        r = agent.representations(subset(ds, torch.arange(i, min(i + chunk, ds["episode"].numel()))))
        for k, v in r.items():
            out.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def ensemble_variance(agents, ds, scale=None) -> torch.Tensor:
    """Varianza entre agentes de las predicciones escalares (sin contexto), media sobre dimensiones."""
    P = torch.stack([a(ds)["scalars"].cpu() for a in agents])                 # [A,B,D]
    if scale is not None:
        P = P / scale
    return P.var(0).mean(-1)


@torch.no_grad()
def novelty_detectors(agents, reference, ds, k: int = 5) -> dict[str, torch.Tensor]:
    """Todas las puntuaciones de novedad sin etiquetas para `ds` (mayor = más nuevo).

    Los detectores por agente se estandarizan con su distribución en `reference` y se promedian
    entre agentes, igual que la colmena promedia sus valores."""
    out = {}
    Xr, Xd = raw_inputs(reference), raw_inputs(ds)
    out["entradas_mahalanobis"] = Mahalanobis().fit(Xr).score(Xd)
    out["entradas_knn"] = knn_score(Xr, Xd, k)
    per = {}

    def add(name, s_ref, s_ds):
        mu, sd = s_ref.mean(), s_ref.std().clamp(min=1e-8)
        per.setdefault(name, []).append((s_ds - mu) / sd)

    for a in agents:
        Rr, Rd = _reps(a, reference), _reps(a, ds)
        mp = Mahalanobis().fit(Rr["pooled"])
        add("rasgos_mahalanobis", mp.score(Rr["pooled"]), mp.score(Rd["pooled"]))
        add("rasgos_knn", _loo_knn(Rr["pooled"], k), knn_score(Rr["pooled"], Rd["pooled"], k))
        if "key" in Rr:
            add("clave_knn_coseno", _loo_knn(Rr["key"], 1, "coseno"), knn_score(Rr["key"], Rd["key"], 1, "coseno"))
            m = Mahalanobis().fit(Rr["key"])
            add("clave_mahalanobis", m.score(Rr["key"]), m.score(Rd["key"]))
        if a.memory is not None and a.memory.mpc is not None and len(a.memory.mpc) > 1:
            for mode in ("coseno", "resistencia", "energia"):
                add(f"plasmodial_{mode}", a.novelty(reference, mode).cpu(), a.novelty(ds, mode).cpu())
    out.update({k: torch.stack(v).mean(0) for k, v in per.items()})
    out["ensamble_varianza"] = ensemble_variance(agents, ds)
    return out


def _loo_knn(R, k, metric="euclidea"):
    """kNN de la referencia contra sí misma, excluyendo el propio punto (para estandarizar)."""
    if metric == "coseno":
        D = 1.0 - F.normalize(R, dim=-1) @ F.normalize(R, dim=-1).T
    else:
        mu, sd = R.mean(0), R.std(0) + 1e-6
        D = torch.cdist((R - mu) / sd, (R - mu) / sd)
    D.fill_diagonal_(float("inf"))
    return D.topk(min(k, R.shape[0] - 1), largest=False).values.mean(-1)


def episode_mean(score: torch.Tensor, episode: torch.Tensor):
    eps = episode.unique()
    return torch.stack([score[episode == e].mean() for e in eps]), eps


# --------------------------------------------------------------------------------------------- qué simular
@torch.no_grad()
def select(strategy: str, gemelo, candidates, budget: int, gen: torch.Generator | None = None,
           reference=None) -> list[int]:
    """Episodios elegidos por una estrategia de línea base (siempre gasta el presupuesto completo,
    salvo "colmena", que puede elegir menos). Devuelve ids de episodio de `candidates`."""
    eps = candidates["episode"].unique()
    if strategy == "aleatoria":
        return eps[torch.randperm(eps.numel(), generator=gen)[:budget]].tolist()
    if strategy == "colmena":
        return [e["episodio"] for e in gemelo.proponer(candidates)]
    if strategy in ("novedad_codiciosa", "valor_codicioso"):
        V, veps, partes = gemelo.valor.episode_values(candidates)
        v = partes["novedad"] if strategy == "novedad_codiciosa" else V
        return veps[v.mean(0).topk(min(budget, veps.numel())).indices].tolist()
    if strategy == "varianza":
        s, veps = episode_mean(ensemble_variance(gemelo.agentes, candidates, gemelo.escala), candidates["episode"])
        return veps[s.topk(min(budget, veps.numel())).indices].tolist()
    if strategy == "coreset":
        if reference is None:
            # lo ya conocido: validación + lo que el agente 0 ya incorporó
            from .dominios.base import concat
            prev = getattr(gemelo, "_acumulado", [[]])[0]
            reference = concat(gemelo.validacion, *prev) if prev else gemelo.validacion
            reference = {k: v for k, v in reference.items() if k in candidates}
        return _coreset(gemelo.agentes[0], reference, candidates, budget)
    raise ValueError(f"estrategia desconocida: {strategy}")


@torch.no_grad()
def _coreset(agent, reference, candidates, budget):
    """k-centros voraz (Sener y Savarese, ICLR 2018) en el espacio de claves, por episodio."""
    space = "key" if agent.memory is not None else "pooled"
    R = _reps(agent, reference)[space]
    C = _reps(agent, candidates)[space]
    eps = candidates["episode"].unique()
    Ce = torch.stack([C[candidates["episode"] == e].mean(0) for e in eps])
    dmin = torch.cdist(Ce, R).min(1).values
    chosen = []
    for _ in range(min(budget, eps.numel())):
        j = int(dmin.argmax())
        chosen.append(int(eps[j]))
        dmin = torch.minimum(dmin, torch.cdist(Ce, Ce[j:j + 1]).squeeze(-1))
        dmin[j] = -1.0
    return chosen


# --------------------------------------------------------------------------------------------- pocos ejemplos
def finetune_fewshot(model_factory, state, ds, k_context, steps=30, lr=1e-3, mode="todo", seed=0):
    """Ajuste fino por episodio con sus k ejemplos. mode: "todo" | "cabeza" (solo los expertos).
    Devuelve errores medios y segundos por episodio (costo de adaptación)."""
    ctx_idx, qry_idx = split_episodes(ds, k_context, seed)
    errs = {"campos_L2rel": 0.0, "escalares_MAE": 0.0}
    n, secs, n_ep = 0, 0.0, 0
    for e in ds["episode"].unique():
        m = _copia(model_factory, state)
        q = qry_idx[ds["episode"][qry_idx] == e]
        c = ctx_idx[ds["episode"][ctx_idx] == e]
        t0 = time.perf_counter()
        if k_context > 0 and steps > 0:
            params = m.moe.parameters() if mode == "cabeza" else m.parameters()
            opt = torch.optim.Adam(params, lr=lr)
            cb = to_device(subset(ds, c), m.device)
            m.train()
            for _ in range(steps):
                out = m(cb)
                loss = sample_losses(out["fields"], out["scalars"], cb).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        secs += time.perf_counter() - t0
        n_ep += 1
        m.eval()
        with torch.no_grad():
            qb = to_device(subset(ds, q), m.device)
            out = m(qb)
            t = qb["fields"]
            if t.numel():
                rel = torch.linalg.vector_norm(out["fields"] - t, dim=(1, 2)) / \
                    torch.linalg.vector_norm(t, dim=(1, 2)).clamp(min=1e-6)
                errs["campos_L2rel"] += float(rel.sum())
            errs["escalares_MAE"] += float((out["scalars"] - qb["scalars"]).abs().mean(-1).sum())
        n += q.numel()
    res = {k: v / max(n, 1) for k, v in errs.items()}
    res["segundos_por_episodio"] = secs / max(n_ep, 1)
    return res


def tuned_finetune(model_factory, state, val, test, k_context, grid=None, seed=0):
    """Ajuste fino con hiperparámetros elegidos en validación (para que la línea base sea justa)."""
    grid = grid or [dict(steps=s, lr=lr, mode=m) for s in (10, 30, 100) for lr in (3e-4, 1e-3, 3e-3)
                    for m in ("todo", "cabeza")]
    best, best_err = None, float("inf")
    for h in grid:
        r = finetune_fewshot(model_factory, state, val, k_context, seed=seed, **h)
        e = r["escalares_MAE"] + r["campos_L2rel"]
        if e < best_err:
            best, best_err = h, e
    r = finetune_fewshot(model_factory, state, test, k_context, seed=seed, **best)
    r["hiperparametros"] = best
    return r


@torch.no_grad()
def memory_adaptation_cost(model, ds, k_context, seed=0):
    """Segundos por episodio de la adaptación por memoria (solo lectura del contexto)."""
    ctx_idx, qry_idx = split_episodes(ds, k_context, seed)
    model.eval()
    t0, n = time.perf_counter(), 0
    for e in ds["episode"].unique():
        q = qry_idx[ds["episode"][qry_idx] == e]
        c = ctx_idx[ds["episode"][ctx_idx] == e]
        model(subset(ds, q), context=subset(ds, c) if k_context > 0 else None)
        n += 1
    return (time.perf_counter() - t0) / max(n, 1)


def _copia(factory, state):
    """Copia del modelo base. Si `state` es el propio modelo, se copia entero: sus memorias tienen
    buffers de tamaño variable (nodos, mapa de novedad) que load_state_dict no puede cargar en un
    modelo recién creado (error de la v0.7 en el ajuste fino)."""
    import copy
    if isinstance(state, torch.nn.Module):
        return copy.deepcopy(state)
    from .train_central import load_flexible
    return load_flexible(factory(), state)
