"""Entrenamiento del módulo central con memoria episódica (aprendizaje en contexto).

Cada lote contiene varios episodios completos; cada muestra lee de las demás de su
episodio. Así el modelo aprende a USAR la memoria para adaptarse con pocos ejemplos.
Las memorias de largo plazo se consolidan con los lotes ya vistos (escala lenta).
"""
import torch

from .model import sample_losses, load_balance_loss
from .dominios.base import subset, split_episodes
from .train import ecology_epoch
from . import to_device


def episode_batches(ds, episodes_per_batch=4, shuffle=True, gen=None):
    eps = ds["episode"].unique()
    order = eps[torch.randperm(eps.numel(), generator=gen)] if shuffle else eps
    for i in range(0, order.numel(), episodes_per_batch):
        sel = torch.isin(ds["episode"], order[i:i + episodes_per_batch])
        yield subset(ds, torch.nonzero(sel).flatten())


def rebuild_memory(model, ds, passes: int = 2):
    """Reconstruye la memoria plasmodial y el mapa de novedad con el codificador FINAL (v0.7).

    Durante el entrenamiento los nodos se escriben con el codificador de cada época; al terminar,
    muchas posiciones son de un codificador que ya no existe y la novedad compara espacios distintos
    (AUROC de la memoria plasmodial en sangre: 0.64-0.74 frente a 0.98 de un kNN con claves actuales).
    Como en la consolidación durante el sueño, se repasa lo vivido con el modelo actual: se vacían la
    red y el mapa y se escriben de nuevo todos los episodios. La memoria neuronal de largo plazo no se
    toca: el modelo aprendió a leerla durante el entrenamiento."""
    model.eval()
    model.novelty_map.reset()
    if model.memory is not None and model.memory.mpc is not None:
        model.memory.mpc.reset()
        for _ in range(passes):                     # la 2.ª pasada adapta conductancias y habituación
            for b in episode_batches(ds, 16, shuffle=False):
                b = to_device(b, model.device)
                with torch.no_grad():
                    _, pooled, _ = model.embed(b)
                    vals = model.dominio.memory_values(b)
                    desc = model.memory.episode_descriptor(pooled, vals, b["episode"])
                    k = model.memory.key_proj(pooled)
                    model.memory.mpc.write(model.memory.long_key(k, desc), model.memory.standardize(vals),
                                           pos=model.memory.long_key(k, None))
    for b in episode_batches(ds, 16, shuffle=False):
        model.map_visit(b)


def fit_central(model, train, val, epochs=20, lr=2e-3, episodes_per_batch=4, eco_warmup=5,
                consolidate_every=1, verbose=False, log=None, anchor_lambda: float = 0.0,
                memory_from=None, rebuild: bool = True, succession: int = 0, schedule: str = "constante",
                evo=None):
    """anchor_lambda > 0: penaliza alejarse de los pesos de partida, λ Σ (w - w0)².
    Sirve al ajustar con datos nuevos para no olvidar lo anterior (regularización tipo L2-SP).
    rebuild: al final, reconstruir memoria plasmodial y mapa de novedad con el codificador final,
    a partir de `memory_from` (por defecto, `train`).
    succession: épocas pioneras (v0.8). Durante ellas no corre Lotka-Volterra ni hay extinciones: las
    abundancias quedan uniformes mientras las aptitudes de los expertos todavía son ruido. La ecología
    empieza cuando ya hay diferencias reales que seleccionar (sucesión ecológica)."""
    log = log if log is not None else {}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.05) \
        if schedule == "coseno" else None
    anchor = [p.detach().clone() for p in model.parameters()] if anchor_lambda > 0 else None
    use_epi = model.memory is not None
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for b in episode_batches(train, episodes_per_batch):
            b = to_device(b, model.device)
            out = model(b, episodic_self=use_epi)
            loss = sample_losses(out["fields"], out["scalars"], b).mean()
            if not model.use_ecology:
                loss = loss + 0.01 * load_balance_loss(out["pi"], model.moe.top_k)
            if anchor is not None:
                loss = loss + anchor_lambda * sum(((p - p0) ** 2).sum() for p, p0 in zip(model.parameters(), anchor))
            if not torch.isfinite(loss):
                log.setdefault("lotes_no_finitos", 0)
                log["lotes_no_finitos"] += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * b["points"].shape[0]
            n += b["points"].shape[0]
            if use_epi and model.memory.ltm is not None and (ep + 1) % consolidate_every == 0 and ep >= eco_warmup:
                model.consolidate(b, pooled=out["pooled"].detach())
        if ep == epochs - 1:
            if rebuild:
                rebuild_memory(model, memory_from if memory_from is not None else train)
            else:
                for b in episode_batches(train, 16, shuffle=False):
                    model.map_visit(b)                              # "dónde he estado"
        if model.use_ecology and ep >= succession:
            # estadísticas de ecología con lectura en contexto, como en el uso real
            # evo (v0.9.3): datos para la selección de expertos, distintos de la validación; si la
            # validación decide qué expertos viven, deja de ser evidencia independiente
            ecology_epoch(_EpisodicView(model), evo if evo is not None else val, log,
                          allow_extinction=ep >= eco_warmup)
        if sched is not None:
            sched.step()
        log.setdefault("perdida", []).append(tot / max(n, 1))
        if verbose:
            print(f"ep {ep:3d} | pérdida {tot / max(n, 1):.4f} | especies {int(model.moe.active.sum())}", flush=True)
    return log


class _EpisodicView(torch.nn.Module):
    """Envuelve el modelo para que ecology_epoch use lectura episódica."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.moe, self.memory, self.use_memory = model.moe, None, False

    @property
    def device(self):
        return self.m.device

    def forward(self, batch, dense=False):
        return self.m(batch, dense=dense, episodic_self=self.m.memory is not None)

    def eval(self):
        self.m.eval()
        return self


@torch.no_grad()
def evaluate_fewshot(model, ds, k_context, seed=0):
    """Error en las muestras de consulta usando k_context ejemplos etiquetados del MISMO episodio
    como contexto (sin gradiente). Devuelve dict con el error medio por componente."""
    model.eval()
    ctx_idx, qry_idx = split_episodes(ds, k_context, seed)
    errs = {"campos_L2rel": 0.0, "escalares_MAE": 0.0}
    n = 0
    for e in ds["episode"].unique():
        q = qry_idx[ds["episode"][qry_idx] == e]
        c = ctx_idx[ds["episode"][ctx_idx] == e]
        qb = subset(ds, q)
        out = model(qb, context=subset(ds, c) if (k_context > 0 and model.memory is not None) else None)
        f, t = out["fields"], to_device(qb, model.device)["fields"]
        rel = torch.linalg.vector_norm(f - t, dim=(1, 2)) / torch.linalg.vector_norm(t, dim=(1, 2)).clamp(min=1e-6)
        sc = (out["scalars"] - to_device(qb, model.device)["scalars"]).abs().mean(-1)
        errs["campos_L2rel"] += float(rel.sum())
        errs["escalares_MAE"] += float(sc.sum())
        n += q.numel()
    return {k: v / n for k, v in errs.items()}


VARIABLE_BUFFERS = ("novelty_map.", "memory.mpc.")     # únicos buffers que pueden cambiar de tamaño


def load_flexible(model, state):
    """Carga un state_dict saltándose los buffers de tamaño variable (nodos de memoria, mapa de novedad),
    que un modelo recién creado tiene vacíos. Sin esto, load_state_dict falla (error presente desde v0.4)."""
    own = model.state_dict()
    skipped = [k for k, v in state.items() if k not in own or own[k].shape != v.shape]
    bad = [k for k in skipped if not k.startswith(VARIABLE_BUFFERS)]
    if bad:                                          # v0.9.3: nunca descartar pesos entrenados en silencio
        raise RuntimeError(f"load_flexible: parámetros incompatibles que no son memorias de tamaño variable: {bad[:5]}")
    model.load_state_dict({k: v for k, v in state.items() if k not in skipped}, strict=False)
    model.claves_omitidas = skipped
    return model


def finetune_fewshot(model_factory, state, ds, k_context, steps=30, lr=1e-3, seed=0):
    """Línea base con gradiente: ajusta una copia del modelo con los k ejemplos de CADA episodio
    y evalúa en sus consultas. Es costosa (un ajuste por episodio), como en la práctica clínica."""
    ctx_idx, qry_idx = split_episodes(ds, k_context, seed)
    errs = {"campos_L2rel": 0.0, "escalares_MAE": 0.0}
    n = 0
    for e in ds["episode"].unique():
        m = model_factory()
        load_flexible(m, state)
        q = qry_idx[ds["episode"][qry_idx] == e]
        c = ctx_idx[ds["episode"][ctx_idx] == e]
        if k_context > 0:
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            cb = to_device(subset(ds, c), m.device)
            m.train()
            for _ in range(steps):
                out = m(cb)
                loss = sample_losses(out["fields"], out["scalars"], cb).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            qb = to_device(subset(ds, q), m.device)
            out = m(qb)
            rel = torch.linalg.vector_norm(out["fields"] - qb["fields"], dim=(1, 2)) / torch.linalg.vector_norm(qb["fields"], dim=(1, 2)).clamp(min=1e-6)
            sc = (out["scalars"] - qb["scalars"]).abs().mean(-1)
        errs["campos_L2rel"] += float(rel.sum())
        errs["escalares_MAE"] += float(sc.sum())
        n += q.numel()
    return {k: v / n for k, v in errs.items()}
