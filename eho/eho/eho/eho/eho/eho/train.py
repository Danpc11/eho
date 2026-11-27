"""Entrenamiento con tres escalas de tiempo.

rápida : descenso de energía de Hopfield (dentro de forward)
media  : gradiente sobre pesos (cada lote)
lenta  : Lotka-Volterra, extinción, especiación por invasión, escritura en memoria (cada época)
"""
import torch

from .model import sample_losses, load_balance_loss
from .data.synthetic import subset
from . import to_device


def batches(ds, bs, shuffle=True, device=None):
    """Los datos viven en CPU; cada lote se copia al dispositivo del modelo."""
    n = ds["points"].shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        b = subset(ds, idx[i:i + bs])
        yield to_device(b, device) if device is not None else b


@torch.no_grad()
def evaluate(model, ds, bs=128):
    model.eval()
    tot, n = 0.0, 0
    parts = {"campos_L2rel": 0.0, "log_eta_MAE": 0.0, "cfl_MAE": 0.0}
    for b in batches(ds, bs, shuffle=False, device=next(model.parameters()).device):
        out = model(b)
        f, t = out["fields"], b["fields"]
        rel = torch.linalg.vector_norm(f - t, dim=(1, 2)) / torch.linalg.vector_norm(t, dim=(1, 2))
        err = (out["scalars"] - b["scalars"]).abs()
        k = b["points"].shape[0]
        parts["campos_L2rel"] += float(rel.sum())
        parts["log_eta_MAE"] += float(err[:, 0].sum())
        parts["cfl_MAE"] += float(err[:, 1].sum())
        n += k
    return {k: v / n for k, v in parts.items()}


@torch.no_grad()
def ecology_epoch(model, val, log, lv_steps=20, allow_extinction=True):
    """Estadísticas densas sobre validación -> Lotka-Volterra -> extinciones."""
    model.eval()
    val = to_device(val, model.device)
    out = model(val, dense=True)
    loss_mix = sample_losses(out["fields"], out["scalars"], val)
    loss_j = {j: sample_losses(f, s, val) for j, (f, s) in out["per_expert"].items()}
    r, A = model.moe.ecology_stats(out["pi"], loss_mix, loss_j)
    V = model.moe.lotka_volterra(r, A, steps=lv_steps)
    dead = model.moe.extinctions(model.memory if model.use_memory else None) if allow_extinction else []
    log.setdefault("lyapunov", []).append(V)
    log.setdefault("extinct", []).append(dead)
    return r, A, loss_mix, out


def try_invasion(model, opt_factory, train, val, log, steps=40, bs=32):
    """Crea un mutante del experto con más error ponderado y lo admite si Λ > 0."""
    model.eval()
    val = to_device(val, model.device)
    with torch.no_grad():
        out = model(val, dense=True)
        loss_j = {j: sample_losses(f, s, val) for j, (f, s) in out["per_expert"].items()}
        burden = {j: float((out["pi"][:, j] * l).mean()) for j, l in loss_j.items()}
    parent = max(burden, key=burden.get)
    j = model.moe.spawn_mutant(parent)
    if j is None:
        return None
    # entrenamiento breve del mutante en las muestras de mayor error del padre
    with torch.no_grad():
        o = model(train, dense=False)
        err = sample_losses(o["fields"], o["scalars"], train)
        hard = err.topk(min(256, err.numel())).indices.cpu()
    hard_ds = subset(train, hard)
    model.moe.admit(j)
    params = list(model.moe.experts[j].parameters()) + [model.moe.keys]
    opt = torch.optim.Adam(params, lr=1e-3)
    keys_before = model.moe.keys.detach().clone()
    other = torch.ones(model.moe.j_max, dtype=torch.bool, device=model.device)
    other[j] = False
    # Modo evaluación (sin actualizar el centro del nicho ni otras estadísticas);
    # solo el mutante y su clave se mueven: la comunidad residente queda intacta.
    model.eval()
    for _ in range(steps):
        for b in batches(hard_ds, bs, device=model.device):
            o = model(b)
            loss = sample_losses(o["fields"], o["scalars"], b).mean()
            opt.zero_grad()
            loss.backward()
            model.moe.keys.grad[other] = 0.0
            opt.step()
            with torch.no_grad():
                model.moe.keys[other] = keys_before[other]       # Adam no toca a los residentes
            break
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        out = model(val, dense=True)
        loss_mix = sample_losses(out["fields"], out["scalars"], val)
        loss_j = {k: sample_losses(f, s, val) for k, (f, s) in out["per_expert"].items()}
        r, A = model.moe.ecology_stats(out["pi"], loss_mix, loss_j)
        lam = model.moe.invasion_fitness(j, r, A)
    admitted = lam > 0
    if not admitted:
        model.moe.active[j] = False
        model.moe.n[j] = 0.0
    log.setdefault("invasions", []).append({"padre": parent, "mutante": j, "Lambda": lam, "admitido": admitted})
    return admitted


@torch.no_grad()
def write_memory(model, ds, frac=0.1):
    """Durante el entrenamiento la memoria guarda estados de alto error (para novedad).

    Las claves cambian mientras el codificador aprende, así que estas entradas
    NO se usan para corregir predicciones; solo para medir novedad y guardar fósiles.
    """
    if not model.use_memory:
        return 0
    model.eval()
    ds = to_device(ds, model.device)
    out = model(ds)
    resid = ds["scalars"] - out["scalars"]
    err = resid.abs().sum(-1)
    k = max(1, int(frac * err.numel()))
    idx = err.topk(k).indices
    model.memory.write(out["pooled"][idx], resid[idx])
    return k


@torch.no_grad()
def calibrate_memory(model, calib, val, betas=(4.0, 8.0, 16.0, 32.0)):
    """Corrección no paramétrica post-entrenamiento (codificador congelado).

    Escribe los residuos del conjunto de calibración y activa la corrección solo
    si reduce el error escalar en validación. Devuelve (mejora relativa, β elegido).
    """
    if not model.use_memory:
        return 0.0, None
    model.eval()
    fossils = model.memory.tags == 1
    model.memory.keys, model.memory.vals, model.memory.tags = (model.memory.keys[fossils],
                                                               model.memory.vals[fossils], model.memory.tags[fossils])
    model.memory_correction = False
    base = evaluate(model, val)
    calib = to_device(calib, model.device)
    out = model(calib)
    model.memory.write(out["pooled"], calib["scalars"] - out["scalars"])
    best, best_beta = base["log_eta_MAE"] + base["cfl_MAE"], None
    model.memory_correction = True
    for b in betas:
        model.memory.beta = b
        for gate in (0.25, 0.5, 1.0):
            model.mem_gate.fill_(float(torch.logit(torch.tensor(gate * 0.999))))
            ev = evaluate(model, val)
            e = ev["log_eta_MAE"] + ev["cfl_MAE"]
            if e < best:
                best, best_beta = e, (b, gate)
    if best_beta is None:
        model.memory_correction = False
        return 0.0, None
    model.memory.beta = best_beta[0]
    model.mem_gate.fill_(float(torch.logit(torch.tensor(best_beta[1] * 0.999))))
    return 1 - best / (base["log_eta_MAE"] + base["cfl_MAE"]), best_beta


def fit(model, train, val, epochs=20, bs=32, lr=2e-3, invade_every=4, eco_warmup=5, log=None,
        verbose=True, replay=None):
    """eco_warmup: épocas sin extinción ni invasión (los expertos aún no se diferencian).
    replay: dataset pequeño de tareas anteriores que se mezcla en cada época (repetición)."""
    log = log if log is not None else {}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    from .data.synthetic import concat
    data = train if replay is None else concat(train, replay)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for b in batches(data, bs, device=model.device):
            out = model(b)
            loss = sample_losses(out["fields"], out["scalars"], b).mean()
            if not model.use_ecology:
                loss = loss + 0.01 * load_balance_loss(out["pi"], model.moe.top_k)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * b["points"].shape[0]
        if model.use_ecology:
            warm = ep >= eco_warmup
            ecology_epoch(model, val, log, allow_extinction=warm)
            if warm and invade_every and (ep + 1) % invade_every == 0:
                try_invasion(model, None, train, val, log)
        if model.use_memory and (ep + 1) % 2 == 0:
            write_memory(model, train)
        ev = evaluate(model, val)
        ev["epoca"], ev["perdida_train"] = ep, tot / data["points"].shape[0]
        ev["especies"] = int(model.moe.active.sum())
        ev["memoria"] = len(model.memory)
        log.setdefault("epocas", []).append(ev)
        if verbose:
            print(f"ep {ep:3d} | train {ev['perdida_train']:.4f} | campos {ev['campos_L2rel']:.4f} | "
                  f"logη {ev['log_eta_MAE']:.4f} | δ/R {ev['cfl_MAE']:.4f} | especies {ev['especies']} | mem {ev['memoria']}")
    return log
