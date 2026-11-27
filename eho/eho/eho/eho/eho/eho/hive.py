"""Construcción de colmenas: varios modelos centrales DIVERSOS + inspección del sitio.

Sin diversidad, los agentes se equivocan juntos y el acuerdo no informa nada (pensamiento de
grupo, observado en v0.2). La diversidad se introduce en cuatro ejes:
    datos        cada agente ve un subconjunto distinto de episodios (bootstrap 80 %)
    simulador    agentes entrenados con simuladores distintos (p. ej. Mirheo frente a HemoCell)
    arquitectura número de rebanadas y ancho distintos
    memoria      (v0.5) todos con memoria completa: la memoria plasmodial es la señal de novedad
"""
import torch

from .central import EHOCentral
from .train_central import fit_central
from .dominios.base import subset, split_episodes

ARCHS = [dict(n_slices=8, d=40), dict(n_slices=12, d=48), dict(n_slices=16, d=56), dict(n_slices=12, d=40)]
# v0.9.4: la diversidad de arquitectura es un DESPLAZAMIENTO respecto a la configuración base
# (modelo.d, modelo.n_slices). Con la base por defecto (48, 12) da exactamente ARCHS; con otra base la
# respeta (antes, ARCHS reemplazaba en silencio lo configurado).
ARCH_DELTAS = [dict(n_slices=-4, d=-8), dict(n_slices=0, d=0), dict(n_slices=4, d=8), dict(n_slices=0, d=-8)]


def diverse_archs(base_d: int, base_slices: int):
    return [dict(n_slices=max(2, base_slices + x["n_slices"]), d=max(8, base_d + x["d"])) for x in ARCH_DELTAS]
# v0.5: todos con memoria completa. Los agentes sin memoria plasmodial daban una señal de
# novedad débil y dividían a la colmena (2 contra 2). La diversidad queda en datos,
# simulador y arquitectura.
MEMS = ["completa", "completa", "completa", "completa"]


def build_hive(dominios, train_sets, val, n_agents=4, diverse=True, epochs=30, d_val=2, seed=0,
               device="cpu", bootstrap=0.8, archs=None, model_kwargs=None, fit_kwargs=None, verbose=False):
    """dominios / train_sets: listas (una por simulador); el agente a usa el índice a % len.

    diverse=False: todos con el mismo simulador (el primero), la misma arquitectura y memoria;
    solo cambia la semilla (colmena homogénea, para comparar).
    """
    model_kwargs = dict(model_kwargs or {})
    explicit = archs is not None and len(archs) > 0
    if not explicit:
        archs = diverse_archs(model_kwargs.get("d", 48), model_kwargs.get("n_slices", 12)) if diverse else \
            [dict(d=model_kwargs.get("d", 48), n_slices=model_kwargs.get("n_slices", 12))]
    mem_default = model_kwargs.pop("memory", "completa")
    fit_kwargs = {"eco_warmup": 5, **(fit_kwargs or {})}
    agents = []
    for a in range(n_agents):
        torch.manual_seed(1000 * seed + a)
        i = a % len(dominios) if diverse else 0
        dom, full = dominios[i], train_sets[i]
        eps = full["episode"].unique()
        keep = eps[torch.randperm(eps.numel())[: int(bootstrap * eps.numel())]] if diverse else eps
        tr = subset(full, torch.nonzero(torch.isin(full["episode"], keep)).flatten())
        arch = archs[a % len(archs)] if diverse else archs[min(1, len(archs) - 1) if explicit else 0]
        mem = mem_default
        kw = {**model_kwargs, **arch}
        m = EHOCentral(dom, memory=mem, d_val=d_val, **kw).to(device)
        fit_central(m, tr, val, epochs=epochs, **fit_kwargs)
        if verbose:
            print(f"  agente {a + 1}/{n_agents} listo ({getattr(dom, 'simulador', dom.name)}, {arch})", flush=True)
        m.eval()
        m.hive_info = {"simulador": getattr(dom, "simulador", dom.name), "memoria": mem, **arch}
        m.init_kwargs = dict(memory=mem, d_val=d_val, **kw)       # v0.9.6: para reconstruirlo sin pickle
        m.indice_dominio = i
        agents.append(m)
    return agents


@torch.no_grad()
def context_inspection(agents, ctx) -> torch.Tensor:
    """Error de inspección por agente [A]: cada medición del contexto se predice con TODAS las demás
    (leave-one-out). v0.9.7: es la ÚNICA implementación, usada en calibración, evaluación y despliegue
    (antes el despliegue usaba como mucho k_contexto auxiliares y la calibración todas: con 5 mediciones
    la señal de confianza no era la misma que la certificada)."""
    n = int(ctx["episode"].numel())
    errs = []
    for m in agents:
        m.eval()
        err = 0.0
        for i in range(n):
            other = torch.cat([torch.arange(i), torch.arange(i + 1, n)])
            o = m(subset(ctx, torch.tensor([i])), context=subset(ctx, other) if other.numel() else None)["scalars"].cpu()
            err += float((o - ctx["scalars"][i:i + 1]).abs().mean())
        errs.append(err / max(n, 1))
    return torch.tensor(errs)


@torch.no_grad()
def predict_with_inspection(agents, ds, k=2, seed=0, return_episodes=False):
    """Predicciones de consulta P [A,Bq,D], objetivos Y [Bq,D] y error de inspección CE [A,Bq]
    (y, con return_episodes, el episodio de cada consulta [Bq]).

    Inspección del sitio: cada agente predice cada ejemplo de contexto usando los DEMÁS como
    contexto; su error medio indica qué tan bien conoce este episodio (paciente, fluido).
    """
    ctx, qry = split_episodes(ds, k, seed)
    P, Y, CE, EP = [], [], [], []
    for e in ds["episode"].unique():
        q = qry[ds["episode"][qry] == e]
        c = ctx[ds["episode"][ctx] == e]
        if q.numel() == 0:
            continue
        P.append(torch.stack([m(subset(ds, q), context=subset(ds, c) if k > 0 else None)["scalars"].cpu()
                              for m in agents]))
        errs = context_inspection(agents, subset(ds, c)) if c.numel() else torch.zeros(len(agents))
        CE.append(errs.unsqueeze(-1).expand(-1, q.numel()))
        Y.append(ds["scalars"][q])
        EP.append(ds["episode"][q])
    out = (torch.cat(P, 1), torch.cat(Y), torch.cat(CE, 1))
    return out + (torch.cat(EP),) if return_episodes else out


@torch.no_grad()
def candidate_scores(agents, candidates, kappa: float = 1.0):
    """Valor de simular cada candidato, por agente: novedad de su memoria + desacuerdo con la colmena.
    Devuelve [A, C]. El desacuerdo es la distancia de la predicción del agente a la media de los demás."""
    nov = torch.stack([m.novelty(candidates).cpu() for m in agents])
    pred = torch.stack([m(candidates)["scalars"].cpu() for m in agents])          # [A,C,D]
    A = pred.shape[0]
    others = (pred.sum(0, keepdim=True) - pred) / max(A - 1, 1)
    dis = (pred - others).abs().mean(-1)
    z = lambda x: (x - x.mean()) / x.std().clamp(min=1e-6)
    return z(nov) + kappa * z(dis)
