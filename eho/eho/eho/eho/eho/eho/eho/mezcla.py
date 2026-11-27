"""Árbitro rápido-lento (v0.8): combinar memoria y ajuste fino por precisión inversa.

Sistemas de aprendizaje complementarios (McClelland et al. 1995; Kumaran, Hassabis y McClelland 2016):
el hipocampo aprende rápido de pocos episodios y la neocorteza extrae estructura lenta. El cerebro
arbitra entre sistemas según su incertidumbre (Daw, Niv y Dayan, Nat. Neurosci. 2005).

En el EHO, la memoria gana con k ≤ 2 ejemplos y el ajuste fino con k = 5 (v0.3). En vez de elegir uno,
se combinan con pesos de varianza mínima, estimados por k y por salida en episodios de VALIDACIÓN:
        α_k = v_ajuste(k) / (v_memoria(k) + v_ajuste(k)),     ŷ = α_k ŷ_memoria + (1 - α_k) ŷ_ajuste
Si los errores de ambos fueran independientes, la combinación tendría varianza mínima; en la práctica
debe quedar en la envolvente inferior de las dos curvas de error contra k (prueba: exp_pocos_ejemplos).
"""
import torch

from .dominios.base import subset, split_episodes
from .model import sample_losses
from . import to_device


@torch.no_grad()
def memory_predictions(model, ds, k, seed=0):
    ctx, qry = split_episodes(ds, k, seed)
    model.eval()
    S, F, Y, T = [], [], [], []
    for e in ds["episode"].unique():
        q, c = qry[ds["episode"][qry] == e], ctx[ds["episode"][ctx] == e]
        out = model(subset(ds, q), context=subset(ds, c) if (k > 0 and model.memory is not None) else None)
        S.append(out["scalars"].cpu()); F.append(out["fields"].cpu())
        Y.append(ds["scalars"][q]); T.append(ds["fields"][q])
    return torch.cat(S), torch.cat(F), torch.cat(Y), torch.cat(T)


def finetune_predictions(factory, state, ds, k, steps=30, lr=1e-3, mode="todo", seed=0):
    ctx, qry = split_episodes(ds, k, seed)
    S, F = [], []
    for e in ds["episode"].unique():
        q, c = qry[ds["episode"][qry] == e], ctx[ds["episode"][ctx] == e]
        m = _copia(factory, state)
        if k > 0 and steps > 0:
            opt = torch.optim.Adam(m.moe.parameters() if mode == "cabeza" else m.parameters(), lr=lr)
            cb = to_device(subset(ds, c), m.device)
            m.train()
            for _ in range(steps):
                o = m(cb)
                loss = sample_losses(o["fields"], o["scalars"], cb).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            o = m(subset(ds, q))
        S.append(o["scalars"].cpu()); F.append(o["fields"].cpu())
    return torch.cat(S), torch.cat(F)


class ArbitroRapidoLento:
    def fit(self, mem_val, ft_val):
        """mem_val, ft_val: (escalares, campos, y, campos_obj) en validación para el mismo k."""
        Sm, Fm, Y, T = mem_val
        Sf, Ff = ft_val[:2]
        vm, vf = ((Sm - Y) ** 2).mean(0), ((Sf - Y) ** 2).mean(0)
        self.a_s = vf / (vm + vf).clamp(min=1e-12)                       # por salida escalar
        wm, wf = ((Fm - T) ** 2).mean(), ((Ff - T) ** 2).mean()
        self.a_f = float(wf / (wm + wf).clamp(min=1e-12)) if T.numel() else 0.5
        return self

    def combine(self, mem, ft):
        Sm, Fm = mem[:2]
        Sf, Ff = ft[:2]
        return self.a_s * Sm + (1 - self.a_s) * Sf, self.a_f * Fm + (1 - self.a_f) * Ff


class ArbitroMultiple:
    """Árbitro entre N predictores (v0.9): pesos proporcionales a la precisión (1 / error cuadrático medio
    en validación), por salida escalar y para los campos. Generaliza el árbitro rápido-lento: también
    arbitra entre memorias (en advección-difusión la episódica sola supera a la completa)."""

    def fit(self, preds_val, Y, T):
        """preds_val: lista de (escalares, campos) en validación."""
        inv_s = torch.stack([1.0 / ((S - Y) ** 2).mean(0).clamp(min=1e-12) for S, _ in [p[:2] for p in preds_val]])
        self.w_s = inv_s / inv_s.sum(0, keepdim=True)                       # [N, D]
        if T.numel():
            inv_f = torch.stack([1.0 / ((F - T) ** 2).mean().clamp(min=1e-12) for _, F in [p[:2] for p in preds_val]])
            self.w_f = inv_f / inv_f.sum()
        else:
            self.w_f = torch.full((len(preds_val),), 1.0 / len(preds_val))
        return self

    def combine(self, preds):
        S = sum(w * p[0] for w, p in zip(self.w_s, preds))
        F = sum(float(w) * p[1] for w, p in zip(self.w_f, preds))
        return S, F


def errors(S, F, Y, T):
    out = {"escalares_MAE": float((S - Y).abs().mean())}
    if T.numel():
        rel = torch.linalg.vector_norm(F - T, dim=(1, 2)) / torch.linalg.vector_norm(T, dim=(1, 2)).clamp(min=1e-6)
        out["campos_L2rel"] = float(rel.mean())
    return out


def _copia(factory, state):
    """Copia del modelo base. Si `state` es el propio modelo, se copia entero: sus memorias tienen
    buffers de tamaño variable (nodos, mapa de novedad) que load_state_dict no puede cargar en un
    modelo recién creado (error de la v0.7 en el ajuste fino)."""
    import copy
    if isinstance(state, torch.nn.Module):
        return copy.deepcopy(state)
    from .train_central import load_flexible
    return load_flexible(factory(), state)
