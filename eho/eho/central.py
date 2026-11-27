"""Módulo central de IA: el EHO de propósito general.

Se conecta a cualquier dominio (eho/dominios) y a los otros dos módulos:
    memoria (eho/memory)   episódica en contexto + neuronal de largo plazo + plasmodial
    colmena (eho/colony)   decisiones colectivas entre varios modelos centrales

tokens -> codificador -> rebanado -> núcleo de Hopfield (energía) -> estado global
       -> + memoria (lectura) -> ecosistema de expertos -> campos y escalares
"""
import torch
import torch.nn as nn

from .slicing import PhysicsSlicing
from .hopfield import EnergyHopfieldCore
from .ecology import EcoMoE, Expert, GenericExpert
from .memory import MemorySystem, NoveltyMap
from . import to_device


class EHOCentral(nn.Module):
    def __init__(self, dominio, d: int = 48, n_slices: int = 12, hopfield_steps: int = 3, j_init: int = 4,
                 j_max: int = 12, top_k: int = 2, memory: str = "completa", d_val: int | None = None,
                 mpc_kwargs=None, ltm_kwargs=None, ecology: bool = True):
        """memory: "ninguna" | "episodica" | "completa" (episódica + neuronal + plasmodial)."""
        super().__init__()
        self.dominio = dominio
        dc, nf, ns = dominio.d_cond, dominio.n_fields, dominio.n_scalars
        self.encoder = nn.Sequential(nn.Linear(dominio.d_point + dc, d), nn.GELU(), nn.Linear(d, d))
        self.slicing = PhysicsSlicing(d, n_slices)
        self.core = EnergyHopfieldCore(d, n_heads=4, d_head=12, n_patterns=32, steps=hopfield_steps, alpha=0.1)
        self.pool = nn.Sequential(nn.Linear(d + dc, d), nn.GELU(), nn.Linear(d, d))
        if dominio.expert == "sangre":
            factory = lambda: Expert(d, dc, nf)
        else:
            factory = lambda: GenericExpert(d, dc, nf, ns)
        self.moe = EcoMoE(d, dc, n_fields=nf, j_init=j_init, j_max=j_max, top_k=top_k, expert_factory=factory)
        self.use_ecology = ecology
        self.memory_mode = memory
        if memory == "ninguna":
            self.memory = None
        else:
            full = memory == "completa"
            dv = d_val or dominio.n_scalars
            self.memory = MemorySystem(d, dv, use_episodic=True, use_ltm=full, use_mpc=full,
                                       mpc_kwargs=mpc_kwargs, ltm_kwargs=ltm_kwargs)
        # mapa de novedad: todos los agentes guardan "dónde han estado". Usa el espacio de claves de la
        # memoria (proyectado y normalizado) si existe: separa lo nuevo mucho mejor que el estado crudo.
        self.novelty_map = NoveltyMap(self.memory.key_proj[0].out_features if self.memory is not None else d,
                                      thresh=0.01)

    @property
    def device(self):
        return next(self.parameters()).device

    def embed(self, batch, return_energy=False):
        pts, cond = batch["points"], batch["cond"]
        B, N, _ = pts.shape
        h = self.encoder(torch.cat([pts, cond.unsqueeze(1).expand(B, N, -1)], -1))
        z, w = self.slicing(h)
        out = self.core(z, return_energy=return_energy)
        zT, energies = out if return_energy else (out, None)
        h_hat = self.slicing.deslice(zT, w) + h
        pooled = self.pool(torch.cat([self.core.norm(zT).mean(1), cond], -1))
        return h_hat, pooled, energies

    def forward(self, batch, dense=False, return_energy=False, episodic_self=False, context=None):
        """episodic_self: en entrenamiento, cada muestra lee de las DEMÁS muestras de su episodio.
        context: lote etiquetado del mismo episodio para adaptación en uso (sin gradiente)."""
        batch = to_device(batch, self.device)
        h_hat, pooled, energies = self.embed(batch, return_energy)
        pooled0 = pooled
        if self.memory is not None:
            if episodic_self:
                ep = batch["episode"]
                mask = (ep.unsqueeze(1) == ep.unsqueeze(0)) & ~torch.eye(ep.numel(), dtype=torch.bool, device=ep.device)
                if self.training:
                    # tamaño de contexto variable (incluido 0) para que el modelo funcione con k = 0, 1, 2, ...
                    keep_p = torch.rand(ep.numel(), 1, device=ep.device)
                    keep_p = torch.where(torch.rand_like(keep_p) < 0.2, torch.zeros_like(keep_p), keep_p)
                    mask = mask & (torch.rand(mask.shape, device=ep.device) < keep_p)
                m = self.memory.read(pooled, pooled, self.dominio.memory_values(batch), mask)
            elif context is not None:
                context = to_device(context, self.device)
                _, cp, _ = self.embed(context)
                m = self.memory.read(pooled, cp, self.dominio.memory_values(context), None)
            else:
                m = self.memory.read(pooled)
            pooled = pooled + m
        extra = batch.get(self.dominio.extra_key) if self.dominio.extra_key else None
        fields, scal, pi, per_expert = self.moe(h_hat, pooled, batch["cond"], extra, dense=dense)
        scal = self.dominio.postprocess(scal)
        return {"fields": fields, "scalars": scal, "pi": pi, "pooled": pooled0,
                "per_expert": per_expert, "energies": energies}

    @torch.no_grad()
    def compact(self):
        """Elimina del modelo los expertos extintos (v0.9). La ecología deja 2 de 6 especies, pero los
        6 expertos seguían guardados: son el 92 % de los parámetros. Las predicciones no cambian (los
        extintos tienen peso cero en el enrutador). Después de compactar no hay invasiones posibles:
        úselo para guardar o desplegar, no a mitad de un entrenamiento con invasión."""
        moe = self.moe
        keep = torch.nonzero(moe.active).flatten()
        j = moe.active.numel()
        if keep.numel() == j:
            return 0
        antes = sum(p.numel() for p in self.parameters())
        moe.experts = torch.nn.ModuleList([moe.experts[i] for i in keep.tolist()])
        moe.keys = torch.nn.Parameter(moe.keys.data[keep].clone())
        for name, buf in list(moe.named_buffers(recurse=False)):
            if buf.dim() >= 1 and buf.shape[0] == j:
                setattr(moe, name, buf[keep].clone())
        for attr in ("j_max", "n_experts"):
            if hasattr(moe, attr):
                setattr(moe, attr, int(keep.numel()))
        return antes - sum(p.numel() for p in self.parameters())

    def _map_key(self, pooled):
        return self.memory.key_proj(pooled) if self.memory is not None else pooled

    @torch.no_grad()
    def map_visit(self, batch):
        """Registra en el mapa de novedad los estados visitados (no requiere etiquetas)."""
        batch = to_device(batch, self.device)
        _, pooled, _ = self.embed(batch)
        return self.novelty_map.write(self._map_key(pooled))

    @torch.no_grad()
    def map_novelty(self, batch):
        batch = to_device(batch, self.device)
        _, pooled, _ = self.embed(batch)
        return self.novelty_map.novelty(self._map_key(pooled))

    @torch.no_grad()
    def consolidate(self, batch, pooled=None):
        """Escribe ejemplos etiquetados en las memorias de largo plazo (neuronal y plasmodial)
        y registra los estados en el mapa de novedad.
        pooled (v0.9): estado global ya calculado en el paso hacia adelante del mismo lote; evita volver
        a pasar por el núcleo de Hopfield (un tercio de los pasos por el núcleo en entrenamiento)."""
        batch = to_device(batch, self.device)
        if pooled is None:
            _, pooled, _ = self.embed(batch)
        self.novelty_map.write(self._map_key(pooled))
        if self.memory is None:
            return {}
        vals = self.dominio.memory_values(batch)
        desc = self.memory.episode_descriptor(pooled, vals, batch["episode"]) if "episode" in batch else None
        return self.memory.write(pooled, vals, epi_desc=desc)

    @torch.no_grad()
    def novelty(self, batch, mode=None):
        """mode: "coseno" | "resistencia" | None (el configurado en la memoria plasmodial)."""
        batch = to_device(batch, self.device)
        _, pooled, _ = self.embed(batch)
        if self.memory is None:
            return torch.ones(pooled.shape[0], device=pooled.device)
        return self.memory.novelty(pooled, mode)

    @torch.no_grad()
    def representations(self, batch):
        """Representaciones para comparar detectores de novedad en el mismo espacio:
        pooled (estado global), key (clave de la memoria plasmodial) y map (clave del mapa de novedad)."""
        batch = to_device(batch, self.device)
        _, pooled, _ = self.embed(batch)
        out = {"pooled": pooled, "map": self._map_key(pooled)}
        if self.memory is not None:
            out["key"] = self.memory.novelty_key(pooled)
        return out
