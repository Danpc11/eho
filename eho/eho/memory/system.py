"""Sistema de memoria de tres partes (sistemas complementarios de aprendizaje).

| Memoria            | Escala de tiempo | Qué guarda                     | Cómo aprende                  |
|--------------------|------------------|--------------------------------|-------------------------------|
| Episódica          | una consulta     | ejemplos etiquetados recientes | atención (Hopfield), sin pesos |
| Neuronal (Titans)  | minutos-horas    | asociaciones clave → resultado | gradiente interno + sorpresa  |
| Plasmodial (MPC)   | días-semanas     | rutas y regiones consolidadas  | adaptación local tipo Physarum|
| Pesos del modelo   | épocas           | la física general              | retropropagación              |

Los VALORES guardados son resultados observados (estandarizados), no representaciones
aprendidas, para que no se vuelvan obsoletos cuando el modelo sigue entrenando.
La salida del sistema empieza en cero (capa final inicializada en cero): sin memoria,
el modelo se comporta igual que antes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .neural_ltm import NeuralLTM
from .plasmodial import PlasmodialMemory


class EpisodicContext(nn.Module):
    """Lectura en contexto: atención de las consultas sobre ejemplos etiquetados (Hopfield moderno).

    Cada ejemplo de contexto se codifica JUNTO con su condición, r_c = MLP([k_c, y_c]), como en
    los procesos neuronales atentos: así el modelo aprende qué dice un resultado sobre los
    parámetros ocultos dada la condición en que se midió. El codificador se aplica en cada
    lectura, de modo que nunca queda obsoleto.
    """

    def __init__(self, d_key: int, d_val: int = 0, d_enc: int = 32):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(2.0))
        self.d_key, self.d_enc = d_key, d_enc
        self.enc = nn.Sequential(nn.Linear(d_key + d_val, 64), nn.GELU(), nn.Linear(64, d_enc)) if d_val else None

    def forward(self, q, ck, cv, mask=None):
        """q [B,dk]; ck [C,dk]; cv [C,dv]; mask [B,C] (True = permitido).

        Devuelve lectura [B,dv] y confianza [B] (máxima similitud permitida; 0 si no hay contexto).
        """
        if ck is None or ck.shape[0] == 0:
            return q.new_zeros(q.shape[0], self.d_enc if self.enc is not None else 1), q.new_zeros(q.shape[0])
        s = F.normalize(q, dim=-1) @ F.normalize(ck, dim=-1).T                # coseno [B,C]
        logits = self.log_beta.exp() * s
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
            s = s.masked_fill(~mask, -1.0)
        has = torch.isfinite(logits).any(-1, keepdim=True)
        a = torch.softmax(torch.where(has, logits, torch.zeros_like(logits)), -1) * has
        conf = torch.where(has.squeeze(-1), s.max(-1).values, torch.zeros_like(s[:, 0]))
        vals = self.enc(torch.cat([ck, cv], -1)) if self.enc is not None else cv
        return a @ vals, conf


class MemorySystem(nn.Module):
    def __init__(self, d: int, d_val: int, d_key: int = 32, use_episodic=True, use_ltm=True, use_mpc=True,
                 mpc_kwargs=None, ltm_kwargs=None, episode_indexed: bool = True):
        """episode_indexed: las claves de las memorias de largo plazo combinan la entrada con el
        descriptor del episodio (lectura episódica del contexto). Sin él, esas memorias solo ven la
        condición visible y no pueden distinguir pacientes con parámetros ocultos distintos."""
        super().__init__()
        self.key_proj = nn.Sequential(nn.Linear(d, d_key), nn.LayerNorm(d_key))
        self.d_enc = 32
        self.episodic = EpisodicContext(d_key, d_val, self.d_enc) if use_episodic else None
        self.episode_indexed = episode_indexed and use_episodic
        self.ltm_key = nn.Sequential(nn.Linear(d_key + self.d_enc, d_key), nn.LayerNorm(d_key)) \
            if self.episode_indexed else None
        self.ltm = NeuralLTM(d_key, d_val, **(ltm_kwargs or {})) if use_ltm else None
        self.mpc = PlasmodialMemory(d_key, d_val, **(mpc_kwargs or {})) if use_mpc else None
        n_in = (self.d_enc + 1) + 2 * (d_val + 1)
        self.fuse = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)
        self.d_val = d_val
        # estadísticas para estandarizar valores (resultados observados)
        self.register_buffer("v_mean", torch.zeros(d_val))
        self.register_buffer("v_std", torch.ones(d_val))
        self.register_buffer("v_count", torch.zeros(()))
        self.context_keys = None      # memoria episódica explícita (adaptación en uso)
        self.context_vals = None

    @property
    def enabled(self):
        return any(m is not None for m in (self.episodic, self.ltm, self.mpc))

    @torch.no_grad()
    def _update_stats(self, v):
        n = self.v_count + v.shape[0]
        w = v.shape[0] / n
        self.v_mean.mul_(1 - w).add_(w * v.mean(0))
        self.v_std.mul_(1 - w).add_(w * v.std(0).clamp(min=0.05) if v.shape[0] > 1 else w * self.v_std)
        self.v_count.copy_(n)

    def standardize(self, v):
        return ((v - self.v_mean) / self.v_std.clamp(min=0.05)).clamp(-10.0, 10.0)

    def read(self, pooled, ctx_pooled=None, ctx_vals=None, ctx_mask=None):
        """Devuelve el vector de memoria [B, d] para sumar al estado del modelo."""
        q = self.key_proj(pooled)
        B, dv = q.shape[0], self.d_val
        parts = []
        # episódica: contexto del lote (entrenamiento) o contexto explícito (uso)
        if self.episodic is not None:
            mask = ctx_mask
            if ctx_pooled is not None:
                ck, cv = self.key_proj(ctx_pooled), self.standardize(ctx_vals)
            elif self.context_keys is not None:
                ck, cv, mask = self.context_keys, self.context_vals, None
            else:
                ck = cv = None
            if ck is not None:
                r_epi, conf = self.episodic(q, ck, cv, mask=mask)
            else:
                r_epi, conf = q.new_zeros(B, self.d_enc), q.new_zeros(B)
            parts += [r_epi, conf.unsqueeze(-1)]
        else:
            r_epi = q.new_zeros(B, self.d_enc)
            parts += [r_epi, q.new_zeros(B, 1)]
        kl = self.long_key(q, r_epi)
        # neuronal de largo plazo
        if self.ltm is not None and int(self.ltm.n_writes) > 0:
            parts += [self.ltm.read(kl), q.new_ones(B, 1)]
        else:
            parts += [q.new_zeros(B, dv), q.new_zeros(B, 1)]
        # plasmodial
        if self.mpc is not None and len(self.mpc) > 0:
            r, _, _ = self.mpc.read(kl)
            parts += [r, q.new_ones(B, 1)]
        else:
            parts += [q.new_zeros(B, dv), q.new_zeros(B, 1)]
        return self.fuse(torch.cat(parts, -1))

    def long_key(self, q, r_epi=None):
        """Clave de las memorias de largo plazo: entrada + descriptor del episodio."""
        if self.ltm_key is None:
            return q
        if r_epi is None:
            r_epi = q.new_zeros(q.shape[0], self.d_enc)
        return self.ltm_key(torch.cat([q, r_epi], -1))

    @torch.no_grad()
    def episode_descriptor(self, pooled, vals, episode):
        """Descriptor de cada muestra a partir de las DEMÁS muestras de su episodio."""
        if self.episodic is None:
            return None
        k = self.key_proj(pooled)
        mask = (episode.unsqueeze(1) == episode.unsqueeze(0)) & ~torch.eye(episode.numel(), dtype=torch.bool,
                                                                            device=episode.device)
        r, _ = self.episodic(k, k, self.standardize(vals), mask=mask)
        return r

    @torch.no_grad()
    def write(self, pooled, vals, epi_desc=None):
        """Consolida ejemplos etiquetados en las memorias de largo plazo (ignora lotes no finitos)."""
        if not (torch.isfinite(pooled).all() and torch.isfinite(vals).all()):
            return {"omitido": True}
        self._update_stats(vals)
        q = self.key_proj(pooled)
        k = self.long_key(q, epi_desc)
        pos = self.long_key(q, None)                 # v0.7: posición sin contexto, para la novedad
        v = self.standardize(vals)
        info = {}
        if self.ltm is not None:
            info["sorpresa"] = self.ltm.write(k, v)
        if self.mpc is not None:
            info.update(self.mpc.write(k, v, pos=pos))
        return info

    @torch.no_grad()
    def set_context(self, pooled, vals):
        """Adaptación en uso: fija ejemplos etiquetados como contexto episódico (sin gradiente)."""
        self.context_keys = self.key_proj(pooled)
        self.context_vals = self.standardize(vals)

    def clear_context(self):
        self.context_keys = self.context_vals = None

    @torch.no_grad()
    def novelty(self, pooled, mode=None):
        """Novedad estructural (memoria plasmodial) de la entrada, sin contexto.
        mode: "coseno" | "resistencia" | None (el de la memoria)."""
        q = self.long_key(self.key_proj(pooled))
        if self.mpc is not None and len(self.mpc) > 0:
            return self.mpc.novelty(q, mode)
        return torch.ones(q.shape[0], device=q.device)

    def novelty_key(self, pooled):
        """La clave en la que vive la memoria plasmodial (para comparar detectores en el MISMO espacio)."""
        return self.long_key(self.key_proj(pooled))
