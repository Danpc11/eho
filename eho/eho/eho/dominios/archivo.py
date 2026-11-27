"""Dominio respaldado por archivos (v0.7): un conjunto de simulaciones YA corridas como reserva.

Sirve para todo lo que no se genera al vuelo: la reserva de 300-600 simulaciones DPD de sangre, The Well
convertido con scripts/convertir_the_well.py, o mediciones. Cada régimen es un archivo .pt con el formato
de eho/dominios/base.py (diccionario de tensores con "episode"). `generate` toma episodios COMPLETOS del
archivo, sin reemplazo dentro de una llamada, así que el aprendizaje activo "simula" revelando episodios
de la reserva, que es exactamente el protocolo del artículo.

    dominio:
      nombre: archivo
      params: {archivos: {sano: datos/dpd_sano.pt, dt2: datos/dpd_dt2.pt}}
      regimenes: [sano]

Las dimensiones (d_point, d_cond, n_fields, n_scalars) se leen del primer archivo. Si el archivo trae
una clave "meta" (dict) con "expert", "extra_key" o "memory_keys", se respetan.
"""
import torch

from .base import Dominio, registrar_dominio, subset


@registrar_dominio("archivo")
class DominioArchivo(Dominio):
    name = "archivo"

    def __init__(self, archivos: dict, episodios_excluidos: dict | None = None):
        """episodios_excluidos: {régimen: [ids]} que esta instancia nunca devuelve (p. ej. los de prueba)."""
        self.archivos = dict(archivos)
        self.regimes = list(self.archivos)
        self.excluidos = {k: set(v) for k, v in (episodios_excluidos or {}).items()}
        self._cache = {}
        ds = self._load(self.regimes[0])
        self.d_point = ds["points"].shape[-1]
        self.d_cond = ds["cond"].shape[-1]
        self.n_fields = ds["fields"].shape[-1] if ds["fields"].dim() == 3 else 0
        self.n_scalars = ds["scalars"].shape[-1]
        meta = getattr(self, "_meta", {}) or {}
        self.expert = meta.get("expert", "generico")
        self.extra_key = meta.get("extra_key")
        self.memory_keys = meta.get("memory_keys")

    def _load(self, regime):
        if regime not in self._cache:
            d = torch.load(self.archivos[regime], weights_only=False)
            self._meta = d.pop("meta", None) if isinstance(d.get("meta"), dict) else None
            self._cache[regime] = d
        return self._cache[regime]

    def episodios(self, regime):
        eps = self._load(regime)["episode"].unique()
        ex = self.excluidos.get(regime, set())
        return eps[~torch.isin(eps, torch.tensor(sorted(ex), dtype=eps.dtype))] if ex else eps

    def generate(self, n_episodes, per_episode, regime, seed=0):
        """Toma n_episodes episodios al azar (per_episode se ignora: se devuelven completos) y los
        renumera 0..n-1. La clave "episodio_original" guarda el id en el archivo."""
        ds = self._load(regime)
        eps = self.episodios(regime)
        g = torch.Generator().manual_seed(seed)
        sel = eps[torch.randperm(eps.numel(), generator=g)[: min(n_episodes, eps.numel())]]
        parts = []
        for new_id, e in enumerate(sel.tolist()):
            p = subset(ds, torch.nonzero(ds["episode"] == e).flatten())
            p["episodio_original"] = p["episode"].clone()
            p["episode"] = torch.full_like(p["episode"], new_id)
            parts.append(p)
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    def memory_values(self, batch):
        if self.memory_keys:
            return torch.cat([batch[k].reshape(batch[k].shape[0], -1).float() for k in self.memory_keys], -1)
        return batch["scalars"]
