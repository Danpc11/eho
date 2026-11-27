"""Interfaz de dominio: lo único que hay que escribir para aplicar el EHO a otro problema.

Un dominio genera (o carga) datos organizados en EPISODIOS: un episodio comparte parámetros
ocultos (un paciente, un fluido, un material) y contiene varias muestras medidas en
condiciones visibles distintas. Así el modelo aprende a adaptarse con pocos ejemplos.

Formato de un conjunto de datos (diccionario de tensores):
    points   [B, N, d_point]   tokens de entrada (partículas, nodos de malla, pasos de tiempo...)
    cond     [B, d_cond]       condiciones VISIBLES
    fields   [B, N, n_fields]  salida por token (opcional: n_fields puede ser 0)
    scalars  [B, n_scalars]    salida global
    episode  [B]               identificador de episodio
    + claves extra que use el experto del dominio (p. ej. "gdot")
"""
import importlib
import torch

# Registro de dominios: el nombre que se escribe en el archivo de configuración -> la clase.
REGISTRO: dict[str, type] = {}


def registrar_dominio(nombre: str):
    """Decorador: @registrar_dominio("mi_enfermedad") sobre una subclase de Dominio."""
    def deco(cls):
        REGISTRO[nombre] = cls
        return cls
    return deco


def crear_dominio(nombre: str, **params):
    """Crea un dominio por nombre registrado o por ruta "paquete.modulo:Clase" (dominios propios
    fuera de este repo, sin tocar el código del EHO)."""
    if ":" in nombre:
        mod, cls = nombre.split(":")
        return getattr(importlib.import_module(mod), cls)(**params)
    if nombre not in REGISTRO:
        raise KeyError(f"dominio '{nombre}' no registrado; disponibles: {sorted(REGISTRO)}")
    return REGISTRO[nombre](**params)


def dominios_disponibles():
    return sorted(REGISTRO)


class Dominio:
    name = "base"
    d_point = 3
    d_cond = 1
    n_fields = 1
    n_scalars = 1
    expert = "generico"      # "generico" o un nombre que el núcleo sepa construir ("sangre")
    extra_key = None          # clave del lote que recibe el experto como `extra`
    regimes: list[str] = []

    def generate(self, n_episodes: int, per_episode: int, regime: str, seed: int = 0) -> dict:
        raise NotImplementedError

    def memory_values(self, batch: dict) -> torch.Tensor:
        """Resultados observados que se guardan en memoria (por defecto, los escalares)."""
        return batch["scalars"]

    def postprocess(self, scalars: torch.Tensor) -> torch.Tensor:
        """Restricciones de dominio sobre las salidas globales (por defecto, ninguna)."""
        return scalars


def concat(*dss):
    out = {}
    offset = 0
    for d in dss:
        for k, v in d.items():
            v = v + offset if k == "episode" else v
            out.setdefault(k, []).append(v)
        offset = int(out["episode"][-1].max()) + 1
    return {k: torch.cat(v) for k, v in out.items()}


def subset(ds, idx):
    return {k: v[idx] for k, v in ds.items()}


def split_episodes(ds, k_context: int, seed: int = 0):
    """Divide cada episodio en contexto (k_context muestras) y consulta (el resto)."""
    g = torch.Generator().manual_seed(seed)
    ctx, qry = [], []
    for e in ds["episode"].unique():
        idx = torch.nonzero(ds["episode"] == e).flatten()
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        ctx.append(idx[:k_context])
        qry.append(idx[k_context:])
    return torch.cat(ctx), torch.cat(qry)
