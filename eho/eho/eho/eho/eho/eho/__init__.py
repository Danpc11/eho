"""EcoHopfield Operator (EHO)."""
import os
import torch


def default_device() -> torch.device:
    """EHO_DEVICE=cpu|cuda|cuda:1 ; por defecto cuda si hay GPU."""
    name = os.environ.get("EHO_DEVICE")
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


__version__ = "0.9.7"


def __getattr__(nombre):
    # importación diferida: `from eho import Gemelo, Config` sin ciclos de importación
    if nombre == "Gemelo":
        from .gemelo import Gemelo
        return Gemelo
    if nombre == "Config":
        from .config import Config
        return Config
    raise AttributeError(nombre)
