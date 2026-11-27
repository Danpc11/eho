"""Dominios enchufables del EHO. Para uno nuevo: `python -m eho nuevo-dominio <nombre>`."""
from .base import (Dominio, concat, subset, split_episodes, registrar_dominio, crear_dominio,
                   dominios_disponibles, REGISTRO)
from .sangre import Sangre
from .edp1d import AdveccionDifusion1D
from .burgers1d import Burgers1D
from .pdebench import PDEBench1D
from .archivo import DominioArchivo

__all__ = ["Dominio", "Sangre", "AdveccionDifusion1D", "Burgers1D", "PDEBench1D", "DominioArchivo", "concat", "subset", "split_episodes",
           "registrar_dominio", "crear_dominio", "dominios_disponibles", "REGISTRO"]
