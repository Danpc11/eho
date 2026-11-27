"""Sistema de memoria del EHO: episódica, neuronal de largo plazo (Titans) y plasmodial (Physarum)."""
from .neural_ltm import NeuralLTM
from .plasmodial import PlasmodialMemory, NoveltyMap
from .system import MemorySystem, EpisodicContext

__all__ = ["NeuralLTM", "PlasmodialMemory", "NoveltyMap", "MemorySystem", "EpisodicContext"]
