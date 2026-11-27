"""Plantilla que escribe `python -m eho nuevo-dominio <nombre>`.

El ejemplo funciona tal cual (un oscilador amortiguado: cada episodio es un "sistema" con
amortiguamiento oculto) para que se pueda entrenar de inmediato y luego reemplazar por el
problema real: otra enfermedad, otro fluido, otro material.
"""

PLANTILLA = '''"""Dominio {nombre} para el EHO.

QUÉ ES UN EPISODIO: un conjunto de muestras que comparten parámetros OCULTOS
(un paciente, un fluido, un material). El modelo aprende a inferirlos con pocos ejemplos.

Reemplaza `generate` por tu simulador o por la lectura de tus datos y ajusta las dimensiones.
Para datos que ya tienes en disco no hace falta `generate`: guarda un diccionario de tensores con
torch.save(...) en el formato de abajo y apúntalo en la configuración (datos_entrenamiento).

Formato (B muestras, N tokens por muestra):
    points   [B, N, d_point]   entradas por token (nodos, partículas, tiempos, genes...)
    cond     [B, d_cond]       condiciones VISIBLES de la medición
    fields   [B, N, n_fields]  salida por token (n_fields puede ser 0: tensor [B, N, 0])
    scalars  [B, n_scalars]    salidas globales
    episode  [B]               id de episodio (entero)
"""
import math
import torch

from eho.dominios import Dominio, registrar_dominio


@registrar_dominio("{nombre}")
class {clase}(Dominio):
    name = "{nombre}"
    d_point = 2          # por token: (t, señal de entrada)
    d_cond = 1           # visible: frecuencia de excitación
    n_fields = 1         # salida por token: respuesta x(t)
    n_scalars = 2        # salidas globales: energía media, pico
    expert = "generico"
    regimes = ["bajo", "alto"]

    RANGOS = {{"bajo": (0.1, 0.4), "alto": (0.6, 1.2), "nuevo": (2.0, 3.0)}}   # amortiguamiento oculto

    def __init__(self, n_points: int = 64):
        self.n_points = n_points

    def generate(self, n_episodes, per_episode, regime, seed=0):
        g = torch.Generator().manual_seed(seed)
        lo, hi = self.RANGOS[regime]
        n = n_episodes * per_episode
        zeta = (lo + (hi - lo) * torch.rand(n_episodes, generator=g)).repeat_interleave(per_episode)
        w = 4 + 8 * torch.rand(n, generator=g)                                   # visible
        t = torch.linspace(0, 1, self.n_points)
        u = torch.sin(2 * math.pi * 3 * t).expand(n, -1)                         # entrada
        x = torch.exp(-zeta.view(-1, 1) * w.view(-1, 1) * t) * torch.cos(w.view(-1, 1) * t)
        pts = torch.stack([t.expand(n, -1), u], -1)
        scal = torch.stack([x.pow(2).mean(-1), x.abs().amax(-1)], -1)
        return {{"points": pts, "cond": (w / 12).unsqueeze(-1), "fields": x.unsqueeze(-1),
                "scalars": scal, "episode": torch.arange(n_episodes).repeat_interleave(per_episode)}}

    # Opcional: qué se guarda en memoria (por defecto, los escalares). Conviene que sean cantidades
    # que delaten los parámetros ocultos del episodio.
    # def memory_values(self, batch):
    #     return batch["scalars"]

    # Opcional: restricciones físicas sobre las salidas (p. ej. positividad).
    def postprocess(self, scalars):
        return torch.nn.functional.softplus(scalars)
'''

CONFIG = '''nombre: {nombre}
dominio:
  nombre: "{modulo}:{clase}"
  params: {{}}
  regimenes: [bajo, alto]
  episodios_por_regimen: 40
  muestras_por_episodio: 8
colmena:
  n_agentes: 4
entrenamiento:
  epocas: 30
  dispositivo: auto
activo:
  presupuesto: 4
'''
