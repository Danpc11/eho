"""Regla de parada por forrajeo (v0.8): cuándo dejar de simular.

Problema (v0.7, Burgers): la colmena paraba tras 3 simulaciones (error 0.997) cuando la regla codiciosa,
con 16, llegaba a 0.599. Usaba el QUÓRUM para dos cosas distintas: elegir candidato y decidir parar.
Las hormigas Temnothorax usan el quórum para comprometerse con un sitio ya evaluado (Pratt y Sumpter,
PNAS 2006); los animales que forrajean deciden cuándo dejar un parche con otra regla.

Teorema del valor marginal (Charnov, Theor. Popul. Biol. 1976): con ganancia acumulada g(n) tras n
simulaciones y costo fijo τ de una campaña (en unidades de simulación), la tasa media es
    R*(n) = g(n) / (n + τ)
y conviene seguir mientras la ganancia marginal supere esa tasa: g'(n) ≥ R*(n). En el óptimo n*,
g'(n*) = R*(n*), y la regla maximiza la ganancia por unidad de cómputo (propiedad demostrable).

Aquí la ganancia de una simulación es lo que se aprendió de ella, medido ANTES de reentrenar con la
inspección del sitio: v = max(s - 1, 0), donde s es el error del caso normalizado por el error típico
de lo conocido (s ≈ 1 = ya se predecía como lo conocido: simularlo no enseñó nada). g'(n) se estima con
una media móvil exponencial de v.

Reformulación v0.9. Con v = max(s - 1, 0), la mitad de los casos CONOCIDOS contaban como ganancia (la
mediana de s en lo conocido es 1), y en una de dos semillas la regla no paraba aunque ya no llegaban
casos nuevos. Ahora v = max(s / s_ref - 1, 0), con s_ref = percentil 90 del error normalizado en lo
conocido (calibrado al entrenar): solo enseña algo un caso que se predice peor que el 90 % de lo conocido.

Tiempo de rendición (giving-up time; McNamara, Am. Nat. 1982): además de g' < R*, se exige que hayan
pasado `rendicion` rondas sin ninguna simulación con v > umbral_util. Y nunca se para antes de `n_min`
simulaciones (arranque en frío: sin evidencia, parar no está justificado).
"""
from dataclasses import dataclass, field


@dataclass
class ReglaParada:
    tau: float = 8.0            # costo de la campaña en simulaciones (mayor = más paciente)
    beta: float = 0.5           # memoria de la media móvil de la ganancia marginal
    rendicion: int = 2          # rondas sin simulaciones útiles antes de rendirse
    umbral_util: float = 0.1    # v mínimo para que una simulación cuente como útil
    referencia: float = 1.0     # v0.9: error normalizado a partir del cual un caso enseña algo
    n_min: int = 4              # simulaciones mínimas antes de poder parar
    ganancias: list = field(default_factory=list)
    marginal: float | None = None
    rondas_sin_util: int = 0
    parada: bool = False

    @property
    def n(self):
        return len(self.ganancias)

    @property
    def tasa_media(self):
        return sum(self.ganancias) / (self.n + self.tau)

    def registrar(self, s_por_simulacion):
        """s_por_simulacion: error normalizado (antes de aprender) de cada simulación de la ronda."""
        util = False
        for s in s_por_simulacion:
            v = max(float(s) / self.referencia - 1.0, 0.0)
            self.ganancias.append(v)
            self.marginal = v if self.marginal is None else self.beta * self.marginal + (1 - self.beta) * v
            util |= v > self.umbral_util
        self.rondas_sin_util = 0 if util else self.rondas_sin_util + 1
        return self

    def continuar(self) -> bool:
        if self.parada:
            return False
        if self.n < self.n_min or self.marginal is None:
            return True
        if self.marginal >= self.tasa_media:
            return True
        if self.rondas_sin_util < self.rendicion:
            return True
        self.parada = True
        return False

    def estado(self):
        return {"simulaciones": self.n, "marginal": self.marginal, "tasa_media": self.tasa_media,
                "rondas_sin_util": self.rondas_sin_util, "parada": self.parada}
