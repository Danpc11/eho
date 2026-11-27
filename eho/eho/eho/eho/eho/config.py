"""Configuración declarativa de un modelo EHO: un archivo YAML (o JSON) describe todo el modelo.

Ejemplo mínimo (configs/sangre.yaml):

    nombre: hemotwin_dt2
    dominio:
      nombre: sangre                 # o "mi_paquete.mi_modulo:MiDominio"
      params: {n_points: 96}
      simuladores:                   # uno por "fuente de datos"; los agentes se reparten entre ellos
        - {simulador: quemada}
        - {simulador: carreau}
      regimenes: [sano, dt2]
    colmena: {n_agentes: 4}
    entrenamiento: {epocas: 30}

Las secciones que no se escriben toman los valores por defecto de abajo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path


@dataclass
class ConfigDominio:
    nombre: str = "sangre"
    params: dict = field(default_factory=dict)            # argumentos comunes del dominio
    simuladores: list = field(default_factory=list)       # variantes por fuente (diversidad de la colmena)
    regimenes: list = field(default_factory=list)         # regímenes para generar datos de entrenamiento
    episodios_por_regimen: int = 40
    muestras_por_episodio: int = 8
    episodios_validacion: int = 6
    # datos propios en lugar de generarlos: rutas a archivos .pt con el formato de dominios/base.py
    datos_entrenamiento: list = field(default_factory=list)   # una ruta por simulador (o una sola)
    datos_validacion: str | None = None
    datos_evolucion: str | None = None                        # v0.9.3: episodios para la selección de expertos
    datos_calibracion: str | None = None                      # v0.7: episodios para la garantía LTT
    episodios_calibracion: int = 60                           # si se generan (uno por paciente cuenta)


@dataclass
class ConfigModelo:
    d: int = 48
    n_slices: int = 12
    hopfield_steps: int = 3
    j_init: int = 4
    j_max: int = 12
    top_k: int = 2
    memoria: str = "completa"          # ninguna | episodica | completa
    d_val: int | None = None           # tamaño de lo que se guarda en memoria (por defecto, los escalares)
    novedad: str = "coseno"            # v0.7: coseno | resistencia (Kirchhoff en la memoria plasmodial)
    ecologia: bool = True              # v0.7: False = expertos con enrutador fijo (ablación del núcleo)
    sucesion: int = 0                  # v0.8: épocas pioneras sin Lotka-Volterra ni extinciones


@dataclass
class ConfigColmena:
    n_agentes: int = 4
    diversa: bool = True
    bootstrap: float = 0.8
    # arquitecturas por agente (se reparten cíclicamente); vacío = las de eho/hive.py
    arquitecturas: list = field(default_factory=list)
    quorum: float = 0.5
    cobertura: float = 0.8             # fracción de casos conocidos que la colmena debe aceptar al predecir
    k_contexto: int = 2                # ejemplos del mismo episodio usados como contexto
    protocolos: list = field(default_factory=list)   # v0.9.6: números de mediciones de contexto con certificado
                                                     # propio, p. ej. [0, 1, 2, 5]; vacío = solo k_contexto
    prediccion: str = "media"          # v0.9.4: media | ganador | ponderada (ver RESULTADOS v0.9.4)
    calibracion: str = "cobertura"     # v0.7: cobertura | ltt (garantía de muestra finita)
    riesgo: float = 0.1                # ltt: P(error > tolerancia | aceptada) ≤ riesgo ...
    delta: float = 0.1                 # ... con probabilidad ≥ 1 - delta
    tolerancia: float = 1.0            # en unidades de la escala de la colmena


@dataclass
class ConfigEntrenamiento:
    epocas: int = 30
    lr: float = 2e-3
    episodios_por_lote: int = 4
    eco_warmup: int = 5
    tasa: str = "constante"            # v0.9: constante | coseno (decaimiento de la tasa de aprendizaje)
    semilla: int = 0
    dispositivo: str = "auto"          # auto | cpu | cuda | cuda:1 ...


@dataclass
class ConfigActivo:
    presupuesto: int = 4               # simulaciones máximas por ronda
    valor_minimo: float = 0.0          # piso duro opcional; con 0 la dinámica de la colmena decide si parar
    usar_error: bool = True            # v0.6: valor = novedad O error de inspección reciente
    exploradoras: int = 1              # si la colmena se abstiene: simulaciones de exploración por ronda
    umbral_inspeccion: float = 0.5     # c(x) por debajo = vecindad aún no inspeccionada
    olvido_error: float = 0.5          # peso que conservan los errores registrados en rondas anteriores
    ancla: float = 1e-2                # λ de la regularización contra el olvido
    repeticion: int = 60               # episodios antiguos que se repasan al aprender lo nuevo
    epocas_ajuste: int = 6
    lr_ajuste: float = 1e-3
    # v0.8: parar por forrajeo y no por quórum (eho/forrajeo.py)
    recalibrar: bool = True            # v0.9.3: recalibrar la abstención después de cada incorporar()
    parada: str = "mvt"                # quorum (v0.7) | mvt (valor marginal + tiempo de rendición)
    tau_campana: float = 8.0           # mvt: costo de la campaña en simulaciones
    rendicion: int = 2                 # mvt: rondas sin simulaciones útiles antes de parar
    n_min: int = 4                     # mvt: simulaciones mínimas
    presupuesto_total: int | None = 24  # tope global de la campaña (None = sin tope); también fija la urgencia σ(t)
    presupuesto_ejecuciones: int | None = None   # v0.9.5: tope de ejecuciones (condiciones x simuladores)
    presupuesto_costo: float | None = None        # v0.9.5: tope de costo (horas de GPU u otra unidad)
    permitir_repetidos: bool = False   # v0.9.6: incorporar dos veces la misma simulación es un error
    presupuesto_estricto: bool = True  # v0.9.4: incorporar() falla si se excede el tope
    ecologia_en_ajuste: bool = True    # v0.9.4: False = abundancias congeladas al incorporar
    gamma_min: float = 0.05            # descubrimiento espontáneo mínimo de la colmena


@dataclass
class Config:
    nombre: str = "eho"
    dominio: ConfigDominio = field(default_factory=ConfigDominio)
    modelo: ConfigModelo = field(default_factory=ConfigModelo)
    colmena: ConfigColmena = field(default_factory=ConfigColmena)
    entrenamiento: ConfigEntrenamiento = field(default_factory=ConfigEntrenamiento)
    activo: ConfigActivo = field(default_factory=ConfigActivo)

    _SECCIONES = {"dominio": ConfigDominio, "modelo": ConfigModelo, "colmena": ConfigColmena,
                  "entrenamiento": ConfigEntrenamiento, "activo": ConfigActivo}

    @classmethod
    def desde_dict(cls, d: dict) -> "Config":
        d = dict(d or {})
        kw = {"nombre": d.pop("nombre", "eho")}
        for sec, tipo in cls._SECCIONES.items():
            sub = d.pop(sec, None) or {}
            validos = {f.name for f in fields(tipo)}
            raros = set(sub) - validos
            if raros:
                raise KeyError(f"claves desconocidas en '{sec}': {sorted(raros)}; válidas: {sorted(validos)}")
            kw[sec] = tipo(**sub)
        if d:
            raise KeyError(f"secciones desconocidas: {sorted(d)}")
        return cls(**kw)

    @classmethod
    def desde_archivo(cls, ruta) -> "Config":
        ruta = Path(ruta)
        txt = ruta.read_text(encoding="utf-8")
        if ruta.suffix in (".yaml", ".yml"):
            import yaml
            return cls.desde_dict(yaml.safe_load(txt))
        return cls.desde_dict(json.loads(txt))

    def a_dict(self) -> dict:
        return {"nombre": self.nombre, **{s: asdict(getattr(self, s)) for s in self._SECCIONES}}

    def guardar(self, ruta):
        ruta = Path(ruta)
        if ruta.suffix in (".yaml", ".yml"):
            import yaml
            ruta.write_text(yaml.safe_dump(self.a_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        else:
            ruta.write_text(json.dumps(self.a_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
