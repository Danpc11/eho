"""Gemelo EHO: la pieza que GENERA y usa modelos a partir de una configuración.

    from eho import Gemelo
    g = Gemelo.desde_config("configs/sangre.yaml")
    g.entrenar()                          # genera o carga datos, entrena la colmena y la calibra
    r = g.predecir(consulta, contexto)    # predicción colectiva con abstención
    elegidos = g.proponer(candidatos)     # qué simular (o medir) después; puede ser "nada"
    g.incorporar(nuevos)                  # aprende lo nuevo sin olvidar lo anterior
    g.guardar("modelos/hemotwin.eho")
    g = Gemelo.cargar("modelos/hemotwin.eho")

Los tres módulos del marco se arman aquí:
    IA central   EHOCentral (eho/central.py), uno por agente
    memoria      episódica + neuronal de largo plazo + plasmodial (eho/memory)
    colmena      decisiones colectivas para predecir y para elegir qué simular (eho/colony)
El dominio (sangre, advección-difusión, o uno nuevo) entra solo por la configuración.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import torch

from . import default_device
from .config import Config
from .dominios.base import crear_dominio, concat, subset
from .hive import build_hive, predict_with_inspection
from .train_central import fit_central, evaluate_fewshot
from .colony import BeeColony, ColonyPredictor, colony_select, urgency_sigma
from .forrajeo import ReglaParada
from .active import SimulationValue

VERSION = "0.9.2"


def _dispositivo(nombre: str):
    if nombre == "auto":
        return default_device()
    return torch.device(nombre)


class Gemelo:
    def __init__(self, cfg: Config, agentes=None):
        self.cfg = cfg
        self.dominios = self._crear_dominios()
        self.agentes = agentes or []
        self.datos_entrenamiento = None      # por simulador; sirven para repasar (no se guardan por defecto)
        self.validacion = None
        self.predictor = None
        self.valor = None
        self.escala = None
        self.historial = []

    # ------------------------------------------------------------------ construcción
    @classmethod
    def desde_config(cls, cfg) -> "Gemelo":
        if not isinstance(cfg, Config):
            cfg = Config.desde_dict(cfg) if isinstance(cfg, dict) else Config.desde_archivo(cfg)
        return cls(cfg)

    def _crear_dominios(self):
        dc = self.cfg.dominio
        variantes = dc.simuladores or [{}]
        return [crear_dominio(dc.nombre, **{**dc.params, **v}) for v in variantes]

    def dominio_variante(self, i: int = 0, **cambios):
        """Una copia del dominio i con parámetros cambiados (p. ej. un régimen nuevo para probar)."""
        dc = self.cfg.dominio
        variantes = dc.simuladores or [{}]
        return crear_dominio(dc.nombre, **{**dc.params, **variantes[i % len(variantes)], **cambios})

    @property
    def dispositivo(self):
        return _dispositivo(self.cfg.entrenamiento.dispositivo)

    def _regimenes(self, dom):
        return self.cfg.dominio.regimenes or (dom.regimes[:1] if dom.regimes else ["base"])

    def generar_datos(self):
        """Datos de entrenamiento (uno por simulador) y validación: de archivo o del simulador del dominio."""
        dc, s = self.cfg.dominio, self.cfg.entrenamiento.semilla
        if dc.datos_entrenamiento:
            rutas = dc.datos_entrenamiento
            trains = [torch.load(rutas[i % len(rutas)], weights_only=False) for i in range(len(self.dominios))]
        else:
            trains = [concat(*[d.generate(dc.episodios_por_regimen, dc.muestras_por_episodio, r, 1000 * s + i)
                               for i, r in enumerate(self._regimenes(d))]) for d in self.dominios]
        if dc.datos_validacion:
            val = torch.load(dc.datos_validacion, weights_only=False)
        else:
            d0 = self.dominios[0]
            val = concat(*[d0.generate(dc.episodios_validacion, dc.muestras_por_episodio, r, 1000 * s + 50 + i)
                           for i, r in enumerate(self._regimenes(d0))])
        return trains, val

    def entrenar(self, trains=None, val=None, verbose=True):
        t0 = time.time()
        if trains is None or val is None:
            g_tr, g_val = self.generar_datos()
            trains, val = trains or g_tr, val if val is not None else g_val
        if isinstance(trains, dict):
            trains = [trains]
        self.datos_entrenamiento, self.validacion = trains, val
        m, c, e = self.cfg.modelo, self.cfg.colmena, self.cfg.entrenamiento
        # tamaño de lo que se guarda en memoria: lo define el dominio (memory_values)
        dv = m.d_val or int(self.dominios[0].memory_values(subset(trains[0], torch.arange(2))).shape[-1])
        if verbose:
            print(f"[{self.cfg.nombre}] entrenando {c.n_agentes} agentes en {self.dispositivo} "
                  f"({len(self.dominios)} fuente(s) de datos)", flush=True)
        self.agentes = build_hive(
            self.dominios, trains, val, n_agents=c.n_agentes, diverse=c.diversa, epochs=e.epocas, d_val=dv,
            seed=e.semilla, device=self.dispositivo, bootstrap=c.bootstrap, archs=c.arquitecturas or None,
            model_kwargs=dict(hopfield_steps=m.hopfield_steps, j_init=m.j_init, j_max=m.j_max, top_k=m.top_k,
                              memory=m.memoria, d=m.d, n_slices=m.n_slices, ecology=m.ecologia,
                              mpc_kwargs={"novelty_mode": m.novedad}),
            fit_kwargs=dict(lr=e.lr, episodes_per_batch=e.episodios_por_lote, eco_warmup=e.eco_warmup,
                            succession=m.sucesion, schedule=e.tasa),
            verbose=verbose)
        self.calibrar(val)
        self.historial.append({"evento": "entrenamiento", "segundos": round(time.time() - t0, 1)})
        if verbose:
            print(f"[{self.cfg.nombre}] listo en {time.time() - t0:.0f} s", flush=True)
        return self

    def datos_calibracion(self):
        """Episodios para la garantía LTT: de archivo o generados (semillas distintas de validación)."""
        dc, s = self.cfg.dominio, self.cfg.entrenamiento.semilla
        if dc.datos_calibracion:
            return torch.load(dc.datos_calibracion, weights_only=False)
        d0 = self.dominios[0]
        regs = self._regimenes(d0)
        n = max(1, dc.episodios_calibracion // len(regs))
        return concat(*[d0.generate(n, dc.muestras_por_episodio, r, 1000 * s + 77 + i) for i, r in enumerate(regs)])

    def calibrar(self, val, calib=None):
        """Quórum calibrado para predecir y escalas de novedad/error para decidir qué simular.

        Con colmena.calibracion = "ltt", la escala y τ salen de `val` y el umbral se certifica con
        OTROS episodios (`calib`), usando una consulta por episodio: así las observaciones son
        intercambiables y la garantía de Learn-then-Test es válida por paciente, no por muestra."""
        c = self.cfg.colmena
        P, Y, CE = predict_with_inspection(self.agentes, val, k=c.k_contexto)
        self.escala = (P.mean(0) - Y).abs().median(0).values.clamp(min=1e-4) * 2.0
        if c.calibracion == "ltt":
            calib = calib if calib is not None else self.datos_calibracion()
            Pc, Yc, CEc, epc = predict_with_inspection(self.agentes, calib, k=c.k_contexto, return_episodes=True)
            idx = torch.tensor([int(torch.nonzero(epc == e)[0]) for e in epc.unique()])
            self.predictor = ColonyPredictor(self.escala).fit(
                P, Y, CE, method="ltt", risk=c.riesgo, delta=c.delta, tolerance=c.tolerancia,
                calib=(Pc[:, idx], Yc[idx], CEc[:, idx]))
        else:
            self.predictor = ColonyPredictor(self.escala).fit(P, Y, CE, target_coverage=c.cobertura)
        self.valor = SimulationValue(self.agentes, k=c.k_contexto, use_error=self.cfg.activo.usar_error,
                                     decay=self.cfg.activo.olvido_error).fit(val)
        self.validacion = val
        return self

    # ------------------------------------------------------------------ uso
    @torch.no_grad()
    def predecir(self, consulta, contexto=None):
        """Predicción colectiva. contexto: muestras etiquetadas del MISMO episodio (paciente, fluido).
        Devuelve escalares (decisión de la colmena), campos (media), aceptado (bool por muestra),
        apoyo, y las predicciones de cada agente."""
        outs = [a(consulta, context=contexto) for a in self.agentes]
        P = torch.stack([o["scalars"].cpu() for o in outs])
        F = torch.stack([o["fields"].cpu() for o in outs])
        if contexto is not None and contexto["episode"].numel() > 1:
            from .active import inspection_errors
            ce = torch.stack([inspection_errors(a, contexto, k=self.cfg.colmena.k_contexto)[:, 0].mean()
                              for a in self.agentes])
            CE = ce.unsqueeze(-1).expand(-1, P.shape[1])
        else:
            CE = None
        pred, ok, apoyo = self.predictor.predict(P, CE)
        return {"escalares": pred, "campos": F.mean(0), "aceptado": ok, "apoyo": apoyo,
                "por_agente": P, "dispersion_campos": F.std(0)}

    @torch.no_grad()
    def novedad(self, lote):
        """Valor de novedad por muestra en [0, 1] (≈0 conocido, ≈1 nuevo), promedio de la colmena."""
        return self.valor.nov.values(lote).mean(0)

    def _regla(self):
        a = self.cfg.activo
        if getattr(self, "parada", None) is None:
            ref = 1.0
            if self.valor is not None and self.valor.err is not None and self.valor.err.q90 is not None:
                ref = max(float(self.valor.err.q90.mean()), 1.0)
            self.parada = ReglaParada(tau=a.tau_campana, rendicion=a.rendicion, n_min=a.n_min, referencia=ref)
        return self.parada

    def proponer(self, candidatos, presupuesto=None, semilla=0):
        """Elige, uno a uno con la colmena, los episodios candidatos que vale la pena simular.
        Puede devolver menos que el presupuesto (o ninguno).

        activo.parada = "quorum" (v0.7): la colmena elige y también decide parar (si no hay quórum, para).
        activo.parada = "mvt" (v0.8): se separan las dos decisiones. La colmena solo ELIGE (quórum =
        compromiso con un candidato; si se traba, se toma el de mayor valor medio). PARAR lo decide la
        regla del valor marginal con tiempo de rendición (eho/forrajeo.py). La inhibición cruzada crece
        con el presupuesto gastado y hay un descubrimiento espontáneo mínimo.
        Devuelve una lista de dicts {episodio, motivo, valor, novedad, error}."""
        a = self.cfg.activo
        presupuesto = presupuesto or a.presupuesto
        mvt = a.parada == "mvt"
        if mvt and not self._regla().continuar():
            return []
        V, eps, partes = self.valor.episode_values(candidatos)
        usadas = getattr(self, "_simulaciones_usadas", 0)
        elegidos, restantes = [], list(range(V.shape[1]))
        for t in range(presupuesto):
            if not restantes:
                break
            sigma = urgency_sigma(usadas + t, a.presupuesto_total) if mvt else 4.0
            col = BeeColony(quorum=self.cfg.colmena.quorum, sigma=sigma, t_max=60.0, noise=0.05,
                            min_value=a.valor_minimo, seed=1000 * semilla + t,
                            gamma_min=a.gamma_min if mvt else 0.0)
            c, _ = colony_select(V[:, restantes], col, normalize=False)
            motivo = "colmena"
            if c is None:
                if not mvt:
                    break
                c, motivo = int(V[:, restantes].mean(0).argmax()), "compromiso_forzado"
            j = restantes.pop(c)
            elegidos.append({"episodio": int(eps[j]), "motivo": motivo, "valor": float(V[:, j].mean()),
                             **{k: float(v[:, j].mean()) for k, v in partes.items()}})
        # exploradoras (v0.6): si la colmena no eligió nada, inspeccionar una vecindad aún no vista
        if not elegidos and a.exploradoras > 0 and "inspeccionado" in partes:
            sin_ver = [j for j in restantes if float(partes["inspeccionado"][:, j].mean()) < a.umbral_inspeccion]
            sin_ver.sort(key=lambda j: -float(V[:, j].mean()))
            for j in sin_ver[: a.exploradoras]:
                elegidos.append({"episodio": int(eps[j]), "motivo": "exploradora", "valor": float(V[:, j].mean()),
                                 **{k: float(v[:, j].mean()) for k, v in partes.items()}})
        return elegidos

    def incorporar(self, nuevos, verbose=False):
        """Aprende casos nuevos ya simulados/medidos.
        nuevos: un conjunto de datos (todos los agentes) o una lista con uno por agente (cada agente
        con los datos de SU simulador). Primero registra cuánto se equivocaba cada agente con ellos
        (inspección, antes de aprender), luego reentrena con ancla + repaso, y consolida la memoria."""
        a = self.cfg.activo
        lista = nuevos if isinstance(nuevos, (list, tuple)) else [nuevos] * len(self.agentes)
        if self.valor is not None and self.valor.err is not None:
            S = self.valor.register(list(lista))                     # [A, B] error normalizado, antes de aprender
            if S is not None:
                ep = lista[0]["episode"]
                s_ep = [float(S[:, ep == e].mean()) for e in ep.unique()]
                self._regla().registrar(s_ep)
        self._simulaciones_usadas = getattr(self, "_simulaciones_usadas", 0) + int(lista[0]["episode"].unique().numel())
        if not hasattr(self, "_acumulado"):
            self._acumulado = [[] for _ in self.agentes]
        for i, (m, nd) in enumerate(zip(self.agentes, lista)):
            self._acumulado[i].append(nd)
            todo = concat(*self._acumulado[i])
            datos = todo
            if self.datos_entrenamiento is not None and a.repeticion > 0:
                tr = self.datos_entrenamiento[i % len(self.datos_entrenamiento)]
                eps = tr["episode"].unique()
                g = torch.Generator().manual_seed(self.cfg.entrenamiento.semilla + i + 17 * len(self.historial))
                rep = eps[torch.randperm(eps.numel(), generator=g)[: a.repeticion]]
                datos = concat(subset(tr, torch.nonzero(torch.isin(tr["episode"], rep)).flatten()), todo)
            # la memoria se reconstruye con TODO lo que el agente conoce (entrenamiento + incorporado)
            completo = todo
            if self.datos_entrenamiento is not None:
                completo = concat(self.datos_entrenamiento[i % len(self.datos_entrenamiento)], todo)
            fit_central(m, datos, self.validacion, epochs=a.epocas_ajuste, lr=a.lr_ajuste, eco_warmup=1000,
                        anchor_lambda=a.ancla, memory_from=completo)
            m.consolidate(nd)                                  # memoria neuronal de largo plazo
            m.eval()
        if self.valor is not None:
            self.valor.refit_novelty(self.validacion)
        self.historial.append({"evento": "incorporar", "episodios": int(lista[0]["episode"].unique().numel())})
        return self

    def evaluar(self, ds, k=None):
        k = self.cfg.colmena.k_contexto if k is None else k
        res = [evaluate_fewshot(m, ds, k) for m in self.agentes]
        return {key: sum(r[key] for r in res) / len(res) for key in res[0]}

    def resumen(self):
        return {"nombre": self.cfg.nombre, "version": VERSION, "dominio": self.cfg.dominio.nombre,
                "agentes": [getattr(m, "hive_info", {}) for m in self.agentes],
                "parametros_por_agente": [sum(p.numel() for p in m.parameters()) for m in self.agentes],
                "historial": self.historial}

    # ------------------------------------------------------------------ persistencia
    def guardar(self, ruta, incluir_datos: bool = False):
        """Guarda la colmena completa (pesos + memorias + calibración).
        incluir_datos=False por defecto: los datos de entrenamiento NO se guardan dentro del modelo
        (importante con datos clínicos: un archivo de modelo no debe llevar registros de pacientes)."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        estado = {"version": VERSION, "config": self.cfg.a_dict(),
                  "agentes": [m.to("cpu") for m in self.agentes], "predictor": self.predictor,
                  "valor": self.valor, "escala": self.escala, "validacion": self.validacion,
                  "historial": self.historial,
                  "datos_entrenamiento": self.datos_entrenamiento if incluir_datos else None}
        torch.save(estado, ruta)
        for m in self.agentes:
            m.to(self.dispositivo)
        return ruta

    @classmethod
    def cargar(cls, ruta, dispositivo: str | None = None) -> "Gemelo":
        estado = torch.load(ruta, map_location="cpu", weights_only=False)
        cfg = Config.desde_dict(estado["config"])
        if dispositivo:
            cfg.entrenamiento.dispositivo = dispositivo
        g = cls(cfg, agentes=estado["agentes"])
        for m in g.agentes:
            m.to(g.dispositivo).eval()
        g.predictor, g.valor, g.escala = estado["predictor"], estado["valor"], estado["escala"]
        g.validacion, g.historial = estado["validacion"], estado["historial"]
        g.datos_entrenamiento = estado.get("datos_entrenamiento")
        if g.valor is not None:                                   # comparte los agentes cargados
            g.valor.nov.agents = g.agentes
            if g.valor.err is not None:
                g.valor.err.agents = g.agentes
        return g
