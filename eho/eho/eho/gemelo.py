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

VERSION = "0.9.4"


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
        self.evolucion = self.datos_evolucion()
        self.agentes = build_hive(
            self.dominios, trains, val, n_agents=c.n_agentes, diverse=c.diversa, epochs=e.epocas, d_val=dv,
            seed=e.semilla, device=self.dispositivo, bootstrap=c.bootstrap, archs=c.arquitecturas or None,
            model_kwargs=dict(hopfield_steps=m.hopfield_steps, j_init=m.j_init, j_max=m.j_max, top_k=m.top_k,
                              memory=m.memoria, d=m.d, n_slices=m.n_slices, ecology=m.ecologia,
                              mpc_kwargs={"novelty_mode": m.novedad}),
            fit_kwargs=dict(lr=e.lr, episodes_per_batch=e.episodios_por_lote, eco_warmup=e.eco_warmup,
                            succession=m.sucesion, schedule=e.tasa, evo=self.evolucion),
            verbose=verbose)
        self.calibrar(val)
        self.historial.append({"evento": "entrenamiento", "segundos": round(time.time() - t0, 1)})
        if verbose:
            print(f"[{self.cfg.nombre}] listo en {time.time() - t0:.0f} s", flush=True)
        return self

    def datos_evolucion(self):
        """Episodios para la selección de expertos (extinción e invasión), distintos de la validación
        (v0.9.3). De archivo, o generados con otra semilla. Si los datos vienen de archivo y no hay
        archivo de evolución, devuelve None: la ecología usa la validación, que entonces forma parte del
        desarrollo del modelo y no debe citarse como evidencia independiente."""
        dc, s = self.cfg.dominio, self.cfg.entrenamiento.semilla
        if dc.datos_evolucion:
            return torch.load(dc.datos_evolucion, weights_only=False)
        if dc.datos_entrenamiento or dc.datos_validacion:
            return None
        d0 = self.dominios[0]
        return concat(*[d0.generate(dc.episodios_validacion, dc.muestras_por_episodio, r, 1000 * s + 60 + i)
                        for i, r in enumerate(self._regimenes(d0))])

    def datos_calibracion(self, ronda: int = 0):
        """Episodios para la garantía LTT: de archivo o generados (semillas distintas de validación).
        ronda: al recalibrar tras incorporar() se generan pacientes NUEVOS (otra semilla); con archivo
        se reutiliza el mismo conjunto, lo que debilita la independencia: provea más pacientes."""
        dc, s = self.cfg.dominio, self.cfg.entrenamiento.semilla
        if dc.datos_calibracion:
            return torch.load(dc.datos_calibracion, weights_only=False)
        d0 = self.dominios[0]
        regs = self._regimenes(d0)
        n = max(1, dc.episodios_calibracion // len(regs))
        return concat(*[d0.generate(n, dc.muestras_por_episodio, r, 1000 * s + 77 + i + 131 * ronda)
                        for i, r in enumerate(regs)])

    def calibrar(self, val, calib=None):
        """Quórum calibrado para predecir y escalas de novedad/error para decidir qué simular.

        Con colmena.calibracion = "ltt", la escala y τ salen de `val` y el umbral se certifica con
        OTROS episodios (`calib`), usando una consulta por episodio: así las observaciones son
        intercambiables y la garantía de Learn-then-Test es válida por paciente, no por muestra."""
        c = self.cfg.colmena
        self._calibrar_predictor(val, calib)
        self.valor = SimulationValue(self.agentes, k=c.k_contexto, use_error=self.cfg.activo.usar_error,
                                     decay=self.cfg.activo.olvido_error).fit(val)
        self.validacion = val
        return self

    def _calibrar_predictor(self, val, calib=None, ronda: int = 0):
        """Escala, τ y umbral de abstención con la versión ACTUAL de los agentes."""
        c = self.cfg.colmena
        P, Y, CE = predict_with_inspection(self.agentes, val, k=c.k_contexto)
        self.escala = (P.mean(0) - Y).abs().median(0).values.clamp(min=1e-4) * 2.0
        if c.calibracion == "ltt":
            calib = calib if calib is not None else self.datos_calibracion(ronda)
            Pc, Yc, CEc, epc = predict_with_inspection(self.agentes, calib, k=c.k_contexto, return_episodes=True)
            idx = torch.tensor([int(torch.nonzero(epc == e)[0]) for e in epc.unique()])
            self.predictor = ColonyPredictor(self.escala, mode=c.prediccion).fit(
                P, Y, CE, method="ltt", risk=c.riesgo, delta=c.delta, tolerance=c.tolerancia,
                calib=(Pc[:, idx], Yc[idx], CEc[:, idx]))
        else:
            self.predictor = ColonyPredictor(self.escala, mode=c.prediccion).fit(P, Y, CE, target_coverage=c.cobertura)
        self.calibracion_vigente = True
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
        # v0.9.3: los campos se combinan con los MISMOS pesos que respaldan la decisión escalar: cada
        # agente pesa según lo cerca que su predicción escalar quedó de la decisión de la colmena
        # (antes: media simple, que podía no corresponder al escalar aceptado). En abstención los
        # pesos son uniformes. La aceptación certifica el escalar, no el campo.
        vigente = getattr(self, "calibracion_vigente", True)
        if not vigente:                    # predicción exploratoria: nada se etiqueta como aceptado
            ok = torch.zeros_like(ok)
        d2 = (((P - pred.unsqueeze(0)) / self.predictor.scale) ** 2).mean(-1)          # [A, B]
        W = torch.softmax(-0.5 * d2, 0)
        W = torch.where(ok.unsqueeze(0), W, torch.full_like(W, 1.0 / W.shape[0]))
        campos = (W.unsqueeze(-1).unsqueeze(-1) * F).sum(0)
        return {"escalares": pred, "campos": campos, "aceptado": ok, "apoyo": apoyo,
                "por_agente": P, "dispersion_campos": F.std(0), "pesos_campos": W,
                "calibracion_vigente": getattr(self, "calibracion_vigente", True)}

    @torch.no_grad()
    def novedad(self, lote):
        """Valor de novedad por muestra en [0, 1] (≈0 conocido, ≈1 nuevo), promedio de la colmena."""
        return self.valor.nov.values(lote).mean(0)

    def presupuesto_restante(self):
        """Simulaciones que quedan del tope global de la campaña (None = sin tope)."""
        tot = self.cfg.activo.presupuesto_total
        return None if tot is None else max(tot - getattr(self, "_simulaciones_usadas", 0), 0)

    def _regla(self):
        a = self.cfg.activo
        if a.parada == "mvt" and not a.usar_error:
            raise ValueError("activo.parada = 'mvt' necesita activo.usar_error = true: la regla de parada mide "
                             "la ganancia con el error de inspección; sin él nunca se actualizaría")
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
        if presupuesto is None:
            presupuesto = a.presupuesto
        if presupuesto < 0:
            raise ValueError("presupuesto negativo")
        mvt = a.parada == "mvt"
        restante = self.presupuesto_restante()     # v0.9.4: el tope global vale en TODOS los modos
        if restante is not None:
            presupuesto = min(presupuesto, restante)
        if presupuesto == 0:
            return []
        if mvt and not self._regla().continuar():
            return []
        V, eps, partes = self.valor.episode_values(candidatos)
        usadas = getattr(self, "_simulaciones_usadas", 0)
        elegidos, restantes = [], list(range(V.shape[1]))
        for t in range(presupuesto):
            if not restantes:
                break
            sigma = urgency_sigma(usadas + t, a.presupuesto_total or 24) if mvt else 4.0
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
            for j in sin_ver[: min(a.exploradoras, presupuesto)]:
                elegidos.append({"episodio": int(eps[j]), "motivo": "exploradora", "valor": float(V[:, j].mean()),
                                 **{k: float(v[:, j].mean()) for k, v in partes.items()}})
        return elegidos

    def incorporar(self, nuevos, verbose=False):
        """Aprende casos nuevos ya simulados/medidos.
        nuevos: un conjunto de datos (todos los agentes) o una lista con uno por agente (cada agente
        con los datos de SU simulador). Primero registra cuánto se equivocaba cada agente con ellos
        (inspección, antes de aprender), luego reentrena con ancla + repaso, y consolida la memoria."""
        a = self.cfg.activo
        if getattr(self, "solo_despliegue", False):
            raise RuntimeError("Este modelo se cargó de un archivo de DESPLIEGUE (sin datos de repaso ni lo "
                               "incorporado antes). Para seguir aprendiendo, guarde con modo='reanudar'.")
        distintos = isinstance(nuevos, (list, tuple))
        lista = nuevos if distintos else [nuevos] * len(self.agentes)
        n_cond = int(lista[0]["episode"].unique().numel())
        restante = self.presupuesto_restante()
        if restante is not None and n_cond > restante:
            if a.presupuesto_estricto:
                raise ValueError(f"incorporar: {n_cond} simulaciones exceden el presupuesto restante ({restante})")
        # la calibración deja de valer ANTES de tocar los modelos: si el ajuste se interrumpe, el
        # estado queda marcado como no calibrado
        self.calibracion_vigente = False
        torch.manual_seed(1000 * self.cfg.entrenamiento.semilla + 7919 * (len(self.historial) + 1))
        if self.valor is not None and self.valor.err is not None:
            S = self.valor.register(list(lista))                     # [A, B] error normalizado, antes de aprender
            if S is not None:
                ep = lista[0]["episode"]
                s_ep = [float(S[:, ep == e].mean()) for e in ep.unique()]
                self._regla().registrar(s_ep)
        self._simulaciones_usadas = getattr(self, "_simulaciones_usadas", 0) + n_cond
        # costo real (v0.9.4): una condición física puede implicar una ejecución por simulador distinto
        n_sim = len(self.dominios) if distintos else 1
        self._ejecuciones = getattr(self, "_ejecuciones", 0) + n_cond * n_sim
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
                        anchor_lambda=a.ancla, memory_from=completo, evo=getattr(self, "evolucion", None),
                        ecology_update=a.ecologia_en_ajuste)
            m.consolidate(nd)                                  # memoria neuronal de largo plazo
            m.eval()
        if self.valor is not None:
            self.valor.refit_novelty(self.validacion)
        self.historial.append({"evento": "incorporar", "episodios": n_cond, "condiciones": n_cond,
                               "ejecuciones": n_cond * n_sim})
        if a.recalibrar and self.predictor is not None:
            self._calibrar_predictor(self.validacion, ronda=len(self.historial))
        return self

    @torch.no_grad()
    def evaluar_colectivo(self, ds, k=None):
        """Error del predictor REAL (v0.9.4). `evaluar` promedia los errores de los agentes, que no es
        el error de la predicción conjunta. Devuelve, para los escalares:
            MAE_agentes       promedio de los errores individuales
            MAE_ensemble      error de la media de las predicciones
            MAE_colmena       error de la decisión colectiva (según colmena.prediccion)
            MAE_aceptados     error de la colmena en lo que acepta
            cobertura         fracción aceptada
            riesgo_selectivo  fracción de aceptados con error escalado > tolerancia"""
        c = self.cfg.colmena
        k = c.k_contexto if k is None else k
        P, Y, CE = predict_with_inspection(self.agentes, ds, k=k)
        pred, ok, _ = self.predictor.predict(P, CE)
        if not getattr(self, "calibracion_vigente", True):
            ok = torch.zeros_like(ok)
        err = (pred - Y).abs().mean(-1)
        falla = ((pred - Y).abs() / self.predictor.scale).mean(-1) > c.tolerancia
        return {"MAE_agentes": float((P - Y).abs().mean()), "MAE_ensemble": float((P.mean(0) - Y).abs().mean()),
                "MAE_colmena": float(err.mean()),
                "MAE_aceptados": float(err[ok].mean()) if ok.any() else float("nan"),
                "cobertura": float(ok.float().mean()),
                "riesgo_selectivo": float(falla[ok].float().mean()) if ok.any() else float("nan")}

    def evaluar(self, ds, k=None):
        """Promedio de los errores INDIVIDUALES de los agentes (no el del predictor colectivo:
        para eso, `evaluar_colectivo`)."""
        k = self.cfg.colmena.k_contexto if k is None else k
        res = [evaluate_fewshot(m, ds, k) for m in self.agentes]
        return {key: sum(r[key] for r in res) / len(res) for key in res[0]}

    def resumen(self):
        return {"nombre": self.cfg.nombre, "version": VERSION, "dominio": self.cfg.dominio.nombre,
                "agentes": [getattr(m, "hive_info", {}) for m in self.agentes],
                "parametros_por_agente": [sum(p.numel() for p in m.parameters()) for m in self.agentes],
                "historial": self.historial}

    # ------------------------------------------------------------------ persistencia
    def guardar(self, ruta, incluir_datos: bool = False, modo: str | None = None):
        """Guarda la colmena completa (pesos + memorias + calibración + estado de la campaña activa).
        incluir_datos=False por defecto: los datos de entrenamiento NO se guardan dentro del modelo.

        AVISO DE PRIVACIDAD (v0.9.3): incluir_datos=False NO es una garantía de privacidad clínica. El
        archivo conserva el conjunto de validación y memorias (nodos, valores) derivadas de los datos de
        entrenamiento. No distribuya un .eho entrenado con datos de pacientes fuera del entorno autorizado.
        AVISO DE SEGURIDAD: el formato es pickle (torch.save). Cargue solo archivos de origen confiable."""
        # v0.9.4: dos formatos. "reanudar": punto de control para continuar el aprendizaje de forma
        # equivalente (datos de entrenamiento, lo incorporado, evolución, campaña). "despliegue": solo
        # inferencia; al cargarlo, incorporar() se niega a continuar (el repaso sería distinto).
        modo = modo or ("reanudar" if incluir_datos else "despliegue")
        if modo not in ("reanudar", "despliegue"):
            raise ValueError("modo debe ser 'reanudar' o 'despliegue'")
        incluir_datos = modo == "reanudar"
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        estado = {"version": VERSION, "config": self.cfg.a_dict(),
                  "agentes": [m.to("cpu") for m in self.agentes], "predictor": self.predictor,
                  "valor": self.valor, "escala": self.escala, "validacion": self.validacion,
                  "historial": self.historial,
                  "parada": getattr(self, "parada", None),
                  "simulaciones_usadas": getattr(self, "_simulaciones_usadas", 0),
                  "calibracion_vigente": getattr(self, "calibracion_vigente", True),
                  "modo": modo, "ejecuciones": getattr(self, "_ejecuciones", 0),
                  "acumulado": getattr(self, "_acumulado", None) if incluir_datos else None,
                  "evolucion": getattr(self, "evolucion", None) if incluir_datos else None,
                  "datos_entrenamiento": self.datos_entrenamiento if incluir_datos else None}
        torch.save(estado, ruta)
        for m in self.agentes:
            m.to(self.dispositivo)
        return ruta

    @classmethod
    def cargar(cls, ruta, dispositivo: str | None = None) -> "Gemelo":
        """Carga una colmena. SEGURIDAD: deserializa objetos Python (pickle); un archivo malicioso puede
        ejecutar código. Cargue únicamente archivos .eho generados por usted o de origen confiable."""
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
        g.parada = estado.get("parada")
        g._simulaciones_usadas = estado.get("simulaciones_usadas", 0)
        g.calibracion_vigente = estado.get("calibracion_vigente", True)
        g._ejecuciones = estado.get("ejecuciones", 0)
        g.evolucion = estado.get("evolucion")
        if estado.get("acumulado") is not None:
            g._acumulado = estado["acumulado"]
        g.solo_despliegue = estado.get("modo", "despliegue") == "despliegue" and g.datos_entrenamiento is None
        if g.valor is not None:                                   # comparte los agentes cargados
            g.valor.nov.agents = g.agentes
            if g.valor.err is not None:
                g.valor.err.agents = g.agentes
        return g
