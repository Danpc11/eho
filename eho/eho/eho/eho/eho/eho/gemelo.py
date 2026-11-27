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

VERSION = "0.9.7"


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
            trains = [torch.load(rutas[i % len(rutas)], weights_only=True) for i in range(len(self.dominios))]
        else:
            trains = [concat(*[d.generate(dc.episodios_por_regimen, dc.muestras_por_episodio, r, 1000 * s + i)
                               for i, r in enumerate(self._regimenes(d))]) for d in self.dominios]
        if dc.datos_validacion:
            val = torch.load(dc.datos_validacion, weights_only=True)
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
        self.auditar_particiones()
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

    def auditar_particiones(self, calib=None, val=None):
        """Auditoría de procedencia (v0.9.5), obligatoria al entrenar y al calibrar: ninguna partición
        puede compartir episodios ni muestras con otra (entrenamiento, validación, evolución, calibración).
        Detecta, por ejemplo, el mismo archivo usado para entrenar y para calibrar."""
        from .dominios import assert_disjoint
        # v0.9.7: se audita la validación que DE VERDAD se usa (no la anterior) y todo lo que ha influido
        # en el modelo: entrenamiento inicial MÁS lo incorporado en el aprendizaje continuo
        partes = {"validación": val if val is not None else self.validacion,
                  "evolución": getattr(self, "evolucion", None), "calibración": calib}
        aprendizaje = list(self.datos_entrenamiento or [])
        for por_agente in getattr(self, "_acumulado", None) or []:
            aprendizaje.extend(por_agente)
        for i, tr in enumerate(aprendizaje):
            for nombre, ds in partes.items():
                if ds is not None:
                    assert_disjoint(tr, ds, f"entrenamiento[{i}] y {nombre}")
        nombres = [n for n, d in partes.items() if d is not None]
        for i, a in enumerate(nombres):
            for b in nombres[i + 1:]:
                assert_disjoint(partes[a], partes[b], f"{a} y {b}")

    def datos_evolucion(self):
        """Episodios para la selección de expertos (extinción e invasión), distintos de la validación
        (v0.9.3). De archivo, o generados con otra semilla. Si los datos vienen de archivo y no hay
        archivo de evolución, devuelve None: la ecología usa la validación, que entonces forma parte del
        desarrollo del modelo y no debe citarse como evidencia independiente."""
        dc, s = self.cfg.dominio, self.cfg.entrenamiento.semilla
        if dc.datos_evolucion:
            return torch.load(dc.datos_evolucion, weights_only=True)
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
            return torch.load(dc.datos_calibracion, weights_only=True)
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
        """Escala, τ y umbral de abstención con la versión ACTUAL de los agentes.

        v0.9.6: un certificado POR PROTOCOLO. Para cada número de mediciones de contexto de
        colmena.protocolos (por defecto, solo k_contexto) se calibra un predictor propio; al predecir se
        usa el que corresponde a las mediciones recibidas. Cada certificado LTT vale por separado con
        probabilidad 1 - δ (para que valgan todos a la vez, divida δ entre el número de protocolos)."""
        c = self.cfg.colmena
        ks = sorted(set(list(c.protocolos) + [c.k_contexto]))
        if c.calibracion == "ltt":
            calib = calib if calib is not None else self.datos_calibracion(ronda)
        self.auditar_particiones(calib, val)
        self.predictores = {}
        # v0.9.7: con varios protocolos, δ se reparte (cota de unión) para que todos valgan a la vez
        delta_k = c.delta / len(ks) if c.garantia_conjunta else c.delta
        for k in ks:
            P, Y, CE = predict_with_inspection(self.agentes, val, k=k)
            if k == c.k_contexto:
                self.escala = (P.mean(0) - Y).abs().median(0).values.clamp(min=1e-4) * 2.0
        for k in ks:
            P, Y, CE = predict_with_inspection(self.agentes, val, k=k)
            if c.calibracion == "ltt":
                Pc, Yc, CEc, epc = predict_with_inspection(self.agentes, calib, k=k, return_episodes=True)
                idx = torch.tensor([int(torch.nonzero(epc == e)[0]) for e in epc.unique()])
                pr = ColonyPredictor(self.escala, mode=c.prediccion).fit(
                    P, Y, CE, method="ltt", risk=c.riesgo, delta=delta_k, tolerance=c.tolerancia,
                    calib=(Pc[:, idx], Yc[idx], CEc[:, idx]))
            else:
                pr = ColonyPredictor(self.escala, mode=c.prediccion).fit(P, Y, CE, target_coverage=c.cobertura)
            self.predictores[k] = pr
        self.predictor = self.predictores[c.k_contexto]
        self.calibracion_vigente = True
        from .centinela import huella_agentes
        self._certificados_emitidos = getattr(self, "_certificados_emitidos", 0) + 1
        self.protocolo = {
            "k_contexto": c.k_contexto, "calibracion": c.calibracion, "prediccion": c.prediccion, "protocolos": ks,
            "huella": huella_agentes(self.agentes),             # versión exacta del modelo certificado
            "certificado": {                                    # v0.9.7: lo que el certificado afirma
                "seleccion_contexto": "k mediciones al azar entre las del paciente (split_episodes)",
                "unidad_estadistica": "una consulta por paciente (episodio)",
                "error": "error absoluto medio de los escalares, escalado por la escala de la colmena",
                "tolerancia": c.tolerancia, "riesgo": c.riesgo, "delta_por_protocolo": delta_k,
                "delta_conjunta": c.delta if c.garantia_conjunta else min(1.0, c.delta * len(ks)),
                "distribucion": {"dominio": self.cfg.dominio.nombre, "regimenes": list(self._regimenes(self.dominios[0]))},
                "emision": self._certificados_emitidos,
                "nota": "cada recalibración emite un certificado nuevo; la probabilidad de que TODOS los "
                        "emitidos en una campaña valgan a la vez es >= 1 - (emisiones x delta)"}}
        return self

    # ------------------------------------------------------------------ uso
    @torch.no_grad()
    def _certificar(self, P, CE, k_ctx: int):
        """ÚNICA función de decisión certificada (v0.9.7), compartida por predecir() y evaluar_colectivo().
        Devuelve (predicción, aceptado, apoyo, info). Solo se marca como aceptado si, a la vez: existe un
        certificado para este número de mediciones, la calibración está vigente y los agentes son
        exactamente los que se calibraron (huella SHA-256)."""
        from .centinela import huella_agentes
        prs = getattr(self, "predictores", None) or {}
        pr = prs.get(k_ctx)
        protocolo_ok = pr is not None
        pr = pr if pr is not None else self.predictor
        pred, ok, apoyo = pr.predict(P, CE)
        vigente = getattr(self, "calibracion_vigente", True)
        prot = getattr(self, "protocolo", None) or {}
        modelo_ok = prot.get("huella") is None or prot["huella"] == huella_agentes(self.agentes)
        if not (vigente and modelo_ok and protocolo_ok):
            ok = torch.zeros_like(ok)
        return pred, ok, apoyo, {"calibracion_vigente": vigente, "protocolo_coincide": protocolo_ok,
                                 "modelo_coincide": modelo_ok, "k_contexto_usado": k_ctx, "predictor": pr}

    @torch.no_grad()
    def predecir(self, consulta, contexto=None):
        """Predicción colectiva. contexto: mediciones etiquetadas del MISMO paciente que la consulta.
        Devuelve escalares, campos, aceptado (bool por muestra), apoyo y las predicciones de cada agente.

        El certificado vale para el protocolo calibrado: k mediciones de contexto de un solo paciente,
        tomadas al azar entre las disponibles, distintas de las consultas. Se comprueba lo que el código
        puede comprobar (un solo episodio, mismo identificador, sin muestras compartidas); que las
        mediciones sean de verdad de ese paciente es responsabilidad de quien llama."""
        from .hive import context_inspection
        from .dominios.base import sample_hashes
        k_ctx = 0
        if contexto is not None and contexto["episode"].numel():
            if contexto["episode"].unique().numel() > 1:
                raise ValueError("predecir: el contexto mezcla varios episodios; debe ser un solo paciente")
            if "episode" in consulta and (consulta["episode"].unique().numel() > 1
                                          or int(consulta["episode"][0]) != int(contexto["episode"][0])):
                raise ValueError("predecir: la consulta y el contexto deben ser del mismo episodio (paciente)")
            if sample_hashes(consulta) & sample_hashes(contexto):
                raise ValueError("predecir: hay muestras que están a la vez en la consulta y en el contexto")
            k_ctx = int(contexto["episode"].numel())
        else:
            contexto = None
        outs = [a(consulta, context=contexto) for a in self.agentes]
        P = torch.stack([o["scalars"].cpu() for o in outs])
        F = torch.stack([o["fields"].cpu() for o in outs])
        ce = context_inspection(self.agentes, contexto) if contexto is not None else torch.zeros(len(self.agentes))
        CE = ce.unsqueeze(-1).expand(-1, P.shape[1])
        pred, ok, apoyo, info = self._certificar(P, CE, k_ctx)
        # los campos se combinan con los pesos que respaldan la decisión escalar; uniformes en abstención.
        # La aceptación certifica el escalar, no el campo.
        d2 = (((P - pred.unsqueeze(0)) / info["predictor"].scale) ** 2).mean(-1)          # [A, B]
        W = torch.softmax(-0.5 * d2, 0)
        W = torch.where(ok.unsqueeze(0), W, torch.full_like(W, 1.0 / W.shape[0]))
        campos = (W.unsqueeze(-1).unsqueeze(-1) * F).sum(0)
        info.pop("predictor")
        return {"escalares": pred, "campos": campos, "aceptado": ok, "apoyo": apoyo,
                "por_agente": P, "dispersion_campos": F.std(0), "pesos_campos": W, **info}

    @torch.no_grad()
    def novedad(self, lote):
        """Valor de novedad por muestra en [0, 1] (≈0 conocido, ≈1 nuevo), promedio de la colmena."""
        return self.valor.nov.values(lote).mean(0)

    def presupuesto_restante(self):
        """Simulaciones que quedan del tope global de la campaña (None = sin tope)."""
        a = self.cfg.activo
        restos = []
        if a.presupuesto_total is not None:
            restos.append(a.presupuesto_total - getattr(self, "_simulaciones_usadas", 0) - getattr(self, "_reservado", 0))
        if a.presupuesto_ejecuciones is not None:           # v0.9.5: condiciones que aún caben en ejecuciones
            restos.append((a.presupuesto_ejecuciones - getattr(self, "_ejecuciones", 0)) // max(len(self.dominios), 1))
        if a.presupuesto_costo is not None and getattr(self, "_costo", 0.0) >= a.presupuesto_costo:
            restos.append(0)
        return None if not restos else max(min(restos), 0)

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
        if self.valor is None:
            raise RuntimeError("Este modelo es de despliegue (solo inferencia): no puede proponer simulaciones. "
                               "Use un archivo guardado con modo='reanudar'.")
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

    def reservar(self, condiciones: int, costo: float | None = None):
        """Reserva presupuesto ANTES de lanzar simulaciones (v0.9.7). incorporar() solo puede rechazar
        resultados cuando las horas de GPU ya se gastaron; para controlar el presupuesto de verdad, el
        flujo es: proponer -> reservar -> ejecutar los simuladores -> incorporar (que consume la reserva y
        registra el costo real). Falla si la reserva no cabe en lo que queda."""
        import math
        a = self.cfg.activo
        if condiciones <= 0:
            raise ValueError("reservar: número de condiciones no válido")
        if costo is not None and (not math.isfinite(costo) or costo < 0):
            raise ValueError("reservar: costo no válido")
        restante = self.presupuesto_restante()
        if restante is not None and condiciones > restante:
            raise ValueError(f"reservar: {condiciones} condiciones no caben en el presupuesto restante ({restante})")
        if a.presupuesto_costo is not None:
            if costo is None:
                raise ValueError("reservar: con presupuesto de costo hay que indicar el costo estimado")
            libre = a.presupuesto_costo - getattr(self, "_costo", 0.0) - getattr(self, "_costo_reservado", 0.0)
            if costo > libre + 1e-12:
                raise ValueError(f"reservar: el costo estimado ({costo}) excede lo disponible ({libre})")
        self._reservado = getattr(self, "_reservado", 0) + int(condiciones)
        self._costo_reservado = getattr(self, "_costo_reservado", 0.0) + (float(costo) if costo is not None else 0.0)
        return {"condiciones_reservadas": self._reservado, "costo_reservado": self._costo_reservado}

    def liberar(self):
        """Cancela las reservas pendientes (simulaciones que no se van a ejecutar)."""
        self._reservado, self._costo_reservado = 0, 0.0

    def incorporar(self, nuevos, verbose=False, costo: float | None = None, manifiesto: dict | None = None,
                   ids: list | None = None):
        """Incorporación TRANSACCIONAL (v0.9.5): si algo falla a mitad del ajuste, se restaura el estado
        anterior completo (agentes, memorias, regla de parada, contadores, calibración).
        costo: costo real de estas simulaciones (p. ej. horas de GPU), para el tope presupuesto_costo."""
        import copy
        campos = ("agentes", "valor", "parada", "predictor", "predictores", "registro", "escala", "_acumulado", "historial",
                  "_reservado", "_costo_reservado", "_certificados_emitidos",
                  "_simulaciones_usadas", "_ejecuciones", "_costo", "calibracion_vigente", "protocolo")
        previo = copy.deepcopy({k: getattr(self, k) for k in campos if hasattr(self, k)})
        try:
            return self._incorporar(nuevos, verbose, costo, manifiesto, ids)
        except BaseException:
            for k in campos:
                if k in previo:
                    setattr(self, k, previo[k])
                elif hasattr(self, k):
                    delattr(self, k)
            raise

    def _incorporar(self, nuevos, verbose=False, costo=None, manifiesto=None, ids_ejecucion=None):
        import math
        # v0.9.7: el costo debe ser finito y no negativo, y obligatorio si hay presupuesto de costo
        if costo is None:
            if self.cfg.activo.presupuesto_costo is not None:
                raise ValueError("incorporar: con presupuesto de costo, el costo de las simulaciones es obligatorio")
        elif not math.isfinite(float(costo)) or float(costo) < 0:
            raise ValueError("incorporar: costo no válido (debe ser finito y no negativo)")
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
        if len(lista) != len(self.agentes):                       # v0.9.5: zip() ya no trunca en silencio
            raise ValueError(f"incorporar: se esperaba un conjunto por agente ({len(self.agentes)}), llegaron {len(lista)}")
        ids0 = lista[0]["episode"].unique()
        for j, d in enumerate(lista[1:], 1):
            if not torch.equal(d["episode"].unique(), ids0):
                raise ValueError(f"incorporar: el conjunto {j} no trae las mismas condiciones (episodios) que el 0")
            if d["cond"].shape != lista[0]["cond"].shape or not torch.allclose(d["cond"], lista[0]["cond"], atol=1e-6):
                raise ValueError(f"incorporar: el conjunto {j} no corresponde a las mismas condiciones físicas que el 0")
        n_cond = int(ids0.numel())
        # v0.9.6: cada simulación tiene un identificador (huella de sus entradas, invariante al orden).
        # Incorporar dos veces la misma se rechaza: se contaría doble en el presupuesto y en el repaso.
        import time as _time
        from .dominios.base import subset as _sub, episode_hashes
        t_ini = _time.perf_counter()
        huellas = [next(iter(episode_hashes(_sub(lista[0], torch.nonzero(lista[0]["episode"] == e).flatten())))) for e in ids0]
        # v0.9.7: la identidad de una simulación es su identificador de EJECUCIÓN (versión del simulador,
        # semilla, parámetros, configuración numérica), que da quien la corre con ids=[...]. La huella de
        # los datos queda como comprobación adicional: la misma simulación en otra malla tiene otra huella.
        if ids_ejecucion is not None:
            if len(ids_ejecucion) != n_cond or len(set(map(str, ids_ejecucion))) != n_cond:
                raise ValueError("incorporar: ids debe tener un identificador distinto por condición")
            ids = [str(x) for x in ids_ejecucion]
        else:
            ids = huellas
        if not hasattr(self, "registro"):
            self.registro = []
        vistos = {r["id"] for r in self.registro} | {r.get("huella") for r in self.registro}
        repetidos = [x for x in ids if x in vistos] + [x for x in huellas if x in vistos] + \
            [x for x in set(ids) if ids.count(x) > 1]
        if repetidos and not a.permitir_repetidos:
            raise ValueError(f"incorporar: {len(repetidos)} simulaciones ya incorporadas o duplicadas (id {repetidos[0][:10]}…)")
        restante = self.presupuesto_restante()
        if restante is not None:
            restante += min(getattr(self, "_reservado", 0), n_cond)      # lo reservado para estas simulaciones
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
        a_ej = a.presupuesto_ejecuciones
        if a_ej is not None and a.presupuesto_estricto and getattr(self, "_ejecuciones", 0) + n_cond * n_sim > a_ej:
            raise ValueError("incorporar: se excede el presupuesto de ejecuciones")
        self._ejecuciones = getattr(self, "_ejecuciones", 0) + n_cond * n_sim
        self._costo = getattr(self, "_costo", 0.0) + (float(costo) if costo is not None else 0.0)
        if a.presupuesto_costo is not None and a.presupuesto_estricto and self._costo > a.presupuesto_costo:
            raise ValueError("incorporar: se excede el presupuesto de costo")
        if not hasattr(self, "_acumulado"):
            self._acumulado = [[] for _ in self.agentes]
        for i, (m, nd) in enumerate(zip(self.agentes, lista, strict=True)):
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
        simuladores = [getattr(d, "simulador", d.name) for d in self.dominios] if distintos else \
            [getattr(self.dominios[0], "simulador", self.dominios[0].name)]
        for x, hx in zip(ids, huellas):                              # manifiesto de procedencia
            # los metadatos del usuario van APARTE: no pueden sobrescribir id, ronda, costo ni simuladores
            self.registro.append({"id": x, "huella": hx, "ronda": len(self.historial), "simuladores": simuladores,
                                  "costo": None if costo is None else float(costo) / n_cond,
                                  "metadatos": dict(manifiesto or {})})
        self._reservado = max(getattr(self, "_reservado", 0) - n_cond, 0)    # la reserva se consume
        if self._reservado == 0:
            self._costo_reservado = 0.0
        self.historial.append({"evento": "incorporar", "episodios": n_cond, "condiciones": n_cond,
                               "ejecuciones": n_cond * n_sim, "costo": costo,
                               "segundos_ajuste": round(_time.perf_counter() - t_ini, 2)})
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
        if k not in (getattr(self, "predictores", None) or {}):
            raise ValueError(f"evaluar_colectivo: no hay calibración para k = {k}; "
                             f"protocolos disponibles: {sorted(getattr(self, 'predictores', {}) or {})}")
        P, Y, CE = predict_with_inspection(self.agentes, ds, k=k)
        pred, ok, _, info = self._certificar(P, CE, k)                 # mismo camino que predecir()
        err = (pred - Y).abs().mean(-1)
        falla = ((pred - Y).abs() / info["predictor"].scale).mean(-1) > c.tolerancia
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
    def guardar(self, ruta, incluir_datos: bool = False, modo: str | None = None, formato: str | None = None):
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
        # v0.9.6: el despliegue usa por defecto el formato SEGURO (solo tensores y JSON, sin pickle)
        formato = formato or ("seguro" if modo == "despliegue" else "pickle")
        if formato == "seguro":
            if modo != "despliegue":
                raise ValueError("el formato seguro solo existe para modo='despliegue'")
            from .persistencia import exportar
            return exportar(self, ruta, VERSION)
        estado = {"version": VERSION, "config": self.cfg.a_dict(),
                  "agentes": [m.to("cpu") for m in self.agentes], "predictor": self.predictor,
                  "valor": self.valor, "escala": self.escala,
                  # v0.9.5: el archivo de despliegue ya no lleva el conjunto de validación
                  "validacion": self.validacion if incluir_datos else None,
                  "protocolo": getattr(self, "protocolo", None), "costo": getattr(self, "_costo", 0.0),
                  "predictores": getattr(self, "predictores", None), "registro": getattr(self, "registro", []),
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
    def cargar(cls, ruta, dispositivo: str | None = None, permitir_pickle: bool = False) -> "Gemelo":
        """Carga una colmena. SEGURIDAD: deserializa objetos Python (pickle); un archivo malicioso puede
        ejecutar código. Cargue únicamente archivos .eho generados por usted o de origen confiable."""
        import pickle
        try:                                   # v0.9.6: primero sin ejecutar código
            estado = torch.load(ruta, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError):
            estado = None
        if isinstance(estado, dict) and estado.get("formato", "").startswith("eho-seguro"):
            from .persistencia import importar
            return importar(cls, estado, dispositivo)
        if not permitir_pickle:
            raise RuntimeError("El archivo no está en formato seguro. Solo si lo generó usted mismo, cárguelo "
                               "con Gemelo.cargar(ruta, permitir_pickle=True).")
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
        g._costo = estado.get("costo", 0.0)
        g.registro = estado.get("registro", [])
        if estado.get("predictores"):
            g.predictores = estado["predictores"]
        if estado.get("protocolo") is not None:
            g.protocolo = estado["protocolo"]
        g.evolucion = estado.get("evolucion")
        if estado.get("acumulado") is not None:
            g._acumulado = estado["acumulado"]
        g.solo_despliegue = estado.get("modo", "despliegue") == "despliegue" and g.datos_entrenamiento is None
        if g.valor is not None:                                   # comparte los agentes cargados
            g.valor.nov.agents = g.agentes
            if g.valor.err is not None:
                g.valor.err.agents = g.agentes
        return g
