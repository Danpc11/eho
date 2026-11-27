"""Formato de despliegue SEGURO (v0.9.6).

Hasta v0.9.5 todo archivo .eho era un pickle (torch.save de objetos Python): cargar un archivo ajeno podía
ejecutar código. El archivo de despliegue ahora contiene SOLO tensores, números y texto JSON, y se lee con
torch.load(weights_only=True), que no ejecuta código. Lleva lo necesario para predecir y abstenerse:
    - configuración (JSON), argumentos de construcción de cada agente (JSON) y su state_dict completo
    - predictor calibrado (escala, τ, umbral, modo, certificado LTT), protocolo de contexto
    - contadores de la campaña y regla de parada (para información)
No lleva datos de pacientes, ni validación, ni el estimador de valor de simular: un modelo de despliegue
predice, pero no propone simulaciones ni sigue aprendiendo. Para eso está modo="reanudar", que sigue siendo
un pickle porque guarda datos y objetos de la campaña: úselo solo en su propio entorno.
"""
import json
from dataclasses import asdict

import torch

FORMATO = "eho-seguro-1"


def _exact_load(model, state):
    """Carga un state_dict completo, ajustando antes el tamaño de los buffers y parámetros de tamaño
    variable (memoria plasmodial, mapa de novedad). Estricto: no puede faltar ni sobrar ninguna clave."""
    own = model.state_dict()
    falta, sobra = set(own) - set(state), set(state) - set(own)
    if falta or sobra:
        raise RuntimeError(f"estado incompatible: faltan {sorted(falta)[:4]}, sobran {sorted(sobra)[:4]}")
    for k, v in state.items():
        if own[k].shape != v.shape:
            *ruta, nombre = k.split(".")
            mod = model
            for r in ruta:
                mod = getattr(mod, r)
            if nombre in mod._parameters:
                mod._parameters[nombre] = torch.nn.Parameter(torch.zeros_like(v))
            elif nombre in mod._buffers:
                mod._buffers[nombre] = torch.zeros_like(v)
            else:
                raise RuntimeError(f"no se puede redimensionar {k}")
    model.load_state_dict(state, strict=True)
    return model


def exportar(g, ruta, version):
    agentes = []
    for m in g.agentes:
        if not hasattr(m, "init_kwargs"):
            raise RuntimeError("Este modelo se entrenó con una versión anterior a la 0.9.6 y no registra cómo "
                               "se construyó cada agente; guárdelo con formato='pickle' o reentrénelo.")
        agentes.append({"kwargs_json": json.dumps(m.init_kwargs), "dominio": int(getattr(m, "indice_dominio", 0)),
                        "expertos": len(m.moe.experts), "info_json": json.dumps(getattr(m, "hive_info", {})),
                        "state": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}})
    def volcar(pr):
        return {"fail_scale": pr.fail_scale if getattr(pr, "fail_scale", None) is not None else torch.zeros(0),
                "scale": pr.scale.detach().cpu(), "tau": pr.tau if pr.tau is not None else torch.zeros(0),
                "threshold": float(pr.threshold), "mode": getattr(pr, "mode", "media"),
                "method": getattr(pr, "method", "cobertura"),
                "ltt_info_json": json.dumps(getattr(pr, "ltt_info", {})),
                "extra_json": json.dumps({k: getattr(pr, k) for k in ("risk", "delta", "tolerance") if hasattr(pr, k)})}
    predictor = volcar(g.predictor)
    predictores = {str(k): volcar(p) for k, p in (getattr(g, "predictores", None) or {}).items()}
    parada = getattr(g, "parada", None)
    campana = {"simulaciones_usadas": getattr(g, "_simulaciones_usadas", 0), "ejecuciones": getattr(g, "_ejecuciones", 0),
               "costo": getattr(g, "_costo", 0.0), "calibracion_vigente": getattr(g, "calibracion_vigente", True),
               "parada": asdict(parada) if parada is not None else None}
    from .centinela import huella_agentes
    estado = {"formato": FORMATO, "version": version, "config_json": json.dumps(g.cfg.a_dict()),
              # v0.9.7: huella de integridad; liga el certificado a estos agentes exactos
              "huella": huella_agentes(g.agentes),
              "agentes": agentes, "predictor": predictor, "predictores": predictores,
              "registro_json": json.dumps(getattr(g, "registro", [])), "escala": g.escala.detach().cpu(),
              "protocolo_json": json.dumps(getattr(g, "protocolo", None)),
              "historial_json": json.dumps(g.historial), "campana_json": json.dumps(campana)}
    torch.save(estado, ruta)
    return ruta


def importar(cls, estado, dispositivo=None):
    from .config import Config
    from .central import EHOCentral
    from .colony import ColonyPredictor
    from .forrajeo import ReglaParada
    cfg = Config.desde_dict(json.loads(estado["config_json"]))
    if dispositivo:
        cfg.entrenamiento.dispositivo = dispositivo
    g = cls(cfg)
    for a in estado["agentes"]:
        kw = json.loads(a["kwargs_json"])
        kw["j_max"] = int(a["expertos"])                       # si se compactó, hay menos expertos
        kw["j_init"] = min(int(kw.get("j_init", 4)), kw["j_max"])
        m = EHOCentral(g.dominios[a["dominio"] % len(g.dominios)], **kw)
        _exact_load(m, a["state"])
        m.init_kwargs, m.indice_dominio, m.hive_info = kw, a["dominio"], json.loads(a["info_json"])
        g.agentes.append(m.to(g.dispositivo).eval())
    def leer(p):
        pr = ColonyPredictor(p["scale"], mode=p["mode"])
        pr.tau = p["tau"] if p["tau"].numel() else None
        fs = p.get("fail_scale")
        pr.fail_scale = fs if fs is not None and fs.numel() else None
        pr.threshold, pr.method = p["threshold"], p["method"]
        pr.ltt_info = json.loads(p["ltt_info_json"])
        for k, v in json.loads(p["extra_json"]).items():
            setattr(pr, k, v)
        return pr
    from .centinela import huella_agentes
    if "huella" in estado and huella_agentes(g.agentes) != estado["huella"]:
        raise RuntimeError("El archivo no supera la comprobación de integridad: los agentes restaurados no "
                           "coinciden con los que se exportaron (archivo dañado o modificado). Esto detecta "
                           "corrupción, no falsificación deliberada: para eso haría falta una firma digital.")
    g.predictor, g.escala = leer(estado["predictor"]), estado["escala"]
    g.predictores = {int(k): leer(p) for k, p in estado.get("predictores", {}).items()}
    g.registro = json.loads(estado.get("registro_json", "[]"))
    prot = json.loads(estado["protocolo_json"])
    if prot is not None:
        g.protocolo = prot
    g.historial = json.loads(estado["historial_json"])
    c = json.loads(estado["campana_json"])
    g._simulaciones_usadas, g._ejecuciones, g._costo = c["simulaciones_usadas"], c["ejecuciones"], c["costo"]
    g.calibracion_vigente = c["calibracion_vigente"]
    g.parada = ReglaParada(**c["parada"]) if c["parada"] else None
    g.valor, g.validacion, g.datos_entrenamiento = None, None, None
    g.solo_despliegue = True
    return g
