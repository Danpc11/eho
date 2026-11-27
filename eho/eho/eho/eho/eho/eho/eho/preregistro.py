"""Prerregistro del análisis (v0.10): valores fijados antes de ver los resultados definitivos.

    python3 -m eho.preregistro            # imprime la huella y si está confirmado
"""
import hashlib
from pathlib import Path

import yaml

RUTA = Path(__file__).resolve().parent.parent / "configs" / "preregistro.yaml"


def cargar(ruta=None) -> dict:
    ruta = Path(ruta) if ruta else RUTA
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    datos["huella"] = hashlib.sha256(ruta.read_bytes()).hexdigest()
    return datos


def para_dominio(dominio: str, ruta=None) -> dict:
    """Parámetros prerregistrados de un dominio, listos para la configuración de la colmena."""
    p = cargar(ruta)
    tol = p["tolerancia"].get(dominio)
    if tol is None:
        raise KeyError(f"no hay tolerancia prerregistrada para el dominio '{dominio}'")
    return {"riesgo": p["abstencion"]["riesgo"], "delta": p["abstencion"]["delta"],
            "protocolos": list(p["abstencion"]["protocolos"]), "tolerancia_absoluta": list(tol["valores"]),
            "margen_relativo": p["no_inferioridad"]["margen_relativo"], "huella": p["huella"],
            "confirmado": bool(p.get("confirmado", False))}


def exigir_confirmado(ruta=None):
    p = cargar(ruta)
    if not p.get("confirmado", False):
        raise RuntimeError("El prerregistro no está confirmado: revise configs/preregistro.yaml y ponga "
                           "`confirmado: true` antes de lanzar las corridas definitivas.")
    return p


if __name__ == "__main__":
    p = cargar()
    print(f"huella: {p['huella']}\nconfirmado: {p.get('confirmado', False)}")
