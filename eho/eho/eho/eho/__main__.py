"""Línea de comandos del EHO: generar, entrenar, usar y hacer crecer modelos.

    python -m eho dominios
    python -m eho nuevo-dominio glucosa                 # crea dominios_propios/glucosa.py + configs/glucosa.yaml
    python -m eho config-ejemplo sangre -o configs/mi.yaml
    python -m eho entrenar configs/sangre.yaml -o modelos/hemotwin.eho
    python -m eho info modelos/hemotwin.eho
    python -m eho evaluar modelos/hemotwin.eho prueba.pt
    python -m eho predecir modelos/hemotwin.eho consulta.pt --contexto ctx.pt -o pred.pt
    python -m eho proponer modelos/hemotwin.eho candidatos.pt --presupuesto 4
    python -m eho incorporar modelos/hemotwin.eho nuevos.pt -o modelos/hemotwin_v2.eho

Los archivos .pt son diccionarios de tensores con el formato de eho/dominios/base.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())          # para dominios propios "paquete.modulo:Clase"


def _cargar_pt(ruta):
    return torch.load(ruta, map_location="cpu", weights_only=True)      # datos: solo tensores


def cmd_dominios(a):
    from .dominios import dominios_disponibles, REGISTRO
    for n in dominios_disponibles():
        cls = REGISTRO[n]
        print(f"{n:12s} d_point={cls.d_point} d_cond={cls.d_cond} campos={cls.n_fields} "
              f"escalares={cls.n_scalars} regímenes={cls.regimes}")


def cmd_nuevo_dominio(a):
    from .plantilla_dominio import PLANTILLA, CONFIG
    nombre = a.nombre.lower()
    clase = "".join(p.capitalize() for p in nombre.split("_"))
    carpeta = Path(a.carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "__init__.py").touch()
    archivo = carpeta / f"{nombre}.py"
    if archivo.exists() and not a.forzar:
        sys.exit(f"{archivo} ya existe (usa --forzar para reemplazarlo)")
    archivo.write_text(PLANTILLA.format(nombre=nombre, clase=clase), encoding="utf-8")
    Path("configs").mkdir(exist_ok=True)
    modulo = ".".join(carpeta.parts + (nombre,))
    cfg = Path("configs") / f"{nombre}.yaml"
    cfg.write_text(CONFIG.format(nombre=nombre, modulo=modulo, clase=clase), encoding="utf-8")
    print(f"creados {archivo} y {cfg}\nsiguiente: edita {archivo} y corre\n"
          f"  python -m eho entrenar {cfg} -o modelos/{nombre}.eho")


def cmd_config_ejemplo(a):
    from .config import Config
    from .dominios import crear_dominio
    cfg = Config()
    cfg.nombre = a.dominio
    cfg.dominio.nombre = a.dominio
    cfg.dominio.regimenes = list(crear_dominio(a.dominio).regimes[:2])
    if a.salida:
        cfg.guardar(a.salida)
        print(f"escrito {a.salida}")
    else:
        import yaml
        print(yaml.safe_dump(cfg.a_dict(), allow_unicode=True, sort_keys=False))


def cmd_entrenar(a):
    from .gemelo import Gemelo
    from .config import Config
    cfg = Config.desde_archivo(a.config)
    if a.dispositivo:
        cfg.entrenamiento.dispositivo = a.dispositivo
    if a.epocas:
        cfg.entrenamiento.epocas = a.epocas
    g = Gemelo(cfg).entrenar()
    if a.prueba:
        print(json.dumps(g.evaluar(_cargar_pt(a.prueba)), indent=1))
    # v0.9.5: por defecto se guarda un punto de control para REANUDAR (lleva los datos de repaso), porque
    # el flujo normal sigue con `incorporar`. Desde v0.9.4 un archivo de despliegue se niega a seguir
    # aprendiendo, y la línea de comandos guardaba despliegue: `incorporar` había dejado de funcionar.
    modo = "despliegue" if getattr(a, "despliegue", False) else "reanudar"
    ruta = g.guardar(a.salida or f"modelos/{cfg.nombre}.eho", modo=modo)
    print(f"modelo guardado en {ruta} (modo {modo})")
    if modo == "reanudar":
        print("AVISO: este archivo contiene los datos de entrenamiento. Para compartirlo, use --despliegue.")


def _gemelo(ruta, dispositivo=None):
    from .gemelo import Gemelo
    return Gemelo.cargar(ruta, dispositivo)


def cmd_info(a):
    print(json.dumps(_gemelo(a.modelo, "cpu").resumen(), ensure_ascii=False, indent=1))


def cmd_evaluar(a):
    g = _gemelo(a.modelo, a.dispositivo)
    datos = _cargar_pt(a.datos)
    print(json.dumps({"agentes": g.evaluar(datos, a.k), "colectivo": g.evaluar_colectivo(datos, a.k)}, indent=1))


def cmd_predecir(a):
    g = _gemelo(a.modelo, a.dispositivo)
    r = g.predecir(_cargar_pt(a.consulta), _cargar_pt(a.contexto) if a.contexto else None)
    if a.salida:
        torch.save(r, a.salida)
    print(json.dumps({"muestras": int(r["aceptado"].numel()),
                      "aceptadas_por_la_colmena": int(r["aceptado"].sum()),
                      "escalares": r["escalares"][: a.mostrar].tolist(),
                      "aceptado": r["aceptado"][: a.mostrar].tolist()}, indent=1))


def cmd_proponer(a):
    g = _gemelo(a.modelo, a.dispositivo)
    elegidos = g.proponer(_cargar_pt(a.candidatos), a.presupuesto)
    print(json.dumps({"elegidos": elegidos,
                      "mensaje": "nada vale la pena simular" if not elegidos else f"{len(elegidos)} a simular"},
                     ensure_ascii=False, indent=1))


def cmd_incorporar(a):
    g = _gemelo(a.modelo, a.dispositivo)
    if a.repaso:
        g.datos_entrenamiento = [_cargar_pt(p) for p in a.repaso]
    g.incorporar(_cargar_pt(a.nuevos))
    modo = "despliegue" if getattr(a, "despliegue", False) else "reanudar"
    print(f"modelo actualizado en {g.guardar(a.salida or a.modelo, modo=modo)} (modo {modo})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m eho", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dominios", help="lista los dominios registrados"); s.set_defaults(f=cmd_dominios)
    s = sub.add_parser("nuevo-dominio", help="crea la plantilla de un dominio nuevo y su configuración")
    s.add_argument("nombre"); s.add_argument("--carpeta", default="dominios_propios")
    s.add_argument("--forzar", action="store_true"); s.set_defaults(f=cmd_nuevo_dominio)
    s = sub.add_parser("config-ejemplo", help="escribe una configuración con todos los valores por defecto")
    s.add_argument("dominio"); s.add_argument("-o", "--salida"); s.set_defaults(f=cmd_config_ejemplo)
    s = sub.add_parser("entrenar", help="genera y entrena un modelo a partir de una configuración")
    s.add_argument("config"); s.add_argument("-o", "--salida"); s.add_argument("--dispositivo")
    s.add_argument("--epocas", type=int); s.add_argument("--prueba", help="datos .pt para evaluar al final")
    s.set_defaults(f=cmd_entrenar)
    for nombre, fn, ayuda in [("info", cmd_info, "resumen del modelo"),
                              ("evaluar", cmd_evaluar, "error con k ejemplos de contexto"),
                              ("predecir", cmd_predecir, "predicción colectiva con abstención"),
                              ("proponer", cmd_proponer, "qué casos vale la pena simular o medir"),
                              ("incorporar", cmd_incorporar, "aprender casos nuevos sin olvidar")]:
        s = sub.add_parser(nombre, help=ayuda)
        s.add_argument("modelo")
        s.add_argument("--dispositivo")
        if nombre == "evaluar":
            s.add_argument("datos"); s.add_argument("-k", type=int, default=2)
        if nombre == "predecir":
            s.add_argument("consulta"); s.add_argument("--contexto"); s.add_argument("-o", "--salida")
            s.add_argument("--mostrar", type=int, default=5)
        if nombre == "proponer":
            s.add_argument("candidatos"); s.add_argument("--presupuesto", type=int)
        if nombre == "incorporar":
            s.add_argument("nuevos"); s.add_argument("-o", "--salida")
            s.add_argument("--repaso", nargs="*", help="datos de entrenamiento originales para repasar")
        if nombre in ("entrenar", "incorporar"):
            s.add_argument("--despliegue", action="store_true",
                           help="guardar solo para inferencia (sin datos; no podrá seguir aprendiendo)")
        s.set_defaults(f=fn)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
