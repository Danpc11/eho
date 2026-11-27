"""Lanza la rejilla de experimentos del artículo en el clúster: 10 semillas, una GPU por trabajo.

    PYTHONPATH=. python3 experiments/rejilla.py --gpus 0,1,2,3 --semillas 0-9 \
        --experimentos novedad,presupuesto,pocos_ejemplos --dominios sangre,burgers1d,edp1d

Cada trabajo corre con EHO_DEVICE=cuda:<g> y escribe su JSON en results/. Los trabajos que ya tienen
JSON se saltan (se puede relanzar tras un corte). Sin gestor de colas: corre dentro de tmux.
Después:  python3 analisis/estadistica.py
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

SALIDA = {"novedad": "results/novedad_{d}_s{s}.json", "presupuesto": "results/presupuesto_{d}_s{s}.json",
          "pocos_ejemplos": "results/pocos_ejemplos_{d}_s{s}.json"}


def rango(txt):
    if "-" in txt:
        a, b = txt.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in txt.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--semillas", default="0-9")
    ap.add_argument("--experimentos", default="novedad,presupuesto,pocos_ejemplos")
    ap.add_argument("--dominios", default="sangre,burgers1d,edp1d")
    ap.add_argument("--epocas", default="")
    a = ap.parse_args()
    gpus = a.gpus.split(",")
    cola = [(e, d, s) for e, d, s in itertools.product(a.experimentos.split(","), a.dominios.split(","),
                                                       rango(a.semillas))
            if not Path(SALIDA[e].format(d=d, s=s)).exists()]
    print(f"{len(cola)} trabajos pendientes en {len(gpus)} GPU", flush=True)
    activos = {}
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    while cola or activos:
        for g in gpus:
            if g not in activos and cola:
                e, d, s = cola.pop(0)
                env = {**os.environ, "EHO_DEVICE": f"cuda:{g}" if g != "cpu" else "cpu", "PYTHONPATH": "."}
                cmd = [sys.executable, f"experiments/exp_{e}.py", d, str(s)] + ([a.epocas] if a.epocas else [])
                log = open(f"results/logs/{e}_{d}_s{s}.log", "w")
                activos[g] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), (e, d, s), log)
                print(f"GPU {g}: {e} {d} s{s}", flush=True)
        for g, (p, job, log) in list(activos.items()):
            if p.poll() is not None:
                log.close()
                print(f"GPU {g}: terminó {job} (código {p.returncode})", flush=True)
                del activos[g]
        time.sleep(5)


if __name__ == "__main__":
    main()
