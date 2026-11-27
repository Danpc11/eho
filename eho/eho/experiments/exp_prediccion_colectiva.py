"""¿Qué debe devolver la colmena? (v0.9.4)

Hasta v0.9.3 el predictor calibrado devolvía siempre la media de los agentes (el quórum interno de 1.01
impedía que el ganador de la dinámica cambiara la predicción). Aquí se comparan, con los mismos agentes
y el mismo umbral de cobertura (80 % en validación):
    media       media aritmética de los agentes
    ponderada   media ponderada por la confianza de inspección
    ganador     centro de la opción con más apoyo en la dinámica de la colmena
en casos conocidos y en un régimen intermedio no visto.

    PYTHONPATH=. python3 experiments/exp_prediccion_colectiva.py burgers1d 0 [epocas]
"""
import json
import os
import sys

import torch

sys.path.insert(0, ".")
from eho import Gemelo
from eho.dominios import concat
from eho.colony import ColonyPredictor
from eho.hive import predict_with_inspection

SETUP = {"sangre": (["sano", "dt2"], "prediabetes", [{"simulador": "quemada"}, {"simulador": "carreau"}]),
         "edp1d": (["difusivo", "advectivo"], "mixto", None), "burgers1d": (["viscoso", "poco_viscoso"], "intermedio", None)}
BASE = 30_000_000                                     # semillas de prueba lejos de entrenamiento y validación


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    regs, inter, sims = SETUP[name]
    dom = {"nombre": name, "regimenes": regs}
    if sims:
        dom["simuladores"] = sims
    g = Gemelo.desde_config({"nombre": f"pc_{name}", "dominio": dom, "colmena": {"n_agentes": 4},
                             "entrenamiento": {"epocas": epochs, "semilla": seed,
                                               "tasa": os.environ.get("EHO_TASA", "constante")}}).entrenar(verbose=False)
    d0 = g.dominio_variante(0)
    sets = {"conocido": concat(*[d0.generate(1, 8, regs[i % len(regs)], BASE + 1000 * seed + i) for i in range(24)]),
            "intermedio": concat(*[d0.generate(1, 8, inter, BASE + 1000 * seed + 500 + i) for i in range(24)])}
    Pv, Yv, CEv = predict_with_inspection(g.agentes, g.validacion, k=2)
    res = {"dominio": name, "semilla": seed, "modos": {}}
    for mode in ("media", "ponderada", "ganador"):
        pr = ColonyPredictor(g.escala, mode=mode).fit(Pv, Yv, CEv, target_coverage=0.8)
        row = {}
        for k, ds in sets.items():
            P, Y, CE = predict_with_inspection(g.agentes, ds, k=2)
            pred, ok, _ = pr.predict(P, CE)
            err = (pred - Y).abs().mean(-1)
            row[k] = {"MAE": float(err.mean()), "MAE_aceptados": float(err[ok].mean()) if ok.any() else float("nan"),
                      "cobertura": float(ok.float().mean()), "MAE_agentes": float((P - Y).abs().mean())}
        res["modos"][mode] = row
        print(mode, {k: {a: round(b, 4) for a, b in v.items()} for k, v in row.items()}, flush=True)
    json.dump(res, open(f"results/prediccion_colectiva_{name}_s{seed}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
