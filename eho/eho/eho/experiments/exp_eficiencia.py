"""Eficiencia (v0.9): ¿cuántas simulaciones, parámetros y segundos hacen falta?

1. Curva de aprendizaje: error con k = 2 ejemplos del paciente frente al número de episodios de
   entrenamiento por régimen (5, 10, 20, 40), con y sin memoria. Es la tesis del artículo en una figura:
   si la memoria sirve, la curva con memoria debe llegar al mismo error con menos simulaciones.
2. Costo: parámetros antes y después de compactar expertos extintos, segundos de entrenamiento e
   inferencia por muestra.
3. Reconstrucción de la memoria plasmodial al final (v0.7): efecto sobre la lectura (rebuild sí/no).

    PYTHONPATH=. python3 experiments/exp_eficiencia.py burgers1d 0 [epocas]
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, ".")
from eho.dominios import crear_dominio, concat
from eho.central import EHOCentral
from eho.train_central import fit_central, evaluate_fewshot

SETUP = {"sangre": (["sano", "dt2"], "prediabetes"), "edp1d": (["difusivo", "advectivo"], "mixto"),
         "burgers1d": (["viscoso", "poco_viscoso"], "intermedio")}
SIZES = [5, 10, 20, 40]


TASA = os.environ.get("EHO_TASA", "constante")       # coseno: decaimiento de la tasa (v0.9)
ETQ = "cos" if TASA == "coseno" else ""


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    regs, test = SETUP[name]
    dom = crear_dominio(name)
    full = [dom.generate(max(SIZES), 8, r, 1000 * seed + i) for i, r in enumerate(regs)]
    va = concat(*[dom.generate(6, 8, r, 1000 * seed + 50 + i) for i, r in enumerate(regs)])
    te = dom.generate(20, 12, test, 1000 * seed + 90)
    dv = int(dom.memory_values({k: v[:2] for k, v in full[0].items()}).shape[-1])
    take = lambda d, n: {k: v[d["episode"] < n] for k, v in d.items()}
    res = {"dominio": name, "semilla": seed, "curva": {}, "costo": {}, "rebuild": {}}
    for mem in ("ninguna", "episodica", "completa"):
        for n in SIZES:
            tr = concat(*[take(d, n) for d in full])
            torch.manual_seed(seed)
            m = EHOCentral(dom, memory=mem, d_val=dv)
            t0 = time.time()
            fit_central(m, tr, va, epochs=epochs, schedule=TASA)
            secs = time.time() - t0
            m.eval()
            e = {f"k={k}": evaluate_fewshot(m, te, k) for k in (0, 2, 5)}
            res["curva"].setdefault(mem, {})[str(n)] = {**e, "segundos": round(secs, 1)}
            print(mem, n, {k: round(v["escalares_MAE"], 4) for k, v in e.items()}, round(secs, 1), flush=True)
            if n == max(SIZES):
                p0 = sum(p.numel() for p in m.parameters())
                t0 = time.time()
                with torch.no_grad():
                    for _ in range(5):
                        m(te)
                ms = (time.time() - t0) / 5 / te["episode"].numel() * 1000
                ref = m(te)["scalars"].clone() if mem == "ninguna" else None
                ahorro = m.compact()
                same = bool(torch.allclose(ref, m(te)["scalars"], atol=1e-5)) if ref is not None else None
                res["costo"][mem] = {"parametros": p0, "parametros_compacto": p0 - ahorro,
                                     "especies": int(m.moe.active.sum()), "ms_por_muestra": round(ms, 3),
                                     "prediccion_igual_tras_compactar": same}
                print("costo", mem, res["costo"][mem], flush=True)
    for rb in (True, False):                                   # efecto de reconstruir la memoria plasmodial
        tr = concat(*full)
        torch.manual_seed(seed)
        m = EHOCentral(dom, memory="completa", d_val=dv)
        fit_central(m, tr, va, epochs=epochs, rebuild=rb, schedule=TASA)
        m.eval()
        res["rebuild"][str(rb)] = {f"k={k}": evaluate_fewshot(m, te, k) for k in (0, 2, 5)}
        print("rebuild", rb, {k: round(v["escalares_MAE"], 4) for k, v in res["rebuild"][str(rb)].items()}, flush=True)
    json.dump(res, open(f"results/eficiencia_{name}{ETQ}_s{seed}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
