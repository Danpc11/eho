"""Ecología de expertos (v0.8): ¿cuándo ayuda, y por qué?

Aprendizaje continuo en sangre: sano -> prediabetes -> DT2, con repetición de 5 episodios por etapa
previa. Tres variantes del mismo núcleo (sin memoria, para aislar la ecología):
    sin_ecologia     enrutador con balance de carga
    ecologia         Lotka-Volterra desde la primera época (v0.1-v0.7)
    sucesion         épocas pioneras sin Lotka-Volterra y luego ecología (v0.8)
Tras cada etapa: error en las tres tareas (olvido = subida en las anteriores), especies activas y
diagnóstico de coexistencia de Chesson (ρ < κ_i/κ_j < 1/ρ).

    PYTHONPATH=. python3 experiments/exp_ecologia.py 0 [epocas]
"""
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from eho.dominios import Sangre, concat, subset
from eho.central import EHOCentral
from eho.train_central import fit_central, evaluate_fewshot
from eho.ecology import chesson_diagnostic

TASKS = ["sano", "prediabetes", "dt2"]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    dom = Sangre()
    tr = {t: dom.generate(30, 8, t, 1000 * seed + i) for i, t in enumerate(TASKS)}
    va = {t: dom.generate(6, 8, t, 1000 * seed + 50 + i) for i, t in enumerate(TASKS)}
    te = {t: dom.generate(12, 8, t, 9000 + i) for i, t in enumerate(TASKS)}
    res, t0 = {"semilla": seed, "variantes": {}}, time.time()
    for name, eco, suc in [("sin_ecologia", False, 0), ("ecologia", True, 0), ("sucesion", True, 5)]:
        torch.manual_seed(seed)
        m = EHOCentral(dom, memory="ninguna", ecology=eco)
        replay, rows = [], []
        for si, t in enumerate(TASKS):
            data = concat(tr[t], *replay) if replay else tr[t]
            v = concat(*[va[k] for k in TASKS[:si + 1]])
            fit_central(m, data, v, epochs=epochs, eco_warmup=5 if si == 0 else 2,
                        succession=suc if si == 0 else 0)
            m.eval()
            err = {k: evaluate_fewshot(m, te[k], 0) for k in TASKS}
            row = {"tras": t, **{k: err[k]["escalares_MAE"] + err[k]["campos_L2rel"] for k in TASKS},
                   "especies": int(m.moe.active.sum())}
            if eco:
                row["chesson"] = chesson_diagnostic(m, v)
            rows.append(row)
            eps = tr[t]["episode"].unique()[:5]
            replay.append(subset(tr[t], torch.nonzero(torch.isin(tr[t]["episode"], eps)).flatten()))
            print(name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
        final = rows[-1]
        res["variantes"][name] = {"etapas": rows, "error_final_medio": sum(final[k] for k in TASKS) / 3,
                                  "olvido_sano": final["sano"] - rows[0]["sano"]}
    res["segundos"] = round(time.time() - t0, 1)
    json.dump(res, open(f"results/ecologia_s{seed}.json", "w"), ensure_ascii=False, indent=1)
    print("listo", res["segundos"])


if __name__ == "__main__":
    main()
