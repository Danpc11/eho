"""Experimento B: aprendizaje continuo sano -> prediabetes -> DT2.

Ambos modelos reciben la misma repetición (40 muestras por tarea previa).
Mide el error en cada tarea después de cada etapa (olvido = aumento del error).
"""
import json, sys
import torch

sys.path.insert(0, ".")
from eho.data.synthetic import make_dataset, concat, subset
from eho import default_device
from eho.model import EHO
from eho.train import fit, evaluate

TASKS = ["sano", "prediabetes", "dt2"]

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    train = {t: make_dataset(400, t, seed=100 + i + 1000 * seed) for i, t in enumerate(TASKS)}
    val = {t: make_dataset(100, t, seed=200 + i + 1000 * seed) for i, t in enumerate(TASKS)}
    test = {t: make_dataset(300, t, seed=300 + i) for i, t in enumerate(TASKS)}
    out = {}
    for name, eco in [("EHO completo", True), ("sin ecologia", False)]:
        torch.manual_seed(seed)
        m = EHO(use_ecology=eco).to(default_device())
        replay, rows, species = None, [], []
        for si, t in enumerate(TASKS):
            v = concat(*[val[k] for k in TASKS[:si + 1]])
            log = fit(m, train[t], v, epochs=epochs, verbose=False, replay=replay,
                      eco_warmup=5 if si == 0 else 2, invade_every=3)
            row = {k: evaluate(m, test[k]) for k in TASKS}
            rows.append({"tras": t, **{k: row[k]["log_eta_MAE"] + row[k]["cfl_MAE"] + row[k]["campos_L2rel"] for k in TASKS}})
            species.append(int(m.moe.active.sum()))
            keep = subset(train[t], torch.arange(40))
            replay = keep if replay is None else concat(replay, keep)
            print(name, rows[-1], "especies", species[-1], flush=True)
        out[name] = {"matriz": rows, "especies": species}
    json.dump(out, open("results/continuo_s%d.json" % seed, "w"), ensure_ascii=False, indent=1)
