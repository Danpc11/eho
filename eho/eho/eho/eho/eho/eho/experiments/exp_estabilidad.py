"""Estabilidad del entrenamiento (v0.9): ecología sí/no x tasa constante/coseno, 3 semillas.

    PYTHONPATH=. python3 experiments/exp_estabilidad.py
"""
import sys, json, torch, time
sys.path.insert(0, ".")
from eho.dominios import crear_dominio, concat
from eho.central import EHOCentral
from eho.train_central import fit_central, evaluate_fewshot
SET = {"sangre": (["sano", "dt2"], "prediabetes"), "burgers1d": (["viscoso", "poco_viscoso"], "intermedio")}
res = {}
for name in ("sangre", "burgers1d"):
    regs, test = SET[name]; dom = crear_dominio(name)
    for seed in (0, 1, 2):
        tr = concat(*[dom.generate(40, 8, r, 1000 * seed + i) for i, r in enumerate(regs)])
        va = concat(*[dom.generate(6, 8, r, 1000 * seed + 50 + i) for i, r in enumerate(regs)])
        te = dom.generate(20, 12, test, 1000 * seed + 90)
        for eco in (True, False):
            for sch in ("constante", "coseno"):
                torch.manual_seed(seed)
                m = EHOCentral(dom, memory="ninguna", ecology=eco)
                fit_central(m, tr, va, epochs=20, schedule=sch); m.eval()
                e = evaluate_fewshot(m, te, 0)
                res.setdefault(name, {}).setdefault(f"eco={eco},lr={sch}", []).append([e["escalares_MAE"], e["campos_L2rel"]])
                print(name, seed, eco, sch, round(e["escalares_MAE"], 4), round(e["campos_L2rel"], 4), flush=True)
json.dump(res, open("results/estabilidad.json", "w"), indent=1)
