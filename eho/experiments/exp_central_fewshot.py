"""Experimento F: adaptación con pocos ejemplos (memoria) en dos dominios.

Entrenamiento con dos regímenes; prueba con EPISODIOS NUEVOS de un tercer régimen intermedio
(nuevos pacientes / nuevos fluidos con parámetros ocultos). Para cada episodio de prueba se dan
k ejemplos etiquetados (contexto) y se mide el error en el resto.

Variantes:
  sin memoria        k = 0 (no puede adaptarse)
  ajuste fino        sin memoria + 30 pasos de gradiente con los k ejemplos de cada episodio
  episódica          memoria en contexto, sin gradiente
  completa           episódica + neuronal (Titans) + plasmodial
  oráculo            ve los parámetros ocultos (cota superior), sin memoria
Uso: python experiments/exp_central_fewshot.py <dominio: edp1d|sangre> <semilla> [épocas]
"""
import json, sys, time
import torch

sys.path.insert(0, ".")
from eho import default_device
from eho.dominios import AdveccionDifusion1D, Sangre, concat
from eho.central import EHOCentral
from eho.train_central import fit_central, evaluate_fewshot, finetune_fewshot

DOMS = {"edp1d": (AdveccionDifusion1D, ["difusivo", "advectivo"], "mixto", 5),
        "sangre": (Sangre, ["sano", "dt2"], "prediabetes", 2)}
KS = [0, 2, 5]

if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    Dom, train_regs, test_reg, dv = DOMS[name]
    dev = default_device()
    res = {"dominio": name, "semilla": seed, "variantes": {}}

    def data(dom):
        tr = concat(*[dom.generate(60, 8, r, seed=1000 * seed + i) for i, r in enumerate(train_regs)])
        va = concat(*[dom.generate(10, 8, r, seed=1000 * seed + 10 + i) for i, r in enumerate(train_regs)])
        te = dom.generate(20, 12, test_reg, seed=777)          # mismo conjunto de prueba en todas las semillas
        return tr, va, te

    for variant in ["sin memoria", "episodica", "completa", "oraculo"]:
        torch.manual_seed(seed)
        dom = Dom(**({"params_visible": True} if (variant == "oraculo" and name == "edp1d") else
                     {"theta_visible": True} if variant == "oraculo" else {}))
        tr, va, te = data(dom)
        mem = {"sin memoria": "ninguna", "oraculo": "ninguna", "episodica": "episodica", "completa": "completa"}[variant]
        factory = lambda: EHOCentral(dom, memory=mem, d_val=dv).to(dev)
        m = factory()
        t0 = time.time()
        fit_central(m, tr, va, epochs=epochs, eco_warmup=5)
        row = {"segundos": round(time.time() - t0, 1), "especies": int(m.moe.active.sum())}
        for k in KS:
            row[f"k={k}"] = evaluate_fewshot(m, te, k)
        if variant == "sin memoria":
            state = {k: v.clone() for k, v in m.state_dict().items()}
            for k in KS[1:]:
                row[f"ajuste_fino k={k}"] = finetune_fewshot(factory, state, te, k)
        if m.memory is not None and m.memory.mpc is not None:
            row["mpc"] = m.memory.mpc.stats()
        res["variantes"][variant] = row
        print(name, seed, variant, json.dumps({k: v for k, v in row.items()}, ensure_ascii=False), flush=True)
    json.dump(res, open(f"results/central_fewshot_{name}_s{seed}.json", "w"), ensure_ascii=False, indent=1)
