"""Afirmación 1 (v0.7): adaptación a un episodio nuevo con k ejemplos, sin reentrenar.

Variantes con el MISMO núcleo y los mismos datos:
    sin memoria            k no se usa
    memoria episódica      atención sobre (clave, resultado) del contexto: el proceso neuronal atentivo
    memoria completa       episódica + neuronal de largo plazo + plasmodial
    ajuste fino afinado    sobre el modelo sin memoria; pasos, tasa y "todo"/"cabeza" elegidos en validación
Se reporta el error y los segundos de adaptación por episodio (el costo de cada opción).

    PYTHONPATH=. python3 experiments/exp_pocos_ejemplos.py burgers1d 0 [epocas]
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
from eho.baselines import tuned_finetune, memory_adaptation_cost
from eho.mezcla import memory_predictions, finetune_predictions, ArbitroRapidoLento, ArbitroMultiple, errors

SETUP = {
    "sangre": dict(dom={}, train=["sano", "dt2"], test="prediabetes"),
    "edp1d": dict(dom={}, train=["difusivo", "advectivo"], test="mixto"),
    "burgers1d": dict(dom={}, train=["viscoso", "poco_viscoso"], test="intermedio"),
}
KS = [0, 1, 2, 5]


TASA = os.environ.get("EHO_TASA", "constante")       # coseno: decaimiento de la tasa (v0.9)
ETQ = "cos" if TASA == "coseno" else ""


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    s = SETUP[name]
    t0 = time.time()
    dom = crear_dominio(name, **s["dom"])
    tr = concat(*[dom.generate(40, 8, r, 1000 * seed + i) for i, r in enumerate(s["train"])])
    va = concat(*[dom.generate(6, 8, r, 1000 * seed + 50 + i) for i, r in enumerate(s["train"])])
    va_t = dom.generate(8, 12, s["test"], 1000 * seed + 70)            # validación del ajuste fino
    te = dom.generate(20, 12, s["test"], 1000 * seed + 90)
    dv = int(dom.memory_values({k: v[:2] for k, v in tr.items()}).shape[-1])
    res = {"dominio": name, "semilla": seed, "variantes": {}}
    models = {}
    for mem in ("ninguna", "episodica", "completa"):
        torch.manual_seed(seed)
        m = EHOCentral(dom, memory=mem, d_val=dv)
        fit_central(m, tr, va, epochs=epochs, schedule=TASA)
        m.eval()
        models[mem] = m
        row = {f"k={k}": evaluate_fewshot(m, te, k) for k in KS}
        row["segundos_adaptacion_k=5"] = memory_adaptation_cost(m, te, 5)
        res["variantes"][f"memoria_{mem}"] = row
        print(mem, {k: {kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)
                    for k, v in row.items()}, flush=True)
    m0 = models["ninguna"]
    factory = lambda: EHOCentral(dom, memory="ninguna", d_val=dv)
    state = m0                                   # se copia entero (buffers de memoria de tamaño variable)
    grid = [dict(steps=st, lr=lr, mode=md) for st in (10, 30) for lr in (1e-3, 3e-3) for md in ("todo", "cabeza")]
    row = {}
    for k in (1, 2, 5):
        row[f"k={k}"] = tuned_finetune(factory, state, va_t, te, k, grid=grid, seed=seed)
        print("ajuste fino", k, row[f"k={k}"], flush=True)
    res["variantes"]["ajuste_fino_afinado"] = row
    # v0.8: árbitro rápido-lento (memoria completa + ajuste fino) con pesos de validación por k
    row = {}
    for k in (1, 2, 5):
        h = {kk: v for kk, v in res["variantes"]["ajuste_fino_afinado"][f"k={k}"]["hiperparametros"].items()}
        arb = ArbitroRapidoLento().fit(memory_predictions(models["completa"], va_t, k, seed),
                                       finetune_predictions(factory, state, va_t, k, seed=seed, **h))
        mem = memory_predictions(models["completa"], te, k, seed)
        ft = finetune_predictions(factory, state, te, k, seed=seed, **h)
        S, F = arb.combine(mem, ft)
        row[f"k={k}"] = {**errors(S, F, mem[2], mem[3]), "peso_memoria": arb.a_s.tolist()}
        print("mezcla", k, row[f"k={k}"], flush=True)
    res["variantes"]["mezcla_rapido_lento"] = row
    # v0.9: árbitro entre memoria episódica, memoria completa y ajuste fino
    row = {}
    for k in (1, 2, 5):
        h = res["variantes"]["ajuste_fino_afinado"][f"k={k}"]["hiperparametros"]
        mv = [memory_predictions(models[m_], va_t, k, seed) for m_ in ("episodica", "completa")]
        val = mv + [finetune_predictions(factory, state, va_t, k, seed=seed, **h)]
        arb = ArbitroMultiple().fit(val, mv[0][2], mv[0][3])
        mt = [memory_predictions(models[m_], te, k, seed) for m_ in ("episodica", "completa")]
        S, F = arb.combine(mt + [finetune_predictions(factory, state, te, k, seed=seed, **h)])
        row[f"k={k}"] = {**errors(S, F, mt[0][2], mt[0][3]), "pesos_escalares": arb.w_s.mean(1).tolist()}
        print("arbitro triple", k, row[f"k={k}"], flush=True)
    res["variantes"]["arbitro_triple"] = row
    res["segundos"] = round(time.time() - t0, 1)
    json.dump(res, open(f"results/pocos_ejemplos_{name}{ETQ}_s{seed}.json", "w"), ensure_ascii=False, indent=1)
    print("listo", res["segundos"], "s")


if __name__ == "__main__":
    main()
