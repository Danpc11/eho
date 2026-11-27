"""Memoria bayesiana y diseño de mediciones orientado a un objetivo (v0.10).

Pacientes de un régimen intermedio no visto. De cada paciente: 4 condiciones OBJETIVO (lo que interesa
predecir) y 8 condiciones candidatas que se le podrían medir.

A. Pocos ejemplos (mediciones al azar, k = 0, 1, 2, 5): error en el objetivo de
     base (sin mediciones), memoria del EHO (media del ensamble con contexto) y memoria bayesiana;
     de la bayesiana, además, cobertura del intervalo del 95 % (calibración de la incertidumbre).
B. Qué medir (k = 1, 2, 3 mediciones elegidas): error en el objetivo con la memoria bayesiana según el
     criterio de elección: al azar, desacuerdo entre agentes (el sondeo de v0.8), D-óptimo (información
     sobre el estado latente) y orientado al objetivo (reducción de varianza del objetivo).

    PYTHONPATH=. python3 experiments/exp_memoria_bayes.py burgers1d 0 [epocas]
"""
import json
import os
import sys

import torch

sys.path.insert(0, ".")
from eho import Gemelo
from eho.dominios import concat, subset

SETUP = {"sangre": (["sano", "dt2"], "prediabetes", [{"simulador": "quemada"}, {"simulador": "carreau"}]),
         "edp1d": (["difusivo", "advectivo"], "mixto", None), "burgers1d": (["viscoso", "poco_viscoso"], "intermedio", None)}
BASE, N_PAC, N_OBJ = 70_000_000, 30, 4


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    regs, inter, sims = SETUP[name]
    dom = {"nombre": name, "regimenes": regs}
    if sims:
        dom["simuladores"] = sims
    g = Gemelo.desde_config({"nombre": f"mb_{name}", "dominio": dom, "colmena": {"n_agentes": 4},
                             "entrenamiento": {"epocas": epochs, "semilla": seed,
                                               "tasa": os.environ.get("EHO_TASA", "constante")}}).entrenar(verbose=False)
    g.ajustar_memoria_bayesiana(r=4)
    d0 = g.dominio_variante(0)
    gen = torch.Generator().manual_seed(seed)
    A = {m: {k: [] for k in (0, 1, 2, 5)} for m in ("base", "memoria_eho", "bayes")}
    cob = {k: [] for k in (0, 1, 2, 5)}
    Bres = {c: {k: [] for k in (1, 2, 3)} for c in ("aleatorio", "desacuerdo", "d_optimo", "objetivo")}
    var_obj = {c: {k: [] for k in (1, 2, 3)} for c in Bres}
    for i in range(N_PAC):
        ep = d0.generate(1, 12, inter if i % 2 else regs[(i // 2) % len(regs)], BASE + 1000 * seed + i)
        obj, pool = subset(ep, torch.arange(N_OBJ)), subset(ep, torch.arange(N_OBJ, 12))
        y = obj["scalars"]
        mae = lambda p: float((p - y).abs().mean())
        for k in (0, 1, 2, 5):
            ctx = subset(pool, torch.randperm(8, generator=gen)[:k]) if k else None
            rb = g.predecir_bayes(obj, ctx)
            A["base"][k].append(mae(rb["base"]))
            A["bayes"][k].append(mae(rb["escalares"]))
            A["memoria_eho"][k].append(mae(torch.stack([a(obj, context=ctx)["scalars"].cpu() for a in g.agentes]).mean(0)))
            cob[k].append(float(((rb["escalares"] - y).abs() <= 1.96 * rb["desviacion"]).float().mean()))
        P = torch.stack([a(pool)["scalars"].cpu() for a in g.agentes])
        desac = (P / g.escala).var(0).mean(-1).argsort(descending=True).tolist()
        for k in (1, 2, 3):
            for c in Bres:
                if c == "desacuerdo":
                    idx = desac[:k]
                    d = g.disenar_mediciones(pool, obj, n=0)
                else:
                    d = g.disenar_mediciones(pool, obj, n=k, criterio=c)
                    idx = d["indices"] if c != "aleatorio" else torch.randperm(8, generator=gen)[:k].tolist()
                rb = g.predecir_bayes(obj, subset(pool, torch.tensor(idx)))
                Bres[c][k].append(mae(rb["escalares"]))
                var_obj[c][k].append(float((rb["desviacion"] ** 2).mean()))
    mean = lambda x: float(torch.tensor(x).mean())
    res = {"dominio": name, "semilla": seed, "pacientes": N_PAC,
           "A_pocos_ejemplos": {m: {str(k): mean(v) for k, v in d.items()} for m, d in A.items()},
           "A_cobertura_95_bayes": {str(k): mean(v) for k, v in cob.items()},
           "B_que_medir": {c: {str(k): mean(v) for k, v in d.items()} for c, d in Bres.items()},
           "B_varianza_predicha_objetivo": {c: {str(k): mean(v) for k, v in d.items()} for c, d in var_obj.items()}}
    json.dump(res, open(f"results/memoria_bayes_{name}_s{seed}.json", "w"), ensure_ascii=False, indent=1)
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, {a: ({kk: round(x, 4) for kk, x in b.items()} if isinstance(b, dict) else round(b, 3)) for a, b in v.items()}, flush=True)


if __name__ == "__main__":
    main()
