"""Afirmación 3 (v0.7): qué simular, con el MISMO presupuesto para todas las estrategias.

Protocolo (corrige v0.4-v0.6, donde el azar gastaba 16 simulaciones y la colmena 4-12):
    - Reserva por ronda: N_CONOCIDOS episodios de los regímenes de entrenamiento y N_NUEVOS de un
      régimen nuevo (cambio visible). El azar elige de una reserva desbalanceada, como en la práctica.
    - Cada estrategia de línea base gasta exactamente PRESUPUESTO por ronda; se registra el error
      después de cada ronda: curva de error contra simulaciones acumuladas.
    - La colmena puede gastar menos (decide parar). Su resultado se compara con las curvas de las
      demás en el número de simulaciones que ELLA gastó (no contra el presupuesto completo).
    - Regla de paro: arrepentimiento = error de la colmena al parar - mínimo de la curva del valor
      codicioso (misma señal, sin abejas) hasta ese número de simulaciones.

    PYTHONPATH=. python3 experiments/exp_presupuesto.py edp1d 0 [epocas] [estrategias]
Estrategias: aleatoria, novedad_codiciosa, valor_codicioso, varianza, coreset, colmena
"""
import copy
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from eho import Gemelo
from eho.dominios import concat, subset
from eho.baselines import select

import os
RONDAS, PRESUPUESTO, N_CONOCIDOS, N_NUEVOS = 4, 4, 24, 6
# EHO_AGOTA=r: a partir de la ronda r la reserva ya no trae casos nuevos (solo conocidos). Es el
# escenario donde parar es lo correcto: mide cuántas simulaciones ahorra la regla de parada.
AGOTA = int(os.environ.get("EHO_AGOTA", "-1"))
if AGOTA >= 0:
    RONDAS = 6
NUEVO = {"sangre": {"H_range": (0.55, 0.70)}, "edp1d": {"k_min": 4}, "burgers1d": {"k_min": 4}}
DOMINIO = {
    "sangre": {"nombre": "sangre", "simuladores": [{"simulador": "quemada"}, {"simulador": "carreau"}],
               "regimenes": ["sano", "dt2"]},
    "edp1d": {"nombre": "edp1d", "regimenes": ["difusivo", "advectivo"]},
    "burgers1d": {"nombre": "burgers1d", "regimenes": ["viscoso", "poco_viscoso"]},
}
TODAS = ["aleatoria", "novedad_codiciosa", "valor_codicioso", "varianza", "coreset", "colmena", "colmena_v08"]


def stop_regret(col, ref, key="nuevo_escalares"):
    """Regla de paro de la colmena frente al valor codicioso (misma señal, sin abejas):
    a_mismas_sim: error de la colmena - curva codiciosa interpolada en las simulaciones que ella gastó
                  (negativo = la colmena elige mejor por simulación);
    perdida_por_parar: error de la colmena - error codicioso con el presupuesto completo
                  (positivo = parar costó precisión; es el precio del ahorro)."""
    n, e = col[-1]["acumuladas"], col[-1][key]
    xs, ys = [h["acumuladas"] for h in ref], [h[key] for h in ref]
    j = max(i for i, x in enumerate(xs) if x <= n)
    y = ys[j] if j == len(xs) - 1 or xs[j] == n else ys[j] + (ys[j + 1] - ys[j]) * (n - xs[j]) / (xs[j + 1] - xs[j])
    return {"simulaciones": n, "simulaciones_codicioso": xs[-1], "error_nuevo": e,
            "codicioso_mismas_sim": y, "a_mismas_sim": e - y,
            "codicioso_presupuesto_completo": ys[-1], "perdida_por_parar": e - ys[-1]}


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    strats = sys.argv[4].split(",") if len(sys.argv) > 4 else TODAS
    t0 = time.time()
    cfg = {"nombre": f"presupuesto_{name}", "dominio": DOMINIO[name], "colmena": {"n_agentes": 4},
           "entrenamiento": {"epocas": epochs, "semilla": seed, "tasa": os.environ.get("EHO_TASA", "constante")},
           "activo": {"presupuesto": PRESUPUESTO}}
    base = Gemelo.desde_config(cfg).entrenar(verbose=False)
    regs = base.cfg.dominio.regimenes
    n_src = len(base.dominios)
    vk = {"simulador": "pries"} if name == "sangre" else {}         # en sangre, la verdad es un 3er modelo
    known_t, new_t = base.dominio_variante(0, **vk), base.dominio_variante(0, **vk, **NUEVO[name])
    test_known = concat(*[known_t.generate(1, 10, regs[i % len(regs)], 7000 + i) for i in range(16)])
    test_new = concat(*[new_t.generate(1, 10, regs[i % len(regs)], 7500 + i) for i in range(16)])

    def pool(r, a):
        kn = base.dominio_variante(a % n_src)
        nw = base.dominio_variante(a % n_src, **NUEVO[name])
        parts = [kn.generate(1, 8, regs[i % len(regs)], 100000 * seed + 1000 * r + i) for i in range(N_CONOCIDOS)]
        cual = kn if (AGOTA >= 0 and r >= AGOTA) else nw           # reserva agotada: los "nuevos" son conocidos
        parts += [cual.generate(1, 8, regs[i % len(regs)], 100000 * seed + 1000 * r + 500 + i) for i in range(N_NUEVOS)]
        return concat(*parts)

    def evaluate(g):
        k, n = g.evaluar(test_known), g.evaluar(test_new)
        return {"conocido_escalares": k["escalares_MAE"], "conocido_campos": k["campos_L2rel"],
                "nuevo_escalares": n["escalares_MAE"], "nuevo_campos": n["campos_L2rel"]}

    res = {"dominio": name, "semilla": seed, "presupuesto_por_ronda": PRESUPUESTO, "rondas": RONDAS,
           "inicial": evaluate(base), "estrategias": {}}
    for strat in strats:
        g = copy.deepcopy(base)
        if strat == "colmena_v08":
            g.cfg.activo.parada = "mvt"                       # forrajeo + urgencia + descubrimiento mínimo
            g.cfg.activo.presupuesto_total = RONDAS * PRESUPUESTO
        gen = torch.Generator().manual_seed(seed)
        hist, total = [{"ronda": -1, "acumuladas": 0, **res["inicial"]}], 0
        for r in range(RONDAS):
            pools = [pool(r, a) for a in range(len(g.agentes))]
            ids = select("colmena" if strat == "colmena_v08" else strat, g, pools[0], PRESUPUESTO, gen=gen)
            n_new = sum(1 for e in ids if e >= N_CONOCIDOS) if not (AGOTA >= 0 and r >= AGOTA) else 0
            if ids:
                g.incorporar([subset(p, torch.nonzero(torch.isin(p["episode"], torch.tensor(ids))).flatten())
                              for p in pools])
            total += len(ids)
            ev = evaluate(g) if ids else {k: v for k, v in hist[-1].items() if k not in ("ronda", "acumuladas", "nuevas")}
            hist.append({"ronda": r, "acumuladas": total, "nuevas": n_new, **ev})
            print(strat, r, total, n_new, {k: round(v, 4) for k, v in ev.items()}, flush=True)
        res["estrategias"][strat] = hist
    for c in ("colmena", "colmena_v08"):
        if c in res["estrategias"] and "valor_codicioso" in res["estrategias"]:
            res[f"{c}_paro"] = stop_regret(res["estrategias"][c], res["estrategias"]["valor_codicioso"])
    res["segundos"] = round(time.time() - t0, 1)
    res["agota_en_ronda"] = AGOTA
    tag = ("" if strats == TODAS else "_" + "-".join(strats)) + (f"_agota{AGOTA}" if AGOTA >= 0 else "")
    json.dump(res, open(f"results/presupuesto_{name}_s{seed}{tag}.json", "w"), ensure_ascii=False, indent=1)
    print("listo", res["segundos"], "s")


if __name__ == "__main__":
    main()
