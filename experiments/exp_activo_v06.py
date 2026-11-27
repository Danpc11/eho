"""Experimento I (v0.6): ¿cuándo debe parar la colmena? Novedad sola frente a novedad + inspección.

Igual que el experimento H (v0.5) pero escrito con la API del Gemelo (eho/gemelo.py), que es la
que se usará con datos reales. Cada ronda hay 30 episodios candidatos (24 conocidos + 6 nuevos:
policitemia en sangre, condiciones iniciales de alta frecuencia en advección-difusión); presupuesto
de 4 simulaciones por ronda, 4 rondas.

Estrategias:
  aleatoria     4 candidatos al azar
  colmena_v05   la colmena decide con la NOVEDAD de la memoria plasmodial (se detenía demasiado pronto)
  colmena_v06_sin_exploradoras   novedad O error reciente de inspección (eho/active.py: InspectionValue)
  colmena_v06   lo anterior + exploradoras: si la colmena se abstiene, inspecciona 1 candidato cuya
                vecindad aún no se ha inspeccionado (parar exige evidencia de que ya se predice bien)

Uso: python experiments/exp_activo_v06.py <sangre|edp1d> <semilla> [épocas_base]
"""
import copy, json, sys
import torch

sys.path.insert(0, ".")
from eho import Gemelo
from eho.dominios import concat, subset

ROUNDS, BUDGET, N_KNOWN, N_NEW = 4, 4, 24, 6


def config(name, seed, epochs):
    if name == "sangre":
        dom = {"nombre": "sangre", "simuladores": [{"simulador": "quemada"}, {"simulador": "carreau"}],
               "regimenes": ["sano", "dt2"]}
    else:
        dom = {"nombre": "edp1d", "regimenes": ["difusivo", "advectivo"]}
    return {"nombre": f"exp_{name}", "dominio": dom, "colmena": {"n_agentes": 4},
            "entrenamiento": {"epocas": epochs, "semilla": seed},
            "activo": {"presupuesto": BUDGET, "usar_error": True}}


def nuevo_kw(name):
    return {"H_range": (0.55, 0.70)} if name == "sangre" else {"k_min": 4}


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    base = Gemelo.desde_config(config(name, seed, epochs))
    regs = base.cfg.dominio.regimenes
    n_src = len(base.dominios)
    # validación y prueba con el simulador "real" en sangre (media geométrica de ambos)
    vk = {"simulador": "real"} if name == "sangre" else {}
    known_t, new_t = base.dominio_variante(0, **vk), base.dominio_variante(0, **vk, **nuevo_kw(name))
    treg = (lambda i: "prediabetes") if name == "sangre" else (lambda i: regs[i % 2])
    val = concat(*[known_t.generate(6, 8, r, 1000 * seed + 50 + i) for i, r in enumerate(regs)])
    test_known = concat(*[known_t.generate(1, 10, treg(i), 7000 + i) for i in range(16)])
    test_new = concat(*[new_t.generate(1, 10, treg(i), 7500 + i) for i in range(16)])
    base.entrenar(val=val)

    def pool(r, a):
        """Candidatos de la ronda r tal como los simularía el agente a (con SU fuente)."""
        kn, nw = base.dominio_variante(a % n_src), base.dominio_variante(a % n_src, **nuevo_kw(name))
        parts = [kn.generate(1, 8, regs[i % 2], 100000 * seed + 1000 * r + i) for i in range(N_KNOWN)]
        parts += [nw.generate(1, 8, regs[i % 2], 100000 * seed + 1000 * r + 500 + i) for i in range(N_NEW)]
        return concat(*parts)

    def evaluate(g):
        k, n = g.evaluar(test_known), g.evaluar(test_new)
        return {"conocido_escalares": k["escalares_MAE"], "conocido_campos": k["campos_L2rel"],
                "nuevo_escalares": n["escalares_MAE"], "nuevo_campos": n["campos_L2rel"]}

    res = {"dominio": name, "semilla": seed, "inicial": evaluate(base), "estrategias": {}}
    print("inicial", res["inicial"], flush=True)
    strats = sys.argv[4].split(",") if len(sys.argv) > 4 else ["aleatoria", "colmena_v05", "colmena_v06"]
    out = f"results/activo_v06_{name}_s{seed}.json"
    try:
        res["estrategias"] = json.load(open(out))["estrategias"]       # conservar estrategias ya corridas
    except Exception:
        pass
    for strat in strats:
        g = copy.deepcopy(base)
        if strat == "colmena_v05":
            g.valor.err = None                       # solo novedad
        if strat == "colmena_v06_sin_exploradoras":
            g.cfg.activo.exploradoras = 0
        gen = torch.Generator().manual_seed(seed)
        hist, total = [], 0
        for r in range(ROUNDS):
            pools = [pool(r, a) for a in range(len(g.agentes))]
            eps = pools[0]["episode"].unique()
            if strat == "aleatoria":
                ids = eps[torch.randperm(eps.numel(), generator=gen)[:BUDGET]].tolist()
                info = []
            else:
                info = g.proponer(pools[0], semilla=100 * seed + r)
                ids = [e["episodio"] for e in info]
            n_new = sum(1 for e in ids if e >= N_KNOWN)
            if ids:
                nuevos = [subset(p, torch.nonzero(torch.isin(p["episode"], torch.tensor(ids))).flatten())
                          for p in pools]
                g.incorporar(nuevos)
            total += len(ids)
            ev = evaluate(g)
            ev.update({"ronda": r, "simulaciones": len(ids), "acumuladas": total, "nuevas_elegidas": n_new,
                       "valores": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k != "episodio"} for e in info]})
            hist.append(ev)
            print(strat, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items() if k != "valores"},
                  flush=True)
        res["estrategias"][strat] = hist
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
