"""Experimento H (v0.5): ciclo de aprendizaje activo decidido por la colmena.

Cada ronda hay 30 episodios candidatos (24 conocidos + 6 nuevos: POLICITEMIA, hematocrito 0.55-0.70,
en sangre; condiciones iniciales de alta frecuencia en advección-difusión).
v0.5: todos los agentes con memoria plasmodial (que se fusiona al llenarse) y ajuste con ancla
λ Σ (w - w0)² más repetición de 60 episodios para olvidar menos lo conocido. Presupuesto: hasta 4 simulaciones
por ronda, 3 rondas. Tras simular, cada agente se reentrena con sus datos nuevos (con su propio
simulador) más repetición de los anteriores, y consolida su memoria.

Estrategias:
  aleatoria   4 candidatos al azar
  promedio    los 4 con mayor valor medio de novedad
  colmena     la colmena elige uno a uno (valor absoluto) y PUEDE DETENERSE si nada vale la pena
Métrica: error de los agentes (k = 2 ejemplos de contexto) en prueba conocida y nueva, por ronda.
Uso: python experiments/exp_activo_colmena.py <sangre|edp1d> <semilla> [épocas_base]
"""
import copy, json, sys
import torch

sys.path.insert(0, ".")
from eho import default_device
from eho.dominios import Sangre, AdveccionDifusion1D, concat, subset
from eho.hive import build_hive
from eho.active import NoveltyValue
from eho.colony import BeeColony, colony_select
from eho.train_central import fit_central, evaluate_fewshot

ROUNDS, BUDGET, N_KNOWN, N_NEW = 3, 4, 24, 6


def make_domain(name):
    if name == "sangre":
        sims = [Sangre(simulador="quemada"), Sangre(simulador="carreau")]
        known = lambda sim: Sangre(simulador=sim)
        new = lambda sim: Sangre(simulador=sim, H_range=(0.55, 0.70))      # policitemia
        regs, test_reg, dv = ["sano", "dt2"], "prediabetes", 2
        test_sim = "real"
    else:
        sims = [AdveccionDifusion1D()]
        known = lambda sim: AdveccionDifusion1D()
        new = lambda sim: AdveccionDifusion1D(k_min=4)
        regs, test_reg, dv = ["difusivo", "advectivo"], None, 5
        test_sim = None
    return sims, known, new, regs, test_reg, dv, test_sim


def candidate_pool(known, new, regs, r, seed, sim):
    """30 episodios candidatos: los primeros N_KNOWN conocidos y los últimos N_NEW nuevos.
    Con la misma semilla, todos los simuladores ven exactamente las mismas condiciones."""
    parts = []
    for i in range(N_KNOWN):
        parts.append(known(sim).generate(1, 8, regs[i % 2], 100000 * seed + 1000 * r + i))
    for i in range(N_NEW):
        parts.append(new(sim).generate(1, 8, regs[i % 2], 100000 * seed + 1000 * r + 500 + i))
    return concat(*parts)


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    dev = default_device()
    sims, known, new, regs, test_reg, dv, test_sim = make_domain(name)
    simname = lambda d: getattr(d, "simulador", None)
    trains = [concat(*[s.generate(40, 8, r, 1000 * seed + i) for i, r in enumerate(regs)]) for s in sims]
    vsim = test_sim or simname(sims[0])
    val = concat(*[known(vsim).generate(6, 8, r, 1000 * seed + 50 + i) for i, r in enumerate(regs)])
    treg = lambda i: test_reg or regs[i % 2]
    test_known = concat(*[known(vsim).generate(1, 10, treg(i), 7000 + i) for i in range(16)])
    test_new = concat(*[new(vsim).generate(1, 10, treg(i), 7500 + i) for i in range(16)])

    base = build_hive(sims, trains, val, n_agents=4, diverse=True, epochs=epochs, d_val=dv, seed=seed, device=dev)
    sim_of = [simname(sims[a % len(sims)]) for a in range(4)]

    def evaluate(agents):
        ek = [evaluate_fewshot(m, test_known, 2) for m in agents]
        en = [evaluate_fewshot(m, test_new, 2) for m in agents]
        f = lambda L, k: sum(x[k] for x in L) / len(L)
        return {"conocido_escalares": f(ek, "escalares_MAE"), "conocido_campos": f(ek, "campos_L2rel"),
                "nuevo_escalares": f(en, "escalares_MAE"), "nuevo_campos": f(en, "campos_L2rel")}

    res = {"dominio": name, "semilla": seed, "inicial": evaluate(base), "estrategias": {}}
    print("inicial", res["inicial"], flush=True)
    for strat in ["aleatoria", "promedio", "colmena", "promedio_sin_ancla"]:
        agents = copy.deepcopy(base)
        anchor = 0.0 if strat == "promedio_sin_ancla" else 1e-2
        n_rep = 30 if strat == "promedio_sin_ancla" else 60
        new_data = [[] for _ in agents]
        hist = []
        gen = torch.Generator().manual_seed(seed)
        for r in range(ROUNDS):
            pool = candidate_pool(known, new, regs, r, seed, sim_of[0])
            nv = NoveltyValue(agents).fit(val)
            V, eps = nv.episode_values(pool)                                    # [A, 30]
            if strat == "aleatoria":
                chosen = torch.randperm(V.shape[1], generator=gen)[:BUDGET].tolist()
            elif strat.startswith("promedio"):
                chosen = V.mean(0).topk(BUDGET).indices.tolist()
            else:
                chosen, remaining = [], list(range(V.shape[1]))
                for t in range(BUDGET):
                    c, _ = colony_select(V[:, remaining], BeeColony(quorum=0.5, sigma=4.0, t_max=60.0, noise=0.05,
                                                                  seed=100 * r + t), normalize=False)
                    if c is None:
                        break                                                      # nada vale la pena
                    chosen.append(remaining.pop(c))
            n_new = sum(1 for c in chosen if c >= N_KNOWN)
            # simular lo elegido: cada agente con SU simulador
            for a, m in enumerate(agents):
                pool_a = candidate_pool(known, new, regs, r, seed, sim_of[a])
                for c in chosen:
                    new_data[a].append(subset(pool_a, torch.nonzero(pool_a["episode"] == eps[c]).flatten()))
                if new_data[a]:
                    nd = concat(*new_data[a])
                    all_eps = trains[a % len(trains)]["episode"].unique()
                    rep_eps = all_eps[torch.randperm(all_eps.numel(), generator=torch.Generator().manual_seed(seed + a))[:n_rep]]
                    rep = subset(trains[a % len(trains)], torch.nonzero(torch.isin(trains[a % len(trains)]["episode"], rep_eps)).flatten())
                    fit_central(m, concat(rep, nd), val, epochs=6, lr=1e-3, eco_warmup=1000, anchor_lambda=anchor)
                    m.consolidate(nd)
                    m.eval()
            ev = evaluate(agents)
            ev.update({"ronda": r, "simulaciones": len(chosen), "nuevas_elegidas": n_new})
            hist.append(ev)
            print(strat, ev, flush=True)
        res["estrategias"][strat] = hist
    json.dump(res, open(f"results/activo_colmena_{name}_s{seed}.json", "w"), ensure_ascii=False, indent=1)
