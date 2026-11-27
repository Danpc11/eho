"""Experimento E: qué simular después, con presupuesto de cómputo fijo.

Brazos: aleatorio, adquisición (novedad + desacuerdo)/costo, política REINFORCE.
El "simulador" es el generador sintético; el costo es el proxy de DPD.
"""
import json, sys, math
import torch

sys.path.insert(0, ".")
from eho.data.synthetic import (make_dataset, concat, subset, ground_truth, make_condition_vector,
                                sample_theta, sample_conditions)
from eho import default_device
from eho.model import EHO
from eho.train import fit, evaluate
from eho.active import acquisition, dpd_cost_proxy, DesignPolicy, train_design_policy


def simulate(theta, H, D, g, n_points=128):
    pts, f, s = ground_truth(theta, H, D, g, n_points)
    return {"points": pts, "cond": make_condition_vector(theta, H, D, g), "gdot": g, "fields": f,
            "scalars": s, "theta": theta, "H": H, "D": D, "g": g}


def design_to_inputs(xi):
    theta = xi[:, :4]
    H = 0.20 + 0.30 * xi[:, 4]
    D = torch.exp(math.log(6.0) + (math.log(60.0) - math.log(6.0)) * xi[:, 5])
    g = torch.exp(math.log(500.0) * xi[:, 6])
    return theta, H, D, g


def pool(n, seed):
    torch.manual_seed(seed)
    k = n // 3
    th = torch.cat([sample_theta(k, "sano"), sample_theta(k, "prediabetes"), sample_theta(n - 2 * k, "dt2")])
    H, D, g = sample_conditions(n)
    return th, H, D, g


if __name__ == "__main__":
    rounds, add, budget_epochs = 3, 60, 20
    test = concat(make_dataset(200, "sano", seed=901), make_dataset(200, "prediabetes", seed=902),
                  make_dataset(200, "dt2", seed=903))
    val = concat(make_dataset(50, "sano", seed=904), make_dataset(50, "dt2", seed=905))
    out = {}
    for seed in [0, 1]:
        for arm in ["aleatorio", "adquisicion", "politica_RL"]:
            torch.manual_seed(seed)
            th, H, D, g = pool(120, 50 + seed)
            train = simulate(th, H, D, g)
            cost = float(dpd_cost_proxy(H, D, g).sum())
            m = EHO().to(default_device())
            fit(m, train, val, epochs=budget_epochs, verbose=False)
            curve = [{"muestras": 120, "costo": cost, **evaluate(m, test)}]
            for r in range(rounds):
                cth, cH, cD, cg = pool(1500, 1000 * seed + 10 * r + 7)
                if arm == "aleatorio":
                    idx = torch.randperm(1500)[:add]
                    new = simulate(cth[idx], cH[idx], cD[idx], cg[idx])
                elif arm == "adquisicion":
                    cand = simulate(cth, cH, cD, cg, n_points=64)
                    score = acquisition(m, cand, cost=dpd_cost_proxy(cH, cD, cg))
                    idx = score.topk(add).indices
                    new = simulate(cth[idx], cH[idx], cD[idx], cg[idx])
                else:
                    pol = DesignPolicy(d_ctx=1)

                    def reward(xi):
                        t, h, d, gg = design_to_inputs(xi)
                        c = simulate(t, h, d, gg, n_points=64)
                        return acquisition(m, c, cost=dpd_cost_proxy(h, d, gg))

                    train_design_policy(pol, reward, torch.ones(1, 1), iters=40, batch=64)
                    with torch.no_grad():
                        xi, _ = pol.sample(torch.ones(add, 1))
                    new = simulate(*design_to_inputs(xi))
                train = concat(train, new)
                cost += float(dpd_cost_proxy(new["H"], new["D"], new["g"]).sum())
                fit(m, train, val, epochs=budget_epochs // 2, verbose=False)
                curve.append({"muestras": int(train["points"].shape[0]), "costo": cost, **evaluate(m, test)})
            out[f"{arm}_s{seed}"] = curve
            last = curve[-1]
            print(arm, seed, {k: round(v, 4) for k, v in last.items()}, flush=True)
    json.dump(out, open("results/activo.json", "w"), ensure_ascii=False, indent=1)
