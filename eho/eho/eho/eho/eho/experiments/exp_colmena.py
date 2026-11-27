"""Experimento G (v0.3): colmena calibrada, diversa, con inspección del sitio.

Dominio sangre: los agentes se entrenan con DOS simuladores distintos (Quemada y Carreau-Yasuda,
sustitutos de Mirheo y HemoCell) y la "realidad" es distinta de ambos (media geométrica).
Dominio edp1d: diversidad solo de datos, arquitectura y memoria.

Pruebas:
  dentro de distribución   pacientes nuevos de prediabetes (sangre) / fluidos de régimen mixto (edp1d)
  fuera de distribución    sangre: geometría nueva (D = 3-5.5 μm); edp1d: régimen extremo
Reglas comparadas (todas calibradas en validación a la misma cobertura objetivo, 80 %):
  colmena diversa con inspección    quórum calibrado + confianza por error de contexto
  promedio ponderado diverso        misma señal, abstención por error de contexto medio
  colmena homogénea con inspección  mismos 4 agentes sin diversidad
Además, selección del siguiente experimento (solo sangre): candidatos con 20 % de geometrías nuevas.
Uso: python experiments/exp_colmena.py <sangre|edp1d> <semilla> [épocas]
"""
import json, sys
import torch

sys.path.insert(0, ".")
from eho import default_device
from eho.dominios import Sangre, AdveccionDifusion1D, concat
from eho.hive import build_hive, predict_with_inspection, candidate_scores
from eho.colony import ColonyPredictor, BeeColony, colony_select


def setup(name, seed):
    if name == "sangre":
        doms = [Sangre(simulador="quemada"), Sangre(simulador="carreau")]
        regs = ["sano", "dt2"]
        trains = [concat(*[d.generate(60, 8, r, 1000 * seed + 10 * j + i) for i, r in enumerate(regs)])
                  for j, d in enumerate(doms)]
        real = Sangre(simulador="real")
        val = concat(*[real.generate(8, 8, r, 1000 * seed + 50 + i) for i, r in enumerate(regs)])
        idd = real.generate(20, 10, "prediabetes", 555)
        odd = Sangre(simulador="real", D_range=(3.0, 5.5)).generate(20, 10, "prediabetes", 556)
        return doms, trains, val, idd, odd, 2
    dom = AdveccionDifusion1D()
    regs = ["difusivo", "advectivo"]
    tr = concat(*[dom.generate(60, 8, r, 1000 * seed + i) for i, r in enumerate(regs)])
    val = concat(*[dom.generate(10, 8, r, 1000 * seed + 50 + i) for i, r in enumerate(regs)])
    return [dom], [tr], val, dom.generate(20, 10, "mixto", 555), dom.generate(20, 10, "extremo", 556), 5


def evaluate_rule(acc, pred, Y, scale, is_ood):
    err = ((pred - Y).abs() / scale).mean(-1)
    return {"cobertura": float(acc.float().mean()),
            "error_aceptadas": float(err[acc].mean()) if acc.any() else None,
            "aceptadas_dentro": float(acc[~is_ood].float().mean()),
            "aceptadas_fuera": float(acc[is_ood].float().mean())}


def run_hive(agents, val, idd, odd, target=0.8):
    Pv, Yv, CEv = predict_with_inspection(agents, val)
    scale = (Pv.mean(0) - Yv).abs().median(0).values.clamp(min=1e-4) * 2.0
    Pi, Yi, CEi = predict_with_inspection(agents, idd)
    Po, Yo, CEo = predict_with_inspection(agents, odd)
    P, Y, CE = torch.cat([Pi, Po], 1), torch.cat([Yi, Yo]), torch.cat([CEi, CEo], 1)
    is_ood = torch.cat([torch.zeros(Yi.shape[0], dtype=torch.bool), torch.ones(Yo.shape[0], dtype=torch.bool)])
    out = {"error_promedio_sin_abstencion": float(((P.mean(0) - Y).abs() / scale).mean())}
    # colmena calibrada con inspección
    cp = ColonyPredictor(scale).fit(Pv, Yv, CEv, target_coverage=target)
    pred, acc, _ = cp.predict(P, CE)
    out["colmena_inspeccion"] = evaluate_rule(acc, pred, Y, scale, is_ood)
    # promedio ponderado con la misma señal, calibrado igual
    tau = CEv.median(1).values.clamp(min=1e-6).unsqueeze(-1)
    def wmean(P_, CE_):
        w = torch.exp(-CE_ / (2 * tau)); w = w / w.sum(0, keepdim=True)
        return (w.unsqueeze(-1) * P_).sum(0)
    thr = CEv.mean(0).quantile(target)
    out["promedio_ponderado"] = evaluate_rule(CE.mean(0) <= thr, wmean(P, CE), Y, scale, is_ood)
    # colmena sin inspección (solo acuerdo), calibrada
    cp0 = ColonyPredictor(scale).fit(Pv, Yv, None, target_coverage=target)
    pred0, acc0, _ = cp0.predict(P, None)
    out["colmena_solo_acuerdo"] = evaluate_rule(acc0, pred0, Y, scale, is_ood)
    return out


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    dev = default_device()
    doms, trains, val, idd, odd, dv = setup(name, seed)
    res = {"dominio": name, "semilla": seed}
    for label, diverse in [("diversa", True), ("homogenea", False)]:
        agents = build_hive(doms, trains, val, n_agents=4, diverse=diverse, epochs=epochs, d_val=dv,
                            seed=seed, device=dev)
        res[label] = run_hive(agents, val, idd, odd)
        res[label]["agentes"] = [m.hive_info for m in agents]
        print(label, json.dumps({k: v for k, v in res[label].items() if k != "agentes"}, ensure_ascii=False), flush=True)
        if name == "sangre" and diverse:
            # selección del siguiente experimento: 8 condiciones conocidas + 2 geometrías nuevas
            picks, picks_mean, abst = [], [], 0
            for t in range(10):
                cand = concat(Sangre(simulador="real").generate(8, 1, "prediabetes", 9000 + t),
                              Sangre(simulador="real", D_range=(3.0, 5.5)).generate(2, 1, "prediabetes", 9500 + t))
                S = candidate_scores(agents, cand)
                choice, _ = colony_select(S, BeeColony(quorum=0.5, seed=t))
                picks_mean.append(int(S.mean(0).argmax()) >= 8)
                if choice is None:
                    abst += 1
                else:
                    picks.append(choice >= 8)
            res["seleccion"] = {"rondas": 10, "abstenciones": abst,
                                "colmena_elige_geometria_nueva": float(torch.tensor(picks, dtype=torch.float).mean()) if picks else None,
                                "argmax_medio_elige_geometria_nueva": float(torch.tensor(picks_mean, dtype=torch.float).mean()),
                                "azar": 0.2}
            print("seleccion", res["seleccion"], flush=True)
    json.dump(res, open(f"results/colmena_{name}_s{seed}.json", "w"), ensure_ascii=False, indent=1)
