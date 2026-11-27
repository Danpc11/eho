"""¿Hace falta la dinámica de abejas para abstenerse? (v0.9.5)

Mismos agentes congelados, misma predicción (la media) y misma certificación Learn-then-Test para todas
las puntuaciones de confianza:
    desviacion     -desviación estándar entre agentes
    mad            -desviación absoluta mediana entre agentes
    distancia      -distancia media entre pares de agentes
    inspeccion     -error de inspección del contexto (media de los agentes)
    confianza      confianza media exp(-error / 2τ)
    clasificador   regresión logística del fallo con las tres primeras (entrenada en validación)
    colmena        apoyo final de la dinámica de abejas
Métricas en prueba: AUROC para detectar fallos (error escalado > tolerancia), área bajo la curva
riesgo-cobertura (menor es mejor), cobertura certificada por LTT con el riesgo pedido, riesgo real en lo
aceptado y milisegundos por predicción.

    PYTHONPATH=. python3 experiments/exp_abstencion.py burgers1d 0 [epocas]
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, ".")
from eho import Gemelo
from eho.dominios import concat
from eho.colony import ColonyPredictor, ltt_grid, ltt_threshold
from eho.hive import predict_with_inspection
from eho.baselines import auroc

SETUP = {"sangre": (["sano", "dt2"], "prediabetes", [{"simulador": "quemada"}, {"simulador": "carreau"}]),
         "edp1d": (["difusivo", "advectivo"], "mixto", None), "burgers1d": (["viscoso", "poco_viscoso"], "intermedio", None)}
BASE, RISK, DELTA, TOL = 60_000_000, 0.1, 0.1, 1.0


def scores(P, CE, scale, pr, clf=None):
    Ps = P / scale
    sd = Ps.std(0).mean(-1)
    med = Ps.median(0).values
    mad = (Ps - med).abs().median(0).values.mean(-1)
    A = P.shape[0]
    dist = torch.stack([(Ps[i] - Ps[j]).abs().mean(-1) for i in range(A) for j in range(i + 1, A)]).mean(0)
    insp = CE.mean(0)
    conf = pr.confidence(CE).mean(0)
    t0 = time.perf_counter()
    _, _, sup = pr.predict(P, CE)
    ms_col = (time.perf_counter() - t0) / P.shape[1] * 1000
    out = {"desviacion": -sd, "mad": -mad, "distancia": -dist, "inspeccion": -insp, "confianza": conf, "colmena": sup}
    X = torch.stack([sd, mad, insp], -1).log1p()
    if clf is not None:
        out["clasificador"] = -torch.sigmoid(((X - clf["mu"]) / clf["sd"]) @ clf["w"] + clf["b"])
    return out, X, ms_col


def fit_clf(X, fail):
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    w, b = torch.zeros(X.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(((X - mu) / sd) @ w + b, fail.float()) + 1e-2 * (w ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return {"mu": mu, "sd": sd, "w": w.detach(), "b": b.detach()}


def aurc(score, err):
    e = err[score.argsort(descending=True)]
    return float((e.cumsum(0) / torch.arange(1, e.numel() + 1)).mean())


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    regs, inter, sims = SETUP[name]
    dom = {"nombre": name, "regimenes": regs}
    if sims:
        dom["simuladores"] = sims
    g = Gemelo.desde_config({"nombre": f"abs_{name}", "dominio": dom, "colmena": {"n_agentes": 4},
                             "entrenamiento": {"epocas": epochs, "semilla": seed,
                                               "tasa": os.environ.get("EHO_TASA", "constante")}}).entrenar(verbose=False)
    d0 = g.dominio_variante(0)
    mix = lambda base, n: concat(*[d0.generate(1, 8, (regs + [inter])[i % 3], base + i) for i in range(n)])
    calib, test = mix(BASE + 100_000 * seed, 90), mix(BASE + 100_000 * seed + 50_000, 60)
    pr = g.predictor
    data = {}
    for k, ds in (("val", g.validacion), ("calib", calib), ("test", test)):
        P, Y, CE, ep = predict_with_inspection(g.agentes, ds, k=2, return_episodes=True)
        err = ((P.mean(0) - Y).abs() / g.escala).mean(-1)
        data[k] = dict(P=P, CE=CE, err=err, fail=err > TOL, ep=ep)
    _, Xv, _ = scores(data["val"]["P"], data["val"]["CE"], g.escala, pr)
    clf = fit_clf(Xv, data["val"]["fail"])
    S = {k: scores(d["P"], d["CE"], g.escala, pr, clf) for k, d in data.items()}
    one = torch.tensor([int(torch.nonzero(data["calib"]["ep"] == e)[0]) for e in data["calib"]["ep"].unique()])
    res = {"dominio": name, "semilla": seed, "riesgo": RISK, "delta": DELTA,
           "tasa_fallo_prueba": float(data["test"]["fail"].float().mean()),
           "ms_por_prediccion_colmena": S["test"][2], "metodos": {}}
    for m in S["test"][0]:
        sv, sc, st = S["val"][0][m], S["calib"][0][m], S["test"][0][m]
        thr, info = ltt_threshold(sc[one], data["calib"]["fail"][one], RISK, DELTA, grid=ltt_grid(sv, one.numel(), RISK, DELTA))
        acc = st >= thr
        res["metodos"][m] = {"auroc_fallo": auroc(-st[~data["test"]["fail"]], -st[data["test"]["fail"]]),
                             "aurc": aurc(st, data["test"]["err"]), "certificado": info["certificado"],
                             "cobertura_certificada": float(acc.float().mean()),
                             "riesgo_real": float(data["test"]["fail"][acc].float().mean()) if acc.any() else float("nan")}
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["metodos"][m].items()}, flush=True)
    json.dump(res, open(f"results/abstencion_{name}_s{seed}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
