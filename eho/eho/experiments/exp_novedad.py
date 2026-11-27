"""Afirmación 2 (v0.7): ¿qué detecta lo nuevo, y por qué?

Una colmena entrenada; tres conjuntos de prueba por dominio:
    conocido   mismos regímenes que el entrenamiento, episodios nuevos
    visible    el cambio se ve en las entradas (alta frecuencia; policitemia en sangre)
    oculto     el cambio está solo en los parámetros ocultos o en la física (régimen extremo;
               en sangre, la ley de Pries como verdad: mismas entradas, otra física)

Detectores: todos los de eho/baselines.py (entradas, rasgos, clave, plasmodial coseno y
resistencia, ensamble) más la inspección del sitio, que sí usa etiquetas del contexto.
AUROC por muestra y por episodio, con intervalo bootstrap del 95 %.

    PYTHONPATH=. python3 experiments/exp_novedad.py edp1d 0 [epocas]
    PYTHONPATH=. python3 experiments/exp_novedad.py burgers1d 0
    PYTHONPATH=. python3 experiments/exp_novedad.py sangre 0
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, ".")
from eho import Gemelo
from eho.dominios import concat
from eho.baselines import novelty_detectors, auroc_ic, episode_mean
from eho.active import inspection_errors
from eho.centinela import Centinela, elegir_sondeo

SETUP = {
    "edp1d": dict(dominio={"nombre": "edp1d", "regimenes": ["difusivo", "advectivo"]},
                  visible=dict(cambios={"k_min": 4}), oculto=dict(regimen="extremo")),
    "burgers1d": dict(dominio={"nombre": "burgers1d", "regimenes": ["viscoso", "poco_viscoso"]},
                      visible=dict(cambios={"k_min": 4}), oculto=dict(regimen="extremo")),
    "sangre": dict(dominio={"nombre": "sangre", "simuladores": [{"simulador": "quemada"}, {"simulador": "carreau"}],
                            "regimenes": ["sano", "dt2"]},
                   visible=dict(cambios={"H_range": (0.55, 0.70)}), oculto=dict(cambios={"simulador": "pries"})),
}


def build_sets(g, name, seed, n_ep=16, per=8):
    s = SETUP[name]
    regs = g.cfg.dominio.regimenes
    known_d = g.dominio_variante(0)
    vis_d = g.dominio_variante(0, **s["visible"].get("cambios", {}))
    hid_d = g.dominio_variante(0, **s["oculto"].get("cambios", {}))
    mk = lambda d, reg_fn, base: concat(*[d.generate(1, per, reg_fn(i), base + i) for i in range(n_ep)])
    known = mk(known_d, lambda i: regs[i % len(regs)], 40_000_000 + 1000 * seed)
    visible = mk(vis_d, lambda i: regs[i % len(regs)], 40_300_000 + 1000 * seed)
    hreg = s["oculto"].get("regimen")
    hidden = mk(hid_d, (lambda i: hreg) if hreg else (lambda i: regs[i % len(regs)]), 40_600_000 + 1000 * seed)
    return known, visible, hidden


TASA = os.environ.get("EHO_TASA", "constante")       # coseno: decaimiento de la tasa (v0.9)
ETQ = "cos" if TASA == "coseno" else ""


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    t0 = time.time()
    cfg = {"nombre": f"novedad_{name}", "dominio": SETUP[name]["dominio"], "colmena": {"n_agentes": 4},
           "entrenamiento": {"epocas": epochs, "semilla": seed, "tasa": TASA}}
    g = Gemelo.desde_config(cfg).entrenar(verbose=False)
    known, visible, hidden = build_sets(g, name, seed)
    ref = g.validacion
    res = {"dominio": name, "semilla": seed, "epocas": epochs, "detectores": {}}
    S = {k: novelty_detectors(g.agentes, ref, ds) for k, ds in
         [("conocido", known), ("visible", visible), ("oculto", hidden)]}
    # inspección del sitio (usa etiquetas): error medio de los agentes con k = 2 de contexto
    insp = {k: torch.stack([inspection_errors(a, ds, k=2)[:, 0] for a in g.agentes]).mean(0)
            for k, ds in [("conocido", known), ("visible", visible), ("oculto", hidden)]}
    for k in S:
        S[k]["inspeccion_con_etiquetas"] = insp[k]
    # v0.8: centinela (sorpresa normalizada ACh/NE + martingala conformal) y sondeo activo
    calib = concat(*[g.dominio_variante(0).generate(1, 8, regs_[i % len(regs_)], 41_000_000 + 1000 * seed + i)
                     for i, regs_ in [(i, g.cfg.dominio.regimenes) for i in range(60)]])
    cen = Centinela(g.agentes, g.escala, k=None).fit(calib, seed=seed)
    sets = {"conocido": known, "visible": visible, "oculto": hidden}
    alarmas = {}
    for k, ds in sets.items():
        ne, lm, eps_ = cen.episode_scores(ds, seed=seed)
        bcast = lambda v: torch.stack([v[eps_ == e][0] for e in ds["episode"]])
        S[k]["centinela_sorpresa"] = bcast(ne)
        S[k]["centinela_martingala"] = bcast(lm)
        alarmas[k] = float((lm >= torch.log(torch.tensor(1 / cen.alpha))).float().mean())
        for nombre, aleat in (("sondeo_aleatorio_3", True), ("sondeo_activo_3", False)):
            idx = elegir_sondeo(g.agentes, ds, 3, g.escala, seed=seed, aleatorio=aleat)
            ne3, _, _ = cen.episode_scores(ds, idx=idx, seed=seed)
            S[k][f"centinela_{nombre}"] = bcast(ne3)
    res["tasa_alarma_martingala"] = alarmas
    # v0.9.2: alarma por paciente con p-valor conformal entre pacientes; 64 conocidos extra para medir
    # bien la tasa de falsas alarmas
    extra = concat(*[g.dominio_variante(0).generate(1, 8, g.cfg.dominio.regimenes[i % len(g.cfg.dominio.regimenes)],
                                                     42_000_000 + 1000 * seed + i) for i in range(64)])
    res["alarma_por_paciente"] = {k: float(cen.patient_alarms(ds, seed=seed)[0].float().mean())
                                  for k, ds in {**sets, "conocido_64": extra}.items()}
    res["n_calibracion_pacientes"] = int(cen.cal_ep.numel())
    eps = {"conocido": known["episode"], "visible": visible["episode"], "oculto": hidden["episode"]}
    for det in S["conocido"]:
        row = {}
        for shift in ("visible", "oculto"):
            a, lo, hi = auroc_ic(S["conocido"][det], S[shift][det], n_boot=500, seed=seed)
            ek, _ = episode_mean(S["conocido"][det], eps["conocido"])
            es, _ = episode_mean(S[shift][det], eps[shift])
            ae, loe, hie = auroc_ic(ek, es, n_boot=500, seed=seed)
            row[shift] = {"auroc": a, "ic95": [lo, hi], "auroc_episodio": ae, "ic95_episodio": [loe, hie]}
        res["detectores"][det] = row
    res["segundos"] = round(time.time() - t0, 1)
    out = f"results/novedad_{name}{ETQ}_s{seed}.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"{name} s{seed} ({res['segundos']} s)")
    print(f"{'detector':28s} {'visible':>16s} {'oculto':>16s}")
    for det, r in res["detectores"].items():
        f = lambda x: f"{x['auroc']:.2f} [{x['ic95'][0]:.2f},{x['ic95'][1]:.2f}]"
        print(f"{det:28s} {f(r['visible']):>16s} {f(r['oculto']):>16s}")


if __name__ == "__main__":
    main()
