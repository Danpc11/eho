"""Experimentos C y D.

C: red microvascular con la viscosidad aprendida por el EHO (sano vs DT2).
D: inferencia inversa por flow matching usando el EHO como simulador rápido,
   evaluada sobre observaciones del simulador "verdadero" (sintético).
"""
import json, sys, math
import torch

sys.path.insert(0, ".")
from eho.data.synthetic import (make_dataset, concat, sample_theta, ground_truth,
                                make_condition_vector, THETA_NAMES)
from eho import default_device
from eho.model import EHO
from eho.train import fit
from eho.network import PhaseSeparation
from eho.inference import FlowMatchingPosterior
from tests.test_garantias import arbol_binario

# Condiciones de medición (H, D [μm], γ̇ [1/s]) que forman el vector observado x
OBS = [(0.45, 50.0, 1.0), (0.45, 50.0, 10.0), (0.45, 50.0, 200.0),
       (0.35, 10.0, 50.0), (0.45, 8.0, 50.0), (0.45, 25.0, 50.0)]


def eho_observables(model, theta, n_points=64):
    """x = [log η_app en las 6 condiciones, δ/R en 2 de ellas] usando el EHO."""
    xs = []
    B = theta.shape[0]
    with torch.no_grad():
        for H, D, g in OBS:
            Ht, Dt, gt = torch.full((B,), H), torch.full((B,), D), torch.full((B,), g)
            pts, _, _ = ground_truth(theta, Ht, Dt, gt, n_points)          # solo geometría
            b = {"points": pts, "cond": make_condition_vector(theta, Ht, Dt, gt), "gdot": gt}
            xs.append(model(b)["scalars"].cpu())
    x = torch.stack(xs, 1)                                                 # [B,6,2]
    return torch.cat([x[:, :, 0], x[:, 3:5, 1]], -1)


def true_observables(theta, noise=0.02):
    xs = []
    B = theta.shape[0]
    for H, D, g in OBS:
        _, _, s = ground_truth(theta, torch.full((B,), H), torch.full((B,), D), torch.full((B,), g), 8)
        xs.append(s)
    x = torch.stack(xs, 1)
    x = torch.cat([x[:, :, 0], x[:, 3:5, 1]], -1)
    return x + noise * torch.randn_like(x)


def mixed_prior(n):
    k = n // 3
    return torch.cat([sample_theta(k, "sano"), sample_theta(k, "prediabetes"), sample_theta(n - 2 * k, "dt2")])


if __name__ == "__main__":
    torch.manual_seed(0)
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    INFLATE = "--discrepancia" in sys.argv
    tr = concat(make_dataset(300, "sano", seed=1), make_dataset(300, "prediabetes", seed=2), make_dataset(300, "dt2", seed=3))
    va = concat(make_dataset(60, "sano", seed=4), make_dataset(60, "dt2", seed=5))
    model = EHO().to(default_device())
    fit(model, tr, va, epochs=epochs, verbose=False)
    model.eval()
    res = {}

    # ---------- C: red microvascular ----------
    def viscosity_from_eho(theta, gdot=100.0):
        def fn(D, H):
            E = D.shape[0]
            th = theta.expand(E, -1)
            g = torch.full((E,), gdot)
            pts, _, _ = ground_truth(th, H.float(), D.float(), g, 64)
            b = {"points": pts, "cond": make_condition_vector(th, H.float(), D.float(), g), "gdot": g}
            with torch.no_grad():
                return model(b)["scalars"][:, 0].exp().double().cpu()
        return fn

    net_res = {}
    for name, th in [("sano", torch.tensor([[0.2, 0.2, 0.2, 0.3]])), ("dt2", torch.tensor([[0.8, 0.8, 0.6, 0.6]]))]:
        net = arbol_binario(niveles=4)
        r = net.solve(viscosity_from_eho(th), PhaseSeparation())
        if not r["valida"]:                                     # v0.9.5: no usar soluciones físicamente inválidas
            raise RuntimeError(f"Solución microvascular inválida ({name}): {r['comprobaciones']}")
        Q = r["Q"].abs()
        terminal = Q[-16:]                                                  # 16 ramas terminales
        net_res[name] = {"flujo_total": float(Q[0]), "CV_perfusion_terminal": float(terminal.std() / terminal.mean()),
                         "H_terminal_min": float(r["H"][-16:].detach().min()), "H_terminal_max": float(r["H"][-16:].detach().max()),
                         "residuo_masa_max": float(r["mass_residual"].abs().max()),
                         "residuo_globulos_max": float(r["rbc_residual"].abs().max()),
                         "valida": r["valida"], "iteraciones": r["iteraciones"],
                         "residuo_constitutivo": r["residuo_constitutivo"], "H_maximo": r["H_maximo"]}
    net_res["caida_de_flujo_dt2_vs_sano_%"] = 100 * (1 - net_res["dt2"]["flujo_total"] / net_res["sano"]["flujo_total"])
    res["red"] = net_res
    print(json.dumps(net_res, indent=1), flush=True)

    # ---------- D: inferencia inversa ----------
    # Discrepancia del modelo: error del EHO frente al simulador verdadero en validación.
    th_cal = mixed_prior(500)
    disc = (true_observables(th_cal, noise=0.0) - eho_observables(model, th_cal))
    disc_sd = disc.std(0)
    noise_sd = torch.sqrt(0.02 ** 2 + (disc_sd ** 2 if INFLATE else 0.0))
    res.setdefault("inversa_meta", {})["discrepancia_sd"] = disc_sd.tolist()
    th_train = mixed_prior(6000)
    x_train = torch.cat([eho_observables(model, th_train[i:i + 500]) for i in range(0, 6000, 500)])
    if INFLATE:
        # sesgo medio de la discrepancia (corrección tipo Kennedy-O'Hagan, de orden cero)
        x_train = x_train + disc.mean(0)
        res["inversa_meta"]["discrepancia_media"] = disc.mean(0).tolist()
    x_train = x_train + noise_sd * torch.randn_like(x_train)
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    th_logit = torch.logit(th_train.clamp(1e-3, 1 - 1e-3))
    fm = FlowMatchingPosterior(4, x_train.shape[1])
    opt = torch.optim.Adam(fm.parameters(), lr=1e-3)
    for it in range(4000):
        idx = torch.randint(0, 6000, (256,))
        loss = fm.loss(th_logit[idx], (x_train[idx] - mu) / sd)
        opt.zero_grad(); loss.backward(); opt.step()
    th_test = mixed_prior(150)
    x_test = (true_observables(th_test) - mu) / sd                           # datos "reales"
    cover, err, width = torch.zeros(4), torch.zeros(4), torch.zeros(4)
    for i in range(150):
        s = torch.sigmoid(fm.sample(x_test[i], 300))
        lo, hi = s.quantile(0.05, 0), s.quantile(0.95, 0)
        cover += ((th_test[i] >= lo) & (th_test[i] <= hi)).float()
        err += (s.median(0).values - th_test[i]).abs()
        width += hi - lo
    prior_width = (mixed_prior(20000).quantile(0.95, 0) - mixed_prior(20000).quantile(0.05, 0))
    inv = {n: {"cobertura_90%": float(cover[k] / 150), "error_mediana": float(err[k] / 150),
               "ancho_intervalo": float(width[k] / 150), "ancho_previa": float(prior_width[k])}
           for k, n in enumerate(THETA_NAMES)}
    res["inversa_con_discrepancia" if INFLATE else "inversa"] = inv
    print(json.dumps(inv, indent=1), flush=True)
    json.dump(res, open("results/red_inversa%s.json" % ("_discrepancia" if INFLATE else ""), "w"),
              ensure_ascii=False, indent=1)
