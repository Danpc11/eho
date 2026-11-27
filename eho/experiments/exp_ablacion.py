"""Experimento A: eficiencia de datos y ablaciones (datos sintéticos mezclados).

Compara EHO completo, sin memoria, sin ecología (MoE estático + pérdida de balanceo)
y una línea base MLP por punto, con 150 y 600 muestras de entrenamiento.
"""
import json, sys, time
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from eho.data.synthetic import make_dataset, concat
from eho import default_device
from eho.model import EHO, sample_losses
from eho.train import fit, evaluate, batches, calibrate_memory
from eho.data.synthetic import subset


class PointMLP(nn.Module):
    """Línea base: MLP por punto + MLP global (sin rebanado, Hopfield ni expertos)."""

    def __init__(self, d_cond=7, d=96):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3 + d_cond, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.s = nn.Sequential(nn.Linear(d_cond + 1, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, b, **kw):
        B, N, _ = b["points"].shape
        f = self.f(torch.cat([b["points"], b["cond"].unsqueeze(1).expand(B, N, -1)], -1))
        s = self.s(torch.cat([b["cond"], b["gdot"].log().unsqueeze(-1)], -1))
        return {"fields": f, "scalars": s}


def fit_mlp(m, tr, epochs, bs=32, lr=2e-3):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        m.train()
        for b in batches(tr, bs):
            o = m(b)
            loss = sample_losses(o["fields"], o["scalars"], b).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def data(n, seed):
    k = n // 3
    return concat(make_dataset(k, "sano", seed=seed), make_dataset(k, "prediabetes", seed=seed + 1),
                  make_dataset(n - 2 * k, "dt2", seed=seed + 2))


if __name__ == "__main__":
    seeds = [0, 1, 2]
    epochs_for = {150: 60, 600: 25}          # mismo número aproximado de pasos de gradiente
    test = data(600, 1000)
    val = data(150, 2000)
    results = []
    for n in [150, 600]:
      epochs = epochs_for[n]
      for seed in seeds:
        tr = data(n, 10 + n + 100 * seed)
        for name in ["EHO completo", "sin memoria", "sin ecologia", "MLP base"]:
            torch.manual_seed(seed)
            t0 = time.time()
            if name == "MLP base":
                m = PointMLP(); fit_mlp(m, tr, epochs); log = {}  # la línea base corre en CPU
                species = None
            else:
                m = EHO(use_memory=(name != "sin memoria"), use_ecology=(name != "sin ecologia")).to(default_device())
                # 20 % de los datos de entrenamiento se reservan para calibrar la memoria
                nc = n // 5
                fit_tr, calib = subset(tr, torch.arange(nc, n)), subset(tr, torch.arange(nc))
                if name == "sin memoria":
                    fit_tr = tr
                log = fit(m, fit_tr, val, epochs=epochs, verbose=False)
                log["calibracion"] = calibrate_memory(m, calib, val)
                species = int(m.moe.active.sum())
            ev = evaluate(m, test)
            ev.update({"modelo": name, "n_train": n, "semilla": seed, "segundos": round(time.time() - t0, 1),
                       "parametros": sum(p.numel() for p in m.parameters()), "especies": species,
                       "invasiones": log.get("invasions", []),
                       "calibracion": log.get("calibracion")})
            results.append(ev)
            print(json.dumps({k: v for k, v in ev.items() if k != "invasiones"}, ensure_ascii=False), flush=True)
    json.dump(results, open("results/ablacion.json", "w"), ensure_ascii=False, indent=1)
