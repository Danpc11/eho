"""Ejemplo mínimo: entrena el EHO con datos sintéticos, calibra la memoria y predice.

    PYTHONPATH=. python3 ejemplo_rapido.py
"""
import torch
from eho.data.synthetic import make_dataset, concat, subset
from eho import default_device
from eho.model import EHO
from eho.train import fit, evaluate, calibrate_memory

torch.manual_seed(0)
datos = concat(make_dataset(200, "sano", seed=1), make_dataset(200, "prediabetes", seed=2),
               make_dataset(200, "dt2", seed=3))
calib, entren = subset(datos, torch.arange(120)), subset(datos, torch.arange(120, 600))
val = concat(make_dataset(50, "sano", seed=4), make_dataset(50, "dt2", seed=5))
prueba = concat(make_dataset(150, "sano", seed=6), make_dataset(150, "dt2", seed=7))

modelo = EHO().to(default_device())
log = fit(modelo, entren, val, epochs=25)
print("calibración de memoria (mejora, (β, compuerta)):", calibrate_memory(modelo, calib, val))
print("prueba:", evaluate(modelo, prueba))
print("especies activas:", int(modelo.moe.active.sum()), "| abundancias:", modelo.moe.n[modelo.moe.active].tolist())
print("invasiones:", log.get("invasions"))

# Predicción para un paciente hipotético con sangre rígida y agregada, tubo de 8 μm
p = make_dataset(1, "dt2", seed=99)
out = modelo.eval()(p)
print("dispositivo:", modelo.device)
print("η_rel aparente predicha:", float(out["scalars"][0, 0].exp().detach()), "| verdadera:", float(p["scalars"][0, 0].exp()))
