"""Lectura de simulaciones DPD reales (LAMMPS / Mirheo) al formato del EHO.

Formato estándar de una muestra (archivo .npz por simulación):
    points  [N, d_point]  posiciones normalizadas (+ atributos, p. ej. tipo de partícula)
    fields  [N, n_fields] velocidad axial normalizada, fracción local de glóbulos φ
    scalars [n_scalars]   log η_rel aparente, δ_CFL / R
    theta   [4]           parámetros mecánicos normalizados a [0, 1]
    H, D, gdot            hematocrito, diámetro [μm], tasa de corte [1/s]

`lammps_dump_to_sample` convierte un dump de LAMMPS (columnas id type x y z vx vy vz)
promediado en el tiempo. Ajuste los tipos de partícula a su script de entrada.
"""
import glob
import numpy as np
import torch

from .synthetic import make_condition_vector


def read_lammps_dump(path):
    """Devuelve lista de cuadros: dict(columnas -> np.ndarray)."""
    frames, cols, rows, reading = [], None, [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith("ITEM: ATOMS"):
                if rows:
                    frames.append(dict(zip(cols, np.array(rows, dtype=float).T)))
                cols, rows, reading = line.split()[2:], [], True
            elif line.startswith("ITEM:"):
                if reading and rows:
                    frames.append(dict(zip(cols, np.array(rows, dtype=float).T)))
                    rows = []
                reading = False
            elif reading:
                rows.append(line.split())
    if rows:
        frames.append(dict(zip(cols, np.array(rows, dtype=float).T)))
    return frames


def lammps_dump_to_sample(path, theta, H, D_um, gdot, rbc_types=(2,), axis="z",
                          n_points=512, n_bins=40, seed=0):
    """Promedia en el tiempo y en la dirección axial; muestrea n_points en la sección."""
    frames = read_lammps_dump(path)
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    perp = [c for i, c in enumerate("xyz") if i != ax]
    allx = np.concatenate([f[perp[0]] for f in frames])
    ally = np.concatenate([f[perp[1]] for f in frames])
    cx, cy = allx.mean(), ally.mean()
    R = np.sqrt((allx - cx) ** 2 + (ally - cy) ** 2).max()
    r = np.sqrt((allx - cx) ** 2 + (ally - cy) ** 2) / R
    v = np.concatenate([f["v" + "xyz"[ax]] for f in frames])
    is_rbc = np.isin(np.concatenate([f["type"] for f in frames]), rbc_types)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    u_prof = np.array([v[idx == k].mean() if np.any(idx == k) else 0.0 for k in range(n_bins)])
    phi_prof = np.array([is_rbc[idx == k].mean() if np.any(idx == k) else 0.0 for k in range(n_bins)])
    u_prof = u_prof / max(u_prof[0], 1e-12)
    rng = np.random.default_rng(seed)
    rr = np.sqrt(rng.random(n_points))
    ang = 2 * np.pi * rng.random(n_points)
    k = np.clip(np.digitize(rr, edges) - 1, 0, n_bins - 1)
    pts = np.stack([rr * np.cos(ang), rr * np.sin(ang), rr], -1)
    fields = np.stack([u_prof[k], phi_prof[k]], -1)
    # δ_CFL: primer radio (desde la pared) donde φ supera 5 % de su máximo
    thr = 0.05 * phi_prof.max()
    inside = np.where(phi_prof > thr)[0]
    delta_over_R = 1.0 - edges[inside.max() + 1] if inside.size else 0.0
    return {"points": pts, "fields": fields, "scalars": np.array([np.nan, delta_over_R]),
            "theta": np.asarray(theta, float), "H": H, "D": D_um, "gdot": gdot}


def save_sample(sample, path):
    np.savez_compressed(path, **sample)


def load_dataset(pattern):
    """Carga todos los .npz que coincidan y arma el diccionario de tensores del EHO.

    La viscosidad aparente (scalars[0]) debe calcularse desde el caudal y la caída
    de presión de la simulación; las muestras con NaN se excluyen.
    """
    files = sorted(glob.glob(pattern))
    S = [dict(np.load(f)) for f in files]
    S = [s for s in S if np.all(np.isfinite(s["scalars"]))]
    t = lambda k: torch.tensor(np.stack([s[k] for s in S]), dtype=torch.float32)
    theta, H, D, g = t("theta"), t("H"), t("D"), t("gdot")
    return {"points": t("points"), "fields": t("fields"), "scalars": t("scalars"), "theta": theta,
            "H": H, "D": D, "g": g, "gdot": g, "cond": make_condition_vector(theta, H, D, g)}
