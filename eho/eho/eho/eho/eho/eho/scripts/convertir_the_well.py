"""Convierte archivos HDF5 de The Well (p. ej. active_matter) al formato de episodios del EHO.

NO probado aún con un archivo real (no se descargó en el entorno donde se escribió). Primero
inspeccione un archivo; el script se detiene con un mensaje claro si la estructura no es la esperada.

    pip install h5py the_well
    # descargar una fracción (ver la documentación de the_well para el comando exacto de su versión)
    python scripts/convertir_the_well.py --inspeccionar datos/active_matter/train/<archivo>.hdf5
    python scripts/convertir_the_well.py datos/active_matter/train/*.hdf5 -o datos/well_am.pt --rejilla 32

Estructura esperada (formato de The Well): grupos "t0_fields" (campos escalares [traj, t, x, y]),
"t1_fields" (vectoriales [traj, t, x, y, 2]), opcional "t2_fields", y "scalars" o atributos con los
parámetros físicos. Cada ARCHIVO tiene un juego de parámetros (en active_matter, α y ζ): es un régimen.

Salida: un .pt por régimen o uno combinado (con --separar, uno por archivo), con
    points  [B, G², 2 + c]   (x, y, campos en t) en una rejilla gruesa G × G
    fields  [B, G², c]       campos en t + Δ
    scalars [B, 2]           log energía cinética media y log varianza de la concentración en t + Δ
    cond    [B, 1]           t normalizado
    episode [B]              grupo de muestras del mismo archivo (parámetros ocultos compartidos)
    params  [B, p]           parámetros físicos (solo para análisis; el modelo no los ve)
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def inspeccionar(ruta):
    import h5py
    with h5py.File(ruta, "r") as f:
        print("atributos raíz:", dict(f.attrs))
        f.visititems(lambda n, o: print(f"  {n}: {getattr(o, 'shape', '(grupo)')} {dict(o.attrs) if o.attrs else ''}"))


def _campos(f):
    """Lista de (nombre, dataset) con forma [traj, t, x, y, (c)]."""
    out = []
    for grupo in ("t0_fields", "t1_fields"):
        if grupo not in f:
            continue
        for nombre, ds in f[grupo].items():
            if ds.ndim >= 4:
                out.append((f"{grupo}/{nombre}", ds))
    if not out:
        sys.exit(f"sin t0_fields/t1_fields reconocibles; use --inspeccionar y adapte _campos()")
    return out


def _parametros(f):
    vals = []
    if "scalars" in f:
        for k, ds in f["scalars"].items():
            try:
                vals.append(float(ds[()]))
            except Exception:
                pass
    for k in f.attrs.get("simulation_parameters", []):
        k = k.decode() if isinstance(k, bytes) else k
        if k in f.attrs:
            vals.append(float(f.attrs[k]))
    return torch.tensor(vals or [0.0])


def convertir(ruta, rejilla, n_muestras, por_episodio, delta, gen):
    import h5py
    with h5py.File(ruta, "r") as f:
        campos = _campos(f)
        n_traj, n_t = campos[0][1].shape[:2]
        params = _parametros(f)
        traj = torch.randint(n_traj, (n_muestras,), generator=gen)
        t = torch.randint(0, n_t - delta, (n_muestras,), generator=gen)
        X, Y = [], []
        for i, ti in zip(traj.tolist(), t.tolist()):
            def tomar(tt):
                chans = []
                for _, ds in campos:
                    a = torch.as_tensor(ds[i, tt]).float()                  # [x, y] o [x, y, c]
                    a = a.unsqueeze(-1) if a.dim() == 2 else a.reshape(*a.shape[:2], -1)
                    chans.append(a)
                a = torch.cat(chans, -1).permute(2, 0, 1).unsqueeze(0)    # [1, c, x, y]
                return F.adaptive_avg_pool2d(a, rejilla)[0].permute(1, 2, 0).reshape(rejilla * rejilla, -1)
            X.append(tomar(ti))
            Y.append(tomar(ti + delta))
    X, Y = torch.stack(X), torch.stack(Y)
    g1 = torch.linspace(-1, 1, rejilla)
    xy = torch.stack(torch.meshgrid(g1, g1, indexing="ij"), -1).reshape(-1, 2).expand(X.shape[0], -1, -1)
    # convención: canal 0 = concentración (t0) y los siguientes 2 = velocidad (t1), si existen
    vel = Y[..., 1:3] if Y.shape[-1] >= 3 else Y
    ke = vel.pow(2).sum(-1).mean(-1).clamp(min=1e-10).log()
    cv = Y[..., 0].var(-1).clamp(min=1e-10).log()
    n_ep = n_muestras // por_episodio
    m = n_ep * por_episodio
    return {"points": torch.cat([xy, X], -1)[:m], "fields": Y[:m], "scalars": torch.stack([ke, cv], -1)[:m],
            "cond": ((t.float() / max(n_t - delta, 1) - 0.5) / 0.3).unsqueeze(-1)[:m],
            "episode": torch.arange(n_ep).repeat_interleave(por_episodio),
            "params": params.expand(m, -1).clone()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rutas", nargs="+")
    ap.add_argument("--inspeccionar", action="store_true")
    ap.add_argument("-o", "--salida", default="datos/the_well.pt")
    ap.add_argument("--rejilla", type=int, default=32)
    ap.add_argument("--muestras", type=int, default=160, help="muestras por archivo")
    ap.add_argument("--por-episodio", type=int, default=8)
    ap.add_argument("--delta", type=int, default=4, help="pasos de tiempo entre entrada y salida")
    ap.add_argument("--separar", action="store_true", help="un .pt por archivo (un régimen por archivo)")
    a = ap.parse_args()
    if a.inspeccionar:
        for r in a.rutas:
            inspeccionar(r)
        return
    gen = torch.Generator().manual_seed(0)
    partes, offset = [], 0
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    for r in a.rutas:
        d = convertir(r, a.rejilla, a.muestras, a.por_episodio, a.delta, gen)
        print(f"{Path(r).name}: {d['episode'].numel()} muestras, parámetros {d['params'][0].tolist()}", flush=True)
        if a.separar:
            torch.save(d, Path(a.salida).with_name(Path(r).stem + ".pt"))
        else:
            d["episode"] = d["episode"] + offset
            offset = int(d["episode"].max()) + 1
            partes.append(d)
    if partes:
        torch.save({k: torch.cat([p[k] for p in partes]) for k in partes[0]}, a.salida)
        print("guardado en", a.salida)


if __name__ == "__main__":
    main()
