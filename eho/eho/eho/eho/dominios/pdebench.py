"""Archivos 1D de PDEBench como dominio (v0.7): datos públicos generados por otros (Takamoto et al.,
NeurIPS 2022 Datasets and Benchmarks).

Cada archivo HDF5 de PDEBench 1D (p. ej. 1D_Advection_Sols_beta0.4.hdf5 o 1D_Burgers_Sols_Nu0.01.hdf5)
contiene muchas trayectorias con UN valor del parámetro físico. Aquí cada régimen es un archivo, y cada
episodio es un grupo de trayectorias de ese archivo: el parámetro queda oculto y el modelo tiene que
inferirlo con pocos ejemplos. Retener archivos enteros da pruebas de interpolación (β intermedio) y de
extrapolación con cambio oculto (β extremo).

Formato esperado (el de PDEBench): un conjunto "tensor" [n_traj, n_t, n_x] y, opcionalmente,
"x-coordinate" y "t-coordinate". Para revisar un archivo antes de usarlo:
    python -c "import h5py; f = h5py.File('ruta.hdf5'); f.visit(print); print(f['tensor'].shape)"

Configuración (configs/pdebench_burgers.yaml):
    dominio:
      nombre: pdebench1d
      params:
        archivos: {nu0.001: datos/1D_Burgers_Sols_Nu0.001.hdf5, nu0.01: datos/..., nu0.1: ...}
      regimenes: [nu0.01, nu0.1]
"""
import math

import torch

from .base import Dominio, registrar_dominio


@registrar_dominio("pdebench1d")
class PDEBench1D(Dominio):
    name = "pdebench1d"
    d_point = 3
    d_cond = 1
    n_fields = 1
    n_scalars = 2
    expert = "generico"

    def __init__(self, archivos: dict, n_grid: int = 64, t_index_range=(5, 100), max_traj: int | None = None,
                 traj_range=(0.0, 1.0)):
        """archivos: {nombre_del_régimen: ruta}. t_index_range: índices de tiempo de salida (T visible).
        traj_range: fracción de trayectorias de cada archivo que usa esta instancia. Use (0, 0.8) para
        entrenar y (0.8, 1) para la prueba, así ninguna trayectoria de prueba se vio al entrenar."""
        self.archivos = dict(archivos)
        self.regimes = list(self.archivos)
        self.n_grid, self.t_index_range, self.max_traj = n_grid, t_index_range, max_traj
        self.traj_range = tuple(traj_range)
        self._shape = {}

    def _open(self, regime):
        import h5py
        if regime not in self.archivos:
            raise KeyError(f"régimen '{regime}' sin archivo; disponibles: {self.regimes}")
        f = h5py.File(self.archivos[regime], "r")
        if "tensor" not in f:
            raise KeyError(f"{self.archivos[regime]}: no hay conjunto 'tensor' (claves: {list(f.keys())})")
        return f

    def generate(self, n_episodes, per_episode, regime, seed=0):
        g = torch.Generator().manual_seed(seed)
        with self._open(regime) as f:
            ten = f["tensor"]
            n_traj, n_t, n_x = ten.shape
            n_traj = min(n_traj, self.max_traj or n_traj)
            t_all = torch.as_tensor(f["t-coordinate"][:]).float() if "t-coordinate" in f else torch.arange(n_t).float()
            stride = max(1, n_x // self.n_grid)
            t0, t1 = self.t_index_range[0], min(self.t_index_range[1], n_t - 1)
            n = n_episodes * per_episode
            lo, hi = int(self.traj_range[0] * n_traj), int(self.traj_range[1] * n_traj)
            traj = lo + torch.randperm(hi - lo, generator=g)[:n]
            if traj.numel() < n:
                raise ValueError(f"{regime}: se piden {n} trayectorias y el rango tiene {hi - lo}")
            traj = traj.sort().values
            ti = torch.randint(t0, t1 + 1, (n,), generator=g)
            u0, uT = [], []
            for i, t in zip(traj.tolist(), ti.tolist()):
                u0.append(torch.as_tensor(ten[i, 0, ::stride]).float())
                uT.append(torch.as_tensor(ten[i, t, ::stride]).float())
        u0, uT = torch.stack(u0)[:, : self.n_grid], torch.stack(uT)[:, : self.n_grid]
        T = t_all[ti.clamp(max=t_all.numel() - 1)]
        N = u0.shape[1]
        x = torch.arange(N) / N
        pts = torch.stack([torch.cos(2 * math.pi * x).expand(n, -1), torch.sin(2 * math.pi * x).expand(n, -1), u0], -1)
        e0, eT = u0.pow(2).mean(-1).sqrt(), uT.pow(2).mean(-1).sqrt()
        grad = (torch.roll(uT, -1, -1) - uT).abs().max(-1).values * N
        scalars = torch.stack([(eT / e0.clamp(min=1e-6)).clamp(min=1e-6).log(), grad.clamp(min=1e-3).log()], -1)
        tmax = float(t_all.max()) if t_all.numel() else 1.0
        cond = ((T / tmax - 0.5) / 0.3).unsqueeze(-1)
        return {"points": pts, "cond": cond, "fields": uT.unsqueeze(-1), "scalars": scalars,
                "episode": torch.arange(n_episodes).repeat_interleave(per_episode), "T": T}
