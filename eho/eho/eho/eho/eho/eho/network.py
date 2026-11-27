"""Red microvascular con conservación exacta (sección 6).

- Kirchhoff: (B R⁻¹ Bᵀ) p = b, resuelto con un solver lineal diferenciable.
  La masa de plasma se conserva a precisión de máquina.
- Separación de fases en bifurcaciones divergentes: ley logística de Pries
  + corrección neuronal ψ_net (inicializada en cero); F_β = 1 - F_α,
  así el número de glóbulos rojos también se conserva exactamente.
- Acoplamiento hematocrito-resistencia por punto fijo.

Unidades: diámetros en μm, longitudes en μm, presiones arbitrarias.
"""
import math
import torch
import torch.nn as nn


def pries_in_vitro_relative_viscosity(D_um: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Efecto Fåhræus-Lindqvist, ajuste in vitro de Pries et al. (1992)."""
    eta45 = 220.0 * torch.exp(-1.3 * D_um) + 3.2 - 2.44 * torch.exp(-0.06 * D_um ** 0.645)
    sig = 1.0 / (1.0 + 1e-11 * D_um ** 12)
    C = (0.8 + torch.exp(-0.075 * D_um)) * (-1.0 + sig) + sig
    return 1.0 + (eta45 - 1.0) * ((1.0 - H) ** C - 1.0) / ((1.0 - 0.45) ** C - 1.0)


class PhaseSeparation(nn.Module):
    """Fracción de glóbulos que va a la rama α en una bifurcación divergente."""

    def __init__(self, d_theta: int = 0, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 + d_theta, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, FQ_a, D_a, D_b, D_f, H_f, theta=None):
        Df = D_f.clamp(min=1.0)
        X0 = (0.964 * (1 - H_f) / Df).clamp(max=0.45)
        A = -13.29 * ((D_a ** 2 / D_b ** 2 - 1) / (D_a ** 2 / D_b ** 2 + 1)) * (1 - H_f) / Df
        Bp = 1 + 6.98 * (1 - H_f) / Df
        x = ((FQ_a - X0) / (1 - 2 * X0)).clamp(1e-4, 1 - 1e-4)
        feats = torch.stack([FQ_a, D_a / Df, D_b / Df, H_f], -1)
        if theta is not None:
            feats = torch.cat([feats, theta.expand(feats.shape[0], -1)], -1)
        logit = A + Bp * torch.log(x / (1 - x)) + self.net(feats).squeeze(-1)
        F_a = torch.sigmoid(logit)
        # zonas de desnatado de plasma: sin glóbulos si FQ < X0, todos si FQ > 1 - X0
        F_a = torch.where(FQ_a <= X0, torch.zeros_like(F_a), F_a)
        F_a = torch.where(FQ_a >= 1 - X0, torch.ones_like(F_a), F_a)
        return F_a


class MicrovascularNetwork:
    """Grafo dirigido por construcción: edges[e] = (i, j)."""

    def __init__(self, n_nodes, edges, lengths_um, diams_um, p_boundary: dict, H_inlet: dict):
        # validaciones (v0.9.4)
        if any(float(d) <= 0 for d in diams_um) or any(float(l) <= 0 for l in lengths_um):
            raise ValueError("MicrovascularNetwork: diámetros y longitudes deben ser positivos")
        if any(not (0.0 <= float(h) <= 0.95) for h in H_inlet.values()):
            raise ValueError("MicrovascularNetwork: hematocrito de entrada fuera de [0, 0.95]")
        if len(p_boundary) < 2:
            raise ValueError("MicrovascularNetwork: se necesitan al menos dos nodos con presión impuesta")
        # v0.9.6: cada componente conexa necesita al menos un nodo con presión; si no, su presión queda
        # indeterminada y el sistema lineal es singular
        padre = list(range(n_nodes))

        def raiz(i):
            while padre[i] != i:
                padre[i] = padre[padre[i]]
                i = padre[i]
            return i
        for a, b in edges:
            padre[raiz(int(a))] = raiz(int(b))
        con_presion = {raiz(int(i)) for i in p_boundary}
        sueltos = sorted({raiz(i) for i in range(n_nodes)} - con_presion)
        if sueltos:
            raise ValueError(f"MicrovascularNetwork: {len(sueltos)} componente(s) de la red sin ningún nodo con presión impuesta")
        self.n_nodes = n_nodes
        self.edges = torch.as_tensor(edges, dtype=torch.long)
        self.L = torch.as_tensor(lengths_um, dtype=torch.float64)
        self.D = torch.as_tensor(diams_um, dtype=torch.float64)
        self.pb = p_boundary                      # {nodo: presión}
        self.H_in = H_inlet                       # {nodo de entrada: hematocrito}
        E = self.edges.shape[0]
        self.Bmat = torch.zeros(n_nodes, E, dtype=torch.float64)
        self.Bmat[self.edges[:, 0], torch.arange(E)] = 1.0
        self.Bmat[self.edges[:, 1], torch.arange(E)] = -1.0

    def solve_flow(self, eta_rel: torch.Tensor):
        """Q_e de Poiseuille con Kirchhoff exacto. eta_rel [E]."""
        G = math.pi * self.D ** 4 / (128.0 * eta_rel * self.L)          # conductancia
        K = self.Bmat @ torch.diag(G) @ self.Bmat.T                       # laplaciano ponderado
        n = self.n_nodes
        bnodes = torch.tensor(sorted(self.pb), dtype=torch.long)
        inner = torch.tensor([i for i in range(n) if i not in self.pb], dtype=torch.long)
        p = torch.zeros(n, dtype=torch.float64)
        p[bnodes] = torch.tensor([self.pb[i] for i in sorted(self.pb)], dtype=torch.float64)
        rhs = -K[inner][:, bnodes] @ p[bnodes]
        p_in = torch.linalg.solve(K[inner][:, inner], rhs)
        p = p.index_put((inner,), p_in)
        Q = G * (self.Bmat.T @ p)                                         # Q_e = G_e (p_i - p_j)
        return p, Q

    def apparent_shear(self, Q):
        """Tasa de cizalla aparente de pared en un tubo, γ̇ = 32 |Q| / (π D³).

        UNIDADES (v0.9.7). El solver trabaja en unidades internas: longitudes y diámetros en µm,
        viscosidad RELATIVA al plasma y presión en la unidad de `p_boundary`. Con ellas,
        Q_interno = π D⁴ ΔP / (128 η_rel L) [µm³ · unidad de presión]. El caudal físico es
        Q = Q_interno / μ_plasma: con ΔP en Pa y μ_plasma en Pa·s, Q queda en µm³/s y γ̇ en s⁻¹.
        Por eso `shear_scale` debe ser 1 / μ_plasma en esas unidades (≈ 833 para μ_plasma = 1.2 mPa·s).
        Con shear_scale = 1 la cizalla está en unidades internas y NO es comparable con s⁻¹."""
        return self.shear_scale * 32.0 * Q.abs() / (math.pi * self.D ** 3)

    shear_scale = 1.0

    def solve(self, viscosity_fn, phase_sep: PhaseSeparation, iters: int = 100, theta=None, tol: float = 1e-6,
              shear_coupled: bool = False, shear0: float = 100.0, shear_scale: float | None = None):
        """shear_coupled (v0.9.5): viscosity_fn(D, H, γ̇) recibe la cizalla LOCAL de cada vaso, calculada
        con el caudal de la iteración anterior; viscosidad, flujo, hematocrito y cizalla convergen juntos
        (sangre no newtoniana). Sin esto, viscosity_fn(D, H) usa la cizalla que el llamador fije."""
        if shear_scale is not None:
            self.shear_scale = float(shear_scale)
        if shear_coupled:
            base_fn, estado = viscosity_fn, {"Q": None}

            def viscosity_fn(D, H):
                g = torch.full_like(D, shear0) if estado["Q"] is None else self.apparent_shear(estado["Q"])
                return base_fn(D, H, g)
        """Punto fijo hematocrito ↔ resistencia. Devuelve dict con Q, H, p y residuos."""
        E = self.edges.shape[0]
        H = torch.full((E,), float(next(iter(self.H_in.values()))), dtype=torch.float64)
        converged, n_it, delta = False, 0, float("inf")
        Q_ant, eta_ant, r_Q, r_eta = None, None, float("inf"), float("inf")
        for n_it in range(1, iters + 1):
            eta = viscosity_fn(self.D, H)
            self._check_viscosity(eta)
            p, Q = self.solve_flow(eta)
            if shear_coupled:                      # amortiguado, como H: evita oscilar con adelgazamiento fuerte
                estado["Q"] = Q if estado["Q"] is None else 0.5 * estado["Q"] + 0.5 * Q
            H_new, rbc = self._propagate_hematocrit(Q, phase_sep, theta)
            delta = float(torch.max(torch.abs(H_new - H)).detach())
            # v0.9.7: con cizalla acoplada no basta con que H deje de cambiar (en un vaso sin
            # bifurcaciones H es constante y el solver "convergía" en 1 iteración con 10.8 % de error en
            # el caudal). Se exige también que caudal y viscosidad hayan dejado de cambiar.
            if Q_ant is not None:
                r_Q = float(((Q - Q_ant).abs().max() / (Q_ant.abs().max() + 1e-300)).detach())
                r_eta = float(((eta - eta_ant).abs().max() / (eta_ant.abs().max() + 1e-300)).detach())
            Q_ant, eta_ant = Q, eta
            if delta < tol and (not shear_coupled or (r_Q < tol and r_eta < tol)):
                H, converged = H_new, True
                break
            H = 0.5 * H + 0.5 * H_new
        H_prev = H
        eta_fin = viscosity_fn(self.D, H)
        self._check_viscosity(eta_fin)
        p, Q = self.solve_flow(eta_fin)
        # residuo de flujo: el caudal devuelto frente al de la última iteración (misma viscosidad si convergió)
        r_Q_fin = float(((Q - Q_ant).abs().max() / (Q_ant.abs().max() + 1e-300)).detach()) if Q_ant is not None else 0.0
        dP = p[self.edges[:, 0]] - p[self.edges[:, 1]]
        disipacion = float((Q * dP).sum().detach())
        H, rbc = self._propagate_hematocrit(Q, phase_sep, theta)
        # residuo constitutivo (v0.9.4): Q se calculó con el H de la iteración anterior; r_H mide cuánto
        # difiere del H devuelto. valida = convergió, sin aristas sin resolver y con ambos residuos pequeños
        r_H = float((H - H_prev).abs().max().detach())
        # v0.9.3: diagnósticos. identidad = max |Q_rbc - H Q| (debe ser ~0); sin_resolver = aristas a las
        # que no llegó hematocrito (nodo fuente sin H asignado o ciclo); convergio = punto fijo alcanzado
        return {"p": p, "Q": Q, "H": H, "rbc_flux": rbc,
                "mass_residual": self.node_residual(Q), "rbc_residual": self.node_residual(rbc),
                "convergio": converged, "iteraciones": n_it, "cambio_final_H": delta,
                "identidad_rbc": float((rbc - H * Q).abs().max().detach()),
                "residuo_constitutivo": r_H,
                "sin_resolver": int(self._unresolved), "H_max": self.H_MAX,
                "residuo_caudal": r_Q_fin, "disipacion": disipacion,
                "viscosidad_min": float(eta_fin.min().detach()),
                **self._validez(p, Q, H, rbc, converged, r_H, tol, r_Q_fin if shear_coupled else 0.0,
                                disipacion, eta_fin)}

    @staticmethod
    def _check_viscosity(eta):
        """v0.9.7: contrato físico de la ley constitutiva. Una viscosidad nula, negativa o no finita
        hace que el caudal vaya contra el gradiente de presión; antes se aceptaba y se marcaba válida."""
        if not bool(torch.isfinite(eta).all()) or float(eta.min().detach()) <= 0:
            raise ValueError("MicrovascularNetwork: la viscosidad debe ser finita y estrictamente positiva "
                             f"(mínimo recibido: {float(eta.min().detach()):.3g})")

    def _validez(self, p, Q, H, rbc, converged, r_H, tol, r_Q=0.0, disipacion=0.0, eta=None):
        """v0.9.5: una solución es válida solo si cumple TODO: convergió, nada quedó sin resolver,
        variables finitas, hematocrito en [0, H_MAX], conservación de volumen y de glóbulos en los
        nodos internos, identidad Q_rbc = H Q y consistencia constitutiva."""
        finito = bool(all(torch.isfinite(x).all() for x in (p, Q, H, rbc)))
        mx = lambda x: float(x.abs().max().detach()) if x.numel() else 0.0      # red sin nodos internos
        esc_q = float(Q.abs().max().detach()) + 1e-300
        esc_r = float(rbc.abs().max().detach()) + 1e-300
        checks = {
            "convergio": bool(converged), "resuelta": self._unresolved == 0, "finita": finito,
            "H_en_rango": finito and float(H.min().detach()) >= -1e-9 and float(H.max().detach()) <= self.H_MAX + 1e-6,
            "conserva_volumen": finito and mx(self.node_residual(Q)) <= 1e-6 * esc_q,
            "conserva_globulos": finito and mx(self.node_residual(rbc)) <= 1e-6 * esc_r,
            "identidad": finito and float((rbc - H * Q).abs().max().detach()) <= 1e-9 * esc_q,
            "constitutiva": r_H < 10 * tol,
            "caudal_consistente": r_Q < 10 * tol,                  # cizalla-viscosidad-caudal en su punto fijo
            "disipacion_no_negativa": disipacion >= -1e-9 * (abs(disipacion) + 1.0),   # Σ Q ΔP ≥ 0
            "viscosidad_positiva": eta is None or (bool(torch.isfinite(eta).all()) and float(eta.min().detach()) > 0),
        }
        return {"valida": all(checks.values()), "comprobaciones": checks,
                "H_min": float(H.min().detach()) if finito else float("nan"),
                "H_maximo": float(H.max().detach()) if finito else float("nan")}

    def node_residual(self, flux):
        """Σ entradas - Σ salidas en nodos internos (debe ser ~0)."""
        inner = [i for i in range(self.n_nodes) if i not in self.pb]
        return (self.Bmat @ flux)[inner]

    H_MAX = 0.95

    def _propagate_hematocrit(self, Q, phase_sep, theta):
        E = self.edges.shape[0]
        src = torch.where(Q >= 0, self.edges[:, 0], self.edges[:, 1])
        dst = torch.where(Q >= 0, self.edges[:, 1], self.edges[:, 0])
        q = Q.abs()
        rbc = torch.zeros(E, dtype=torch.float64)
        done = torch.zeros(E, dtype=torch.bool)
        self._unresolved_nodes = set()
        node_in = {i: [] for i in range(self.n_nodes)}
        node_out = {i: [] for i in range(self.n_nodes)}
        for e in range(E):
            node_out[int(src[e])].append(e)
            node_in[int(dst[e])].append(e)
        for _ in range(E + 1):
            progressed = False
            for node in range(self.n_nodes):
                outs = [e for e in node_out[node] if not done[e]]
                if not outs:
                    continue
                if node in self.H_in and not node_in[node]:
                    for e in outs:
                        rbc[e] = self.H_in[node] * q[e]
                        done[e] = True
                    progressed = True
                    continue
                if not all(done[e] for e in node_in[node]):
                    continue
                if not node_in[node]:                                    # fuente sin hematocrito asignado
                    for e in outs:
                        rbc[e] = 0.0
                        done[e] = True
                    self._unresolved_nodes.add(node)
                    progressed = True
                    continue
                inflow_rbc = sum(rbc[e] for e in node_in[node])
                inflow_q = sum(q[e] for e in node_in[node])
                if node in self.pb:
                    # v0.9.5: nodo con presión impuesta Y aristas de entrada y de salida. Intercambia
                    # sangre con el exterior: Q_ext = Σ Q_sal - Σ Q_ent. Antes se trataba como nodo
                    # interno, se repartían los glóbulos entrantes en un caudal distinto y el hematocrito
                    # podía salir de 4000 con valida = True.
                    #   Q_ext > 0 (entra sangre): trae el hematocrito de entrada del nodo, H_in[nodo];
                    #   Q_ext < 0 (sale sangre): extracción bien mezclada, las salidas conservan H.
                    out_q = sum(q[e] for e in outs)
                    ext = out_q - inflow_q
                    if float(out_q.detach()) < 1e-30:
                        total = 0.0 * inflow_rbc
                    elif float(ext.detach()) > 0:
                        if node in self.H_in:
                            total = inflow_rbc + self.H_in[node] * ext
                        else:                                            # entra sangre sin hematocrito definido
                            self._unresolved_nodes.add(node)
                            total = inflow_rbc * out_q / inflow_q.clamp(min=1e-30)
                    else:
                        total = inflow_rbc * out_q / inflow_q.clamp(min=1e-30)
                    for e in outs:
                        rbc[e] = total * q[e] / out_q.clamp(min=1e-30)
                        done[e] = True
                    progressed = True
                    continue
                H_f = inflow_rbc / inflow_q.clamp(min=1e-30)
                if len(outs) == 1:
                    rbc[outs[0]] = inflow_rbc
                elif len(outs) == 2:
                    a, b = outs
                    if float((q[a] + q[b]).detach()) < 1e-30:                       # dos ramas sin caudal: sin 0/0
                        rbc[a] = rbc[b] = 0.0
                        done[a] = done[b] = True
                        progressed = True
                        continue
                    FQ = q[a] / (q[a] + q[b])
                    D_f = torch.stack([self.D[e] for e in node_in[node]]).max()      # v0.9.5: sin float()
                    Fa = phase_sep(FQ.view(1).float(), self.D[a].view(1).float(), self.D[b].view(1).float(),
                                   D_f.view(1).float(), H_f.view(1).float(), theta).double().squeeze()
                    # v0.9.3: la fracción se restringe para que NINGUNA hija supere H_MAX. Antes se
                    # recortaba H después, y el flujo eritrocitario devuelto dejaba de cumplir
                    # Q_rbc = H Q. El intervalo es no vacío siempre que H de entrada ≤ H_MAX.
                    # (v0.9.4: los límites son tensores, sin .detach(): el gradiente respecto al caudal y
                    # al hematocrito se conserva cuando la restricción está activa)
                    if float(inflow_rbc.detach()) > 0:
                        hi = torch.clamp(self.H_MAX * q[a] / inflow_rbc, max=1.0)
                        lo = torch.clamp(1.0 - self.H_MAX * q[b] / inflow_rbc, min=0.0)
                        if float((hi - lo).detach()) >= 0:
                            Fa = torch.minimum(torch.maximum(Fa, lo), hi)
                    rbc[a] = Fa * inflow_rbc
                    rbc[b] = (1 - Fa) * inflow_rbc                       # conservación exacta
                else:                                                     # >2 salidas: reparto por flujo
                    qsum = sum(q[k] for k in outs)
                    for e in outs:
                        rbc[e] = inflow_rbc * q[e] / qsum if float(qsum.detach()) > 1e-30 else 0.0 * inflow_rbc
                for e in outs:
                    done[e] = True
                progressed = True
            if bool(done.all()) or not progressed:
                break
        self._unresolved = int((~done).sum()) + len(self._unresolved_nodes)
        H = torch.where(q > 1e-30, rbc / q.clamp(min=1e-30), torch.zeros_like(q)).clamp(min=0.0)
        # v0.9.4: se devuelve el flujo PROPAGADO (con el signo de la arista), no H·Q reconstruido; así
        # identidad_rbc = max |R_propagado - H Q| es una comprobación independiente y no una identidad
        return H, rbc * torch.sign(Q + (Q == 0).double())
