"""Memoria Plasmodial de Conductancias (MPC), inspirada en Physarum polycephalum.

Física (Tero et al.; Kramar y Alim, PNAS 2021; Bhattacharyya et al., PRL 2022):
la memoria vive en la ESTRUCTURA de una red de flujo. Los conductos usados se
ensanchan y los no usados se encogen.

    nodos       prototipos con clave k_i y valor v_i
    aristas     k vecinos más cercanos, longitud ℓ_e = ||k_i - k_j||, conductancia D_e
    consulta    inyecta b_i = softmax_i(β k_i·q) - 1/n   (Σ b = 0)
    flujo       L(D) p = b  (Kirchhoff),   Q_e = (D_e/ℓ_e)(p_i - p_j)
    lectura     π_i ∝ |b_i| + ½ Σ_{e∋i} |Q_e|,   r(q) = Σ_i π_i v_i
    adaptación  dD_e/dt = 1[|Q_e| > θ] f(|Q_e|) - γ(z_e) D_e,   f(Q) = Q^μ/(1+Q^μ)
    umbral      solo cambian las aristas con señal grande (menos olvido; Chatterjee et al. 2025)
    bloqueo     z_e biestable con histéresis: consolida aristas usadas de forma sostenida
    habituación θ_i sigue el uso reciente del nodo; lo nuevo llega a nodos poco habituados
    crecimiento nodo nuevo si la novedad supera n*; D_min > 0 conserva la plasticidad
    deriva      las claves siguen a las consultas que reciben (k-medias en línea)

Novedad (v0.7). Hasta v0.6, novelty() era 1 - similitud coseno máxima con un nodo (+ habituación):
una distancia al vecino más cercano que no usaba ni el flujo ni las conductancias. Ahora hay dos modos:

    "coseno"       el de v0.6 (se conserva para la ablación y para decidir cuándo crear nodos)
    "resistencia"  resistencia efectiva de Kirchhoff entre la consulta y la "tierra" de la red.
                   La consulta se conecta a sus k nodos más cercanos con conductancia g_j = 1/ℓ_qj;
                   cada nodo se descarga a tierra con conductancia γ_i = g0 (h0 + hab_i), mayor en lo
                   habituado; los tubos internos tienen D_e/ℓ_e. Con A = L(D) + diag(γ) y C = A⁻¹[J,J]:

                       R(q) = 1 / ( Σ_j g_j  -  gᵀ C (I + diag(g) C)⁻¹ g ),    novedad = log R(q)

                   (complemento de Schur + Woodbury; A se invierte una vez por llamada, cada consulta
                   solo resuelve un sistema de k × k). Una consulta lejana (g pequeño) o que cae en una
                   región de tubos delgados y poco usados tiene R grande; una que cae sobre la red
                   consolidada, R pequeña. Es la cantidad física que el plasmodio "siente".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlasmodialMemory(nn.Module):
    def __init__(self, d_key: int, d_val: int, max_nodes: int = 512, k_nn: int = 8, beta: float = 8.0,
                 mu: float = 1.5, gamma: float = 0.05, gamma_lock: float = 0.002, dt: float = 0.5,
                 q_thresh: float = 1e-3, d_min: float = 0.05, d_on: float = 2.0, d_off: float = 0.5,
                 t_consolidate: int = 3, novelty_new: float = 0.05, key_lr: float = 0.05,
                 tau_hab: float = 20.0, novelty_mode: str = "coseno", ground: float = 10.0,
                 ground_floor: float = 0.1):
        super().__init__()
        self.cfg = dict(max_nodes=max_nodes, k_nn=k_nn, beta=beta, mu=mu, gamma=gamma, gamma_lock=gamma_lock,
                        dt=dt, q_thresh=q_thresh, d_min=d_min, d_on=d_on, d_off=d_off,
                        t_consolidate=t_consolidate, novelty_new=novelty_new, key_lr=key_lr, tau_hab=tau_hab,
                        novelty_mode=novelty_mode, ground=ground, ground_floor=ground_floor)
        self.register_buffer("keys", torch.zeros(0, d_key))
        self.register_buffer("vals", torch.zeros(0, d_val))
        self.register_buffer("pos", torch.zeros(0, d_key))        # v0.7: posición sin contexto (novedad)
        self.register_buffer("hab", torch.zeros(0))              # umbral de habituación por nodo
        self.register_buffer("edges", torch.zeros(0, 2, dtype=torch.long))
        self.register_buffer("D", torch.zeros(0))                 # conductancias
        self.register_buffer("z", torch.zeros(0, dtype=torch.bool))  # bloqueo (consolidación)
        self.register_buffer("above", torch.zeros(0, dtype=torch.long))  # contador para histéresis
        self.d_val = d_val

    def __len__(self):
        return self.keys.shape[0]

    @torch.no_grad()
    def reset(self):
        """Vacía la red (v0.7: se reconstruye al final de cada entrenamiento con el codificador final)."""
        dev, dk, dv = self.keys.device, self.keys.shape[1], self.vals.shape[1]
        self.keys, self.vals = torch.zeros(0, dk, device=dev), torch.zeros(0, dv, device=dev)
        self.pos, self.hab = torch.zeros(0, dk, device=dev), torch.zeros(0, device=dev)
        self.edges = torch.zeros(0, 2, dtype=torch.long, device=dev)
        self.D, self.z = torch.zeros(0, device=dev), torch.zeros(0, dtype=torch.bool, device=dev)
        self.above = torch.zeros(0, dtype=torch.long, device=dev)

    # ---------- topología ----------
    @torch.no_grad()
    def _rebuild_edges(self):
        n, c = len(self), self.cfg
        if n < 2:
            self.edges = torch.zeros(0, 2, dtype=torch.long, device=self.keys.device)
            self.D = torch.zeros(0, device=self.keys.device)
            self.z = torch.zeros(0, dtype=torch.bool, device=self.keys.device)
            self.above = torch.zeros(0, dtype=torch.long, device=self.keys.device)
            return
        old = {(int(a), int(b)): (float(d), bool(zz), int(ab))
               for (a, b), d, zz, ab in zip(self.edges.tolist(), self.D, self.z, self.above)}
        kn = F.normalize(self.keys, dim=-1)
        dist = torch.cdist(kn, kn)
        dist.fill_diagonal_(float("inf"))
        k = min(c["k_nn"], n - 1)
        nbr = dist.topk(k, largest=False).indices
        pairs = set()
        for i in range(n):
            for j in nbr[i].tolist():
                pairs.add((min(i, j), max(i, j)))
        pairs = sorted(pairs)
        dev = self.keys.device
        self.edges = torch.tensor(pairs, dtype=torch.long, device=dev)
        self.D = torch.tensor([old.get(p, (1.0, False, 0))[0] for p in pairs], device=dev)
        self.z = torch.tensor([old.get(p, (1.0, False, 0))[1] for p in pairs], dtype=torch.bool, device=dev)
        self.above = torch.tensor([old.get(p, (1.0, False, 0))[2] for p in pairs], dtype=torch.long, device=dev)

    def _lengths(self):
        kn = F.normalize(self.keys, dim=-1)
        return (kn[self.edges[:, 0]] - kn[self.edges[:, 1]]).norm(dim=-1).clamp(min=1e-3)

    # ---------- flujo ----------
    def _injection(self, q):
        """b [B, n]: fuente softmax de afinidad, sumidero uniforme; Σ b = 0."""
        a = torch.softmax(self.cfg["beta"] * F.normalize(q, dim=-1) @ F.normalize(self.keys, dim=-1).T, -1)
        return a - 1.0 / len(self), a

    def _solve(self, b):
        """Kirchhoff con conductancias D: devuelve presiones p [B,n] y flujos Q [B,E].

        Se resuelve en float64 con una regularización relativa (fija el nivel de presión
        y vuelve invertible el laplaciano aunque la red tenga componentes desconectadas).
        """
        n = len(self)
        G = (self.D / self._lengths()).double()
        i, j = self.edges[:, 0], self.edges[:, 1]
        L = torch.zeros(n, n, device=b.device, dtype=torch.float64)
        L.index_put_((i, j), -G, accumulate=True)
        L.index_put_((j, i), -G, accumulate=True)
        L.index_put_((i, i), G, accumulate=True)
        L.index_put_((j, j), G, accumulate=True)
        L = L + 1e-6 * (1.0 + float(G.max())) * torch.eye(n, device=b.device, dtype=torch.float64)
        p = torch.linalg.solve(L, b.double().T).T
        Q = G * (p[:, i] - p[:, j])
        return p.to(b.dtype), Q.to(b.dtype)

    def read(self, q: torch.Tensor):
        """q [B, d_key] -> lectura [B, d_val], pesos π [B, n], flujos Q [B, E]."""
        B = q.shape[0]
        if len(self) == 0:
            return q.new_zeros(B, self.d_val), None, None
        b, a = self._injection(q)
        if len(self) < 2 or self.edges.shape[0] == 0:
            return a @ self.vals, a, None
        _, Q = self._solve(b)
        node_flux = torch.zeros(B, len(self), device=q.device, dtype=q.dtype)
        node_flux.index_add_(1, self.edges[:, 0], Q.abs())
        node_flux.index_add_(1, self.edges[:, 1], Q.abs())
        pi = b.abs() + 0.5 * node_flux
        pi = pi / pi.sum(-1, keepdim=True).clamp(min=1e-12)
        return pi @ self.vals, pi, Q

    @torch.no_grad()
    def _pos(self):
        """Posición de cada nodo en el espacio de las consultas SIN contexto.

        Corrección de v0.7: los nodos se escriben con la clave indexada por episodio (entrada +
        descriptor del contexto), pero la novedad de un candidato se pregunta sin contexto (no tiene
        etiquetas). Comparar una cosa con la otra mezclaba dos espacios y hundía la AUROC (sangre 0.64
        frente a 0.98 del kNN en el mismo espacio). Ahora cada nodo guarda además su posición sin
        contexto, que se mueve, se fusiona y se crea igual que la clave."""
        pos = getattr(self, "pos", None)
        return pos if (pos is not None and pos.shape[0] == len(self)) else self.keys

    def cosine_novelty(self, q, space: str = "pos"):
        """Novedad de v0.6: falta de afinidad + nodos poco habituados (1 - máx. coseno + habituación).
        space="key" se usa para decidir si se crea un nodo; space="pos" para la novedad de un candidato."""
        if len(self) == 0:
            return torch.ones(q.shape[0], device=q.device)
        ref = self._pos() if space == "pos" else self.keys
        sim = F.normalize(q, dim=-1) @ F.normalize(ref, dim=-1).T
        a = torch.softmax(self.cfg["beta"] * sim, -1)
        unhab = (a * (1.0 - self.hab.clamp(0, 1))).sum(-1)
        return (1.0 - sim.max(-1).values) + 0.1 * unhab

    @torch.no_grad()
    def resistance_novelty(self, q):
        """log de la resistencia efectiva consulta → tierra (ver el encabezado del módulo)."""
        n, c = len(self), self.cfg
        if n < 2 or self.edges.shape[0] == 0:
            return self.cosine_novelty(q)
        dev = q.device
        kn = F.normalize(self._pos(), dim=-1).double()
        qn = F.normalize(q, dim=-1).double()
        # red con tierra: A = L(D) + diag(γ), definida positiva
        G = (self.D / self._lengths()).double()
        i, j = self.edges[:, 0], self.edges[:, 1]
        A = torch.zeros(n, n, device=dev, dtype=torch.float64)
        A.index_put_((i, j), -G, accumulate=True)
        A.index_put_((j, i), -G, accumulate=True)
        A.index_put_((i, i), G, accumulate=True)
        A.index_put_((j, j), G, accumulate=True)
        gam = c["ground"] * (c["ground_floor"] + self.hab.clamp(0, 1).double())
        A = A + torch.diag(gam)
        Ainv = torch.linalg.inv(A)
        # conexión de cada consulta con sus k nodos más cercanos
        k = min(c["k_nn"], n)
        dist = torch.cdist(qn, kn)                                        # [B, n]
        dJ, J = dist.topk(k, largest=False)                               # [B, k]
        g = 1.0 / dJ.clamp(min=1e-3)                                      # [B, k]
        C = Ainv[J.unsqueeze(-1), J.unsqueeze(-2)]                        # [B, k, k]
        I = torch.eye(k, device=dev, dtype=torch.float64).expand_as(C)
        y = torch.linalg.solve(I + g.unsqueeze(-1) * C, g.unsqueeze(-1))  # (I + diag(g) C) y = g
        val = (g.unsqueeze(-2) @ C @ y).squeeze(-1).squeeze(-1)           # gᵀ C y
        R = 1.0 / (g.sum(-1) - val).clamp(min=1e-12)
        return R.log().to(q.dtype)

    @torch.no_grad()
    def energy_novelty(self, q, beta: float | None = None):
        """Discrepancia tipo CA1 (v0.8): energía de Hopfield moderna de la consulta frente a los nodos,
            E(q) = -(1/β) log Σ_i exp(β cos(q, pos_i)) + (1/β) log n.
        CA1 compara lo que llega (corteza entorrinal) con lo que la memoria completa (CA3); una energía
        alta significa que ningún patrón guardado explica la entrada. A diferencia del vecino más cercano,
        suma la evidencia de TODOS los nodos parecidos (densidad), no solo la del más próximo.
        En aprendizaje automático: Zhang et al., ICLR 2023 (energía de Hopfield para detección OOD)."""
        if len(self) == 0:
            return torch.ones(q.shape[0], device=q.device)
        import math
        b = beta or self.cfg["beta"]
        sim = F.normalize(q, dim=-1) @ F.normalize(self._pos(), dim=-1).T
        return -(torch.logsumexp(b * sim, -1) - math.log(len(self))) / b

    @torch.no_grad()
    def novelty(self, q, mode: str | None = None):
        mode = mode or self.cfg.get("novelty_mode", "coseno")
        if mode == "resistencia":
            return self.resistance_novelty(q)
        if mode == "coseno":
            return self.cosine_novelty(q)
        if mode == "energia":
            return self.energy_novelty(q)
        raise ValueError(f"modo de novedad desconocido: {mode}")

    # ---------- escritura y adaptación ----------
    @torch.no_grad()
    def write(self, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor | None = None):
        """Graba un lote: crea nodos para lo nuevo, adapta conductancias, mueve claves, habitúa.
        pos: posición sin contexto de cada muestra (por defecto, la propia clave)."""
        k, v = k.detach(), v.detach()
        pos = k if pos is None else pos.detach()
        if not (torch.isfinite(k).all() and torch.isfinite(v).all()):
            return {"nuevos": 0, "omitido": True}
        c = self.cfg
        if len(self) == 0:
            self._add_nodes(k[:1], v[:1], pos[:1])
            k, v, pos = k[1:], v[1:], pos[1:]
            if k.shape[0] == 0:
                return {"nuevos": 1}
        nov = self.cosine_novelty(k, space="key")
        new = nov > c["novelty_new"]
        # dentro del mismo lote, no crear dos nodos casi idénticos
        if new.sum() > 1:
            kn = F.normalize(k[new], dim=-1)
            sim = kn @ kn.T
            keep = torch.ones(kn.shape[0], dtype=torch.bool, device=k.device)
            for a in range(kn.shape[0]):
                if keep[a]:
                    dup = (sim[a] > 1 - c["novelty_new"]) & (torch.arange(kn.shape[0], device=k.device) > a)
                    keep &= ~dup
            idx_new = torch.nonzero(new).flatten()
            new = torch.zeros_like(new)
            new[idx_new[keep]] = True
        n_new = 0
        if new.any():
            idx = torch.nonzero(new).flatten()[: max(1, c["max_nodes"] // 8)]
            room = c["max_nodes"] - len(self)
            if room < idx.numel():
                self._merge_closest(idx.numel() - room)          # llena: fusionar en vez de dejar de crecer
            self._add_nodes(k[idx], v[idx], pos[idx])
            n_new = int(idx.numel())
        old = ~new
        if old.any():
            ko, vo, po = k[old], v[old], pos[old]
            _, pi, Q = self.read(ko)
            # deriva de claves y valores hacia lo que recibe cada nodo
            w = pi / pi.sum(0, keepdim=True).clamp(min=1e-12)          # [B, n] normalizado por nodo
            mass = pi.sum(0)
            upd = (mass > 1e-3).float().unsqueeze(-1) * c["key_lr"]
            self.keys.mul_(1 - upd).add_(upd * (w.T @ ko))
            self.vals.mul_(1 - upd).add_(upd * (w.T @ vo))
            if self._pos() is not self.keys:
                self.pos.mul_(1 - upd).add_(upd * (w.T @ po))
            # habituación
            use = mass / mass.max().clamp(min=1e-12)
            self.hab.add_((use - self.hab) / c["tau_hab"])
            if Q is not None and torch.isfinite(Q).all():
                self._adapt(Q.abs().mean(0))
        return {"nuevos": n_new}

    @torch.no_grad()
    def _adapt(self, Qm):
        c = self.cfg
        Qn = Qm / Qm.max().clamp(min=1e-12)                             # flujo relativo
        f = Qn ** c["mu"] / (1 + Qn ** c["mu"])
        gate = (Qm > c["q_thresh"]).float()
        gam = torch.where(self.z, torch.full_like(self.D, c["gamma_lock"]), torch.full_like(self.D, c["gamma"]))
        self.D.add_(c["dt"] * (gate * f - gam * self.D)).clamp_(min=c["d_min"])
        # bloqueo biestable con histéresis y consolidación espaciada (cuenta escrituras sostenidas)
        self.above = torch.where(self.D > c["d_on"], self.above + 1, torch.zeros_like(self.above))
        self.z = (self.z | (self.above >= c["t_consolidate"])) & (self.D > c["d_off"])

    @torch.no_grad()
    def _merge_closest(self, n_merge: int):
        """Fusiona los pares de nodos más parecidos (sin bloqueo) para liberar n_merge lugares.
        Un mapa lleno que deja de crecer declara 'conocido' a todo (visto en v0.4)."""
        for _ in range(n_merge):
            kn = F.normalize(self.keys, dim=-1)
            sim = kn @ kn.T
            sim.fill_diagonal_(-2.0)
            i, j = divmod(int(sim.argmax()), sim.shape[1])
            self.keys[i] = 0.5 * (self.keys[i] + self.keys[j])
            self.vals[i] = 0.5 * (self.vals[i] + self.vals[j])
            has_pos = self._pos() is not self.keys
            if has_pos:
                self.pos[i] = 0.5 * (self.pos[i] + self.pos[j])
            self.hab[i] = torch.maximum(self.hab[i], self.hab[j])
            keep = torch.ones(len(self), dtype=torch.bool, device=self.keys.device)
            keep[j] = False
            self.keys, self.vals, self.hab = self.keys[keep], self.vals[keep], self.hab[keep]
            if has_pos:
                self.pos = self.pos[keep]
            # aristas: conservar las que no tocan j y reindexar
            e = self.edges
            ok = (e != j).all(-1)
            e = e[ok]
            e = e - (e > j).long()
            self.edges, self.D, self.z, self.above = e, self.D[ok], self.z[ok], self.above[ok]
        self._rebuild_edges()

    @torch.no_grad()
    def _add_nodes(self, k, v, p=None):
        if getattr(self, "pos", None) is not None and self.pos.shape[0] == len(self):
            self.pos = torch.cat([self.pos, k if p is None else p])
        self.keys = torch.cat([self.keys, k])
        self.vals = torch.cat([self.vals, v])
        self.hab = torch.cat([self.hab, torch.zeros(k.shape[0], device=k.device)])
        self._rebuild_edges()

    def stats(self):
        return {"nodos": len(self), "aristas": int(self.edges.shape[0]),
                "bloqueadas": int(self.z.sum()) if self.z.numel() else 0,
                "D_media": float(self.D.mean()) if self.D.numel() else 0.0}


class NoveltyMap(nn.Module):
    """Mapa de novedad: el "dónde he estado" de cada agente (v0.5).

    Es la parte de la memoria plasmodial que solo guarda posiciones (claves), sin resolver
    flujos, así que es barata y admite muchos más nodos. Se crea un nodo cuando lo visto está
    lejos de todo lo anterior (1 - similitud coseno máxima > umbral). Cuando se llena, se
    fusionan los dos nodos más parecidos en lugar de dejar de crecer: así un mapa lleno no
    declara "conocido" a todo, que es lo que pasó en v0.4 (memoria plasmodial a tope).
    """

    def __init__(self, d_key: int, max_nodes: int = 4096, thresh: float = 0.02):
        super().__init__()
        self.max_nodes, self.thresh = max_nodes, thresh
        self.register_buffer("keys", torch.zeros(0, d_key))
        self.register_buffer("counts", torch.zeros(0))

    def __len__(self):
        return self.keys.shape[0]

    @torch.no_grad()
    def reset(self):
        self.keys = torch.zeros(0, self.keys.shape[1], device=self.keys.device)
        self.counts = torch.zeros(0, device=self.keys.device)

    @torch.no_grad()
    def novelty(self, q):
        if len(self) == 0:
            return torch.ones(q.shape[0], device=q.device)
        sim = F.normalize(q, dim=-1) @ F.normalize(self.keys, dim=-1).T
        return 1.0 - sim.max(-1).values

    @torch.no_grad()
    def write(self, k):
        k = k.detach()
        if not torch.isfinite(k).all():
            return 0
        added = 0
        for chunk in k.split(256):
            nov = self.novelty(chunk)
            new = chunk[nov > self.thresh]
            if new.shape[0]:
                # deduplicar dentro del lote
                kn = F.normalize(new, dim=-1)
                keep = torch.ones(new.shape[0], dtype=torch.bool, device=k.device)
                sim = kn @ kn.T
                for a in range(new.shape[0]):
                    if keep[a]:
                        keep &= ~((sim[a] > 1 - self.thresh) & (torch.arange(new.shape[0], device=k.device) > a))
                new = new[keep]
                self.keys = torch.cat([self.keys, new])
                self.counts = torch.cat([self.counts, torch.ones(new.shape[0], device=k.device)])
                added += new.shape[0]
            while len(self) > self.max_nodes:
                self._merge_closest()
        return added

    @torch.no_grad()
    def _merge_closest(self):
        kn = F.normalize(self.keys, dim=-1)
        sim = kn @ kn.T
        sim.fill_diagonal_(-2.0)
        i, j = divmod(int(sim.argmax()), sim.shape[1])
        w = self.counts[i] / (self.counts[i] + self.counts[j])
        self.keys[i] = w * self.keys[i] + (1 - w) * self.keys[j]
        self.counts[i] += self.counts[j]
        keep = torch.ones(len(self), dtype=torch.bool, device=self.keys.device)
        keep[j] = False
        self.keys, self.counts = self.keys[keep], self.counts[keep]
