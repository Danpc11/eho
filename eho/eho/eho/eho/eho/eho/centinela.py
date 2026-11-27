"""Centinela (v0.8): detectar que el modelo ya no sirve para ESTE paciente cuando las entradas se ven igual.

Si el cambio está solo en los parámetros ocultos, p(x) no cambia y ningún detector sin etiquetas puede
separarlo: una AUROC ≈ 0.5 es un resultado de identificabilidad, no un defecto. Los organismos lo
resuelven comparando lo que predicen con lo que pasa, y actuando para obtener la observación más útil.

1. Sorpresa normalizada (acetilcolina / noradrenalina; Yu y Dayan, Neuron 2005). La ACh señala la
   incertidumbre ESPERADA; la NE, la INESPERADA. Un error grande donde el modelo ya esperaba error no
   debe alarmar; el mismo error donde esperaba acertar, sí:
        z_i = mean_D |ȳ_i - y_i| / s  /  sqrt( mean_D (σ_agentes,i / s)² + s0² )
   ȳ_i es la predicción de la colmena dejando fuera a i (los demás ejemplos del episodio como contexto),
   σ_agentes,i la dispersión entre agentes (incertidumbre epistémica esperada) y s0 el piso aleatorio.
2. Dos procedimientos distintos, con garantías distintas (v0.9.3):
   a) ALARMA POR PACIENTE (`patient_alarms`): p-valor conformal de la sorpresa media del paciente frente
      a pacientes de calibración. Garantía marginal bajo intercambiabilidad de pacientes:
      P(alarma | paciente conocido) ≤ α. Es la única garantía que el código puede afirmar.
   b) Evidencia acumulada (`log_martingale`): S_n = Π_i mezcla_ε ε p_i^(ε-1) (martingala de potencia de
      Vovk et al.). AQUÍ ES SOLO UNA PUNTUACIÓN, SIN GARANTÍA: los p_i comparten el mismo conjunto de
      calibración fijo y las muestras de un paciente comparten parámetros ocultos, así que no son
      condicionalmente válidos y la desigualdad de Ville no aplica. No afirmar control de falsas alarmas
      "en cualquier momento" con ella (en sangre dio 0-12.5 % con α = 5 %).
3. Comparación con adaptación (quimiotaxis de E. coli; Alon et al., Nature 1999): `adaptativo` acumula
   log z frente a una media móvil de su propio nivel, de modo que responde a cambios y no al nivel basal.
4. Sondeo activo (peces eléctricos, murciélagos): `elegir_sondeo` escoge qué k muestras de un paciente
   conviene medir, las de mayor desacuerdo entre agentes (sin etiquetas), donde un θ distinto se notaría más.
"""
import math

import torch

from .dominios.base import subset

EPS_GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


@torch.no_grad()
def loo_predictions(agents, ds, k: int | None = 2, seed: int = 0, idx=None):
    """Predicción de cada muestra con k OTRAS del mismo episodio como contexto, para cada agente.
    idx: muestras permitidas como contexto y como objetivo (p. ej. las elegidas por sondeo).
    k=None (v0.9): contexto = TODAS las demás muestras permitidas del episodio, en un solo paso por
    agente (lectura episódica enmascarada); A pasos en vez de A·B.
    Devuelve P [A, B, D] (NaN fuera de idx) e Y [B, D]."""
    g = torch.Generator().manual_seed(seed)
    B, D = ds["scalars"].shape
    if k is None:
        sel = torch.arange(B) if idx is None else torch.as_tensor(idx).sort().values
        sub = subset(ds, sel)
        P = torch.full((len(agents), B, D), float("nan"))
        for a, m in enumerate(agents):
            m.eval()
            for i in range(0, sel.numel(), 256):               # los episodios son contiguos
                j = min(i + 256, sel.numel())
                while j < sel.numel() and sub["episode"][j] == sub["episode"][j - 1]:
                    j += 1
                if i >= j:
                    continue
                part = subset(sub, torch.arange(i, j))
                P[a, sel[i:j]] = m(part, episodic_self=m.memory is not None)["scalars"].cpu()
        return P, ds["scalars"].float()
    allowed = torch.zeros(B, dtype=torch.bool)
    allowed[idx if idx is not None else torch.arange(B)] = True
    P = torch.full((len(agents), B, D), float("nan"))
    for e in ds["episode"].unique():
        ids = torch.nonzero((ds["episode"] == e) & allowed).flatten()
        for i in ids:
            others = ids[ids != i]
            c = others[torch.randperm(others.numel(), generator=g)[:k]]
            one = subset(ds, i.view(1))
            ctx = subset(ds, c) if c.numel() else None
            for a, m in enumerate(agents):
                m.eval()
                P[a, i] = m(one, context=ctx if m.memory is not None else None)["scalars"].cpu()[0]
    return P, ds["scalars"].float()


def huella_agentes(agents, protocolo=None) -> str:
    """SHA-256 del estado COMPLETO de los agentes (v0.9.5): nombre, forma, tipo y contenido de cada
    parámetro y de cada buffer (memorias, abundancias), más el protocolo de uso. La versión anterior
    (número, suma y suma de cuadrados) no cambiaba al permutar pesos y no veía los buffers."""
    import hashlib
    h = hashlib.sha256()
    for i, a in enumerate(agents):
        for k, v in a.state_dict().items():
            t = v.detach().cpu().contiguous()
            h.update(f"{i}|{k}|{tuple(t.shape)}|{t.dtype}".encode())
            h.update(t.numpy().tobytes())
    h.update(repr(protocolo).encode())
    return h.hexdigest()


class Centinela:
    def __init__(self, agents, scale, k: int | None = None, alpha: float = 0.05, tau_adapt: float = 5.0):
        self.agents, self.scale, self.k, self.alpha, self.tau_adapt = agents, scale.float(), k, alpha, tau_adapt
        self.cal = None
        self.s0 = None

    def _z(self, P, Y):
        ok = ~torch.isnan(P[0, :, 0])
        mean, sd = P.nanmean(0), torch.nan_to_num(P, nan=0.0).std(0)
        e = ((mean - Y).abs() / self.scale).mean(-1)
        u2 = ((sd / self.scale) ** 2).mean(-1)
        z = e / torch.sqrt(u2 + self.s0 ** 2)
        return torch.where(ok, z, torch.full_like(z, float("nan")))

    def fit(self, calib, seed: int = 0, referencia=None, sondeo: dict | None = None):
        """Calibración con episodios del dominio conocido (distintos de los de entrenamiento).

        v0.9.4, tres condiciones para que el p-valor por paciente sea válido:
        1. La función de sorpresa se CONGELA antes de calibrar: s0 se estima con `referencia` (otros
           pacientes). Si no se da, se aparta un tercio de `calib` para s0 y se calibra con el resto.
           Antes s0 salía de los mismos pacientes de calibración, y sus puntuaciones no eran simétricas
           con las de un paciente nuevo.
        2. Mismo protocolo de medición en calibración y en uso: `sondeo` = {"n": 3, "aleatorio": False}
           fija qué muestras de cada paciente se usan; `patient_pvalues` aplica el mismo.
        3. La calibración queda ligada a la versión de los agentes (huella); si cambian, hay que recalibrar."""
        eps = calib["episode"].unique()
        if referencia is None:
            n_ref = max(1, eps.numel() // 3)
            ref_mask = torch.isin(calib["episode"], eps[:n_ref])
            referencia, calib = subset(calib, torch.nonzero(ref_mask).flatten()), subset(calib, torch.nonzero(~ref_mask).flatten())
        self.sondeo = dict(sondeo) if sondeo else None
        Pr, Yr = loo_predictions(self.agents, referencia, self.k, seed, self._idx(referencia, seed))
        e = ((Pr.nanmean(0) - Yr).abs() / self.scale).mean(-1)
        self.s0 = float(e[~torch.isnan(e)].median().clamp(min=1e-6))
        self.huella = huella_agentes(self.agents, self._protocolo())
        P, Y = loo_predictions(self.agents, calib, self.k, seed, self._idx(calib, seed))
        z = self._z(P, Y)
        self.cal = z[~torch.isnan(z)].sort().values
        # v0.9.2: calibración POR PACIENTE. Las muestras de un paciente comparten parámetros ocultos y
        # no son independientes, así que la martingala por muestra no garantiza α por paciente (en sangre
        # dio 0-12.5 % de falsas alarmas). Los pacientes sí son intercambiables entre sí: el estadístico
        # es la sorpresa media del paciente y su p-valor conformal se calcula frente a los pacientes de
        # calibración. P(p ≤ α | paciente conocido) ≤ α exactamente (hacen falta ≥ 1/α - 1 pacientes).
        medias = torch.stack([z[calib["episode"] == e].nanmean() for e in calib["episode"].unique()])
        # v0.9.7: un paciente de calibración sin ninguna medición válida no aporta nada: se excluye
        # (un NaN en la calibración corrompía el rango conformal)
        self.cal_ep = medias[~torch.isnan(medias)].sort().values
        self.n_calibracion_descartados = int(torch.isnan(medias).sum())
        return self

    def _idx(self, ds, seed=0):
        if not getattr(self, "sondeo", None):
            return None
        return elegir_sondeo(self.agents, ds, self.sondeo["n"], self.scale, seed=seed,
                             aleatorio=self.sondeo.get("aleatorio", False))

    def _protocolo(self):
        return {"k": self.k, "sondeo": getattr(self, "sondeo", None), "alpha": self.alpha}

    def vigente(self) -> bool:
        return getattr(self, "huella", None) == huella_agentes(self.agents, self._protocolo())

    def patient_pvalues(self, ds, seed: int = 0):
        """p-valor conformal por paciente (sorpresa media frente a los pacientes de calibración), con el
        MISMO protocolo de medición que se usó al calibrar. Falla si los agentes cambiaron."""
        if not self.vigente():
            raise RuntimeError("Centinela: los agentes cambiaron desde la calibración; vuelva a llamar a fit()")
        ne, _, eps = self.episode_scores(ds, self._idx(ds, seed), seed)
        n = self.cal_ep.numel()
        ge = n - torch.searchsorted(self.cal_ep, torch.nan_to_num(ne, nan=0.0), right=False)
        p = (ge.float() + 1.0) / (n + 1.0)
        return torch.where(torch.isnan(ne), torch.full_like(p, float("nan")), p), eps   # NaN = sin evidencia

    def patient_alarms(self, ds, seed: int = 0):
        p, eps = self.patient_pvalues(ds, seed)
        return p <= self.alpha, eps                      # NaN <= α es False: sin evidencia no hay alarma

    def patient_states(self, ds, seed: int = 0):
        """Tres estados por paciente (v0.9.7): "compatible" con lo conocido, "fuera_de_dominio" (alarma) o
        "evidencia_insuficiente" (ninguna medición válida). La falta de datos no es ni normalidad ni anomalía."""
        p, eps = self.patient_pvalues(ds, seed)
        return ["evidencia_insuficiente" if x != x else ("fuera_de_dominio" if x <= self.alpha else "compatible")
                for x in p.tolist()], eps

    @staticmethod
    def threshold_at_false_alarm(scores_known: torch.Tensor, alpha: float) -> float:
        """Umbral conformal de un detector cualquiera (mayor = más anómalo) para una tasa de falsas alarmas
        α sobre pacientes conocidos: el ⌈(n+1)(1-α)⌉-ésimo valor. Para comparar detectores (centinela,
        CUSUM, ...) hay que fijarles a TODOS la misma tasa de falsas alarmas y entonces medir potencia."""
        import math
        s = scores_known[~torch.isnan(scores_known)].sort().values
        k = math.ceil((s.numel() + 1) * (1 - alpha))
        return float("inf") if k > s.numel() else float(s[k - 1])

    def scores(self, ds, idx=None, seed: int = 0):
        """Sorpresa normalizada z por muestra (NaN para las que no están en idx)."""
        P, Y = loo_predictions(self.agents, ds, self.k, seed, idx)
        return self._z(P, Y)

    def pvalues(self, z):
        n = self.cal.numel()
        ge = n - torch.searchsorted(self.cal, z, right=False)
        return (ge.float() + 1.0) / (n + 1.0)

    def log_martingale(self, z):
        """log S_n (mezcla uniforme de martingalas de potencia) a lo largo de la secuencia z."""
        z = z[~torch.isnan(z)]
        if z.numel() == 0:
            return torch.zeros(0)
        p = self.pvalues(z).double()
        logs = torch.stack([torch.cumsum(math.log(e) + (e - 1) * p.log(), 0) for e in EPS_GRID])
        return (torch.logsumexp(logs, 0) - math.log(len(EPS_GRID))).float()

    def alarm(self, z):
        """Índice de la primera vez que la evidencia acumulada supera 1/α, o None.
        Heurístico: sin garantía de falsas alarmas (ver el encabezado). Para decidir, use patient_alarms."""
        lm = self.log_martingale(z)
        hit = torch.nonzero(lm >= math.log(1.0 / self.alpha)).flatten()
        return int(hit[0]) if hit.numel() else None

    def cusum(self, z, kappa: float = 0.5):
        """Línea base clásica (Page, 1954): CUSUM unilateral sobre log z estandarizado con la calibración.
        S_t = max(0, S_{t-1} + u_t - κ). Devuelve el máximo de S a lo largo de la secuencia.
        Dos condiciones para usarlo bien: (1) z debe venir en un orden con sentido físico (tiempo o
        posición); sobre un orden arbitrario no detecta un cambio de régimen, solo acumula; (2) su umbral
        debe fijarse con `threshold_at_false_alarm` a la misma α que los demás detectores."""
        z = z[~torch.isnan(z)].clamp(min=1e-6).log()
        lc = self.cal.clamp(min=1e-6).log()
        u = (z - lc.mean()) / lc.std().clamp(min=1e-6)
        s, best = 0.0, 0.0
        for v in u.tolist():
            s = max(0.0, s + v - kappa)
            best = max(best, s)
        return best

    def adaptativo(self, z):
        """Respuesta relativa log z_t - log m_t con m adaptándose a su propio nivel (retroalimentación integral)."""
        z = z[~torch.isnan(z)].clamp(min=1e-6)
        m, out = float(self.cal.median()), []
        for v in z.tolist():
            out.append(math.log(v) - math.log(m))
            m += (v - m) / self.tau_adapt
        return torch.tensor(out)

    def episode_scores(self, ds, idx=None, seed: int = 0):
        """Por episodio: sorpresa media (NE) y log de la e-valor conformal de sus muestras."""
        z = self.scores(ds, idx, seed)
        eps = ds["episode"].unique()
        ne, lm = [], []
        for e in eps:
            ze = z[ds["episode"] == e]
            ze = ze[~torch.isnan(ze)]
            ne.append(ze.mean() if ze.numel() else torch.tensor(float("nan")))
            l = self.log_martingale(ze)
            lm.append(l[-1] if l.numel() else torch.tensor(0.0))
        return torch.stack(ne), torch.stack(lm), eps


@torch.no_grad()
def elegir_sondeo(agents, ds, n: int, scale=None, seed: int = 0, aleatorio: bool = False):
    """Índices de las n muestras de cada episodio que conviene medir (sin usar etiquetas).
    Sondeo activo: las de mayor desacuerdo entre agentes. aleatorio=True: la línea base."""
    g = torch.Generator().manual_seed(seed)
    if not aleatorio:
        P = torch.stack([a.eval()(ds)["scalars"].cpu() for a in agents])
        v = (P / (scale if scale is not None else 1.0)).var(0).mean(-1)
    out = []
    for e in ds["episode"].unique():
        ids = torch.nonzero(ds["episode"] == e).flatten()
        if aleatorio:
            out.append(ids[torch.randperm(ids.numel(), generator=g)[:n]])
        else:
            out.append(ids[v[ids].topk(min(n, ids.numel())).indices])
    return torch.cat(out)
