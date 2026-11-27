"""Colmena: decisiones colectivas entre varios modelos, como un enjambre de abejas.

Modelo de selección de nido (Seeley et al., Science 2012; Pais et al., PLoS ONE 2013;
Reina et al., best-of-N). Ψ_i = fracción de exploradoras comprometidas con la opción i,
Ψ_U = 1 - Σ Ψ_j las no comprometidas:

    dΨ_i/dt = γ_i Ψ_U  -  α_i Ψ_i  +  ρ_i Ψ_U Ψ_i  -  σ Ψ_i Σ_{j≠i} Ψ_j
              descubrimiento  abandono  reclutamiento   inhibición cruzada

con γ_i, ρ_i ∝ valor v_i y α_i ∝ 1 - v_i. Se decide cuando una opción alcanza el
quórum q (no hace falta consenso). Si ninguna lo alcanza en el tiempo máximo, la
colmena se ABSTIENE. La regla es sensible al valor: si todas las opciones son malas,
no se compromete.

Aquí cada "exploradora" pertenece a un agente (un EHO). Cada agente evalúa las opciones
por su cuenta; las exploradoras de un agente descubren y reclutan según la valoración de
SU agente. Así la decisión surge de evaluaciones independientes, no de copiar.
"""
from dataclasses import dataclass
import torch


@dataclass
class Decision:
    choice: int | None            # índice de la opción elegida, o None si se abstiene
    support: torch.Tensor          # Ψ final por opción
    time: float                    # tiempo hasta el quórum (o el máximo)
    trajectory: torch.Tensor       # Ψ en el tiempo [T, O]


class BeeColony:
    def __init__(self, quorum: float = 0.5, gamma: float = 0.5, rho: float = 2.0, alpha: float = 0.5,
                 sigma: float = 2.0, dt: float = 0.02, t_max: float = 30.0, min_value: float = 0.0,
                 noise: float = 0.02, seed: int = 0, gamma_min: float = 0.0):
        """noise: fluctuación de las exploradoras. Rompe empates entre opciones igual de BUENAS
        (con inhibición cruzada fuerte el empate simétrico es inestable), pero no basta para
        comprometerse con opciones malas: la abstención por valor bajo se conserva."""
        self.quorum, self.gamma, self.rho, self.alpha, self.sigma = quorum, gamma, rho, alpha, sigma
        self.dt, self.t_max, self.min_value, self.noise = dt, t_max, min_value, noise
        # v0.8: descubrimiento espontáneo mínimo (Seeley et al. 2012): las exploradoras nunca dejan de
        # descubrir del todo, así la colmena no se queda quieta cuando todo vale poco (arranque en frío)
        self.gamma_min = gamma_min
        self.seed = seed

    def decide(self, values: torch.Tensor, weights: torch.Tensor | None = None) -> Decision:
        """values [A, O] en [0, 1]: valoración de cada agente para cada opción.

        weights [A] (opcional): fracción de exploradoras de cada agente (p. ej. su fiabilidad).
        Dinámica de campo medio por agente: Ψ_{a,i} con tasas que dependen de v_{a,i};
        la inhibición cruzada y el quórum se calculan sobre la colmena completa.
        """
        # v0.9.4: generador nuevo en cada llamada. Con uno persistente, el apoyo de una consulta dependía
        # de cuántas consultas se habían hecho antes y en qué orden.
        gen = torch.Generator().manual_seed(self.seed)
        V = values.clamp(0, 1).double()
        A, O = V.shape
        w = (torch.full((A,), 1.0 / A) if weights is None else weights / weights.sum()).double()
        V = torch.where(V < self.min_value, torch.zeros_like(V), V)
        Psi = torch.zeros(A, O, dtype=torch.float64)                   # comprometidas, por agente
        traj, t = [], 0.0
        while t < self.t_max:
            tot = (w.unsqueeze(-1) * Psi).sum(0)                          # Ψ_i de la colmena
            U = 1.0 - Psi.sum(-1, keepdim=True)                           # no comprometidas del agente
            others = tot.sum() - tot                                      # Σ_{j≠i} Ψ_j
            dPsi = (torch.clamp(self.gamma * V, min=self.gamma_min) * U - self.alpha * (1 - V) * Psi
                    + self.rho * V * U * tot.unsqueeze(0) - self.sigma * Psi * others.unsqueeze(0))
            eta = self.noise * (self.dt ** 0.5) * torch.randn(Psi.shape, generator=gen, dtype=torch.float64)
            Psi = (Psi + self.dt * dPsi + eta * Psi.clamp(min=1e-3).sqrt()).clamp(min=0.0)
            Psi = Psi / Psi.sum(-1, keepdim=True).clamp(min=1.0)          # Σ_i Ψ_{a,i} ≤ 1
            tot = (w.unsqueeze(-1) * Psi).sum(0)
            traj.append(tot.clone())
            t += self.dt
            if tot.max() >= self.quorum:
                return Decision(int(tot.argmax()), tot, t, torch.stack(traj))
            # v0.9: si la deriva determinista ya es nula y el apoyo está lejos del quórum, seguir
            # integrando solo gasta tiempo (antes: 1,500 pasos por cada muestra en abstención).
            # Criterio: el apoyo no cambió más de 0.01 en 2 unidades de tiempo y ni siquiera la suma de
            # todos los apoyos alcanzaría el quórum (un empate entre opciones buenas NO se corta: el ruido lo rompe)
            lag = int(round(2.0 / self.dt))
            if len(traj) > 2 * lag and len(traj) % lag == 0 and float(tot.sum()) < 0.8 * self.quorum \
                    and float((tot - traj[-lag]).abs().max()) < 0.01:
                break
        return Decision(None, tot, t, torch.stack(traj))


def cluster_options(preds: torch.Tensor, scale):
    """Agrupa predicciones compatibles (distancia escalada < 1) en opciones distintas.

    preds [A, D] -> centros [O, D]. Agrupamiento voraz en orden; suficiente para pocos agentes.
    """
    centers, members = [], []
    for a in range(preds.shape[0]):
        for o, c in enumerate(centers):
            if (((preds[a] - c) / scale) ** 2).sum().sqrt() < 1.0:
                members[o].append(a)
                centers[o] = preds[members[o]].mean(0)
                break
        else:
            centers.append(preds[a].clone())
            members.append([a])
    return torch.stack(centers), members


def agreement_values(preds: torch.Tensor, centers: torch.Tensor, scale, conf: torch.Tensor | None = None):
    """v_{a,o} = exp(-||p_a - c_o||² / (2 s²)) · conf_a : cuánto apoya el agente a la opción o. [A, O]."""
    d2 = (((preds.unsqueeze(1) - centers.unsqueeze(0)) / scale) ** 2).sum(-1)
    v = torch.exp(-0.5 * d2)
    if conf is not None:
        v = v * conf.clamp(0, 1).unsqueeze(-1)
    return v


def colony_predict(preds: torch.Tensor, scale, colony: BeeColony | None = None, conf=None, mode: str = "ganador"):
    """Predicción colectiva por muestra. preds [A, B, D]; scale: tolerancia por dimensión.

    Las predicciones compatibles se agrupan en opciones; cada agente apoya las opciones
    cercanas a su propia predicción. Si una opción alcanza el quórum, la predicción es su
    centro; si no, la muestra queda en abstención (se devuelve la media solo como referencia).
    mode: "ganador" (centro de la opción con más apoyo), "media" (media de los agentes) o
    "ponderada" (media ponderada por la confianza de inspección).
    Devuelve (predicción [B, D], aceptada [B] bool, apoyo ganador [B]).
    """
    colony = colony or BeeColony()
    A, B, D = preds.shape
    out = preds.mean(0).clone()
    accepted = torch.zeros(B, dtype=torch.bool)
    support = torch.zeros(B)
    for b in range(B):
        centers, _ = cluster_options(preds[:, b], scale)
        c = None if conf is None else conf[:, b]
        dec = colony.decide(agreement_values(preds[:, b], centers, scale, c))
        winner = int(dec.support.argmax())
        support[b] = float(dec.support[winner])
        accepted[b] = dec.choice is not None
        # v0.9.4: el ganador de la dinámica y el cumplimiento del quórum son cosas distintas. Hasta
        # v0.9.3 la predicción solo cambiaba si había quórum; el predictor calibrado usa quórum 1.01
        # (para que la dinámica corra completa), así que devolvía SIEMPRE la media de los agentes.
        if mode == "ganador":
            out[b] = centers[winner]
        elif mode == "ponderada" and c is not None and float(c.sum()) > 0:
            w = c.clamp(0, 1) / c.clamp(0, 1).sum()
            out[b] = (w.unsqueeze(-1) * preds[:, b]).sum(0)
        elif mode not in ("media", "ganador", "ponderada"):
            raise ValueError(f"modo de predicción desconocido: {mode}")
    return out, accepted, support


def urgency_sigma(used: float, total: float, sigma0: float = 1.0, sigma_max: float = 6.0) -> float:
    """Inhibición cruzada que crece con el presupuesto gastado (v0.8; Pais et al., PLoS ONE 2013:
    una σ mayor rompe antes el empate entre opciones casi iguales). Al principio coexisten las opciones
    parecidas (se evita comprometerse con la primera); al final se fuerza la decisión (se evita trabarse)."""
    f = min(max(used / max(total, 1e-9), 0.0), 1.0)
    return sigma0 + (sigma_max - sigma0) * f


def colony_select(scores: torch.Tensor, colony: BeeColony | None = None, top: int = 3, normalize: bool = True):
    """Elegir el siguiente experimento. scores [A, C] (mayor = mejor) de cada agente.

    normalize=True: se reescalan a [0, 1] con la escala conjunta (valor relativo).
    normalize=False: los valores ya están en una escala absoluta [0, 1] (p. ej. InformationValue);
    así la colmena puede abstenerse cuando ningún candidato vale la pena.
    Se corre la colmena sobre las `top` candidatas con mayor valoración media (por defecto 3).
    Con 6 u 8 opciones casi igual de buenas, la inhibición cruzada suprime a todas y la colmena
    se traba (Ψ máx ≈ 0.1; comprobado). Las abejas lo evitan porque las exploradoras descubren
    pocos sitios a la vez; aquí, la colmena decide entre las pocas mejores de cada turno.
    Devuelve (índice de candidata o None, decisión).
    """
    colony = colony or BeeColony(sigma=4.0, t_max=60.0, noise=0.05)
    if normalize:
        lo, hi = scores.min(), scores.max()
        V = (scores - lo) / (hi - lo).clamp(min=1e-12)
    else:
        V = scores.clamp(0, 1)
    idx = V.mean(0).topk(min(top, V.shape[1])).indices
    dec = colony.decide(V[:, idx])
    return (int(idx[dec.choice]) if dec.choice is not None else None), dec


class ColonyPredictor:
    """Colmena con quórum CALIBRADO.

    La dinámica de abeja corre hasta el tiempo máximo y devuelve el apoyo final de la opción
    ganadora (Ψ máx). En validación se elige el umbral de apoyo que logra una cobertura
    objetivo (o un error máximo en lo aceptado). Así el quórum deja de ser un número fijo
    y se ajusta a lo bien que se conocen los agentes entre sí.

    La confianza de cada agente viene de "inspeccionar el sitio": su error al predecir los
    ejemplos etiquetados del contexto (conf = exp(-error / (2 τ)), τ = mediana en validación).
    """

    def __init__(self, scale, colony: BeeColony | None = None, mode: str = "media"):
        self.mode = mode
        self.scale = scale
        self.colony = colony or BeeColony(quorum=1.01)        # nunca se detiene antes: apoyo final
        self.tau = None
        self.threshold = 0.0

    def confidence(self, ctx_err):
        return torch.exp(-ctx_err / (2 * self.tau))

    def _run(self, preds, ctx_err):
        conf = None if (ctx_err is None or self.tau is None) else self.confidence(ctx_err)
        return colony_predict(preds, self.scale, self.colony, conf=conf, mode=getattr(self, "mode", "media"))

    def fit(self, preds, y, ctx_err=None, target_coverage: float | None = 0.8, max_error: float | None = None,
            method: str = "cobertura", risk: float = 0.1, delta: float = 0.1, tolerance: float = 1.0,
            calib=None):
        """preds [A,B,D], y [B,D], ctx_err [A,B]. Elige τ y el umbral de apoyo.

        method="cobertura": umbral = cuantil del apoyo para la cobertura objetivo (v0.3-v0.6).
        method="ltt": Learn-then-Test (Angelopoulos, Bates et al. 2021). Garantía de muestra finita:
            con probabilidad ≥ 1 - δ sobre el conjunto de calibración,
                P( error escalado > tolerancia  |  la colmena acepta ) ≤ risk.
            Para que la garantía valga, τ y la escala deben estimarse con datos DISTINTOS de los que se
            usan para elegir el umbral: pase calib=(preds_c, y_c, ctx_err_c) con otros episodios.
        """
        if ctx_err is not None:
            self.tau = ctx_err.median(1).values.clamp(min=1e-6).unsqueeze(-1)
        if method == "ltt":
            if calib is None:                       # v0.9.3: la calibración independiente es obligatoria
                raise ValueError("LTT requiere calib=(preds, y, ctx_err) con episodios independientes de "
                                 "los usados para la escala y τ; sin ellos la garantía no es válida")
            # rejilla de umbrales fijada ANTES de ver la calibración, con el apoyo del primer conjunto
            _, _, sup0 = self._run(preds, ctx_err)
            preds, y, ctx_err = calib
            pred, _, support = self._run(preds, ctx_err)
            err = ((pred - y).abs() / self.scale).mean(-1)
            grid = ltt_grid(sup0, support.numel(), risk, delta)
            self.threshold, self.ltt_info = ltt_threshold(support, err > tolerance, risk, delta, grid=grid)
            self.method, self.risk, self.delta, self.tolerance = "ltt", risk, delta, tolerance
            return self
        self.method = "cobertura"
        pred, _, support = self._run(preds, ctx_err)
        err = ((pred - y).abs() / self.scale).mean(-1)
        if max_error is not None:
            # el umbral más bajo cuyo error medio en lo aceptado no supera max_error
            order = support.argsort(descending=True)
            cum = err[order].cumsum(0) / torch.arange(1, err.numel() + 1)
            ok = torch.nonzero(cum <= max_error).flatten()
            self.threshold = float(support[order][ok.max()]) if ok.numel() else float(support.max()) + 1e-6
        else:
            self.threshold = float(support.quantile(1 - target_coverage))
        return self

    def predict(self, preds, ctx_err=None):
        pred, _, support = self._run(preds, ctx_err)
        if getattr(self, "method", "") == "ltt" and not self.ltt_info.get("certificado", False):
            # sin umbral certificado la abstención es un ESTADO, no un umbral aproximado: se rechaza
            # todo, también un apoyo futuro mayor que cualquiera visto en calibración
            return pred, torch.zeros_like(support, dtype=torch.bool), support
        return pred, support >= self.threshold, support


def binom_pvalue(n_fail: int, n: int, risk: float) -> float:
    """p-valor exacto de H0: tasa de fallo > risk, observando n_fail fallos en n casos aceptados.
    p = P(Binomial(n, risk) ≤ n_fail). Con n = 0 no hay evidencia (p = 1)."""
    if n == 0:
        return 1.0
    from scipy.stats import binom
    return float(binom.cdf(n_fail, n, risk))


def ltt_grid(reference_support: torch.Tensor, n_calib: int, risk: float, delta: float, points: int = 15):
    """Rejilla decreciente de umbrales para la secuencia fija, calculada con datos AJENOS a la calibración
    (v0.9.3). La prueba de secuencia fija exige que el orden de las hipótesis se fije antes de ver los
    datos con los que se prueban; ordenar por los apoyos de la propia calibración no lo cumple.
    El primer umbral es el cuantil del apoyo de referencia con el que se esperan ~1.5 m_min casos
    aceptados en la calibración (con menos, ni 0 fallos dan p ≤ δ). Lista vacía = nada certificable."""
    import math
    m_min = max(1, math.ceil(math.log(delta) / math.log(1 - risk)))
    frac = min(1.0, 1.5 * m_min / max(n_calib, 1))
    if n_calib < m_min:
        return []
    qs = torch.linspace(1.0 - frac, 0.0, points)
    return sorted({float(torch.quantile(reference_support.double(), float(q))) for q in qs}, reverse=True)


def ltt_threshold(support: torch.Tensor, fail: torch.Tensor, risk: float = 0.1, delta: float = 0.1, grid=None):
    """Learn-then-Test con prueba de secuencia fija sobre umbrales decrecientes de apoyo.

    Se recorren los umbrales del más estricto al más permisivo; se acepta cada umbral mientras su
    p-valor sea ≤ δ y se para en el primer fallo (control del error de familia sin corrección de
    Bonferroni, porque el orden se fija antes de ver los datos). Devuelve el umbral más permisivo
    certificado y un resumen. Si ninguno se certifica, el umbral queda por encima de todo apoyo
    observado: la colmena se abstiene siempre (lo correcto si no hay evidencia suficiente).
    """
    s, f = support.double(), fail.bool()
    if grid is None:
        raise ValueError("ltt_threshold exige una rejilla fijada con datos ajenos a la calibración "
                         "(ltt_grid). El umbral ordenado con la propia calibración es heuristic_threshold, sin garantía.")
    if True:
        # secuencia fija sobre una rejilla predefinida: se acepta cada umbral mientras p ≤ δ
        best = float("inf")
        info = {"aceptados_calibracion": 0, "cobertura_calibracion": 0.0, "tasa_fallo_calibracion": float("nan"),
                "n_calibracion": int(s.numel()), "rejilla": len(grid), "certificado": False}
        for lam in grid:
            acc = s >= lam
            m, k = int(acc.sum()), int((f & acc).sum())
            if binom_pvalue(k, m, risk) > delta:
                break
            best = float(lam)
            info.update(aceptados_calibracion=m, cobertura_calibracion=m / max(s.numel(), 1),
                        tasa_fallo_calibracion=k / max(m, 1), certificado=True)
        return best, info


def heuristic_threshold(support: torch.Tensor, fail: torch.Tensor, risk: float = 0.1, delta: float = 0.1):
    """Umbral HEURÍSTICO, SIN GARANTÍA ESTADÍSTICA (era la rama grid=None de ltt_threshold hasta v0.9.3).
    Ordena los umbrales con los apoyos de la propia calibración, lo que invalida la prueba de secuencia
    fija. Se conserva solo para comparar; no lo use para afirmar control del riesgo."""
    s, f = support.double(), fail.bool()
    order = s.argsort(descending=True)
    s_sorted, f_sorted = s[order], f[order]
    best = float(s.max()) + 1e-6
    certified, cov, rate = 0, 0.0, float("nan")
    cum_fail = f_sorted.long().cumsum(0)
    n_total = s.numel()
    # con menos de m_min aceptados ni siquiera 0 fallos dan p ≤ δ: esas hipótesis no se pueden rechazar,
    # así que la secuencia empieza en m_min (si empezara en 1, se detendría siempre en el primer paso)
    import math
    m_min = max(1, math.ceil(math.log(delta) / math.log(1 - risk)))
    for m in range(m_min, n_total + 1):
        if m < n_total and s_sorted[m] == s_sorted[m - 1]:
            continue                                                  # umbrales solo entre valores distintos
        p = binom_pvalue(int(cum_fail[m - 1]), m, risk)
        if p > delta:
            break
        best, certified = float(s_sorted[m - 1]), m
        cov, rate = m / n_total, float(cum_fail[m - 1]) / m
    return best, {"aceptados_calibracion": certified, "cobertura_calibracion": cov,
                  "tasa_fallo_calibracion": rate, "n_calibracion": n_total, "certificado": False,
                  "heuristico": True}
