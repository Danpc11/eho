"""Datos sintéticos de sangre en tubo, para probar el EHO antes de tener DPD.

NO sustituye a las simulaciones DPD: es un banco de pruebas con física razonable.
- Viscosidad del núcleo: modelo de Quemada con k(γ̇) dependiente de θ.
- Efecto Fåhræus-Lindqvist: modelo de dos capas (núcleo + capa libre de células).
- Campos en la sección transversal: velocidad de Poiseuille de dos capas y
  fracción de glóbulos φ(r) con capa libre suavizada.

θ = (rigidez s, agregación a, déficit de volumen v, factor de plasma p) ∈ [0,1]^4
c = (hematocrito H, diámetro D [μm], tasa de corte γ̇ [1/s])
"""
import math
import torch

REGIMES = {                       # parámetros Beta(α, β) por componente de θ
    "sano":        {"s": (2, 6), "a": (2, 6), "v": (2, 6), "p": (2, 5)},
    "prediabetes": {"s": (4, 4), "a": (4, 4), "v": (3, 5), "p": (3, 4)},
    "dt2":         {"s": (6, 2), "a": (6, 2), "v": (5, 3), "p": (5, 3)},
}
THETA_NAMES = ["s", "a", "v", "p"]


def sample_theta(n, regime="sano", generator=None):
    cols = []
    for k in THETA_NAMES:
        a, b = REGIMES[regime][k]
        cols.append(torch.distributions.Beta(float(a), float(b)).sample((n,)))
    return torch.stack(cols, -1)


def sample_conditions(n):
    H = 0.20 + 0.30 * torch.rand(n)
    D = torch.exp(math.log(6.0) + (math.log(60.0) - math.log(6.0)) * torch.rand(n))
    g = torch.exp(math.log(1.0) + (math.log(500.0) - math.log(1.0)) * torch.rand(n))
    return H, D, g


def quemada_relative(theta, H, gdot):
    s, a, v, _ = theta.unbind(-1)
    k0 = 2.6 + 1.0 * a
    kinf = 1.6 + 0.6 * s + 0.3 * v
    gc = 2.0 * (1.0 + a)
    sq = torch.sqrt(gdot / gc)
    k = (k0 + kinf * sq) / (1 + sq)
    return (1 - (0.5 * k * H).clamp(max=0.9)) ** -2


def cell_free_layer(theta, H, D):
    s, a, _, _ = theta.unbind(-1)
    delta = (2.6 - 3.0 * H + 1.2 * (1 - s) + 0.6 * a) * (1 - torch.exp(-D / 15.0))
    return delta.clamp(min=0.3).clamp(max=0.45 * D / 2)


def carreau_relative(theta, H, gdot):
    """Segundo modelo de núcleo (Carreau-Yasuda), con otra estructura: sustituye a un
    segundo simulador (p. ej. HemoCell frente a Mirheo) con discrepancia de modelo real."""
    s, a, v, _ = theta.unbind(-1)
    eta_inf = torch.exp(2.0 * H / (1 - 0.5 * H)) * (1 + 0.3 * s + 0.15 * v)
    eta_0 = eta_inf * (1 + 6.0 * (0.5 + a) * H)
    lam = 2.0 * (1 + a)
    return eta_inf + (eta_0 - eta_inf) * (1 + (lam * gdot) ** 2) ** ((0.4 - 1) / 2)


def cell_free_layer_b(theta, H, D):
    s, a, _, _ = theta.unbind(-1)
    delta = (2.4 - 2.7 * H + 1.4 * (1 - s) + 0.5 * a) * (1 - torch.exp(-D / 12.0))
    return delta.clamp(min=0.3).clamp(max=0.45 * D / 2)


def pries_in_vitro(D, H):
    """Viscosidad relativa aparente en tubos de vidrio a corte alto: ley empírica de Pries, Neuhaus y
    Gaehtgens (Am J Physiol 1992), ajustada a mediciones experimentales. D en μm, H = hematocrito de descarga."""
    eta45 = 220.0 * torch.exp(-1.3 * D) + 3.2 - 2.44 * torch.exp(-0.06 * D ** 0.645)
    w = 1.0 / (1.0 + 1e-11 * D ** 12)
    C = (0.8 + torch.exp(-0.075 * D)) * (-1.0 + w) + w
    return 1.0 + (eta45 - 1.0) * ((1.0 - H) ** C - 1.0) / ((1.0 - 0.45) ** C - 1.0)


def pries_casson(theta, H, D, gdot):
    """Tercer modelo, que NINGÚN agente ve al entrenar (v0.7): ley de Pries para la dependencia con el
    diámetro y el hematocrito, multiplicada por un factor de corte tipo Casson que depende de θ
    (agregación a corte bajo; rigidez y volumen a corte alto). Familia funcional distinta de Quemada y
    de Carreau-Yasuda, así que la discrepancia de modelo es real (ya no es el promedio de los agentes).
    Devuelve (η_aparente, δ) con δ de la media de las dos leyes de capa libre."""
    s, a, v, _ = theta.unbind(-1)
    gc = 0.4 * (1.0 + 3.0 * a) * (H / 0.45) ** 2                  # corte característico de agregación
    shear = (1.0 + torch.sqrt(gc / gdot)) ** 2 / (1.0 + torch.sqrt(gc / 500.0)) ** 2
    mech = 1.0 + 0.25 * s + 0.10 * v
    eta_app = 1.0 + (pries_in_vitro(D, H) - 1.0) * shear * mech
    delta = 0.5 * (cell_free_layer(theta, H, D) + cell_free_layer_b(theta, H, D))
    return eta_app, delta


def core_and_layer(theta, H, D, gdot, simulador="quemada"):
    """quemada | carreau | real (media geométrica de ambos) | pries (tercer modelo, fuera del
    entrenamiento). Para "pries" se devuelve la viscosidad del núcleo que reproduce la aparente."""
    if simulador == "quemada":
        return quemada_relative(theta, H, gdot), cell_free_layer(theta, H, D)
    if simulador == "carreau":
        return carreau_relative(theta, H, gdot), cell_free_layer_b(theta, H, D)
    if simulador == "real":
        e = torch.sqrt(quemada_relative(theta, H, gdot) * carreau_relative(theta, H, gdot))
        return e, 0.5 * (cell_free_layer(theta, H, D) + cell_free_layer_b(theta, H, D))
    if simulador == "pries":
        eta_app, delta = pries_casson(theta, H, D, gdot)
        lam4 = (1 - delta / (D / 2)) ** 4
        # invertir el modelo de dos capas: η_app = 1 / (1 - λ⁴ (1 - 1/η_core))
        inv_core = 1.0 - (1.0 - 1.0 / eta_app) / lam4
        return 1.0 / inv_core.clamp(min=1e-3), delta
    raise ValueError(simulador)


def ground_truth(theta, H, D, gdot, n_points=128, generator=None, simulador="quemada"):
    B = theta.shape[0]
    R = D / 2
    eta_core, delta = core_and_layer(theta, H, D, gdot, simulador)   # relativa a plasma
    lam = (1 - delta / R)                                         # R_c / R
    eta_app = 1.0 / (1.0 - lam ** 4 * (1.0 - 1.0 / eta_core))     # dos capas (Sharan-Popel)
    # puntos uniformes en el disco unitario
    rr = torch.sqrt(torch.rand(B, n_points, generator=generator))
    ang = 2 * math.pi * torch.rand(B, n_points, generator=generator)
    pts = torch.stack([rr * torch.cos(ang), rr * torch.sin(ang), rr], -1)
    lc = lam.unsqueeze(-1)
    ec = eta_core.unsqueeze(-1)
    u_out = (1 - rr ** 2)                                         # r >= R_c (plasma, η=1)
    u_in = (1 - lc ** 2) + (lc ** 2 - rr ** 2) / ec               # r < R_c (núcleo)
    u = torch.where(rr >= lc, u_out, u_in)
    u = u / ((1 - lc ** 2) + lc ** 2 / ec)                        # normaliza por u(0)
    H_core = (H / lam ** 2).clamp(max=0.95).unsqueeze(-1)         # conserva hematocrito de tubo
    width = 0.3 / R.unsqueeze(-1)                                 # 0.3 μm de transición
    phi = H_core * torch.sigmoid((lc - rr) / width)
    fields = torch.stack([u, phi], -1)
    scalars = torch.stack([eta_app.log(), delta / R], -1)
    return pts, fields, scalars


def make_condition_vector(theta, H, D, gdot):
    return torch.cat([theta, ((H - 0.35) / 0.1).unsqueeze(-1),
                      ((D.log() - math.log(19.0)) / 0.7).unsqueeze(-1),
                      ((gdot.log() - math.log(22.0)) / 1.8).unsqueeze(-1)], -1)


def make_dataset(n, regime="sano", n_points=128, seed=0):
    torch.manual_seed(seed)
    theta = sample_theta(n, regime)
    H, D, g = sample_conditions(n)
    pts, fields, scalars = ground_truth(theta, H, D, g, n_points)
    return {"points": pts, "cond": make_condition_vector(theta, H, D, g), "gdot": g,
            "fields": fields, "scalars": scalars, "theta": theta, "H": H, "D": D, "g": g}


def subset(ds, idx):
    return {k: v[idx] for k, v in ds.items()}


def concat(*dss):
    return {k: torch.cat([d[k] for d in dss]) for k in dss[0]}
