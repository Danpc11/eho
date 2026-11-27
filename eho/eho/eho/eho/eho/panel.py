"""Panel sanguíneo ampliado (sección "Panel sanguíneo ampliado").

1. PanelEncoder: cada análisis presente es un token (variable, valor, tiempo);
   la ausencia de un dato no se imputa: simplemente no hay token.
2. PlasmaViscosity: η_p(b_p) = η_w exp(Σ_k s_k ψ_k(x_k)) con ψ_k crecientes
   y signos fijos: + fibrinógeno, globulinas, triglicéridos, colesterol; − HDL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulario por capas; extensible sin reentrenar el resto (nuevo embedding).
PANEL_VARIABLES = [
    # glóbulo rojo
    "hematocrit", "mcv", "mchc", "rdw", "reticulocytes",
    # proteínas del plasma
    "fibrinogen", "globulins", "albumin", "total_protein",
    # lípidos
    "triglycerides", "cholesterol", "ldl", "hdl",
    # metabolismo
    "glucose", "hba1c", "lactate",
    # inflamación
    "crp", "esr", "wbc", "ferritin",
    # coagulación
    "platelets", "mpv", "d_dimer",
    # iones
    "magnesium", "zinc",
    # función de órganos (desenlaces, no entradas, pero se pueden codificar)
    "creatinine", "egfr", "urea", "bilirubin",
    # medición directa
    "plasma_viscosity", "elongation_index",
]
VAR_INDEX = {v: i for i, v in enumerate(PANEL_VARIABLES)}

# Signo del efecto sobre la viscosidad del plasma (literatura hemorreológica).
PLASMA_SIGNS = {"fibrinogen": +1, "globulins": +1, "triglycerides": +1, "cholesterol": +1,
                "ldl": +1, "total_protein": +1, "hdl": -1}


class PanelEncoder(nn.Module):
    def __init__(self, d: int = 64, n_vars: int = len(PANEL_VARIABLES), n_heads: int = 4):
        super().__init__()
        self.var_emb = nn.Embedding(n_vars, d)
        self.val = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))   # (valor z, Δt)
        self.query = nn.Parameter(torch.randn(1, 1, d) / d ** 0.5)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.out = nn.LayerNorm(d)

    def forward(self, var_ids, values, dt, mask):
        """var_ids, values, dt: [B, K]; mask [B, K] True = observado.

        Los valores deben venir estandarizados (z-score poblacional).
        """
        tok = self.var_emb(var_ids) + self.val(torch.stack([values, dt], -1))
        q = self.query.expand(var_ids.shape[0], -1, -1)
        empty = ~mask.any(-1)
        safe_mask = mask.clone()
        safe_mask[empty, 0] = True                     # evita NaN si no hay ningún dato
        o, _ = self.attn(q, tok, tok, key_padding_mask=~safe_mask)
        o = o.squeeze(1) * (~empty).float().unsqueeze(-1)
        return self.out(o)


class PlasmaViscosity(nn.Module):
    """Viscosidad del plasma monótona en cada componente (mPa·s a 37 °C)."""

    ETA_WATER_37C = 0.69

    def __init__(self, variables=tuple(PLASMA_SIGNS), n_basis: int = 8):
        super().__init__()
        self.variables = list(variables)
        self.signs = torch.tensor([PLASMA_SIGNS[v] for v in self.variables], dtype=torch.float32)
        k = len(self.variables)
        self.raw_w = nn.Parameter(torch.full((k, n_basis), -2.0))
        self.centers = nn.Parameter(torch.linspace(-2, 2, n_basis).repeat(k, 1))
        # base ajustada para que el plasma "promedio" quede ≈ 1.2 mPa·s (rango 1.10-1.30)
        self.log_base = nn.Parameter(torch.tensor(1.2 / self.ETA_WATER_37C).log())

    def psi(self, x):
        """ψ_k(x) = Σ_b w_kb softplus(x - c_kb) - (mismo en x=0): creciente, ψ(0) = 0."""
        w = F.softplus(self.raw_w)
        f = (w * F.softplus(x.unsqueeze(-1) - self.centers)).sum(-1)
        f0 = (w * F.softplus(-self.centers)).sum(-1)
        return f - f0

    def forward(self, x, observed):
        """x: [B, k] valores estandarizados; observed: [B, k] bool. Faltante => aporte 0."""
        contrib = self.signs.to(x.device) * self.psi(x) * observed.float()
        return self.ETA_WATER_37C * torch.exp(self.log_base + contrib.sum(-1))
