"""Pruebas de la v0.8: forrajeo, colmena con urgencia, centinela, árbitro y Chesson."""
import math

import torch

from eho.forrajeo import ReglaParada
from eho.colony import BeeColony, urgency_sigma
from eho.centinela import Centinela
from eho.mezcla import ArbitroRapidoLento


def test_regla_parada_no_para_en_frio_y_para_con_rendimientos_decrecientes():
    r = ReglaParada(tau=4, rendicion=2, n_min=4)
    assert r.continuar()
    r.registrar([3.0, 2.5, 2.0])                    # aprende mucho: sigue
    assert r.continuar()
    for _ in range(4):
        r.registrar([1.0, 1.05, 0.9, 1.0])           # ya predice como lo conocido: nada que aprender
    assert not r.continuar() and r.parada
    assert not r.continuar()                         # una vez parada, no reabre


def test_regla_parada_paciente_si_cada_simulacion_ensena():
    r = ReglaParada(tau=4, rendicion=2, n_min=2)
    for _ in range(6):
        r.registrar([2.0, 2.2])
        assert r.continuar()


def test_urgencia_y_descubrimiento_minimo():
    assert urgency_sigma(0, 10) < urgency_sigma(5, 10) < urgency_sigma(10, 10)
    V = torch.full((4, 3), 0.02)                     # todo vale poco: sin piso, nadie se compromete
    sin = BeeColony(quorum=0.3, t_max=20, seed=0).decide(V)
    con = BeeColony(quorum=0.3, t_max=20, seed=0, gamma_min=0.5).decide(V)
    assert float(con.support.sum()) > float(sin.support.sum())


class _Fake:
    def __init__(self, cal):
        self.cal = cal


def test_martingala_conformal_controla_falsas_alarmas():
    torch.manual_seed(0)
    c = Centinela([], torch.ones(1), alpha=0.05)
    c.cal = torch.randn(2000).abs().sort().values
    falsas = sum(c.alarm(torch.randn(40).abs()) is not None for _ in range(300))
    assert falsas / 300 <= 0.05 + 0.03                 # Ville: ≤ α (más holgura por 300 repeticiones)
    detect = sum(c.alarm(3 * torch.randn(40).abs()) is not None for _ in range(100))
    assert detect / 100 > 0.9


def test_arbitro_pesa_al_mejor():
    Y = torch.randn(200, 2)
    T = torch.zeros(200, 4, 1)
    bueno, malo = Y + 0.1 * torch.randn(200, 2), Y + 1.0 * torch.randn(200, 2)
    a = ArbitroRapidoLento().fit((bueno, T, Y, T), (malo, T))
    assert (a.a_s > 0.9).all()
    S, _ = a.combine((bueno, T), (malo, T))
    assert ((S - Y) ** 2).mean() <= ((bueno - Y) ** 2).mean() * 1.05


def test_gemelo_v08_mvt_y_chesson():
    from eho import Gemelo
    from eho.ecology import chesson_diagnostic
    cfg = {"nombre": "t", "dominio": {"nombre": "edp1d", "regimenes": ["difusivo"], "episodios_por_regimen": 8,
                                      "muestras_por_episodio": 4, "episodios_validacion": 4},
           "modelo": {"d": 24, "n_slices": 4, "sucesion": 1},
           "colmena": {"n_agentes": 2},
           "entrenamiento": {"epocas": 2, "dispositivo": "cpu"},
           "activo": {"parada": "mvt", "presupuesto": 2, "epocas_ajuste": 1, "repeticion": 4}}
    g = Gemelo.desde_config(cfg).entrenar(verbose=False)
    cand = g.dominio_variante(0).generate(6, 4, "difusivo", seed=3)
    elegidos = g.proponer(cand)
    assert 1 <= len(elegidos) <= 2                     # en frío, la regla no deja parar
    ids = torch.tensor([e["episodio"] for e in elegidos])
    g.incorporar(cand_sub := {k: v[torch.isin(cand["episode"], ids)] for k, v in cand.items()})
    assert g.parada.n == len(elegidos)
    d = chesson_diagnostic(g.agentes[0], g.validacion)
    assert d["especies"] >= 1
