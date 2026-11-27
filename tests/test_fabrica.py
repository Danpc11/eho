"""v0.6: fábrica de modelos (configuración, registro de dominios, Gemelo, persistencia, CLI)."""
import os
import subprocess
import sys

import pytest
import torch

from eho import Gemelo, Config
from eho.dominios import crear_dominio, dominios_disponibles, Sangre
from eho.active import InspectionValue

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MINI = {"nombre": "mini", "dominio": {"nombre": "edp1d", "regimenes": ["difusivo", "advectivo"],
                                      "episodios_por_regimen": 6, "episodios_validacion": 3},
        "colmena": {"n_agentes": 2}, "entrenamiento": {"epocas": 2, "dispositivo": "cpu"}}


@pytest.fixture(scope="module")
def gemelo():
    torch.manual_seed(0)
    return Gemelo.desde_config(MINI).entrenar(verbose=False)


def test_registro_y_creacion():
    assert {"sangre", "edp1d"} <= set(dominios_disponibles())
    d = crear_dominio("sangre", H_range=(0.55, 0.7))
    assert isinstance(d, Sangre) and d.H_range == (0.55, 0.7)
    d2 = crear_dominio("eho.dominios.edp1d:AdveccionDifusion1D", k_min=4)
    assert d2.k_min == 4
    with pytest.raises(KeyError):
        crear_dominio("no_existe")


def test_config_ida_y_vuelta(tmp_path):
    cfg = Config.desde_dict(MINI)
    for ext in ("yaml", "json"):
        p = tmp_path / f"c.{ext}"
        cfg.guardar(p)
        assert Config.desde_archivo(p).a_dict() == cfg.a_dict()
    with pytest.raises(KeyError):
        Config.desde_dict({"modelo": {"capas_magicas": 3}})
    for f in ("sangre.yaml", "edp1d.yaml"):
        Config.desde_archivo(os.path.join(RAIZ, "configs", f))


def test_gemelo_predice_propone_y_persiste(gemelo, tmp_path):
    d = gemelo.dominios[0]
    cand = d.generate(3, 8, "difusivo", 5)
    r = gemelo.predecir(subset3(cand, slice(0, 3)), subset3(cand, slice(3, 6)))
    assert r["escalares"].shape == (3, 2) and r["aceptado"].dtype == torch.bool
    elegidos = gemelo.proponer(cand)
    assert len(elegidos) <= gemelo.cfg.activo.presupuesto
    ruta = gemelo.guardar(tmp_path / "m.eho")
    g2 = Gemelo.cargar(ruta, "cpu")
    q = subset3(cand, slice(0, 4))
    for a, b in zip(gemelo.agentes, g2.agentes):
        assert torch.allclose(a(q)["scalars"], b(q)["scalars"], atol=1e-6)
    assert g2.datos_entrenamiento is None             # los datos no viajan dentro del modelo


def subset3(ds, sl):
    return {k: v[sl] for k, v in ds.items()}


def test_incorporar_registra_error_antes_de_aprender(gemelo):
    import copy
    g = copy.deepcopy(gemelo)
    nuevo = g.dominio_variante(0, k_min=4).generate(2, 8, "difusivo", 9)
    g.incorporar(nuevo)
    ds, s, w = g.valor.err.records[-1]
    assert s.shape == (2, 16) and w == 1.0
    V, eps, partes = g.valor.episode_values(nuevo)
    assert set(partes) == {"novedad", "error", "inspeccionado"} and (V >= partes["novedad"] - 1e-6).all()


def test_valor_por_inspeccion_sube_cerca_de_errores_altos(gemelo):
    iv = InspectionValue(gemelo.agentes, k=2)
    conocido = gemelo.dominios[0].generate(3, 8, "difusivo", 21)
    iv.fit(conocido)
    assert float(iv.values(conocido).max()) == 0.0              # sin registros: sin valor por error
    malo = gemelo.dominio_variante(0, k_min=4).generate(2, 8, "advectivo", 22)
    s = iv.register(malo)
    v_malo = iv.values(malo).mean()
    # registro artificial de error muy bajo en lo conocido: su valor debe quedar cerca de 0
    iv.records.append(([conocido] * 2, torch.zeros(2, conocido["episode"].numel()), 1.0))
    v_con = iv.values(conocido)
    assert 0.0 <= float(v_malo) <= 1.0 and s.shape == (2, 16)
    assert float(v_con.mean()) < 0.5


def test_cli_nuevo_dominio_y_entrenar(tmp_path):
    env = {**os.environ, "PYTHONPATH": RAIZ}
    run = lambda *a: subprocess.run([sys.executable, "-m", "eho", *a], cwd=tmp_path, env=env,
                                    capture_output=True, text=True, check=True)
    run("nuevo-dominio", "prueba_x")
    cfg = (tmp_path / "configs" / "prueba_x.yaml").read_text()
    cfg = cfg.replace("n_agentes: 4", "n_agentes: 2").replace("epocas: 30", "epocas: 1") \
             .replace("episodios_por_regimen: 40", "episodios_por_regimen: 4")
    (tmp_path / "configs" / "prueba_x.yaml").write_text(cfg)
    out = run("entrenar", "configs/prueba_x.yaml", "-o", "m.eho", "--dispositivo", "cpu")
    assert (tmp_path / "m.eho").exists(), out.stderr
    info = run("info", "m.eho")
    assert "prueba_x" in info.stdout
