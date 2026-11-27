"""Resume los JSON de results/ con la estadística que pide el plan: media, IC 95 % por bootstrap
entre semillas y prueba de Wilcoxon pareada (por semilla) contra la línea base de referencia.

    python3 analisis/estadistica.py            # imprime tablas en Markdown
    python3 analisis/estadistica.py --csv      # además escribe results/resumen_*.csv
"""
import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np


def ic_bootstrap(x, n=5000, nivel=0.95, seed=0):
    x = np.asarray(x, float)
    if x.size < 2:
        return float(x.mean()) if x.size else float("nan"), float("nan"), float("nan")
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, x.size)).mean(1)
    return float(x.mean()), float(np.quantile(m, (1 - nivel) / 2)), float(np.quantile(m, 1 - (1 - nivel) / 2))


def wilcoxon(a, b):
    """p-valor bilateral de Wilcoxon pareado (scipy); nan si hay menos de 5 pares."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 5 or np.allclose(a, b):
        return float("nan")
    from scipy.stats import wilcoxon as w
    return float(w(a, b).pvalue)


def no_inferioridad(nuevo, referencia, margen, nivel=0.95, metodo="t"):
    """Prueba de NO INFERIORIDAD pareada por semilla (menor es mejor). Δ_s = nuevo_s - referencia_s.

    nuevo y referencia: diccionarios {semilla: valor} (recomendado: el emparejamiento es explícito) o
    vectores de la misma longitud y en el mismo orden de semillas. Se usan solo las semillas presentes en
    ambos. El margen (deterioro máximo aceptable) debe fijarse ANTES de ver los datos.
    Se declara no inferioridad si el límite superior unilateral de la media de Δ es < margen.
      metodo="t": intervalo t de Student pareado (supone Δ aproximadamente normal; con pocas semillas es
                  más fiable que el bootstrap percentil, que subcubre).
      metodo="bootstrap": percentil; úselo con muchas semillas.
    Qué demuestra: que lo nuevo no es peor que la referencia por más del margen. NO demuestra
    equivalencia (para eso hacen falta dos pruebas unilaterales, TOST, con margen a ambos lados).
    v0.9.7: antes no se comprobaba el emparejamiento y NumPy difundía un solo valor contra cinco."""
    if isinstance(nuevo, dict) != isinstance(referencia, dict):
        raise ValueError("no_inferioridad: pase dos diccionarios {semilla: valor} o dos vectores")
    if isinstance(nuevo, dict):
        semillas = sorted(set(nuevo) & set(referencia))
        x = np.array([nuevo[s] for s in semillas], dtype=float)
        y = np.array([referencia[s] for s in semillas], dtype=float)
    else:
        x, y = np.asarray(nuevo, dtype=float), np.asarray(referencia, dtype=float)
        semillas = list(range(x.size))
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("no_inferioridad: se requieren vectores de una dimensión")
        if x.shape != y.shape:
            raise ValueError(f"no_inferioridad: semillas no pareadas ({x.size} frente a {y.size})")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("no_inferioridad: hay resultados no finitos")
    if not (np.isfinite(margen) and margen > 0):
        raise ValueError("no_inferioridad: el margen debe ser finito y positivo")
    d = x - y
    base = {"n": int(d.size), "semillas": semillas, "delta_media": float(d.mean()) if d.size else float("nan"),
            "margen": margen, "metodo": metodo}
    if d.size < 5:
        return {**base, "limite_superior": float("nan"), "no_inferior": None,
                "nota": "se necesitan al menos 5 semillas pareadas"}
    if metodo == "t":
        from scipy.stats import t
        sup = float(d.mean() + t.ppf(nivel, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size))
    elif metodo == "bootstrap":
        sup = float(np.quantile(np.random.default_rng(0).choice(d, (10000, d.size)).mean(1), nivel))
    else:
        raise ValueError("metodo debe ser 't' o 'bootstrap'")
    return {**base, "limite_superior": sup, "no_inferior": bool(sup < margen)}


def fmt(m, lo, hi, d=3):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if np.isfinite(lo) else f"{m:.{d}f}"


def cargar(patron):
    out = defaultdict(dict)
    for f in glob.glob(patron):
        mt = re.search(r"_([a-z0-9]+)_s(\d+)\.json$", f)
        if mt:
            out[mt.group(1)][int(mt.group(2))] = json.load(open(f))
    return out


def novedad():
    datos = cargar("results/novedad_*_s*.json")
    for dom, por_semilla in sorted(datos.items()):
        semillas = sorted(por_semilla)
        dets = list(por_semilla[semillas[0]]["detectores"])
        print(f"\n### Novedad — {dom} ({len(semillas)} semillas): AUROC media [IC 95 %]\n")
        print("| Detector | Cambio visible | Cambio oculto |\n| --- | --- | --- |")
        for d in dets:
            cel = []
            for sh in ("visible", "oculto"):
                x = [por_semilla[s]["detectores"][d][sh]["auroc"] for s in semillas]
                cel.append(fmt(*ic_bootstrap(x), d=2))
            print(f"| {d} | {cel[0]} | {cel[1]} |")


def presupuesto(ref="aleatoria"):
    datos = cargar("results/presupuesto_*_s*.json")
    for dom, por_semilla in sorted(datos.items()):
        semillas = sorted(por_semilla)
        estr = list(por_semilla[semillas[0]]["estrategias"])
        print(f"\n### Qué simular — {dom} ({len(semillas)} semillas): error en lo nuevo al final\n")
        print(f"| Estrategia | Simulaciones | Escalares, nuevo | Campos, nuevo | Escalares, conocido | p vs {ref} |")
        print("| --- | --- | --- | --- | --- | --- |")
        fin = lambda e, k: [por_semilla[s]["estrategias"][e][-1][k] for s in semillas]
        for e in estr:
            p = wilcoxon(fin(e, "nuevo_escalares"), fin(ref, "nuevo_escalares")) if e != ref else float("nan")
            print(f"| {e} | {np.mean(fin(e, 'acumuladas')):.1f} | {fmt(*ic_bootstrap(fin(e, 'nuevo_escalares')))} | "
                  f"{fmt(*ic_bootstrap(fin(e, 'nuevo_campos')))} | {fmt(*ic_bootstrap(fin(e, 'conocido_escalares')))} | "
                  f"{p:.3g} |")
        for k, txt in (("a_mismas_sim", "frente al codicioso con las mismas simulaciones"),
                       ("perdida_por_parar", "frente al codicioso con el presupuesto completo")):
            arr = [por_semilla[s]["colmena_paro"][k] for s in semillas if "colmena_paro" in por_semilla[s]]
            if arr:
                print(f"\nColmena, error en lo nuevo {txt}: {fmt(*ic_bootstrap(arr), d=4)}")


def pocos_ejemplos():
    datos = cargar("results/pocos_ejemplos_*_s*.json")
    for dom, por_semilla in sorted(datos.items()):
        semillas = sorted(por_semilla)
        vars_ = list(por_semilla[semillas[0]]["variantes"])
        print(f"\n### Pocos ejemplos — {dom} ({len(semillas)} semillas): error escalar\n")
        print("| Variante | k = 1 | k = 2 | k = 5 |\n| --- | --- | --- | --- |")
        for v in vars_:
            cel = []
            for k in (1, 2, 5):
                x = [por_semilla[s]["variantes"][v][f"k={k}"]["escalares_MAE"] for s in semillas
                     if v in por_semilla[s]["variantes"]]
                cel.append(fmt(*ic_bootstrap(x), d=4))
            print(f"| {v} | {' | '.join(cel)} |")


def agotamiento():
    """Reserva que se agota: simulaciones gastadas y error final de la colmena v0.8 frente al codicioso."""
    fs = sorted(glob.glob("results/presupuesto_*_agota*.json"))
    if not fs:
        return
    print("\n### Regla de parada con reserva que se agota\n")
    print("| Dominio | Semilla | Codicioso: sim. / error | Colmena v0.8: sim. / error | Ahorro | Pérdida |")
    print("| --- | --- | --- | --- | --- | --- |")
    for f in fs:
        r = json.load(open(f))
        g, c = r["estrategias"]["valor_codicioso"][-1], r["estrategias"]["colmena_v08"][-1]
        print(f"| {r['dominio']} | {r['semilla']} | {g['acumuladas']} / {g['nuevo_escalares']:.3f} | "
              f"{c['acumuladas']} / {c['nuevo_escalares']:.3f} | {1 - c['acumuladas'] / g['acumuladas']:.0%} | "
              f"{c['nuevo_escalares'] - g['nuevo_escalares']:+.3f} |")


def ecologia():
    fs = sorted(glob.glob("results/ecologia_s*.json"))
    if not fs:
        return
    rs = [json.load(open(f)) for f in fs]
    print(f"\n### Ecología en aprendizaje continuo ({len(rs)} semillas)\n")
    print("| Variante | Error final medio | Olvido en sano | Especies | Pares que coexisten (Chesson) |")
    print("| --- | --- | --- | --- | --- |")
    for v in rs[0]["variantes"]:
        fin = [r["variantes"][v]["error_final_medio"] for r in rs]
        olv = [r["variantes"][v]["olvido_sano"] for r in rs]
        esp = [r["variantes"][v]["etapas"][-1]["especies"] for r in rs]
        ch = [r["variantes"][v]["etapas"][-1].get("chesson", {}).get("fraccion_coexistencia", float("nan")) for r in rs]
        print(f"| {v} | {fmt(*ic_bootstrap(fin))} | {fmt(*ic_bootstrap(olv))} | {np.mean(esp):.1f} | {np.nanmean(ch):.2f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    novedad()
    presupuesto()
    pocos_ejemplos()
    agotamiento()
    ecologia()
