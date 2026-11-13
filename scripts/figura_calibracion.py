"""Calibración del modelo final en el conjunto de prueba (2014J), sección 2.2.

Comprueba si la probabilidad estimada, que el sistema muestra al profesor, puede leerse como
riesgo: agrupa a los estudiantes en deciles de probabilidad y compara la probabilidad media de
cada decil con la proporción de estudiantes realmente en riesgo. Solo lee el modelo fijado en
04_modelo_final; no recalibra ni cambia nada.

Uso (desde la raíz de Proposal):
    ABRIR_PRUEBA=1 .venv/bin/python scripts/figura_calibracion.py
La figura se guarda en figuras/Training/calibracion_prueba.pdf.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import comun as c  # noqa: E402

c.thesis_style()
d = pd.read_parquet(c.DATASET)
d["cut"] = d["cut"].astype(str)

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), sharey=True, gridspec_kw={"wspace": 0.12})
filas = {}
for ax, cut in zip(axes, c.CUTS):
    f = joblib.load(c.ARTIFACTS / f"final_{cut}.joblib")
    t = c.load_test(cut)
    p = f["model"].predict_proba(t[f["features"]])[:, 1]
    g = pd.DataFrame({"p": p, "y": t[c.TARGET].values})
    g["decil"] = pd.qcut(g["p"].rank(method="first"), 10, labels=False)
    dec = g.groupby("decil").agg(p=("p", "mean"), y=("y", "mean"))
    train = d[(d.cut == cut) & (d.code_presentation != c.TEST_PERIOD)]
    filas[c.CUT_NAMES[cut]] = {"riesgo_entrenamiento": train[c.TARGET].mean(),
                               "riesgo_prueba": g["y"].mean(), "p_media_prueba": g["p"].mean(),
                               "umbral": f["threshold"], "max_sobreestimacion": (dec.p - dec.y).max()}

    ax.plot([0, 1], [0, 1], ":", color="#aaaaaa", lw=0.8)
    ax.axvline(f["threshold"], ls="--", color="#555555", lw=0.8)
    ax.plot(dec["p"], dec["y"], "o-", color=c.CUT_COLORS[cut], ms=3.5, lw=1.2)
    ax.set_title(c.CUT_NAMES[cut])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Probabilidad estimada")
    c.comma_axis(ax, "x", 1)
    c.comma_axis(ax, "y", 1)
axes[0].set_ylabel("Proporción en riesgo")

c.save_figure(fig, "calibracion_prueba")
print(pd.DataFrame(filas).round(3).to_string())
