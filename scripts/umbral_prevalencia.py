"""Umbral = proporción de desaprobados del entrenamiento (reescritura del Cap. 2 por Davier, 30/10).

Sustituye al umbral F2 de 04_modelo_final (E14) en la evaluación del modelo final en 2014J. No
cambia el modelo: mismos coeficientes, variables y C; el umbral no usa datos de prueba (es la
proporción de desaprobados de los periodos de entrenamiento en cada corte). Calcula:

- con el umbral: % de marcados, sensibilidad, precisión y especificidad, del modelo y de la regla
  de la nota (nota media con ceros < 40);
- con la misma lista: el modelo marca a tantos estudiantes como la regla (los de mayor riesgo) y se
  comparan sensibilidad y precisión;
- la evolución de la alerta entre cortes consecutivos;

y genera las figuras de la sección 2.4 (curvas ROC con el punto de cada criterio, comparación con la
misma lista, calibración con el umbral nuevo y evolución de la alerta).

Uso (desde la raíz de Proposal):
    ABRIR_PRUEBA=1 MPLBACKEND=Agg .venv/bin/python scripts/umbral_prevalencia.py
Figuras en figuras/Training/ y cifras en artifacts/06_umbral_prevalencia.json.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import comun as c  # noqa: E402

c.thesis_style()
final = {cut: joblib.load(c.ARTIFACTS / f"final_{cut}.joblib") for cut in c.CUTS}
test = {cut: c.load_test(cut) for cut in c.CUTS}
train = {cut: c.load_train(cut) for cut in c.CUTS}
umbral = {cut: float(train[cut][c.TARGET].mean()) for cut in c.CUTS}
prob = {cut: final[cut]["model"].predict_proba(test[cut][final[cut]["features"]])[:, 1] for cut in c.CUTS}


def metricas(y, yhat):
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"marcados_%": 100 * yhat.mean(), "sensibilidad": tp / (tp + fn), "precision": tp / (tp + fp),
            "especificidad": tn / (tn + fp), "n_marcados": int(yhat.sum())}


filas, misma = [], []
etiquetas = {}
for cut in c.CUTS:
    y = test[cut][c.TARGET].to_numpy()
    modelo = (prob[cut] > umbral[cut]).astype(int)
    regla = c.baseline_label(test[cut])
    etiquetas[cut] = modelo
    filas.append({"corte": cut, "umbral": umbral[cut], "n_entrenamiento": len(train[cut]),
                  **{f"modelo_{k}": v for k, v in metricas(y, modelo).items()},
                  **{f"regla_{k}": v for k, v in metricas(y, regla).items()}})
    # Misma lista: el modelo marca a los n de mayor probabilidad, con n = los que marca la regla.
    n = int(regla.sum())
    top = np.zeros_like(y)
    top[np.argsort(-prob[cut], kind="stable")[:n]] = 1
    misma.append({"corte": cut, "n": n, "marcados_%": 100 * n / len(y),
                  **{f"modelo_{k}": v for k, v in metricas(y, top).items() if k in ("sensibilidad", "precision")},
                  **{f"regla_{k}": v for k, v in metricas(y, regla).items() if k in ("sensibilidad", "precision")}})
tabla = pd.DataFrame(filas).set_index("corte")
tabla_misma = pd.DataFrame(misma).set_index("corte")

# Curvas ROC con el punto de cada criterio.
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8), sharey=True)
for ax, cut in zip(axes, c.CUTS):
    t, y = test[cut], test[cut][c.TARGET].to_numpy()
    nota = c.baseline_score(t)
    for nombre, s, ls, color in [("Modelo", prob[cut], "-", c.CUT_COLORS[cut]), ("Regla de la nota", nota, "--", "#555555")]:
        fpr, tpr, _ = roc_curve(y, s)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, ls, color=color, lw=1.4, label=f"{nombre} ({c.comma(auc)})")
    r = tabla.loc[cut]
    ax.plot(1 - r["modelo_especificidad"], r["modelo_sensibilidad"], "o", color=c.CUT_COLORS[cut], ms=5)
    ax.plot(1 - r["regla_especificidad"], r["regla_sensibilidad"], "s", color="#555555", ms=4.5)
    ax.plot([0, 1], [0, 1], ":", color="#aaaaaa", lw=0.8)
    ax.set_title(c.CUT_NAMES[cut])
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.5)
    c.comma_axis(ax, "x", 1)
    c.comma_axis(ax, "y", 1)
axes[0].set_ylabel("Sensibilidad")
fig.tight_layout()
c.save_figure(fig, "curvas_roc_prueba_umbral")
plt.close(fig)

# Misma lista: sensibilidad y precisión.
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
x = np.arange(3)
for ax, met, titulo in zip(axes, ["sensibilidad", "precision"], ["Sensibilidad", "Precisión"]):
    for k, (quien, color, nombre) in enumerate([("modelo", "#2a6fdb", "Modelo"), ("regla", "#8c8c8c", "Regla de la nota")]):
        v = tabla_misma[f"{quien}_{met}"].to_numpy()
        ax.bar(x + (k - 0.5) * 0.36, v, 0.36, color=color, label=nombre)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * 0.36, vi + 0.02, c.comma(vi, 2), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [c.CUT_NAMES[cut] for cut in c.CUTS], fontsize=8)
    ax.set_title(titulo)
    ax.set_ylim(0, 1)
    c.comma_axis(ax, "y", 1)
axes[0].legend(loc="upper left", fontsize=7)
fig.tight_layout()
c.save_figure(fig, "comparacion_misma_lista")
plt.close(fig)

# Calibración con el umbral nuevo.
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, cut in zip(axes, c.CUTS):
    g = pd.DataFrame({"p": prob[cut], "y": test[cut][c.TARGET].to_numpy()})
    g["decil"] = pd.qcut(g["p"].rank(method="first"), 10, labels=False)
    dec = g.groupby("decil").agg(p=("p", "mean"), y=("y", "mean"))
    tabla.loc[cut, "max_sobreestimacion"] = float((dec.p - dec.y).max())
    ax.plot([0, 1], [0, 1], ":", color="#aaaaaa", lw=0.8)
    ax.axvline(umbral[cut], ls="--", color="#555555", lw=0.8)
    ax.plot(dec["p"], dec["y"], "o-", color=c.CUT_COLORS[cut], ms=3.5, lw=1.2)
    ax.set_title(c.CUT_NAMES[cut])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Probabilidad estimada")
    c.comma_axis(ax, "x", 1)
    c.comma_axis(ax, "y", 1)
axes[0].set_ylabel("Proporción en riesgo académico")
c.save_figure(fig, "calibracion_prueba_umbral")
plt.close(fig)

# Evolución de la alerta entre cortes consecutivos.
KEY = ["id_student", "code_module"]
GRUPOS = {(0, 0): "Sin alerta\nen ambos", (1, 0): "Sale de\nla alerta", (0, 1): "Entra en\nla alerta", (1, 1): "Con alerta\nen ambos"}
trans = []
for a, b in [("first", "mid"), ("mid", "last")]:
    A = test[a][KEY].assign(alerta_a=etiquetas[a])
    B = test[b][KEY + [c.TARGET]].assign(alerta_b=etiquetas[b])
    m = A.merge(B, on=KEY)
    for (ga, gb), g in m.groupby(["alerta_a", "alerta_b"]):
        trans.append({"paso": f"{a}→{b}", "grupo": GRUPOS[ga, gb].replace("\n", " "), "estudiantes": len(g),
                      "desaprobados_%": 100 * g[c.TARGET].mean()})
trans = pd.DataFrame(trans)
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
orden = [v.replace("\n", " ") for v in GRUPOS.values()]
for ax, (paso, titulo) in zip(axes, [("first→mid", "Del primer al segundo corte"), ("mid→last", "Del segundo al tercer corte")]):
    t = trans[trans["paso"] == paso].set_index("grupo").reindex(orden)
    ax.bar(range(4), t["desaprobados_%"], color=["#9ecae1", "#1b9e77", "#d9622b", "#7a5195"])
    for i, (pct, n) in enumerate(zip(t["desaprobados_%"], t["estudiantes"])):
        ax.text(i, pct + 1.5, f"{c.comma(pct, 1)} %\n(n = {n if n < 10000 else f'{n:,}'.replace(',', ' ')})", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4), list(GRUPOS.values()), fontsize=7.5)
    ax.set_title(titulo)
    ax.set_ylim(0, 110)
axes[0].set_ylabel("En riesgo académico (%)")
fig.tight_layout()
c.save_figure(fig, "evolucion_alerta_prueba_umbral")
plt.close(fig)

c.save_json({"umbral": tabla.reset_index().to_dict("records"), "misma_lista": tabla_misma.reset_index().to_dict("records"),
             "transiciones": trans.to_dict("records")}, c.ARTIFACTS / "06_umbral_prevalencia.json")
pd.set_option("display.width", 200)
print(tabla.round(3).T.to_string())
print(tabla_misma.round(3).T.to_string())
print(trans.round(1).to_string())
