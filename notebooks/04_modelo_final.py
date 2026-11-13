# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python (Proposal)
#     language: python
#     name: proposal
# ---

# %% [markdown]
# # 04 · Modelo final
#
# **Pregunta.** ¿Qué variables, qué C y qué umbral tiene el modelo final, el mismo para los tres
# cortes?
#
# **Regla de decisión** (E10). El mismo procedimiento que se validó en el notebook 03, ahora con
# los tres periodos de entrenamiento:
# 1. RF y GB con la configuración común de mayor ROC-AUC media sobre los cortes en la búsqueda
#    dejando fuera un periodo de tres (`artifacts/search_<modelo>_<corte>.joblib`), ajustados con
#    todo el entrenamiento; su importancia por impureza da el orden común de las variables.
# 2. Búsqueda de (k, C) de la logística L2 dejando fuera cada periodo (tres pliegues), media
#    sobre cortes, y regla de un error típico.
# 3. Umbral de cada corte: el que maximiza F2 en las predicciones fuera de pliegue de esa
#    búsqueda. Reajuste con los tres periodos.
#
# Después de este notebook no se cambia nada del modelo. Solo periodos de entrenamiento.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src") if Path.cwd().name == "notebooks" else "src")
import comun as c  # noqa: E402

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import fbeta_score, precision_score, recall_score  # noqa: E402

c.thesis_style()
pd.set_option("display.precision", 3)
pd.set_option("display.width", 160)
data = {cut: c.load_train(cut) for cut in c.CUTS}
Cs = c.GRIDS["lr"]["logisticregression__C"]

# %% [markdown]
# ## 1. Árboles con la configuración común y orden de las variables

# %%
tree_params, imps = {}, {}
for m in ["rf", "gb"]:
    joint = c.joint_table({cut: joblib.load(c.ARTIFACTS / f"search_{m}_{cut}.joblib") for cut in c.CUTS})
    tree_params[m] = joint.loc[0, "params"]
    for cut, d in data.items():
        feats = c.features_for(cut)
        c.thermal_guard()
        model = clone(c.make_model(m)).set_params(**tree_params[m]).fit(d[feats], d[c.TARGET])
        imps[m, cut] = pd.Series(model.feature_importances_, index=feats)
order_s = c.tree_ranking(imps)
order = list(order_s.index)
print(tree_params)
norm = pd.DataFrame({k: v / v.sum() for k, v in imps.items()}).reindex(order)
norm.assign(media=order_s).rename(index=c.LABEL)

# %%
fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.2), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.7]})
y = np.arange(len(order))
w = 0.27
for ax, m, title in [(axes[0], "rf", "Random Forest"), (axes[1], "gb", "Gradient Boosting")]:
    for j, cut in enumerate(c.CUTS):
        ax.barh(y + (j - 1) * w, norm[(m, cut)].fillna(0), height=w, color=c.CUT_COLORS[cut], label=c.CUT_NAMES[cut])
    ax.set_title(title)
    c.comma_axis(ax, "x", 1)
axes[2].barh(y, order_s.values, height=0.6, color="#555555")
axes[2].set_title("Media (orden)")
c.comma_axis(axes[2], "x", 1)
axes[0].set_yticks(y, [f"{i + 1}. {c.LABEL[f]}" for i, f in enumerate(order)], fontsize=7.5)
axes[0].invert_yaxis()
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, ncols=3, loc="upper center", bbox_to_anchor=(0.6, 1.05))
fig.supxlabel("Importancia relativa por reducción de impureza (suma 1 por modelo y corte)", fontsize=9)
fig.tight_layout()
c.save_figure(fig, "importancias_arboles")
plt.show()

# %% [markdown]
# ## 2. Búsqueda de (k, C) con los tres periodos y regla de un error típico

# %%
path = c.ARTIFACTS / "sel_final.joblib"
search = joblib.load(path) if path.exists() else None
if search is None or set(Cs) - set(search["C"]):
    search = c.selection_search(data, order, Cs, previous=search)
    joblib.dump(search, path)
search = search[search["C"].isin(Cs)].reset_index(drop=True)  # grilla de C de E13
choice = c.one_se_choice(search)
K, C = int(choice["k"]), float(choice["C"])
choice

# %%
best_per_k = search.loc[search.groupby("k")["auc"].idxmax()].set_index("k")
best_per_k

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.plot(best_per_k.index, best_per_k["auc"], marker="o", ms=4, color="black", label="ROC-AUC media (mejor C)")
for p, mk in zip(c.TRAIN_PERIODS, ["o", "s", "^"]):
    ax.plot(best_per_k.index, best_per_k[p], ls=":", marker=mk, ms=3, lw=0.8, color="#888888", label=f"Validado en {p}")
ax.axhline(choice["best_auc"] - choice["se"], color="#d9622b", ls="--", lw=1, label="Máximo − 1 error típico")
ax.axvline(K, color="#2a6fdb", lw=1, label=f"Elegido: k = {K}")
ax.set_xticks(best_per_k.index)
ax.set_xlabel("Número de variables (k primeras del orden común)")
ax.set_ylabel("ROC-AUC (media de los cortes)")
c.comma_axis(ax, "y", 3)
ax.legend(fontsize=7, loc="lower right", ncols=2)
fig.tight_layout()
c.save_figure(fig, "seleccion_k")
plt.show()

# %% [markdown]
# ## 3. Umbral F2 y ajuste final

# %%
final, rows = {}, []
for cut, d in data.items():
    feats = c.top_k(order, K, cut)
    model = c.make_model("lr").set_params(logisticregression__C=C)
    oof = c.lopo_oof(model, d, feats)
    thr = c.f2_threshold(d[c.TARGET], oof)
    yhat = oof >= thr
    y = d[c.TARGET]
    rows.append({"corte": cut, "variables": len(feats), "umbral": thr, "sensibilidad (fuera de pliegue)": recall_score(y, yhat),
                 "precisión": precision_score(y, yhat), "F2": fbeta_score(y, yhat, beta=2), "marcados (%)": 100 * yhat.mean()})
    c.thermal_guard()
    fitted = clone(model).fit(d[feats], y)
    final[cut] = {"model": fitted, "features": feats, "C": C, "threshold": thr, "oof": oof}
    joblib.dump(final[cut], c.ARTIFACTS / f"final_{cut}.joblib")
summary = pd.DataFrame(rows).set_index("corte")
summary

# %%
coef = pd.DataFrame({c.CUT_NAMES[cut]: pd.Series(f["model"][-1].coef_[0], index=f["features"]) for cut, f in final.items()}
                    ).reindex(c.top_k(order, K, "last"))
coef.rename(index=c.LABEL)

# %%
c.save_json({"tree_params": tree_params, "order": order, "order_importance": order_s.to_dict(),
             "choice": choice.to_dict(), "summary": summary.reset_index().to_dict("records"),
             "coef": coef.to_dict()}, c.ARTIFACTS / "04_modelo_final.json")

# %% [markdown]
# ## Conclusiones
#
# - **Orden común** (RF con hojas de 50 y √p; GB con 0,02 · profundidad 3 · 300 árboles): la nota
#   media (valor), `last_score` (valor), la nota media (percentil) y la actividad reciente, por
#   este orden; las de VLE salvo la actividad reciente y la recencia pesan poco.
# - **Selección**: el máximo de la búsqueda está en k = 13 (0,890), pero desde k = 4 la curva es
#   plana dentro de un error típico (0,005). **k = 4, C = 0,001.**
# - **Modelo final (E14)**: logística L2 con `mean_score_with_zeros` en valor y en percentil,
#   `last_score` en valor (no en el primer corte) y `recent_clicks_28d`; mismo C en los tres
#   cortes. Umbral F2: 0,213 · 0,235 · 0,263. Fuera de pliegue detecta al 94 % · 90 % · 91 % de
#   los estudiantes en riesgo y marca al 81 % · 55 % · 46 %. Todos los coeficientes, negativos
#   (más nota o más actividad reciente, menos riesgo). No se cambia más.
