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
# # 03 · Selección de variables, umbral y elección del modelo
#
# **Pregunta.** Con la selección de variables hecha por los árboles dentro de la validación,
# ¿cuántas variables necesita la regresión logística y cuánto discrimina en un periodo que no ha
# visto? ¿La supera algún modelo de árboles de forma consistente? ¿Qué sensibilidad y precisión
# cabe esperar con el umbral F2?
#
# **Protocolo** (E10), en cada pliegue externo y solo con sus dos periodos de entrenamiento:
# 1. Orden común de las variables: importancia por impureza del RF y del GB del pliegue
#    (notebook 02), normalizada y promediada sobre modelos y cortes (`comun.tree_ranking`).
# 2. Búsqueda de (k, C) de la logística L2 con las k primeras variables, dejando fuera cada uno
#    de los dos periodos, con la ROC-AUC media sobre los tres cortes (`comun.selection_search`).
#    Regla de un error típico (`comun.one_se_choice`).
# 3. Umbral de cada corte: el que maximiza F2 en las predicciones fuera de pliegue de esos dos
#    periodos (`comun.f2_threshold`). Se reajusta y se predice el periodo externo.
#
# **Regla de decisión D9** (sin cambios): el modelo final es la logística con su selección,
# salvo que el RF o el GB la superen en los tres cortes con (1) IC del 95 % de la diferencia
# pareada por encima de 0 y (2) diferencia mayor que la desviación entre periodos de la
# logística. Si un modelo de árboles cumpliera, se para y se pregunta a Jose (E10).
#
# Solo periodos de entrenamiento (2013B, 2013J, 2014B).

# %%
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src") if Path.cwd().name == "notebooks" else "src")
import comun as c  # noqa: E402

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score  # noqa: E402

c.thesis_style()
pd.set_option("display.precision", 3)
pd.set_option("display.width", 160)
data = {cut: c.load_train(cut) for cut in c.CUTS}
res = {m: c.nested_cv_joint(m, data) for m in c.GRIDS}
Cs = c.GRIDS["lr"]["logisticregression__C"]

# %% [markdown]
# ## 1. Orden de las variables en cada pliegue externo

# %%
orders = {}
for p in c.TRAIN_PERIODS:
    imps = {(m, cut): next(f["importances"] for f in res[m][cut]["folds"] if f["period"] == p)
            for m in ["rf", "gb"] for cut in c.CUTS}
    orders[p] = c.tree_ranking(imps)
pd.DataFrame({f"sin {p}": o.rank(ascending=False).astype(int) for p, o in orders.items()}).rename(index=c.LABEL)

# %% [markdown]
# ## 2. Selección de (k, C), umbral y predicción del periodo externo
#
# Cada búsqueda se guarda en `artifacts/sel_inner_<periodo>.joblib`.

# %%
oof = {cut: np.full(len(data[cut]), np.nan) for cut in c.CUTS}
labels = {cut: np.full(len(data[cut]), -1) for cut in c.CUTS}
folds = []
for p in c.TRAIN_PERIODS:
    train = {cut: d[d["code_presentation"] != p].reset_index(drop=True) for cut, d in data.items()}
    order = list(orders[p].index)
    path = c.ARTIFACTS / f"sel_inner_{p}.joblib"
    table = joblib.load(path) if path.exists() else None
    if table is None or set(Cs) - set(table["C"]):  # primera búsqueda, o grilla ampliada
        t = time.time()
        table = c.selection_search(train, order, Cs, previous=table)
        joblib.dump(table, path)
        print(f"sin {p}: {time.time() - t:5.0f} s", flush=True)
    table = table[table["C"].isin(Cs)].reset_index(drop=True)  # grilla de C de E13
    choice = c.one_se_choice(table)
    k, C = int(choice["k"]), float(choice["C"])
    for cut, d in data.items():
        feats = c.top_k(order, k, cut)
        model = c.make_model("lr").set_params(logisticregression__C=C)
        thr = c.f2_threshold(train[cut][c.TARGET], c.lopo_oof(model, train[cut], feats))
        tr, va = (d["code_presentation"] != p).to_numpy(), (d["code_presentation"] == p).to_numpy()
        fitted = model.fit(d.loc[tr, feats], d.loc[tr, c.TARGET])
        oof[cut][va] = fitted.predict_proba(d.loc[va, feats])[:, 1]
        labels[cut][va] = oof[cut][va] >= thr
        y = d.loc[va, c.TARGET]
        folds.append({"periodo fuera": p, "corte": cut, "k": k, "C": C, "variables": feats, "umbral": thr,
                      "auc": roc_auc_score(y, oof[cut][va]),
                      "auc_train": roc_auc_score(d.loc[tr, c.TARGET], fitted.predict_proba(d.loc[tr, feats])[:, 1]),
                      "sensibilidad": recall_score(y, labels[cut][va]), "precisión": precision_score(y, labels[cut][va]),
                      "F2": fbeta_score(y, labels[cut][va], beta=2), "marcados (%)": 100 * labels[cut][va].mean(),
                      "k_max": int(choice["best_k"]), "auc_max": choice["best_auc"], "auc_k": choice["auc"], "se": choice["se"]})
folds = pd.DataFrame(folds)
folds[["periodo fuera", "corte", "k", "C", "k_max", "auc_max", "auc_k", "se", "umbral", "auc", "auc_train"]]

# %%
folds.groupby("periodo fuera").first()["variables"].map(lambda v: [c.LABEL[f] for f in v]).to_frame()

# %% [markdown]
# ## 3. ROC-AUC en el periodo dejado fuera: logística con selección frente al resto

# %%
def baseline_fold_aucs(d):
    return {p: roc_auc_score(g[c.TARGET], c.baseline_score(g)) for p, g in d.groupby("code_presentation")}


def row(cut, name, per):
    vals = list(per.values())
    return {"corte": cut, "modelo": name, **per, "media": np.mean(vals), "desv.": np.std(vals, ddof=1)}


rows = []
for cut in c.CUTS:
    f = folds[folds["corte"] == cut]
    rows.append(row(cut, "lr_sel", dict(zip(f["periodo fuera"], f["auc"]))) | {"entrenamiento": f["auc_train"].mean()})
    for m in ["lr", "rf", "gb"]:
        fl = res[m][cut]["folds"]
        rows.append(row(cut, m, {x["period"]: x["auc_val"] for x in fl}) | {"entrenamiento": np.mean([x["auc_train"] for x in fl])})
    rows.append(row(cut, "nota", baseline_fold_aucs(data[cut])))
table = pd.DataFrame(rows).set_index(["corte", "modelo"])
table

# %%
SHOW = ["lr_sel", "rf", "gb", "nota"]
short = {"lr_sel": "Log.", "rf": "RF", "gb": "GB", "nota": "Nota"}
markers = {"2013B": "o", "2013J": "s", "2014B": "^"}
fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.0), sharey=True)
for ax, cut in zip(axes, c.CUTS):
    for i, m in enumerate(SHOW):
        r = table.loc[(cut, m)]
        for p in c.TRAIN_PERIODS:
            ax.scatter(i, r[p], marker=markers[p], s=22, color=c.CUT_COLORS[cut], alpha=0.7, zorder=3)
        ax.hlines(r["media"], i - 0.28, i + 0.28, color="black", lw=1.4, zorder=4)
    ax.set_xticks(range(len(SHOW)), [short[m] for m in SHOW], fontsize=8)
    ax.set_xlim(-0.6, len(SHOW) - 0.4)
    ax.set_title(c.CUT_NAMES[cut])
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    c.comma_axis(ax, "y", 2)
axes[0].set_ylabel("ROC-AUC en el periodo dejado fuera")
handles = [plt.Line2D([], [], ls="", marker=markers[p], color="#555555", label=p) for p in c.TRAIN_PERIODS]
handles.append(plt.Line2D([], [], color="black", lw=1.4, label="Media"))
fig.legend(handles=handles, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.07))
fig.tight_layout()
c.save_figure(fig, "resultados_validacion")
plt.show()

# %% [markdown]
# ## 4. Diferencias pareadas y regla D9
#
# Estadístico: la media de la ROC-AUC de cada periodo externo (la de la tabla), con bootstrap de
# estudiantes dentro de cada clase (E13).

# %%
boot = {}
for cut in c.CUTS:
    d = data[cut]
    scores = {"lr_sel": oof[cut], "lr": res["lr"][cut]["oof"], "rf": res["rf"][cut]["oof"],
              "gb": res["gb"][cut]["oof"], "nota": c.baseline_score(d)}
    boot[cut] = c.bootstrap_auc_by_period(d[c.TARGET], scores, c.class_id(d), d["code_presentation"], reference="lr_sel")
pd.concat(boot, names=["corte", "modelo"])

# %%
d9 = []
for cut in c.CUTS:
    sd = table.loc[(cut, "lr_sel"), "desv."]
    for m in ["rf", "gb"]:
        b = boot[cut].loc[m]
        d9.append({"corte": cut, "modelo": m, "dif": b["dif"], "dif_inf": b["dif_inf"], "dif_sup": b["dif_sup"],
                   "desv. logística": sd, "(1) IC > 0": b["dif_inf"] > 0, "(2) dif > desv.": b["dif"] > sd})
d9 = pd.DataFrame(d9)
d9["cumple"] = d9["(1) IC > 0"] & d9["(2) dif > desv."]
d9

# %%
winners = [m for m in ["rf", "gb"] if d9.loc[d9["modelo"] == m, "cumple"].all()]
print("Modelos de árboles que cumplen D9 en los tres cortes:", winners or "ninguno")
print("Modelo final:", "regresión logística con selección de variables" if not winners else "PARAR Y PREGUNTAR A JOSE")

# %%
# Frente a la regla de la nota (logística con selección − nota) y frente a la logística con todas
vs = pd.DataFrame({cut: {"lr_sel − nota": -boot[cut].loc["nota", "dif"],
                         "IC inf": -boot[cut].loc["nota", "dif_sup"], "IC sup": -boot[cut].loc["nota", "dif_inf"],
                         "lr (todas) − lr_sel": boot[cut].loc["lr", "dif"],
                         "IC inf ": boot[cut].loc["lr", "dif_inf"], "IC sup ": boot[cut].loc["lr", "dif_sup"]}
                   for cut in c.CUTS}).T
vs

# %% [markdown]
# ## 5. El umbral F2 en el periodo dejado fuera

# %%
thr_table = folds.groupby("corte")[["umbral", "sensibilidad", "precisión", "F2", "marcados (%)"]].agg(["mean", "std"]).reindex(c.CUTS)
thr_table

# %%
c.save_json({"table": table.reset_index().to_dict("records"), "folds": folds.drop(columns="variables").to_dict("records"),
             "variables": {p: v for p, v in folds.groupby("periodo fuera").first()["variables"].items()},
             "orders": {p: list(o.index) for p, o in orders.items()},
             "d9": d9.to_dict("records"), "winners": winners, "vs": vs.to_dict(),
             "umbral": thr_table.set_axis([f"{a} {b}" for a, b in thr_table.columns], axis=1).to_dict()},
            c.ARTIFACTS / "03_seleccion.json")

# %% [markdown]
# ## Conclusiones
#
# - **Orden de las variables** muy estable entre pliegues: `mean_score_with_zeros` (valor) primera
#   siempre; luego su percentil, `last_score` (valor) y `recent_clicks_28d`; `repeated_course`,
#   `clicks_per_active_day` y `pre_course_clicks`, últimas.
# - **Selección**: la regla de un error típico elige k = 5, 4, 4 (el máximo estaba en 5, 5, 10),
#   con C = 0,001, 0,001, 0,03. La grilla de C se cerró en 0,001 (E13).
# - **ROC-AUC externa** de la logística con selección: 0,817 · 0,906 · 0,938; frente a la logística
#   con todas, 0,005 · 0,002 · 0,002 menos (IC excluye el 0, pero menor que la desviación entre
#   periodos): con 4–5 variables rinde de forma equivalente.
# - **D9** (por periodo, E13): RF y GB la superan por 0,005–0,008; solo en el tercer corte supera
#   a la desviación entre periodos. **Modelo final: regresión logística con selección (E14).**
# - **Frente a la regla de la nota**: +0,016 · +0,013 · +0,008, con IC por encima de 0.
# - **Umbral F2** en el periodo externo: sensibilidad 0,93 · 0,90 · 0,90, precisión 0,53 · 0,66 ·
#   0,71; marca al 78 % · 54 % · 45 % de los estudiantes (prevalencia 32–46 %).
