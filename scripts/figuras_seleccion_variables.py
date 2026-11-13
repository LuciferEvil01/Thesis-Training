"""Figuras de la sección 2.1 (selección de variables) y Figura 2.11, con todo el texto en español.

Sustituye a las versiones generadas en «Armando el Dataset/studies», que tenían títulos,
leyendas y categorías en inglés (first/mid/last). Usa el estilo común de la tesis
(`comun.thesis_style`) y solo los periodos de entrenamiento (2013B, 2013J, 2014B). Hasta la
ronda Q (E30) la presencia y el porcentaje de clics por tipo de actividad usaban todo OULAD.

Uso (desde la raíz de Proposal):
    .venv/bin/python scripts/figuras_seleccion_variables.py
Las figuras se guardan en figuras/FeatureSelection/ y figuras/Training/.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import comun as c  # noqa: E402

RAW = ROOT / "data" / "raw" / "oulad"
OUT = ROOT / "figuras" / "FeatureSelection"
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ["id_student", "code_module", "code_presentation"]
SHORT = {"first": "Primer corte", "mid": "Segundo corte", "last": "Tercer corte"}

c.thesis_style()
d = pd.read_parquet(c.DATASET)
d["cut"] = d["cut"].astype(str)
train = d[d.code_presentation != c.TEST_PERIOD].copy()
courses = pd.read_csv(RAW / "courses.csv")


# Tipos de actividad de OULAD en español (R-06); el nombre original entre paréntesis solo en la
# figura de presencia por asignatura (O2-25: la Figura 2.2 de la tesis va sin él).
TIPOS = {"oucontent": "Contenido del curso", "forumng": "Foro", "quiz": "Cuestionario",
         "homepage": "Página de inicio", "subpage": "Subpágina", "resource": "Recurso",
         "ouwiki": "Wiki", "url": "Enlace externo", "oucollaborate": "Sala de colaboración",
         "glossary": "Glosario", "questionnaire": "Encuesta", "externalquiz": "Cuestionario externo",
         "page": "Página", "dataplus": "Base de datos", "ouelluminate": "Aula virtual",
         "dualpane": "Panel doble", "htmlactivity": "Actividad HTML", "folder": "Carpeta",
         "sharedsubpage": "Subpágina compartida", "repeatactivity": "Actividad repetida"}


def tipo(t):
    return f"{TIPOS.get(t, t)} ({t})"


def guardar(fig, nombre):
    fig.savefig(OUT / f"{nombre}.pdf")
    plt.close(fig)
    print("guardada", nombre)


# --- Figura 2.1: posición de los cortes ---------------------------------------------------
x = (train.drop_duplicates(["code_module", "code_presentation", "cut"])
     .merge(courses, on=["code_module", "code_presentation"]))
x["pct"] = 100 * x.cutoff_day / x.module_presentation_length
pres = sorted(x[["code_module", "code_presentation"]].drop_duplicates().itertuples(index=False))
fig, ax = plt.subplots(figsize=(6.3, 4.2))
ypos = {p: i for i, p in enumerate(reversed(pres))}
for cut in c.CUTS:
    s = x[x.cut == cut]
    ax.scatter(s.pct, [ypos[(m, p)] for m, p in zip(s.code_module, s.code_presentation)],
               s=22, color=c.CUT_COLORS[cut], label=SHORT[cut], zorder=3)
for v in (25, 50, 75):
    ax.axvline(v, color="#9a9a9a", lw=0.8, ls=(0, (2, 2)), zorder=1)
ax.set_yticks(list(ypos.values()), [f"{m} {p}" for m, p in ypos])
ax.tick_params(axis="y", labelsize=8)
ax.set_xlim(0, 100)
ax.set_xlabel("Porcentaje del curso transcurrido")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower right", frameon=True, framealpha=1, edgecolor="#e3e3e3")
guardar(fig, "assessment_cuts")

# --- Figuras 2.2 y 2.3: tipos de actividad ------------------------------------------------
vle = pd.read_csv(RAW / "vle.csv")
vle = vle[vle.code_presentation != c.TEST_PERIOD]  # solo entrenamiento (E30)
mat = vle.groupby(["activity_type", "code_module"]).id_site.nunique().unstack()
n_mod = mat.notna().sum(axis=1)
orden = mat.assign(n=n_mod, tot=mat.sum(axis=1)).sort_values(["n", "tot"], ascending=False).index
mat, n_mod = mat.loc[orden], n_mod.loc[orden]
fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.imshow(np.log1p(mat.fillna(0)), cmap="Blues", aspect="auto", vmin=0)
ax.set_xticks(range(mat.shape[1]), mat.columns)
ax.set_yticks(range(mat.shape[0]), [f"{tipo(t)}, {n}" for t, n in zip(mat.index, n_mod)])
ax.tick_params(labelsize=8)
ax.set_xlabel("Asignatura")
ax.set_ylabel("Tipo de actividad (asignaturas que lo usan)")
ax.grid(False)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.iloc[i, j]
        if not pd.isna(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 60 else "#1a1a1a")
guardar(fig, "activity_presence")

sv = pd.read_csv(RAW / "studentVle.csv", usecols=["code_presentation", "id_site", "sum_click"])
sv = sv[sv.code_presentation != c.TEST_PERIOD].drop(columns="code_presentation")
share = (sv.merge(vle[["id_site", "activity_type"]], on="id_site").groupby("activity_type").sum_click.sum()
         .sort_values(ascending=False).pipe(lambda s: 100 * s / s.sum()))
print(f"Tipos: {len(n_mod)}; en las siete asignaturas {int((n_mod == 7).sum())}; en una o dos "
      f"{int((n_mod <= 2).sum())}; cinco más usados {share.iloc[:5].sum():.1f} %; por debajo del 0,3 % "
      f"{int((share < 0.3).sum())}")
sva = pd.read_csv(RAW / "studentVle.csv", usecols=["code_module", "code_presentation", "id_site", "sum_click"])
sva = sva[sva.code_presentation != c.TEST_PERIOD].merge(vle[["id_site", "activity_type"]], on="id_site")
por_tipo = sva.groupby(["code_module", "activity_type"]).sum_click.sum().unstack(fill_value=0)
evaluacion = 100 * por_tipo.reindex(columns=["quiz", "externalquiz", "questionnaire"], fill_value=0).sum(axis=1) \
    / por_tipo.sum(axis=1)
print("Clics en actividades de evaluación (cuestionarios, cuestionarios externos y encuestas) por asignatura, %:",
      evaluacion.round(1).to_dict())
del sva


def pct(v):
    if round(v, 1) >= 0.1:  # 0,095 se escribe 0,1 % y no 0,10 %
        return f"{c.comma(v, 1)} %"
    if v >= 0.01:
        return f"{c.comma(v, 2)} %"
    return "< 0,01 %"


fig, ax = plt.subplots(figsize=(5.6, 4.6))
y = list(range(len(share)))[::-1]
ax.barh(y, share.values, height=0.6, color=c.CUT_COLORS["first"])
ax.set_yticks(y, [TIPOS.get(t, t) for t in share.index], fontsize=7.5)
ax.tick_params(axis="y", labelsize=8)
for yi, v in zip(y, share.values):
    ax.text(v + 0.4, yi, pct(v), va="center", fontsize=7.5, color="#444444")
ax.axhline(len(share) - 5.5, color="#666666", lw=0.8, ls=(0, (3, 3)))
ax.text(31, len(share) - 6.0, f"los cinco tipos más usados reúnen el {c.comma(share.iloc[:5].sum(), 1)} % de los clics",
        ha="right", va="top", fontsize=8)
ax.set_xlim(0, 32)
ax.set_xlabel("Porcentaje del total de clics de los periodos de entrenamiento")
ax.grid(axis="y", visible=False)
c.comma_axis(ax, "x", 0)
guardar(fig, "activity_click_share")

# --- Figura 2.4: dependencia de la asignatura ---------------------------------------------
SHOW = c.VLE
mid = train[train.cut == "mid"]
med = mid.groupby("code_module")[SHOW].median()
ratio = med / mid[SHOW].median()
fig, ax = plt.subplots(figsize=(6.6, 3.2))
im = ax.imshow(np.log2(ratio.T), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(med)), med.index)
ax.set_yticks(range(len(SHOW)), [s.replace("_", r"\_") if False else s for s in SHOW])
ax.tick_params(labelsize=8)
ax.set_xlabel("Asignatura")
ax.grid(False)
for i, f in enumerate(SHOW):
    for j, m in enumerate(med.index):
        ax.text(j, i, f"{med.loc[m, f]:.0f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(np.log2(ratio.loc[m, f])) > 1.2 else "#1a1a1a")
cb = fig.colorbar(im, ax=ax, ticks=[-2, -1, 0, 1, 2])
cb.ax.set_yticklabels(["÷4", "÷2", "=", "×2", "×4"])
cb.set_label("Mediana de la asignatura /\nmediana global", fontsize=8)
guardar(fig, "vle_course_dependence")

# --- Figura 2.5: nota de la última TMA frente a la nota media ------------------------------
ass = pd.read_csv(RAW / "assessments.csv", na_values="?")
tma = ass[ass.assessment_type == "TMA"].sort_values(["code_module", "code_presentation", "date"])
tma["k"] = tma.groupby(["code_module", "code_presentation"]).cumcount() + 1
sa = pd.read_csv(RAW / "studentAssessment.csv", na_values="?")
rows = train[KEYS + ["cut", "anchor_tma", "cutoff_day", "mean_score_with_zeros_pct"]]
last = (rows.merge(tma[["code_module", "code_presentation", "id_assessment", "k"]],
                   left_on=["code_module", "code_presentation", "anchor_tma"],
                   right_on=["code_module", "code_presentation", "k"])
        .merge(sa[["id_assessment", "id_student", "date_submitted", "score"]],
               on=["id_assessment", "id_student"], how="left"))
last["ultima"] = last.score.where(last.date_submitted <= last.cutoff_day, 0).fillna(0)
last["last_pct"] = last.groupby(["code_module", "code_presentation", "cut"])["ultima"].rank(pct=True)
rho = {cut: last[last.cut == cut][["last_pct", "mean_score_with_zeros_pct"]].corr("spearman").iloc[0, 1]
       for cut in c.CUTS}
print("Spearman:", {k: round(v, 3) for k, v in rho.items()})
fig, ax = plt.subplots(figsize=(3.6, 2.6))
bars = ax.bar([SHORT[k] for k in c.CUTS], [rho[k] for k in c.CUTS], width=0.55,
              color=[c.CUT_COLORS[k] for k in c.CUTS])
for b, k in zip(bars, c.CUTS):
    ax.text(b.get_x() + b.get_width() / 2, rho[k] + 0.015, c.comma(rho[k], 2), ha="center", fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Correlación de Spearman")
ax.tick_params(axis="x", labelsize=8.5)
ax.grid(axis="x", visible=False)
c.comma_axis(ax, "y", 1)
guardar(fig, "assessment_last_vs_mean")

# --- Figura 2.11: regularización del Random Forest -----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6))
etiquetas = {"sqrt": r"$\sqrt{p}$ variables por división", 0.5: r"$0{,}5\,p$ variables por división"}
for ax, cut in zip(axes, c.CUTS):
    for mf, ls in [("sqrt", "-"), (0.5, "--")]:
        curvas = []
        for p in c.TRAIN_PERIODS:
            inner = joblib.load(c.ARTIFACTS / f"inner_rf_{cut}_{p}.joblib")
            sel = inner[inner["params"].map(lambda q: q["max_features"] == mf)]
            curvas.append(sel.set_index(sel["params"].map(lambda q: q["min_samples_leaf"]))["auc"].sort_index())
        curva = pd.concat(curvas, axis=1).mean(axis=1)
        ax.plot(range(len(curva)), curva.values, ls, marker="o", ms=3, color=c.CUT_COLORS[cut],
                label=etiquetas[mf])
    hojas = c.GRIDS["rf"]["min_samples_leaf"]
    ax.set_xticks(range(len(hojas)), [str(v) for v in hojas], fontsize=7)
    ax.set_xlabel("Tamaño mínimo de hoja", fontsize=9)
    ax.set_title(c.CUT_NAMES[cut])
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, steps=[1, 2, 5, 10]))
    c.comma_axis(ax, "y", 3)
axes[0].set_ylabel("ROC-AUC (validación interna)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.08), handlelength=3)
fig.tight_layout()
c.save_figure(fig, "regularizacion_rf")
print("guardada regularizacion_rf")
