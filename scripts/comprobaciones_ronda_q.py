"""Comprobaciones de la tercera revisión del oponente (ronda Q, 07-11-2026; E30).

Solo periodos de entrenamiento (2013B, 2013J, 2014B); no abre 2014J ni cambia el modelo.

A2. Entregas tardías: una TMA entregada después de `cutoff_day - RETRASO_CORRECCION` se trata como
    todavía sin nota en el corte, (i) sin contar en la nota media (como hace el sistema con las
    notas que faltan) o (ii) como no entregada (cero). Validación dejando fuera un periodo con la
    configuración del modelo final, como E25.
D4. Percentiles en grupos de 20: los percentiles de la nota media y de la actividad se recalculan
    dentro de grupos aleatorios de 20 estudiantes de cada clase y se aplica el modelo ajustado con
    las clases completas (fuera de pliegue).
D1. Calibración fuera de pliegue: diferencia entre la probabilidad media y la proporción real por
    decil y por clase.
D2. Composición de los estudiantes en riesgo (bajas y desaprobados) y ROC-AUC de cada grupo frente a
    los que aprueban (fuera de pliegue).
F2. Conversión de la escala 2-5: porcentaje de marcados y sensibilidad del modelo final con su umbral
    en los periodos de entrenamiento, con la conversión equipercentil entre aprobados y con los
    valores fijos (50, 70, 90) que usa el sistema mientras no hay cursos anteriores. Sustituye a las
    cifras de marcados de 2014J de `revisions/Davier/2026-11-3/simulaciones/sim_escala2.py` (E29).

Uso (desde la raíz de Proposal): nice -n 19 .venv/bin/python scripts/comprobaciones_ronda_q.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import build_dataset as bd  # noqa: E402
import comun as c  # noqa: E402

RETRASO_CORRECCION = 14  # días entre la entrega y su nota (diez días hábiles, E25)
FEATURES = {cut: m["features"] for cut, m in
            ((cut, joblib.load(ROOT / "artifacts" / f"final_{cut}.joblib")) for cut in c.CUTS)}
FINAL = {cut: joblib.load(ROOT / "artifacts" / f"final_{cut}.joblib")["model"] for cut in c.CUTS}
rng = np.random.default_rng(0)


def nuevo_modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=3000))


def lopo(df, cut):
    """Predicciones fuera de pliegue (un periodo fuera) con la configuración del modelo final."""
    d = df[df.cut == cut].copy()
    d["p"] = np.nan
    for T in c.TRAIN_PERIODS:
        tr, te = d.code_presentation != T, d.code_presentation == T
        m = nuevo_modelo().fit(d.loc[tr, FEATURES[cut]], d.loc[tr, c.TARGET])
        d.loc[te, "p"] = m.predict_proba(d.loc[te, FEATURES[cut]])[:, 1]
    return d


def auc_por_periodo(d, col, signo=1):
    return np.array([roc_auc_score(g[c.TARGET], signo * g[col]) for _, g in d.groupby("code_presentation")])


# --- Datos base (una sola lectura de OULAD) -------------------------------------------------------
students, tmas, submissions, vle = bd.load()
students = students[students.code_presentation != c.TEST_PERIOD]
tmas_tr = tmas[tmas.code_presentation != c.TEST_PERIOD]
base = students.merge(bd.tma_cuts(tmas_tr), on=bd.COURSE)
base = base[~(base["date_unregistration"] <= base["cutoff_day"])].reset_index(drop=True)
vf = bd.vle_features(vle, base)
del vle


def ensambla(af):
    df = (base[bd.ROW + ["anchor_tma", "cutoff_day", "repeated_course", "at_risk", "final_result"]]
          .merge(vf, on=bd.ROW).merge(af, on=bd.ROW))
    for col in bd.VLE:
        df[f"{col}_pct"] = bd.class_percentile(df, col, invert=(col == "days_since_last_activity"))
    for col in bd.GRADES:
        df[f"{col}_raw"] = df[col] / 100
        df[f"{col}_pct"] = bd.class_percentile(df, col)
    df["cut"] = df["cut"].astype(str)
    return df


def notas_tardias(modo):
    """assessment_features con las entregas tardías sin nota en el corte."""
    due = base[bd.ROW + ["cutoff_day", "anchor_tma"]].merge(tmas_tr[bd.COURSE + ["id_assessment", "k"]], on=bd.COURSE)
    due = due[due["k"] <= due["anchor_tma"]]
    due = due.merge(submissions[["id_assessment", "id_student", "date_submitted", "score"]],
                    on=["id_assessment", "id_student"], how="left")
    entregada = due["date_submitted"] <= due["cutoff_day"]
    sin_nota = entregada & (due["date_submitted"] > due["cutoff_day"] - RETRASO_CORRECCION)
    due["submitted"] = entregada if modo == "sin contar" else entregada & ~sin_nota
    due["score0"] = due["score"].where(due["submitted"], 0)
    if modo == "sin contar":
        due.loc[sin_nota, "score0"] = np.nan
    g = due.groupby(bd.ROW)
    f = pd.DataFrame({"submission_rate": g["submitted"].mean(), "mean_score_with_zeros": g["score0"].mean()})
    f["last_score"] = due[due["k"] == due["anchor_tma"]].set_index(bd.ROW)["score0"]
    f = base[bd.ROW].merge(f.reset_index(), on=bd.ROW, how="left")
    med = f.groupby([f[k] for k in bd.CLASS])["mean_score_with_zeros"].transform("median")
    f["mean_score_with_zeros"] = f["mean_score_with_zeros"].fillna(med)
    f["last_score"] = f["last_score"].fillna(f["mean_score_with_zeros"])
    f.loc[f["cut"] == "first", "last_score"] = np.nan
    return f, sin_nota[entregada].mean()


ref = ensambla(bd.assessment_features(tmas_tr, submissions, base))
oof = {cut: lopo(ref, cut) for cut in c.CUTS}

print("== A2. Entregas tardías (ROC-AUC media de los tres periodos: modelo, regla, diferencia)")
for cut in c.CUTS:
    d = oof[cut]
    m, r = auc_por_periodo(d, "p"), auc_por_periodo(d, "mean_score_with_zeros_raw", -1)
    print(f"  {cut:5s} referencia      {m.mean():.3f} {r.mean():.3f} {np.mean(m - r):+.3f}")
for modo in ["sin contar", "como no entregada"]:
    af, prop = notas_tardias(modo)
    df = ensambla(af)
    print(f"  [{modo}] entregas sin nota en el corte: {100 * prop:.1f} % de las entregadas")
    for cut in c.CUTS:
        d = lopo(df, cut)
        m, r = auc_por_periodo(d, "p"), auc_por_periodo(d, "mean_score_with_zeros_raw", -1)
        print(f"  {cut:5s} {modo:17s} {m.mean():.3f} {r.mean():.3f} {np.mean(m - r):+.3f}  "
              f"(cambio del modelo {m.mean() - auc_por_periodo(oof[cut], 'p').mean():+.3f})")

print("\n== D4. Percentiles en grupos aleatorios de 20 estudiantes de cada clase (fuera de pliegue)")
for cut in c.CUTS:
    d = ref[ref.cut == cut].copy()
    d["grupo"] = d.groupby(bd.CLASS).cumcount().map(lambda i: i // 20)
    d["grupo"] = d.groupby(bd.CLASS)["grupo"].transform(lambda s: rng.permutation(s.to_numpy()))
    g20 = d.copy()
    for col, base_col, inv in [("mean_score_with_zeros_pct", "mean_score_with_zeros", False),
                               ("recent_clicks_28d_pct", "recent_clicks_28d", False)]:
        x = g20[base_col].astype(float)
        by = x.groupby([g20[k] for k in bd.CLASS + ["grupo"]])
        g20[col] = by.rank(pct=True, method="average").mask(by.transform("nunique") <= 1, 0.5)
    conj, por_grupo, pierde, n_grupos = [], [], 0, 0
    for T in c.TRAIN_PERIODS:
        tr = d[d.code_presentation != T]
        te = g20[g20.code_presentation == T].copy()
        m = nuevo_modelo().fit(tr[FEATURES[cut]], tr[c.TARGET])
        te["p"] = m.predict_proba(te[FEATURES[cut]])[:, 1]
        conj.append(roc_auc_score(te[c.TARGET], te["p"]))
        for _, g in te.groupby(bd.CLASS + ["grupo"]):
            if g[c.TARGET].nunique() < 2 or len(g) < 20:
                continue
            am, ar = roc_auc_score(g[c.TARGET], g["p"]), roc_auc_score(g[c.TARGET], -g["mean_score_with_zeros_raw"])
            por_grupo.append((am, ar)); n_grupos += 1; pierde += am < ar
    pg = np.array(por_grupo)
    print(f"  {cut:5s} conjunta {auc_por_periodo(oof[cut], 'p').mean():.3f} -> {np.mean(conj):.3f} | "
          f"media por grupo modelo {pg[:, 0].mean():.3f} regla {pg[:, 1].mean():.3f} | modelo por debajo de la "
          f"regla en {100 * pierde / n_grupos:.0f} % de {n_grupos} grupos | percentiles 10-90 del modelo "
          f"{np.percentile(pg[:, 0], 10):.2f}-{np.percentile(pg[:, 0], 90):.2f}")

print("\n== D1. Calibración fuera de pliegue (probabilidad media - proporción real)")
for cut in c.CUTS:
    d = oof[cut]
    dec_min, dec_max, clase = [], [], []
    for T, g in d.groupby("code_presentation"):
        q = pd.qcut(g["p"].rank(method="first"), 10, labels=False)
        diff = g.groupby(q)["p"].mean() - g.groupby(q)[c.TARGET].mean()
        dec_min.append(diff.min()); dec_max.append(diff.max())
        cl = g.groupby(bd.CLASS)
        clase.append((cl["p"].mean() - cl[c.TARGET].mean()).abs().max())
    print(f"  {cut:5s} decil más infraestimado {min(dec_min):+.3f} | más sobrestimado {max(dec_max):+.3f} | "
          f"mayor desviación de una clase {max(clase):.3f} (por periodo {' '.join(f'{x:.3f}' for x in clase)})")

print("\n== D2. Composición de los estudiantes en riesgo y ROC-AUC de cada grupo frente a los que aprueban")
for cut in c.CUTS:
    d = oof[cut]
    riesgo = d[d[c.TARGET] == 1]
    baja = (riesgo["final_result"] == "Withdrawn").mean()
    aprueba = d[d[c.TARGET] == 0]
    out = []
    for nombre, grupo in [("bajas", "Withdrawn"), ("desaprobados", "Fail")]:
        g = pd.concat([aprueba, d[d["final_result"] == grupo]])
        out.append(f"{nombre} {auc_por_periodo(g, 'p').mean():.3f}")
    print(f"  {cut:5s} bajas {100 * baja:.1f} % de los estudiantes en riesgo | ROC-AUC " + " · ".join(out))

print("\n== F2. Conversión de la escala 2-5: marcados y sensibilidad del modelo final con su umbral (entrenamiento)")
orig = submissions["score"].astype(float)
tr_mask = submissions["id_assessment"].isin(tmas_tr["id_assessment"])
ref_notas = orig[tr_mask].dropna()
BANDAS = {"iguales": [40, 60, 80], "exigente": [50, 70, 85]}


def a_categoria(s, cortes):
    return pd.cut(s, [-1, *cortes, 101], right=False, labels=[2, 3, 4, 5]).astype(float)


def equipercentil_aprobados(cat_tr):
    out = {2.0: ref_notas[ref_notas < 40].mean()}
    cc, r = cat_tr[cat_tr >= 3], np.sort(ref_notas[ref_notas >= 40].to_numpy())
    lo = 0.0
    for k, hi in cc.value_counts(normalize=True).sort_index().cumsum().items():
        a, b = int(lo * len(r)), max(int(hi * len(r)), int(lo * len(r)) + 1)
        out[k] = r[a:b].mean(); lo = hi
    return out


def evalua(scores, nombre):
    sub = submissions.copy(); sub["score"] = scores
    df = ensambla(bd.assessment_features(tmas_tr, sub, base))
    fila = []
    for cut in c.CUTS:
        d = df[df.cut == cut]
        p = FINAL[cut].predict_proba(d[FEATURES[cut]])[:, 1]
        thr = d[c.TARGET].mean()
        marc = p > thr
        sens = marc[d[c.TARGET].to_numpy() == 1].mean()
        fila.append(f"{cut} AUC {roc_auc_score(d[c.TARGET], p):.3f} marcados {100 * marc.mean():.1f} % sens {sens:.3f}")
    print(f"  {nombre:42s} " + " | ".join(fila), flush=True)


evalua(orig, "original (0-100)")
for nb, cortes in BANDAS.items():
    cat = a_categoria(orig, cortes)
    m = equipercentil_aprobados(cat[tr_mask].dropna())
    evalua(cat.map(m), f"[{nb}] equipercentil entre aprobados")
    evalua(cat.map({2.0: m[2.0], 3.0: 50.0, 4.0: 70.0, 5.0: 90.0}), f"[{nb}] valores fijos 50/70/90")
    print(f"    equivalencias [{nb}]: " + ", ".join(f"{int(k)} -> {v:.1f}" for k, v in sorted(m.items())))

print("\n== D4/N6. Tamaño de las clases y posición del plazo de la última TMA (entrenamiento)")
for cut in c.CUTS:
    s = ref[ref.cut == cut].groupby(bd.CLASS).size()
    print(f"  {cut:5s} estudiantes por clase: mínimo {s.min()}, mediana {s.median():.0f}, máximo {s.max()}")
pos = tmas_tr.sort_values("date").groupby(bd.COURSE).tail(1)["pct"]
print(f"  plazo de la última TMA: entre el {pos.min():.1f} y el {pos.max():.1f} % del curso, mediana {pos.median():.1f} %")
