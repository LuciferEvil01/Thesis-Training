"""Críticas C8 (regla de un error típico con tres periodos) y C9 (C en el borde, coeficientes y
explicaciones). Solo periodos de entrenamiento. Uso: .venv/bin/python scripts/penalizacion_y_regla.py
"""
import json, sys
import numpy as np
from scipy.stats import chi2, spearmanr
sys.path.insert(0, "src")
import comun as c
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

order = json.load(open("artifacts/04_modelo_final.json"))["order"]
DATA = {cut: c.load_train(cut) for cut in c.CUTS}
def feats(cut, k): return [f for f in order[:k] if not (cut == "first" and f.startswith("last_score"))]  # como la tesis: k puestos del orden común
def lr(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
def per_period(k, C):
    per = np.zeros(3)
    for cut in c.CUTS:
        df, F = DATA[cut], feats(cut, k)
        for i, T in enumerate(c.TRAIN_PERIODS):
            tr, te = df[df.code_presentation != T], df[df.code_presentation == T]
            per[i] += roc_auc_score(te[c.TARGET], lr(C).fit(tr[F], tr[c.TARGET]).predict_proba(te[F])[:, 1]) / 3
    return per

print("== C8: regla de un error típico")
lo, hi = np.sqrt(2 / chi2.ppf(0.975, 2)), np.sqrt(2 / chi2.ppf(0.025, 2))
print(f"  IC 95 % de la desviación verdadera con 3 valores: [{lo:.2f}; {hi:.2f}] veces la estimada")
CS = [0.001, 0.01, 0.1, 1, 10, 100]
rows = []
for k in range(1, 14):
    best = max(((C, per_period(k, C)) for C in CS), key=lambda t: t[1].mean())
    rows.append((k, best[0], best[1].mean(), best[1].std(ddof=1) / np.sqrt(3)))
    print(f"  k={k:2d} C={best[0]:<6} AUC {rows[-1][2]:.4f}  et {rows[-1][3]:.4f}")
top = max(rows, key=lambda r: r[2])
for mult in [0.25, 0.5, 1, 2, 3, 6]:
    k = min(r[0] for r in rows if r[2] >= top[2] - mult * top[3])
    print(f"  error típico ×{mult}: k = {k}")

print("\n== C9: ROC-AUC del modelo final (k = 4) según C")
for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
    print(f"  C={C:<7g} AUC {per_period(4, C).mean():.4f}")
print("\n== C9: coeficientes (ajuste con los tres periodos) según C")
for cut in c.CUTS:
    df, F = DATA[cut], feats(cut, 4)
    print(f"  {cut}: correlación valor-percentil de la nota media {df['mean_score_with_zeros_raw'].corr(df['mean_score_with_zeros_pct']):.3f}")
    base = None
    for C in [0.001, 1, 100]:
        m = lr(C).fit(df[F], df[c.TARGET]); b = m[-1].coef_[0]
        Z = m[0].transform(df[F]); contrib = Z * b
        top_var = contrib.argmax(axis=1)  # variable que más aumenta el riesgo de cada estudiante
        if base is None: base = top_var
        print(f"    C={C:<6g} coef " + " ".join(f"{f.replace('_with_zeros','').replace('_28d','')}={x:+.3f}" for f, x in zip(F, b))
              + f" | misma variable principal que con C=0,001: {np.mean(top_var == base):.1%}")
