"""Descomposición de la ventaja del modelo sobre la regla (crítica B4, 06-11-2026).

Validación dejando fuera un periodo, configuración del modelo final (logística L2, C = 0,001).
No abre 2014J. Uso: .venv/bin/python scripts/aporte_actividad.py
"""
import sys
import numpy as np
sys.path.insert(0, "src")
import comun as c
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SETS = {
    "regla (nota media)": None,
    "solo notas del modelo": ["mean_score_with_zeros_raw", "last_score_raw", "mean_score_with_zeros_pct"],
    "modelo final": ["mean_score_with_zeros_raw", "last_score_raw", "mean_score_with_zeros_pct", "recent_clicks_28d_pct"],
    "solo actividad del modelo": ["recent_clicks_28d_pct"],
    "las siete de actividad": c.VLE_PCT,
}
for cut in c.CUTS:
    df = c.load_train(cut)
    res = {}
    for name, F in SETS.items():
        aucs = []
        for T in c.TRAIN_PERIODS:
            tr, te = df[df.code_presentation != T], df[df.code_presentation == T]
            if F is None:
                aucs.append(roc_auc_score(te[c.TARGET], -te["mean_score_with_zeros_raw"]))
                continue
            F2 = [f for f in F if not (cut == "first" and f.startswith("last_score"))]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=2000)).fit(tr[F2], tr[c.TARGET])
            aucs.append(roc_auc_score(te[c.TARGET], m.predict_proba(te[F2])[:, 1]))
        res[name] = np.array(aucs)
    r, g, f = res["regla (nota media)"], res["solo notas del modelo"], res["modelo final"]
    print(f"== {cut}")
    for k, v in res.items():
        print(f"  {k:28s} {v.mean():.3f}  (por periodo {' '.join(f'{x:.3f}' for x in v)})")
    print(f"  ventaja total sobre la regla      {np.mean(f - r):.3f}  por periodo {' '.join(f'{x:.3f}' for x in f - r)}")
    print(f"  de ella, recombinar las notas     {np.mean(g - r):.3f}  por periodo {' '.join(f'{x:.3f}' for x in g - r)}")
    print(f"  de ella, añadir la actividad      {np.mean(f - g):.3f}  por periodo {' '.join(f'{x:.3f}' for x in f - g)}")
