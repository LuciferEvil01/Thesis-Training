"""Umbral = proporción en riesgo del entrenamiento, evaluado fuera de pliegue (crítica A3, 06-11-2026).

Para cada periodo de entrenamiento dejado fuera, el modelo final (logística L2, C = 0,001, primeras
variables del orden común) se ajusta con los otros dos y el umbral es su proporción en riesgo. No abre
2014J. Uso: .venv/bin/python scripts/umbral_validacion.py
"""
import json, sys
import numpy as np
sys.path.insert(0, "src")
import comun as c
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

order = json.load(open("artifacts/04_modelo_final.json"))["order"]
for cut in c.CUTS:
    df = c.load_train(cut)
    F = [f for f in order[:4] if not (cut == "first" and f.startswith("last_score"))]  # 3 en el primer corte
    y_all, p_all, flag_all = [], [], []
    for T in c.TRAIN_PERIODS:
        tr, te = df[df.code_presentation != T], df[df.code_presentation == T]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=2000)).fit(tr[F], tr[c.TARGET])
        thr = tr[c.TARGET].mean()
        y_all.append(te[c.TARGET].to_numpy()); flag_all.append(m.predict_proba(te[F])[:, 1] >= thr)
    y, f = np.concatenate(y_all), np.concatenate(flag_all)
    tp, fp, fn, tn = (f & (y == 1)).sum(), (f & (y == 0)).sum(), (~f & (y == 1)).sum(), (~f & (y == 0)).sum()
    print(f"{cut:5s} marcados {f.mean():.3f}  en riesgo {y.mean():.3f}  sensibilidad {tp/(tp+fn):.3f}  "
          f"especificidad {tn/(tn+fp):.3f}  precisión {tp/(tp+fp):.3f}")
