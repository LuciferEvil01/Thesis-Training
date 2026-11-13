"""¿Depende el resultado de los catorce días entre el plazo de la TMA y su nota? (crítica D10, 07-11-2026)

Reconstruye el conjunto con otros retrasos (sin tocar data/processed/dataset.parquet) y repite la
validación dejando fuera un periodo con la configuración del modelo final y la regla de la nota.
Solo periodos de entrenamiento. Uso: .venv/bin/python scripts/sensibilidad_retraso.py
"""
import json, sys
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import build_dataset as bd
import comun as c
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

order = json.load(open("artifacts/04_modelo_final.json"))["order"]
for lag in [7, 14, 21, 28]:
    bd.GRADE_LAG = lag
    df = bd.build()
    df["cut"] = df["cut"].astype(str)
    df = df[df.code_presentation != c.TEST_PERIOD]
    out = []
    for cut in c.CUTS:
        d = df[df.cut == cut]
        F = [f for f in order[:4] if not (cut == "first" and f.startswith("last_score"))]
        m_auc, r_auc = [], []
        for T in c.TRAIN_PERIODS:
            tr, te = d[d.code_presentation != T], d[d.code_presentation == T]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=3000)).fit(tr[F], tr[c.TARGET])
            m_auc.append(roc_auc_score(te[c.TARGET], m.predict_proba(te[F])[:, 1]))
            r_auc.append(roc_auc_score(te[c.TARGET], -te["mean_score_with_zeros_raw"]))
        out.append(f"{cut}: n={len(d)} modelo {np.mean(m_auc):.3f} regla {np.mean(r_auc):.3f} dif {np.mean(m_auc)-np.mean(r_auc):.3f}")
    print(f"retraso {lag:2d} días | " + " | ".join(out), flush=True)
