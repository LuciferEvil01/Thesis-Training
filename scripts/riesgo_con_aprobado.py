"""Estudiantes en riesgo que todavía aprueban en el corte: cuántos marca el modelo (crítica B5, 06-11-2026).

Validación dejando fuera un periodo, configuración del modelo final y umbral = proporción en riesgo de
los periodos de ajuste. La regla de la nota no marca a nadie con media >= 0,40 (40 en la escala de OULAD; las notas están en [0, 1]). No abre 2014J.
Uso: .venv/bin/python scripts/riesgo_con_aprobado.py
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
    rows = []
    for T in c.TRAIN_PERIODS:
        tr, te = df[df.code_presentation != T], df[df.code_presentation == T]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=2000)).fit(tr[F], tr[c.TARGET])
        flag = m.predict_proba(te[F])[:, 1] >= tr[c.TARGET].mean()
        rows.append((te[c.TARGET].to_numpy(), te["mean_score_with_zeros_raw"].to_numpy() >= 0.40, flag))
    y = np.concatenate([r[0] for r in rows]); ok = np.concatenate([r[1] for r in rows]); f = np.concatenate([r[2] for r in rows])
    risk = y == 1
    print(f"{cut:5s} en riesgo: {risk.sum()}; de ellos con media aprobada: {(risk & ok).sum()} "
          f"({(risk & ok).sum() / risk.sum():.1%}); el modelo marca a {(risk & ok & f).sum()} "
          f"({(risk & ok & f).sum() / (risk & ok).sum():.1%}) | aprueban con media aprobada: {(~risk & ok).sum()}, "
          f"marcados {(~risk & ok & f).sum() / (~risk & ok).sum():.1%} | precisión entre los de media aprobada: "
          f"{(risk & ok & f).sum() / (ok & f).sum():.3f}")
