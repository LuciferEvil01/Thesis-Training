"""Comprobación de la intercambiabilidad de los periodos (crítica A1 del oponente, 06-11-2026).

Solo periodos de entrenamiento; no abre 2014J. Uso: .venv/bin/python scripts/validacion_temporal.py
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import comun as c
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
spec = json.load(open("artifacts/04_modelo_final.json"))
order = spec["order"]
def feats(cut, k): return [f for f in order[:k] if not (cut == "first" and f.startswith("last_score"))]  # como la tesis: k puestos del orden común
def model(C=0.001): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
rng = np.random.default_rng(42)
R = 50
print("== 1-2. Un periodo fuente -> un periodo objetivo, mismo tamaño (submuestreo, 50 repeticiones)")
pairs = {"2013B": ["2013J", "2014B"], "2013J": ["2013B", "2014B"], "2014B": ["2013B", "2013J"]}
for cut in c.CUTS:
    df = c.load_train(cut); F = feats(cut, 4)
    n = min((df.code_presentation == p).sum() for p in c.TRAIN_PERIODS)
    for T, S in pairs.items():
        te = df[df.code_presentation == T]
        out = []
        for s in S:
            tr = df[df.code_presentation == s]
            aucs = []
            for r in range(R):
                sub = tr.sample(n=n, random_state=int(rng.integers(1e9)))
                m = model().fit(sub[F], sub[c.TARGET])
                aucs.append(roc_auc_score(te[c.TARGET], m.predict_proba(te[F])[:, 1]))
            rel = "pasado" if s < T else "futuro"
            out.append(f"{s} ({rel}, tipo {s[-1]}): {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        # común: solo asignaturas presentes en fuente y objetivo
        print(f"  {cut:5s} objetivo {T}: " + " | ".join(out))
print("\n== 2b. Igual, solo con las asignaturas comunes a fuente y objetivo")
for cut in c.CUTS:
    df = c.load_train(cut); F = feats(cut, 4)
    for T, S in pairs.items():
        out = []
        for s in S:
            mods = set(df[df.code_presentation == T].code_module) & set(df[df.code_presentation == s].code_module)
            te = df[(df.code_presentation == T) & df.code_module.isin(mods)]
            tr = df[(df.code_presentation == s) & df.code_module.isin(mods)]
            m = model().fit(tr[F], tr[c.TARGET])
            out.append(f"{s}: {roc_auc_score(te[c.TARGET], m.predict_proba(te[F])[:,1]):.4f} (mód. {''.join(sorted(x[0] for x in mods))})")
        print(f"  {cut:5s} objetivo {T}: " + " | ".join(out))
print("\n== 3. Dejar fuera un periodo (tesis) frente a validación hacia delante, modelo final")
for cut in c.CUTS:
    df = c.load_train(cut); F = feats(cut, 4)
    lopo = {}
    for T in c.TRAIN_PERIODS:
        tr, te = df[df.code_presentation != T], df[df.code_presentation == T]
        lopo[T] = roc_auc_score(te[c.TARGET], model().fit(tr[F], tr[c.TARGET]).predict_proba(te[F])[:, 1])
    fwd = {}
    for T, past in [("2013J", ["2013B"]), ("2014B", ["2013B", "2013J"])]:
        tr, te = df[df.code_presentation.isin(past)], df[df.code_presentation == T]
        fwd[T] = roc_auc_score(te[c.TARGET], model().fit(tr[F], tr[c.TARGET]).predict_proba(te[F])[:, 1])
    print(f"  {cut:5s} LOPO: " + ", ".join(f"{k} {v:.4f}" for k, v in lopo.items()) +
          f" | hacia delante: " + ", ".join(f"{k} {v:.4f}" for k, v in fwd.items()))
print("\n== 4. Selección de k y C con validación hacia delante (regla de un error típico)")
Cs = [0.001, 0.01, 0.1, 1, 10, 100]
rows = []
for k in range(1, 14):
    for C in Cs:
        per = {}
        for T, past in [("2013J", ["2013B"]), ("2014B", ["2013B", "2013J"])]:
            vals = []
            for cut in c.CUTS:
                df = c.load_train(cut); F = feats(cut, k)
                tr, te = df[df.code_presentation.isin(past)], df[df.code_presentation == T]
                vals.append(roc_auc_score(te[c.TARGET], model(C).fit(tr[F], tr[c.TARGET]).predict_proba(te[F])[:, 1]))
            per[T] = np.mean(vals)
        rows.append((k, C, np.mean(list(per.values())), np.std(list(per.values()), ddof=1) / np.sqrt(2)))
best = max(rows, key=lambda r: r[2])
ok = [r for r in rows if r[2] >= best[2] - best[3]]
choice = min(ok, key=lambda r: (r[0], r[1]))
print(f"  mejor: k={best[0]}, C={best[1]}, AUC={best[2]:.4f} (et {best[3]:.4f})")
print(f"  elegido (1 ET, menos variables y C menor): k={choice[0]}, C={choice[1]}, AUC={choice[2]:.4f}")
print("  k=4, C=0.001:", [f"{r[2]:.4f}" for r in rows if r[0] == 4 and r[1] == 0.001])
