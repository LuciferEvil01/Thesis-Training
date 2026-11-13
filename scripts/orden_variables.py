"""¿Cambia el modelo elegido si el orden de las variables se obtiene de otra forma? (crítica C7, 06-11-2026)

Órdenes comparados: (a) reducción de impureza en RF y GB (el de la tesis); (b) permutación en el
periodo dejado fuera con RF y GB; (c) permutación en el periodo dejado fuera con la regresión logística
de las 13 columnas; (d) selección hacia delante con la regresión logística. Con cada orden, la regla de
un error típico elige k y C como en la tesis. Solo periodos de entrenamiento. Uso:
.venv/bin/python scripts/orden_variables.py
"""
import json, sys
import numpy as np
sys.path.insert(0, "src")
import comun as c
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CS = [0.001, 0.01, 0.1, 1, 10, 100]
DATA = {cut: c.load_train(cut) for cut in c.CUTS}
ALL = c.ALL_FEATURES
def lr(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
def feats(cut, F): return [f for f in F if not (cut == "first" and f.startswith("last_score"))]

def auc_lopo(F, C):
    """Media sobre cortes de la ROC-AUC de cada periodo dejado fuera; devuelve el vector por periodo."""
    per = np.zeros(3)
    for cut in c.CUTS:
        df = DATA[cut]; F2 = feats(cut, F)
        if not F2:  # p. ej., solo la última nota en el primer corte, donde no existe
            per += 0.5 / 3
            continue
        for i, T in enumerate(c.TRAIN_PERIODS):
            tr, te = df[df.code_presentation != T], df[df.code_presentation == T]
            per[i] += roc_auc_score(te[c.TARGET], lr(C).fit(tr[F2], tr[c.TARGET]).predict_proba(te[F2])[:, 1]) / 3
    return per

def choose(order):
    rows = []
    for k in range(1, len(order) + 1):
        best = max(((C, auc_lopo(order[:k], C)) for C in CS), key=lambda t: t[1].mean())
        rows.append((k, best[0], best[1].mean(), best[1].std(ddof=1) / np.sqrt(3)))
    top = max(rows, key=lambda r: r[2])
    k, C, auc, _ = min((r for r in rows if r[2] >= top[2] - top[3]), key=lambda r: r[0])
    return order[:k], C, auc, top

def perm_order(make):
    imp = {f: 0.0 for f in ALL}; n = {f: 0 for f in ALL}
    for cut in c.CUTS:
        df = DATA[cut]; F = feats(cut, ALL)
        for T in c.TRAIN_PERIODS:
            tr, te = df[df.code_presentation != T], df[df.code_presentation == T]
            m = make().fit(tr[F], tr[c.TARGET])
            r = permutation_importance(m, te[F], te[c.TARGET], scoring="roc_auc", n_repeats=5, random_state=c.SEED, n_jobs=2)
            tot = max(r.importances_mean.clip(min=0).sum(), 1e-12)
            for f, v in zip(F, r.importances_mean):
                imp[f] += max(v, 0) / tot; n[f] += 1
    return sorted(ALL, key=lambda f: -imp[f] / n[f])

orders = {}
def report(name, o):
    orders[name] = o
    chosen, C, auc, top = choose(o)
    print(f"{name}\n  orden: {', '.join(o[:6])}, …\n  elegidas ({len(chosen)}): {', '.join(chosen)}; C = {C}; "
          f"ROC-AUC {auc:.4f} (máxima {top[2]:.4f} con k = {top[0]})", flush=True)
report("a) impureza (tesis)", json.load(open("artifacts/04_modelo_final.json"))["order"])
rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features="sqrt", random_state=c.SEED, n_jobs=2)
gb = lambda: GradientBoostingClassifier(learning_rate=0.02, max_depth=3, n_estimators=300, subsample=0.8, random_state=c.SEED)
o_rf, o_gb = perm_order(rf), perm_order(gb)
pos = {f: (o_rf.index(f) + o_gb.index(f)) / 2 for f in ALL}
report("b) permutación, árboles", sorted(ALL, key=lambda f: pos[f]))
report("c) permutación, logística", perm_order(lambda: lr(0.001)))
fwd, rest = [], list(ALL)
while rest:
    nxt = max(rest, key=lambda f: max(auc_lopo(fwd + [f], C).mean() for C in [0.001, 1]))
    fwd.append(nxt); rest.remove(nxt)
report("d) hacia delante, logística", fwd)
