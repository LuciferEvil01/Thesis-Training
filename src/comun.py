"""Código común del entrenamiento: datos, validación cruzada anidada, bootstrap y figuras.

Lo importan todos los notebooks (`import comun as c`). Reúne lo que no debe variar entre
notebooks, para que cada uno contenga solo su experimento:

- Rutas, cortes, periodos y listas de variables de cada corte.
- Carga del entrenamiento (2013B, 2013J, 2014B) y, solo en la evaluación final, de la prueba (2014J).
- Validación cruzada anidada dejando fuera un periodo lectivo (plan.md, decisión D4).
- ROC-AUC con intervalo de confianza por bootstrap dentro de cada clase.
- Estilo de las figuras de la tesis (matplotlib, PDF, fondo claro).
- Protección térmica: el portátil de Jose tiene 2 núcleos y se calienta.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # raíz de Proposal
DATASET = ROOT / "data" / "processed" / "dataset.parquet"
ARTIFACTS = ROOT / "artifacts"  # resultados intermedios entre notebooks (predicciones, búsquedas)
FIGURES = ROOT / "figuras" / "Training"  # figuras que van a la tesis

SEED = 42
N_JOBS = int(os.environ.get("N_JOBS", "1"))  # núcleos por ajuste; 1 deja libre el otro núcleo

CUTS = ["first", "mid", "last"]
CUT_NAMES = {"first": "Primer corte", "mid": "Segundo corte", "last": "Tercer corte"}
TEST_PERIOD = "2014J"
TRAIN_PERIODS = ["2013B", "2013J", "2014B"]
TARGET = "at_risk"
CLASS = ["code_module", "code_presentation", "cut"]  # clase = asignatura, periodo y corte

VLE = ["total_clicks", "active_days", "days_since_last_activity", "recent_clicks_28d",
       "n_unique_sites", "clicks_per_active_day", "pre_course_clicks"]
VLE_PCT = [f"{v}_pct" for v in VLE]
GRADES_RAW = ["mean_score_with_zeros_raw", "last_score_raw"]
GRADES_PCT = ["mean_score_with_zeros_pct", "last_score_pct"]
OTHER = ["submission_rate", "repeated_course"]
ALL_FEATURES = VLE_PCT + GRADES_RAW + GRADES_PCT + OTHER

FAMILY = {**{f: "Actividad en el VLE" for f in VLE_PCT},
          **{f: "Evaluación" for f in GRADES_RAW + GRADES_PCT + ["submission_rate"]},
          "repeated_course": "Historial académico"}


def features_for(cut: str) -> list[str]:
    """Variables candidatas del corte: last_score no existe en el primero (sección 2.1.4.2)."""
    return [f for f in ALL_FEATURES if not (cut == "first" and f.startswith("last_score"))]


# ---------------------------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------------------------

def _read() -> pd.DataFrame:
    if not DATASET.exists():
        raise FileNotFoundError(f"Falta {DATASET}. Ejecuta: .venv/bin/python scripts/build_dataset.py")
    df = pd.read_parquet(DATASET)
    df["cut"] = df["cut"].astype(str)
    return df


def load_train(cut: str) -> pd.DataFrame:
    """Filas de un corte en los periodos de entrenamiento. 2014J no se carga."""
    df = _read()
    df = df[(df["cut"] == cut) & (df["code_presentation"] != TEST_PERIOD)].reset_index(drop=True)
    assert TEST_PERIOD not in set(df["code_presentation"])
    return df


def load_test(cut: str) -> pd.DataFrame:
    """Filas de un corte en el periodo de prueba (2014J).

    Solo se llama en el notebook de la evaluación final, después de anotar el modelo final en
    DECISIONES.md. Exige la variable de entorno ABRIR_PRUEBA=1 para que no se abra por descuido.
    """
    if os.environ.get("ABRIR_PRUEBA") != "1":
        raise PermissionError("El conjunto de prueba solo se abre en la evaluación final (ABRIR_PRUEBA=1).")
    df = _read()
    return df[(df["cut"] == cut) & (df["code_presentation"] == TEST_PERIOD)].reset_index(drop=True)


def class_id(df: pd.DataFrame) -> np.ndarray:
    return (df["code_module"] + "_" + df["code_presentation"] + "_" + df["cut"]).to_numpy()


# ---------------------------------------------------------------------------------------------
# Protección térmica
# ---------------------------------------------------------------------------------------------

# °C; se pueden cambiar con variables de entorno, y THERMAL_GUARD=0 desactiva la protección
TEMP_PAUSE = float(os.environ.get("TEMP_PAUSE", "80"))
TEMP_RESUME = float(os.environ.get("TEMP_RESUME", "68"))
MAX_PAUSE_S = 300  # una pausa nunca dura más de 5 minutos, aunque la CPU no llegue a enfriarse
THERMAL_ON = os.environ.get("THERMAL_GUARD", "1") != "0"


def cpu_temp() -> float | None:
    """Temperatura de la CPU (k10temp, coretemp o x86_pkg_temp), en °C; None si no se puede leer."""
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (h / "name").read_text().strip() in ("k10temp", "coretemp"):
                return int((h / "temp1_input").read_text()) / 1000
        except OSError:
            pass
    return None


def thermal_guard() -> None:
    """Si la CPU llega a TEMP_PAUSE, espera a que baje a TEMP_RESUME (como mucho MAX_PAUSE_S)."""
    t = cpu_temp() if THERMAL_ON else None
    if t is None or t < TEMP_PAUSE:
        return
    print(f"[pausa térmica: CPU a {t:.0f} °C]", flush=True)
    start = time.time()
    while (t := cpu_temp()) is not None and t > TEMP_RESUME and time.time() - start < MAX_PAUSE_S:
        time.sleep(15)


# ---------------------------------------------------------------------------------------------
# Validación cruzada dejando fuera un periodo, y su versión anidada
# ---------------------------------------------------------------------------------------------

def _grid(params: dict[str, list]) -> list[dict]:
    keys = list(params)
    return [dict(zip(keys, values)) for values in itertools.product(*(params[k] for k in keys))]


def _fit_predict(model, X_tr, y_tr, X_va) -> np.ndarray:
    thermal_guard()
    m = clone(model).fit(X_tr, y_tr)
    return m.predict_proba(X_va)[:, 1]


def grid_search_lopo(model, grid: dict[str, list], df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Búsqueda en rejilla con validación dejando fuera un periodo.

    Devuelve una fila por combinación de hiperparámetros, con la ROC-AUC de cada periodo validado y
    su media. Con dos periodos hay dos pliegues; con tres, tres.
    """
    X, y, periods = df[features], df[TARGET].to_numpy(), df["code_presentation"].to_numpy()
    rows = []
    for params in _grid(grid):
        m = clone(model).set_params(**params)
        aucs = {}
        for p in sorted(set(periods)):
            tr, va = periods != p, periods == p
            aucs[p] = roc_auc_score(y[va], _fit_predict(m, X[tr], y[tr], X[va]))
        rows.append({"params": params, **aucs, "auc": float(np.mean(list(aucs.values())))})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def nested_cv(model, grid: dict[str, list], df: pd.DataFrame, features: list[str], name: str | None = None) -> dict:
    """Validación cruzada anidada dejando fuera un periodo (decisión D4).

    Bucle externo: cada periodo de entrenamiento se deja fuera una vez. Bucle interno: con los otros
    dos periodos se eligen los hiperparámetros (grid_search_lopo, dos pliegues), se reajusta con
    ambos y se predice el periodo de fuera. Así la ROC-AUC del bucle externo estima el rendimiento
    del procedimiento completo, ajuste incluido, sin el optimismo de elegir y evaluar con los
    mismos datos.

    Devuelve las probabilidades fuera de pliegue (alineadas con df), la ROC-AUC de cada periodo
    externo, la ROC-AUC en el entrenamiento de cada pliegue (para diagnosticar sobreajuste), los
    hiperparámetros elegidos en cada pliegue y las tablas de las búsquedas internas. Si se pasa
    `name`, el resultado se guarda en artifacts/ y, si ya existe, se carga en lugar de recalcularlo.
    """
    path = ARTIFACTS / f"nested_{name}.joblib" if name else None
    if path is not None and path.exists():
        return joblib.load(path)

    X, y, periods = df[features], df[TARGET].to_numpy(), df["code_presentation"].to_numpy()
    oof = np.full(len(df), np.nan)
    folds = []
    for p in sorted(set(periods)):
        tr, va = periods != p, periods == p
        inner = grid_search_lopo(model, grid, df[tr], features)
        best = clone(model).set_params(**inner.loc[0, "params"])
        thermal_guard()
        best.fit(X[tr], y[tr])
        oof[va] = best.predict_proba(X[va])[:, 1]
        folds.append({"period": p, "params": inner.loc[0, "params"],
                      "auc_val": roc_auc_score(y[va], oof[va]),
                      "auc_train": roc_auc_score(y[tr], best.predict_proba(X[tr])[:, 1]),
                      "inner": inner})
    out = {"oof": oof, "folds": folds, "features": list(features)}
    if path is not None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        joblib.dump(out, path)
    return out


def summarize_nested(results: dict[str, dict]) -> pd.DataFrame:
    """Tabla modelo × periodo con la ROC-AUC del bucle externo, su media y su desviación."""
    rows = []
    for name, r in results.items():
        row = {"modelo": name, **{f["period"]: f["auc_val"] for f in r["folds"]}}
        vals = [f["auc_val"] for f in r["folds"]]
        row.update({"media": np.mean(vals), "desviación": np.std(vals, ddof=1),
                    "entrenamiento": np.mean([f["auc_train"] for f in r["folds"]])})
        rows.append(row)
    return pd.DataFrame(rows).set_index("modelo")


# ---------------------------------------------------------------------------------------------
# Modelos y rejillas (decisión D5)
# ---------------------------------------------------------------------------------------------

MODEL_NAMES = {"lr": "Regresión logística", "rf": "Random Forest", "gb": "Gradient Boosting"}


def make_model(name: str):
    """Modelo sin ajustar de cada familia; los hiperparámetros de GRIDS se eligen en la validación."""
    if name == "lr":  # penalización L2, la habitual (E10: la logística ya no selecciona variables)
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=N_JOBS)
    if name == "gb":
        return GradientBoostingClassifier(subsample=0.8, random_state=SEED)
    raise ValueError(name)


# Rejillas de D5, ampliadas según su punto de decisión (DECISIONES.md, E2)
GRIDS = {
    "lr": {"logisticregression__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},  # E13
    "rf": {"min_samples_leaf": [1, 5, 20, 50, 100, 200, 400], "max_features": ["sqrt", 0.5]},
    "gb": {"learning_rate": [0.02, 0.05, 0.1], "max_depth": [1, 2, 3, 4], "n_estimators": [50, 100, 300]},
}


# ---------------------------------------------------------------------------------------------
# Una configuración para los tres cortes (DECISIONES.md, E10)
# ---------------------------------------------------------------------------------------------

def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def joint_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une las búsquedas de los cortes: la ROC-AUC de cada periodo validado, promediada sobre los cortes.

    Cada tabla es una salida de grid_search_lopo (una fila por combinación). La columna `auc` es la
    media sobre periodos de esas medias, y la tabla se ordena de mayor a menor.
    """
    long = pd.concat([t.assign(key=t["params"].map(_key)) for t in tables.values()])
    periods = [p for p in long.columns if p not in ("params", "auc", "key")]
    out = long.groupby("key", sort=False).agg(params=("params", "first"), **{p: (p, "mean") for p in periods})
    out["auc"] = out[periods].mean(axis=1)
    return out.sort_values("auc", ascending=False).reset_index(drop=True)


def _inner_tables(m: str, data: dict, period_out: str) -> dict[str, pd.DataFrame]:
    """Búsquedas internas de un pliegue externo, por corte; reutiliza las ya guardadas (E2, E10)."""
    tables = {}
    for cut, df in data.items():
        path = ARTIFACTS / f"inner_{m}_{cut}_{period_out}.joblib"
        old = ARTIFACTS / f"nested_{m}_{cut}.joblib"  # validación anidada por corte de la primera ronda
        if path.exists():
            tables[cut] = joblib.load(path)
            continue
        if old.exists():
            tables[cut] = next(f["inner"] for f in joblib.load(old)["folds"] if f["period"] == period_out)
        else:
            tr = df["code_presentation"] != period_out
            tables[cut] = grid_search_lopo(make_model(m), GRIDS[m], df[tr], features_for(cut))
        joblib.dump(tables[cut], path)
    return tables


def nested_cv_joint(m: str, data: dict[str, pd.DataFrame]) -> dict:
    """Validación cruzada anidada con una configuración común a los cortes (E10).

    Como nested_cv, pero en cada pliegue externo se elige la combinación de hiperparámetros con
    mayor ROC-AUC media sobre los tres cortes, y con ella se ajusta cada corte por separado. Guarda
    también la importancia propia de los modelos de árboles en cada pliegue (para el ranking de la
    selección de variables). Se guarda en artifacts/nested_joint_<m>.joblib.
    """
    path = ARTIFACTS / f"nested_joint_{m}.joblib"
    if path.exists():
        return joblib.load(path)
    out = {cut: {"oof": np.full(len(df), np.nan), "folds": []} for cut, df in data.items()}
    out["inner"] = {}
    for p in TRAIN_PERIODS:
        joint = joint_table(_inner_tables(m, data, p))
        out["inner"][p] = joint
        params = joint.loc[0, "params"]
        for cut, df in data.items():
            feats = features_for(cut)
            tr, va = (df["code_presentation"] != p).to_numpy(), (df["code_presentation"] == p).to_numpy()
            thermal_guard()
            model = clone(make_model(m)).set_params(**params).fit(df.loc[tr, feats], df.loc[tr, TARGET])
            out[cut]["oof"][va] = model.predict_proba(df.loc[va, feats])[:, 1]
            out[cut]["folds"].append({
                "period": p, "params": params,
                "auc_val": roc_auc_score(df.loc[va, TARGET], out[cut]["oof"][va]),
                "auc_train": roc_auc_score(df.loc[tr, TARGET], model.predict_proba(df.loc[tr, feats])[:, 1]),
                "importances": pd.Series(model.feature_importances_, index=feats)
                if hasattr(model, "feature_importances_") else None})
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, path)
    return out


# ---------------------------------------------------------------------------------------------
# Selección de variables con los árboles y umbral F2 (E10)
# ---------------------------------------------------------------------------------------------

def tree_ranking(importances: dict[tuple[str, str], pd.Series]) -> pd.Series:
    """Orden común de las variables: importancia por impureza, normalizada a suma 1 en cada
    (modelo, corte), y promediada sobre los modelos y los cortes en que existe la variable."""
    norm = pd.DataFrame({k: v / v.sum() for k, v in importances.items()})
    return norm.mean(axis=1).sort_values(ascending=False)


def top_k(order: list[str], k: int, cut: str) -> list[str]:
    """Las k primeras variables del orden común que existen en el corte."""
    return [f for f in order[:k] if f in features_for(cut)]


def lopo_oof(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Predicciones fuera de pliegue dejando fuera cada periodo de df."""
    oof = np.full(len(df), np.nan)
    periods = df["code_presentation"].to_numpy()
    for p in sorted(set(periods)):
        tr, va = periods != p, periods == p
        oof[va] = _fit_predict(model, df.loc[tr, features], df.loc[tr, TARGET], df.loc[va, features])
    return oof


def selection_search(data: dict[str, pd.DataFrame], order: list[str], Cs: list[float],
                     previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """Búsqueda de (k, C) de la logística con una configuración común a los cortes.

    Una fila por (k, C) con la ROC-AUC de cada periodo validado (media sobre cortes) y su media.
    Si se pasa `previous` (una búsqueda anterior con el mismo orden), solo se calculan los C nuevos.
    """
    done = set() if previous is None else set(zip(previous["k"], previous["C"]))
    rows = [] if previous is None else previous.to_dict("records")
    for k in range(1, len(order) + 1):
        for C in Cs:
            if (k, C) in done:
                continue
            per_cut = {}
            for cut, df in data.items():
                feats = top_k(order, k, cut)
                model = make_model("lr").set_params(logisticregression__C=C)
                per_cut[cut] = grid_search_lopo(model, {}, df, feats).iloc[0] if feats else None
            if any(v is None for v in per_cut.values()):
                continue
            periods = [p for p in per_cut[next(iter(per_cut))].index if p not in ("params", "auc")]
            vals = {p: float(np.mean([per_cut[cut][p] for cut in per_cut])) for p in periods}
            rows.append({"k": k, "C": C, **vals, "auc": float(np.mean(list(vals.values())))})
    return pd.DataFrame(rows).sort_values(["k", "C"]).reset_index(drop=True)


def one_se_choice(table: pd.DataFrame) -> pd.Series:
    """Regla de un error típico (Hastie et al., 2009, §7.10): la menor k cuya ROC-AUC no baja del
    máximo menos el error típico del máximo; con esa k, el C de mayor ROC-AUC."""
    periods = [p for p in table.columns if p not in ("k", "C", "auc")]
    best = table.loc[table["auc"].idxmax()]
    se = np.std(best[periods].to_numpy(dtype=float), ddof=1) / np.sqrt(len(periods))
    k = table.loc[table["auc"] >= best["auc"] - se, "k"].min()
    row = table[table["k"] == k].sort_values("auc", ascending=False).iloc[0].copy()
    row["se"], row["best_auc"], row["best_k"] = se, best["auc"], best["k"]
    return row


def f2_threshold(y, prob) -> float:
    """Umbral que maximiza F2 (van Rijsbergen, 1979) en la clase en riesgo."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, prob)
    f2 = 5 * prec[:-1] * rec[:-1] / np.maximum(4 * prec[:-1] + rec[:-1], 1e-12)
    return float(thr[np.argmax(f2)])


# ---------------------------------------------------------------------------------------------
# Bootstrap dentro de cada clase
# ---------------------------------------------------------------------------------------------

def _class_resamples(classes: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(classes == k) for k in np.unique(classes)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx])


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC por rangos (Mann-Whitney); igual que roc_auc_score, pero más rápida en el bootstrap."""
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def bootstrap_auc(y, scores: dict[str, np.ndarray], classes, reference: str | None = None,
                  n_boot: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """ROC-AUC de cada puntuación con su IC del 95 % por bootstrap percentil.

    Se remuestrean los estudiantes dentro de cada clase, de modo que cada réplica conserva la
    composición por asignatura. Todas las puntuaciones usan las mismas réplicas; si se indica
    `reference`, se añade la diferencia pareada con ella y su IC del 95 %.
    """
    y, classes = np.asarray(y), np.asarray(classes)
    names = list(scores)
    S = np.column_stack([np.asarray(scores[n], dtype=float) for n in names])
    boot = np.array([[_auc(y[ix], S[ix, j]) for j in range(len(names))]
                     for ix in _class_resamples(classes, n_boot, seed)])
    point = np.array([_auc(y, S[:, j]) for j in range(len(names))])
    out = pd.DataFrame({"auc": point, "ic_inf": np.percentile(boot, 2.5, axis=0),
                        "ic_sup": np.percentile(boot, 97.5, axis=0)}, index=names)
    if reference is not None:
        d = boot - boot[:, [names.index(reference)]]
        out["dif"] = point - point[names.index(reference)]
        out["dif_inf"] = np.percentile(d, 2.5, axis=0)
        out["dif_sup"] = np.percentile(d, 97.5, axis=0)
    return out


def bootstrap_auc_by_period(y, scores: dict[str, np.ndarray], classes, periods, reference: str,
                            n_boot: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Como bootstrap_auc, pero el estadístico es la media de la ROC-AUC de cada periodo (E13).

    Es la métrica de la validación anidada: cada periodo externo se predice con un modelo
    distinto, y juntar sus predicciones mezclaría escalas de probabilidad distintas. Las clases
    están anidadas en los periodos, así que remuestrear dentro de cada clase conserva los periodos.
    """
    y, classes, periods = np.asarray(y), np.asarray(classes), np.asarray(periods)
    names = list(scores)
    S = np.column_stack([np.asarray(scores[n], dtype=float) for n in names])
    groups = [periods == p for p in np.unique(periods)]

    def stat(ix):
        yy, SS, gg = y[ix], S[ix], [g[ix] for g in groups]
        return [np.mean([_auc(yy[g], SS[g, j]) for g in gg]) for j in range(len(names))]

    boot = np.array([stat(ix) for ix in _class_resamples(classes, n_boot, seed)])
    point = np.array(stat(np.arange(len(y))))
    d = boot - boot[:, [names.index(reference)]]
    return pd.DataFrame({"auc": point, "ic_inf": np.percentile(boot, 2.5, axis=0),
                         "ic_sup": np.percentile(boot, 97.5, axis=0),
                         "dif": point - point[names.index(reference)],
                         "dif_inf": np.percentile(d, 2.5, axis=0), "dif_sup": np.percentile(d, 97.5, axis=0)},
                        index=names)


# ---------------------------------------------------------------------------------------------
# Línea base (decisión D8)
# ---------------------------------------------------------------------------------------------

PASS_MARK = 40  # nota de aprobado de una evaluación en OULAD


def baseline_score(df: pd.DataFrame) -> np.ndarray:
    """Puntuación de riesgo de la regla de la nota: cuanto menor la nota media, mayor el riesgo."""
    return -df["mean_score_with_zeros"].to_numpy(dtype=float)


def baseline_label(df: pd.DataFrame) -> np.ndarray:
    """Etiqueta de la regla de la nota: en riesgo si la nota media (con ceros) está por debajo del aprobado."""
    return (df["mean_score_with_zeros"].to_numpy(dtype=float) < PASS_MARK).astype(int)


# ---------------------------------------------------------------------------------------------
# Figuras de la tesis
# ---------------------------------------------------------------------------------------------

CUT_COLORS = {"first": "#2a6fdb", "mid": "#d9622b", "last": "#1b9e77"}
MODEL_COLORS = ["#2a6fdb", "#d9622b", "#1b9e77", "#7a5195", "#8c8c8c"]


def thesis_style() -> None:
    """Estilo común: fondo blanco, tipografía con serifa como la tesis, sin marco superior ni derecho."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "font.family": "serif", "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 9, "legend.frameon": False, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.formatter.use_locale": False,
    })


def save_figure(fig, name: str) -> Path:
    """Guarda la figura en PDF en figuras/Training/ y la deja visible en el notebook."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path)
    return path


def comma(x: float, digits: int = 3) -> str:
    """Número con coma decimal, como en la tesis."""
    return f"{x:.{digits}f}".replace(".", ",")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))


# Nombres de las variables en las figuras: el de la sección 2.1, con la representación de la nota
LABEL = {**{f"{v}_pct": v for v in VLE},
         "mean_score_with_zeros_raw": "mean_score_with_zeros (valor)",
         "mean_score_with_zeros_pct": "mean_score_with_zeros (percentil)",
         "last_score_raw": "last_score (valor)", "last_score_pct": "last_score (percentil)",
         "submission_rate": "submission_rate", "repeated_course": "repeated_course"}


def comma_axis(ax, axis: str = "y", digits: int = 2) -> None:
    """Coma decimal en las marcas de un eje."""
    fmt = mpl.ticker.FuncFormatter(lambda x, _: comma(x, digits))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
