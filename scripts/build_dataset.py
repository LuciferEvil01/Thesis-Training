"""Construye el conjunto de datos `at_risk` de la sección 2.1 a partir de OULAD.

Una fila por estudiante, presentación de la asignatura y corte. En cada corte solo se usa la
información con fecha <= cutoff_day, y se excluye a quien ya se había dado de baja.

    at_risk = 1  si final_result en {Fail, Withdrawn}
    at_risk = 0  si final_result en {Pass, Distinction}

Cortes anclados a las TMA (sección 2.1.1): en cada presentación, la TMA de plazo más próximo al
25 % del curso (first), la más próxima al 50 % (mid) y la última (last). El día de corte es el
plazo de esa TMA + 14 días.

Columnas de la salida
---------------------
- Identificación (no son variables): id_student, code_module, code_presentation, cut,
  anchor_tma, cutoff_day.
- Objetivo: at_risk. También final_result, solo para describir los errores; nunca es variable.
- Valores originales, para la exploración y la línea base: las 7 de VLE, submission_rate,
  mean_score_with_zeros y last_score (notas de 0 a 100), repeated_course.
- Variables del modelo (ver src/comun.py):
    * VLE en percentil dentro de la clase: <variable>_pct (sección 2.1.3.3).
    * Notas en sus dos representaciones (revisión del tutor, hoja 3):
        <nota>_raw = nota / 100        (frontera absoluta del aprobado)
        <nota>_pct = percentil en la clase (posición respecto a los compañeros)
      para mean_score_with_zeros y last_score. last_score no existe en el corte first.
    * submission_rate (0-1) y repeated_course (0/1), sin transformar.

El percentil no usa la variable objetivo y cada clase (asignatura, periodo y corte) se normaliza
solo con sus propios estudiantes; como cada clase cae entera en un periodo, calcularlo aquí no
produce fugas entre particiones.

Uso (desde la raíz de Proposal):
    python scripts/build_dataset.py
Salida: data/processed/dataset.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "oulad"
OUT = ROOT / "data" / "processed" / "dataset.parquet"

TMA_TARGETS = {"first": 25, "mid": 50}  # % del curso al que se acercan los dos primeros cortes
GRADE_LAG = 14  # días entre el plazo de una TMA y la publicación de su nota (supuesto, sección 2.1.1)
RECENT_DAYS = 28  # ventana de recent_clicks_28d (sección 2.1.3.1)

KEYS = ["id_student", "code_module", "code_presentation"]
COURSE = ["code_module", "code_presentation"]
ROW = KEYS + ["cut"]
CLASS = COURSE + ["cut"]
AT_RISK = {"Fail": 1, "Withdrawn": 1, "Pass": 0, "Distinction": 0}

VLE = ["total_clicks", "active_days", "days_since_last_activity", "recent_clicks_28d",
       "n_unique_sites", "clicks_per_active_day", "pre_course_clicks"]
GRADES = ["mean_score_with_zeros", "last_score"]


def read(name, **kwargs):
    return pd.read_csv(RAW / f"{name}.csv", na_values="?", **kwargs)


def load():
    if not RAW.exists():
        sys.exit(f"No existe {RAW}. Ejecuta antes: python scripts/download_oulad.py")
    courses = read("courses")
    info = read("studentInfo", usecols=KEYS + ["final_result", "num_of_prev_attempts"])
    registration = read("studentRegistration", usecols=KEYS + ["date_unregistration"])
    assessments = read("assessments")
    student_assessment = read("studentAssessment")
    student_vle = read("studentVle", dtype={"id_student": "int32", "id_site": "int32",
                                            "date": "int16", "sum_click": "int32"})
    if info.duplicated(KEYS).any():
        sys.exit("studentInfo tiene claves (estudiante, curso) repetidas")

    students = info.merge(registration, on=KEYS, how="left").merge(courses, on=COURSE)
    students["at_risk"] = students["final_result"].map(AT_RISK).astype("int8")
    students["repeated_course"] = (students["num_of_prev_attempts"] > 0).astype("int8")

    # TMA numeradas por orden de plazo dentro de cada presentación
    tmas = assessments[assessments["assessment_type"] == "TMA"].merge(courses, on=COURSE)
    tmas = tmas.sort_values(COURSE + ["date"])
    tmas["k"] = tmas.groupby(COURSE).cumcount() + 1
    tmas["pct"] = tmas["date"] / tmas["module_presentation_length"] * 100
    submissions = student_assessment.merge(tmas[["id_assessment"]], on="id_assessment")
    return students, tmas, submissions, student_vle


def tma_cuts(tmas):
    """TMA de anclaje y día de corte de cada presentación y corte."""
    rows = []
    for course, g in tmas.groupby(COURSE):
        anchors = {cut: g["k"].iloc[(g["pct"] - target).abs().argmin()] for cut, target in TMA_TARGETS.items()}
        anchors["last"] = g["k"].max()
        for cut, k in anchors.items():
            due = int(g.loc[g["k"] == k, "date"].iloc[0])
            rows.append((*course, cut, int(k), due + GRADE_LAG))
    return pd.DataFrame(rows, columns=COURSE + ["cut", "anchor_tma", "cutoff_day"])


def vle_features(student_vle, base):
    """Actividad en el VLE hasta el día de corte, incluida la anterior al inicio del curso."""
    v = student_vle.merge(base[ROW + ["cutoff_day"]], on=KEYS)
    v = v[v["date"] <= v["cutoff_day"]]
    g = v.groupby(ROW)
    f = pd.DataFrame({
        "total_clicks": g["sum_click"].sum(),
        "active_days": g["date"].nunique(),
        "n_unique_sites": g["id_site"].nunique(),
        "days_since_last_activity": g["cutoff_day"].first() - g["date"].max(),
    })
    f["clicks_per_active_day"] = f["total_clicks"] / f["active_days"]
    f["pre_course_clicks"] = v[v["date"] < 0].groupby(ROW)["sum_click"].sum()
    f["recent_clicks_28d"] = v[v["date"] > v["cutoff_day"] - RECENT_DAYS].groupby(ROW)["sum_click"].sum()

    # Sin actividad registrada: 0 clics; la recencia queda vacía (no está definida)
    counts = [c for c in f.columns if c != "days_since_last_activity"]
    f = base[ROW].merge(f.reset_index(), on=ROW, how="left")
    f[counts] = f[counts].fillna(0)
    return f


def assessment_features(tmas, submissions, base):
    """TMA 1..anchor_tma de cada corte; entregada = date_submitted <= cutoff_day; sin pesos."""
    due = base[ROW + ["cutoff_day", "anchor_tma"]].merge(tmas[COURSE + ["id_assessment", "k"]], on=COURSE)
    due = due[due["k"] <= due["anchor_tma"]]
    due = due.merge(submissions[["id_assessment", "id_student", "date_submitted", "score"]],
                    on=["id_assessment", "id_student"], how="left")
    due["submitted"] = due["date_submitted"] <= due["cutoff_day"]
    due["score0"] = due["score"].where(due["submitted"], 0)

    g = due.groupby(ROW)
    f = pd.DataFrame({"submission_rate": g["submitted"].mean(),
                      "mean_score_with_zeros": g["score0"].mean()})
    f["last_score"] = due[due["k"] == due["anchor_tma"]].set_index(ROW)["score0"]
    f = base[ROW].merge(f.reset_index(), on=ROW, how="left")

    # Entregas sin nota en OULAD (unas decenas de filas): la nota media se calcula con las demás
    # (pandas ignora los vacíos) y la del corte toma la nota media del propio estudiante. Si no
    # queda ninguna nota (5 filas del primer corte), la media toma la mediana de su clase.
    class_median = f.groupby([f[c] for c in CLASS])["mean_score_with_zeros"].transform("median")
    f["mean_score_with_zeros"] = f["mean_score_with_zeros"].fillna(class_median)
    f["last_score"] = f["last_score"].fillna(f["mean_score_with_zeros"])
    f.loc[f["cut"] == "first", "last_score"] = np.nan  # no se usa en el primer corte (sección 2.1.4.2)
    return f


def class_percentile(df, col, invert=False):
    """Percentil dentro de la clase: rango promedio en los empates; 0,5 si la clase no varía."""
    x = df[col].astype(float)
    if invert:  # más días sin actividad = menos actividad; sin ningún clic = el peor percentil
        x = -x.fillna(np.inf)
    by_class = x.groupby([df[c] for c in CLASS])
    pct = by_class.rank(pct=True, method="average")
    return pct.mask(by_class.transform("nunique") <= 1, 0.5)


def build():
    print("Cargando OULAD...")
    students, tmas, submissions, student_vle = load()
    base = students.merge(tma_cuts(tmas), on=COURSE)
    gone = base["date_unregistration"] <= base["cutoff_day"]
    n_gone = gone.groupby(base["cut"]).sum()
    base = base[~gone].reset_index(drop=True)

    df = (base[ROW + ["anchor_tma", "cutoff_day", "repeated_course", "at_risk", "final_result"]]
          .merge(vle_features(student_vle, base), on=ROW)
          .merge(assessment_features(tmas, submissions, base), on=ROW))

    for c in VLE:
        df[f"{c}_pct"] = class_percentile(df, c, invert=(c == "days_since_last_activity"))
    for c in GRADES:
        df[f"{c}_raw"] = df[c] / 100
        df[f"{c}_pct"] = class_percentile(df, c)
    df.loc[df["cut"] == "first", "last_score_pct"] = np.nan

    df["cut"] = pd.Categorical(df["cut"], ["first", "mid", "last"], ordered=True)
    df = df.sort_values(CLASS + ["id_student"]).reset_index(drop=True)
    for cut, d in df.groupby("cut", observed=True):
        print(f"  {cut:>5}: {len(d):6d} filas ({n_gone[cut]} dadas de baja antes del corte), "
              f"at_risk = {d['at_risk'].mean():.1%}")
    return df


def main():
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Escrito {OUT.relative_to(ROOT)}: {len(df)} filas, {df.shape[1]} columnas")


if __name__ == "__main__":
    main()
