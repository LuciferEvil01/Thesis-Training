# Thesis-Training · entrenamiento y evaluación del modelo

Código del **entrenamiento y la evaluación del modelo de predicción** del trabajo de diploma
*Detección temprana de estudiantes en riesgo académico: un sistema basado en los registros de Moodle*
(José Agustín del Toro González, Facultad de Matemática y Computación, Universidad de La Habana, 2026),
Capítulo 2. Aquí se construye el conjunto de datos a partir de OULAD (Kuzilek et al., 2017), se
entrenan y validan los modelos, se fija el modelo final y se hacen las comprobaciones que respaldan
las cifras de la tesis. El sistema que aplica el modelo en Moodle está en
[Thesis-Software](https://github.com/LuciferEvil01/Thesis-Software).

La historia de todas las decisiones, con su porqué y su resultado, está en `DECISIONES.md`.

## El modelo final

| | Primer corte | Segundo corte | Tercer corte |
|---|---|---|---|
| Modelo | `StandardScaler` → `LogisticRegression` (L2), C = 0,001 | ídem | ídem |
| Variables | `mean_score_with_zeros` (valor y percentil), `recent_clicks_28d_pct` | las 3 + `last_score` (valor) | ídem |
| Umbral | 0,433 | 0,389 | 0,353 |
| ROC-AUC en 2014J [IC 95 %] | 0,787 [0,778; 0,797] | 0,883 [0,875; 0,891] | 0,923 [0,917; 0,930] |

Entrenado con 2013B, 2013J y 2014B; 2014J es la prueba. El umbral es la proporción en riesgo de
los periodos de entrenamiento de cada corte (E18). Modelos en `artifacts/final_<corte>.joblib` (se generan al ejecutar el paso 4).

## El flujo, paso a paso

| Paso | Qué hace | Dónde | Decisiones | Salida |
|---|---|---|---|---|
| 1 · Datos | Descarga OULAD, fija los cortes y construye las variables (secciones 2.1 y 2.2) | `scripts/download_oulad.py`, `scripts/build_dataset.py`; revisión en `notebooks/01_datos` | D1, D2, E1 | `data/processed/dataset.parquet`, `artifacts/01_datos.json` |
| 2 · Validación anidada | Logística L2, Random Forest y Gradient Boosting, dejando fuera un periodo, con una configuración común a los tres cortes | `notebooks/02_validacion_anidada` | E10, E11 | `artifacts/02_validacion.json`; figura `esquema_validacion_anidada` |
| 3 · Selección de variables y elección del modelo | Orden común con la impureza de los árboles, k y C con la regla de un error típico, regla D9 por periodo | `notebooks/03_seleccion` | E10, E12–E14 | `artifacts/03_seleccion.json`; figura `resultados_validacion` |
| 4 · Modelo final | Ajuste con los tres periodos, antes de la segunda evaluación en la prueba | `notebooks/04_modelo_final` | E14 | `artifacts/final_<corte>.joblib`; figuras `importancias_arboles`, `seleccion_k` |
| 5 · Prueba en 2014J | ROC-AUC del modelo y de la regla de la nota con IC (segunda y última evaluación del modelo; ver la nota sobre 2014J) | `notebooks/05_prueba` | E15 | `artifacts/05_prueba.json` |
| 6 · Umbral | Proporción en riesgo del entrenamiento; clasificación en 2014J, misma lista que la regla, calibración y evolución de la alerta | `scripts/umbral_prevalencia.py` | E18 | `artifacts/06_umbral_prevalencia.json`; figuras `*_umbral`, `comparacion_misma_lista` |

Del paso 4 y del 5 solo vale lo que no depende del umbral: el umbral F2 que todavía calculan esos
notebooks (y sus figuras) quedó sustituido por el del paso 6.

**Lecturas de 2014J.** El modelo se evaluó en la prueba dos veces: con el de la primera ronda (E7) y
con el final (E15). Después, la prueba se volvió a leer sin cambiar el modelo para la calibración
(E17) y para el umbral del paso 6 (E18), que se adoptó después de verla (E20 lo evalúa fuera de
pliegue). Las simulaciones de las decisiones del sistema (03-11 y 05-11) también leen 2014J; E29 enumera todas
las lecturas.

### Comprobaciones posteriores (revisores)

Ninguna cambia el modelo; todas usan solo los periodos de entrenamiento, salvo la primera.

| Comprobación | Script | Decisión |
|---|---|---|
| Calibración en 2014J | `scripts/figura_calibracion.py` (la figura vigente es la del paso 6) | E17 |
| Intercambiabilidad de los periodos y validación hacia delante | `scripts/validacion_temporal.py` | E19, E23 |
| Umbral del paso 6 fuera de pliegue | `scripts/umbral_validacion.py` | E20, E23 |
| De dónde sale la ventaja sobre la regla de la nota | `scripts/aporte_actividad.py`, `scripts/riesgo_con_aprobado.py` | E21, E23 |
| La selección no depende del orden de importancia | `scripts/orden_variables.py` | E22 |
| Regla de un error típico con tres periodos y efecto de C | `scripts/penalizacion_y_regla.py` | E24 |
| Sensibilidad al retraso de la nota | `scripts/sensibilidad_retraso.py` | E25 |
| Entregas tardías, cursos de veinte, calibración por curso, clase positiva y conversión de la escala | `scripts/comprobaciones_ronda_q.py` (salida en `artifacts/ronda_q.txt`) | E30 |

`scripts/figuras_seleccion_variables.py` genera en español las figuras de las secciones 2.1 y 2.2.

### Lo que falta: validación con EVEA

Los datos de EVEA son un segundo conjunto de prueba: **no se reentrena**. El protocolo está fijado
antes de recibirlos (E26, con la etiqueta precisada en E31) y la herramienta está en
[Thesis-Software](https://github.com/LuciferEvil01/Thesis-Software)
(`python -m alerta validar-evea`, E27).

## Qué hay aquí

| Ruta | Qué es |
|---|---|
| `DECISIONES.md` | Bitácora de todas las decisiones (D1–D14 y E1–E32) |
| `ESTADO.md` | Estado actual y qué falta |
| `src/comun.py` | Código común: carga, validación anidada, *bootstrap*, figuras. Semilla fija (`SEED = 42`) |
| `notebooks/` | Pasos 1–5 (formato *percent* de jupytext y su `.ipynb` ejecutado) |
| `scripts/` | Datos, umbral, figuras y comprobaciones |
| `figuras/` | Figuras de la tesis en PDF. `correlaciones_last`, `regularizacion_rf`, `activity_presence` y `assessment_last_vs_mean` están al día, pero la tesis no las usa |
| `preparar.sh` | Crea `.venv`, descarga OULAD y construye los datos |
| `data/`, `artifacts/`, `.venv/` | Se generan; no se versionan |

## Reproducir

Requiere Python 3.11 o posterior; las versiones de las bibliotecas están fijadas en `requirements.txt`
(scikit-learn 1.9.1).

```bash
./preparar.sh                                 # entorno y datos (lo hecho no se repite)
nice -n 19 .venv/bin/jupytext --to ipynb --execute notebooks/0N_nombre.py
ABRIR_PRUEBA=1 MPLBACKEND=Agg .venv/bin/python scripts/umbral_prevalencia.py
```

Los datos de OULAD se descargan de su fuente oficial; se distribuyen con licencia CC BY 4.0
(Kuzilek, J., Hlosta, M. y Zdrahal, Z., 2017, *Open University Learning Analytics dataset*,
Scientific Data 4, 170171, https://doi.org/10.1038/sdata.2017.171).

Los notebooks y scripts que leen 2014J exigen `ABRIR_PRUEBA=1`.
