# Bitácora de decisiones

Cada entrada lleva número, fecha, qué se decidió, por qué y con qué resultado. Las nuevas se
añaden al final; si una cambia otra anterior, lo dice y la anterior no se borra.

## Decisiones previas (21-10-2026)

Las decisiones D1–D14 se fijaron antes de ver ningún resultado, a partir de la
revisión del tutor y del diagnóstico del trabajo anterior. En resumen: los datos de la sección 2.1
con las notas en sus dos representaciones (D1, D2); un modelo por corte (D3); 2014J como prueba y
validación cruzada anidada dejando fuera un periodo (D4); regresión logística con L1, Random Forest
y Gradient Boosting (D5); ROC-AUC como única métrica de comparación (D6); sin calibración ni
umbral a medida (D7); la regla de la nota como única línea base (D8); la logística como modelo
final salvo mejora consistente (D9); importancia con los mecanismos de cada modelo (D10);
evaluación única en la prueba con IC, curvas ROC y matrices de confusión (D11, D12); lo que no se
hace (D13); la redacción (D14).

## Decisiones

(La primera será la E1.)

### E1 · 21-10-2026 · Fase 1: los datos son aptos y se pasa a la fase 2 sin cambios

**Qué.** El conjunto de entrenamiento se usa tal como lo construye `scripts/build_dataset.py`
(D1, D2). No se cambia ninguna variable.
**Por qué.** `notebooks/01_datos`: 17 512 · 16 245 · 15 344 filas por corte, 15 clases por corte,
sin vacíos y todas las variables en [0, 1]; prevalencia del 32 % al 46 % (desbalance moderado,
se mantiene D7). Las dos representaciones de cada nota tienen correlación de Spearman de
0,95–0,97 (idénticas dentro de cada clase por construcción; difieren solo entre clases), y las
siete de VLE son muy redundantes (el primer componente principal explica el 69 % de su varianza).
Esto anticipa que la importancia por permutación se repartirá entre variables correlacionadas
(D10).
**Nota de transparencia.** Al comprobar el archivo de datos se imprimió una vez el número de
filas de cada corte y periodo, 2014J incluido (8 764 · 7 955 · 7 603). No se miraron sus
variables ni su variable objetivo; el número de filas no interviene en ninguna decisión.
**Figuras.** `prevalencia_periodos`, `distribucion_variables_mid`, `correlaciones_last`.

### E2 · 21-10-2026 · Fase 2: se amplían las rejillas (punto de decisión de D5)

**Qué.** Primera ronda de la validación anidada con las rejillas de D5 (resultados guardados en
`artifacts/rejilla_v1/`). El hiperparámetro elegido cayó en un extremo de la rejilla en al
menos 2 de los 3 pliegues externos en estos casos:
- Gradient Boosting: `learning_rate` = 0,05 (mínimo) en los tres cortes; `max_depth` = 2 (mínimo)
  en el primero; `n_estimators` = 100 (mínimo) en los tres. Es decir, el óptimo está en el
  extremo de menor complejidad.
- Random Forest: `min_samples_leaf` = 100 (máximo) en el primer corte.
- Regresión logística: `C` = 10 (máximo) en el segundo corte.

Las rejillas se amplían en esa dirección, para los tres cortes, de modo que cada modelo tenga una
sola rejilla:
- logística, `C` hasta 30 y 100;
- Random Forest, `min_samples_leaf` hasta 200 y 400;
- Gradient Boosting, `learning_rate` ∈ {0,02; 0,05; 0,1}, `max_depth` ∈ {1, 2, 3, 4} y
  `n_estimators` ∈ {50, 100, 300}.

Se repite la validación anidada completa.
**Por qué.** Es la regla escrita en D5: un óptimo en el borde indica que la rejilla no contiene el
mejor valor.
**Resultado de la primera ronda** (para que conste; no es el definitivo). Media ± desviación entre
periodos de la ROC-AUC externa:

| Corte | Logística | Random Forest | Gradient Boosting | Regla de la nota |
|---|---|---|---|---|
| Primero | 0,822 ± 0,009 | 0,823 ± 0,008 | 0,822 ± 0,010 | 0,801 ± 0,010 |
| Segundo | 0,908 ± 0,012 | 0,911 ± 0,011 | 0,912 ± 0,008 | 0,893 ± 0,019 |
| Tercero | 0,939 ± 0,007 | 0,946 ± 0,002 | 0,946 ± 0,002 | 0,929 ± 0,015 |

Con D9, ninguno de los modelos complejos superaba a la logística en los tres cortes: en el primero
no había diferencia, y en el segundo la diferencia (~0,006) no llegaba a la desviación entre
periodos (0,012).

### E3 · 21-10-2026 · Fase 2: no se amplía más la rejilla del Gradient Boosting

**Qué.** Tras la ampliación de E2, la logística y el Random Forest tienen el óptimo dentro de la
rejilla en todos los cortes. El Gradient Boosting aún toca algún borde en 2 de 3 pliegues:
`learning_rate` = 0,1 (máximo) y `max_depth` = 1 (mínimo) en el primer corte;
`learning_rate` = 0,02 (mínimo) y `n_estimators` = 300 (máximo) en el tercero. No se hace una
tercera ronda.
**Por qué.** (1) Los bordes cambian de dirección respecto a la primera ronda (entonces,
`learning_rate` en el mínimo en el primer corte; ahora, en el máximo). (2) Son bordes de
`learning_rate` y `n_estimators`, que se compensan: una tasa menor con más árboles da un ajuste
equivalente (Friedman, 2001). La rejilla contiene en su interior combinaciones del mismo
«presupuesto» de *boosting* (por ejemplo, 0,05 × 100 frente a 0,02 × 300). (3) La ROC-AUC
externa del GB no cambió con la ampliación (0,822 · 0,912 · 0,946 en ambas rondas), así que la
superficie es plana en esa dirección. (4) D9 no puede cambiar: en el primer corte los tres
modelos son idénticos (diferencia GB − logística de 0,000, IC [−0,001; 0,001]). Esto interpreta
el punto de decisión de D5 (que no dice cuántas rondas); si Jose prefiere aplicarlo
literalmente, una tercera ronda solo del GB costaría alrededor de una hora.

### E4 · 21-10-2026 · Fase 2: el modelo final es la regresión logística con L1 (regla D9)

**Qué.** Modelo final en los tres cortes: regresión logística con penalización L1 (sus
hiperparámetros se fijan en la fase 4).
**Por qué.** Aplicación literal de D9 sobre la validación anidada con las rejillas de E2
(`notebooks/02_validacion_anidada`). ROC-AUC externa, media ± desviación entre periodos:

| Corte | Logística | Random Forest | Gradient Boosting | Regla de la nota |
|---|---|---|---|---|
| Primero | 0,822 ± 0,009 | 0,823 ± 0,008 | 0,822 ± 0,010 | 0,801 ± 0,010 |
| Segundo | 0,908 ± 0,012 | 0,911 ± 0,011 | 0,912 ± 0,010 | 0,893 ± 0,019 |
| Tercero | 0,939 ± 0,007 | 0,946 ± 0,002 | 0,946 ± 0,001 | 0,929 ± 0,015 |

Diferencia pareada con la logística (predicciones fuera de pliegue, bootstrap por clase):

| Corte | RF − logística [IC 95 %] | GB − logística [IC 95 %] | Desv. de la logística | ¿Cumple D9? |
|---|---|---|---|---|
| Primero | 0,000 [−0,002; 0,002] | 0,000 [−0,001; 0,001] | 0,009 | No (1 y 2) |
| Segundo | 0,005 [0,004; 0,007] | 0,007 [0,005; 0,008] | 0,012 | No (2) |
| Tercero | 0,009 [0,007; 0,010] | 0,009 [0,007; 0,010] | 0,007 | Sí |

Ningún modelo complejo cumple en los tres cortes. **Resultado adicional:** la logística supera a
la regla de la nota en los tres cortes: 0,017 [0,013; 0,020] · 0,012 [0,010; 0,015] ·
0,006 [0,004; 0,008]. La diferencia entre la ROC-AUC de entrenamiento y la externa es pequeña en
todos los modelos (≤ 0,024), porque la regularización se eligió en el bucle interno (el RF elige
hojas de 20 a 100 estudiantes; figura `regularizacion_rf`).

### E5 · 21-10-2026 · Fase 3: se quedan las dos representaciones de la nota (regla de D10)

**Qué.** No se retira ninguna variable; las notas siguen entrando como valor y como percentil.
**Regla aplicada** (concretada en `notebooks/03_importancia` antes de ejecutarlo): se retira una
representación si su coeficiente es nulo en la logística en los 9 pliegues externos (3 cortes ×
3 periodos) y su puesto medio en las importancias del RF y del GB está en el último cuarto.
**Resultado.** Ninguna de las dos es nula en ningún pliegue. Puesto medio en los árboles: valor
2,2, percentil 4,4 (de 11–13 variables). El valor es la representación preferida; el percentil
aporta algo en la logística y en el RF.
**Otros resultados** (`coeficientes_logistica`, `importancias_modelos`). La familia de evaluación
reúne del 48 % al 87 % de la importancia, según el modelo y el corte; de las de VLE pesan sobre
todo la actividad reciente y la recencia. La L1 elige penalizaciones débiles (C = 0,3–10 con los
tres periodos) y casi no anula coeficientes. La permutación en el periodo dejado fuera confirma
el orden en lo esencial.

### E6 · 21-10-2026 · Fase 4: modelo final de cada corte (antes de abrir la prueba)

**Qué.** El modelo final queda fijado así y no se cambia después:

| Corte | Modelo | Variables | C | Entrenamiento |
|---|---|---|---|---|
| Primero | `StandardScaler` → `LogisticRegression(l1_ratio=1, solver="liblinear", C=0,3)` | las 11 candidatas | 0,3 | 17 512 filas (2013B, 2013J, 2014B) |
| Segundo | ídem, `C=10` | las 13 candidatas | 10 | 16 245 filas |
| Tercero | ídem, `C=0,3` | las 13 candidatas | 0,3 | 15 344 filas |

Las variables son las 7 de VLE en percentil, `submission_rate`, `mean_score_with_zeros` en valor y
en percentil, `last_score` en valor y en percentil (no en el primer corte) y `repeated_course`.
Ningún coeficiente es nulo. Modelos guardados en `artifacts/final_<corte>.joblib`. La línea base
de la evaluación es la regla de la nota (D8): puntuación = −`mean_score_with_zeros`, y etiqueta =
en riesgo si esa nota es menor que 40. El umbral de la matriz de confusión del modelo es 0,5 (D7).
**Por qué.** E4 (familia), E5 (variables) y la búsqueda de D4 con los tres periodos
(`notebooks/04_modelo_final`). La ROC-AUC es plana para C ≥ 0,1 (diferencias en la cuarta cifra),
así que el valor concreto de C no tiene consecuencias.

### E7 · 21-10-2026 · Fase 5: evaluación única en 2014J

**Qué.** Se ejecutó `notebooks/05_prueba` con el modelo de E6. Antes se ensayó el código en una
copia temporal que sustituía la prueba por 2014B, sin abrir la prueba ni guardar nada. En ese
ensayo se corrigió el formato de las etiquetas de la matriz de confusión. La ejecución real se
repitió una vez más, con los mismos números, solo para añadir la celda de conclusiones. No se
cambió nada del modelo.
**Resultados.**

| Corte | Modelo final [IC 95 %] | Regla de la nota [IC 95 %] | Diferencia [IC 95 %] |
|---|---|---|---|
| Primero | 0,794 [0,785; 0,803] | 0,754 [0,744; 0,765] | 0,039 [0,034; 0,044] |
| Segundo | 0,889 [0,882; 0,897] | 0,845 [0,836; 0,854] | 0,045 [0,040; 0,049] |
| Tercero | 0,927 [0,921; 0,934] | 0,899 [0,892; 0,907] | 0,028 [0,024; 0,032] |

Con el umbral 0,5, el modelo tiene una sensibilidad de 0,653 · 0,731 · 0,797 y una precisión de
0,645 · 0,709 · 0,748; la regla tiene 0,378 · 0,488 · 0,582 y 0,680 · 0,773 · 0,822. F1: 0,649 ·
0,720 · 0,771 frente a 0,486 · 0,598 · 0,681. La ROC-AUC en 2014J es menor que en la validación
cruzada en los tres cortes, y la regla de la nota cae en la misma medida o más: la diferencia se
atribuye al periodo, no al ajuste. Así se dirá en la tesis, sin afirmar que el periodo sea «más
difícil» (revisión, hoja 8).

### E8 · 21-10-2026 · Fase 6: redacción de la sección 2.2 y retoque de la 2.1

**Qué.**
- El texto de la sección de entrenamiento de la tesis, nuevo, con la estructura del plan: partición y validación, métrica,
  modelos y línea base, resultados de la validación, importancia, modelo final, prueba y
  limitaciones. Tiene 8 figuras y 7 tablas.
- El texto de la sección de variables, retocado en tres sitios:
  - el párrafo de la representación de las notas (ahora entran como valor y como percentil);
  - el párrafo y la tabla resumen;
  - se quita la frase que prometía comparar en la prueba el modelo con las variables
    seleccionadas frente al modelo con todas: el plan no lo hace, porque la selección va
    dentro del ajuste (D5, D10).
- Un informe para el tutor, con la correspondencia punto por punto.

**Cambios técnicos sin efecto en los resultados.**
- Las figuras se regeneraron al ancho de página (6,5 pulgadas) para que la letra sea legible.
  Para ello se volvieron a ejecutar los notebooks 01–03 y 05; este último usa el mismo modelo y
  da los mismos números que en E7.
- El esquema de la validación anidada se dibuja en el notebook 02.
- Se corrigió el apellido de Oğul en la bibliografía de la tesis (decía «Ogğul»).

**Figuras sin usar en el texto** (retiradas después, E28):
`prevalencia_periodos` (sustituida por la Tabla 2.3), `distribucion_variables_mid` y
`busqueda_C` (la curva plana de C se dice en una frase).

### E9 · 21-10-2026 · Jose confirma E3

**Qué.** Jose está de acuerdo con E3: no se hace una tercera ronda de la rejilla del Gradient
Boosting. E3 queda como interpretación aceptada del punto de decisión de D5.

## Segunda ronda: revisión del tutor, hojas 9–12 (22-10-2026)

Davier revisó la sección 2.2 de E8. Aprueba el trabajo y la redacción en general, pero pide
tres cambios de método (hojas 9 y 12) y una lista de correcciones de redacción (hojas 10–12).
Lo que sigue **cambia E4–E7**; esas entradas no se borran. Jose pide que la tesis presente solo
el protocolo definitivo, no la historia de las rondas.

### E10 · 22-10-2026 · Protocolo de la segunda ronda (fijado antes de ver resultados)

**Qué cambia y por qué.**
1. **Una sola configuración para los tres cortes** (hoja 9: «que el modelo intermedio use un C
   diferente»; hoja 12: «me sentiría mucho mejor si usáramos exactamente el mismo modelo en las
   tres etapas»). Cada familia de modelos usa una única combinación de hiperparámetros en los
   tres cortes: en cada búsqueda (bucle interno y búsqueda final) se elige la combinación de
   mayor ROC-AUC media sobre los tres cortes. Cada corte se sigue ajustando con sus propios
   datos (D3), porque las variables disponibles cambian (`last_score` no existe en el primero).
   Se aplica a los tres modelos, para que la comparación sea homogénea. Sustituye en parte a D4
   (búsqueda por corte).
2. **Selección de variables con los modelos de árboles, dentro de la validación** (hoja 9:
   «que se trabaje con todos los features, desaprovechando los insights que da el feature
   selection»; hoja 12: parsimonia, «¿no es mejor simplificar?»; hoja 11: «quítale a lo de la
   feature selection lo relativo a la regresión logística»).
   - *Ranking*: importancia por reducción de impureza del Random Forest y del Gradient
     Boosting, normalizada a suma 1 en cada modelo y corte, y promediada sobre los dos modelos
     y los cortes en que existe la variable. Un único orden para los tres cortes (en el primero
     se omiten las de `last_score`, que allí coinciden con la nota media). En cada pliegue
     externo se calcula solo con sus datos de entrenamiento, con los árboles reajustados con la
     configuración que eligió su bucle interno.
   - *Número de variables k*: se busca en el bucle interno junto con C (k = 1, …, 13 primeras
     del orden). Regla de **un error típico** (Hastie et al., 2009, §7.10): la menor k cuya
     ROC-AUC media (sobre cortes y periodos internos) no baje del máximo menos el error típico
     de ese máximo (desviación entre periodos internos, ya promediados sobre cortes, entre √K).
     Con esa k, el C de mayor ROC-AUC.
   - La logística deja de usarse para seleccionar: pasa de L1 a **L2** (la penalización
     habitual, más estable con variables correlacionadas). Sustituye a la L1 de D5 y a la regla
     de parsimonia de D10 (E5).
3. **Umbral del modelo final** (hoja 9: «que no se seleccione un umbral»; hoja 12: «en ningún
   momento se discute el umbral»). Jose fija el objetivo: detectar al mayor número posible de
   estudiantes en riesgo; la precisión es secundaria (la lista se puede priorizar después por
   probabilidad). Criterio: **maximizar F2** (van Rijsbergen, 1979; la sensibilidad pesa el doble
   que la precisión), calculado sobre la clase en riesgo, por lo que tiene en cuenta el
   desbalance. Un umbral por corte (la prevalencia cambia). Se elige con predicciones fuera de
   pliegue de entrenamiento: en la validación anidada, con las del bucle interno, y se aplica al
   periodo externo (así se estima también la sensibilidad y la precisión esperadas); en el
   modelo final, con las de la búsqueda con los tres periodos. Sustituye a D7 (umbral 0,5) en lo
   relativo al umbral; sigue sin calibrarse.
4. **Elección de la familia (D9, sin cambios en la regla).** Se compara la logística *con su
   selección de variables* (el procedimiento completo de 2) frente al RF y al GB con todas las
   candidatas, todos con configuración común a los cortes. Si un modelo de árboles cumpliera D9,
   se para y se pregunta a Jose antes de seguir.
5. **Evolución entre cortes** (petición de Jose). Sin cambiar el modelo: en la prueba, para los
   estudiantes presentes en dos cortes consecutivos, cómo cambia su alerta (sale de riesgo,
   sigue, entra) y qué proporción está realmente en riesgo en cada grupo. Una figura y un
   párrafo. OULAD no registra tutorías, así que solo describe cambios espontáneos.

**Reutilización.** Las búsquedas internas de E2 (por corte) del RF y del GB son exactamente las
tablas que necesita la elección común: se promedian, no se recalculan. La logística se recalcula
(cambia a L2). Los resultados de la primera ronda pasan a `artifacts/ronda1/`.

**Transparencia sobre la prueba.** 2014J ya se abrió una vez (E7) con el modelo de la primera
ronda. Se volverá a evaluar una sola vez con el modelo de esta ronda. Los cambios vienen de la
revisión del tutor, no de los resultados en 2014J, y todas las reglas (1–4) quedan fijadas aquí
antes de reabrirla.

### E11 · 22-10-2026 · Validación con configuración común: sin cambios de grilla en los árboles

**Qué.** `notebooks/02_validacion_anidada` con la elección común a los cortes (E10.1).
Configuraciones elegidas en los pliegues externos: RF con `min_samples_leaf` = 50 y
`max_features` = √p en los tres; GB con (0,02; profundidad 2; 300 árboles) en dos y
(0,05; 3; 100) en uno. El GB toca el mínimo de `learning_rate` y el máximo de `n_estimators` en
2 de 3 pliegues: es el caso compensado de E3 (aceptado por Jose en E9), así que no se amplía.
**Resultado.** Usar una configuración común cuesta en la búsqueda interna menos de 0,001 de
ROC-AUC en cualquier corte y modelo. La ROC-AUC externa del RF y del GB queda como en E4.

### E12 · 22-10-2026 · Se amplía la grilla de C hacia abajo (regla de D5)

**Qué.** En la selección de variables de la logística (`notebooks/03_seleccion`, primera
ejecución), la regla de un error típico eligió k = 5, 4, 4 variables, con C = 0,001 (el mínimo
de la grilla) en 2 de los 3 pliegues externos. Por la regla de D5 se añade C ∈ {0,0001; 0,0003}
a la grilla de la logística (una sola grilla, también para la logística con todas las variables
del notebook 02) y se repiten ambos notebooks; solo se calculan las combinaciones nuevas.
**Por qué.** Regla escrita. Con pocas variables muy correlacionadas (las notas y la actividad
reciente), una penalización L2 fuerte reparte el peso entre ellas y generaliza algo mejor a otro
periodo; las diferencias entre valores de C son de la tercera cifra decimal.
**Resultado de la primera ejecución** (para que conste): ningún árbol cumple D9; la logística con
4–5 variables tiene una ROC-AUC externa de 0,817 · 0,906 · 0,938, frente a 0,822 · 0,908 · 0,940
con todas. El umbral F2 dio, en el periodo externo, una sensibilidad de 0,93 · 0,90 · 0,90 y
marcó al 78 % · 54 % · 45 % de los estudiantes.

### E13 · 22-10-2026 · D9 por periodo y grilla de C cerrada en 0,001 (consultado con Jose)

**Qué.** (1) En D9, la diferencia pareada y su IC se calculan sobre la **media de la ROC-AUC de
cada periodo externo** (la métrica de la tabla de resultados), con bootstrap de estudiantes
dentro de cada clase (`comun.bootstrap_auc_by_period`), en lugar de juntar las predicciones de
los tres periodos. (2) La grilla de C vuelve a {0,001, …, 100}; se deshace la ampliación de E12.
**Por qué.** Con la grilla de E12, la regla de un error típico eligió C = 0,0001 en dos pliegues y
0,03 en el tercero. Con C tan pequeño las probabilidades se aplastan hacia la prevalencia, cada
pliegue en una escala distinta, y al juntar las predicciones la ROC-AUC conjunta de la logística
cayó a 0,790 · 0,881 · 0,918, aunque la de cada periodo apenas cambió (0,816 · 0,906 · 0,937).
Con D9 literal «ganaban» los árboles en los tres cortes por 0,025–0,028, una diferencia que mide
la incoherencia de escalas entre pliegues y no la discriminación. Con el estadístico por
periodo, la diferencia RF − logística es 0,006 · 0,005 · 0,008 (desviación de la logística
0,008 · 0,013 · 0,006) con cualquiera de las dos grillas: ningún árbol cumple D9. Bajar C de
0,001 no cambia la ROC-AUC (la L2 converge a un orden fijo cuando C → 0) y vuelve a tocar el
borde, así que la ampliación no aporta y perjudica al umbral. Jose eligió esta opción.
**Nota técnica.** Durante la sesión se desinstaló Python 3.13 del sistema (`apt autoremove`), y
el entorno se recreó con Python 3.14 y las mismas versiones fijas de `requirements.txt`.

### E14 · 22-10-2026 · Modelo final de la segunda ronda (antes de abrir la prueba)

**Qué.** Aplicación de D9 por periodo (E13) en `notebooks/03_seleccion`: ningún modelo de árboles
supera a la logística con selección en los tres cortes (RF − logística: 0,006 [0,004; 0,007] ·
0,005 [0,004; 0,007] · 0,008 [0,006; 0,009]; desviación de la logística entre periodos 0,009 ·
0,013 · 0,006). La logística con selección rinde 0,817 ± 0,009 · 0,906 ± 0,013 · 0,938 ± 0,006
en la validación anidada; la regla de la nota, 0,801 · 0,893 · 0,929 (diferencia +0,016 ·
+0,013 · +0,008, IC por encima de 0). Frente a la logística con las 13 variables pierde 0,005 ·
0,002 · 0,002, menos que la variación entre periodos. El umbral F2, validado en el periodo
externo, detectó al 93 % · 90 % · 90 % de los estudiantes en riesgo con precisión 0,53 · 0,66 ·
0,71. Con esto, el modelo final queda fijado así (`notebooks/04_modelo_final`) y no se cambia:

| | Primer corte | Segundo corte | Tercer corte |
|---|---|---|---|
| Modelo | `StandardScaler` → `LogisticRegression` (L2), C = 0,001 | ídem | ídem |
| Variables | `mean_score_with_zeros` (valor y percentil), `recent_clicks_28d` | las 3 + `last_score` (valor) | ídem |
| Umbral (F2) | 0,213 | 0,235 | 0,263 |
| Entrenamiento | 17 512 filas (2013B, 2013J, 2014B) | 16 245 | 15 344 |

El orden común de las variables viene del RF (hojas de 50, √p) y del GB (0,02 · profundidad 3 ·
300 árboles) con la configuración común de la búsqueda con los tres periodos; la regla de un
error típico eligió k = 4 (máximo en k = 13, 0,890; error típico 0,005). Modelos guardados en
`artifacts/final_<corte>.joblib`. Línea base: regla de la nota (D8).

### E15 · 22-10-2026 · Evaluación en 2014J del modelo de E14 (segunda y última apertura)

**Qué.** `notebooks/05_prueba` con el modelo de E14. Antes, ensayo en una copia temporal con
2014B en lugar de la prueba (sin abrirla ni guardar nada). Se ejecutó una segunda vez, con los
mismos números, solo para corregir las etiquetas de la figura de evolución y añadir las
conclusiones. No se cambió nada del modelo.
**Resultados.**

| Corte | Modelo final [IC 95 %] | Regla de la nota [IC 95 %] | Diferencia [IC 95 %] |
|---|---|---|---|
| Primero | 0,787 [0,778; 0,797] | 0,754 [0,744; 0,765] | 0,033 [0,029; 0,038] |
| Segundo | 0,883 [0,875; 0,891] | 0,845 [0,836; 0,854] | 0,038 [0,034; 0,043] |
| Tercero | 0,923 [0,917; 0,930] | 0,899 [0,892; 0,907] | 0,024 [0,020; 0,028] |

Con el umbral F2, sensibilidad 0,929 · 0,894 · 0,887 y precisión 0,459 · 0,532 · 0,616 (la regla:
0,378 · 0,488 · 0,582 y 0,680 · 0,773 · 0,822); F2 0,771 · 0,787 · 0,816 frente a 0,415 · 0,527 ·
0,618. Reciben alerta el 66 % · 36 % · 22 % de quienes aprueban. Evolución de la alerta: de quienes
salen de ella, está realmente en riesgo el 9,5 % · 11,7 %; de quienes entran, el 43,9 % · 33,3 %;
de quienes siguen, el 53,7 % · 64,3 %; de quienes nunca la tuvieron, el 4,5 % · 3,4 %.
Respecto a la primera ronda (E7), la ROC-AUC es algo menor (0,794 · 0,889 · 0,927 con 11–13
variables), dentro de la variación entre periodos; se usan 3–4 variables.

### E16 · 22-10-2026 · Redacción de la segunda versión de la sección 2.2

**Qué.** El texto de la sección de entrenamiento de la tesis, reescrito con el protocolo de E10–E15 y las correcciones de
redacción de las hojas 10–12: fuera el segundo párrafo de la introducción, la crítica a la
partición aleatoria, el párrafo de la incertidumbre, el tercer párrafo de la métrica, el de la
regla de elección del modelo, el de la ROC-AUC menor en 2014J y las frases que solo respondían
a la revisión (pocos modelos, histogramas, PCA, calibración, reponderación). Los modelos, en
párrafo; «grilla» y *GridSearchCV*; la línea base, definida por el umbral de 40. Subsecciones
nuevas: selección de variables (2.2.4) y umbral de decisión (2.2.5), y el párrafo de la evolución
de la alerta entre cortes en la prueba. 9 figuras y 8 tablas. Se reformularon tres frases que
comentaban diferencias de la tercera cifra decimal. El texto de la sección de variables no cambia: ya
remite la selección supervisada al entrenamiento. Nueva referencia: van Rijsbergen (1979).
Figuras ya sin uso en el texto: `coeficientes_logistica`, `importancias_modelos`,
`busqueda_C`, `prevalencia_periodos` y `distribucion_variables_mid` (de la primera ronda).
`INFORME_PARA_EL_TUTOR.md` y su PDF, rehechos para las hojas 9–12.

### E17 · 29-10-2026 · Comprobación de la calibración del modelo final en 2014J

**Qué.** `scripts/figura_calibracion.py` (con `ABRIR_PRUEBA=1`, solo lectura): diagrama de
calibración por deciles del modelo de E14 en 2014J. No se recalibra ni se cambia nada.
Proporción en riesgo (entrenamiento / prueba): 0,433 / 0,377 · 0,389 / 0,314 · 0,353 / 0,282;
probabilidad media en la prueba 0,431 · 0,369 · 0,324; mayor sobreestimación de un decil 0,135 ·
0,102 · 0,125; proporción real en riesgo a ±0,05 del umbral 0,170 · 0,159 · 0,193.
**Por qué.** El sistema muestra la probabilidad al profesor y el umbral se lee en costes (Elkan,
2001), lo que supone calibración (consulta de Davier sobre el umbral, 29-10).

### E18 · 02-11-2026 · Umbral en la proporción de desaprobados del entrenamiento (reescritura de Davier)

**Qué.** El umbral F2 de E14 (0,213 · 0,235 · 0,263) se sustituye por la proporción de
desaprobados de los periodos de entrenamiento en cada corte: 0,433 · 0,389 · 0,353. No cambia el
modelo (coeficientes, variables ni C). `scripts/umbral_prevalencia.py` reproduce las cifras de la
reescritura del Cap. 2 de Davier (30-10) y genera las figuras de la sección 2.4
(`curvas_roc_prueba_umbral`, `comparacion_misma_lista`, `calibracion_prueba_umbral`,
`evolucion_alerta_prueba_umbral`); cifras en `artifacts/06_umbral_prevalencia.json`.
**Resultado en 2014J.** Marca al 43,2 · 37,2 · 34,7 %; sensibilidad 0,698 · 0,781 · 0,836;
precisión 0,609 · 0,660 · 0,679; especificidad 0,729 · 0,816 · 0,845. Con la misma lista que la
regla de la nota (≈ 20 %): sensibilidad 0,417 · 0,538 · 0,628 frente a 0,378 · 0,488 · 0,582, y
precisión 0,750 · 0,851 · 0,887 frente a 0,680 · 0,773 · 0,822. Coincide con la reescritura salvo
un estudiante por grupo en la evolución de la alerta (Davier usó los umbrales redondeados).
**Por qué.** Crítica de Davier (30-10, D4-35 a D4-42): el umbral F2 marcaba al 76 % en el primer
corte y β y los costes no tenían fundamento. El nuevo no tiene parámetros y no usa datos de prueba.
**Nota de protocolo.** Se adopta después de la segunda y última apertura de la prueba (E10, E15).
La ROC-AUC no depende del umbral; las cifras de clasificación de 2014J sí son posteriores a esa
apertura. Pendiente de que Jose decida si la tesis lo menciona.

### E19 · 06-11-2026 · Comprobación de la intercambiabilidad de los periodos (crítica A1 del oponente)

**Qué.** `scripts/validacion_temporal.py`, con la configuración del modelo final (logística L2,
C = 0,001, primeras 4 variables del orden común, 3 en el primer corte) y solo los periodos de
entrenamiento. No cambia nada del modelo.
**Resultado.** (1) Con un solo periodo fuente del mismo tamaño, entrenar con el pasado o con el
futuro da la misma ROC-AUC: objetivo 2013J, 2013B (pasado) 0,814 · 0,911 · 0,944 frente a 2014B
(futuro) 0,811 · 0,911 · 0,942; en los seis pares, diferencias ≤ 0,007 y sin ventaja del futuro.
(2) El pliegue de 2014B ya se entrena solo con periodos anteriores; en el de 2013J, entrenar solo
con 2013B en lugar de 2013B y 2014B cambia la ROC-AUC en menos de 0,001 (0,8129/0,8138 ·
0,9106/0,9106 · 0,9427/0,9435). (3) La elección de k y C con validación hacia delante (2013B →
2013J; 2013B + 2013J → 2014B) y la regla de un error típico da k = 4 y C = 0,001, el modelo final.
(4) En 2014J, la ROC-AUC del modelo baja respecto a la validación 0,035 · 0,025 · 0,017 y la de la
regla de la nota 0,047 · 0,048 · 0,030: la caída es del periodo, no del sentido del tiempo.
**Por qué.** Dos de los tres pliegues de D4 entrenan con periodos posteriores; la tesis decía que la
validación respetaba el orden temporal. El autor decide (06-11) declarar el supuesto de
intercambiabilidad y justificarlo con esta comprobación; 2014J sigue como prueba porque reproduce el
uso y su estimación no depende del supuesto.

### E20 · 06-11-2026 · El umbral de E18, evaluado fuera de pliegue (crítica A3 del oponente)

**Qué.** `scripts/umbral_validacion.py`: en cada periodo de entrenamiento dejado fuera, el modelo
final se ajusta con los otros dos y el umbral es la proporción en riesgo de esos dos. No abre 2014J
ni cambia nada.
**Resultado.** Marca al 42,3 · 39,6 · 38,8 %; sensibilidad 0,694 · 0,798 · 0,853; especificidad
0,783 · 0,860 · 0,865; precisión 0,709 · 0,784 · 0,775. Es el mismo comportamiento que en 2014J
(E18), y el del umbral F2 de E14 en la validación era marcar al 81 · 55 · 46 %.
**Por qué.** E18 se adoptó después de abrir la prueba. El autor decide (06-11) decirlo en la tesis:
el defecto del umbral F2 ya se veía en la validación, el nuevo no tiene parámetros ni usa datos de
prueba, y esta comprobación muestra que su comportamiento se conocía sin la prueba.

### E21 · 06-11-2026 · De dónde sale la ventaja del modelo (críticas B4, B5 y B6 del oponente)

**Qué.** Tres comprobaciones en la validación dejando fuera un periodo, con la configuración del
modelo final y sin abrir 2014J: `scripts/aporte_actividad.py`, `scripts/riesgo_con_aprobado.py` y la
regla de la última nota (cálculo de la misma sesión).
**Resultado.** (1) ROC-AUC: regla de la media 0,801 · 0,893 · 0,929; logística solo con las notas del
modelo 0,806 · 0,898 · 0,933; modelo final 0,817 · 0,907 · 0,939; solo la actividad reciente 0,741 ·
0,845 · 0,891. Recombinar las notas aporta 0,005 · 0,006 · 0,003 y la actividad 0,011 · 0,008 · 0,006
(69 · 57 · 60 % de la ventaja), positiva en los tres periodos de cada corte. (2) Regla de la última
nota: 0,883 · 0,886 en los cortes 2 y 3, peor que la media. (3) Con el umbral de E18 fuera de pliegue,
de los estudiantes en riesgo tienen todavía la media aprobada (≥ 0,40) el 63,0 · 46,6 · 34,9 %; la
regla no marca a ninguno y el modelo marca al 51,4 · 56,7 · 58,3 %, con precisión 0,574 · 0,598 ·
0,497 entre los de media aprobada.
**Por qué.** El oponente (B4–B6) dice que la actividad aporta poco, que la detección no es temprana y
que la referencia es débil. El autor decide (06-11) explicar en la tesis de dónde sale la ventaja, qué
significa «temprana» y por qué no se predice antes de la primera nota.

### E22 · 06-11-2026 · La selección de variables no depende del orden de importancia (crítica C7 del oponente)

**Qué.** `scripts/orden_variables.py`: la regla de un error típico de la tesis (k y C, media de los
tres cortes, validación dejando fuera un periodo) con cuatro órdenes de las variables. Solo periodos
de entrenamiento; no cambia nada.
**Resultado.** (a) Impureza en RF y GB (el de la tesis): 3 de notas + `recent_clicks_28d_pct`,
C = 0,001, ROC-AUC 0,8876. (b) Permutación en el periodo dejado fuera con RF y GB: 3 de notas +
`days_since_last_activity_pct`, C = 0,001, 0,8871. (c) Permutación con la logística (C = 0,001, 13
columnas): 4 de notas + `days_since_last_activity_pct`, C = 0,001, 0,8867. (d) Selección hacia
delante con la logística: `mean_score_with_zeros_pct`, `recent_clicks_28d_pct`,
`days_since_last_activity_pct`, C = 100, 0,8878. Máximo con todas las variables: 0,8899. En (d),
`repeated_course` sube al quinto puesto (último con la impureza), pero no entra en ninguna selección.
**Por qué.** El oponente señala el sesgo de la impureza hacia las variables continuas (Strobl et al.,
2007) y que el orden de los árboles se aplica a la logística. El autor decide (06-11) decirlo en la
tesis con esta comprobación; el método y el modelo final no cambian.

### E23 · 07-11-2026 · Corrección: variables del primer corte en las comprobaciones E19–E21

**Qué.** `validacion_temporal.py`, `umbral_validacion.py` y `riesgo_con_aprobado.py` construían las
variables del primer corte quitando primero `last_score` y tomando después las cuatro primeras, lo
que añadía `days_since_last_activity_pct` (cuatro variables en lugar de tres). La tesis toma los k
primeros puestos del orden común y quita después `last_score` (`04_modelo_final`: tres variables en
el primer corte). Se corrigen los tres scripts y `penalizacion_y_regla.py`; `aporte_actividad.py`
y `orden_variables.py` ya eran correctos.
**Resultado corregido.** E19: objetivo 2013J en el primer corte, 0,812 con 2013B y con 2014B (antes
0,814 y 0,811); validación hacia delante igual a la de la tesis en el pliegue de 2013J (0,8121); la
selección hacia delante elige **k = 5** y C = 0,001 (ROC-AUC 0,8874; k = 4 da 0,8845), no k = 4: la
quinta variable es `days_since_last_activity_pct`. E20: sensibilidad del primer corte 0,691 y
especificidad 0,781 (antes 0,694 y 0,783). E21 (3): el modelo marca al 50,9 % de los estudiantes en
riesgo con media aprobada en el primer corte, con precisión 0,569 (antes 51,4 % y 0,574). Los cortes
segundo y tercero no cambian.
**Consecuencia en la tesis.** En 2.3.4 se corrigen las cifras del primer corte y la frase de la
selección hacia delante («elige el mismo valor de C y una variable más»); las cifras de 2.4.1
redondean igual.

### E24 · 07-11-2026 · Regla de un error típico con tres periodos y efecto de C (críticas C8 y C9)

**Qué.** `scripts/penalizacion_y_regla.py`, con las variables del primer corte ya corregidas (E23).
Solo periodos de entrenamiento; no cambia nada.
**Resultado.** (C8) Con tres valores, el IC del 95 % de la desviación real es [0,52; 6,28] veces la
estimada. La regla elige k = 4 si el error típico es 0,5, 1 o 2 veces el estimado; k = 5 con 0,25;
k = 2 con 3; k = 1 con 6. ROC-AUC: k = 3 0,8792; k = 4 0,8876; k = 5 0,8893; k = 13 0,8899. (C9) ROC-AUC
del modelo final según C: 0,8880 (1e-5) · 0,8876 (0,001) · 0,8855 (1) · 0,8854 (100). Coeficientes con
C = 1 (frente a C = 0,001): primer corte −0,643 · −0,557 · −0,502 (−0,489 · −0,497 · −0,406); tercer
corte nota media −1,060, percentil −0,911, nota de la TMA −0,208, actividad −0,961 (−0,614 · −0,575 ·
−0,438 · −0,588). La variable que más aumenta el riesgo de cada estudiante coincide con la de
C = 0,001 en el 93,3 · 82,7 · 77,5 % de los estudiantes. Correlación entre el valor y el percentil de
la nota media: 0,869 · 0,915 · 0,929.
**Por qué.** El autor decide (07-11) decir en la tesis que la elección de k se mantiene entre la
mitad y el doble del error típico, que la ROC-AUC es casi plana respecto a C y que las contribuciones
de cada variable describen el modelo y no la importancia causal.

### E25 · 07-11-2026 · Sensibilidad al retraso de la nota y fuente de la Open University (crítica D10)

**Qué.** `scripts/sensibilidad_retraso.py` reconstruye el conjunto con `GRADE_LAG` = 7, 21 y 28 días
(sin tocar `dataset.parquet`) y repite la validación dejando fuera un periodo con la configuración del
modelo final. Solo periodos de entrenamiento.
**Resultado.** ROC-AUC del modelo (cortes 1 · 2 · 3) y ventaja sobre la regla: 7 días 0,816 · 0,903 ·
0,937 (0,015 · 0,012 · 0,009); 14 días 0,817 · 0,907 · 0,939 (0,016 · 0,014 · 0,010); 21 días 0,816 ·
0,909 · 0,939 (0,016 · 0,016 · 0,010); 28 días 0,822 · 0,913 · 0,942 (0,023 · 0,020 · 0,013). Un retraso
real mayor no inflaría los resultados: con 21 o 28 días el modelo rinde igual o algo mejor.
**Fuente.** The Open University, *TMA and iCMA Policy*, v. 2.2 (en vigor desde el 1 de julio de 2025),
§ 5.3.1: «Tutors usually aim to mark and return your TMA within ten working days from the submission
cut-off date». Copia en `Teorico/repo/state_of_art/papers/ou2025tmapolicy.pdf`. No se encontró la
versión de 2013–2014.
**Por qué.** El oponente señala que los catorce días se justificaban solo con el reglamento cubano. El
autor decide (07-11) fundamentarlos en la práctica de la Open University, mantener la norma cubana como
coincidencia con el lugar de uso y decir que el resultado no depende de la cifra.

### E26 · 07-11-2026 · Protocolo de validación con datos históricos de EVEA (fijado antes de recibirlos)

**Contexto.** Respuesta a la crítica D11 del oponente (distancia OULAD–EVEA). La solicitud de datos
(`Thesis-Software/Solicitud_datos_EVEA.pdf`, 31-10) pide las tablas estándar de Moodle de cursos
cerrados de al menos cuatro periodos. El autor confirma (07-11) que **no se reentrena**: los datos de
EVEA son un segundo conjunto de prueba del modelo ajustado con OULAD (decisión del 02-11). Este
protocolo se escribe antes de recibir ni ver ningún dato y no se cambia después de verlos; si algo
resulta inaplicable (p. ej., faltan tablas), la desviación se anota aquí con su motivo antes de
calcular ningún resultado.

1. **Modelo y umbral, sin cambios.** El paquete activo del sistema (coeficientes de la Tabla 2.6) y
   el umbral de la proporción en riesgo de OULAD (0,433 · 0,389 · 0,353), con la regla de ajuste del
   sistema según la posición de la evaluación en el curso (fronteras en el 37,5 y el 62,5 %).
2. **Conversión de notas.** La equipercentil del sistema, con el periodo más antiguo de los datos como
   «cursos anteriores» de la plataforma. Ese periodo no se evalúa; se evalúan los posteriores. Dentro
   de un periodo evaluado, los cursos anteriores son todos los de periodos previos.
3. **Etiqueta.** En riesgo = nota final 2 o baja; las bajas se reconstruyen con los eventos de
   matrícula del registro (sección 3.3 de la solicitud). Sin nota final conocida, el estudiante se
   excluye y se cuenta.
4. **Momentos de predicción.** (a) Principal: los tres momentos equivalentes a los cortes del Cap. 2
   (la evaluación con fecha límite más próxima al 25 % y al 50 % del curso y la última), una vez
   publicada su nota según la regla del sistema (90 % de las notas de quienes entregaron). (b)
   Secundario: cada vez que una tarea queda completa, como hace el sistema.
5. **Cursos.** Los que cumplen los criterios de la solicitud y tienen en el momento de predicción al
   menos 20 estudiantes, al menos 5 en riesgo y al menos 5 que aprueban. Se informa cuántos cursos y
   estudiantes quedan fuera en cada filtro.
6. **Métricas.** ROC-AUC del modelo y de la regla de la nota (orden por la nota media con ceros) por
   curso y momento, y su media sobre cursos; diferencia modelo − regla con IC del 95 % por *bootstrap*
   de cursos (1000 réplicas); calibración por deciles; porcentaje de marcados, sensibilidad,
   especificidad y precisión con el umbral.
7. **Hipótesis.** Principal: en EVEA el modelo supera a la regla de la nota (IC de la diferencia por
   encima de 0) en cada uno de los tres momentos. Secundarias: la caída de la ROC-AUC respecto a 2014J
   (0,787 · 0,883 · 0,923) y si la calibración y el porcentaje de marcados se mantienen.
8. **Compromiso.** Se informan los resultados sean cuales sean, también si el modelo no supera a la
   regla. El modelo, el umbral y este protocolo no se modifican tras ver los datos de EVEA. Los datos
   se anonimizan antes de procesarlos y en la tesis solo aparecen resultados agregados.

### E27 · 08-11-2026 · Herramienta de validación con EVEA lista y probada (antes de recibir los datos)

**Qué.** En `Thesis-Software`: `alerta/fuentes/tablas_moodle.py` (tablas de la base de datos de Moodle,
en el formato de la solicitud, → esquema canónico), `alerta/validacion_evea.py` (protocolo E26) y el
subcomando `python -m alerta validar-evea --tablas <carpeta> --salida <carpeta>`. Prueba:
`scripts/exportar_tablas_lab.py` exporta del Moodle de laboratorio el curso que reproduce AAA_2014J y
`tests/test_tablas_moodle.py` comprueba que las tablas dan los mismos estudiantes y las mismas trece
variables que OULAD en los tres cortes (diferencia 0) y la misma ROC-AUC (0,7922 · 0,8301 · 0,9219).
Batería completa: 78 pruebas superadas.
**Hallazgos para los datos reales.** (1) Moodle borra de `grade_grades` las notas de quien se
desmatricula y solo las guarda en `grade_grades_history`; la fuente las recupera (107 en la prueba).
(2) La fecha de publicación de cada nota es la más antigua entre el historial y `grade_grades`. (3) Si
`resultado_final.csv` trae la fecha de baja, prevalece sobre el registro de matrícula. (4) Los periodos se
deducen de la fecha de inicio (semestres de la UH) hasta que la institución entregue su calendario.
**Pendiente al recibir los datos.** Comprobar que llegan las tablas y columnas esperadas; cualquier ajuste
de la lectura se anota aquí antes de calcular resultados (E26).

### E28 · 08-11-2026 · Reordenamiento de la carpeta

**Qué.** Para que la carpeta muestre solo el flujo que sigue la tesis, se retira de ella lo que ya no
forma parte de él: las figuras de la primera ronda (`coeficientes_logistica`, `importancias_modelos`,
`busqueda_C`, `prevalencia_periodos`, `distribucion_variables_mid`), las de la prueba con el umbral F2
(`curvas_roc_prueba`, `matrices_confusion_prueba`, `evolucion_alertas_prueba`) y la calibración de E17
(`calibracion_prueba`), el borrador de la sección 2.2 de E16 con el informe para el tutor, las notas de
contexto teórico y los artefactos de la primera ronda y de la primera rejilla, salvo `01_datos.json`,
el resumen de E1, que sigue vigente. `README.md` y `ESTADO.md` se reescriben con el flujo final (E10–E27).
**Por qué.** Petición de Jose. Las entradas anteriores de esta bitácora no se tocan, incluido el
registro de la primera apertura de 2014J (E7).
**Resultado.** No cambia ningún código, dato, modelo ni cifra. `01_datos`, `05_prueba` y
`figura_calibracion.py` vuelven a generar las figuras retiradas si se ejecutan.

### E29 · 08-11-2026 · Lecturas del conjunto de prueba 2014J (crítica R1 del oponente, ronda Q)

**Qué.** Relación completa de las veces que se leyó 2014J, para que conste en un solo sitio. No cambia
ninguna entrada anterior.
- **Evaluaciones del modelo:** E7 (modelo de la primera ronda) y E15 (modelo final). Antes de cada una,
  ensayo con 2014B en lugar de la prueba, sin abrirla.
- **Relecturas sin cambiar el modelo:** E17 (calibración, `scripts/figura_calibracion.py`) y E18 (umbral
  de la proporción en riesgo, `scripts/umbral_prevalencia.py`, adoptado después de verla; E20 lo evalúa
  fuera de pliegue).
- **Lecturas fuera de esta carpeta, sin `ABRIR_PRUEBA`:** las simulaciones del sistema del Cap. 3 en
  `Teorico/repo/revisions/Davier/2026-11-3/simulaciones/` (`sim_escala.py`, `sim_escala2.py`,
  `sim_parcial.py`, `sim_regla.py`, 03-11) y `2026-11-05/simulaciones/sim_cma.py` (05-11) importan
  `build_dataset` y calculan con las etiquetas de 2014J. Informaron decisiones del sistema: la conversión
  de la escala 2-5 (los porcentajes de marcados que separaban las conversiones eran de 2014J), el umbral
  del 90 % de notas, la regla del ajuste por posición y el tratamiento de las CMA. Ninguna cambió el
  modelo ni su umbral, y las cifras que la tesis da de esas decisiones son de los periodos de
  entrenamiento.
**Consecuencia.** La comprobación de la conversión se repite en los periodos de entrenamiento, también
para los marcados (E30, F2). Desde aquí ninguna comprobación lee 2014J.
**Nota (crítica R7).** El punto 4 de E19 (caídas de 0,035 · 0,025 · 0,017) se calculó antes de la
corrección de E23; la tesis usa las de la Tabla 2.5 (entre 0,015 y 0,030).

### E30 · 08-11-2026 · Comprobaciones de la ronda Q (tercera pasada del oponente)

**Qué.** `scripts/comprobaciones_ronda_q.py` (salida en `artifacts/ronda_q.txt`), con la configuración del
modelo final y solo los periodos de entrenamiento; y `scripts/figuras_seleccion_variables.py`, que pasa a
usar solo esos periodos (Fig. 2.2 regenerada). No cambia nada del modelo.
**Resultado.**
- *A2, entregas tardías:* el 4,1 % de las entregas llega en los catorce días anteriores al corte. Tratadas
  como sin nota, la ROC-AUC del modelo cambia −0,006 · −0,003 · −0,001 (sin contarlas) y −0,003 · −0,009 ·
  −0,002 (como no entregadas); la ventaja sobre la regla, +0,018 · +0,015 · +0,010 y +0,022 · +0,016 · +0,012.
- *D4, grupos de veinte:* ROC-AUC conjunta 0,817 → 0,811, 0,907 → 0,904, 0,939 → 0,935; el modelo queda por
  debajo de la regla en el 38 · 29 · 31 % de los grupos; percentiles 10-90 de la ROC-AUC por grupo
  0,67-0,95 · 0,80-1,00 · 0,86-1,00. Clases de entrenamiento: de 326 a 1905 estudiantes.
- *D1, calibración fuera de pliegue:* decil más sobrestimado +0,122 · +0,119 · +0,116; más infraestimado
  −0,143 · −0,152 · −0,176; mayor desviación de una clase 0,227 · 0,145 · 0,145.
- *D2, clase positiva:* las bajas son el 33,2 · 19,8 · 6,5 % de los estudiantes en riesgo; ROC-AUC frente a
  los que aprueban, bajas 0,784 · 0,887 · 0,914, desaprobados 0,834 · 0,912 · 0,941.
- *F2, conversión:* en entrenamiento, ROC-AUC como mucho −0,005; marcados, equipercentil entre −0,7 y +2,0
  puntos, valores fijos 50/70/90 entre +0,5 y +4,8 puntos, según el supuesto de bandas.
- *N5:* 19 tipos de actividad en entrenamiento, 5 en las siete asignaturas, 7 en una o dos, los cinco más
  usados 91,8 %, 11 por debajo del 0,3 %; actividades de evaluación entre el 0 (AAA) y el 45,3 % (CCC).
- *N6:* el plazo de la última TMA vence entre el 59,3 y el 83,3 % del curso (mediana 77,6 %).
**Por qué.** Críticas del oponente de la ronda Q. El autor pide resolverlas todas (07-11).

### E31 · 08-11-2026 · Etiqueta de EVEA: convocatoria ordinaria (antes de recibir los datos)

**Qué.** Precisa el punto 3 de E26: en riesgo = nota final 2 en la **convocatoria ordinaria** o baja,
según la condición final que registre la institución (libro de calificaciones de EVEA o fuera de él,
sección 3.6 de la solicitud). Si los datos no distinguen convocatorias, se usa la nota final registrada y
se declara.
**Por qué.** Crítica D2 del oponente: la etiqueta de EVEA no estaba definida del todo. La convocatoria
ordinaria es la que puede cambiar una intervención durante el curso. Se fija antes de recibir ni ver
ningún dato, como exige E26.

### E32 · 13-11-2026 · Publicación del repositorio (Thesis-Training)

**Qué.** La carpeta se publica en `github.com/LuciferEvil01/Thesis-Training` con el flujo final: datos,
notebooks, scripts, figuras y esta bitácora. Las figuras pasan a la carpeta `figuras/`, y la
preparación del entorno y de los datos queda en `preparar.sh`. No se publican los borradores, las
notas de trabajo ni la configuración del asistente de programación. El repositorio publicado empieza
con un historial nuevo; el historial de trabajo se conserva fuera de él.
**Por qué.** Petición de Jose, para que la tesis remita al código (auditoría del 12-11-2026, AU-14).
**Resultado.** No cambia ningún código de cálculo, dato, modelo ni cifra.
