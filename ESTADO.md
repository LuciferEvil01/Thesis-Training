# Estado

**08-11-2026.** El modelo final está fijado y evaluado; la tesis (Cap. 2) ya recoge sus resultados.
Solo queda validarlo con los datos de EVEA, sin reentrenar.

| Paso | Dónde | Estado |
|---|---|---|
| 1 · Datos | `scripts/build_dataset.py`, `notebooks/01_datos` | Hecho (E1) |
| 2 · Validación anidada | `notebooks/02_validacion_anidada` | Hecho (E10, E11) |
| 3 · Selección de variables y elección del modelo | `notebooks/03_seleccion` | Hecho (E12–E14) |
| 4 · Modelo final | `notebooks/04_modelo_final` | Hecho y fijado (E14) |
| 5 · Prueba en 2014J | `notebooks/05_prueba` | Hecho; segunda y última evaluación del modelo (E15) |
| 6 · Umbral | `scripts/umbral_prevalencia.py` | Hecho (E18) |
| Comprobaciones de los revisores | `scripts/` | Hechas (E17, E19–E25, E30) |
| Validación con EVEA | `Thesis-Software`, `python -m alerta validar-evea` | Protocolo fijado (E26, E31) y herramienta probada (E27); esperando los datos |

## Al recibir los datos de EVEA

1. Comprobar que llegan las tablas y columnas de la solicitud.
2. Si hay que ajustar la lectura, anotarlo en `DECISIONES.md` **antes** de calcular nada (E26).
3. Ejecutar la validación y anotar los resultados, sean cuales sean (E26, punto 8).

## Notas

- El modelo, el umbral y el protocolo de EVEA no se cambian. El modelo se evaluó en 2014J dos veces
  (E7, E15) y la prueba se leyó después para la calibración (E17) y el umbral (E18); E29 enumera todas las lecturas.
- `.venv` usa Python 3.14 (E13).
