# Respuesta a la segunda auditoría externa (sobre la v0.9.3)

Fecha: 27 de noviembre de 2026. Versión que responde: v0.9.4. Pruebas: `tests/test_v094.py` (14 nuevas).

## Predicción y garantías estadísticas

| # | Hallazgo | Veredicto | Cambio en v0.9.4 | Prueba |
|---|---|---|---|---|
| 3.1 | El predictor calibrado devolvía siempre la media de los agentes | **Confirmado.** Es el hallazgo más importante: explica por qué "la colmena no supera a un promedio" desde la v0.3 (era un promedio) | El ganador de la dinámica se separa del quórum. `colmena.prediccion`: `media`, `ponderada` o `ganador`, igual en calibración e inferencia. Comparación de las tres en `RESULTADOS.md` | `test_calibrated_colony_selects_winner` |
| 3.2 | Sin umbral certificado, un apoyo futuro alto se aceptaba | **Confirmado** | Umbral = ∞ y estado explícito: el predictor rechaza todo si LTT no certificó | `test_uncertified_ltt_rejects_unseen_support` |
| 3.3 | Quedaba la rama LTT con umbrales de la propia calibración | **Confirmado** | `ltt_threshold` exige rejilla; la rama vieja es `heuristic_threshold`, documentada sin garantía | misma prueba |
| 3.4 | Se aceptaba con la calibración caducada | **Confirmado** | Con calibración no vigente nada se marca como aceptado; la invalidación ocurre antes de modificar los modelos | `test_stale_calibration_rejects` |
| 3.5 | El apoyo dependía del orden de las consultas | **Confirmado** | `BeeColony.decide` usa un generador nuevo por llamada | `test_query_order_invariance` |

## Aprendizaje continuo y persistencia

| # | Hallazgo | Veredicto | Cambio | Prueba |
|---|---|---|---|---|
| 5.1 | Guardar y cargar cambiaba el aprendizaje futuro | **Confirmado** | Dos formatos: `guardar(modo="reanudar")` (datos, lo incorporado, evolución, campaña) y `"despliegue"`; un modelo de despliegue se niega a `incorporar()`. `incorporar` fija su semilla, así la campaña reanudada es idéntica a la continua | `test_resume_training_equivalence` |
| 5.2 | El presupuesto global podía excederse | **Confirmado** | Tope en todos los modos, también para las exploradoras; `incorporar()` falla si se excede (`presupuesto_estricto`). **Pendiente:** identificadores únicos de simulación | `test_budget_all_modes` |
| 5.3 | `usar_error: false` con `parada: mvt` | **Confirmado** | La combinación se rechaza con un error claro | `test_budget_all_modes` |
| 5.4 | El número de simulaciones no es el costo real | **Confirmado**, arreglo parcial | Se cuentan condiciones físicas y ejecuciones por simulador. **Pendiente:** horas de CPU/GPU | `test_resume_training_equivalence` |

## Centinela

| # | Hallazgo | Veredicto | Cambio | Prueba |
|---|---|---|---|---|
| 6.1 | s0 se estimaba con los pacientes de calibración | **Confirmado** | s0 se fija con un conjunto de referencia aparte (o un tercio de la calibración) antes de calibrar | `test_conformal_fixed_score_and_protocol` |
| 6.2 | El protocolo de sondeo debe coincidir | **Confirmado** | `fit(sondeo=...)` fija el protocolo y `patient_pvalues` aplica el mismo | misma prueba |
| 6.3 | El centinela no detectaba cambios en los agentes | **Confirmado** | Huella de los agentes; si cambian, exige recalibrar | misma prueba |

## Red microvascular

| # | Hallazgo | Veredicto | Cambio | Prueba |
|---|---|---|---|---|
| 7.2 | `identidad_rbc` era una identidad por construcción | **Confirmado** | Se devuelve el flujo propagado y se compara con H·Q | `test_network_residuals_and_validation` |
| 7.3 | H y Q finales podían no ser del mismo punto fijo | **Confirmado** | `residuo_constitutivo` y bandera `valida` (convergió, nada sin resolver, residuo pequeño) | misma prueba |
| 7.4 | Casos degenerados | **Confirmado**, arreglo parcial | División protegida con más de dos salidas; se validan diámetros, longitudes, hematocrito de entrada y condiciones de presión. **Pendiente:** detectar componentes desconectados | misma prueba |
| 7.5 | Límites desprendidos del grafo | **Confirmado** | Límites como tensores; el gradiente coincide con diferencias finitas con la restricción activa | `test_network_phase_gradient` |

## Entrenamiento y evaluación

| # | Hallazgo | Veredicto | Cambio | Prueba |
|---|---|---|---|---|
| 8.1 | `evaluar()` promedia errores individuales | **Confirmado** | `evaluar_colectivo()` da MAE de agentes, del ensamble, de la colmena, de lo aceptado, cobertura y riesgo selectivo; `exp_presupuesto.py` reporta ambos | `test_evaluate_collective_prediction` |
| 8.2 | Las arquitecturas por defecto pisaban la configuración | **Confirmado** | La diversidad es un desplazamiento respecto a `modelo.d` y `modelo.n_slices`; con la base por defecto da lo mismo que antes | `test_config_architecture_respected` |
| 8.3 | Sin extinciones, la ecología seguía cambiando | **Confirmado** | `activo.ecologia_en_ajuste: false` congela abundancias; al incorporar se usa el conjunto de evolución y no la validación | `test_frozen_ecology` |
| 8.4 | `load_flexible` no revisaba claves faltantes | **Confirmado** | Error si falta un parámetro que no sea memoria de tamaño variable | `test_missing_checkpoint_parameter` |
| 8.5 | Semillas compartidas entre entrenamiento y reserva | **Confirmado**: con semilla 0 coincidían | Semillas de reserva y prueba en rangos separados; comprobación por huellas (`assert_disjoint`) dentro de `exp_presupuesto.py` | `test_seed_dataset_overlap` |

## Consecuencia para los resultados anteriores

- Todo error reportado con `Gemelo.evaluar()` (los experimentos de presupuesto y parada) es el promedio
  de los errores de los agentes, no el del predictor colectivo.
- Con semilla 0, la reserva de `exp_presupuesto.py` compartía semillas con el entrenamiento.
- Los resultados de sondeo activo no usaban el mismo protocolo en calibración y en uso.
Estos resultados deben regenerarse antes de citarse.

## De acuerdo, sin cambio de código todavía

Secciones 9 a 12: prueba de no inferioridad con margen fijado de antemano, CUSUM y GLR como líneas base,
ablaciones a presupuesto igual, más semillas, datos DPD independientes y la función objetivo global como
dirección teórica. También la recomendación de no añadir más mecanismos y consolidar tres aportaciones:
adaptación por memoria, límites del conocimiento y administración del presupuesto.
