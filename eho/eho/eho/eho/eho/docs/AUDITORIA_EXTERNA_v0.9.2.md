# Respuesta a la auditoría externa de la v0.9.2

Fecha: 27 de noviembre de 2026. Versión que responde: v0.9.3. Cada punto indica si se confirmó en el
código, qué se cambió y qué prueba lo cubre (`tests/test_v093.py`).

## Errores y vulnerabilidades (sección 3 de la auditoría)

| # | Hallazgo | Veredicto | Cambio en v0.9.3 | Prueba |
|---|---|---|---|---|
| 3.1 | La regla de parada no para con ganancia cero | **Confirmado** | Desigualdad estricta; el tiempo de rendición se evalúa antes; se validan `referencia`, `tau`, `beta` y la finitud de las ganancias | `test_stop_zero_gain`, `test_stop_validates_inputs` |
| 3.2 | La ganancia marginal depende del orden dentro del lote | **Confirmado** | La media móvil se actualiza una vez por ronda con la ganancia media | `test_stop_batch_permutation` |
| 3.3 | La garantía secuencial del centinela no está justificada | **Confirmado** (ya se había visto: 0-12.5 % de falsas alarmas en sangre) | Se retira la afirmación. La martingala queda como puntuación sin garantía; la única garantía es la alarma por paciente (marginal, bajo intercambiabilidad) | `test_alarma_por_paciente_respeta_alfa` |
| 3.4 | `incorporar()` no recalibra la abstención | **Confirmado** | Recalibración automática tras cada incorporación (`activo.recalibrar`), con pacientes de calibración nuevos si se generan; `calibracion_vigente` se reporta en `predecir()` | `test_resume_campaign_and_recalibrate` |
| 3.5 | Campos = media simple; escalares = decisión colectiva | **Confirmado** | Los campos se combinan con pesos según la cercanía de cada agente a la decisión escalar; uniformes en abstención. **Pendiente:** restricción integral campo-escalar y calibración propia de los campos | `test_field_weights_follow_scalar_decision` |
| 3.6 | Guardar y cargar pierde el estado de la campaña | **Confirmado** | Se guardan `parada`, simulaciones usadas y vigencia de la calibración | `test_resume_campaign_and_recalibrate` |
| 3.7 | `torch.load(weights_only=False)`; privacidad de `incluir_datos=False` | **Confirmado**, arreglo parcial | Avisos explícitos de seguridad y privacidad en `guardar`/`cargar`. **Pendiente:** formato `state_dict`/safetensors + JSON y separar la validación del archivo desplegable | — |
| 3.8 | `presupuesto=0` se sustituía por el predeterminado | **Confirmado** | `None` → predeterminado; negativo → error; 0 → lista vacía; nunca se excede `presupuesto_total` | `test_zero_and_total_budget` |
| 3.9 | LTT: umbrales ordenados con la propia calibración; `calib` opcional | **Confirmado** | `calib` obligatorio (error si falta); rejilla de umbrales fijada con el otro conjunto (`ltt_grid`); abstención explícita si nada se certifica; verificación Monte Carlo | `test_ltt_requires_independent_calibration`, `test_ltt_no_certified_threshold_abstains`, `test_ltt_grid_controls_risk` |
| 3.10 | H se recortaba sin corregir el flujo eritrocitario | **Confirmado** | La fracción de separación de fases se restringe para que ninguna rama supere H_max; H = Q_rbc/Q sin recorte posterior | `test_rbc_flux_consistency_and_bound` |
| 3.11 | Casos degenerados y falta de indicadores de convergencia | **Confirmado**, y peor de lo descrito | Fuente sin H, ramas sin caudal y aristas sin resolver se manejan y se reportan; `solve` devuelve `convergio`, `iteraciones`, `cambio_final_H`, `identidad_rbc`, `sin_resolver`. Hallazgo propio: con la tolerancia fija de 1e-10 el árbol de 4 niveles NUNCA convergía (se estanca en ~1e-7) y se devolvía sin aviso; ahora `tol=1e-6`, 100 iteraciones | `test_zero_flow_and_convergence_flags` |
| 3.12 | La validación decide qué expertos viven | **Confirmado** | Conjunto de evolución separado (`fit_central(evo=...)`, `Gemelo.datos_evolucion`, `dominio.datos_evolucion`). Con datos de archivo sin conjunto de evolución se usa la validación y se documenta | (cubierto por las pruebas del Gemelo) |
| 3.13 | `load_flexible` descarta pesos en silencio | **Confirmado** | Solo se omiten buffers de memoria de tamaño variable; cualquier otra incompatibilidad lanza error; se reportan las claves omitidas | `test_flexible_load_equivalence_and_strictness` |

## Robustez matemática (sección 4)

- **4.1 Hopfield discreto.** De acuerdo: la prueba de energía decreciente cubre la dinámica continua y
  casos de prueba, no una cota para el paso discreto. Pendiente: búsqueda de paso por retroceso o cota de
  estabilidad, y reportar la distribución de violaciones.
- **4.2 Lyapunov de la ecología.** De acuerdo: vale con r y A congelados. No debe presentarse como
  estabilidad del sistema acoplado. Coincide con el diagnóstico de Chesson de v0.8.1 (0 pares coexisten).
- **4.3 Consistencia termodinámica.** De acuerdo: "informado por la física" para el cierre de viscosidad
  y la red; no "operador termodinámicamente consistente" para el sistema completo.
- **4.4 Forrajeo.** De acuerdo: la ganancia es el error ANTES de aprender, no la mejora medida. No se
  afirma optimalidad. Pendiente: adquisición por reducción esperada de riesgo por unidad de costo.

## Resultados (sección 5) y originalidad (sección 6)

De acuerdo en todo. Ya estaba escrito en `RESULTADOS.md`: la potencia del centinela en sangre es baja
(23 %), dos o tres semillas no bastan, el ahorro de simulaciones demuestra la regla de parada y no la
selección colectiva, y el árbitro triple no mejora al rápido-lento. Se añade a pendientes: pruebas de no
inferioridad, curvas de potencia frente al número de mediciones, CUSUM/GLR como líneas base y los siete
escenarios de parada. Requieren el clúster.

## Repositorio (sección 9)

- Las versiones sí están etiquetadas en Git (v0.9.2 y siguientes).
- Pendientes que dependen de una decisión del autor: `LICENSE` y `CITATION.cff`.
- Pendientes técnicos: integración continua, `environment.lock.yml`, `REPRODUCIBILITY.md` y vincular cada
  resultado con commit, configuración, semillas y hardware.
