# Errores que persistían tras la v0.9.5: qué se corrigió en la v0.9.6 y qué queda

Fecha: 27 de noviembre de 2026. Pruebas nuevas: `tests/test_v096.py` (8).

| # | Error que persistía | Estado en v0.9.6 | Prueba |
|---|---|---|---|
| 1 | El archivo de modelo era un pickle | **Corregido para despliegue.** Formato seguro (`eho/persistencia.py`): solo tensores, números y JSON; se lee con `weights_only=True`. Predice exactamente igual (diferencia 0.0), también tras `compact()`. `Gemelo.cargar(ruta, permitir_pickle=False)` rechaza cualquier pickle. El punto de control para reanudar sigue siendo pickle (lleva datos y objetos de la campaña): solo para uso propio | `test_safe_deployment_roundtrip` |
| 2 | Un solo umbral certificado por modelo | **Corregido.** `colmena.protocolos: [0, 1, 2, 5]` calibra un certificado por número de mediciones; al predecir se usa el que corresponde, y con un protocolo no calibrado no se certifica | `test_certificate_per_context_protocol` |
| 2b | La secuencia LTT se detenía con un solo fallo temprano y no certificaba nada (hallazgo propio: en sangre certificaba con riesgo 0.2 y no con 0.3) | **Corregido.** La rejilla de umbrales se elige con los datos de referencia (independientes de la calibración): pocos umbrales, entre la mitad y el total de la cobertura que la referencia sugiere alcanzable | `test_informed_ltt_grid_is_valid_and_more_powerful` |
| 3 | Identidad de los datos solo por contenido; simulaciones repetidas se contaban doble | **Corregido en parte.** Cada simulación incorporada tiene identificador y entrada en `Gemelo.registro` (ronda, simuladores, costo y los campos del `manifiesto` que se pasen). Incorporar dos veces la misma da error. Falta que los dominios generen el manifiesto desde el simulador real | `test_unique_simulation_ids_and_manifest` |
| 4 | El costo no se medía | **En parte.** Se registran los segundos de cada ajuste y el costo declarado por simulación. Las horas de GPU del simulador las tiene que aportar quien lo corre (`incorporar(costo=...)`) | misma prueba |
| 5 | La red no detectaba componentes desconectados | **Corregido.** Error si una componente no tiene nodo con presión | `test_disconnected_network_rejected` |
| 6 | Energía de Hopfield: sin cota para el paso discreto | **Medido, no demostrado.** `EHOCentral.energy_trace()` da la energía en cada paso. En 480 muestras por dominio (Burgers, sangre, advección-difusión), con el modelo sin entrenar y entrenado: 0 % de pasos con energía creciente. Es evidencia empírica; la cota teórica sigue pendiente | `test_energy_trace_reports_discrete_violations` |
| 7 | Sin pruebas de no inferioridad ni líneas base clásicas del centinela | **Herramientas listas.** `analisis/estadistica.no_inferioridad()` (margen fijado de antemano; exige ≥ 5 semillas) y `Centinela.cusum()`, incluido en `exp_novedad.py`. Faltan las corridas | `test_non_inferiority`, `test_cusum_baseline` |
| 8 | `RESULTADOS.md` remitía a una lectura que no estaba escrita | **Corregido** (sección v0.9.6 de `RESULTADOS.md`) | — |

## Lo que sigue pendiente

- **Requiere el clúster:** 10 semillas, regenerar todos los resultados anteriores a la v0.9.4, pruebas de
  no inferioridad con margen fijado, CUSUM/GLR frente al centinela, datos DPD reales.
- **Requiere trabajo de modelado:** usar la cizalla local acoplada en `exp_red_inversa.py` con unidades
  físicas y sin `no_grad`; separación de fases en nodos de frontera (hoy "bien mezclada", una hipótesis);
  cota teórica del paso discreto de Hopfield; estabilidad del sistema acoplado con ecología.
- **Requiere una decisión tuya:** licencia y `CITATION.cff`; si la colmena pasa a ser solo una ablación.
- **Formato:** el punto de control para reanudar sigue siendo pickle.
