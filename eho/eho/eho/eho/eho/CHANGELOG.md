# Cambios

## v0.9.7 (27 de noviembre de 2026): cuarta auditoría externa
Ver `docs/AUDITORIA_EXTERNA_v0.9.6.md`.
- Una sola función de decisión certificada (`_certificar`) y una sola de inspección (`context_inspection`).
- Certificado descrito, ligado a la huella del modelo y con δ conjunta; consulta y contexto del mismo paciente.
- Auditoría de particiones con lo incorporado; reserva de presupuesto antes de simular; costos validados; identificadores de ejecución; metadatos aparte.
- Red: convergencia de H, caudal y viscosidad; viscosidad positiva; disipación no negativa; unidades de la cizalla.
- `no_inferioridad` con emparejamiento real e intervalo t; pickle desactivado por defecto; huella de integridad del archivo.
- Centinela: tres estados y umbral a igual tasa de falsas alarmas. Hopfield: modo `armijo`. Rejilla LTT con respaldo.

## v0.9.6 (27 de noviembre de 2026): errores que persistían
Ver `docs/PENDIENTES_v0.9.6.md`.
- Formato de despliegue seguro (sin pickle); `cargar(permitir_pickle=False)`.
- Un certificado LTT por protocolo de contexto (`colmena.protocolos`); rejilla LTT informada por la referencia.
- Identificador único y manifiesto por simulación; no se puede incorporar dos veces la misma.
- Red: componentes desconectados. Hopfield: `energy_trace`. Centinela: CUSUM. Estadística: no inferioridad.
- `exp_abstencion.py` con frontera riesgo-cobertura.

## v0.9.5 (27 de noviembre de 2026): tercera auditoría externa y auditoría propia
Ver `docs/AUDITORIA_EXTERNA_v0.9.4.md`.
- Red: nodos de frontera con paso de flujo, `valida` con todas las comprobaciones físicas, gradiente del vaso padre, cizalla local acoplada.
- Centinela: huella SHA-256 del estado completo y del protocolo.
- Datos: huellas invariantes al orden y por muestra; auditoría de procedencia obligatoria; carga sin pickle.
- LTT ligado al protocolo de contexto; contexto de un solo paciente.
- Presupuesto en condiciones, ejecuciones y costo; `incorporar()` transaccional y con validación de conjuntos.
- Despliegue sin datos de validación; línea de comandos corregida (rota en v0.9.4).
- `experiments/exp_abstencion.py`: comparación de señales de abstención con la misma certificación.

## v0.9.4 (27 de noviembre de 2026): segunda auditoría externa
Ver `docs/AUDITORIA_EXTERNA_v0.9.3.md`. 21 hallazgos confirmados; 18 corregidos, 3 parciales. 97 pruebas.
- Colmena: el predictor calibrado devolvía siempre la media; ahora `colmena.prediccion` (media | ponderada | ganador). La media sigue por defecto porque el ganador da +5 % de error.
- LTT: sin certificado se rechaza todo; `ltt_threshold` exige rejilla; la rama antigua es `heuristic_threshold`.
- Calibración caducada: nada se marca como aceptado; se invalida antes de modificar los modelos.
- Inferencia determinista: el apoyo ya no depende del orden de las consultas.
- Persistencia: `guardar(modo="reanudar" | "despliegue")`; la campaña reanudada es idéntica a la continua.
- Presupuesto global en todos los modos; `mvt` exige `usar_error`; conteo de ejecuciones por simulador.
- Centinela: s0 con referencia aparte, protocolo de sondeo fijo, huella de los agentes.
- Red: flujo propagado independiente, residuo constitutivo, bandera `valida`, validaciones, gradientes correctos.
- `evaluar_colectivo()`; arquitecturas relativas a la configuración; ecología congelable; claves faltantes; huellas de datos y semillas separadas.

## v0.9.3 (27 de noviembre de 2026): respuesta a la auditoría externa
Ver `docs/AUDITORIA_EXTERNA_v0.9.2.md`. 13 hallazgos confirmados; 11 corregidos y 2 con arreglo parcial.
83 pruebas (12 nuevas en `tests/test_v093.py`).
- Regla de parada: para con ganancia cero; invariante al orden del lote; entradas validadas.
- Abstención: recalibración tras `incorporar()`; LTT con calibración independiente obligatoria y rejilla predefinida.
- Centinela: se retira la garantía secuencial; queda la garantía marginal por paciente.
- Gemelo: `presupuesto=0` respetado y tope global; estado de campaña en el archivo; campos con los pesos de la decisión escalar.
- Red microvascular: separación de fases conservativa y acotada; indicadores de convergencia; tolerancia realista.
- Ecología: conjunto de evolución separado de la validación.
- `load_flexible`: sin descartes silenciosos.

## v0.9.2 y anteriores
Ver `RESULTADOS.md`.
