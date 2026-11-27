# Cambios

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
