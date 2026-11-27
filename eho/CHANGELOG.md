# Cambios

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
