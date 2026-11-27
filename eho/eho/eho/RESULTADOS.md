# Resultados v0.9.5: tercera auditoría externa y auditoría propia (27 de noviembre de 2026)

Detalle en `docs/AUDITORIA_EXTERNA_v0.9.4.md`: 14 hallazgos del evaluador confirmados (11 corregidos,
3 parciales) y 4 errores propios encontrados y corregidos.

## ¿Hace falta la dinámica de abejas para abstenerse?

`experiments/exp_abstencion.py`: mismos agentes, misma predicción (la media) y la misma certificación
Learn-then-Test (riesgo 0.1, δ = 0.1, 90 pacientes de calibración) para siete puntuaciones de confianza.
Prueba: 60 pacientes de regímenes conocidos y de uno intermedio. **Una semilla por dominio** (la segunda
se cortó por tiempo). AUROC: detectar fallos (mayor es mejor). AURC: área bajo la curva riesgo-cobertura
(menor es mejor).

| Puntuación de confianza | burgers1d: AUROC / AURC | edp1d: AUROC / AURC | sangre: AUROC / AURC |
|---|---|---|---|
| desviacion | 0.67 / 0.759 | 0.57 / 0.700 | 0.60 / 0.612 |
| mad | 0.56 / 0.984 | 0.52 / 0.795 | 0.54 / 0.645 |
| distancia | 0.66 / 0.763 | 0.56 / 0.704 | 0.61 / 0.612 |
| inspeccion | 0.48 / 1.025 | 0.47 / 0.772 | 0.77 / 0.510 |
| confianza | 0.48 / 1.027 | 0.46 / 0.787 | 0.77 / 0.509 |
| colmena | 0.58 / 0.853 | 0.52 / 0.718 | 0.67 / 0.543 |
| clasificador | 0.66 / 0.739 | 0.54 / 0.706 | 0.73 / 0.542 |

Certificación con el mismo procedimiento:

| Puntuación | burgers1d: cobertura certificada / riesgo real | edp1d: cobertura certificada / riesgo real | sangre: cobertura certificada / riesgo real |
|---|---|---|---|
| desviacion | sin certificar | sin certificar | sin certificar |
| mad | sin certificar | sin certificar | sin certificar |
| distancia | sin certificar | sin certificar | sin certificar |
| inspeccion | sin certificar | sin certificar | sin certificar |
| confianza | sin certificar | sin certificar | sin certificar |
| colmena | sin certificar | sin certificar | sin certificar |
| clasificador | sin certificar | sin certificar | sin certificar |

(burgers1d: 34% de fallos, colmena 37.6 ms por predicción; edp1d: 25% de fallos, colmena 37.6 ms por predicción; sangre: 20% de fallos, colmena 38.5 ms por predicción.)

Las conclusiones de esta tabla están en `docs/AUDITORIA_EXTERNA_v0.9.4.md` y deben repetirse con más
semillas antes de cambiar valores por defecto.

## Aclaración sobre la comparación de la v0.9.4

Los tres mecanismos de agregación se calibraron para una cobertura objetivo del 80 % en validación: no
comparten umbral numérico ni cobertura exacta en la prueba, y no se usó LTT. Esa tabla compara el error
de la agregación, no la abstención certificada.

---

