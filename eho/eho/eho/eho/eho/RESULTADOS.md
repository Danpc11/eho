# Resultados v0.9.7: cuarta auditoría externa (27 de noviembre de 2026)

Detalle en `docs/AUDITORIA_EXTERNA_v0.9.6.md`. Esta versión corrige errores de método y de código; no
trae resultados nuevos salvo dos comprobaciones.

## Señal de abstención elegida solo con la referencia (sangre, semilla 1)

La señal se elige con el conjunto de referencia y se congela antes de mirar calibración y prueba:
**clasificador** (AUROC 0.94 en la referencia). El certificado es por observación
(una consulta al azar por paciente); las otras dos columnas son lecturas clínicas NO certificadas.

| Riesgo pedido | Cobertura certificada | Riesgo por observación | Riesgo medio por paciente | Pacientes con algún fallo |
|---|---|---|---|---|
| 0.1 | nada | — | — | — |
| 0.2 | 54% | 0.11 | 0.12 | 24% |
| 0.3 | 97% | 0.26 | 0.26 | 60% |

Lectura: que el riesgo por observación esté acotado no significa que un paciente no tenga ningún fallo.
Para afirmar algo por paciente hay que certificar con esa unidad. Una semilla: orientativo.

## Energía del núcleo de Hopfield: descenso garantizado

Con el paso por defecto (α = 0.1), 0 % de pasos con energía creciente (como en v0.9.6). Con pasos
grandes (α = 2 y α = 20) el paso fijo viola en ~25 % de los pasos; el modo `armijo` da 0 % en todos los
casos, por construcción. Con α = 0.1 ambos modos dan exactamente la misma energía final.

## Sin cambios en los resultados anteriores

La inspección unificada no cambia los números con 2 mediciones de contexto (la corrida de comprobación
reproduce los de la v0.9.6); sí cambia el comportamiento con 5. Los resultados previos a la v0.9.4 siguen
pendientes de regenerar.

---

