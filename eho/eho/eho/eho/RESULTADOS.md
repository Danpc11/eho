# Resultados v0.9.6: corrección de los errores que persistían (27 de noviembre de 2026)

Detalle en `docs/PENDIENTES_v0.9.6.md`.

## Señales de abstención (2 semillas por dominio): AUROC para detectar predicciones fallidas

| Señal de confianza | burgers1d | edp1d | sangre |
|---|---|---|---|
| desviacion | 0.70 | 0.49 | 0.69 |
| mad | 0.61 | 0.48 | 0.58 |
| distancia | 0.70 | 0.49 | 0.69 |
| inspeccion | 0.53 | 0.46 | 0.73 |
| confianza | 0.53 | 0.46 | 0.73 |
| colmena | 0.63 | 0.46 | 0.66 |
| clasificador | 0.69 | 0.56 | 0.76 |

Lectura: **el apoyo de la colmena no es la mejor señal en ningún dominio.** La desviación entre agentes
la supera en Burgers y en sangre, y un clasificador ligero (desviación, MAD e inspección) es la mejor o
empata en los tres. En advección-difusión ninguna señal separa bien (≈ 0.5). La colmena cuesta unos
38 ms por predicción; las demás, casi nada.

## Cobertura certificable con la rejilla LTT corregida

**burgers1d, semilla 1** (27% de fallos en la prueba). Cobertura certificada (riesgo real):

| Señal | riesgo 0.1 | riesgo 0.2 | riesgo 0.3 |
|---|---|---|---|
| desviacion | nada | nada | 83% (0.20) |
| mad | nada | nada | nada |
| distancia | nada | nada | 84% (0.21) |
| inspeccion | nada | nada | nada |
| confianza | nada | nada | nada |
| colmena | nada | nada | nada |
| clasificador | nada | nada | 79% (0.19) |

**sangre, semilla 1** (27% de fallos en la prueba). Cobertura certificada (riesgo real):

| Señal | riesgo 0.1 | riesgo 0.2 | riesgo 0.3 |
|---|---|---|---|
| desviacion | nada | 65% (0.14) | 100% (0.27) |
| mad | nada | nada | 99% (0.27) |
| distancia | nada | 69% (0.15) | 99% (0.27) |
| inspeccion | nada | 48% (0.14) | 78% (0.21) |
| confianza | nada | 68% (0.18) | 68% (0.18) |
| colmena | nada | nada | 91% (0.25) |
| clasificador | nada | 54% (0.11) | 97% (0.26) |

Lectura:
- Hasta v0.9.5 no se certificaba nada con ningún riesgo: la secuencia de umbrales empezaba con muy pocos
  casos y un solo fallo la detenía. Con la rejilla informada por los datos de referencia sí se certifica,
  y **el riesgo real queda por debajo del riesgo pedido en todos los casos certificados**.
- Con riesgo 0.1 sigue sin certificarse nada cuando el 27 % de las predicciones falla: es un límite de
  los datos y de la tolerancia elegida (2 veces el error mediano), no del procedimiento.
- La colmena certifica menos cobertura que la desviación entre agentes o el clasificador con el mismo riesgo.
- Una semilla por dominio en esta tabla: orientativo.

## Energía del núcleo de Hopfield con pasos discretos

`EHOCentral.energy_trace()`: 480 muestras por dominio (Burgers, sangre, advección-difusión), 3 pasos, con
el modelo sin entrenar y entrenado: **0 % de pasos con energía creciente**. Evidencia empírica; no es
una cota.

## Formato de despliegue seguro

El archivo de despliegue se lee sin ejecutar código y da predicciones idénticas al modelo original
(diferencia 0.0 en escalares, campos, apoyo y aceptación), también después de compactar.

---

