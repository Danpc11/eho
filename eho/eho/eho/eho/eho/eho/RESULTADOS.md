# Resultados v0.10: memoria bayesiana, diseño de mediciones y ganancia medida (27 de noviembre de 2026)

Detalle en `docs/v0.10_MEMORIA_BAYESIANA_Y_PRERREGISTRO.md`. **Una semilla por dominio, en CPU y con
datos sintéticos: indicativo.** 30 pacientes (mitad de un régimen intermedio no visto); de cada uno, 4
condiciones objetivo y 8 candidatas a medirse. Error absoluto medio en el objetivo.

## A. Adaptación con pocas mediciones (elegidas al azar)

| Dominio | Método | k = 0 | k = 1 | k = 2 | k = 5 |
|---|---|---|---|---|---|
| burgers1d | Sin mediciones (base) | 0.5099 | 0.5099 | 0.5099 | 0.5099 |
| burgers1d | Memoria del EHO | 0.5099 | 0.3220 | 0.3020 | 0.2775 |
| burgers1d | Memoria bayesiana | 0.5099 | 0.2434 | 0.2373 | 0.1914 |
| burgers1d | Cobertura del intervalo del 95 % (bayesiana) | 1.00 | 0.99 | 0.97 | 0.98 |
| sangre | Sin mediciones (base) | 0.0419 | 0.0419 | 0.0419 | 0.0419 |
| sangre | Memoria del EHO | 0.0419 | 0.0283 | 0.0255 | 0.0252 |
| sangre | Memoria bayesiana | 0.0419 | 0.0144 | 0.0105 | 0.0073 |
| sangre | Cobertura del intervalo del 95 % (bayesiana) | 1.00 | 0.98 | 0.98 | 0.99 |

- La memoria bayesiana reduce el error frente a la memoria del EHO en los dos dominios y en todos los k.
- Su intervalo del 95 % cubre entre el 97 % y el 99 %: la incertidumbre está bien calibrada, algo conservadora.
- No hay ningún paso de gradiente al adaptar: es álgebra lineal de r x r por paciente.

## B. Qué medirle a un paciente (predicción con la memoria bayesiana)

| Dominio | Criterio para elegir | 1 medición | 2 mediciones | 3 mediciones |
|---|---|---|---|---|
| burgers1d | Al azar | 0.2625 | 0.2326 | 0.2078 |
| burgers1d | Desacuerdo entre agentes | 0.2720 | 0.2107 | 0.2079 |
| burgers1d | D-óptimo | 0.2014 | 0.1968 | 0.1966 |
| burgers1d | Orientado al objetivo | 0.1996 | 0.1959 | 0.1932 |
| sangre | Al azar | 0.0142 | 0.0093 | 0.0094 |
| sangre | Desacuerdo entre agentes | 0.0121 | 0.0099 | 0.0088 |
| sangre | D-óptimo | 0.0122 | 0.0092 | 0.0080 |
| sangre | Orientado al objetivo | 0.0113 | 0.0084 | 0.0077 |

- Elegir por información (D-óptimo u orientado al objetivo) supera al azar y al desacuerdo entre agentes.
  En Burgers, UNA medición bien elegida da el error de TRES al azar.
- El desacuerdo entre agentes (el sondeo de v0.8) no es consistentemente mejor que el azar: gana en unos
  casos y pierde en otros, como ya se vio en v0.9.1.
- La diferencia entre D-óptimo y orientado al objetivo es pequeña aquí, porque el objetivo son otras
  condiciones del mismo paciente; debería crecer cuando el objetivo sea una cantidad derivada (perfusión).

## C. Regla de parada con ganancia medida (Burgers, reserva que se agota en la ronda 2)

| Regla de parada | Simulaciones usadas | Error en lo nuevo (agentes) | Error en lo nuevo (ensamble) |
|---|---|---|---|
| Sorpresa (v0.9) | 16 de 24 | 0.678 | 0.654 |
| Ganancia medida (v0.10) | 12 de 24 | 0.577 | 0.537 |

- Con ganancia medida la campaña se detiene a las 12 simulaciones, justo cuando la reserva deja de traer
  casos nuevos; con la sorpresa seguía hasta 16. Es un ahorro del 50 % frente al presupuesto de 24.
- El error final no es peor (0.577 frente a 0.678), aunque de esas 12 simulaciones 3 se apartaron como
  referencia y no se usaron para aprender. Que sea MENOR es probablemente ruido de una semilla.
- "Agentes" es el promedio de los errores individuales; "ensamble", el error de la media de predicciones.
- Una semilla y un dominio: hay que repetirlo antes de afirmarlo, y con la prueba de no inferioridad
  prerregistrada frente al codicioso con presupuesto completo (que aquí no se corrió).

## Advertencias

- La comparación A no incluye el ajuste fino afinado ni el árbitro en este mismo protocolo: hay que
  añadirlos antes de afirmar que la memoria bayesiana es la mejor opción.
- La memoria bayesiana no participa todavía en la abstención certificada ni en el centinela.
- Advección-difusión no se corrió en esta ronda.

---

