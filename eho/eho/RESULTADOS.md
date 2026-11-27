# Resultados v0.9.4: segunda auditoría externa (27 de noviembre de 2026)

97 pruebas pasan (2 solo con GPU). Detalle punto por punto en `docs/AUDITORIA_EXTERNA_v0.9.3.md`.
Los 21 hallazgos de código se confirmaron; 18 corregidos y 3 con arreglo parcial.

## El hallazgo principal: la colmena calibrada era un promedio

Hasta v0.9.3 el predictor calibrado devolvía siempre la media aritmética de los agentes: el ganador de
la dinámica de abejas nunca cambiaba el valor predicho. Corregido, se compararon las tres opciones con
los mismos agentes y el mismo umbral (`experiments/exp_prediccion_colectiva.py`; error escalar MAE,
2 semillas por dominio; sangre con decaimiento coseno):

| Dominio (semillas) | Conjunto | Promedio de errores de agentes | Media | Ponderada | Ganador |
|---|---|---|---|---|---|
| burgers1d (2) | conocido | 0.3315 | 0.2974 | 0.2971 | 0.3112 |
| burgers1d (2) | intermedio | 0.2864 | 0.2451 | 0.2457 | 0.2606 |
| edp1d (2) | conocido | 0.1359 | 0.1067 | 0.1080 | 0.1199 |
| edp1d (2) | intermedio | 0.1365 | 0.1105 | 0.1120 | 0.1241 |
| sangre (2) | conocido | 0.0350 | 0.0298 | 0.0285 | 0.0290 |
| sangre (2) | intermedio | 0.0295 | 0.0235 | 0.0233 | 0.0232 |

- **El centro ganador de la colmena es peor que la media**: +5.2 % de error en promedio (peor en Burgers
  y en advección-difusión; igual en sangre).
- **La media ponderada por confianza no se distingue de la media simple** (-0.5 %).
- **Promediar sí ayuda**: la media de las predicciones tiene 10-22 % menos error que el promedio de los
  errores individuales de los agentes.
- Por eso `colmena.prediccion` queda en `media` por defecto; `ganador` y `ponderada` son opciones.

**Consecuencia para el artículo:** la dinámica de abejas no mejora la predicción. Lo que aporta la
colmena es el apoyo con el que se decide abstenerse (y, en el aprendizaje activo, la regla de parada).
Debe describirse como "ensamble con abstención colectiva calibrada", no como "decisión colectiva de la
predicción".

## Resultados anteriores que quedan invalidados hasta regenerarlos

- Los errores de los experimentos de presupuesto y parada (v0.7-v0.9.3) son el promedio de los errores de
  los agentes, no el error del predictor conjunto. `exp_presupuesto.py` ahora reporta ambos.
- Con semilla 0, la reserva de candidatos compartía semillas con el entrenamiento. Las semillas están
  ahora en rangos separados y el experimento comprueba por huellas que no haya episodios repetidos.
- Las alarmas por paciente (v0.9.2) estimaban s0 con los mismos pacientes de calibración, y el sondeo
  activo no usaba el mismo protocolo al calibrar. Corregido; falta repetir las cifras.
- Sigue pendiente de la v0.9.3: repetir todo con el conjunto de evolución separado.

Nada de esto se ha vuelto a correr todavía, salvo la comparación de arriba.

---

# Resultados v0.9.3: correcciones de la auditoría externa (27 de noviembre de 2026)

83 pruebas pasan (2 solo con GPU). Detalle punto por punto en `docs/AUDITORIA_EXTERNA_v0.9.2.md`.

Los 13 hallazgos de código de la auditoría se confirmaron. 11 están corregidos; 2 tienen arreglo parcial
(formato de archivo seguro; consistencia integral entre campos y escalares).

**Afirmaciones que se retiran o se acotan:**
- La martingala del centinela NO garantiza falsas alarmas "en cualquier momento". La única garantía es
  la alarma por paciente (marginal, bajo intercambiabilidad de pacientes).
- La garantía Learn-then-Test solo vale con calibración independiente (ahora obligatoria) y con la
  rejilla de umbrales fijada con otros datos.
- La función de Lyapunov de la ecología vale con parámetros congelados; no demuestra estabilidad del
  sistema que aprende. La energía de Hopfield decrece en tiempo continuo; el paso discreto no tiene cota.
- La regla de forrajeo no es óptima en sentido de aprendizaje activo: usa el error antes de aprender.

**Hallazgo propio al corregir la red microvascular:** con la tolerancia fija de 1e-10, el acoplamiento
hematocrito-flujo del árbol de 4 niveles nunca convergía (se estanca en ~1e-7) y el resultado se
devolvía sin aviso. Ahora hay tolerancia de 1e-6, 100 iteraciones e indicador `convergio`.

**Comprobación de que la regla de parada corregida sigue funcionando** (reserva agotada en la ronda 2,
semilla 0): para a las 16 simulaciones en los dos dominios, como antes. Error en lo nuevo: sangre 0.064
(antes 0.060), Burgers 0.594 (antes 0.581). Una semilla; las diferencias están dentro del ruido visto.
Estas corridas ya incluyen la recalibración tras cada incorporación.

Los resultados de versiones anteriores NO se han repetido con el conjunto de evolución separado ni con
la rejilla LTT nueva: hay que regenerarlos en el clúster antes de citarlos.

---

# Resultados v0.9.2: alarma por paciente y regla de parada en sangre (26 de noviembre de 2026)

71 pruebas pasan (2 solo con GPU). Sangre con decaimiento coseno. 3 semillas en la alarma, 2 en la parada.

## 1. Alarma calibrada por paciente

La martingala por muestra daba 0-12.5 % de falsas alarmas en sangre porque las muestras de un paciente
no son independientes. Ahora (`Centinela.patient_alarms`) el estadístico es la sorpresa media del
paciente y su p-valor conformal se calcula frente a 60 pacientes de calibración: los pacientes sí son
intercambiables, así que P(alarma | paciente conocido) ≤ α.

| Semilla | Falsas alarmas en 64 conocidos nuevos | Alarma en cambio visible | Alarma en cambio oculto | AUROC oculto |
|---|---|---|---|---|
| 0 | 3.1 % | 25 % | 25 % | 0.66 |
| 1 | 7.8 % | 19 % | 19 % | 0.84 |
| 2 | 1.6 % | 25 % | 25 % | 0.90 |
| Media | **4.2 % (8 de 192)**, α = 5 % | 23 % | 23 % | 0.80 |

- **La cota se cumple**: 4.2 % en promedio con α = 5 %. La garantía es marginal (en promedio sobre
  conjuntos de calibración), por eso una semilla puede quedar en 7.8 %.
- **La potencia es baja en sangre**: con α = 5 % solo se alarma el 19-25 % de los pacientes con cambio,
  aunque la AUROC sea 0.80. Separa en promedio, pero pocos pacientes superan el umbral estricto. Con 8
  muestras por paciente y la discrepancia pequeña de la ley de Pries, no da para más: es un límite real.
- En Burgers (semilla 0, tasa constante) la potencia es alta: falsas alarmas 4.7 % en 64 conocidos;
  alarma en el 94 % de los pacientes con cambio visible y en el 75 % con cambio oculto (AUROC 0.98 y 0.91).

## 2. Regla de parada en sangre (reserva que se agota en la ronda 2; verdad = ley de Pries)

Error escalar en lo nuevo (policitemia). Inicial: 0.092 (semilla 0).

| Semilla | Colmena: simulaciones / error | Codicioso con 16 | Codicioso con 20 | Codicioso con 24 |
|---|---|---|---|---|
| 0 | 16 / 0.060 | 0.060 | 0.048 | 0.072 |
| 1 | 16 / 0.063 | 0.063 | 0.072 | 0.052 |

- **La regla para a las 16 simulaciones en las dos semillas: 33 % de ahorro.** Tras agotarse los casos
  nuevos gasta dos rondas más (el tiempo de rendición) y se detiene.
- **Sin pérdida medible**: 0.060 y 0.063 frente a 0.072 y 0.052 del codicioso con 24 (media 0.062 en
  ambos). Seguir simulando casos conocidos no mejora: el error del codicioso sube y baja entre 0.048 y
  0.072, que es ruido de reentrenamiento.
- Hasta la simulación 16 la colmena y el codicioso eligen exactamente lo mismo en sangre: la colmena
  aporta la decisión de parar, no una mejor elección.
- Junto con Burgers (v0.9: 16 simulaciones en las dos semillas, pérdidas de 0.01 y 0.10), la afirmación
  defendible es: "la regla de parada ahorra un tercio de las simulaciones cuando la reserva deja de
  traer casos nuevos, sin pérdida en sangre y con una pérdida pequeña en Burgers". 2 semillas por dominio.

---

# Resultados v0.9.1: sangre repetida con decaimiento coseno (26 de noviembre de 2026)

Un núcleo de CPU. Novedad: 3 semillas con cada tasa. Pocos ejemplos y curvas: 2 semillas. IC del 95 %
por bootstrap entre semillas (con 2-3 semillas son orientativos). `EHO_TASA=coseno` en los experimentos;
`entrenamiento.tasa: coseno` en la configuración (ya activado en `configs/sangre.yaml`).

## 1. Pocos ejemplos en sangre (error escalar)

| Variante | Tasa constante: k = 1 / 2 / 5 | Decaimiento coseno: k = 1 / 2 / 5 |
|---|---|---|
| Sin memoria | 0.046 / 0.046 / 0.047 | 0.033 / 0.033 / 0.034 |
| Memoria completa | 0.038 / 0.036 / 0.036 | 0.024 / **0.019** / 0.017 |
| Ajuste fino afinado | 0.025 / 0.018 / 0.013 | 0.025 / 0.022 / 0.020 |
| Árbitro rápido-lento | 0.020 / 0.016 / 0.012 | 0.018 / **0.015** / **0.012** |
| Árbitro triple | (no corrido) | **0.018** / 0.016 / 0.013 |

- **Con decaimiento coseno, la memoria vuelve a competir en sangre**: iguala o supera al ajuste fino en
  k = 2 (0.019 frente a 0.022) y k = 5 (0.017 frente a 0.020). Los intervalos se solapan (2 semillas):
  es "empata o gana", no "gana".
- Con tasa constante, el ajuste fino ganaba claramente (0.013 frente a 0.036 en k = 5).
- **El árbitro es lo más sólido**: el mejor o empatado con el mejor en las dos condiciones y en todos
  los k, con los intervalos más estrechos (k = 5: [0.0117, 0.0118] con coseno).

## 2. Cambios en sangre (3 semillas, AUROC media [IC 95 %])

| Detector | Visible, constante | Visible, coseno | Oculto, constante | Oculto, coseno |
|---|---|---|---|---|
| Mahalanobis en entradas crudas | 1.00 | 1.00 | 0.44 | 0.44 |
| Plasmodial coseno | 0.82 [0.71, 0.89] | 0.95 [0.92, 0.99] | 0.49 | 0.45 |
| Varianza del ensamble | 0.74 [0.59, 0.85] | 0.96 [0.90, 1.00] | 0.48 | 0.46 |
| Inspección del sitio | 0.75 | 0.79 | 0.68 [0.63, 0.72] | 0.68 [0.59, 0.78] |
| **Centinela** | 0.75 [0.48, 0.95] | 0.71 [0.65, 0.77] | **0.83 [0.67, 0.97]** | **0.80 [0.66, 0.90]** |
| Centinela, 3 muestras al azar | 0.71 | 0.64 | 0.76 | 0.72 |
| Centinela, 3 muestras por sondeo | 0.66 | 0.56 | 0.73 | 0.64 |

- **El cambio oculto en sangre (la ley de Pries como verdad) sí se detecta**: centinela 0.80-0.83 frente
  a 0.68 de la inspección y ~0.45 de todo detector sin etiquetas. En v0.7-v0.8 parecía indetectable; lo
  que cambió es usar todo el contexto del paciente (v0.9). El intervalo es ancho.
- En el cambio visible (policitemia) el centinela no es la herramienta: basta Mahalanobis sobre las
  entradas (1.00). El decaimiento coseno sí mejora mucho los detectores basados en el modelo
  (plasmodial 0.82 → 0.95; ensamble 0.74 → 0.96) y estrecha los intervalos.
- **El sondeo activo empeora en sangre** (oculto 0.64-0.73 frente a 0.72-0.76 al azar). Ayuda en
  Burgers y perjudica en sangre: no es general y no debe afirmarse como mecanismo.
- Falsas alarmas de la martingala en pacientes conocidos: 0-12.5 % por corrida (0, 1 o 2 pacientes de
  16), media 5 %. Con α = 0.05 está en el límite: las muestras de un mismo paciente no son
  independientes y la garantía es por secuencia intercambiable. Hay que calibrar por paciente
  (una e-valor por paciente) antes de afirmar la cota en sangre.

## 3. Curvas de aprendizaje en sangre con coseno (k = 2, 2 semillas)

| Episodios por régimen | Sin memoria | Memoria completa |
|---|---|---|
| 5 | 0.098 | 0.115 |
| 10 | 0.041 | 0.051 |
| 20 | 0.055 | 0.036 |
| 40 | 0.033 | **0.019** |

Misma conclusión que con tasa constante: la memoria ayuda a partir de unos 20 episodios por régimen y
estorba con 5-10. La curva sin memoria sigue sin ser monótona (0.041 con 10, 0.055 con 20): con 2
semillas no se distingue de ruido.

## Qué queda de las afirmaciones del artículo, en sangre

1. Pocos ejemplos: defendible como "la memoria iguala al ajuste fino sin pasos de gradiente, y el
   árbitro supera a ambos", con decaimiento coseno. No como "la memoria gana".
2. Novedad: el cambio oculto se detecta con el centinela (0.80-0.83); el visible, con las entradas.
3. Menos simulaciones: no demostrado en sangre (la regla de parada solo se probó en Burgers).
4. Abstención: la cota de falsas alarmas está en el límite en sangre; falta la calibración por paciente.

---

# Resultados v0.9: auditoría de eficiencia y reformulaciones (26 de noviembre de 2026)

70 pruebas pasan (2 solo con GPU). Todo en UN núcleo de CPU. Semillas: 3 en novedad (Burgers y
advección-difusión) y estabilidad; 2 en curvas de aprendizaje y parada; 1 en sangre y en el árbitro triple.

## 1. Dónde se iba el costo

| Hallazgo | Cambio | Efecto medido |
|---|---|---|
| El 92 % de los parámetros estaba en 12 expertos; la ecología deja vivos 2 | `EHOCentral.compact()` | 193 mil → 54 mil parámetros (−72 %); predicciones idénticas |
| El centinela hacía un paso por muestra y por agente | LOO enmascarado en un paso por agente (`k=None`) | A pasos en vez de A·B; además detecta mejor (tabla 2) |
| `consolidate` recalculaba el estado del lote | reutiliza el del paso hacia adelante | un paso menos por el núcleo de Hopfield por lote (no medido aparte) |
| La colmena integraba 1,500 pasos por cada muestra en abstención | corte cuando el apoyo deja de cambiar y la suma de apoyos no alcanza el quórum | los empates entre opciones buenas no se cortan (prueba) |

Costo actual: 20 épocas con 80 episodios tardan 9-18 s; la inferencia, 0.16-0.24 ms por muestra.

## 2. Centinela con todo el contexto del paciente (3 semillas, AUROC media [IC 95 %])

| Detector | Burgers visible | Burgers oculto | Adv.-dif. visible | Adv.-dif. oculto |
|---|---|---|---|---|
| Inspección del sitio (v0.7) | 0.88 [0.87, 0.90] | 0.74 [0.68, 0.85] | 0.91 | 0.89 [0.86, 0.94] |
| Centinela v0.8 (contexto de 2) | 0.99 | 0.77 [0.72, 0.84] | 0.96 | 1.00 |
| **Centinela v0.9 (contexto completo)** | 0.96 [0.92, 0.99] | **0.93 [0.91, 0.96]** | 0.97 | **1.00** |
| 3 muestras al azar | 0.83 [0.66, 0.93] | 0.75 [0.70, 0.80] | 0.95 | 0.96 |
| **3 muestras por sondeo activo** | **0.93 [0.90, 0.96]** | **0.82 [0.79, 0.85]** | 0.97 | 0.98 |

- El cambio oculto en Burgers pasa de 0.77 a 0.93 al usar las 7 muestras restantes como contexto.
- **El sondeo activo ahora sí se sostiene con 3 semillas** en Burgers (oculto 0.82 frente a 0.75;
  visible 0.93 frente a 0.83, con intervalos que apenas se tocan). En v0.8.1, con contexto de 2, no.
- Martingala (α = 0.05): 0 falsas alarmas en conocidos en las 6 corridas. Alarma en el cambio oculto:
  75-100 % en advección-difusión, 0-31 % en Burgers (8 muestras por paciente dan poca evidencia).
- **Sangre (1 semilla) sigue siendo el punto débil**: centinela 0.48 visible y 0.67 oculto; inspección
  0.78 y 0.63. Los resultados de sangre cambian mucho de una corrida a otra (ver punto 5).

## 3. Regla de parada reformulada (Burgers, reserva que se agota en la ronda 2)

Antes, la ganancia de una simulación era max(s − 1, 0): la mitad de los casos conocidos contaban como
aprendizaje. Ahora es max(s / s_ref − 1, 0), con s_ref = percentil 90 del error en lo conocido.

| Semilla | Colmena v0.9: simulaciones / error | Codicioso con 16 | Codicioso con 24 |
|---|---|---|---|
| 0 | 16 / 0.581 | 0.564 | 0.571 |
| 1 | 16 / 0.657 | 0.611 | 0.561 |

Para a las 16 en las dos semillas (ahorro del 33 %; en v0.8 paraba en una de dos). Pierde 0.01 y 0.10
frente al codicioso con 24. Nota: estas corridas usaron `umbral_util = 0`; el valor por defecto quedó
en 0.1 para que la regla también pare cuando la referencia no está calibrada.

## 4. Árbitro entre tres predictores (memoria episódica, completa y ajuste fino; 1 semilla)

| Dominio | Variante | k = 1 | k = 2 | k = 5 |
|---|---|---|---|---|
| Adv.-dif. | Mejor predictor solo (memoria episódica) | 0.113 | 0.107 | 0.101 |
| | Árbitro rápido-lento (v0.8) | 0.115 | 0.112 | 0.109 |
| | **Árbitro triple** | **0.102** | **0.100** | **0.098** |
| Burgers | Árbitro rápido-lento / triple, k = 5 | | | 0.249 / 0.248 |

## 5. Estabilidad del entrenamiento (sin memoria, 3 semillas, error escalar)

| Dominio | Ecología | Tasa constante (por semilla) | Decaimiento coseno (por semilla) |
|---|---|---|---|
| Sangre | sí | 0.056, 0.036, 0.102 | **0.035, 0.030, 0.034** |
| Sangre | no | 0.059, 0.045, 0.049 | 0.039, 0.030, 0.034 |
| Burgers | sí | 0.335, 0.359, 0.349 | 0.349, 0.301, 0.319 |
| Burgers | no | 0.353, 0.337, 0.329 | 0.353, 0.319, 0.340 |

En sangre el decaimiento coseno baja el error a la mitad y casi elimina la variación entre semillas.
En Burgers los escalares cambian poco y los campos empeoran (0.50 → 0.55 con ecología). Queda como
opción (`fit_central(schedule="coseno")`), no por defecto. `experiments/exp_estabilidad.py`.

## 6. Curvas de aprendizaje: ¿la memoria ahorra simulaciones? (k = 2, 2 semillas)

| Episodios por régimen | Sangre: sin / con memoria | Burgers: sin / con memoria |
|---|---|---|
| 5 | 0.093 / 0.066 | 0.349 / 0.387 |
| 10 | 0.050 / 0.053 | 0.388 / 0.370 |
| 20 | 0.036 / 0.032 | 0.334 / 0.329 |
| 40 | 0.046 / 0.020 | 0.349 / 0.267 |

**La memoria no reduce las simulaciones necesarias cuando hay muy pocas.** Su ventaja crece con los
datos: necesita 20-40 episodios por régimen para aprender a leer. Lo que los datos respaldan es que
aprovecha mejor las simulaciones, no que hagan falta menos. La curva sin memoria en sangre no es
monótona (0.036 con 20, 0.046 con 40) por la inestabilidad del punto 5.

## 7. Descartado

- Reconstruir la memoria plasmodial al final NO causaba la regresión que se sospechaba: con y sin
  reconstrucción, 0.0201 / 0.0202 en sangre, 0.267 / 0.261 en Burgers, 0.129 / 0.123 en adv.-dif. (k = 2).

## Pendiente

- Repetir sangre con decaimiento coseno en todos los experimentos (novedad, pocos ejemplos, curvas).
- Árbitro triple y curvas de aprendizaje con más semillas; regla de parada con `umbral_util = 0.1`.
- 10 semillas en el clúster. No implementados: agentes no informados, gating por contexto, MC dropout, MAML.

---

# Resultados v0.8.1: lo que se pudo resolver sin el clúster (25 de noviembre de 2026)

Todo en CPU. **Tres semillas donde se indica, una en el resto.** IC del 95 % por bootstrap entre semillas
(con 3 semillas son anchos y orientativos). Las tablas se regeneran con `python3 analisis/estadistica.py`.

## 1. Cambio oculto y visible: centinela (3 semillas en Burgers y advección-difusión)

| Detector | Burgers visible | Burgers oculto | Adv.-dif. visible | Adv.-dif. oculto |
|---|---|---|---|---|
| Mahalanobis en la clave (mejor sin etiquetas) | 0.81 [0.77, 0.87] | 0.47 | 0.99 | 0.51 |
| Plasmodial resistencia | 0.78 [0.66, 0.89] | 0.46 | 0.99 | 0.50 |
| Varianza del ensamble | 0.79 [0.76, 0.82] | 0.51 | 0.48 | 0.51 |
| Inspección del sitio (v0.7) | 0.89 [0.89, 0.90] | 0.69 [0.58, 0.81] | 0.91 | 0.92 [0.89, 0.94] |
| **Centinela: sorpresa normalizada** | **0.99 [0.98, 1.00]** | **0.77 [0.72, 0.84]** | 0.96 [0.91, 0.99] | **1.00** |
| Centinela: martingala | 0.99 | 0.77 [0.73, 0.82] | 0.96 | 1.00 |
| Centinela, 3 muestras al azar | 0.92 | 0.61 [0.56, 0.66] | 0.93 | 0.98 |
| Centinela, 3 muestras por sondeo activo | 0.92 | 0.63 [0.57, 0.72] | 0.95 | 0.99 |

- **El centinela se sostiene con 3 semillas**: supera a la inspección en el cambio oculto (0.77 frente a 0.69
  en Burgers; 1.00 frente a 0.92 en advección-difusión) y en el visible de Burgers (0.99 frente a 0.89).
- **El sondeo activo no se sostiene**: la ventaja de la semilla 0 (0.72 frente a 0.60) era ruido; con 3
  semillas queda en 0.63 frente a 0.61. No debe afirmarse en el artículo tal como está.
- Con 3 semillas, los detectores sin etiquetas son más débiles en Burgers de lo que parecía con una
  (kNN en la clave 0.68 [0.51, 0.86]); ninguno pasa de 0.5 en el cambio oculto, como predice la teoría.
- Detector tipo CA1 (energía de Hopfield sobre los nodos): visible 0.66-0.68 en Burgers, 0.99 en
  advección-difusión, 0.75 en sangre. No mejora a los demás: se deja como opción (`novedad: energia`).
- Martingala (α = 0.05, 8 muestras por paciente): falsas alarmas en conocidos 0 %, 0 %, 0 %, 0 %, 6 %
  y 6 % (una de 16) en las seis corridas; cambio oculto 94-100 % en advección-difusión y 0 % en Burgers
  (8 muestras no acumulan evidencia suficiente aunque la AUROC sea 0.77).
- **Sangre (1 semilla): el centinela ayuda poco** (visible 0.80, oculto 0.60; la inspección daba 0.57 y
  0.52). En el cambio visible gana Mahalanobis sobre las entradas (0.99). La ley de Pries como cambio
  oculto sigue casi indetectable: la colmena la predice casi igual de bien que lo conocido.

## 2. Pocos ejemplos: árbitro rápido-lento (error escalar MAE)

| Dominio | Variante | k = 1 | k = 2 | k = 5 |
|---|---|---|---|---|
| Burgers (3 semillas) | Memoria completa | 0.296 | **0.254** | 0.237 |
| | Ajuste fino afinado | 0.375 | 0.361 | 0.298 |
| | Árbitro | **0.284** | 0.263 | **0.225** |
| Adv.-dif. (1 semilla) | Memoria episódica | **0.113** | **0.107** | **0.101** |
| | Memoria completa | 0.172 | 0.181 | 0.184 |
| | Ajuste fino afinado | 0.175 | 0.150 | 0.134 |
| | Árbitro (completa + ajuste) | 0.132 | 0.128 | 0.118 |
| Sangre (1 semilla) | Memoria completa | 0.058 | 0.056 | 0.055 |
| | Ajuste fino afinado | **0.024** | **0.018** | **0.011** |
| | Árbitro | 0.024 | 0.018 | 0.011 |

- El árbitro nunca queda lejos del mejor de sus dos componentes y a veces los supera (Burgers k = 1 y 5).
  Cumple su papel: no hay que elegir a mano entre memoria y ajuste fino.
- **En sangre, con esta configuración (40 episodios por régimen, 20 épocas), la memoria no adapta nada**
  (0.058 → 0.055) y el ajuste fino afinado gana por mucho. Con la configuración original (60 episodios,
  30 épocas, `exp_central_fewshot.py`) la memoria sí adapta: 0.057 sin contexto → 0.0149 con k = 2 →
  0.0145 con k = 5, frente a 0.0151 y ~0.010 del ajuste fino. Es decir, en sangre la memoria empata con
  el ajuste fino en k = 2 y pierde en k = 5, y su ventaja depende de cuánto se entrenó.
  **La afirmación 1 no se sostiene en sangre tal como está escrita.**
- Aviso: en esa corrida, la memoria completa sin contexto empeoró frente a v0.6 (0.026 → 0.057) y con
  contexto quedó algo peor (0.0115 → 0.0149). Puede ser efecto de reconstruir la memoria plasmodial al
  final del entrenamiento (v0.7) o ruido de una semilla; hay que medirlo con `rebuild=False`.
- En advección-difusión, la memoria episódica sola es la mejor y la completa la empeora: el árbitro
  debería arbitrar también entre memorias.

## 3. Regla de parada con reserva que se agota (Burgers; casos nuevos solo en las rondas 0 y 1)

| Semilla | Codicioso: simulaciones / error | Colmena v0.8: simulaciones / error | Ahorro | Pérdida |
|---|---|---|---|---|
| 0 | 24 / 0.571 | 16 / 0.619 | 33 % | +0.047 |
| 1 | 24 / 0.561 | 24 / 0.606 | 0 % | +0.046 |

La regla paró en una semilla de dos. En la otra siguió gastando aunque ya no había casos nuevos: los
conocidos todavía dejaban ganancia por encima del umbral. El ahorro existe pero no es fiable todavía;
hay que afinar `umbral_util` y `rendicion` con más semillas. La colmena queda ~0.05 por encima del
codicioso en error en ambas.

## 4. Ecología en aprendizaje continuo (sangre: sano → prediabetes → DT2; 3 semillas)

| Variante | Error final medio | Olvido en sano | Especies |
|---|---|---|---|
| Sin ecología | 0.177 [0.111, 0.306] | -0.122 | 4 |
| Ecología (Lotka-Volterra) | **0.119 [0.114, 0.123]** | -0.312 | 2 |
| Ecología con sucesión | 0.118 [0.115, 0.122] | -0.300 | 2 |

- **La ecología sí ayuda en aprendizaje continuo**: menor error y, sobre todo, mucha menos variación
  entre semillas (sin ecología una semilla se fue a 0.306). Confirma lo que predice la teoría: sirve
  cuando el entorno cambia, no en tareas estáticas.
- **La sucesión no cambia nada** aquí (0.118 frente a 0.119).
- **Diagnóstico de Chesson: 0 de 1 pares cumple la desigualdad de coexistencia** en todas las etapas y
  semillas; quedan 2 especies con solapamiento de nicho alto (0.62-0.99). Las dos especies que
  sobreviven no coexisten de forma estable según el criterio: se mantienen por el piso de abundancia,
  no por reparto de nichos. El beneficio viene de la selección (extinguir expertos malos), no de la
  diversidad. Así debe describirse.

## 5. Errores corregidos

- `load_state_dict` fallaba en el ajuste fino con memorias de tamaño variable (`exp_central_fewshot.py`
  estaba roto desde v0.4): ahora `train_central.load_flexible`.

## Pendiente

- 10 semillas en el clúster; sangre con más de una semilla en todo.
- Medir el efecto de reconstruir la memoria (`rebuild`) sobre la lectura en sangre.
- No implementados: agentes no informados (Couzin 2011), gating por contexto (Masse 2018), MC dropout, MAML.

---

# Resultados v0.8: mecanismos biológicos con modelo matemático (26 de noviembre de 2026)

66 pruebas pasan (2 solo con GPU). **Una semilla, en CPU: indicativo.** Las cifras del artículo salen
de `experiments/rejilla.py` con 10 semillas.

## Qué se implementó

| Mecanismo | Organismo y referencia | Archivo | Propiedad que se prueba |
|---|---|---|---|
| Parar por valor marginal + tiempo de rendición | Forrajeo; Charnov 1976, McNamara 1982 | `forrajeo.py`, `gemelo.proponer` | Maximiza ganancia por simulación; nunca para en frío |
| Separar compromiso (quórum) de parada | *Temnothorax*; Pratt y Sumpter 2006 | `gemelo.proponer` (`activo.parada: mvt`, por defecto) | Si la colmena se traba, se compromete con el de mayor valor |
| Inhibición cruzada creciente σ(t) y descubrimiento mínimo γ_min | Abejas; Seeley et al. 2012, Pais et al. 2013 | `colony.py` (`urgency_sigma`, `gamma_min`) | Sin congelamiento ni abstención perpetua |
| Sorpresa normalizada ACh/NE | Yu y Dayan 2005 | `centinela.py` | Error relativo a la incertidumbre esperada |
| Martingala de prueba conformal | Vovk et al. 2003, 2021 | `centinela.py` | Falsa alarma ≤ α en cualquier momento (Ville) |
| Adaptación por retroalimentación integral | Quimiotaxis de *E. coli* | `Centinela.adaptativo` | Responde a cambios, no al nivel basal |
| Sondeo activo | Peces eléctricos, murciélagos | `centinela.elegir_sondeo` | Qué k muestras medir, sin etiquetas |
| Árbitro rápido-lento por precisión inversa | Sistemas complementarios; Daw et al. 2005 | `mezcla.py` | Varianza mínima: envolvente inferior de memoria y ajuste fino |
| Sucesión (épocas pioneras sin Lotka-Volterra) | Sucesión ecológica | `train_central.py` (`modelo.sucesion`) | Sin selección mientras las aptitudes son ruido |
| Diagnóstico de coexistencia de Chesson | Chesson 2000 | `ecology.chesson_diagnostic` | ρ < κ_i/κ_j < 1/ρ: explica cuándo la ecología puede ayudar |

Error corregido: el ajuste fino de `baselines.py` (v0.7) cargaba el estado en un modelo nuevo y fallaba
con las memorias de tamaño variable; ahora se copia el modelo entero.

## Parada (Burgers, presupuesto de 4 por ronda, 4 rondas)

| Estrategia | Simulaciones | Error escalar en lo nuevo | Campos en lo nuevo |
|---|---|---|---|
| Inicial | 0 | 1.436 | 6.09 |
| Valor codicioso | 16 | **0.546** | 2.47 |
| Colmena v0.7 (quórum decide parar) | 4 | 0.972 | 3.37 |
| Colmena v0.8 (forrajeo + urgencia) | 16 | 0.647 (0.565 con 12) | **2.29** |

La v0.8 ya no se detiene antes de tiempo: elige 4 de 4 casos nuevos en cada ronda y baja de 0.972 a
0.647. Aquí la regla no paró, y era lo correcto: cada ronda traía 6 candidatos nuevos que seguían
enseñando (ganancia marginal por encima de la tasa media). Queda 0.10 por encima del codicioso al
final, aunque en la ronda 3 estuvo por debajo (0.565 contra 0.589): con una semilla es ruido. Falta un
escenario donde parar sea lo correcto (reserva que se agota de casos nuevos) para medir el ahorro.

## Cambio oculto y visible (AUROC, semilla 0)

| Detector | Burgers visible | Burgers oculto | Adv.-dif. visible | Adv.-dif. oculto |
|---|---|---|---|---|
| Mejor sin etiquetas (v0.7) | 0.89 | 0.49 | 1.00 | 0.54 |
| Inspección del sitio (v0.7) | 0.89 | 0.68 | 0.92 | 0.89 |
| **Centinela: sorpresa normalizada** | **0.98** | **0.76** | 0.98 | **1.00** |
| Centinela: martingala conformal | 0.98 | 0.73 | 0.98 | 1.00 |
| Centinela con 3 muestras al azar | 0.96 | 0.60 | 0.97 | 0.99 |
| Centinela con 3 muestras por sondeo activo | 0.93 | **0.72** | 0.92 | 1.00 |

Tasa de alarma de la martingala (α = 0.05; 8 muestras por paciente): conocidos 0 % en ambos dominios
(control de falsas alarmas); oculto 100 % en advección-difusión y 0 % en Burgers (la evidencia de 8
muestras no basta para cruzar 1/α = 20); visible 56-75 %.

**Lectura.** Normalizar el error por la incertidumbre esperada es lo que más ayuda: sube el cambio
oculto de 0.68 a 0.76 en Burgers y de 0.89 a 1.00 en advección-difusión. Con solo 3 muestras por
paciente, elegirlas por desacuerdo entre agentes sube el cambio oculto de 0.60 a 0.72 en Burgers; para
el cambio visible el sondeo no ayuda (0.93 frente a 0.96).

## Pocos ejemplos (Burgers, régimen intermedio, error escalar MAE / campos L2 relativo)

| Variante | k = 1 | k = 2 | k = 5 |
|---|---|---|---|
| Sin memoria | 0.334 / 0.50 | 0.334 / 0.49 | 0.344 / 0.49 |
| Memoria completa | 0.335 / 0.59 | 0.290 / 0.60 | 0.268 / 0.60 |
| Ajuste fino afinado | 0.365 / 0.59 | 0.317 / 0.56 | 0.287 / 0.49 |
| **Árbitro rápido-lento** | **0.326 / 0.48** | **0.274 / 0.47** | **0.249 / 0.46** |

El árbitro queda por debajo de ambas curvas en todos los k, en escalares y en campos (la propiedad de
varianza mínima). En Burgers la memoria completa ya le gana al ajuste fino también con k = 5 (0.268
frente a 0.287), pero empeora los campos; el árbitro conserva lo mejor de cada uno. El ajuste fino
cuesta 0.06-0.07 s por paciente y la memoria 0.006 s.

## Pendiente

- 10 semillas (`experiments/rejilla.py`) y el escenario de reserva que se agota para la regla de parada.
- Medir el diagnóstico de Chesson y la sucesión en aprendizaje continuo (sano → prediabetes → DT2).
- Discrepancia CA1 (energía de Hopfield) como detector, agentes no informados (Couzin 2011) y gating por
  contexto (Masse 2018): recomendados en la revisión, no implementados.

---

# Resultados v0.7: auditoría para Nature Computational Science (26 de noviembre de 2026)

60 pruebas pasan (2 solo corren con GPU). **Una sola semilla por experimento, en CPU: son
indicativos.** Las cifras para el artículo salen de `experiments/rejilla.py` con 10 semillas.

## Cambios

1. **Memoria plasmodial: dos errores corregidos.** (a) Los nodos se guardaban con la clave del episodio y
   la novedad de un candidato se preguntaba sin contexto: dos espacios distintos. Ahora cada nodo guarda
   también su posición sin contexto. (b) Los nodos escritos durante el entrenamiento quedaban en el espacio
   de codificadores viejos. Ahora la red se reconstruye con el codificador final (`rebuild_memory`).
2. **Novedad por resistencia efectiva de Kirchhoff** (`modelo.novedad: resistencia`). Un primer intento con
   conductancia a tierra baja (0.1) no separaba lo lejano de lo cercano: la resistencia a tierra tapaba la
   distancia. Con 10 sí separa, y un nodo usado de forma sostenida tiene menos resistencia que uno poco usado.
   La fórmula cerrada coincide con la solución directa del laplaciano aumentado (prueba).
3. **Abstención con garantía Learn-then-Test** (`colmena.calibracion: ltt`), una consulta por paciente.
   La secuencia de umbrales empieza en el mínimo número de casos que puede dar significancia
   (22 con riesgo 0.1 y δ = 0.1); si empezara en 1 se detendría siempre en el primer paso.
4. Líneas base, Burgers con resolvedor real, ley de Pries como verdad oculta, PDEBench, dominio desde
   archivo, The Well, rejilla y estadística: ver README.

## Novedad: AUROC frente a casos conocidos (`exp_novedad.py`, semilla 0)

Cambio visible: alta frecuencia (advección-difusión, Burgers) o policitemia (sangre).
Cambio oculto: régimen extremo de parámetros ocultos; en sangre, la ley de Pries como verdad.

| Detector | Adv.-dif. visible | Adv.-dif. oculto | Burgers visible | Burgers oculto | Sangre visible | Sangre oculto |
|---|---|---|---|---|---|---|
| Mahalanobis, entradas crudas | 0.94 | 0.47 | 0.52 | 0.47 | **0.99** | 0.52 |
| kNN coseno, clave de memoria | **1.00** | 0.54 | 0.86 | 0.44 | 0.98 | 0.46 |
| Mahalanobis, estado global | 0.98 | 0.48 | **0.89** | 0.47 | 0.96 | 0.51 |
| Plasmodial coseno (v0.6 → v0.7) | 0.98 → 0.95 | 0.47 | 0.42 → 0.82 | 0.47 | 0.64 → 0.86 | 0.49 |
| Plasmodial resistencia | 0.98 | 0.48 | **0.89** | 0.43 | 0.81 | 0.46 |
| Varianza del ensamble | 0.43 | 0.49 | 0.82 | 0.52 | 0.62 | 0.48 |
| Inspección del sitio (usa etiquetas) | 0.92 | **0.89** | **0.89** | **0.68** | 0.57 | 0.52 |

**Lectura.**
- Las correcciones de la memoria plasmodial importan: en Burgers pasa de 0.42 a 0.82 (coseno) y 0.89
  (resistencia); en sangre de 0.64 a 0.86.
- **Pero la memoria plasmodial no supera a un detector simple en el mismo espacio.** Con un cambio visible,
  kNN o Mahalanobis sobre representaciones del modelo llegan igual o más alto. En sangre y en
  advección-difusión basta con las entradas crudas: esas pruebas son fáciles. Burgers es el único dominio
  donde hacen falta rasgos aprendidos, por eso debe ser el dominio principal de la figura 3.
- La resistencia supera al coseno en Burgers (0.89 frente a 0.82) y en advección-difusión (0.98 frente a
  0.95), y queda por debajo en sangre (0.81 frente a 0.86). Con una semilla no se puede decidir; el valor por
  defecto sigue siendo `coseno`.
- **Con un cambio oculto, ningún detector sin etiquetas funciona** (AUROC ≈ 0.5). Solo la inspección del
  sitio lo detecta (0.89 y 0.68). Esto es lo que el artículo puede afirmar con fuerza.
- En sangre, la ley de Pries no se detecta ni por inspección (0.52): la colmena extrapola bien a ese tercer
  modelo y la discrepancia es pequeña frente al error típico.

## Qué simular a presupuesto igual (`exp_presupuesto.py`, Burgers, semilla 0)

4 rondas; reserva por ronda de 24 episodios conocidos y 6 nuevos (alta frecuencia); 4 simulaciones por
ronda para las líneas base. Error escalar (MAE) y de campos (L2 relativo) en lo nuevo.

| Estrategia | Simulaciones | Escalares, nuevo | Campos, nuevo | Escalares, conocido |
|---|---|---|---|---|
| Inicial | 0 | 1.436 | 6.09 | 0.307 |
| Aleatoria | 16 | 0.840 | 4.89 | 0.250 |
| Novedad codiciosa | 16 | 0.688 | **2.09** | 0.251 |
| Valor codicioso (la señal de la colmena, sin abejas) | 16 | **0.599** | 2.44 | **0.244** |
| Varianza del ensamble | 16 | 0.605 | 2.17 | 0.290 |
| Core-set | 16 | 0.724 | 2.58 | 0.273 |
| Colmena | **3** | 0.997 | 6.58 | 0.257 |

**Lectura.**
- Cualquier estrategia informada le gana al azar con el mismo presupuesto (0.60-0.72 frente a 0.84).
  Valor codicioso y varianza del ensamble quedan empatados.
- **La colmena para demasiado pronto en Burgers.** Por simulación elige bien: con 3 simulaciones su error
  (0.997) es menor que el de la curva codiciosa interpolada en 3 (1.18). Pero se detiene tras 3 y termina
  en 0.997 frente a 0.599 del codicioso con 16. Sus campos empeoran en la última ronda (4.88 → 6.58).
- Consecuencia para el artículo: la afirmación 3 no se sostiene hoy en Burgers. Antes de 10 semillas hay
  que revisar el umbral de paro (por ejemplo, exigir que la inspección haya visto la región nueva antes de
  parar, o calibrar `valor_minimo`), y reportar ahorro y pérdida juntos.

## Pendiente

- 10 semillas en el clúster (`experiments/rejilla.py`), incluidos `exp_pocos_ejemplos.py` (no se corrió aquí).
- Probar el lector de PDEBench y el conversor de The Well con archivos reales.
- MC dropout y MAML como líneas base (no implementados).

---

# Resultados v0.6: fábrica de modelos y regla de paro de la colmena (25 de noviembre de 2026)

48 pruebas pasan. Datos sintéticos, 2 semillas, 4 rondas con hasta 4 simulaciones por ronda.
Script: `experiments/exp_activo_v06.py`, escrito con la API del Gemelo (la misma que se usará con datos reales).

## Cambios

1. **Fábrica de modelos.** Configuración en YAML (`configs/`), registro de dominios, clase `Gemelo`
   (entrenar, calibrar, predecir, proponer, incorporar, guardar, cargar), línea de comandos
   (`python -m eho ...`) y plantilla de dominio nuevo. Un modelo para otra enfermedad o campo se genera
   sin tocar el código del EHO. El archivo `.eho` no guarda los datos de entrenamiento.
2. **Valor por inspección.** Cada caso simulado se compara con lo que la colmena predecía ANTES de
   aprenderlo, y ese error se guarda junto al caso. Un candidato vale la pena si es nuevo **o** si los casos
   simulados cerca de él salieron mal: v = 1 − (1 − v_novedad)(1 − v_error).
3. **Exploradoras.** Si la colmena se abstiene, se simula un candidato cuya vecindad aún no se ha
   inspeccionado. Así la colmena deja de simular cuando ya comprobó que predice bien, no cuando
   simplemente no sabe.

## Error final en casos nuevos (escalares MAE; campos L2 relativo) y simulaciones usadas

| Corrida | Inicial | Al azar | Colmena v0.5 (novedad) | v0.6 sin exploradoras | v0.6 con exploradoras |
|---|---|---|---|---|---|
| Adv.-dif. s0, campos | 5.08 | 2.57 (16 sim.) | 1.73 (10) | **1.38** (12) | 2.37 (7) |
| Adv.-dif. s1, campos | 4.77 | 2.82 (16) | 1.59 (8) | **1.50** (7) | **1.50** (9) |
| Sangre s0, escalares | 0.081 | **0.036** (16) | 0.054 (1) | 0.054 (1) | 0.039 (4) |
| Sangre s1, escalares | 0.093 | 0.036 (16) | 0.056 (6) | **0.035** (9) | 0.040 (8) |

Error en casos conocidos: sin cambios relevantes entre estrategias (advección-difusión 0.46–0.49 en
campos, desde 0.54–0.57 al inicio; sangre 0.020–0.038 en escalares, desde 0.029–0.039).

## Lectura

- **El valor por inspección corrige el paro prematuro.** En 3 de 4 corridas, v0.6 sin exploradoras
  termina con menos error en lo nuevo que v0.5 (en la cuarta empata). En sangre s1 iguala al azar con
  9 simulaciones en vez de 16. En advección-difusión s0 sigue simulando en la última ronda, cuando
  v0.5 ya se había detenido.
- **Arranque en frío.** En sangre s0 la novedad nunca se dispara con la policitemia. Como nada se
  simula, tampoco hay error que medir, y la colmena se queda quieta (v0.5 y v0.6 sin exploradoras:
  1 simulación en 4 rondas). Las exploradoras lo resuelven: con 4 simulaciones baja de 0.081 a 0.039,
  cerca del azar con 16.
- **El precio de explorar.** En advección-difusión s0, las exploradoras gastaron dos rondas en
  candidatos conocidos y el resultado empeoró (2.37 contra 1.38). Hay dependencia de la trayectoria:
  cada simulación cambia el modelo y con él lo que la colmena valora después.
- **Por defecto: exploradoras = 1.** Una colmena que nunca aprende lo nuevo es peor en clínica que una
  que gasta de más en algunas rondas. Con 2 semillas no se puede decidir más. Correr 10 semillas en el
  clúster cuesta poco y lo resolvería.


---

# Resultados v0.5: colmena más robusta, menos olvido y policitemia (25 de noviembre de 2026)

42 pruebas pasan. Datos sintéticos, 2 semillas. Script: `experiments/exp_activo_colmena.py`.

## Cambios

1. **Por qué se abstenía la colmena (v0.4):** los agentes se dividían 2 contra 2. Uno tenía la memoria
   plasmodial llena (512 nodos) y todo le parecía conocido; otro no tenía memoria plasmodial y su
   distancia a vecinos no separaba lo nuevo. Ahora todos tienen memoria plasmodial, y cuando se llena
   fusiona los nodos más parecidos en vez de dejar de crecer. AUROC de novedad en advección-difusión:
   1.00, 1.00, 0.99 y 0.23; la mayoría de 3 contra 1 basta para decidir.
2. **Menos olvido:** ajuste con ancla λ Σ (w − w0)² (λ = 0.01) y repetición de 60 episodios elegidos al
   azar. En v0.4 la repetición tomaba los primeros episodios, que eran todos de un solo régimen.
3. **Policitemia como novedad difícil en sangre:** hematocrito de 0.55 a 0.70 (hiperviscosidad), visible
   en las entradas. El mapa de novedad la detecta con AUROC de 0.80 a 0.97 y el error inicial es de 2 a 3
   veces el de lo conocido.

## Ciclo de aprendizaje activo (3 rondas, hasta 4 simulaciones por ronda)

| Dominio | Semilla | Estrategia | Simulaciones | Nuevas | Nuevo: escalares | Nuevo: campos | Conocido: campos |
|---|---|---|---|---|---|---|---|
| Advección-difusión | 0 | Inicial | — | — | 0.867 | 4.75 | 0.543 |
| | | Aleatoria | 12 | 2 | 0.756 | 2.96 | 0.461 |
| | | Promedio + ancla | 12 | 12 | 0.309 | 1.37 | 0.483 |
| | | Promedio sin ancla | 12 | 12 | **0.279** | **1.36** | 0.515 |
| | | Colmena + ancla | **5** | 5 | 0.443 | 1.58 | 0.475 |
| Advección-difusión | 1 | Inicial | — | — | 0.687 | 5.35 | 0.552 |
| | | Aleatoria | 12 | 3 | 0.633 | 3.83 | 0.474 |
| | | Promedio + ancla | 12 | 12 | 0.284 | 1.39 | 0.485 |
| | | Promedio sin ancla | 12 | 11 | **0.280** | **1.29** | 0.511 |
| | | Colmena + ancla | **4** | 4 | 0.455 | 2.72 | 0.483 |
| Sangre (policitemia) | 0 | Inicial | — | — | 0.048 | 0.149 | 0.076 |
| | | Aleatoria | 12 | 2 | 0.033 | 0.103 | 0.059 |
| | | Promedio + ancla | 12 | 8 | **0.028** | **0.091** | 0.060 |
| | | Promedio sin ancla | 12 | 12 | 0.033 | 0.091 | 0.067 |
| | | Colmena + ancla | **3** | 3 | 0.034 | 0.121 | 0.059 |
| Sangre (policitemia) | 1 | Inicial | — | — | 0.097 | 0.186 | 0.077 |
| | | Aleatoria | 12 | 3 | 0.043 | 0.111 | 0.064 |
| | | Promedio + ancla | 12 | 6 | 0.035 | 0.110 | 0.062 |
| | | Promedio sin ancla | 12 | 5 | **0.034** | **0.109** | 0.069 |
| | | Colmena + ancla | 1 | 0 | 0.077 | 0.159 | 0.061 |

**Lectura.**
- **Elegir lo nuevo funciona en los dos dominios.** Con policitemia, el promedio de novedad baja el error
  escalar en lo nuevo de 0.048-0.097 a 0.028-0.035, casi al nivel de lo conocido; al azar, a 0.033-0.043.
- **El ancla reduce el olvido.** El error de campos en lo conocido mejora en vez de empeorar (advección-
  difusión: 0.54-0.55 → 0.48; en v0.4 subía a 0.65-0.72). Sin ancla, 0.51. El costo es pequeño en lo
  nuevo (escalares 0.309 frente a 0.279 en una semilla).
- **La colmena ya decide, pero para demasiado pronto.** Elige casi siempre casos nuevos y en
  advección-difusión logra cerca del 85 % de la mejora con 5 simulaciones en vez de 12 (semilla 0).
  Pero en cuanto su memoria registra los primeros casos nuevos, ese tipo deja de parecerle nuevo
  aunque el modelo todavía lo prediga mal. En sangre, semilla 1, paró tras una sola simulación.
- **Lección:** novedad no es lo mismo que error. La regla de parada debe combinar novedad con
  inspección del sitio (el error en los casos recién simulados), como una abeja que vuelve a evaluar
  el sitio antes de dejar de bailar.

---

# Resultados v0.4: qué simular después, decidido por la colmena (25 de noviembre de 2026)

41 pruebas pasan. Datos sintéticos, 2 semillas. Script: `experiments/exp_activo_colmena.py`.

## Hallazgo 1: el desacuerdo entre expertos no sirve como valor; la memoria plasmodial sí

Para elegir qué simular, cada agente necesita estimar el valor de un candidato con un criterio
común. Probamos dos señales para distinguir candidatos nuevos de conocidos (AUROC por agente):

| Señal | Advección-difusión (alta frecuencia) | Sangre (D = 3-5.5 μm) |
|---|---|---|
| Desacuerdo entre los expertos del agente | 0.00-0.46 | 0.11-0.73 |
| Novedad de la memoria plasmodial | **1.00** | **0.80** |
| Distancia a 5 vecinos (estado global) | 0.30-0.73 | 0.99-1.00 |

Los expertos coinciden con exceso de confianza fuera de distribución: los modos de alta frecuencia
decaen casi a cero y todos predicen "casi nada". La memoria plasmodial sí sabe qué regiones ya
recorrió. `NoveltyValue` usa esa novedad (o la distancia a vecinos si el agente no tiene memoria
plasmodial) en una escala absoluta: lo típico vale ≈ 0.02 y lo nuevo se acerca a 1.

## Hallazgo 2: con muchas opciones igual de buenas, la colmena se traba

Con 6 u 8 candidatos casi igual de valiosos, la inhibición cruzada suprime a todos y solo
se compromete el 8-12 % de las exploradoras; con 2 o 3 decide. Las abejas evitan esto porque
descubren pocos sitios a la vez. Ahora la colmena decide entre las 3 mejores de cada turno.

## Ciclo de aprendizaje activo: 3 rondas, hasta 4 simulaciones por ronda

30 candidatos por ronda (24 conocidos, 6 nuevos). Error medio de los 4 agentes con 2 ejemplos de
contexto, después de la última ronda.

| Dominio | Semilla | Estrategia | Simulaciones | Nuevas | Campos, nuevo | Escalares, nuevo | Campos, conocido |
|---|---|---|---|---|---|---|---|
| Advección-difusión | 0 | Inicial | — | — | 5.18 | 0.77 | 0.53 |
| Advección-difusión | 0 | Aleatoria | 12 | 2 | 3.64 | 0.73 | 0.50 |
| Advección-difusión | 0 | Promedio de novedad | 12 | 12 | **1.65** | **0.28** | 0.65 |
| Advección-difusión | 0 | Colmena | 0 | 0 | 5.18 | 0.77 | 0.53 |
| Advección-difusión | 1 | Inicial | — | — | 5.93 | 0.66 | 0.56 |
| Advección-difusión | 1 | Aleatoria | 12 | 3 | 3.29 | 0.61 | 0.62 |
| Advección-difusión | 1 | Promedio de novedad | 12 | 12 | **1.13** | **0.25** | 0.72 |
| Advección-difusión | 1 | Colmena | **4** | 4 | 1.29 | 0.39 | 0.74 |
| Sangre | 0 | Aleatoria | 12 | 2 | 0.079 | 0.029 | 0.071 |
| Sangre | 0 | Promedio de novedad | 12 | 11 | 0.085 | 0.035 | 0.069 |
| Sangre | 0 | Colmena | 8 | 7 | 0.073 | 0.040 | 0.075 |
| Sangre | 1 | Aleatoria | 12 | 3 | 0.071 | 0.035 | 0.069 |
| Sangre | 1 | Promedio de novedad | 12 | 12 | 0.071 | 0.033 | 0.091 |
| Sangre | 1 | Colmena | 9 | 8 | 0.069 | 0.038 | 0.078 |

**Lectura.**
- **Elegir por novedad funciona.** En advección-difusión, simular lo nuevo baja el error en
  condiciones nuevas de 5.2-5.9 a 1.1-1.6; al azar, con el mismo presupuesto, solo a 3.3-3.6.
  El precio es algo de olvido en lo conocido (campos 0.53-0.56 → 0.65-0.72).
- **La colmena puede ahorrar cómputo.** En la semilla 1 logró casi toda la mejora (1.29 frente a 1.13)
  con 4 simulaciones en vez de 12 y luego decidió parar: tras consolidar su memoria, lo nuevo ya no
  le parecía nuevo. En la semilla 0 se abstuvo desde el inicio y no aprendió nada: el umbral de
  valor todavía no es robusto.
- **En sangre la elección importa poco:** el modelo ya extrapola bien a tubos de 3-5.5 μm, y las
  tres estrategias terminan con errores similares.

## Qué sigue

1. Hacer robusto el umbral de valor de la colmena (evitar la abstención total de la semilla 0),
   por ejemplo calibrando q90 con candidatos y no solo con validación.
2. Reducir el olvido de lo conocido al aprender lo nuevo (más repetición o consolidación de memoria).
3. Probar en un caso donde lo nuevo sea difícil en sangre (por ejemplo, otra enfermedad).

---

# Resultados v0.3: memoria indexada por episodio, colmena calibrada y diversa (25 de noviembre de 2026)

Mismos protocolos que v0.2 (abajo). 38 pruebas pasan. Datos sintéticos; 3 semillas en F, 2 en G.

## F (v0.3). Pocos ejemplos con memoria de largo plazo indexada por episodio

| Dominio | Variante | Escalares, k = 5 | Campos, k = 5 | Cambio frente a v0.2 |
|---|---|---|---|---|
| Advección-difusión | Memoria completa | 0.105 ± 0.024 | 0.557 ± 0.034 | Mejora (v0.2: 0.129 y 0.644); escalares al nivel del oráculo (0.100) |
| Advección-difusión | Memoria episódica | 0.118 ± 0.019 | **0.489 ± 0.039** | Sigue siendo la mejor en campos |
| Advección-difusión | Sin memoria | 0.131 ± 0.025 | 0.661 ± 0.025 | — |
| Sangre | Memoria completa | 0.017 ± 0.007 | 0.051 ± 0.005 | Igual o algo peor (v0.2: 0.013), dentro del ruido |
| Sangre | Ajuste fino (gradiente) | **0.0097 ± 0.0007** | **0.024 ± 0.001** | — |
| Sangre | Sin memoria | 0.029 ± 0.001 | 0.061 ± 0.016 | — |

**Lectura.** Indexar las memorias de largo plazo por episodio arregla el problema de v0.2 en
advección-difusión: la memoria completa ya no estorba y sus escalares igualan al oráculo. En
sangre no aporta; ahí la memoria completa reduce el error a la mitad respecto a no tener memoria,
pero el ajuste fino con 5 ejemplos sigue siendo mejor.

## G (v0.3). Colmena con quórum calibrado (cobertura objetivo 80 %), diversa frente a homogénea

Error relativo a la tolerancia de la colmena. "Fuera" = régimen extremo (advección-difusión) o
geometría nueva con D = 3-5.5 μm (sangre).

| Dominio | Semilla | Colmena | Regla | Cobertura | Error aceptadas | Aceptadas dentro | Aceptadas fuera |
|---|---|---|---|---|---|---|---|
| Advección-difusión | 0 | Diversa | Inspección + quórum | 0.38 | 0.78 | 0.76 | **0.01** |
| Advección-difusión | 0 | Diversa | Promedio ponderado | 0.40 | 0.83 | 0.80 | 0.00 |
| Advección-difusión | 0 | Diversa | Solo acuerdo | 0.51 | 1.23 | 0.83 | 0.19 |
| Advección-difusión | 0 | Homogénea | Inspección + quórum | 0.39 | 0.91 | 0.71 | 0.08 |
| Advección-difusión | 1 | Diversa | Inspección + quórum | 0.40 | 0.88 | 0.74 | **0.06** |
| Advección-difusión | 1 | Diversa | Promedio ponderado | 0.28 | 0.82 | 0.55 | 0.00 |
| Advección-difusión | 1 | Homogénea | Inspección + quórum | 0.52 | 0.93 | 0.86 | 0.17 |
| Sangre | 0 | Diversa | Inspección + quórum | 0.82 | 0.43 | 0.87 | 0.78 |
| Sangre | 0 | Diversa | Promedio ponderado | 0.90 | 0.36 | 0.85 | 0.95 |
| Sangre | 0 | Homogénea | Inspección + quórum | 0.48 | 0.73 | 0.71 | 0.24 |
| Sangre | 1 | Diversa | Inspección + quórum | 0.79 | 0.63 | 0.76 | 0.81 |
| Sangre | 1 | Diversa | Promedio ponderado | 0.75 | 0.54 | 0.75 | 0.75 |
| Sangre | 1 | Homogénea | Inspección + quórum | 0.80 | 0.63 | 0.83 | 0.78 |

**Lectura.**
- **El quórum calibrado funciona:** la aceptación dentro de distribución queda entre 71 % y 87 %,
  cerca del objetivo de 80 % (en v0.2 el quórum fijo aceptaba entre 3 % y 23 %).
- **La inspección del sitio sigue siendo lo decisivo:** frente a "solo acuerdo", reduce el error en lo
  aceptado (0.78-0.88 frente a 1.13-1.23) y casi elimina lo fuera de distribución en advección-difusión.
- **La diversidad ayuda en advección-difusión** (fuera aceptadas: 1-6 % frente a 8-17 % de la
  homogénea). En sangre el efecto es mixto: la colmena diversa con dos simuladores es mucho mejor en
  la semilla 0 (error 0.43 frente a 0.73) e igual en la 1.
- **La geometría nueva en sangre se acepta** (75-81 %) y sin mucho más error: el modelo extrapola
  razonablemente a tubos de 3-5.5 μm, así que rechazarla no sería lo correcto aquí.
- **La colmena no le gana al promedio ponderado con la misma señal.** Su valor es la abstención
  calibrada y el marco de decisión; la precisión es similar.
- **La elección del siguiente experimento sigue sin funcionar:** la colmena se abstiene en 9-10 de 10
  rondas (los agentes no coinciden en qué candidato es más valioso) y el promedio elige la geometría
  nueva 30 % de las veces (azar: 20 %). Hace falta una señal de valor común, por ejemplo la ganancia
  de información esperada sobre los parámetros ocultos.

---

# Resultados v0.2: marco general con memoria y colmena (25 de noviembre de 2026)

Datos sintéticos, CPU de 2 núcleos, 3 semillas (2 en la colmena). Scripts en `experiments/`;
números en `results/central_fewshot_*.json` y `results/colmena_s*.json`. 35 pruebas pasan.

## F. Adaptación con pocos ejemplos (memoria), episodios nuevos de un régimen intermedio

Cada episodio de prueba (un paciente o un fluido nuevo) tiene parámetros ocultos. Se dan k
ejemplos etiquetados del mismo episodio y se mide el error en los demás. "Ajuste fino" entrena
una copia del modelo sin memoria con esos k ejemplos (30 pasos de gradiente por episodio).

**Sangre** (entrenado con sano y DT2, prueba en prediabetes). Error escalar (log η y δ/R):

| Variante | k = 0 | k = 2 | k = 5 | Campos, k = 5 |
|---|---|---|---|---|
| Sin memoria | 0.029 ± 0.002 | 0.029 ± 0.002 | 0.029 ± 0.001 | 0.061 ± 0.016 |
| Ajuste fino (gradiente) | — | 0.020 ± 0.001 | **0.0097 ± 0.0007** | **0.024 ± 0.001** |
| Memoria episódica | 0.037 ± 0.018 | 0.039 ± 0.014 | 0.039 ± 0.014 | 0.057 ± 0.006 |
| Memoria completa | 0.031 ± 0.010 | **0.014 ± 0.003** | 0.013 ± 0.004 | 0.048 ± 0.010 |
| Oráculo (ve θ) | 0.007 ± 0.002 | | | 0.039 ± 0.004 |

**Advección-difusión** (entrenado con difusivo y advectivo, prueba en mixto):

| Variante | Escalares, k = 5 | Campos, k = 0 | Campos, k = 5 |
|---|---|---|---|
| Sin memoria | 0.131 ± 0.025 | 0.673 ± 0.026 | 0.661 ± 0.025 |
| Ajuste fino (gradiente) | 0.136 ± 0.003 | | 0.664 ± 0.021 |
| Memoria episódica | **0.103 ± 0.004** | 0.630 ± 0.014 | **0.520 ± 0.031** |
| Memoria completa | 0.129 ± 0.062 | 0.688 ± 0.036 | 0.644 ± 0.023 |
| Oráculo (ve κ y c) | 0.100 ± 0.009 | 0.470 ± 0.068 | 0.458 ± 0.065 |

**Lectura.**
- La memoria permite adaptarse sin reentrenar. En sangre, con 2 ejemplos la memoria completa
  reduce el error escalar a la mitad (0.029 → 0.014) y supera al ajuste fino con 2 ejemplos (0.020).
  Con 5 ejemplos, el ajuste fino gana (0.0097 frente a 0.013) y también en campos.
- En advección-difusión, la memoria episódica es la mejor variante: con 5 ejemplos su error escalar
  (0.103) iguala al del oráculo (0.100) y reduce el error de campos de 0.66 a 0.52. El ajuste fino no
  ayuda aquí.
- La memoria completa (con las memorias de largo plazo) ayuda en sangre pero no en advección-difusión,
  donde añade varianza. Las memorias de largo plazo guardan regularidades de la población; no
  distinguen episodios cuyos parámetros ocultos no se ven en las entradas.

## G. Colmena de 4 modelos (advección-difusión, fuera de distribución = régimen extremo)

| Semilla | Regla | Cobertura | Error en aceptadas | Aceptadas dentro | Aceptadas fuera |
|---|---|---|---|---|---|
| 0 | Colmena (solo acuerdo) | 0.68 | 1.51 | 0.79 | 0.56 |
| 0 | Promedio, misma cobertura | 0.68 | 1.48 | 0.78 | 0.57 |
| 0 | **Colmena con inspección del contexto** | 0.23 | 0.61 | 0.45 | **0.00** |
| 0 | Promedio ponderado con la misma señal | 0.23 | **0.58** | 0.45 | 0.00 |
| 1 | Colmena (solo acuerdo) | 0.13 | 1.86 | 0.16 | 0.11 |
| 1 | Promedio, misma cobertura | 0.13 | 0.89 | 0.23 | 0.03 |
| 1 | **Colmena con inspección del contexto** | 0.03 | **0.70** | 0.05 | 0.01 |
| 1 | Promedio ponderado con la misma señal | 0.05 | 0.83 | 0.10 | 0.00 |

Errores relativos a la tolerancia de la colmena (error típico en validación × 2).

**Lectura.**
- **El acuerdo solo no basta:** los cuatro agentes se equivocan juntos fuera de distribución
  (pensamiento de grupo) y la colmena acepta más de la mitad de esos casos.
- **La novedad de la memoria no detecta lo nuevo aquí** (AUROC 0.44 a 0.50): un fluido extremo se ve
  igual en las entradas; solo difiere en sus parámetros ocultos.
- **Inspeccionar el sitio sí funciona:** cada agente predice los ejemplos etiquetados del contexto y,
  si falla, baja su voto. Ese error detecta lo fuera de distribución con AUROC de 0.81 a 1.0 en 7 de 8
  agentes. Con esa señal, la colmena rechaza casi todo lo fuera de distribución y el error en lo
  aceptado baja de 1.5 a 0.6-0.7.
- **La ganancia viene de la señal, no de la regla de las abejas:** un promedio ponderado con la misma
  señal logra lo mismo en una semilla; la colmena gana en la otra. Y la colmena es muy conservadora
  (acepta 3-23 %): hay que calibrar el quórum.
- **Elegir el siguiente experimento no funcionó en este dominio:** sin etiquetas, los candidatos
  nuevos no se distinguen por sus entradas. La selección por colmena necesita un dominio donde la
  novedad sea visible en las entradas (por ejemplo, geometrías nuevas).

## Qué sigue (v0.2)

1. Calibrar el quórum de la colmena para una cobertura objetivo (por ejemplo 80 %).
2. Diversidad real entre agentes: simuladores distintos (Mirheo frente a HemoCell), datos distintos.
3. Memorias de largo plazo indexadas por episodio (no solo por entrada) para que distingan pacientes.
4. Repetir F y G con 5 semillas y en datos DPD reales.

---

# Resultados v0.1 con datos sintéticos (24 de noviembre de 2026)

Todos los números salen de `results/*.json` y se reproducen con los scripts de `experiments/`.
CPU de 2 núcleos; cada experimento tarda de 1 a 6 minutos. **Son datos sintéticos 2D, no DPD.**
Media ± desviación estándar entre semillas.

## Garantías matemáticas (18 pruebas, `tests/test_garantias.py`): todas pasan

| Garantía | Resultado |
|---|---|
| Prop. 1: la energía de Hopfield decrece en cada paso | Sí, 5 semillas × 30 pasos |
| Prop. 2: V(n) de Lotka-Volterra decrece | Sí, con dt = 0.05 y con el dt = 0.2 del entrenamiento (50 comunidades) |
| Prop. 3: disipación ≥ 0, Φ convexa, η_rel ≥ 1, curva de flujo monótona | Sí |
| GENERIC: L antisimétrica, M simétrica PSD, L∇S = 0, M∇E = 0, dE/dt = 0, dS/dt ≥ 0 | Sí |
| Red microvascular: conservación de plasma y glóbulos rojos | Residuo máximo ≈ 1e-12 (precisión de máquina) |
| Viscosidad del plasma monótona por componente; datos faltantes sin imputar | Sí |

## B. Aprendizaje continuo: sano → prediabetes → DT2 (3 semillas)

Error total (campos + log η + δ/R) en cada tarea después de cada etapa. Ambos modelos reciben la
misma repetición (40 muestras por tarea anterior).

| Modelo | Después de | Sano | Prediabetes | DT2 |
|---|---|---|---|---|
| EHO completo | sano | 0.237 ± 0.158 | 0.257 ± 0.180 | 0.301 ± 0.198 |
| EHO completo | prediabetes | 0.073 ± 0.020 | 0.074 ± 0.018 | 0.089 ± 0.018 |
| EHO completo | DT2 | **0.058 ± 0.016** | **0.057 ± 0.013** | **0.054 ± 0.016** |
| Sin ecología | sano | 0.130 ± 0.048 | 0.141 ± 0.031 | 0.162 ± 0.014 |
| Sin ecología | prediabetes | 0.068 ± 0.007 | 0.067 ± 0.008 | 0.082 ± 0.013 |
| Sin ecología | DT2 | 0.091 ± 0.028 | 0.087 ± 0.033 | 0.080 ± 0.023 |

**Lectura.** Al final de la secuencia, el EHO con ecología tiene entre 30 % y 35 % menos error. Sin ecología,
el error en sano sube de 0.068 a 0.091 al entrenar con DT2 (olvido); con ecología sigue bajando.
Punto débil: en la primera etapa el EHO completo es más lento e inestable (0.237 ± 0.158), porque
las primeras extinciones e invasiones perturban el entrenamiento cuando aún hay pocos datos.

## A. Tarea estática (3 semillas)

Error en prueba (600 muestras mezcladas). Campos = L2 relativo de velocidad y φ.

| n entrenamiento | Modelo | Campos | log η (MAE) | δ/R (MAE) |
|---|---|---|---|---|
| 150 | EHO completo | 0.062 ± 0.012 | 0.020 ± 0.004 | 0.0083 ± 0.0015 |
| 150 | EHO sin memoria | **0.046 ± 0.007** | 0.019 ± 0.008 | 0.011 ± 0.002 |
| 150 | EHO sin ecología | 0.077 ± 0.025 | 0.031 ± 0.014 | 0.016 ± 0.012 |
| 150 | MLP base | 0.092 ± 0.001 | **0.008 ± 0.001** | **0.0043 ± 0.0003** |
| 600 | EHO completo | 0.049 ± 0.009 | 0.015 ± 0.005 | 0.0063 ± 0.0017 |
| 600 | EHO sin memoria | **0.039 ± 0.010** | 0.022 ± 0.012 | 0.017 ± 0.017 |
| 600 | EHO sin ecología | 0.048 ± 0.004 | 0.016 ± 0.004 | 0.0069 ± 0.0008 |
| 600 | MLP base | 0.085 ± 0.003 | **0.010 ± 0.003** | 0.0058 ± 0.0029 |

**Lectura.** Las variantes del EHO predicen los campos con entre 45 % y 55 % del error del MLP. En los
escalares gana el MLP, porque aquí son funciones suaves de 7 entradas. La memoria calibrada reduce
el error escalar y su variación (log η a 600: 0.015 ± 0.005 frente a 0.022 ± 0.012 sin memoria), pero
usa el 20 % de los datos para calibrar y eso cuesta precisión en campos.

## C. Red microvascular con la viscosidad aprendida

Árbol de 4 niveles (16 ramas terminales), misma caída de presión.

| | Sano | DT2 |
|---|---|---|
| Flujo total (unidades arbitrarias) | 561.7 | 479.5 |
| Caída de flujo de DT2 frente a sano | | 14.6 % |
| Coeficiente de variación de la perfusión terminal | 0.186 | 0.185 |
| Residuo máximo de masa / de glóbulos | 7e-13 / 1e-14 | 1e-12 / 0 |

La heterogeneidad no cambia porque la corrección neuronal de separación de fases aún no está
entrenada (falta el dataset de bifurcaciones resueltas célula a célula).

## D. Inferencia inversa por flow matching (150 pacientes sintéticos)

El EHO es el simulador para entrenar la posterior; se evalúa con observaciones del simulador
"verdadero". Incluye la discrepancia del modelo (sesgo medio + varianza) medida en 500 casos.

| Parámetro | Cobertura del intervalo 90 % | Error de la mediana | Ancho del intervalo | Ancho de la previa |
|---|---|---|---|---|
| Rigidez s | 95 % | 0.029 | 0.15 | 0.80 |
| Agregación a | 98 % | 0.105 | 0.54 | 0.80 |
| Déficit de volumen v | 94 % | 0.123 | 0.52 | 0.71 |
| Factor de plasma p | 87 % | 0.153 | 0.57 | 0.70 |

En una corrida previa sin la corrección de discrepancia, la cobertura de la rigidez fue 68 % a 71 %.
El parámetro p no afecta la viscosidad relativa en estos datos y su posterior queda cerca de la previa.

## E. Qué simular después (2 semillas, 120 muestras iniciales + 3 rondas de 60)

| Estrategia | Costo DPD relativo | Campos | log η | δ/R |
|---|---|---|---|---|
| Aleatoria | 626 | 0.061 | **0.022** | **0.0065** |
| Adquisición (novedad + desacuerdo)/costo | **286** | **0.058** | 0.025 | 0.0084 |
| Política de refuerzo (REINFORCE) | **279** | 0.094 | 0.025 | 0.012 |

La adquisición logra un error similar con el 46 % del cómputo. La política de refuerzo,
simplificada, todavía no supera a la regla fija. Con 2 semillas el resultado es indicativo.

## Revisión del código (24 de noviembre)

Errores encontrados y corregidos antes de esta corrida:

1. Al entrenar a un mutante, el optimizador movía también las claves de los expertos residentes y el
   centro del nicho se actualizaba con muestras sesgadas. Ahora solo se mueven el mutante y su clave.
2. La regla de extinción no extinguía a nadie si quedaban menos de 2 especies. Ahora sobreviven las 2
   más abundantes.
3. En el aprendizaje activo, la novedad podía ser negativa y al dividirla entre el costo se preferían
   simulaciones caras. Ahora se desplaza a valores positivos.

Pendientes conocidos (no son errores, son partes sin implementar):

- La memoria externa no entra en la energía de Hopfield (solo en la lectura y la novedad).
- Los fósiles se guardan, pero aún no se recuperan como semilla de mutantes.
- La red microvascular no es diferenciable de extremo a extremo (la separación de fases pasa por
  conversiones de tipo); falta la diferenciación implícita.
- El codificador del panel sanguíneo, la viscosidad del plasma y la celda GENERIC están probados, pero
  no conectados al entrenamiento del EHO.

## Qué sigue

1. Reemplazar los datos sintéticos por DPD (Mirheo/LAMMPS) con `eho/data/dpd_io.py`.
2. Un periodo de calentamiento más largo en la primera tarea, para evitar la inestabilidad inicial.
3. Probar la ecología con un cambio de distribución más fuerte: bifurcaciones o sepsis con leucocitos.
4. Entrenar la separación de fases con simulaciones de bifurcación.
5. Sustituir REINFORCE por DAD o RL-DAD y penalizar la falta de cobertura del espacio de diseño.
6. Cinco semillas o más antes de reportar cualquier cifra en un artículo.
