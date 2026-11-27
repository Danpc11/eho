# eho

HemoTwin-EHO: EcoHopfield Operator. Repositorio: https://github.com/Danpc11/eho

> **v0.10:** memoria bayesiana de bajo rango con incertidumbre explícita, diseño de mediciones orientado
> a un objetivo, regla de parada con ganancia medida y prerregistro del análisis
> (`docs/v0.10_MEMORIA_BAYESIANA_Y_PRERREGISTRO.md`). El prerregistro está SIN confirmar.
>
> **v0.9.7:** correcciones de método y de código previas al clúster (`docs/AUDITORIA_EXTERNA_v0.9.6.md`,
> con la lista de comprobación y la hoja de ruta matemática).
>
> **v0.9.6:** formato de despliegue seguro (sin pickle), un certificado de abstención por protocolo de
> contexto, identificador y manifiesto por simulación (`docs/PENDIENTES_v0.9.6.md`).
>
> **v0.9.5:** tercera auditoría externa y auditoría propia (`docs/AUDITORIA_EXTERNA_v0.9.4.md`).
>
> **v0.9.4:** segunda auditoría externa (21 hallazgos; `docs/AUDITORIA_EXTERNA_v0.9.3.md`). La colmena es un
> ensamble con abstención colectiva calibrada: su dinámica no mejora la predicción frente a la media.
>
> **v0.9.3:** correcciones de una auditoría externa (13 hallazgos; ver `docs/AUDITORIA_EXTERNA_v0.9.2.md`
> y `CHANGELOG.md`). Estado: todo está probado solo con datos sintéticos.
>
> **v0.9:** auditoría de eficiencia. `compact()` (−72 % de parámetros), centinela en un paso por agente,
> regla de parada calibrada, árbitro entre tres predictores, decaimiento coseno opcional y curvas de
> aprendizaje (`experiments/exp_eficiencia.py`, `exp_estabilidad.py`). Ver `RESULTADOS.md`.
>
> **v0.8:** mecanismos biológicos con modelo matemático: parada por forrajeo (`forrajeo.py`), centinela
> ACh/NE con martingala conformal y sondeo activo (`centinela.py`), árbitro rápido-lento (`mezcla.py`),
> sucesión y diagnóstico de Chesson. Ver `RESULTADOS.md`.
>
> **v0.7 (26 de noviembre de 2026): auditoría para *Nature Computational Science*.** Ver la sección
> "Qué cambió en la v0.7" al final de este archivo y `RESULTADOS.md`.

Implementación en PyTorch del **EcoHopfield Operator (EHO)** descrito en el documento de diseño
"EcoHopfield Operator: arquitectura y matemáticas de un modelo de IA evolutivo para hemorreología multiescala".

El EHO aprende la reología de la sangre resuelta célula a célula, la lleva a redes microvasculares
e invierte los parámetros mecánicos de cada paciente con incertidumbre. Su estructura cambia con los
datos: los expertos forman un ecosistema que compite (Lotka-Volterra), se extingue y admite mutantes
solo si su aptitud de invasión es positiva.

> **Estado:** el código está completo y verificado con **datos sintéticos** (modelo de Quemada + efecto
> Fåhræus-Lindqvist de dos capas). No sustituye a las simulaciones DPD: es el banco de pruebas
> mientras se generan en el clúster. Ver `RESULTADOS.md`.

## Marco general (v0.2): tres módulos

El EHO ya no es solo un modelo de sangre: es un marco de propósito general con tres módulos.
La sangre es un dominio más.

| Módulo | Archivo | Qué hace | Inspiración |
|---|---|---|---|
| **Central de IA** | `eho/central.py` | Rebanado → núcleo de Hopfield con energía → ecosistema de expertos (Lotka-Volterra). Funciona con cualquier dominio | Transolver, Energy Transformer, ecología de MacArthur |
| **Memoria** | `eho/memory/` | Episódica en contexto (adaptación con pocos ejemplos, sin gradiente) + neuronal de largo plazo que aprende en uso con sorpresa y olvido + plasmodial (conductancias tipo *Physarum*, refuerzo por uso, umbral, consolidación biestable, habituación) | Titans (NeurIPS 2025); Kramar y Alim (PNAS 2021); Tero et al.; Chatterjee et al. (2025) |
| **Colmena** | `eho/colony.py` | Varios modelos centrales deciden por reclutamiento proporcional al valor, inhibición cruzada y quórum; se abstienen si no hay acuerdo o si todo es malo; eligen el siguiente experimento | Seeley et al. (Science 2012); Pais et al. (PLoS ONE 2013) |

**Dominios** (`eho/dominios/`): una clase con `generate(...)` y, opcionalmente, `memory_values` y
`postprocess`. Los datos se organizan en episodios (un paciente, un fluido, un material) con
parámetros ocultos; el modelo aprende a inferirlos con pocos ejemplos del mismo episodio.
Incluidos: `Sangre` (paciente con mecánica de glóbulos rojos oculta) y `AdveccionDifusion1D`
(fluido con difusividad y velocidad ocultas).

```python
from eho.dominios import AdveccionDifusion1D
from eho.central import EHOCentral
from eho.train_central import fit_central, evaluate_fewshot

dom = AdveccionDifusion1D()
modelo = EHOCentral(dom, memory="completa", d_val=5)       # "ninguna" | "episodica" | "completa"
fit_central(modelo, dom.generate(60, 8, "difusivo"), dom.generate(10, 8, "difusivo", seed=1))
prueba = dom.generate(20, 12, "mixto", seed=2)
print(evaluate_fewshot(modelo, prueba, k_context=5))      # 5 ejemplos del episodio, sin reentrenar
```

Para un dominio nuevo basta escribir su clase `Dominio`; el núcleo, la memoria y la colmena no cambian.

Experimentos del marco general (ver `RESULTADOS.md`, sección v0.2):

```bash
for s in 0 1 2; do for d in sangre edp1d; do PYTHONPATH=. python3 experiments/exp_central_fewshot.py $d $s 30; done; done
for s in 0 1; do PYTHONPATH=. python3 experiments/exp_colmena.py $s 30; done
```

Colmenas diversas (v0.3): `eho/hive.py` construye agentes con datos, simulador, arquitectura y
memoria distintos; `ColonyPredictor` calibra el quórum para una cobertura objetivo y usa la
inspección del sitio (error de cada agente en los ejemplos de contexto) como confianza.

```python
from eho.hive import build_hive, predict_with_inspection
from eho.colony import ColonyPredictor
agentes = build_hive([dom_a, dom_b], [datos_a, datos_b], val, n_agents=4, diverse=True)
Pv, Yv, CEv = predict_with_inspection(agentes, val)
colmena = ColonyPredictor(escala).fit(Pv, Yv, CEv, target_coverage=0.8)
P, _, CE = predict_with_inspection(agentes, nuevos)      # predicciones y error de inspección
pred, aceptada, apoyo = colmena.predict(P, CE)
```

Qué simular después (v0.4): `NoveltyValue` (en `eho/active.py`) da a cada agente un valor
absoluto en [0, 1] basado en la novedad de su memoria plasmodial, y `colony_select(..., normalize=False)`
elige entre las 3 mejores o se abstiene si nada vale la pena. Experimento: `experiments/exp_activo_colmena.py`.

Lo que mostraron, en breve (ver `RESULTADOS.md`): la memoria permite adaptarse a un paciente o
fluido nuevo con 2 a 5 ejemplos sin reentrenar; la colmena calibrada acepta cerca del 80 % de lo
conocido y, con inspección del sitio y agentes diversos, rechaza casi todo lo fuera de distribución
en advección-difusión, aunque no supera en precisión a un promedio ponderado con la misma señal.

## Generar un modelo (v0.6)

Desde la v0.6 el repositorio es una **fábrica de modelos**: un archivo de configuración describe el
modelo completo (dominio, arquitectura, colmena, entrenamiento, aprendizaje activo) y la línea de
comandos lo genera, lo guarda y lo hace crecer. No hace falta tocar el código del EHO para otra
enfermedad u otro campo.

```bash
# 1. ver qué dominios hay y crear uno nuevo (plantilla que ya funciona)
python -m eho dominios
python -m eho nuevo-dominio glucosa          # crea dominios_propios/glucosa.py y configs/glucosa.yaml

# 2. entrenar la colmena y guardarla (pesos + memorias + calibración en un solo archivo)
python -m eho entrenar configs/sangre.yaml -o modelos/hemotwin.eho --dispositivo cuda

# 3. usarla
python -m eho info      modelos/hemotwin.eho
python -m eho predecir  modelos/hemotwin.eho consulta.pt --contexto mediciones_del_paciente.pt
python -m eho proponer  modelos/hemotwin.eho candidatos.pt --presupuesto 4   # qué simular después (o nada)
python -m eho incorporar modelos/hemotwin.eho simulados.pt -o modelos/hemotwin_v2.eho
```

Lo mismo desde Python:

```python
from eho import Gemelo
g = Gemelo.desde_config("configs/sangre.yaml").entrenar()
r = g.predecir(consulta, contexto)        # r["escalares"], r["campos"], r["aceptado"] (abstención de la colmena)
elegidos = g.proponer(candidatos)         # [{episodio, valor, novedad, error}, ...]; puede ser []
g.incorporar(simulados)                   # registra el error previo, reentrena con ancla + repaso, consolida memoria
g.guardar("modelos/hemotwin.eho")         # sin datos de entrenamiento dentro (seguro con datos clínicos)
```

| Pieza | Archivo |
|---|---|
| Configuración declarativa (YAML/JSON) | `eho/config.py`, `configs/*.yaml` |
| Registro de dominios (`@registrar_dominio`, o `"paquete.modulo:Clase"` para dominios fuera del repo) | `eho/dominios/base.py` |
| Gemelo: construir, entrenar, calibrar, predecir, proponer, incorporar, guardar, cargar | `eho/gemelo.py` |
| Línea de comandos | `eho/__main__.py` |
| Plantilla de dominio nuevo | `eho/plantilla_dominio.py` |

**Datos propios.** Un dominio solo necesita `generate(...)` si hay simulador. Con datos ya
existentes (DPD, HemoCell, mediciones), se guardan como diccionario de tensores (`torch.save`) con el
formato de `eho/dominios/base.py` y se apuntan en la configuración (`datos_entrenamiento`,
`datos_validacion`). El archivo `.eho` **no** guarda los datos de entrenamiento salvo que se pida
(`guardar(..., incluir_datos=True)`): un modelo entrenado con datos de MIMIC no debe llevar
registros de pacientes dentro.

## Estructura (v0.1, modelo de sangre original)

| Archivo | Sección del documento | Qué hace | Garantía verificada en `tests/` |
|---|---|---|---|
| `eho/slicing.py` | 2. Rebanado físico | N partículas → M estados físicos, costo O(N·M·d) | — |
| `eho/hopfield.py` | 3. Núcleo de Hopfield | Energía de atención + memoria; T pasos con pesos compartidos; memoria clave-valor, novedad ν, fósiles, consolidación | Prop. 1: la energía decrece; jacobiano de g simétrico y PSD |
| `eho/ecology.py` | 4. Ecología de expertos | Nicho por afinidad, Lotka-Volterra, extinción, mutantes, aptitud de invasión Λ | Prop. 2: V(n) es función de Lyapunov |
| `eho/thermo.py` | 5. Cierre termodinámico | Curva de flujo monótona aprendida ⇒ potencial de disipación convexo; celda GENERIC | Prop. 3: disipación ≥ 0, Φ convexa, η_rel ≥ 1; L∇S = 0, M∇E = 0, dS/dt ≥ 0 |
| `eho/network.py` | 6. Red microvascular | Kirchhoff exacto, separación de fases de Pries + red neuronal, punto fijo | Conservación de plasma y glóbulos a precisión de máquina |
| `eho/inference.py` | 7. Inferencia inversa | Flow matching posterior + calibración SBC | — (ver experimento D) |
| `eho/panel.py` | Panel sanguíneo ampliado | Codificador que tolera análisis faltantes; viscosidad del plasma monótona por componente | Monotonía por signo (HDL negativo) y faltantes sin imputar |
| `eho/active.py` | Aprendizaje activo / refuerzo | Adquisición (novedad + desacuerdo)/costo; política REINFORCE de diseño experimental | — (ver experimento E) |
| `eho/train.py` | Entrenamiento | Tres escalas de tiempo; calibración de memoria post-entrenamiento | — |
| `eho/data/synthetic.py` | — | Datos sintéticos por régimen: sano, prediabetes, DT2 | — |
| `eho/data/dpd_io.py` | — | Lector de dumps de LAMMPS y formato `.npz` para datos DPD reales | — |

## Instalación en el servidor (4 × RTX A4000)

```bash
git clone <url-del-repositorio> && cd hemotwin_eho
bash scripts/instalar_servidor.sh      # entorno conda, PyTorch con CUDA, pruebas en GPU, velocidad
conda activate hemotwin
EHO_DEVICE=cuda:0 PYTHONPATH=. python3 ejemplo_rapido.py
```

`EHO_DEVICE` elige el dispositivo (`cpu`, `cuda:0` … `cuda:3`); por defecto usa la GPU si existe.
Para correr cuatro experimentos a la vez, uno por GPU, asigne `EHO_DEVICE=cuda:0`, `cuda:1`, etc.
El servidor no tiene gestor de colas: lance trabajos largos dentro de `tmux` o con `nohup`.

## Uso rápido

```bash
pip install torch numpy scipy matplotlib pytest
PYTHONPATH=. python3 -m pytest -q tests          # 18 pruebas de garantías (+2 que solo corren con GPU)
PYTHONPATH=. python3 ejemplo_rapido.py           # entrena, calibra y predice (~1 min en CPU)
```

Experimentos (cada uno escribe en `results/`):

```bash
PYTHONPATH=. python3 experiments/exp_ablacion.py                     # A: ablaciones, 3 semillas
for s in 0 1 2; do PYTHONPATH=. python3 experiments/exp_continuo.py 15 $s; done   # B: 3 semillas
PYTHONPATH=. python3 experiments/exp_red_inversa.py 25 --discrepancia # C y D: red e inversión
PYTHONPATH=. python3 experiments/exp_activo.py                       # E: qué simular después
```

## Conectar datos DPD reales

1. Cada simulación se guarda como `.npz` con `points`, `fields`, `scalars`, `theta`, `H`, `D`, `gdot`
   (ver `eho/data/dpd_io.py`). `lammps_dump_to_sample` convierte un dump de LAMMPS promediado.
2. La viscosidad aparente (`scalars[0] = log η_rel`) se calcula a partir del caudal y la caída de presión.
3. `load_dataset("ruta/*.npz")` devuelve el diccionario que consume `fit`.
4. Para 3D completo, aumente `n_slices`, `d` y use GPU: el código es agnóstico al dispositivo
   (mueva modelo y tensores con `.to("cuda")`).

## Decisiones de diseño que cambiaron al implementar

El código difiere del documento en cuatro puntos, todos por evidencia de los experimentos:

1. **Curva de flujo en vez de red convexa.** Parametrizar ψ con una red convexa daba viscosidades
   iniciales absurdas y peor precisión. Ahora se aprende el esfuerzo τ(γ̇) ≥ 0 monótono, y ψ = ∫τ
   es convexa por construcción. La Proposición 3 se mantiene exactamente.
2. **Nicho por afinidad, abundancia solo en la mezcla.** Si la abundancia decide qué datos recibe cada
   experto, el dominante los acapara y hay exclusión competitiva (lo observamos). La aptitud se mide
   en el nicho propio, no contra la mezcla.
3. **Representaciones centradas para el nicho.** Sin centrar, todas las muestras caen en el mismo nicho
   (α ≈ 1) y sobrevive una sola especie. Con centrado coexisten especies con solapamiento parcial.
4. **Memoria: corrección solo después de entrenar.** Escrita durante el entrenamiento, la memoria
   guardaba claves obsoletas y empeoraba el modelo. Ahora, durante el entrenamiento solo mide novedad
   y guarda fósiles; la corrección se escribe al final sobre un conjunto de calibración y se activa
   solo si mejora la validación.

## Limitaciones

- Todos los resultados son con datos sintéticos de sección transversal 2D; el desempeño con DPD 3D
  está por medirse.
- La ecología aporta en aprendizaje continuo (entre 30 % y 35 % menos error final y sin olvido, 3 semillas),
  pero no en tareas estáticas, y desestabiliza la primera etapa del entrenamiento. Hace falta probarla
  con cambios de distribución más fuertes (otra geometría, otra enfermedad).
- Pendientes de implementación: memoria externa dentro de la energía, recuperación de fósiles,
  diferenciación implícita de la red, conexión del panel sanguíneo y de GENERIC al entrenamiento.
- La separación de fases usa la ley de Pries sin entrenar la corrección neuronal (falta el dataset
  de bifurcaciones resueltas célula a célula).
- La política de refuerzo para diseñar simulaciones es una versión simplificada (REINFORCE con
  recompensa aproximada); DAD o RL-DAD son el reemplazo natural.


## Qué cambió en la v0.7

La v0.7 sale de una auditoría de las cinco afirmaciones del artículo. Cada cambio corrige algo que
un revisor señalaría, o añade la línea base que pediría.

| Cambio | Archivo | Por qué |
|---|---|---|
| La memoria plasmodial guarda la posición sin contexto de cada nodo, además de su clave por episodio | `memory/plasmodial.py`, `memory/system.py` | La novedad de un candidato (sin etiquetas) se comparaba con claves que llevan el descriptor del episodio: dos espacios distintos |
| La memoria plasmodial y el mapa de novedad se reconstruyen con el codificador final | `train_central.py` (`rebuild_memory`), `gemelo.py` | Los nodos escritos en épocas tempranas quedaban en un espacio que ya no existía |
| Novedad por resistencia efectiva de Kirchhoff (`modelo.novedad: resistencia`) | `memory/plasmodial.py` | Hasta v0.6 la "novedad plasmodial" era un vecino más cercano: no usaba ni el flujo ni las conductancias |
| Abstención con garantía de muestra finita (`colmena.calibracion: ltt`) | `colony.py`, `gemelo.py` | Learn-then-Test: P(error > tolerancia \| aceptada) ≤ riesgo con probabilidad ≥ 1 − δ, una consulta por paciente |
| `modelo.ecologia: false` | `central.py`, `config.py` | Ablación del núcleo sin Lotka-Volterra |
| Líneas base | `baselines.py` | 9 detectores de novedad en el mismo espacio, 5 estrategias de qué simular, ajuste fino afinado con costo medido |
| Burgers 1D con resolvedor pseudoespectral | `dominios/burgers1d.py` | Física no lineal sin solución cerrada; validado contra Cole-Hopf (error 1e-8) |
| Ley de Pries (in vitro, 1992) como tercer modelo de sangre | `data/synthetic.py` (`simulador: pries`) | La "realidad" de v0.6 era el promedio de los simuladores de los propios agentes |
| Lector de PDEBench 1D | `dominios/pdebench.py` | Datos públicos generados por otros; cada archivo es un parámetro oculto |
| Dominio desde archivo (reserva de simulaciones) | `dominios/archivo.py` | La reserva DPD de 300-600 simulaciones y The Well entran sin código nuevo |
| Conversor de The Well | `scripts/convertir_the_well.py` | Sin probar con un archivo real: use primero `--inspeccionar` |
| Experimentos del artículo | `experiments/exp_novedad.py`, `exp_presupuesto.py`, `exp_pocos_ejemplos.py` | Presupuesto igual para todas las estrategias; curvas; arrepentimiento de la regla de paro |
| Rejilla de 10 semillas en las 4 GPU y estadística | `experiments/rejilla.py`, `analisis/estadistica.py` | IC 95 % por bootstrap y Wilcoxon pareado |

```bash
PYTHONPATH=. python3 -m pytest -q tests                                   # todas las pruebas
PYTHONPATH=. python3 experiments/rejilla.py --gpus 0,1,2,3 --semillas 0-9  # en el clúster, dentro de tmux
python3 analisis/estadistica.py                                            # tablas con IC y p-valores
```

`h5py` solo hace falta para PDEBench y The Well (`pip install h5py`).
