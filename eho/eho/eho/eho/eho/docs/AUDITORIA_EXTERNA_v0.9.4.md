# Respuesta a la tercera auditoría externa (sobre la v0.9.4) y auditoría propia

Fecha: 27 de noviembre de 2026. Versión que responde: v0.9.5. Pruebas nuevas: `tests/test_v095.py` (12).

## Hallazgos del evaluador

| # | Hallazgo | Veredicto | Cambio en v0.9.5 | Prueba |
|---|---|---|---|---|
| 3.1 | Hematocrito imposible (H ≈ 4000) marcado como válido en nodos de frontera con paso de flujo | **Confirmado** | Los nodos con presión impuesta y aristas de entrada y salida intercambian sangre con el exterior: si entra, trae `H_in[nodo]` (sin él, la red queda sin resolver); si sale, la extracción es bien mezclada. `valida` exige además variables finitas, H en [0, H_max], conservación de volumen y de glóbulos, identidad y consistencia constitutiva; se devuelve `comprobaciones` | `test_boundary_node_with_throughflow`, `test_valid_flag_requires_physical_range` |
| 3.2 | La huella del centinela colisiona al permutar pesos y no ve buffers | **Confirmado** | SHA-256 del `state_dict` completo (nombre, forma, tipo, contenido) más el protocolo (k, sondeo, α) | `test_fingerprint_changes_on_permutation_and_buffers` |
| 3.3 | El hash de episodios depende del orden de las muestras | **Confirmado** | Huella invariante al orden y comprobación adicional por muestra (detecta el mismo paciente con menos observaciones). **Pendiente:** identificadores persistentes de paciente y simulación | `test_episode_hash_order_invariant_and_partial_overlap` |
| 3.4 | El certificado LTT no estaba ligado al protocolo de contexto | **Confirmado** | Se guarda el protocolo de calibración; con otro número de mediciones la predicción se devuelve pero no se certifica (`protocolo_coincide`); un contexto con varios pacientes da error. **Pendiente:** umbrales separados para k = 0, 1, 2, 5 | `test_ltt_certificate_bound_to_context_protocol` |
| 3.5 | La independencia de los datos dependía del usuario | **Confirmado** | `auditar_particiones()` es obligatoria al entrenar y al calibrar (entrenamiento, validación, evolución, calibración) | `test_provenance_audit_is_mandatory` |
| 4.1 | El presupuesto solo contaba condiciones físicas | **Confirmado** | Tres topes independientes: condiciones, ejecuciones (`presupuesto_ejecuciones`) y costo (`presupuesto_costo`, con `incorporar(costo=...)`) | `test_execution_and_cost_budgets` |
| 4.2 | No se validaba la correspondencia entre conjuntos por simulador | **Confirmado** | Un conjunto por agente, mismos episodios y mismas condiciones visibles; `zip(strict=True)`. **Pendiente:** manifiesto por ejecución | `test_incorporar_validates_and_is_transactional` |
| 4.3 | El aprendizaje continuo no era transaccional | **Confirmado** | `incorporar()` restaura el estado completo si algo falla | misma prueba |
| 4.4 | El archivo de despliegue llevaba la validación | **Confirmado**, parcial | El despliegue ya no incluye validación ni datos. Los archivos de DATOS se cargan sin pickle (`weights_only=True`). **Pendiente:** el archivo de modelo sigue siendo pickle | `test_deployment_file_has_no_validation_data` |
| 5.2 | No se había demostrado que la colmena sea necesaria para abstenerse | **De acuerdo** | Experimento nuevo `exp_abstencion.py` (ver `RESULTADOS.md`) | — |
| 5.3 | La comparación de v0.9.4 usaba cobertura, no LTT | **De acuerdo** | Aclarado en `RESULTADOS.md`; el experimento nuevo usa LTT para todos | — |
| 7.1 | El experimento de red inversa no revisaba `valida` | **Confirmado** | Falla si la solución no es válida y guarda residuos e iteraciones | — |
| 7.2 | El diámetro del vaso padre no era diferenciable | **Confirmado** | Sin `float()` ni `torch.tensor([...])`; gradiente comprobado con diferencias finitas | `test_parent_diameter_gradient` |
| 7.3 | Viscosidad con cizalla fija de 100 s⁻¹ | **Confirmado**, parcial | `solve(shear_coupled=True)` pasa a la viscosidad la cizalla local 32\\|Q\\|/(πD³) y converge con ella. **Pendiente:** usarla en `exp_red_inversa.py` con unidades físicas y quitar el `no_grad` para el problema inverso de extremo a extremo | `test_shear_coupled_solve` |

## Auditoría propia (errores que el evaluador no señaló)

| # | Hallazgo | Cómo se encontró | Estado |
|---|---|---|---|
| P1 | **La línea de comandos quedó rota en la v0.9.4**: `eho entrenar` guardaba un archivo de despliegue y `eho incorporar` se negaba a continuar | Lectura de `__main__.py` tras el cambio de formatos | Corregido: se guarda para reanudar por defecto, `--despliegue` para exportar; `test_cli_train_then_incorporate` |
| P2 | Una red sin nodos internos hacía fallar el cálculo de residuos | Al probar el caso del evaluador | Corregido |
| P3 | Los archivos de datos (.pt) se cargaban con pickle sin necesidad | Búsqueda de `weights_only=False` | Corregido en 6 de 7 sitios; queda el archivo de modelo |
| P4 | `eho evaluar` solo mostraba el promedio de errores de los agentes | Lectura de la línea de comandos | Corregido: muestra también el colectivo |

Comprobaciones propias que salieron bien (script de auditoría, no en la batería):
- Entrenar dos veces con la misma semilla da predicciones idénticas (diferencia 0.0).
- Predecir por lotes o de uno en uno da el mismo escalar (1e-7) y el mismo apoyo.
- La inferencia no modifica la memoria; las etiquetas de la consulta no influyen en la predicción.
- `incorporar` funciona con un conjunto por simulador y después de `compact()`.
- El error de inspección se calcula igual en calibración y en uso cuando el contexto tiene k muestras.

## Errores y debilidades que persisten

1. **Formato de modelo inseguro** (pickle). Requiere separar pesos (`state_dict`/safetensors), configuración (JSON) y estado de calibración. Es el pendiente de seguridad más importante.
2. **Un solo umbral LTT por modelo.** Fuera del protocolo calibrado no se certifica nada; faltan certificados por número de mediciones.
3. **Identidad de los datos por contenido, no por procedencia.** Falta un manifiesto (paciente, simulador, versión, parámetros, semilla, costo).
4. **Costo en ejecuciones y en una cifra dada por el usuario**; no se miden horas de GPU.
5. **Red microvascular:** no detecta componentes desconectados; la separación de fases en nodos de frontera con paso de flujo es "bien mezclada" (una hipótesis, no una ley); el acoplamiento con cizalla local no se usa aún en el experimento.
6. **Garantías matemáticas acotadas:** energía de Hopfield sin cota para el paso discreto; Lyapunov de la ecología solo con parámetros congelados; la regla de parada usa el error antes de aprender.
7. **Estadística:** 1-3 semillas; sin pruebas de no inferioridad; sin CUSUM/GLR como líneas base del centinela.
8. **Todo sigue siendo sintético**, y casi todos los resultados previos a la v0.9.4 hay que regenerarlos.

## Cómo mejorarlo (orden sugerido)

1. Congelar la arquitectura: núcleo + memoria episódica + ensamble con abstención + centinela + regla de parada. Ecología, memoria plasmodial y colmena predictiva, como ablaciones.
2. Formato de modelo seguro y manifiesto de datos (pendientes 1 y 3) antes de tocar datos clínicos.
3. Certificados LTT por protocolo (pendiente 2).
4. Regenerar resultados en el clúster con 10 semillas y margen de no inferioridad fijado de antemano.
5. Datos DPD reales y la figura hemorreológica con cizalla local acoplada.
