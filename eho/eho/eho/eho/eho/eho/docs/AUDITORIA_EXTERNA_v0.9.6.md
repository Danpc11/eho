# Respuesta a la cuarta auditoría externa (sobre la v0.9.6)

Fecha: 27 de noviembre de 2026. Versión que responde: v0.9.7. Pruebas nuevas: `tests/test_v097.py` (15).
Objetivo de esta versión: dejar corregidos los errores de método y de código ANTES de lanzar el clúster.

## Errores de código y de método

| # | Hallazgo | Veredicto | Cambio en v0.9.7 | Prueba |
|---|---|---|---|---|
| 3.1 | `evaluar_colectivo()` usaba el certificado por defecto, no el del contexto | **Confirmado** | Una sola función de decisión certificada, `Gemelo._certificar()`, compartida por `predecir()` y `evaluar_colectivo()`; evaluar con un k sin calibración da error | `test_evaluar_uses_certificate_of_its_protocol` |
| 3.2 | Con 5 mediciones, la inspección en despliegue no era la de calibración | **Confirmado** | Una sola función de inspección, `hive.context_inspection()` (leave-one-out con todas las demás), usada en calibración, evaluación y despliegue | `test_same_inspection_in_calibration_and_deployment` (k = 5) |
| 3.3 | El número de mediciones no define el protocolo | **Confirmado**, arreglo parcial | `predecir()` exige consulta y contexto del mismo episodio y sin muestras compartidas. El certificado queda descrito (`protocolo["certificado"]`: selección del contexto, unidad estadística, error, tolerancia, riesgo, δ, distribución) y ligado a la huella SHA-256 del modelo: si los agentes cambian, nada se certifica. Que las mediciones sean de verdad de ese paciente no lo puede comprobar el código | `test_context_must_belong_to_query_patient`, `test_certificate_is_bound_to_model_and_described` |
| 3.4 | La auditoría ignoraba lo incorporado y usaba la validación anterior | **Confirmado** | Audita entrenamiento inicial + todo lo incorporado, y la validación que se pasa | `test_audit_covers_incorporated_data_and_current_validation` |
| 3.5 | Certificados individuales frente a garantía conjunta | **De acuerdo** | `colmena.garantia_conjunta` (por defecto): δ se reparte entre protocolos. Las recalibraciones sucesivas se cuentan (`emision`) y la nota del certificado da la cota conjunta; no hay gasto de α automático entre recalibraciones | misma prueba |
| 4.1 | El manifiesto podía sobrescribir el identificador | **Confirmado** | Los metadatos del usuario van en `metadatos` | `test_manifest_cannot_override_and_costs_validated` |
| 4.2 | Identidad solo por huella de datos | **Confirmado** | `incorporar(ids=[...])`: identificador de ejecución dado por quien corre el simulador; la huella queda como segunda comprobación (los mismos datos con otro id se rechazan) | misma prueba |
| 4.3 | Costos negativos, no finitos o ausentes | **Confirmado** | Costo finito y no negativo; obligatorio si hay presupuesto de costo | misma prueba |
| 4.4 | El presupuesto se comprobaba después de simular | **Confirmado** | `reservar()` / `liberar()`: la reserva descuenta del presupuesto antes de ejecutar; `incorporar()` la consume. Flujo: proponer → reservar → simular → incorporar | `test_reserve_before_running` |
| 5.1 | Con cizalla acoplada solo se comprobaba la convergencia de H | **Confirmado** (10.8 % de error en el caudal) | Convergencia conjunta de H, caudal y viscosidad; `residuo_caudal` y comprobación `caudal_consistente` en `valida`; el caudal usado para la cizalla se amortigua | `test_shear_coupled_convergence_checks_flow` |
| 5.2 | Unidades de la cizalla sin definir | **Confirmado**, arreglo parcial | Unidades internas documentadas y `shear_scale` (= 1 / μ_plasma) para obtener s⁻¹. **Pendiente:** todo el solver en SI | — |
| 6 | `no_inferioridad` aceptaba vectores sin parear | **Confirmado** | Diccionarios por semilla o vectores de igual longitud; valores finitos; margen positivo. Intervalo t pareado por defecto (el bootstrap percentil subcubre con pocas semillas). El texto ya no dice que demuestre equivalencia | `test_non_inferiority_requires_real_pairs` |
| 7.1 | La carga con pickle estaba activa por defecto | **Confirmado** | `permitir_pickle=False` por defecto; en la línea de comandos, `--permitir-pickle` | `test_pickle_disabled_by_default` |
| 7.2 | Carga segura no es privacidad | **De acuerdo** | Documentado: las memorias y el registro pueden derivar de datos de pacientes; la privacidad clínica no está validada | — |
| 7.3 | El certificado no estaba ligado al archivo | **Confirmado** | Huella de integridad en el archivo; al cargar se recalcula y, si no coincide, error. Detecta corrupción, no falsificación (haría falta firma digital) | `test_export_integrity_hash` |
| 8.4 | Unidad estadística del riesgo | **De acuerdo** | El certificado declara su unidad (una consulta por paciente). `exp_abstencion.py` reporta además riesgo medio por paciente y pacientes con algún fallo, sin certificar | — |
| 8.5 | Comparaciones múltiples entre señales y riesgos | **De acuerdo** | La señal se elige solo con la referencia (`senal_elegida_en_referencia`) y se congela; las demás son comparación | — |
| 8.6 | La rejilla LTT devolvía vacío si la referencia no sugería nada | **Confirmado** | Vuelve a la rejilla ciega en vez de abstenerse; el margen es un parámetro. Se retira la frase "es solo un límite de los datos": también dependen la potencia, el tamaño de calibración y la búsqueda | `test_ltt_grid_falls_back_instead_of_abstaining` |
| 9.1 | El centinela mezclaba datos ausentes con anómalos | **Confirmado** | Pacientes sin mediciones válidas se excluyen de la calibración; tres estados: compatible, fuera de dominio, evidencia insuficiente | `test_sentinel_insufficient_evidence_and_equal_false_alarm` |
| 9.2 | CUSUM debe compararse a igual tasa de falsas alarmas | **De acuerdo** | `Centinela.threshold_at_false_alarm()`; el docstring exige un orden con sentido físico | misma prueba |
| 10 | La prueba de energía no comprobaba que la energía bajara | **Confirmado** | Modo `armijo` en el núcleo (búsqueda por retroceso por muestra): E_{t+1} ≤ E_t por construcción. La prueba comprueba descenso; con un paso absurdo (α = 20) el paso fijo viola en ~25 % de los pasos y Armijo en 0 % | `test_armijo_guarantees_energy_descent` |
| 11 | Viscosidad negativa aceptada y marcada válida | **Confirmado** | Viscosidad no positiva o no finita: error. `valida` exige además disipación Σ Q ΔP ≥ 0 | `test_negative_viscosity_rejected_and_dissipation_reported` |

Error propio encontrado al corregir: al unificar la inspección, `predict_with_inspection` perdió su
`no_grad` y dejaba tensores con grafo en el predictor (fallaba la copia del Gemelo). Corregido antes de empaquetar.

## Decisión sobre los mecanismos (de acuerdo con la tabla del evaluador)

| Componente | Estado en el código |
|---|---|
| Operador informado por física, núcleo de Hopfield, memoria episódica, memoria de largo plazo, ensamble | Se conservan |
| Colmena para predecir | Retirada como valor por defecto desde v0.9.4 (`colmena.prediccion: media`); queda como opción |
| Colmena para abstenerse | Comparación experimental (`exp_abstencion.py`); no es la mejor señal |
| Colmena para adquirir, regla de parada, centinela, LTT | Se conservan; pendientes de validar con 10 semillas |
| Ecología, memoria plasmodial | Se conservan como ablaciones (`modelo.ecologia`, `modelo.memoria`) |

## Desarrollo matemático propuesto: qué se adoptó y qué queda como hoja de ruta

Adoptado ya: descenso energético discreto garantizado (Armijo, su Proposición 1 en la práctica); contrato
físico de la red (viscosidad positiva, disipación no negativa, residuos completos); certificados con
unidad estadística y δ conjunta; tamaño mínimo de calibración m_min = ⌈log δ / log(1 − r)⌉ (ya en `ltt_grid`).

Hoja de ruta (son funciones nuevas, no correcciones; conviene decidirlas antes de congelar la arquitectura):
1. Memoria bayesiana de bajo rango (actualización tipo Kalman del estado latente del paciente, con Σ_k ⪯ Σ_{k−1}).
2. Adquisición orientada a un objetivo físico, Δ_G(x) / C(x), y multifidelidad (qué simulador usar).
3. Regla de parada con ganancia medida (riesgo antes y después en una referencia independiente).
4. Potencial constitutivo con contexto independiente de la cizalla (su Proposición 4) y viscosidad derivada del mismo potencial.
5. Red como problema variacional y sensibilidades por diferenciación implícita o adjunto.
6. Proyección centrada P en el paso de Hopfield y cota de paso; estabilidad del sistema acoplado.

## Antes de lanzar el clúster

- [x] Una sola función de inferencia certificada y una sola de inspección.
- [x] Auditoría de particiones con lo incorporado; semillas en rangos separados.
- [x] Certificados descritos, ligados al modelo y con δ conjunta.
- [x] Presupuesto con reserva previa, costos validados e identificadores de ejecución.
- [x] Red con convergencia completa y contrato físico.
- [x] Prueba de no inferioridad con emparejamiento real.
- [ ] Fijar de antemano: margen de no inferioridad, riesgo y tolerancia por dominio, y la señal de abstención.
- [ ] Decidir si la hoja de ruta 1-3 entra antes de las 10 semillas (cambiaría los resultados).
