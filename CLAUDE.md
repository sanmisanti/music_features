# CLAUDE.md

Archivo de configuracion para Claude Code en este repositorio.

## DIRECTIVA CRITICA: EJECUCION DE SCRIPTS

**NUNCA ejecutar scripts, comandos o tests directamente.**
- SIEMPRE avisar al usuario antes de ejecutar cualquier comando
- ESPERAR que el usuario ejecute el script y muestre la salida
- DESPUES analizar los resultados y continuar segun corresponda
- Aplica a: python scripts, bash commands, tests, jupyter notebooks

## DIRECTIVA CRITICA: COMUNICACION PROFESIONAL

**NUNCA utilizar emojis, frases cortas o comunicacion informal.** Toda interaccion debe mantener rigor tecnico y profesionalismo de proyecto de tesis de Ingenieria Informatica. Respuestas como comunicacion entre ingenieros informaticos profesionales con explicaciones tecnicas fundamentadas.

---

## ESTADO DEL PROYECTO

**Fase actual: Informe LaTeX completo - pendiente compilacion y pruebas de calidad externas (Abril 2026)**

El proyecto se encuentra en proceso de re-ejecucion desde cero. La primera iteracion (2025) produjo resultados funcionales pero con problemas metodologicos identificados durante evaluacion academica. La re-ejecucion preserva el mismo dataset y objetivos, pero reconstruye codigo y documentacion con rigor mejorado.

### Objetivo de la re-ejecucion

Desarrollar el proyecto paso a paso, produciendo simultaneamente codigo funcional y el informe de tesis en LaTeX. Cada etapa genera artefactos de codigo y contenido de capitulos del informe.

### Progreso

- [x] Etapa 1: Fundacion (estructura repositorio, esqueleto LaTeX 7 capitulos + 3 anexos, config centralizada, APA 7 via biblatex+biber)
- [x] Investigacion bibliografica (7 documentos en `thesis/investigacion/`, ~147 fuentes, ~316KB)
- [x] Capitulo 2: Estado de la Cuestion (7 secciones completas, ~58 entradas en bibliography.bib)
- [x] Etapa 2: Datos — codigo EDA ejecutado, redaccion LaTeX completada (§5.2.1 Fuentes, §5.2.2 Preprocesamiento)
- [x] Etapa 3: Features — preprocesamiento, vectorizacion E5 (17,964 x 384), normalizacion musical (17,964 x 13, key circular), unificacion NPZ (28.11 MB, 8 arrays), redaccion LaTeX completada (§5.3, §5.4, §5.5)
- [x] Etapa 4: Clustering — codigo completado (Hopkins, multi-algoritmo, purificacion, UMAP 2D), H1-H4 confirmadas. Ver **[`src/clustering/CLAUDE_ETAPA4.md`](src/clustering/CLAUDE_ETAPA4.md)**
- [x] Etapa 5: Recomendacion — sistema hibrido implementado (k-NN + fusion tardia), α óptimo = 0.80, P@10 = 0.4447, cobertura catalogo 95.5%
- [x] Redaccion LaTeX completa — 7 capitulos. §4 Marco Teorico escrito (§4.1, §4.2, §4.4, §4.5 con 8 subsecciones, §4.6); §5.7 Sistema de Recomendacion; §6 Resultados completo; §7 Conclusiones; §1 Introduccion ampliada. Los 3 anexos (A-C) se eliminaron el 2026-06-14 por ser demasiado tecnicos; su contenido util se integro al Cap 6 (ver tabla ESTRUCTURA DEL INFORME)
- [ ] Etapa 6: Sintesis — pendiente compilacion final, pruebas de calidad externas (ver **[`PLAN_FINALIZACION.md`](PLAN_FINALIZACION.md)** y **[`src/recommendation/PRUEBAS_CALIDAD.md`](src/recommendation/PRUEBAS_CALIDAD.md)**)

---

## DECISIONES TECNICAS

### Modelo de embeddings semanticos

**Seleccionado: `multilingual-e5-small`** (Microsoft) en reemplazo de `paraphrase-multilingual-MiniLM-L12-v2`.

| Propiedad | MiniLM (v1) | E5-small (v2) |
|-----------|-------------|---------------|
| Dimensiones | 384 | 384 |
| Ventana de tokens | 128 | 512 |
| Idiomas | 50+ | 100+ |
| Cobertura de letras | ~50% sin truncamiento | **53.5% real** (tokenizer E5) |

Justificacion: misma dimensionalidad (metricas de clustering comparables), 4x mas contexto, soporte multilingue superior. **Actualizado en `src/config.py`** (BERT_MODEL_NAME y BERT_TARGET_TOKENS).

**Cobertura validada con tokenizer real (Etapa 3)**: 53.48% (9,607 / 17,964 canciones dentro de ventana 512). La heuristica del EDA (1.33 tok/palabra) estimaba 58.8%, resultando optimista. El subword tokenization multilingue genera mas tokens por palabra de lo estimado. Distribucion: media=607, mediana=486, DE=433, p90=1115, p95=1451, p99=2324.

### Estrategia de chunking para letras largas

**Decision**: Implementar chunking con agregacion por promedio para eliminar perdida de informacion por truncamiento.

El 46.5% de canciones excede la ventana de 512 tokens. Truncar introduce sesgo sistematico: canciones cortas tienen representacion completa mientras que canciones largas tienen representacion parcial. Dos canciones con contenido tematico identico pero diferente longitud producirian embeddings distintos por un artefacto del pipeline, no por diferencia semantica real.

Parametros de chunking:
- **Tamano de chunk**: 450 tokens de contenido (overhead real: 5 tokens = 3 prefijo + 2 especiales)
- **Overlap**: 50 tokens entre chunks consecutivos, stride=400 (preserva continuidad semantica en fronteras)
- **Agregacion**: promedio simple de embeddings de chunks + re-normalizacion L2
- **Canciones cortas (<=507 tokens contenido)**: pasan directamente sin chunking
- **Output final**: [17964, 384] float32 normalizado L2 — mismo formato, cobertura 100%

Resultados de chunking (ejecucion en curso):
- 9,583 canciones sin chunking (53.3%) + 8,381 con chunking (46.7%)
- 32,303 chunks totales, promedio 1.80 chunks/cancion (2.71 chunks/cancion chunkeada)

### Estilo de redaccion del informe

Principios establecidos durante la redaccion del Capitulo 2:
- Lenguaje accesible: explicar TODOS los terminos tecnicos cuando se introducen
- No hacer dumps de terminologia (no listar conceptos sin explicar)
- Explicar la metodologia detras de cada resultado citado (como midieron, que midieron)
- Estado de la Cuestion = QUE y POR QUE (accesible); Marco Teorico = COMO (tecnico con formulas)
- `\textcite` para autor como sujeto, `\parencite` para parentetico
- No referenciar "la primera ejecucion" en el informe; presentar situaciones de forma general
- No descartar opciones prematuramente; presentar el panorama
- **Tono y posicionamiento**: el proyecto es una tesis para CONSTRUIR un recomendador, NO un paper con contribuciones metodologicas. No reclamar formalizacion de procedimientos, vacios del campo como espacios a explotar, ni estandares de reproducibilidad para que otros investigadores verifiquen. Limitar la descripcion a lo que el sistema SI hace; no enumerar lo que NO hace.
- **Rol del clustering en el informe**: herramienta analitica/validacion (Hopkins de estructura, NMI de complementariedad cross-modal). NO es mecanismo de recomendacion. La recomendacion opera sobre similitudes k-NN.
- **Division Estado de la Cuestion vs Solucion Propuesta**: el Estado de la Cuestion describe el panorama y trade-offs en terminos generales del campo. Las decisiones especificas del proyecto (con numeros concretos: 384+13=397, α=0.80, etc.) van en Solucion Propuesta, no en el Estado de la Cuestion.
- **Sin guiones largos** (`---`) en el texto del informe. Reemplazar por comas, paréntesis o dos puntos según corresponda.
- **Capitulo 3 con tono de planificacion, no retrospectivo**: el Capitulo 3 (Definicion del Problema) describe el proyecto en perspectiva de planificacion. Riesgos como "mitigacion prevista" (no "adoptada"), cronograma como distribucion planificada del equipo, sin parrafos que narren si los riesgos se materializaron o si las heuristicas resultaron acertadas. El analisis a posteriori va en §6 (Resultados) y §7 (Conclusiones).
- **Estudio economico**: se adopta la escala de honorarios COPAIPA (Salta, enero 2026) sobre un equipo profesional planificado de 4 perfiles (Lider de Proyecto, Analista Funcional, Responsable de Desarrollo, Desarrollador), no sobre la dedicacion real de los coautores. Costo total resultante: $17.274.662 ARS (~USD 12.300). El PDF de referencia esta en `docs/COPAIPA/`.

---

## METODOLOGIA DE RE-EJECUCION

### Principio central: Desarrollo y documentacion simultaneos

Cada etapa del proyecto produce DOS entregables:
1. **Codigo**: Modulos funcionales, validados, reproducibles
2. **LaTeX**: Secciones del informe escritas con los resultados obtenidos

Las tablas y figuras del informe se generan desde codigo. No se editan manualmente.

### Etapas del proyecto

| Etapa | Descripcion | Capitulos LaTeX asociados |
|-------|-------------|---------------------------|
| 1. Fundacion | Estructura repositorio, esqueleto LaTeX, configuracion | Introduccion (borrador) |
| 2. Datos | Carga, exploracion, analisis descriptivo del dataset | Marco Teorico (datos), Solucion Propuesta (fuentes) |
| 3. Features | Vectorizacion E5, features musicales, unificacion | Marco Teorico (NLP/BERT), Solucion Propuesta (vectorizacion) |
| 4. Clustering | Hopkins, evaluacion multi-algoritmo, purificacion — **planificacion en [`src/clustering/CLAUDE_ETAPA4.md`](src/clustering/CLAUDE_ETAPA4.md)** | Marco Teorico (clustering), Solucion Propuesta (clustering), Resultados |
| 5. Recomendacion | Sistema hibrido, optimizacion pesos, evaluacion | Solucion Propuesta (integracion), Resultados |
| 6. Sintesis | Conclusiones, revision coherencia, compilacion final | Conclusiones, Apendices |

---

## LECCIONES APRENDIDAS DE LA PRIMERA EJECUCION

Problemas identificados que la re-ejecucion debe evitar:

### Problemas de datos
- El dataset final de 7,811 canciones fue resultado residual de filtros, no disenado intencionalmente. **Correccion**: Documentar presupuesto de perdida explicito en cada etapa del pipeline.
- El analisis de sesgo (linguistico, por genero, por instrumentalidad) se realizo post-hoc. **Correccion**: Analizar sesgo ANTES de cada filtrado, no despues.
- El 57.7% de perdida de datos no tenia justificacion intencional. **Correccion**: Registrar por que se excluye cada cancion.

### Problemas de codigo
- El performance predictor usaba 20 observaciones sinteticas hardcoded. **Correccion**: Solo usar resultados reales; eliminar datos inventados.
- Dos funciones devolvian valores placeholder (gap analysis, natural structure score). **Correccion**: Toda funcion devuelve resultados reales o se elimina.
- Asignacion de genero incorrecta en script de exportacion. **Correccion**: Validar integridad en cada paso de transformacion.

### Problemas de evaluacion
- Metricas proxy (genero como ground truth) sin declaracion explicita. **Correccion**: Declarar upfront que el ground truth es proxy, no verdad absoluta.
- Pesos de fusion suboptimos (55/45 vs 20/80 optimo) presentados como "decision de diseno". **Correccion**: Si se eligen pesos suboptimos, documentar como limitacion, no como decision positiva.
- Hipotesis formuladas despues de ver resultados (post-hoc). **Correccion**: Pre-registrar hipotesis antes de ejecutar experimentos.

### Problemas de documentacion
- Documentacion escrita retroactivamente, no junto al desarrollo. **Correccion**: Escribir LaTeX en cada etapa como entregable obligatorio.
- Sin controles de reproducibilidad (seeds, versionado de intermedios). **Correccion**: Seeds fijos en configuracion centralizada, resultados versionados.

---

## DATASET

| Dataset | Ubicacion | Registros | Descripcion |
|---------|-----------|-----------|-------------|
| Fuente con letras | `data/2_with_lyrics/spotify_songs_fixed.csv` | 18,454 | Spotify Kaggle + Genius lyrics, separador `@@` |
| Seleccionado | `data/3_selected/selected_dataset.csv` | 17,964 | Post-preprocesamiento, separador `@@`, 40.72 MB |

| Embeddings semanticos | `data/4_vectorized/embeddings.npy` | 17,964 x 384 | float32, normalizado L2, 26.31 MB |
| Track IDs vectorizados | `data/4_vectorized/track_ids.npy` | 17,964 | Alineado con embeddings |
| Token counts | `data/4_vectorized/token_counts.npy` | 17,964 | Conteo de tokens por cancion (con prefijo E5) |
| Features musicales norm. | `data/4_vectorized/musical_features.npy` | 17,964 x 13 | float32, circular key + z-score |
| Nombres de features | `data/4_vectorized/feature_names.npy` | 13 | Nombres de las 13 features musicales |
| **Dataset unificado** | `data/5_unified/unified_dataset.npz` | 17,964 | 8 arrays: embeddings 384D + musical 13D + metadatos |

---

## MATERIAL DE REFERENCIA (PRIMERA EJECUCION)

El codigo y documentacion de la primera ejecucion permanecen en el repositorio como referencia. No se reutilizan directamente; se reimplementan con las correcciones indicadas.

### Documentacion de referencia

| Documento | Contenido |
|-----------|-----------|
| `docs/FULL_PROJECT.md` | Documento maestro v1: metodologia, resultados, 56 configuraciones |
| `docs/SOLUCION_PROPUESTA_SEMANTICO.md` | Componente semantico completo |
| `docs/SEMANTIC_RECOMMENDATIONS_ARCHITECTURE.md` | Decision vectores BERT directos vs clustering |

### Codigo de referencia

| Modulo | Ubicacion |
|--------|-----------|
| Seleccion clustering-aware | `data_selection/clustering_aware/select_optimal_10k_from_18k.py` |
| Vectorizacion BERT | `clustering/algorithms/lyrics/vectorization/bert_vectorizer.py` |
| Funcion objetivo multi-criterio | `clustering/evaluation_project/phase3_.../config/evaluation_metrics.py` |
| Purificacion hibrida | `scripts/cluster_purification.py` |
| Sistema recomendacion | `recommendation_system/scripts/` |

### Resultados de referencia (benchmark v1) y resultados v2

| Metrica | Valor v1 | Valor v2 | Notas |
|---------|----------|----------|-------|
| Hopkins Semantico (384D) | 0.7752 +/- 0.0015 | **0.9472 +/- 0.0028** | +0.172, E5 + chunking + dataset mayor |
| Hopkins Musical (12/13D) | 0.7871 +/- 0.0022 | **0.8298 +/- 0.0103** | +0.043, dataset mayor + key circular |
| Silhouette mejor semantico | ~0.06-0.11 | **0.9137 (UMAP 30D)** | HDBSCAN k=3, transformacion radical con UMAP |
| Silhouette post-purif. musical | 0.2893 | **0.1351** | K-Means k=5, no comparable (dataset y features distintos) |
| NMI cross-modal | 0.0567 | **0.0298** | Entre UMAP k=3 y musical K-Means k=5 (cobertura total); antes 0.0096 con 384D k=5 vs HDBSCAN k=2 (84% ruido) |
| Precision@10 hibrido | 0.398 | Pendiente (Etapa 5) | Benchmark de recomendacion |

---

## ESTRUCTURA DEL INFORME

Formato: APA 7ma Edicion, compilacion con biblatex + biber.

| Archivo | Capitulo | Estado |
|---------|----------|--------|
| `thesis/main.tex` | Documento maestro | Completo (incluye `pdflscape` para Gantt apaisado) |
| `thesis/chapters/01_introduccion.tex` | Cap 1: Introduccion | Ampliado (§1.1-§1.4) |
| `thesis/chapters/03_estado_cuestion.tex` | Cap 2: Estado de la Cuestion | **Completo** (revisado 2026-05-13) |
| `thesis/chapters/04_definicion_problema.tex` | Cap 3: Definicion del Problema | **Completo** (rediseno cronograma + estudio economico COPAIPA + riesgos 2026-05-20) |
| `thesis/chapters/02_marco_teorico.tex` | Cap 4: Marco Teorico | Completo (§4.1-§4.6, 8 subsecciones en §4.5) |
| `thesis/chapters/05_solucion_propuesta.tex` | Cap 5: Solucion Propuesta | Completo (§5.1-§5.7) |
| `thesis/chapters/06_resultados.tex` | Cap 6: Resultados Experimentales | Completo (§6.1 Clustering, §6.2 Recomendacion, §6.3 Costo Computacional, §6.4 Comparativo, §6.5 Discusion, §6.6 Plan de Pruebas) |
| `thesis/chapters/07_conclusiones.tex` | Cap 7: Conclusiones y Futuras Lineas | Completo (§7.1-§7.4) |
| `thesis/bibliography.bib` | Referencias (~65 entradas) | En progreso |

**Anexos eliminados (2026-06-14)**: los tres anexos (A Configuraciones, B Metricas, C Reproducibilidad) se eliminaron por ser excesivamente tecnicos para un proyecto de grado. Su contenido util se integro al Cap 6: las 3 tablas exhaustivas de clustering a §6.1, la cobertura por genero a §6.2, y los tiempos de ejecucion a §6.3 (Costo Computacional). La distribucion de clusters en recomendaciones, el volcado de `config.py` (Anexo A) y la guia de reproducibilidad (Anexo C) se descartaron; el detalle reproducible permanece en el codigo y los JSON de `results/metrics/`.

Las dos tablas integradas se generan desde codigo (no se editan a mano):
- **Cobertura por genero** (`tab:coverage_per_genre`): `generate_genre_coverage_table()` en `src/recommendation/tables.py`, invocada desde `run_recommendation.py`. Lee `genre_coverage` de las metricas de Etapa 5.
- **Tiempos de ejecucion** (`tab:tiempos_ejecucion`): `generate_timing_table()` en **`src/reporting/timing.py`** (`python -m src.reporting.timing`). Agrega el tiempo total de cada etapa leyendo `results/metrics/`. Requiere instrumentacion: cada `run_*.py` ahora registra `total_elapsed_seconds` de nivel superior en su JSON. Hasta que cada etapa se re-ejecute con esta instrumentacion, el generador usa valores de respaldo de logs de la ejecucion previa (solo vectorizacion y recomendacion ya tienen su tiempo persistido en JSON). Re-ejecutar las etapas baratas (segundos) y clustering (~40 min) hace que la tabla provenga unicamente de los JSON; la vectorizacion (2h) no necesita repetirse.

**Nota**: Los nombres de archivo (01\_, 02\_, etc.) no corresponden al orden de capitulos. El orden real esta definido por la secuencia de `\input` en `main.tex`.

### Investigacion bibliografica

7 documentos de revision sistematica en `thesis/investigacion/`:

| Archivo | Dominio | Fuentes |
|---------|---------|---------|
| `01_sistemas_recomendacion.md` | Recomendacion musical | 23 |
| `02_music_information_retrieval.md` | MIR y letras | 21 |
| `03_nlp_bert_embeddings.md` | NLP, BERT, embeddings | 18 |
| `04_clustering_algoritmos.md` | Clustering y evaluacion | 25 |
| `05_fusion_multimodal.md` | Fusion multimodal | 18 |
| `06_evaluacion_experimental.md` | Evaluacion y reproducibilidad | 20 |
| `07_ingenieria_datos_features.md` | Ingenieria de datos y features | 22 |

### Configuracion centralizada

`src/config.py` define todos los paths, seeds, y parametros globales. Ningun script debe definir estos valores localmente.

### Modulos Etapa 2: EDA

| Modulo | Ubicacion | Contenido |
|--------|-----------|-----------|
| Carga y validacion | `src/data/loader.py` | `load_source_dataset()`, `LoadReport`, validacion de features y letras |
| Analisis exploratorio | `src/data/eda.py` | Perfiles estadisticos, correlaciones, analisis de sesgo, data loss budget |
| Visualizacion | `src/data/plots.py` | 7 figuras PDF (distribuciones, correlaciones, sesgo) |
| Tablas LaTeX | `src/data/tables.py` | 5 tablas booktabs (descriptivas, genero, letras, correlaciones, overview) |
| Orquestador | `src/data/run_eda.py` | `python -m src.data.run_eda` — ejecuta pipeline completo |
| API publica | `src/data/__init__.py` | Exports del modulo |

Artefactos generados (ejecucion exitosa 2026-03-01, 12.2s):
- 7 PDFs en `results/figures/` y `thesis/figures/`
- 5 `.tex` en `results/tables/` y `thesis/tables/`
- `results/metrics/etapa2_eda.json`

### Resultados clave del EDA

| Metrica | Valor | Nota |
|---------|-------|------|
| Filas / columnas | 18,454 / 25 | 61.31 MB |
| Letras validas | 18,202 (98.6%) | 252 nulas |
| Loudness fuera de rango | 4 | Valores > 0 dB |
| Correlaciones |r| >= 0.5 | 2 pares | energy-loudness (0.67), energy-acousticness (-0.55) |
| Balance de genero (entropia) | 0.9842 (alto) | 6 generos |
| Idioma dominante | ingles 83.5% | 35 idiomas, entropia 0.21 |
| Vocal vs instrumental | 95.5% / 4.5% | Umbral instrumentalness > 0.5 |
| Cobertura ventana 512 tokens | **53.48% real** | Validado con tokenizer E5; heuristica EDA (58.8%) era optimista |
| Feature con mas outliers IQR | instrumentalness (21.1%) | Distribucion concentrada cerca de 0 |

### Modulos Etapa 3 Paso 1: Preprocesamiento

| Modulo | Ubicacion | Contenido |
|--------|-----------|-----------|
| Preprocesamiento | `src/data/preprocessor.py` | `filter_null_lyrics()`, `filter_by_word_count()`, `clip_loudness_anomalies()`, `preprocess_dataset()`, `save_selected_dataset()` |
| Orquestador | `src/data/run_preprocessing.py` | `python -m src.data.run_preprocessing` — ejecuta pipeline completo |

Artefactos generados (ejecucion exitosa 2026-03-01, 4.4s):
- `data/3_selected/selected_dataset.csv` (17,964 filas, 25 columnas)
- `results/tables/data_loss_budget.tex` + `thesis/tables/data_loss_budget.tex`
- `results/metrics/etapa3_preprocessing.json`

### Resultados del preprocesamiento

| Filtro | Eliminados | % | Acumulado |
|--------|-----------|---|-----------|
| Letras nulas/vacias | 252 | 1.37% | 18,202 |
| Letras < 10 palabras | 108 | 0.59% | 18,094 |
| Letras > 2000 palabras | 130 | 0.72% | 17,964 |
| **Total perdida** | **490** | **2.66%** | **17,964** |

Adicionalmente, 4 valores de loudness > 0 dB fueron corregidos (clip a 0 dB, sin remocion de filas).

**Contraste con v1**: La v1 redujo de 18K a 7,811 (57.7% de perdida) mediante seleccion clustering-aware sin justificacion intencional. La v2 aplica unicamente filtros de calidad documentados, con 2.66% de perdida total.

### Modulos Etapa 3 Paso 2: Vectorizacion semantica

| Modulo | Ubicacion | Contenido |
|--------|-----------|-----------|
| Vectorizador | `src/features/vectorizer.py` | `prepare_lyrics_for_encoding()`, `measure_token_coverage()`, `vectorize_lyrics()` (con chunking) |
| Tablas LaTeX | `src/features/tables.py` | `generate_token_coverage_table()` |
| Orquestador | `src/features/run_vectorization.py` | `python -m src.features.run_vectorization` — pipeline 8 pasos |
| API publica | `src/features/__init__.py` | Exports del modulo |

Artefactos generados (ejecucion exitosa 2026-03-02, ~2h05min):
- `data/4_vectorized/embeddings.npy` (17,964 x 384, float32, 26.31 MB)
- `data/4_vectorized/track_ids.npy` (17,964)
- `data/4_vectorized/token_counts.npy` (17,964)
- `results/tables/token_coverage.tex` + `thesis/tables/token_coverage.tex`
- `results/metrics/etapa3_vectorization.json`

### Resultados de la vectorizacion

| Metrica | Valor |
|---------|-------|
| Canciones vectorizadas | 17,964 / 17,964 (0 fallos) |
| Sin chunking | 9,583 (53.3%) |
| Con chunking | 8,381 (46.7%) |
| Chunks totales | 32,303 (1.80 chunks/cancion) |
| Norma L2 media | 1.0000 (std=0.0000) |
| Vectores cero | 0 |
| NaN | 0 |
| Dimensiones muertas | 0 |
| Tiempo encoding | 7,378 s (~2h03min CPU) |

### Modulos Etapa 3 Paso 3: Normalizacion de features musicales

| Modulo | Ubicacion | Contenido |
|--------|-----------|-----------|
| Normalizador | `src/features/normalizer.py` | `NormalizationReport`, `circular_encode_key()`, `extract_musical_features()`, `normalize_features()` (circular key + z-score) |
| Tablas LaTeX | `src/features/tables.py` | `generate_normalization_table()` |
| Orquestador | `src/features/run_normalization.py` | `python -m src.features.run_normalization` — pipeline 7 pasos |
| API publica | `src/features/__init__.py` | Exports actualizados |

Artefactos generados (pendiente re-ejecucion con codificacion circular):
- `data/4_vectorized/musical_features.npy` (17,964 x 13, float32)
- `data/4_vectorized/feature_names.npy` (13 nombres)
- `results/tables/normalization_stats.tex` + `thesis/tables/normalization_stats.tex`
- `results/metrics/etapa3_normalization.json`

### Resultados de la normalizacion

| Metrica | Valor |
|---------|-------|
| Muestras x features | 17,964 x 13 |
| Metodo | codificacion circular de key + z-score |
| NaN antes / despues | 0 / 0 |
| Track IDs alineados con embeddings | 17,964 verificados |

Nota sobre variables categoricas:
- `key` (0-11): codificacion circular sin/cos (key_sin, key_cos) para preservar adyacencia tonal. Reemplaza 1 columna por 2 (12->13 features).
- `mode` (0/1): z-score directo, matematicamente valido para binarias.

### Modulos Etapa 3 Paso 4: Unificacion de features

| Modulo | Ubicacion | Contenido |
|--------|-----------|-----------|
| Unificador | `src/features/unifier.py` | `UnificationReport`, `load_feature_components()`, `load_metadata()`, `build_unified_dataset()` |
| Tablas LaTeX | `src/features/tables.py` | `generate_unification_table()` |
| Orquestador | `src/features/run_unification.py` | `python -m src.features.run_unification` — pipeline 6 pasos |
| API publica | `src/features/__init__.py` | Exports actualizados |

Artefactos generados (ejecucion exitosa 2026-03-02, 0.8s):
- `data/5_unified/unified_dataset.npz` (17,964 muestras, 8 arrays, 28.11 MB)
- `results/tables/unified_summary.tex` + `thesis/tables/unified_summary.tex`
- `results/metrics/etapa3_unification.json`

### Resultados de la unificacion

| Metrica | Valor |
|---------|-------|
| Muestras | 17,964 |
| Embeddings semanticos | 17,964 x 384, float32 |
| Features musicales | 17,964 x 13, float32 |
| Generos unicos | 6 (edm: 1,853, latin: 2,132, pop: 3,879, r&b: 3,314, rap: 3,279, rock: 3,507) |
| NaN semanticos / musicales | 0 / 0 |
| Norma L2 media | 1.000000 |
| Z-score media abs | 7.35e-08 |
| Track names nulos | 1 (rellenado con string vacio) |
| Arrays en NPZ | 8 |
| Tamano NPZ | 28.11 MB |

Nota: `load_metadata()` lee del dataset fuente original (`2_with_lyrics/`) en lugar del CSV seleccionado (`3_selected/`) para evitar ambiguedad del separador `@@` cuando campos contienen `@`. Un track_name nulo (cancion con nombre `@`) se rellena con string vacio.

### Contenido del NPZ unificado

| Key | Shape | Dtype | Origen |
|-----|-------|-------|--------|
| `semantic_embeddings` | [17964, 384] | float32 | `4_vectorized/embeddings.npy` |
| `musical_features` | [17964, 13] | float32 | `4_vectorized/musical_features.npy` |
| `track_ids` | [17964] | object | `4_vectorized/track_ids.npy` |
| `feature_names` | [13] | U16 | `4_vectorized/feature_names.npy` |
| `genre_labels` | [17964] | object | `2_with_lyrics/` col `playlist_genre` |
| `track_names` | [17964] | object | `2_with_lyrics/` col `track_name` |
| `track_artists` | [17964] | object | `2_with_lyrics/` col `track_artist` |
| `token_counts` | [17964] | int32 | `4_vectorized/token_counts.npy` |

---

## DIRECTIVA: ACTUALIZACION DE ESTE ARCHIVO

Este archivo se actualiza al completar cada etapa:
1. Marcar etapa completada en la seccion "Progreso"
2. Agregar referencias a nuevos modulos/documentos creados
3. Registrar decisiones tecnicas importantes en la seccion correspondiente
4. Mantener actualizada la seccion de lecciones aprendidas si surgen nuevas
