# PLAN_FINALIZACION.md — Plan de finalización del proyecto

Documento maestro de trabajo para cerrar el proyecto de tesis. Integra la redacción LaTeX pendiente, las pruebas de calidad pendientes, y la actualización de documentación del proyecto.

**Última actualización**: 2026-04-20

---

## 1. OBJETIVO

Completar de forma correcta y coherente:

1. Los capítulos LaTeX pendientes del informe (§5.7, §6, §4 parcial, §7, §1 ampliado, 3 anexos)
2. Las pruebas de calidad pendientes que afectan la honestidad del informe (Pruebas A, B del documento `src/recommendation/PRUEBAS_CALIDAD.md`)
3. La actualización de documentación del proyecto (`CLAUDE.md`, memoria)

**Criterio de éxito**: informe compilable end-to-end, con todas las secciones escritas, todas las tablas/figuras referenciadas, todas las citaciones con entradas bib, y con honestidad sobre limitaciones del sistema.

---

## 2. ESTADO ACTUAL DEL INFORME

### 2.1 Capítulos completados

| Archivo | Estado | Notas |
|---------|--------|-------|
| `03_estado_cuestion.tex` | Completo | 7 secciones, ~58 entradas bib citadas |
| `04_definicion_problema.tex` | Completo | §3.1 Planteamiento, §3.2 Propósito, §3.3 Objetivos |
| `05_solucion_propuesta.tex` (§5.1-§5.6) | Completo | Arquitectura, pipeline de datos, vectorización, normalización, unificación, clustering |

### 2.2 Capítulos pendientes

| Archivo | Sección pendiente | Líneas actuales | Dependencias |
|---------|-------------------|-----------------|--------------|
| `05_solucion_propuesta.tex` | §5.7 completa | 0 en §5.7 | Pruebas A+B (baselines) ideales |
| `06_resultados.tex` | Todo (§6.1-§6.6) | 31 | Depende de Pruebas A+B |
| `02_marco_teorico.tex` | §4.1, §4.2, §4.4, §4.5, §4.6 | 523 con esqueletos anotados | Independiente de resultados |
| `07_conclusiones.tex` | §7.1-§7.4 | 16 | Depende de §6 y §4 |
| `01_introduccion.tex` | §1.2-§1.5 | 16 | Depende de §3 ya escrito |
| `A_configuraciones.tex` | Todo | 4 | Independiente (volcado de config.py) |
| `B_metricas.tex` | Todo | 4 | Depende de Pruebas A+B |
| `C_reproducibilidad.tex` | Todo | 4 | Independiente |

### 2.3 Artefactos generados disponibles

**Tablas LaTeX** (21 archivos en `thesis/tables/`):
- Datos (5): dataset_overview, feature_descriptive_stats, lyrics_quality_summary, genre_subgenre_distribution, data_loss_budget
- Features (3): token_coverage, normalization_stats, unified_summary
- Clustering (8): hopkins_results, clustering_best_configs, clustering_comparison_x3, cross_modal_nmi, purification_results, high_correlations
- Recomendación (5): recommendation_fusion_strategy, recommendation_grid_search, recommendation_optimal_metrics, recommendation_precision_per_genre, recommendation_v1_comparison

**Figuras PDF** (23 archivos en `thesis/figures/`):
- EDA (6): correlation_matrix, feature_boxplots, feature_distributions, features_by_genre, genre_distribution, language_distribution, lyrics_length_distribution
- Clustering (10): hopkins_results, silhouette_comparison_x3, cluster_sizes_x3, purification_comparison, umap_clusters_x2, umap_genres_x2
- Recomendación (4): recommendation_alpha_precision, recommendation_precision_genre, recommendation_score_distributions, recommendation_diversity_coverage

**Métricas JSON** (9 archivos en `results/metrics/`): etapa2_eda, etapa3_{preprocessing, vectorization, normalization, unification}, etapa4_{hopkins, clustering, purification}, etapa5_recommendation.

---

## 3. RIESGOS Y RESTRICCIONES

### 3.1 Riesgo: escribir antes de tener datos honestos

Si escribimos §6 (Resultados) y §7 (Conclusiones) **antes** de ejecutar la Prueba B (baselines numéricos), la narrativa va a ser más débil de lo necesario:

- No podemos contextualizar P@10 = 0.4447 contra baselines triviales
- No podemos verificar si el BERT aporta sobre TF-IDF
- El análisis comparativo quedaría incompleto

**Mitigación**: priorizar Pruebas A+B **antes** de escribir §5.7 y §6.

### 3.2 Riesgo: incoherencia entre capítulos

§4 (Marco Teórico) debe explicar **exactamente** los conceptos que se usan en §5 y §6. Si escribimos §4 sin chequear qué usamos, podemos incluir teoría irrelevante u omitir teoría necesaria.

**Mitigación**: antes de escribir cada subsección de §4, verificar en qué sección de §5 o §6 se usa el concepto (esto ya está anotado en los comentarios `% USADO EN:` de `02_marco_teorico.tex`).

### 3.3 Riesgo: bibliografía incompleta

Las nuevas secciones pueden requerir entradas bib no presentes en las 63 actuales. Errores de compilación por citaciones faltantes.

**Mitigación**: en cada sección, después de escribirla, verificar que todas las claves `\parencite`/`\textcite` existen en `bibliography.bib`. Agregar entradas faltantes inmediatamente.

### 3.4 Restricción: comunicación profesional

CLAUDE.md establece: sin emojis, sin comunicación informal, rigor técnico de tesis de Ingeniería Informática. Aplica a LaTeX y a respuestas.

### 3.5 Restricción: no ejecutar scripts directamente

Toda ejecución de código Python se avisa al usuario y se espera la salida antes de continuar.

---

## 4. FASES DEL PLAN

### FASE 0 — Auditoría y preparación (estimado: 1 hora)

**Objetivo**: confirmar estado actual antes de escribir, identificar dependencias.

**Tareas**:

0.1. **Revisar coherencia entre §3 (Definición del Problema) y lo efectivamente implementado**.
   - Leer completo `04_definicion_problema.tex`.
   - Verificar que los objetivos específicos declarados (§3.3) se corresponden con lo que hicimos.
   - Si hay discrepancias (ej: §3 menciona "Spectral clustering" pero nosotros usamos HDBSCAN), decidir si ajustar §3 o dejar constancia en §6.
   - **Entregable**: lista de inconsistencias detectadas (si las hay).

0.2. **Inventariar entradas bib disponibles y posibles faltantes para §4**.
   - `02_marco_teorico.tex` tiene anotaciones `% REFERENCIAS: clave_bib` en cada subsección.
   - Extraer todas esas claves y verificar que existen en `bibliography.bib`.
   - Documentar claves faltantes para agregarlas durante la redacción.
   - **Entregable**: lista de claves bib a agregar.

0.3. **Verificar referencias cruzadas en secciones ya escritas**.
   - `05_solucion_propuesta.tex` §5.6 referencia §sec:clustering-teoria, §subsec:hopkins, etc. Estas labels están en `02_marco_teorico.tex` pero las secciones correspondientes están en esqueleto anotado (sin contenido real). Al compilar, `\ref{}` resuelve el número pero la sección apunta a un esqueleto vacío.
   - **Entregable**: mapa de referencias cruzadas a resolver.

0.4. **Consultar con el usuario la prioridad de Pruebas A y B**.
   - Opción 1: hacer Pruebas A y B antes de escribir §5.7 y §6 (recomendado)
   - Opción 2: escribir §5.7 y §6 ahora con datos actuales, dejar baselines para un anexo posterior
   - **Entregable**: decisión documentada.

---

### FASE 1 — Pruebas de calidad críticas (estimado: 3-4 horas)

Solo si la FASE 0 determina que se hacen antes de escribir. Si se deciden posponer, saltar a FASE 2 y marcar este punto como "pendiente" en el plan final.

**1.1. Prueba A — Ablación visual por α** (1 hora).

Ver `src/recommendation/PRUEBAS_CALIDAD.md` §5.1 para metodología detallada.

**Pasos**:
- Crear script `src/recommendation/ablation_analysis.py` que, dadas 5-6 queries, genere un reporte markdown comparando top 10 bajo α ∈ {0.0, 0.5, 0.8, 1.0}
- Ejecutar sobre queries representativas
- Documentar hallazgos en `PRUEBAS_CALIDAD.md` sección histórico
- **Entregable**: reporte markdown de ablación + actualización de PRUEBAS_CALIDAD.md

**1.2. Prueba B — Baselines numéricos** (2-3 horas).

Ver `src/recommendation/PRUEBAS_CALIDAD.md` §5.2 para metodología detallada.

**Pasos**:
- Crear módulo `src/recommendation/baselines.py` con implementación de:
  - `random_baseline()` — recomendaciones aleatorias
  - `random_same_genre_baseline()` — aleatorio dentro del mismo género
  - `popularity_baseline()` — top-K por track_popularity
  - `tfidf_baseline()` — TF-IDF sobre letras + coseno
- Crear script `src/recommendation/run_baselines.py` que compute P@10 para cada baseline sobre las 17,964 queries
- Generar tabla LaTeX comparativa `recommendation_baselines_comparison.tex`
- Generar figura comparativa `recommendation_baselines_barplot.pdf`
- Actualizar `etapa5_recommendation.json` con resultados de baselines
- **Entregable**: tabla, figura, JSON actualizado, actualización de PRUEBAS_CALIDAD.md

**Validación**: los baselines deben producir números coherentes:
- Random total: ~0.18 (proporcional a distribución de géneros)
- Random-same-genre: ~1.00 por construcción
- Popularity: 0.30-0.40 esperable
- TF-IDF: desconocido (hipótesis central)
- Nuestro sistema: 0.4447

Si algún número es inesperado, auditar la implementación antes de incorporarlo al informe.

---

### FASE 2 — Núcleo del informe (estimado: 8-12 horas)

**2.1. Completar §5.7 Sistema de Recomendación Híbrido** (3-4 horas).

Ubicación: `05_solucion_propuesta.tex` líneas 351-363.

**Estructura a escribir**:

- **§5.7 introducción** (1-2 párrafos): link con la decisión de k-NN de §5.6. Explicar que la sección describe el diseño del sistema.

- **§5.7.1 Estrategia de Fusión** (2-3 páginas):
  - Justificación de fusión tardía sobre temprana
  - Métrica para espacio semántico: similitud coseno (link a §4.5 y §5.3)
  - Métrica para espacio musical: distancia euclídea + conversión con kernel gaussiano
  - Cálculo de σ por mediana de distancias pairwise (heurística estándar)
  - Fórmula de fusión lineal: `score = α · s_sem + (1-α) · s_mus`
  - Cita tabla `recommendation_fusion_strategy.tex`

- **§5.7.2 Optimización de Pesos** (2-3 páginas):
  - Justificación de grid search (no optimización continua)
  - Rango de α: 21 valores de 0.00 a 1.00, paso 0.05
  - Métrica de selección: Precision@10 con género como proxy (re-referenciar §5.6.2 declarando el proxy)
  - Procedimiento: para cada α, evaluar P@K sobre las 17,964 queries
  - Cita tabla `recommendation_grid_search.tex` y figura `recommendation_alpha_precision.pdf`

- **§5.7.3 Evaluación** (nueva subsección, 1-2 páginas):
  - Definición formal de las cuatro métricas usadas:
    - Precision@K
    - Intra-List Diversity (ILD)
    - Cobertura del catálogo
    - Cobertura por género
  - Declarar nuevamente el género como proxy y sus limitaciones

**Entradas bib requeridas**: verificar existencia de:
- Fusión multimodal: Baltrušaitis 2019, Liang 2024, Vogt 1999 (fusión lineal)
- Grid search: Bergstra & Bengio 2012 (opcional)
- Kernel gaussiano / heurística mediana: Garreau, Jitkrittum & Kanagawa 2017

**2.2. Completar §6 Resultados Experimentales** (5-8 horas).

Ubicación: `06_resultados.tex` (actualmente solo títulos).

**Estructura a escribir**:

- **§6.1 Configuración Experimental** (1 página):
  - Hardware/plataforma
  - Seeds fijos (RANDOM_SEED=42, NUMPY_SEED=42, BOOTSTRAP_SEED=123)
  - Software (Python 3.14, sklearn, scipy, transformers, umap-learn, hdbscan)
  - Reproducibilidad: código disponible, parámetros centralizados en `config.py`

- **§6.2 Análisis Descriptivo del Dataset** (2-3 páginas):
  - Referenciar tablas ya incluidas en §5.2 sin duplicar
  - Agregar: tabla `high_correlations.tex` (no usada aún)
  - Figuras EDA: `feature_distributions`, `correlation_matrix`, `genre_distribution`, `language_distribution`
  - Hallazgos clave: 98.6% letras válidas, entropía género 0.98, 83.5% inglés, 4.5% instrumental

- **§6.3 Resultados de Clustering** (4-5 páginas):
  - §6.3.1 Tendencia al Agrupamiento (Hopkins 4 espacios): tabla `hopkins_results.tex`, figura `hopkins_results.pdf`, H1 confirmada
  - §6.3.2 Comparación de Algoritmos: tablas `clustering_comparison_*.tex` y `clustering_best_configs.tex`, figuras `silhouette_comparison_*.pdf` y `cluster_sizes_*.pdf`, H2 confirmada
  - §6.3.3 Complementariedad Cross-modal: tabla `cross_modal_nmi.tex`, H3 confirmada
  - §6.3.4 Efecto de la Purificación: tabla `purification_results.tex`, figura `purification_comparison.pdf`, H4 confirmada
  - §6.3.5 Visualización UMAP 2D: figuras `umap_clusters_*.pdf`, `umap_genres_*.pdf`
  - Síntesis: las cuatro hipótesis pre-registradas confirmadas

- **§6.4 Resultados de Recomendación** (3-4 páginas):
  - §6.4.1 Sigma y Matrices de Similitud: σ = 4.7634, distribuciones de similitudes
  - §6.4.2 Grid Search de α: tabla `recommendation_grid_search.tex`, figura `recommendation_alpha_precision.pdf`, α óptimo = 0.80
  - §6.4.3 Métricas al α Óptimo: tabla `recommendation_optimal_metrics.tex`
  - §6.4.4 Análisis por Género: tabla `recommendation_precision_per_genre.tex`, figura `recommendation_precision_genre.pdf`, varianza entre géneros (rap 0.69 vs edm 0.31)
  - §6.4.5 Diversidad y Cobertura: figura `recommendation_diversity_coverage.pdf`
  - §6.4.6 Distribución de Scores: figura `recommendation_score_distributions.pdf`
  - **Si Fase 1 completada**: §6.4.7 Comparación con Baselines: tabla `recommendation_baselines_comparison.tex`, figura `recommendation_baselines_barplot.pdf`

- **§6.5 Análisis Comparativo con Literatura** (2 páginas):
  - Tabla `recommendation_v1_comparison.tex`
  - Discusión contra trabajos citados en §2 (Hu 2010, Vystrčilová 2020)
  - Ventajas y limitaciones metodológicas de nuestro enfoque

- **§6.6 Discusión** (2-3 páginas):
  - Análisis crítico honesto
  - Relación entre Hopkins alto y Silhouette bajo en 384D (efecto de dimensionalidad)
  - Por qué α=0.80 indica dominancia semántica pero solo-semántico no es suficiente
  - **Si Fase 1 completada**: discusión de aporte del BERT sobre TF-IDF
  - Limitaciones del género como proxy de ground truth
  - Incorporar hallazgos cualitativos de `PRUEBAS_CALIDAD.md` (caso Vuelve como evidencia fuerte, caso duplicados como problema)

**Entradas bib requeridas**: verificar existencia de trabajos comparativos y baselines citados.

---

### FASE 3 — Marco Teórico (estimado: 10-15 horas, se puede hacer en paralelo con FASE 2)

Ubicación: `02_marco_teorico.tex`. Cinco secciones en esqueleto anotado a completar.

**3.1. §4.1 Sistemas de Recomendación Musical** (2-3 páginas).
- §4.1.1 Filtrado Colaborativo y por Contenido
- §4.1.2 Sistemas Híbridos
- §4.1.3 Evaluación de Sistemas de Recomendación

Fuentes del esqueleto anotado. Usar investigacion/01_sistemas_recomendacion.md como referencia.

**3.2. §4.2 NLP aplicado a Letras** (4-5 páginas).
- §4.2.1 Representación Vectorial de Texto
- §4.2.2 Arquitectura Transformer y Mecanismo de Atención
- §4.2.3 BERT y Modelos Bidireccionales
- §4.2.4 Sentence Embeddings: de Sentence-BERT a E5
- §4.2.5 Tokenización Subword

La sección técnica más extensa. Referencias clave: Vaswani 2017 (attention), Devlin 2019 (BERT), Wang 2022 (E5), Reimers 2019 (Sentence-BERT).

**3.3. §4.4 Normalización de Features** (2 páginas).
- §4.4.1 Estandarización Z-Score
- §4.4.2 Normalización L2 en Espacios de Embeddings
- §4.4.3 Variables Categóricas en Espacios Continuos (codificación circular)

**3.4. §4.5 Análisis de Clustering** (6-8 páginas — la más densa).
- §4.5.1 Tendencia al Agrupamiento: Estadístico de Hopkins
- §4.5.2 K-Means y K-Means++
- §4.5.3 Clustering Jerárquico: Método de Ward
- §4.5.4 Clustering Basado en Densidad: HDBSCAN
- §4.5.5 Evaluación Interna de Clustering
- §4.5.6 Maldición de la Dimensionalidad
- §4.5.7 Reducción Dimensional: UMAP
- §4.5.8 Purificación de Clusters

Fuentes clave ya en bib: Arthur 2007, Ward 1963, Murtagh 2014, Campello 2013, McInnes 2018, Allaoui 2020, Rousseeuw 1987, Chicco 2025, Aggarwal 2001, Dinh 2021, Lawson 1990 (Hopkins — agregada en Etapa 4).

**3.5. §4.6 Representaciones Multimodales y Fusión** (2-3 páginas).
- §4.6.1 Taxonomía de Fusión Multimodal
- §4.6.2 Fusión por Combinación Lineal de Scores
- §4.6.3 Complementariedad entre Modalidades (NMI)

**Metodología de escritura**:
- Seguir estrictamente las anotaciones `% CONCEPTO:`, `% USADO EN:`, `% REFERENCIAS:`, `% CONTENIDO:` ya presentes en el esqueleto
- Mantener estilo: explicar conceptos accesiblemente, con fórmulas donde sean centrales
- Cada subsección termina justificando por qué el concepto es relevante para este proyecto (link explícito a §5 o §6)

**Validación**: después de cada sección, verificar que toda clave citada existe en bib, sin excepción.

---

### FASE 4 — Cierre del informe (estimado: 4-6 horas)

**4.1. §7 Conclusiones** (3-4 páginas).

Ubicación: `07_conclusiones.tex`.

- §7.1 Conclusiones: síntesis por etapa, hallazgos clave
- §7.2 Contribuciones del Trabajo:
  - Metodológicas: pre-registro de hipótesis, data loss budget explícito, purificación formalizada, declaración upfront del proxy de género
  - Empíricas: evidencia de complementariedad multimodal (NMI=0.0096), α=0.80 óptimo empírico, mejora sobre v1
- §7.3 Limitaciones (honesto, incorporar hallazgos de PRUEBAS_CALIDAD.md):
  - Evaluación exclusiva con proxy de género
  - Ausencia de evaluadores humanos
  - Duplicados no filtrados del dataset
  - Etiquetas de género inconsistentes
  - Ambigüedad entre similitud sonora vs semántica en ciertos casos
  - Si no se ejecutaron baselines: reconocer la ausencia
- §7.4 Futuras Líneas:
  - Evaluación con usuarios reales
  - Deduplicación automática del dataset
  - Exploración de fusión intermedia (contrastive learning)
  - Contexto temporal/situacional
  - Pruebas C y D del documento PRUEBAS_CALIDAD.md

**4.2. §1 Introducción expandida** (2-3 páginas adicionales).

Ubicación: `01_introduccion.tex`. Agregar al borrador existente (16 líneas actuales):

- §1.2 Planteamiento del Problema (breve, diferir a §3)
- §1.3 Objetivos (general y específicos — coincidir con §3.3)
- §1.4 Alcance y Limitaciones
- §1.5 Estructura del Documento

**4.3. Revisión de §3** (1 hora).

Si la FASE 0 detectó inconsistencias entre §3 y lo implementado, aplicar correcciones mínimas ahora.

---

### FASE 5 — Anexos (estimado: 3-5 horas)

**5.1. Anexo A — Configuraciones Experimentales** (2 páginas).

Ubicación: `thesis/appendices/A_configuraciones.tex`.

Contenido: volcado estructurado de `src/config.py` con explicación breve de cada parámetro:
- Seeds
- Paths
- Modelo E5 y chunking
- Hopkins
- Clustering multi-algoritmo
- UMAP
- Purificación
- Recomendación (α, K, σ)

**5.2. Anexo B — Métricas Detalladas** (3-5 páginas).

Ubicación: `thesis/appendices/B_metricas.tex`.

Contenido: resultados que no caben en §6:
- Tabla completa de las 47+ configuraciones de clustering probadas (desde `etapa4_clustering.json`)
- Tabla completa de los 21 valores de α del grid search (desde `etapa5_recommendation.json`)
- Métricas por cluster (Silhouette per-cluster, tamaños)
- Detalle de los 6 géneros con estadísticas extendidas

Estas tablas se pueden generar programáticamente desde los JSONs.

**5.3. Anexo C — Guía de Reproducibilidad** (2-3 páginas).

Ubicación: `thesis/appendices/C_reproducibilidad.tex`.

Contenido: guía técnica paso a paso:
- Requisitos: Python 3.14, GPU opcional, 8 GB RAM mínimo
- Estructura del repositorio
- Orden de ejecución de scripts:
  1. `python -m src.data.run_eda`
  2. `python -m src.data.run_preprocessing`
  3. `python -m src.features.run_vectorization` (~2h)
  4. `python -m src.features.run_normalization`
  5. `python -m src.features.run_unification`
  6. `python -m src.clustering.run_hopkins`
  7. `python -m src.clustering.run_clustering` (~40min)
  8. `python -m src.clustering.run_purification`
  9. `python -m src.recommendation.run_recommendation` (~2min)
- Tiempos esperados por etapa
- Outputs esperados con tamaños
- Verificaciones post-ejecución

---

### FASE 6 — Revisión de coherencia y compilación final (estimado: 3-4 horas)

**6.1. Compilación completa del informe**.
- Intentar compilar `thesis/main.tex` con pdflatex + biber
- Identificar errores: citaciones faltantes, referencias rotas, warnings
- Corregir iterativamente

**6.2. Revisión de referencias cruzadas**.
- Script simple que busca todas las `\ref{}` y `\cite{}` en los .tex y verifica que apuntan a labels/claves existentes

**6.3. Revisión de coherencia narrativa**.
- Lectura secuencial de los 7 capítulos
- Verificar que los términos técnicos se introducen antes de usarse
- Verificar que las afirmaciones cuantitativas son consistentes entre capítulos (mismo valor de P@10, misma cantidad de canciones, etc.)

**6.4. Revisión de bibliografía**.
- Verificar que no hay entradas bib sin uso
- Verificar que todas las entradas tienen campos mínimos para APA 7

---

### FASE 7 — Actualización de documentación técnica (estimado: 1 hora)

**7.1. `CLAUDE.md`**.
- Marcar Etapa 5 como completada con resultados
- Marcar Etapa 6 según corresponda tras finalización
- Agregar tabla comparativa v1 vs v2 completa
- Link a PRUEBAS_CALIDAD.md

**7.2. `MEMORY.md` y archivos de memoria**.
- Crear `project_etapa5_recommendation.md` con resultados, decisiones, σ, α óptimo, métricas principales
- Actualizar `MEMORY.md` con referencia al nuevo archivo
- Actualizar estado global: "Proyecto completado" o "Informe LaTeX completo"

**7.3. Crear `src/recommendation/CLAUDE_ETAPA5.md`** (opcional, análogo a CLAUDE_ETAPA4.md).
- Planificación retrospectiva de la etapa
- Decisiones técnicas detalladas
- Hipótesis post-hoc (no había pre-registradas para Etapa 5)

---

## 5. ORDEN DE EJECUCIÓN Y DEPENDENCIAS

```
FASE 0 (auditoría)
  ↓
[decisión usuario: hacer pruebas A+B?]
  ↓                                 ↓
FASE 1 (pruebas A+B)              (saltar)
  ↓                                 ↓
  └─────────────────┬───────────────┘
                    ↓
FASE 2 (§5.7 + §6)  ←── puede hacerse en paralelo con ──→  FASE 3 (§4)
  ↓                                                          ↓
  └──────────────────────┬──────────────────────────────────┘
                         ↓
                    FASE 4 (§7 + §1)
                         ↓
                    FASE 5 (anexos)
                         ↓
                    FASE 6 (coherencia + compilación)
                         ↓
                    FASE 7 (docs técnicos)
```

**Paralelización posible**: FASE 2 y FASE 3 son independientes. Si hubiera dos sesiones paralelas (raro en este workflow) se podrían abordar simultáneamente. En práctica, hacemos secuencial.

---

## 6. ESTIMACIÓN TOTAL DE ESFUERZO

| Fase | Estimación | Acumulado |
|------|-----------|-----------|
| FASE 0 | 1h | 1h |
| FASE 1 (opcional) | 3-4h | 4-5h |
| FASE 2 | 8-12h | 12-17h |
| FASE 3 | 10-15h | 22-32h |
| FASE 4 | 4-6h | 26-38h |
| FASE 5 | 3-5h | 29-43h |
| FASE 6 | 3-4h | 32-47h |
| FASE 7 | 1h | 33-48h |

**Rango total**: 33-48 horas de trabajo efectivo. Dependiendo de cadencia de sesiones y disponibilidad.

---

## 7. CRITERIOS DE CALIDAD PARA CADA ENTREGABLE

**Cada sección LaTeX escrita cumple**:
1. Tono técnico-académico (sin emojis, sin informalidad)
2. Todo término técnico introducido se explica
3. Todas las tablas referenciadas existen físicamente en `thesis/tables/`
4. Todas las figuras referenciadas existen físicamente en `thesis/figures/`
5. Todas las citaciones `\parencite`/`\textcite` apuntan a entradas existentes en `bibliography.bib`
6. Todas las `\ref{}` apuntan a labels existentes
7. Consistencia de valores numéricos con los JSON de `results/metrics/`
8. Honestidad sobre limitaciones (no inflar resultados ni omitir problemas)

**Cada prueba de calidad cumple**:
1. Metodología documentada antes de ejecutar
2. Resultados registrados en PRUEBAS_CALIDAD.md
3. Código reproducible en `src/recommendation/`
4. Artefactos (tablas, figuras) generados vía código, no manualmente

---

## 8. INTEGRACIÓN CON DOCUMENTOS EXISTENTES

Este plan complementa y NO reemplaza:

- `CLAUDE.md` — contexto general del proyecto y directivas operativas
- `src/clustering/CLAUDE_ETAPA4.md` — planificación y resultados de la Etapa 4
- `src/recommendation/PRUEBAS_CALIDAD.md` — detalle de pruebas cualitativas ya hechas y pendientes

Relación:
- PRUEBAS_CALIDAD.md es fuente de las Pruebas A-D que se ejecutan en FASE 1 (y potencialmente FASE 4).
- CLAUDE_ETAPA4.md es referencia para redactar §6.3 (resultados de clustering).
- CLAUDE.md se actualiza en FASE 7 con estado final.

---

## 9. PRIMER HITO SUGERIDO: FASE 0 COMPLETA

Antes de seguir, proponer al usuario:

1. Revisar §3 (definición del problema) — ¿consistente con lo implementado?
2. Auditar bib — ¿qué falta?
3. Decidir: ¿se ejecutan las Pruebas A y B antes de escribir §5.7/§6?

Una vez obtenida la decisión sobre el punto 3, se ejecuta FASE 1 o se salta a FASE 2.

---

## 10. DECISIONES TOMADAS (2026-04-20)

1. **FASE 1 pospuesta**: no se ejecutan Pruebas A ni B ahora. Se escribirá §5.7 y §6 con los datos actuales. Las pruebas se incorporarán en una sesión futura y los resultados se reflejarán retroactivamente.
2. **Duplicados del dataset**: se documentan como limitación explícita en §7.3 (Limitaciones) y §6.6 (Discusión). La deduplicación queda como trabajo pendiente, no bloquea la redacción.
3. **Orden FASE 2 antes de FASE 3**: primero núcleo del informe (§5.7 + §6), después marco teórico.

## 11. HISTORIAL DEL PLAN

| Fecha | Acción |
|-------|--------|
| 2026-04-20 | Creación del plan |
| 2026-04-20 | Decisiones tomadas: sin pruebas ahora, duplicados como limitación, FASE 2 primero |
| 2026-04-20 | FASE 0 (auditoría) ejecutada: §3 completa y coherente, 63 entradas bib antes de comenzar, referencias clave verificadas |
| _pospuesta_ | FASE 1 (pruebas A y B) — decisión del usuario |
| 2026-04-20 | FASE 2 completada: §5.7 (3 subsecciones + 4 ecuaciones) y §6 (6 secciones completas con tablas, figuras e interpretación) escritas. Agregada entrada bib `ziegler_2005_topic_diversification`. Agregados paquetes `subcaption` y `threeparttable` a main.tex. Todas las citas y referencias cruzadas verificadas. |
| 2026-04-20 | FASE 3 completada: §4 Marco Teórico redactado (§4.1 Sistemas de Recomendación, §4.2 NLP aplicado a Letras, §4.4 Normalización de Features, §4.5 Análisis de Clustering con 8 subsecciones, §4.6 Representaciones Multimodales y Fusión). Todas las citas verificadas contra bibliography.bib. |
| 2026-04-20 | FASE 4 completada: §7 Conclusiones (4 secciones: síntesis, contribuciones, limitaciones, futuras líneas) y §1 Introducción ampliada (§1.2 Objetivos, §1.3 Alcance y Limitaciones, §1.4 Estructura del Documento). |
| 2026-04-20 | FASE 5 completada: Anexo A (Configuraciones Experimentales), Anexo B (Métricas Detalladas con comparación clustering exhaustiva, cobertura por género, distribución de clusters, benchmarks v1 vs v2, tiempos), Anexo C (Guía de Reproducibilidad). |
| 2026-04-20 | FASE 6 parcialmente completada: verificaciones automáticas de citas bib y referencias cruzadas realizadas durante la escritura. Compilación LaTeX end-to-end pendiente (requiere ejecución manual del usuario). |
| 2026-04-20 | FASE 7 completada: CLAUDE.md actualizado (fase actual, progreso Etapa 5 y LaTeX), MEMORY.md actualizado, creado memory/project_etapa5_recommendation.md con resultados completos y pruebas cualitativas. |
