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

**Estructura obligatoria**: Introduccion contextual, desarrollo tecnico con fundamentacion cientifica, analisis comparativo cuando aplique, implicaciones practicas, sintesis y conclusiones.

---

## DOCUMENTACION DEL PROYECTO

### Documento Maestro (OBLIGATORIO LEER PRIMERO)

| Documento | Contenido |
|-----------|-----------|
| **docs/FULL_PROJECT.md** | Documento base para tesis: metodologia cientifica completa, resultados experimentales, contribuciones, 56 configuraciones validadas |

### Documentacion de Arquitectura

| Documento | Contenido |
|-----------|-----------|
| **docs/SEMANTIC_RECOMMENDATIONS_ARCHITECTURE.md** | Decision arquitectural: vectores BERT directos vs clustering |

### Documentacion de Modulos

| Modulo | Archivo | Contenido |
|--------|---------|-----------|
| Datos | `data/CLAUDE.md` | Estructura datasets, rutas, formatos, flujo 18,454 -> 10,000 -> 7,811 |
| Seleccion | `data_selection/CLAUDE.md` | Pipeline clustering-aware, Hopkins validation |
| Clustering | `clustering/CLAUDE.md` | Arquitectura musical/semantico, componentes activos |
| Exploratorio | `exploratory_analysis/CLAUDE.md` | Capacidades, tests 82/82 |
| FASE 2 | `clustering/evaluation_project/phase2_clustering_readiness/` | Hopkins comparativo post-unificacion (Dic 2025) |
| FASE 3 | `clustering/evaluation_project/phase3_multimodal_clustering/README.md` | Clustering multimodal exhaustivo |

### Documentacion Archivada

Documentos historicos movidos a `archive/docs_legacy/`:
- ANALYSIS_RESULTS.md, DOCS.md, DIRECTIVAS.md
- FASE_4_COMPLETION_REPORT.md, OPTIMIZATION_DOCUMENTATION.md
- PROYECTO_COMPLETO_DOCUMENTACION.md

---

## ESTADO DEL PROYECTO

**Sistema de Clustering Musical Optimizado - COMPLETADO**

- **Resultado Principal**: Silhouette Score 0.1554 -> 0.2893 (+86.1%)
- **Dataset Final**: 7,811 canciones multimodal unificado
- **Metodologia**: Hybrid Purification Strategy

**Hopkins Statistic Post-Unificacion (FASE 2 - Dic 2025)**:
- Semantico (384D): 0.7752 ± 0.0015 (Excellent clustering tendency)
- Musical (12D): 0.7871 ± 0.0022 (Excellent clustering tendency)
- Validacion estadistica: p < 0.001, Cohen's d = 4.02

**Consultar docs/FULL_PROJECT.md para detalles completos.**

---

## COMANDOS PRINCIPALES

```bash
# Clustering completo (8-10 segundos)
python run_final_clustering.py

# Analisis rapido de datasets
python quick_analysis.py --dataset optimal

# Recomendador musical (<100ms)
python run_music_recommender.py

# FASE 3: Evaluacion multimodal
cd clustering_evaluation_project/phase3_multimodal_clustering
python run_multimodal_clustering_evaluation.py --dataset ../phase1_dataset_unification/unified_multimodal_dataset_20250822_004929.pkl --output ./results
```

---

## DATASETS PRINCIPALES

| Dataset | Ubicacion | Registros |
|---------|-----------|-----------|
| Multimodal final | `data/5_unified/` | 7,811 |
| Musical optimizado | `data/3_selected/` | 10,000 |
| Fuente con letras | `data/2_with_lyrics/` | 18,454 |
| Embeddings BERT | `data/4_vectorized/` | 9,753 |

**Formatos**: Ver `data/CLAUDE.md` para separadores especificos por carpeta.

---

## ARQUITECTURA

### Sistemas Activos
- `cluster_purification.py` - Sistema clustering musical (800+ lineas)
- `clustering/algorithms/lyrics/` - Clustering semantico BERT
- `clustering/evaluation_project/` - Evaluacion multimodal (FASE 1, 2, 3)

### Sistemas Legacy
- `scripts/legacy/clustering/` - Algoritmos baseline
- `scripts/legacy/data_selection/` - Pipeline 1.2M legacy
- `archive/legacy_recommender/` - Recomendador anterior

---

## DIRECTIVA: ACTUALIZACION DE DOCUMENTACION

Cada nuevo archivo .md con informacion tecnica DEBE:
1. Agregarse como referencia en este archivo
2. Incluir descripcion breve del contenido
3. Mantener orden logico de importancia

**FULL_PROJECT.md es el documento base para la tesis** - toda decision tecnica, experimento, y resultado debe documentarse alli con rigor academico.
