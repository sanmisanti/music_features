# CLUSTERING - Modulo de Clustering Musical y Semantico

## Resumen Ejecutivo

Este modulo contiene los algoritmos de clustering para el sistema multimodal de recomendacion musical. Implementa dos subsistemas complementarios: clustering de caracteristicas musicales (12D Spotify) y clustering semantico de letras (384D BERT).

**Estado**: PRODUCCION - Sistema validado con +86.1% mejora Silhouette Score

---

## Arquitectura Actual

```
clustering/
├── algorithms/
│   ├── lyrics/              # Sistema BERT semantico (ACTIVO)
│   │   ├── vectorization/   # BertVectorizer, batch processing
│   │   ├── clustering/      # SemanticKMeans, HierarchicalClusterer
│   │   ├── preprocessing/   # TextCleaner, MultilingualNormalizer
│   │   ├── evaluation/      # ClusterEvaluator, metricas
│   │   ├── recommendation/  # Sistema recomendacion semantica
│   │   ├── config/          # Configuracion centralizada
│   │   ├── cache/           # Sistema cache multi-nivel
│   │   ├── models/          # Cache BERT embeddings
│   │   ├── scripts/         # Scripts de ejecucion
│   │   ├── tests/           # Suite de testing
│   │   ├── utils/           # Utilidades
│   │   └── docs/            # Documentacion tecnica
│   └── musical/             # Sistema musical optimizado (ACTIVO)
│       ├── __init__.py
│       └── clustering_optimized.py
├── models/
│   └── musical_models/      # Estructura para modelos
├── scripts/
│   └── analyze_lyrics_dataset.py  # Analisis EDA
├── CLAUDE.md                # Este archivo
└── requirements.txt
```

---

## Componentes Activos

### 1. Clustering Semantico (algorithms/lyrics/)

Sistema de clustering basado en embeddings BERT para letras musicales multilingues.

| Componente | Descripcion |
|------------|-------------|
| **Modelo BERT** | `paraphrase-multilingual-MiniLM-L12-v2` (384D) |
| **Idiomas** | Ingles 84.4%, Espanol 7.5%, Aleman 1.3%, Portugues 1.0% |
| **Cache** | Multi-nivel (RAM + Disco + Database) |
| **Performance** | >4 canciones/segundo |

**Documentacion detallada**: `algorithms/lyrics/README.md`

### 2. Clustering Musical (algorithms/musical/)

Sistema optimizado de clustering K-Means para caracteristicas Spotify.

| Componente | Descripcion |
|------------|-------------|
| **Algoritmo** | K-Means optimizado con MiniBatch |
| **Features** | 12 caracteristicas Spotify normalizadas |
| **Metricas** | Silhouette, Calinski-Harabasz, Davies-Bouldin |

**Script principal**: `clustering_optimized.py`

---

## Sistema de Produccion

El sistema de clustering de produccion NO reside en este modulo. Los scripts activos son:

| Script | Ubicacion | Funcion |
|--------|-----------|---------|
| **cluster_purification.py** | `scripts/` | Sistema Hybrid Purification (+86.1%) |
| **run_final_clustering.py** | `scripts/` | Orquestador de clustering final |
| **optimized_music_recommender.py** | `archive/legacy_recommender/` | Recomendador hibrido |

### Metricas de Produccion

| Metrica | Valor |
|---------|-------|
| Silhouette Score | 0.2893 (+86.1% vs baseline 0.1554) |
| Algoritmo | Hierarchical Clustering, K=3 |
| Dataset | 10,000 canciones optimizadas |
| Performance | 2,209 canciones/segundo |

---

## Datasets Utilizados

| Dataset | Ubicacion | Registros | Uso |
|---------|-----------|-----------|-----|
| Musical optimizado | `data/3_selected/picked_data_optimal.csv` | 10,000 | Clustering musical |
| Embeddings BERT | `data/4_vectorized/embeddings_bert_9753x384.npy` | 9,753 | Clustering semantico |
| Multimodal unificado | `data/5_unified/unified_multimodal_7811.pkl` | 7,811 | Sistema hibrido |

**Documentacion de datos**: `data/CLAUDE.md`

---

## Componentes Legacy

Los siguientes componentes fueron archivados en `scripts/legacy/clustering/`:

| Directorio | Contenido | Razon de Archivado |
|------------|-----------|-------------------|
| `algorithms_legacy/` | clustering.py, clustering_pca.py, etc. | Superados por cluster_purification.py |
| `models_baseline/` | Modelos 0.231-0.314 Silhouette | Superados por Hybrid Purification 0.2893 |
| `recommender_old/` | music_recommender.py | Superado por optimized_music_recommender.py |
| `notebooks/` | cluster.ipynb | Experimental, no produccion |

**NO USAR** estos componentes en codigo nuevo.

---

## Uso Rapido

### Clustering Musical (Produccion)

```bash
# Sistema completo de clustering optimizado
python scripts/run_final_clustering.py
```

### Vectorizacion BERT

```bash
# Generar embeddings de letras
python clustering/algorithms/lyrics/scripts/run_complete_vectorization.py
```

### Analisis de Dataset

```bash
# EDA de dataset musical
python clustering/scripts/analyze_lyrics_dataset.py
```

---

## Dependencias

```
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.0.0
sentence-transformers>=2.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.64.0
joblib>=1.1.0
```

---

## Referencias

- **data/CLAUDE.md** - Documentacion de datasets
- **data_selection/CLAUDE.md** - Pipeline de seleccion de datos
- **algorithms/lyrics/README.md** - Sistema BERT semantico
- **algorithms/lyrics/IMPLEMENTATION_PLAN.md** - Plan de implementacion detallado
- **scripts/legacy/clustering/README.md** - Componentes archivados
