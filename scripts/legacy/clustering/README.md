# Clustering - Componentes Legacy

**Fecha de archivado**: 2025-12

## Razon de Archivado

Estos componentes fueron desarrollados en fases tempranas del proyecto y fueron **superados** por el sistema Hybrid Purification que logro +86.1% mejora en Silhouette Score (0.1554 -> 0.2893).

## Contenido Archivado

### algorithms_legacy/

Scripts de clustering originales superados por `scripts/cluster_purification.py`:

| Archivo | Proposito Original | Silhouette Logrado |
|---------|-------------------|-------------------|
| clustering.py | K-Means basico educativo | ~0.23 |
| clustering_pca.py | K-Means con PCA | 0.251-0.314 |
| clustering_comparative.py | Analisis comparativo | N/A (herramienta) |
| run_fase2_complete.py | Orquestador FASE 2 | N/A (script) |
| test_clustering_comparative.py | Tests comparativos | N/A (tests) |

### models_baseline/

Modelos entrenados con metodologia anterior (2025-01-28):

| Modelo | Silhouette | Metodo |
|--------|------------|--------|
| method1_pca5 | 0.314 | PCA 5 componentes |
| method2_pca8 | 0.251 | PCA 8 componentes |
| method3_optimized | 0.231 | K-Means sin PCA |

**Nota**: Estos modelos usaban `picked_data_0.csv` (9,677 canciones), NO el dataset optimizado actual.

### recommender_old/

Sistema de recomendacion original basado en modelos baseline:

| Archivo | Proposito |
|---------|-----------|
| music_recommender.py | Recomendador estandar |
| music_recommender_full.py | Recomendador dataset 1.2M |

**Superado por**: `archive/legacy_recommender/optimized_music_recommender.py`

### notebooks/

| Archivo | Proposito |
|---------|-----------|
| cluster.ipynb | Exploracion inicial de clustering |

## Sistema Actual

El sistema de clustering activo esta en:

```
scripts/
├── cluster_purification.py    # Sistema Hybrid Purification (+86.1%)
└── run_final_clustering.py    # Orquestador principal

clustering/
├── algorithms/lyrics/         # Sistema BERT semantico
└── algorithms/musical/        # clustering_optimized.py
```

## NO USAR

Estos archivos son **solo referencia historica**. Para clustering de produccion:

```bash
python scripts/run_final_clustering.py
```

## Metricas Comparativas

| Sistema | Silhouette | Estado |
|---------|------------|--------|
| **Hybrid Purification** | **0.2893** | PRODUCCION |
| PCA 5 baseline | 0.314 | LEGACY |
| PCA 8 baseline | 0.251 | LEGACY |
| Optimized baseline | 0.231 | LEGACY |

**Nota**: El Silhouette 0.314 de PCA5 fue sobre un dataset diferente (9,677 canciones sin letras). El sistema actual opera sobre 10,000 canciones optimizadas con Hopkins 0.823.
