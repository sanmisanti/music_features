# INFORME TECNICO DEL PROYECTO
## Sistema de Recomendacion Musical Multimodal con Clustering Optimizado

**Fecha de generacion**: Diciembre 2025
**Estado del proyecto**: SISTEMA FUNCIONAL - Componentes principales completados
**Resultado principal**: Silhouette Score 0.1554 → 0.2893 (+86.1%)

---

## 1. RESUMEN EJECUTIVO

Este proyecto implementa un sistema de recomendacion musical multimodal que integra:

- **Clustering musical optimizado** sobre 12 caracteristicas acusticas de Spotify
- **Vectorizacion semantica** de letras mediante embeddings BERT (384 dimensiones)
- **Fusion hibrida ponderada** (55% musical, 45% semantico) validada experimentalmente

La contribucion metodologica principal es la **Hybrid Purification Strategy**, que mejora la calidad de clustering en un 86.1% mediante eliminacion secuencial de boundary points, outliers estadisticos, y seleccion de caracteristicas discriminativas.

---

## 2. ARQUITECTURA DEL SISTEMA

### 2.1 Estructura de Directorios

```
music_features/
├── CLAUDE.md                    # Configuracion Claude Code
├── INFORME.md                   # Este documento
├── README.md                    # Descripcion general del proyecto
│
├── data/                        # Datasets (ver seccion 3)
│   ├── 0_raw/                   # Spotify original 1.2M (legacy)
│   ├── 1_cleaned/               # Testing rapido 500
│   ├── 2_with_lyrics/           # Fuente principal 18,454
│   ├── 3_selected/              # Produccion musical 10,000
│   ├── 4_vectorized/            # Embeddings BERT 9,753
│   ├── 5_unified/               # Multimodal final 7,811
│   └── auxiliary/               # Cache del sistema
│
├── data_selection/              # Pipeline seleccion clustering-aware
│   └── clustering_aware/        # Implementacion activa
│
├── clustering/                  # Algoritmos de clustering
│   ├── algorithms/
│   │   ├── lyrics/              # Sistema BERT semantico
│   │   └── musical/             # Sistema musical optimizado
│   └── evaluation_project/      # Evaluacion multimodal (3 fases)
│
├── exploratory_analysis/        # Analisis exploratorio de datos
│   ├── config/                  # Configuracion centralizada
│   ├── data_loading/            # Carga y validacion
│   ├── statistical_analysis/    # Estadisticas descriptivas
│   ├── visualization/           # Graficos y heatmaps
│   ├── feature_analysis/        # PCA, t-SNE, Hopkins
│   └── reporting/               # Generacion de reportes
│
├── recommendation_system/       # Sistema de recomendacion final
│   ├── data/                    # Vectores y metadatos (~25 MB)
│   ├── clusters/                # Asignaciones K=10, K=6
│   ├── config/                  # Configuracion validada FASE 3
│   ├── scripts/                 # Motor hibrido (4,828 lineas)
│   └── CLAUDE.md                # Documentacion del modulo
│
├── scripts/                     # Scripts ejecutables
│   ├── cluster_purification.py  # Sistema Hybrid Purification
│   ├── run_final_clustering.py  # Orquestador principal
│   ├── analysis/                # Scripts de analisis
│   ├── generation/              # Generacion de datasets
│   ├── visualization/           # Generacion de graficos
│   └── legacy/                  # Scripts obsoletos (NO USAR)
│
├── docs/                        # Documentacion tecnica
│   ├── FULL_PROJECT.md          # Documento base para tesis
│   └── SEMANTIC_RECOMMENDATIONS_ARCHITECTURE.md
│
├── tests/                       # Suite de testing
└── archive/                     # Componentes archivados
```

---

## 3. FLUJO DE DATOS

### 3.1 Pipeline de Transformacion

```
FUENTE ORIGINAL
┌─────────────────────────────────────────────────────────────┐
│  data/2_with_lyrics/spotify_songs_fixed.csv                 │
│  18,454 canciones | Separador: @@ | Hopkins: 0.823          │
│  Origen: Kaggle + Genius.com (letras integradas)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
ETAPA 1: SELECCION CLUSTERING-AWARE
┌─────────────────────────────────────────────────────────────┐
│  Modulo: data_selection/clustering_aware/                   │
│  Script: select_optimal_10k_from_18k.py (878 lineas)        │
│                                                             │
│  Metodologia:                                               │
│  1. Pre-clustering K=2 (vocal 60% / instrumental 40%)       │
│  2. Seleccion proporcional (6,000 + 4,000)                  │
│  3. MaxMin Sampling con KD-Tree (diversidad maxima)         │
│  4. Validacion Hopkins continua (threshold > 0.70)          │
│                                                             │
│  Tiempo: 239.7 segundos                                     │
│  Hopkins resultante: 0.823                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  data/3_selected/picked_data_optimal.csv                    │
│  10,000 canciones | Separador: ^ | 12 audio features        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
ETAPA 2A: CLUSTERING MUSICAL       ETAPA 2B: VECTORIZACION BERT
┌─────────────────────────────┐   ┌─────────────────────────────┐
│ scripts/cluster_purification│   │ clustering/algorithms/lyrics│
│                             │   │                             │
│ Hybrid Purification:        │   │ Modelo BERT:                │
│ 1. Eliminar Si < 0          │   │ paraphrase-multilingual-    │
│ 2. Remover |z| > 2.5        │   │ MiniLM-L12-v2               │
│ 3. Features discriminativas │   │                             │
│                             │   │ Dimensiones: 384            │
│ Resultado:                  │   │ Normalizacion: L2           │
│ Silhouette: 0.2893 (+86.1%) │   │                             │
│ Algoritmo: Hierarchical K=3 │   │ Embeddings validos: 8,567   │
└─────────────────────────────┘   └─────────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
ETAPA 3: UNIFICACION MULTIMODAL
┌─────────────────────────────────────────────────────────────┐
│  clustering/evaluation_project/phase1_dataset_unification/  │
│  Script: create_unified_multimodal_dataset.py               │
│                                                             │
│  Proceso:                                                   │
│  1. Interseccion track_id (musical ∩ semantico)             │
│  2. StandardScaler sobre features musicales                 │
│  3. Embeddings BERT ya L2-normalizados                      │
│  4. Deduplicacion                                           │
│                                                             │
│  Perdida de datos (efecto residual, no intencional):        │
│  10,000 → 9,753 (letras) → 8,567 (BERT) → 7,811 (final)     │
│  Perdida total: 21.9%                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  data/5_unified/unified_multimodal_7811.pkl                 │
│  7,811 canciones                                            │
│  - semantic_embeddings: [7811, 384]                         │
│  - musical_features_normalized: [7811, 12]                  │
│  - track_metadata: DataFrame                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
ETAPA 4: EVALUACION MULTIMODAL (FASE 3)
┌─────────────────────────────────────────────────────────────┐
│  clustering/evaluation_project/phase3_multimodal_clustering │
│                                                             │
│  56 configuraciones evaluadas:                              │
│  - Musical (12D): Hierarchical, K-Means++, GMM, DBSCAN      │
│  - Semantico (384D): Hierarchical, K-Means++, GMM-Tied      │
│                                                             │
│  Funcion objetivo multi-criterio:                           │
│  Score = 0.3×Silhouette + 0.3×Balance +                     │
│          0.2×Interpretabilidad + 0.1×Cross_Modal +          │
│          0.1×Granularidad                                   │
│                                                             │
│  Configuracion optima:                                      │
│  - Musical: K=10, Silhouette=0.0965, Balance=0.7547         │
│  - Semantico: K=6, Silhouette=0.0329, Interp=0.7284         │
│  - Pesos fusion: 55% musical, 45% semantico                 │
│  - NMI cross-modal: 0.0567 (complementariedad confirmada)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
ETAPA 5: SISTEMA DE RECOMENDACION
┌─────────────────────────────────────────────────────────────┐
│  recommendation_system/                                     │
│                                                             │
│  Componentes:                                               │
│  - MusicDataLoader (339 lineas)                             │
│  - HybridMusicRecommender (409 lineas)                      │
│  - RecommendationExplainer (1080+ lineas)                   │
│  - CLI Interface (643 lineas)                               │
│                                                             │
│  Arquitectura de recomendacion:                             │
│  - Semantico: k-NN directo sobre embeddings BERT            │
│  - Musical: Clustering K=10 + similitud intra-cluster       │
│  - Fusion: Ponderacion 55% musical, 45% semantico           │
│                                                             │
│  Performance: <100ms por recomendacion                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Caracteristicas Musicales (12 features)

```python
clustering_features = [
    'danceability',      # Aptitud para baile [0,1]
    'energy',            # Intensidad perceptual [0,1]
    'key',               # Tonalidad [0-11]
    'loudness',          # Volumen en dB [-60, 0]
    'mode',              # Mayor/menor [0,1]
    'speechiness',       # Presencia de voz hablada [0,1]
    'acousticness',      # Probabilidad acustica [0,1]
    'instrumentalness',  # Probabilidad instrumental [0,1]
    'liveness',          # Presencia de audiencia [0,1]
    'valence',           # Positividad musical [0,1]
    'tempo',             # BPM
    'duration_ms'        # Duracion en milisegundos
]
```

---

## 4. DECISIONES ARQUITECTURALES CLAVE

### 4.1 Seleccion de Dataset Fuente

**Decision**: Usar dataset Kaggle 18K con letras en lugar de Spotify API 1.2M

**Justificacion**:
- Dataset 1.2M producia Hopkins ~0.45 (destruia estructura de clustering)
- Dataset 18K tiene Hopkins 0.823 (estructura natural excelente)
- Letras integradas permiten analisis multimodal sin APIs externas

### 4.2 Vectorizacion Semantica

**Decision**: Usar embeddings BERT directos para recomendaciones semanticas (sin clustering)

**Justificacion** (documentada en docs/SEMANTIC_RECOMMENDATIONS_ARCHITECTURE.md):
- Clustering semantico K-Means: Silhouette 0.1113 pero distribucion util
- Clustering Hierarchical: Silhouette 0.6733 pero distribucion 99.98% vs 0.02%
- Vectores directos: Granularidad maxima (8,567 niveles de similitud)
- Precision: Similitudes 89-99% documentadas experimentalmente
- Performance: <100ms por recomendacion validado

### 4.3 Fusion Multimodal

**Decision**: Ponderacion 55% musical, 45% semantico

**Justificacion**:
- Basada en experimentacion exhaustiva FASE 3 (56 configuraciones)
- Musical tiene mejor estructura tecnica (Silhouette 0.0965 vs 0.0329)
- Semantico aporta complementariedad tematica (NMI 0.0567 confirma independencia)
- Balance optimiza precision sin sacrificar diversidad

---

## 5. MODULOS DEL SISTEMA

### 5.1 Modulos con Documentacion CLAUDE.md Verificada

| Modulo | Archivo | Contenido |
|--------|---------|-----------|
| `data/` | CLAUDE.md | Guia de datasets, flujo de datos, separadores |
| `data/0_raw/` | CLAUDE.md | Dataset Spotify 1.2M (legacy) |
| `data/1_cleaned/` | CLAUDE.md | Dataset testing 500 |
| `data/2_with_lyrics/` | CLAUDE.md | Dataset fuente 18,454 |
| `data/3_selected/` | CLAUDE.md | Dataset produccion 10,000 |
| `data/4_vectorized/` | CLAUDE.md | Embeddings BERT 9,753 |
| `data/5_unified/` | CLAUDE.md | Dataset multimodal 7,811 |
| `data_selection/` | CLAUDE.md | Pipeline clustering-aware |
| `clustering/` | CLAUDE.md | Arquitectura de clustering |
| `exploratory_analysis/` | CLAUDE.md | Analisis exploratorio (actualizado Dic 2025) |

### 5.2 Modulos con CLAUDE.md (actualizados Dic 2025)

| Modulo | Archivo | Contenido |
|--------|---------|-----------|
| `recommendation_system/` | CLAUDE.md | Sistema hibrido, API, CLI, 4,828 lineas |

### 5.3 Modulos con README (sin CLAUDE.md) - A COMPLETAR

| Modulo | Archivo | Estado |
|--------|---------|--------|
| `clustering/evaluation_project/phase3_multimodal_clustering/` | README.md | Completo |
| `scripts/` | README.md | DESACTUALIZADO (ver seccion 7) |

### 5.4 Documentacion Tecnica Principal

| Documento | Ubicacion | Contenido | Estado |
|-----------|-----------|-----------|--------|
| FULL_PROJECT.md | docs/ | Documento base para tesis (341KB) | A ACTUALIZAR |
| SEMANTIC_RECOMMENDATIONS_ARCHITECTURE.md | docs/ | Decision vectores BERT directos | Completo |

---

## 6. METRICAS Y RESULTADOS

### 6.1 Clustering Musical

| Metrica | Baseline | Optimizado | Mejora |
|---------|----------|------------|--------|
| Silhouette Score | 0.1554 | 0.2893 | +86.1% |
| Hopkins Statistic | 0.788 | 0.823 | +4.4% |
| Algoritmo optimo | K-Means | Hierarchical K=3 | - |

### 6.2 Clustering Semantico

| Metrica | Valor | Interpretacion |
|---------|-------|----------------|
| Silhouette Score | 0.0329 | Esperado para 384D |
| Interpretabilidad | 0.7284 | Buena coherencia interna |
| K optimo | 6 | Balance granularidad/cohesion |

### 6.3 Sistema Multimodal

| Metrica | Valor |
|---------|-------|
| Canciones finales | 7,811 |
| Embeddings semanticos | [7811, 384] |
| Features musicales | [7811, 12] |
| Clusters musicales | K=10 |
| Clusters semanticos | K=6 |
| NMI cross-modal | 0.0567 |
| Performance recomendacion | <100ms |

### 6.4 Hopkins Statistic Post-Unificacion (FASE 2 - Dic 2025)

Evaluacion de clustering readiness sobre el dataset final unificado (7,811 canciones).

| Espacio | Hopkins | Std | IC 95% | Interpretacion |
|---------|---------|-----|--------|----------------|
| **Semantico (384D)** | 0.7752 | ±0.0015 | [0.7740, 0.7763] | Excellent clustering tendency |
| **Musical (12D)** | 0.7871 | ±0.0022 | [0.7854, 0.7888] | Excellent clustering tendency |

**Validacion Estadistica**:

| Test | Estadistico | p-value | Significativo |
|------|-------------|---------|---------------|
| t-test pareado | t = 12.70 | p = 4.73×10⁻⁷ | Si (α=0.01) |
| Wilcoxon | W = 0.0 | p = 0.002 | Si (α=0.01) |
| Cohen's d | 4.02 | - | Large effect |

**Interpretacion**: La perdida del 21.9% de datos (10,000 → 7,811) **NO degradó** la estructura de clustering. Ambos valores superan el umbral de 0.75 para "excelente clustering tendency".

### 6.5 Distribucion de Generos (Dataset Final)

| Genero | Cantidad | Porcentaje |
|--------|----------|------------|
| Rock | 1,927 | 24.7% |
| R&B | 1,555 | 19.9% |
| Pop | 1,418 | 18.2% |
| Rap | 1,372 | 17.6% |
| EDM | 782 | 10.0% |
| Latin | 757 | 9.7% |

---

## 7. LIMITACIONES CONOCIDAS

### 7.1 Metodologicas

1. **Dataset multimodal no intencional**: El dataset de 7,811 canciones es resultado residual de filtros tecnicos, no de diseno intencional. Perdida del 21.9% de datos originales.

2. **Validaciones completadas (FASE 2 - Dic 2025)**:
   - Hopkins Statistic post-unificacion: CALCULADO (Semantico 0.7752, Musical 0.7871)
   - Validacion estadistica: p < 0.001, Cohen's d = 4.02 (Large effect)
   - Resultado: Estructura de clustering preservada tras unificacion

3. **Validaciones pendientes**:
   - Analisis de sesgo por disponibilidad de letras
   - Validacion de representatividad vs dataset original
   - Test de estabilidad de clustering (multiples seeds)

4. **Sesgos potenciales no caracterizados**:
   - Canciones con letras "faciles" de procesar posiblemente sobrerrepresentadas
   - Posible sesgo hacia ciertos idiomas/generos
   - Caracteristicas musicales de canciones excluidas vs incluidas no comparadas

### 7.2 Documentales

1. **scripts/README.md desactualizado**: Describe pipeline de 1.2M canciones que fue abandonado. Los scripts mencionados (representative_selector.py, selection_validator.py, main_selection_pipeline.py) fueron movidos a `scripts/legacy/`.

2. **Referencia incorrecta en CLAUDE.md principal**: Menciona `clustering_evaluation_project/` como carpeta raiz, pero la ubicacion real es `clustering/evaluation_project/`.

3. **ANTEPROYECTO/ e INFORME_FINAL/**: Documentos potencialmente desalineados con el estado actual del proyecto. Tratados como documentos independientes.

---

## 8. SCRIPTS PRINCIPALES

### 8.1 Scripts Activos (Produccion)

| Script | Ubicacion | Funcion |
|--------|-----------|---------|
| `cluster_purification.py` | scripts/ | Sistema Hybrid Purification (+86.1%) |
| `run_final_clustering.py` | scripts/ | Orquestador clustering final |
| `quick_analysis.py` | scripts/ | Analisis rapido de datasets |
| `select_optimal_10k_from_18k.py` | data_selection/clustering_aware/ | Seleccion clustering-aware |
| `run_complete_vectorization.py` | clustering/algorithms/lyrics/scripts/ | Vectorizacion BERT |
| `evaluate_clustering_readiness_comparative.py` | clustering/evaluation_project/phase2_*/ | Hopkins comparativo FASE 2 |
| `run_multimodal_clustering_evaluation.py` | clustering/evaluation_project/phase3_*/ | Evaluacion FASE 3 |

### 8.2 Scripts del Sistema de Recomendacion

| Script | Ubicacion | Funcion |
|--------|-----------|---------|
| `load_system.py` | recommendation_system/scripts/ | Cargador centralizado |
| `music_recommender.py` | recommendation_system/scripts/ | Motor hibrido |
| `explain_recommendations.py` | recommendation_system/scripts/ | Sistema explicabilidad |
| `recommend_songs.py` | recommendation_system/scripts/ | CLI principal |

### 8.3 Scripts Legacy (NO USAR)

Ubicacion: `scripts/legacy/`

- `legacy/clustering/` - Algoritmos superados por cluster_purification.py
- `legacy/data_selection/` - Pipeline 1.2M obsoleto

---

## 9. COMANDOS DE USO

### 9.1 Clustering Musical Completo

```bash
python scripts/run_final_clustering.py
```
Tiempo estimado: 8-10 segundos

### 9.2 Recomendaciones Musicales

```bash
# Por track_id
python recommendation_system/scripts/recommend_songs.py --track_id "TRACK_ID"

# Por nombre de cancion
python recommendation_system/scripts/recommend_songs.py --song_name "Bohemian Rhapsody"

# Modo interactivo
python recommendation_system/scripts/recommend_songs.py --interactive

# Demostracion
python recommendation_system/scripts/recommend_songs.py --demo
```

### 9.3 Analisis Rapido de Dataset

```bash
python scripts/quick_analysis.py --dataset optimal
```

### 9.4 Evaluacion Multimodal FASE 3

```bash
cd clustering/evaluation_project/phase3_multimodal_clustering
python run_multimodal_clustering_evaluation.py \
  --dataset ../phase1_dataset_unification/unified_multimodal_dataset_*.pkl \
  --output ./results
```

---

## 10. CARGA DE DATASETS

### 10.1 Dataset Multimodal (Recomendado)

```python
import pickle

with open('data/5_unified/unified_multimodal_7811.pkl', 'rb') as f:
    dataset = pickle.load(f)

track_ids = dataset['data']['track_ids']                     # [7811]
semantic = dataset['data']['semantic_embeddings']            # [7811, 384]
musical_norm = dataset['data']['musical_features_normalized'] # [7811, 12]
metadata = dataset['data']['track_metadata']                 # DataFrame
```

### 10.2 Dataset Musical (Solo Audio Features)

```python
import pandas as pd

df = pd.read_csv('data/3_selected/picked_data_optimal.csv',
                 sep='^',
                 encoding='utf-8')
# 10,000 canciones, 12 audio features
```

### 10.3 Dataset Fuente con Letras

```python
import pandas as pd

df = pd.read_csv('data/2_with_lyrics/spotify_songs_fixed.csv',
                 sep='@@',
                 engine='python',
                 encoding='utf-8')
# 18,454 canciones con letras completas
```

---

## 11. DEPENDENCIAS PRINCIPALES

```
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.0.0
sentence-transformers>=2.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
tqdm>=4.64.0
joblib>=1.1.0
```

---

## 12. REFERENCIAS DOCUMENTALES

### Documentacion Interna

| Documento | Contenido |
|-----------|-----------|
| `CLAUDE.md` | Configuracion Claude Code, directivas, comandos |
| `docs/FULL_PROJECT.md` | Metodologia cientifica completa para tesis |
| `docs/SEMANTIC_RECOMMENDATIONS_ARCHITECTURE.md` | Decision arquitectural vectores BERT |
| `data/CLAUDE.md` | Guia de datasets y flujo de datos |
| `data_selection/CLAUDE.md` | Pipeline clustering-aware |
| `clustering/CLAUDE.md` | Arquitectura de clustering |
| `clustering/evaluation_project/phase2_*/results/` | Reportes Hopkins FASE 2 (Dic 2025) |
| `exploratory_analysis/CLAUDE.md` | Modulo analisis exploratorio |
| `recommendation_system/README.md` | Sistema de recomendacion |

### Referencias Academicas

- Hopkins Statistic: Lawson & Jurs (1990)
- MaxMin Sampling: Gonzalez (1985)
- Silhouette Analysis: Rousseeuw (1987)
- BERT: Devlin et al. (2019)
- Sentence-BERT: Reimers & Gurevych (2019)

---

*Documento generado: Diciembre 2025*
*Ultima actualizacion: 8 Diciembre 2025 (Hopkins FASE 2, limpieza recommendation_system)*
*Estado del proyecto: Sistema funcional con componentes principales completados*
