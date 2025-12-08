# MODULO RECOMMENDATION_SYSTEM

**Ultima actualizacion**: Diciembre 2025
**Estado**: PRODUCCION - Sistema funcional validado

---

## OBJETIVO DEL MODULO

Sistema de recomendacion musical hibrido que combina similitud semantica (embeddings BERT 384D) con similitud musical (caracteristicas Spotify 12D). Implementa arquitectura validada experimentalmente en FASE 3 con pesos optimizados (55% musical, 45% semantico).

---

## ARQUITECTURA

```
recommendation_system/
├── CLAUDE.md                  # Esta documentacion
├── README.md                  # Guia de uso y API
│
├── scripts/                   # Sistema de produccion (4,828 lineas)
│   ├── load_system.py         # Cargador centralizado
│   ├── music_recommender.py   # Motor hibrido
│   ├── explain_recommendations.py  # Sistema explicabilidad
│   ├── recommend_songs.py     # Interface CLI principal
│   ├── analyze_clusters.py    # Analisis de clusters (WIP)
│   └── validate_system.py     # Suite validacion (WIP)
│
├── data/                      # Datos del sistema (~25 MB)
│   ├── semantic_embeddings.npy       # [7811, 384] BERT L2-norm
│   ├── musical_features_normalized.npy  # [7811, 12] StandardScaler
│   ├── songs_metadata.csv            # Metadatos canciones
│   └── track_ids.npy                 # IDs alineacion
│
├── clusters/                  # Asignaciones de clusters
│   ├── musical_clusters_k10.npy      # K-Means++ K=10
│   └── semantic_clusters_k6.npy      # K-Means++ K=6
│
└── config/                    # Configuracion validada
    ├── system_config.json            # Pesos y parametros
    └── fase3_results.json            # Resultados experimentales
```

---

## COMPONENTES PRINCIPALES

### 1. MusicDataLoader (`load_system.py` - 339 lineas)

Cargador centralizado con cache para evitar re-cargas.

```python
from scripts.load_system import MusicDataLoader

loader = MusicDataLoader()
semantic = loader.get_semantic_vectors()     # [7811, 384]
musical = loader.get_musical_vectors()       # [7811, 12]
track_ids = loader.get_track_ids()           # [7811]
metadata = loader.get_metadata()             # DataFrame
clusters_m = loader.get_musical_clusters()   # [7811] K=10
clusters_s = loader.get_semantic_clusters()  # [7811] K=6
config = loader.get_config()                 # dict
```

### 2. HybridMusicRecommender (`music_recommender.py` - 448 lineas)

Motor de recomendacion hibrido con pesos validados FASE 3.

```python
from scripts.music_recommender import HybridMusicRecommender

recommender = HybridMusicRecommender()  # Pesos FASE 3: 55% musical, 45% semantico

# Recomendaciones por track_id
recs = recommender.recommend(track_id="TRACK_ID", n_recommendations=10)

# Pesos personalizados
recommender_custom = HybridMusicRecommender(
    custom_musical_weight=0.60,
    custom_semantic_weight=0.40
)
```

**Metodos principales**:
- `recommend(track_id, n_recommendations)` - Recomendaciones hibridas
- `recommend_musical_only(track_id, n)` - Solo similitud musical
- `recommend_semantic_only(track_id, n)` - Solo similitud semantica
- `get_song_info(track_id)` - Informacion de cancion

### 3. RecommendationExplainer (`explain_recommendations.py` - 1,079 lineas)

Sistema de explicabilidad con analisis de clusters.

```python
from scripts.explain_recommendations import RecommendationExplainer

explainer = RecommendationExplainer()

# Explicacion individual
explanation = explainer.explain_recommendation(
    query_track_id="TRACK_ID",
    recommended_track_id="REC_TRACK_ID"
)

# Batch de explicaciones
explanations = explainer.get_batch_explanations(recommendations)

# Analisis de cluster
cluster_info = explainer.get_cluster_analysis(track_id)
```

### 4. MusicRecommendationInterface (`recommend_songs.py` - 701 lineas)

Interface unificada con CLI completa.

```python
from scripts.recommend_songs import MusicRecommendationInterface

interface = MusicRecommendationInterface()

# Por track_id
result = interface.recommend_by_track_id("TRACK_ID", include_explanations=True)

# Por nombre de cancion
result = interface.recommend_by_name("Bohemian Rhapsody", "Queen")

# Busqueda
matches = interface.search_songs("stairway to heaven")
```

---

## CONFIGURACION DEL SISTEMA

### system_config.json

```json
{
  "dataset_info": {
    "total_songs": 7811,
    "semantic_dimensions": 384,
    "musical_dimensions": 12
  },
  "optimal_configurations": {
    "musical_clustering": {
      "algorithm": "kmeans_plus",
      "n_clusters": 10,
      "silhouette_score": 0.0965
    },
    "semantic_clustering": {
      "algorithm": "kmeans_plus",
      "n_clusters": 6,
      "silhouette_score": 0.0329
    }
  },
  "recommendation_weights": {
    "musical_weight": 0.55,
    "semantic_weight": 0.45
  }
}
```

---

## COMANDOS DE USO

### CLI Principal

```bash
# Por track_id
python scripts/recommend_songs.py --track_id "TRACK_ID" --n_recommendations 10

# Por nombre de cancion
python scripts/recommend_songs.py --song_name "Bohemian Rhapsody" --artist "Queen"

# Busqueda de canciones
python scripts/recommend_songs.py --search "stairway to heaven"

# Modo interactivo
python scripts/recommend_songs.py --interactive

# Demostracion
python scripts/recommend_songs.py --demo
```

### Ejecucion desde raiz del proyecto

```bash
cd recommendation_system
python scripts/recommend_songs.py --interactive
```

---

## DATOS DEL SISTEMA

### Archivos de Datos

| Archivo | Shape | Descripcion |
|---------|-------|-------------|
| `semantic_embeddings.npy` | [7811, 384] | Embeddings BERT L2-normalizados |
| `musical_features_normalized.npy` | [7811, 12] | Features Spotify StandardScaler |
| `track_ids.npy` | [7811] | IDs unicos para alineacion |
| `songs_metadata.csv` | 7811 rows | track_name, artist_name, genre, etc. |

### Archivos de Clusters

| Archivo | Shape | Configuracion |
|---------|-------|---------------|
| `musical_clusters_k10.npy` | [7811] | K-Means++ K=10, Silhouette 0.0965 |
| `semantic_clusters_k6.npy` | [7811] | K-Means++ K=6, Silhouette 0.0329 |

---

## METRICAS DEL SISTEMA

| Metrica | Valor |
|---------|-------|
| Canciones | 7,811 |
| Performance recomendacion | <100ms |
| Pesos hibridos | 55% musical, 45% semantico |
| Clusters musicales | K=10 (Silhouette 0.0965) |
| Clusters semanticos | K=6 (Silhouette 0.0329) |
| NMI cross-modal | 0.0567 |

---

## SCRIPTS EN DESARROLLO

| Script | Estado | Descripcion |
|--------|--------|-------------|
| `analyze_clusters.py` | WIP | Analisis estadistico avanzado de clusters |
| `validate_system.py` | WIP | Suite de testing Precision@K, Recall@K |

Estos scripts estan implementados pero no finalizados. Funcionalidad basica disponible.

---

## DEPENDENCIAS

```python
numpy>=1.20.0
pandas>=1.5.0
scikit-learn>=1.0.0
```

---

## ORIGEN DE DATOS

Los datos de este modulo provienen de:

- **Embeddings semanticos**: `data/5_unified/unified_multimodal_7811.pkl`
- **Clusters**: `clustering/evaluation_project/phase3_multimodal_clustering/results/`
- **Configuracion**: Resultados experimentales FASE 3

---

## REFERENCIAS

- `data/5_unified/CLAUDE.md` - Dataset multimodal fuente
- `clustering/evaluation_project/phase3_multimodal_clustering/README.md` - Experimentacion FASE 3
- `docs/SEMANTIC_RECOMMENDATIONS_ARCHITECTURE.md` - Decision arquitectural

---

*Modulo limpiado: Diciembre 2025*
*Eliminados: testing_scripts/, songs_catalog_basic_*.csv*
