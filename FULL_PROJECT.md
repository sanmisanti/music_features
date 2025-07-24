# FULL_PROJECT.md - Sistema de Recomendación Musical Multimodal

## Visión General del Proyecto

Este documento describe la arquitectura completa del sistema de recomendación musical basado en análisis multimodal que combina características musicales y análisis semántico de letras.

### Objetivo Principal
Desarrollar un sistema de recomendación musical avanzado que utilice tanto las características propias de la música (audio features) como el análisis semántico de las letras para generar recomendaciones precisas y contextualmente relevantes.

## Arquitectura del Sistema Completo

### Componentes Principales

#### 1. **Módulo de Análisis Musical** (Este Proyecto)
- **Función**: Procesar características audio de las canciones
- **Tecnologías**: Spotify Audio Features + OpenL3 + Clustering K-Means
- **Output**: Espacio vectorial de características musicales
- **Estado**: ✅ Implementado (K=7 clusters, silhouette score 0.177)

#### 2. **Módulo de Análisis Semántico de Letras** (Por Desarrollar)
- **Función**: Analizar el contenido semántico y emocional de las letras
- **Tecnologías Propuestas**: BERT, Sentence-BERT, análisis emocional
- **Output**: Espacio vectorial semántico
- **Estado**: 🔄 Pendiente de desarrollo

#### 3. **Módulo de Fusión Multimodal** (Por Desarrollar)
- **Función**: Combinar ambos espacios vectoriales de manera inteligente
- **Tecnologías Propuestas**: CCA, redes neuronales, ensemble methods
- **Output**: Recomendaciones integradas
- **Estado**: 🔄 Pendiente de desarrollo

#### 4. **Sistema de Evaluación y Métricas** (Por Desarrollar)
- **Función**: Evaluar calidad de recomendaciones
- **Métricas**: Precision@K, Recall@K, diversidad, novedad
- **Estado**: 🔄 Pendiente de desarrollo

## Flujo de Datos Completo

```
[Canción de Usuario]
         ↓
    [Extracción de Features]
         ↓
┌─[Audio Features]─────┐    ┌─[Letras]──────────┐
│ • Spotify Features   │    │ • Texto completo  │
│ • OpenL3 Embeddings  │    │ • Preprocessing   │
│ • Librosa Features   │    │ • Tokenización    │
└─────────────────────┘    └───────────────────┘
         ↓                           ↓
┌─[Clustering Musical]─┐    ┌─[Análisis Semántico]─┐
│ • K-Means            │    │ • BERT Embeddings    │
│ • Normalización      │    │ • Análisis Emocional │
│ • Espacio vectorial  │    │ • Temas musicales    │
└─────────────────────┘    └───────────────────────┘
         ↓                           ↓
         └──────────[Fusión Multimodal]──────────┘
                           ↓
                 [Recomendaciones Finales]
                           ↓
                   [Evaluación y Ranking]
```

## Enfoques Técnicos Avanzados por Módulo

### Módulo de Características Musicales (Mejoras Propuestas)

#### Clustering Avanzado
```python
# Ensemble de algoritmos múltiples
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

# Clustering jerárquico multi-escala
# Nivel 1: Géneros principales (Rock, Pop, Jazz, etc.)
# Nivel 2: Subgéneros (Rock alternativo, Pop latino, etc.)
# Nivel 3: Estilos específicos (Grunge, Reggaeton, etc.)

# Optimización automática de hiperparámetros
from sklearn.model_selection import GridSearchCV
```

#### Feature Engineering Avanzado
```python
# Combinación de múltiples fuentes
audio_features = {
    'spotify': ['danceability', 'energy', 'valence', ...],
    'openl3': [512-dim deep embeddings],
    'librosa': ['mfcc', 'chroma', 'spectral_contrast', ...],
    'derived': ['energy_ratio', 'tempo_stability', ...]
}

# Features temporales para canciones
temporal_features = [
    'feature_variance',  # Variabilidad temporal
    'feature_trends',    # Tendencias en la canción
    'structural_breaks'  # Cambios de sección
]
```

### Módulo Semántico de Letras (Diseño Propuesto)

#### Análisis Multi-dimensional
```python
# Embeddings semánticos
from transformers import AutoModel, AutoTokenizer

models_pipeline = {
    'semantic': 'sentence-transformers/all-MiniLM-L6-v2',
    'emotion': 'j-hartmann/emotion-english-distilroberta-base',
    'music_genre': 'music-specific-bert-model',  # Cuando esté disponible
    'themes': 'custom-topic-model'
}

# Análisis estructural de letras
lyric_features = {
    'semantic_embeddings': [768-dim],
    'emotional_profile': ['joy', 'sadness', 'anger', 'love', ...],
    'themes': ['party', 'love', 'social_issues', 'personal_growth', ...],
    'linguistic': ['complexity', 'metaphor_density', 'rhyme_scheme'],
    'cultural': ['language', 'cultural_references', 'slang_usage']
}
```

#### Procesamiento Especializado
```python
# Preprocesamiento específico para letras musicales
def preprocess_lyrics(lyrics):
    # Manejo de repeticiones (chorus, verse)
    # Eliminación de anotaciones [Chorus], [Verse]
    # Normalización de contracciones
    # Detección de idioma automática
    # Traducción opcional para análisis multiidioma
    pass

# Análisis semántico jerárquico
def semantic_analysis(lyrics):
    # Nivel palabra: embeddings individuales
    # Nivel línea: coherencia semántica
    # Nivel estrofa: temas específicos
    # Nivel canción: mensaje general
    pass
```

### Módulo de Fusión Multimodal (Arquitectura Propuesta)

#### Estrategias de Fusión
```python
# 1. Fusión Temprana (Feature-level)
def early_fusion(music_features, lyric_features, weights=[0.6, 0.4]):
    """Concatenación ponderada de features normalizados"""
    combined = np.concatenate([
        weights[0] * normalize(music_features),
        weights[1] * normalize(lyric_features)
    ])
    return combined

# 2. Fusión Tardía (Decision-level)
def late_fusion(music_recs, lyric_recs, strategy='weighted_average'):
    """Combinar rankings de recomendaciones independientes"""
    if strategy == 'weighted_average':
        return 0.7 * music_recs + 0.3 * lyric_recs
    elif strategy == 'rank_aggregation':
        return aggregate_rankings([music_recs, lyric_recs])

# 3. Fusión Híbrida (Neural Networks)
class MultimodalRecommender(nn.Module):
    def __init__(self, music_dim=13, lyric_dim=768, hidden_dim=128):
        super().__init__()
        self.music_encoder = nn.Sequential(
            nn.Linear(music_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.lyric_encoder = nn.Sequential(
            nn.Linear(lyric_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Similarity score
        )
    
    def forward(self, music_features, lyric_features):
        music_encoded = self.music_encoder(music_features)
        lyric_encoded = self.lyric_encoder(lyric_features)
        combined = torch.cat([music_encoded, lyric_encoded], dim=1)
        return self.fusion_layer(combined)
```

#### Alineación de Espacios Vectoriales
```python
# Canonical Correlation Analysis para alinear espacios
from sklearn.cross_decomposition import CCA

def align_vector_spaces(music_embeddings, lyric_embeddings):
    """Encuentra transformaciones para maximizar correlación"""
    cca = CCA(n_components=min(music_embeddings.shape[1], 
                              lyric_embeddings.shape[1]))
    music_aligned, lyric_aligned = cca.fit_transform(
        music_embeddings, lyric_embeddings
    )
    return music_aligned, lyric_aligned, cca

# Cross-modal learning
def learn_cross_modal_mapping(music_data, lyric_data):
    """Aprende mapeo entre espacios usando canciones etiquetadas"""
    # Usar canciones con alta similitud conocida
    # Entrenar modelo para predecir similitud cross-modal
    pass
```

## Sistema de Evaluación Avanzado

### Métricas de Calidad
```python
evaluation_metrics = {
    'accuracy': ['precision@k', 'recall@k', 'f1@k'],
    'ranking': ['ndcg@k', 'map@k', 'mrr'],
    'diversity': ['intra_list_diversity', 'genre_diversity'],
    'novelty': ['catalog_coverage', 'temporal_diversity'],
    'serendipity': ['unexpected_relevance', 'cross_genre_recs'],
    'user_satisfaction': ['click_through_rate', 'listening_time']
}
```

### Benchmarking y Baselines
```python
baseline_methods = {
    'content_based': 'Spotify features only',
    'collaborative': 'User-item interactions',
    'popularity': 'Most popular tracks',
    'random': 'Random recommendations',
    'genre_based': 'Same genre recommendations'
}
```

## Consideraciones Técnicas

### Escalabilidad
```python
# Para millones de canciones
import faiss  # Approximate Nearest Neighbors
import dask   # Distributed computing

# Indexación eficiente
def build_scalable_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index

# Procesamiento distribuido
def process_large_dataset(data_path):
    df = dd.read_csv(data_path)  # Dask dataframe
    return df.map_partitions(process_chunk)
```

### Optimización de Rendimiento
```python
performance_optimizations = {
    'caching': 'Redis para embeddings calculados',
    'batch_processing': 'Procesamiento en lotes para eficiencia',
    'model_compression': 'Quantización de modelos BERT',
    'index_pruning': 'Reducción de dimensionalidad inteligente'
}
```

## Roadmap de Desarrollo

### Fases del Proyecto

**Fase 1: Base Musical** ✅ *Completada*
- [x] Clustering básico con Spotify features
- [x] Pipeline de datos y visualizaciones
- [x] Métricas básicas de clustering

**Fase 2: Optimización Musical** 🔄 *En progreso*
- [ ] Integración de OpenL3 embeddings
- [ ] Ensemble de algoritmos de clustering
- [ ] Feature engineering avanzado
- [ ] Optimización de hiperparámetros

**Fase 3: Módulo Semántico** 📋 *Planeada*
- [ ] Pipeline de procesamiento de letras
- [ ] Implementación de embeddings BERT
- [ ] Análisis emocional y temático
- [ ] Sistema de features semánticas

**Fase 4: Fusión Multimodal** 📋 *Planeada*
- [ ] Estrategias de fusión temprana y tardía
- [ ] Modelo neuronal para fusión híbrida
- [ ] Alineación de espacios vectoriales
- [ ] Sistema de pesos adaptativos

**Fase 5: Evaluación Avanzada** 📋 *Planeada*
- [ ] Métricas de evaluación completas
- [ ] Sistema de benchmarking
- [ ] Validación con usuarios reales
- [ ] Optimización basada en feedback

**Fase 6: Sistema Completo** 📋 *Futura*
- [ ] API de recomendaciones
- [ ] Interfaz de usuario
- [ ] Sistema de monitoreo
- [ ] Deployment en producción

## Consideraciones de Investigación

### Contribuciones Científicas Potenciales
1. **Fusión Multimodal Musical**: Nuevo enfoque para combinar audio y texto en música
2. **Clustering Musical Avanzado**: Mejoras en clustering de características musicales
3. **Embeddings Semánticos Musicales**: Adaptación de BERT para letras musicales
4. **Métricas de Evaluación**: Nuevas métricas para sistemas musicales multimodales

### Posibles Publicaciones
- Conferencias: ISMIR, RecSys, ICML
- Journals: ACM TORS, IEEE Multimedia, Music Information Retrieval

### Datasets y Recursos
- Spotify Million Playlist Dataset
- Last.fm Dataset
- MusicBrainz
- Genius Lyrics API
- AudioSet de Google

## Referencias Técnicas

### Papers Clave
1. "Deep Learning for Music Recommendation Systems" (2019)
2. "Multimodal Deep Learning for Recommendation Systems" (2020)
3. "BERT for Music: Semantic Analysis of Lyrics" (2021)
4. "Cross-Modal Learning for Audio-Text Retrieval" (2022)

### Librerías y Frameworks
```python
core_libraries = {
    'ml': ['scikit-learn', 'pytorch', 'transformers'],
    'audio': ['librosa', 'openl3', 'essentia'],
    'nlp': ['spacy', 'nltk', 'sentence-transformers'],
    'data': ['pandas', 'numpy', 'dask'],
    'viz': ['matplotlib', 'plotly', 'seaborn'],
    'serving': ['fastapi', 'redis', 'docker']
}
```

Este documento debe ser la referencia principal para entender la visión completa del proyecto y guiar el desarrollo de todos los módulos.