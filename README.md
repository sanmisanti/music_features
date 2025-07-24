# Módulo de Análisis de Características Musicales

Sistema de clustering y recomendación basado en características musicales usando Spotify Audio Features y embeddings profundos de audio.

## 🎯 Contexto del Proyecto

Este módulo es parte de un **sistema de recomendación musical multimodal** más amplio que combina:
- **Análisis Musical** (este módulo) - Características de audio y clustering
- **Análisis Semántico** (futuro) - Procesamiento de letras con NLP
- **Fusión Multimodal** (futuro) - Integración de ambos espacios vectoriales

📖 **Ver [FULL_PROJECT.md](./FULL_PROJECT.md) para la visión completa del sistema**

## 🎵 Funcionalidades Actuales

### ✅ Implementado
- **Clustering K-Means** de canciones basado en 13 características de Spotify
- **Pipeline de datos** para limpieza y procesamiento de datasets masivos
- **Sistema de recomendación** cluster-based con múltiples métricas de similitud
- **Análisis OpenL3** para embeddings profundos de audio (512 dimensiones)
- **Visualizaciones** de clusters usando PCA y análisis estadístico

### 🔄 En Desarrollo
- Optimización de hiperparámetros de clustering
- Ensemble de múltiples algoritmos de clustering
- Feature engineering avanzado
- Integración completa de OpenL3 embeddings

## 🏗️ Arquitectura del Módulo

```
[Dataset Spotify] → [Limpieza] → [Normalización] → [Clustering] → [Recomendaciones]
     (1.2M)           clean.py     StandardScaler     K-Means        pred.ipynb
        ↓                                                ↓
[Audio Files] → [OpenL3] → [Embeddings 512D] → [Fusión] → [Similitud Avanzada]
   (.mp3)     aa_openl3.py    (.npy files)    (futuro)      (coseno/euclidiana)
```

## 📊 Dataset y Resultados

### Dataset
- **Fuente**: Spotify Million Playlist Dataset
- **Tamaño**: ~1.2M canciones con 24 features
- **Características**: 13 audio features (danceability, energy, valence, tempo, etc.)
- **Formato**: CSV con separadores `;` y decimales `,` (locale español)

### Resultados de Clustering
- **Algoritmo**: K-Means con normalización StandardScaler
- **K Óptimo**: 7 clusters (silhouette score: 0.177)
- **Distribución**: [42, 197, 110, 7, 37, 26, 81] canciones por cluster
- **Visualización**: PCA 2D para interpretación

## 🚀 Inicio Rápido

### Instalación
```bash
# Clonar repositorio
git clone <repo-url>
cd music_features

# Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn plotly
pip install openl3 librosa soundfile  # Para análisis de audio
```

### Uso Básico

#### 1. Limpiar Dataset
```bash
python clean.py
```
Genera:
- `tracks_features_clean.csv` - Dataset completo limpio
- `tracks_features_500.csv` - Muestra de 500 canciones

#### 2. Ejecutar Clustering
```bash
jupyter notebook clustering/cluster.ipynb
```
Proceso completo:
- Carga y preprocesamiento de datos
- Búsqueda de K óptimo (método del codo + silhouette)
- Análisis de clusters musicales
- Visualizaciones PCA
- Guardado de resultados en `clustering_results.csv`

#### 3. Generar Recomendaciones
```bash
jupyter notebook pred.ipynb
```
Funcionalidades:
- Predicción de cluster para nuevas canciones
- Búsqueda de canciones similares
- Múltiples métricas de similitud

#### 4. Análisis de Audio con OpenL3
```bash
python audio_analysis/aa_openl3.py
```
Genera embeddings de 512 dimensiones para archivos MP3.

## 📁 Estructura del Proyecto

```
music_features/
├── 📁 data/
│   ├── original_data/
│   │   ├── tracks_features.csv      # Dataset original (1.2M)
│   │   └── tracks_features_reduced.csv
│   └── cleaned_data/
│       ├── tracks_features_clean.csv    # Dataset limpio completo
│       └── tracks_features_500.csv      # Muestra de desarrollo
├── 📁 clustering/
│   ├── cluster.ipynb               # Notebook principal de clustering
│   └── clustering_results.csv     # Resultados con asignaciones
├── 📁 audio_analysis/
│   ├── aa_openl3.py               # Análisis OpenL3
│   ├── we_will_rock_you.mp3       # Audio de ejemplo
│   └── *.npy                      # Embeddings generados
├── clean.py                       # Script de limpieza de datos
├── pred.ipynb                     # Sistema de recomendaciones
├── README.md                      # Este archivo
├── CLAUDE.md                      # Guía para Claude Code
└── FULL_PROJECT.md               # Visión completa del sistema
```

## 🎼 Características Musicales Analizadas

| Feature | Descripción | Rango |
|---------|-------------|-------|
| `danceability` | Qué tan bailable es la canción | 0.0 - 1.0 |
| `energy` | Intensidad y potencia percibida | 0.0 - 1.0 |
| `key` | Clave musical de la canción | 0 - 11 |
| `loudness` | Volumen general en decibelios | -60 - 0 dB |
| `mode` | Modalidad (mayor=1, menor=0) | 0, 1 |
| `speechiness` | Presencia de palabras habladas | 0.0 - 1.0 |
| `acousticness` | Medida de si la canción es acústica | 0.0 - 1.0 |
| `instrumentalness` | Predice si no tiene voz | 0.0 - 1.0 |
| `liveness` | Detección de audiencia en vivo | 0.0 - 1.0 |
| `valence` | Positividad musical transmitida | 0.0 - 1.0 |
| `tempo` | Tempo estimado en BPM | ~0 - 250 |
| `duration_ms` | Duración en milisegundos | Variable |
| `time_signature` | Compás estimado | 3 - 7 |

## 🔬 Metodología

### Pipeline de Clustering
1. **Preprocesamiento**: Limpieza y normalización con `StandardScaler`
2. **Optimización**: Búsqueda de K óptimo usando silhouette score y método del codo
3. **Clustering**: K-Means con múltiples inicializaciones aleatorias
4. **Evaluación**: Análisis de distribución y calidad de clusters
5. **Visualización**: Reducción PCA para interpretación 2D

### Sistema de Recomendación
1. **Asignación**: Predicción de cluster para nueva canción
2. **Filtrado**: Selección de canciones del mismo cluster
3. **Similitud**: Cálculo usando coseno, euclidiana o Manhattan
4. **Ranking**: Ordenamiento por similitud descendente
5. **Resultado**: Top-N canciones más similares

## 📈 Métricas y Evaluación

### Métricas de Clustering
- **Silhouette Score**: Calidad de separación entre clusters
- **Inercia (WCSS)**: Compactidad interna de clusters
- **Distribución**: Balance entre clusters

### Métricas de Recomendación
- **Similitud Coseno**: Orientación vectorial
- **Distancia Euclidiana**: Distancia geométrica
- **Distancia Manhattan**: Distancia por coordenadas

## 🎯 Próximos Pasos

### Mejoras Inmediatas
- [ ] **Optimización de clustering**: Ensemble de algoritmos (DBSCAN, Hierarchical)
- [ ] **Feature engineering**: Ratios entre features, features temporales
- [ ] **Hiperparámetros**: Grid search automático
- [ ] **Evaluación**: Métricas de recomendación (Precision@K, NDCG)

### Integración con Sistema Completo
- [ ] **API de recomendaciones**: FastAPI para servir el modelo
- [ ] **Preparación para fusión**: Estandardización de outputs
- [ ] **Escalabilidad**: Optimización para datasets masivos
- [ ] **Monitoring**: Métricas de rendimiento y calidad

## 🔧 Configuración Técnica

### Dependencias Principales
```python
pandas>=1.3.0          # Manipulación de datos
numpy>=1.21.0           # Computación numérica
scikit-learn>=1.0.0     # Machine learning
matplotlib>=3.4.0       # Visualización básica
seaborn>=0.11.0         # Visualización estadística
plotly>=5.0.0           # Visualización interactiva
openl3>=0.4.0           # Embeddings de audio
librosa>=0.8.0          # Procesamiento de audio
soundfile>=0.10.0       # I/O de archivos de audio
```

### Configuración de Ambiente
```python
# Variables importantes
RANDOM_STATE = 42       # Reproducibilidad
SAMPLE_SIZE = 500       # Tamaño de muestra para desarrollo
CSV_SEPARATOR = ';'     # Separador de campos
DECIMAL_SEPARATOR = ',' # Separador decimal (locale español)
```

## 📚 Referencias y Recursos

### Papers Relevantes
- "Audio Set: An ontology and human-labeled dataset for audio events" (Google, 2017)
- "Look, Listen and Learn More: Design Choices for Deep Audio Embeddings" (OpenL3, 2019)
- "The Million Playlist Dataset: Recsys Challenge 2018" (Spotify)

### APIs y Datasets
- [Spotify Web API](https://developer.spotify.com/documentation/web-api/) - Características de audio
- [OpenL3](https://github.com/marl/openl3) - Embeddings profundos de audio
- [Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) - Dataset principal

## 📞 Soporte

Para desarrollo con Claude Code:
- **CLAUDE.md**: Instrucciones específicas para IA
- **FULL_PROJECT.md**: Contexto completo del sistema
- **Issues**: Reporte de problemas y sugerencias

---

**Parte del proyecto de tesis**: Sistema de Recomendación Musical Multimodal  
**Estado**: Módulo base implementado, en proceso de optimización  
**Siguiente módulo**: Análisis Semántico de Letras