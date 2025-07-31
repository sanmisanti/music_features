# 🎵 Módulo de Clustering Musical

Sistema de clustering avanzado para análisis y recomendación musical basado en características de audio de Spotify. Implementa algoritmos K-Means optimizados con múltiples métricas de evaluación y sistemas de recomendación por similitud.

## 📁 Estructura del Proyecto

```
clustering/
├── algorithms/              # Algoritmos de clustering
│   ├── clustering.py           # K-Means básico (desarrollo/educativo)
│   ├── clustering_optimized.py # K-Means optimizado para datasets grandes
│   └── clustering_pca.py       # K-Means con reducción PCA
├── models/                  # Modelos entrenados y validados
│   └── final_models/           # 3 mejores métodos de clustering
│       ├── method1_pca5_silhouette0314/      # 🏆 MEJOR (Producción)
│       ├── method2_pca8_silhouette0251/      # Alternativo
│       └── method3_optimized_silhouette0231/ # Referencia
├── recommender/             # Sistemas de recomendación
│   ├── music_recommender.py      # Recomendador estándar
│   └── music_recommender_full.py # Recomendador con dataset completo (1.2M)
├── notebooks/               # Análisis exploratorio
│   └── cluster.ipynb            # Jupyter notebook interactivo
├── results/                 # Resultados de clustering
│   └── clustering_results.csv   # Dataset con clusters asignados
├── tests/                   # Pruebas y validación
│   └── test_compatibility.py    # Validación de compatibilidad
├── utils/                   # Utilidades (reservado para futuras implementaciones)
├── requirements.txt         # Dependencias Python
└── README.md               # Esta documentación
```

## 🎯 Funcionalidad Principal

### Sistema de Clustering
- **Entrada**: Canciones con 13 características musicales de Spotify API
- **Algoritmo**: K-Means con búsqueda automática de K óptimo
- **Métricas**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
- **Optimizaciones**: PCA opcional, normalización StandardScaler, manejo de outliers

### Sistema de Recomendación
- **Predicción de Clusters**: Asigna nuevas canciones a clusters existentes
- **Métricas de Similitud**: Manhattan, Euclidiana, Coseno
- **Salida**: Top-N canciones más similares del mismo cluster

## 🏆 Modelos Validados

### 🥇 Method 1: PCA 5 Componentes (PRODUCCIÓN)
- **Silhouette Score**: 0.314 (+37.2% vs baseline)
- **K óptimo**: 3 clusters
- **Varianza explicada**: 66.6%
- **Uso**: Sistema principal de producción
- **comando**: `python recommender/music_recommender.py --models-dir models/final_models/method1_pca5_silhouette0314`

### 🥈 Method 2: PCA 8 Componentes (ALTERNATIVO)
- **Silhouette Score**: 0.251 (+8.7% vs baseline)
- **K óptimo**: 3 clusters
- **Varianza explicada**: ~75%
- **Uso**: Sistema alternativo más conservador

### 🥉 Method 3: Optimizado Base (REFERENCIA)
- **Silhouette Score**: 0.231 (baseline)
- **K óptimo**: 3 clusters
- **Features**: 13 características originales completas
- **Uso**: Sistema de referencia sin reducción dimensional

## 🚀 Guía de Uso Rápido

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 2. Clustering de Nuevos Datos
```bash
# Clustering básico (desarrollo)
python algorithms/clustering.py

# Clustering optimizado para datasets grandes
python algorithms/clustering_optimized.py --dataset ../data/picked_data_0.csv --save-models

# Clustering con PCA (recomendado)
python algorithms/clustering_pca.py --dataset ../data/picked_data_0.csv --pca-components 5 --save-models
```

### 3. Recomendaciones Musicales
```bash
# Recomendación por ID de canción (recomendado)
python recommender/music_recommender.py \
    --models-dir models/final_models/method1_pca5_silhouette0314 \
    --song-id "5zlcxSrYyFmCmSRbere3c5" \
    --top-n 5

# Recomendación por características JSON
python recommender/music_recommender.py \
    --models-dir models/final_models/method1_pca5_silhouette0314 \
    --song-features song.json \
    --top-n 10

# Modo interactivo
python recommender/music_recommender.py \
    --models-dir models/final_models/method1_pca5_silhouette0314 \
    --interactive
```

### 4. Dataset Completo (1.2M canciones)
```bash
python recommender/music_recommender_full.py \
    --models-dir models/final_models/method1_pca5_silhouette0314 \
    --song-id "ID" \
    --top-n 10
```

## 📊 Datos y Formato

### Entrada Esperada
- **Formato**: CSV con separador `;` y decimal `,` (formato español)
- **Encoding**: UTF-8
- **Características musicales requeridas** (13):
  - `danceability`, `energy`, `key`, `loudness`, `mode`
  - `speechiness`, `acousticness`, `instrumentalness`
  - `liveness`, `valence`, `tempo`, `duration_ms`, `time_signature`

### Salida Generada
- **clustering_results.csv**: Dataset original + columna `cluster`
- **clustering_metrics.json**: Métricas detalladas de evaluación
- **Modelos entrenados**: `scaler.pkl`, `kmeans_k3.pkl`, `pca_Xcomp.pkl`

## 🔧 Configuración Avanzada

### Parámetros de Clustering Optimizado
```bash
python algorithms/clustering_optimized.py \
    --dataset ../data/picked_data_0.csv \
    --k-range 3 15 \
    --algorithm minibatch \
    --batch-size 2000 \
    --save-models \
    --save-plots
```

### Métricas de Similitud
- `--similarity-metric cosine`: Similitud coseno (0-1)
- `--similarity-metric euclidean`: Distancia euclidiana invertida
- `--similarity-metric manhattan`: Distancia Manhattan invertida (por defecto)

## 📈 Métricas de Evaluación

### Silhouette Score
- **Rango**: [-1, 1]
- **Interpretación**: 
  - > 0.7: Excelente
  - > 0.5: Bueno  
  - > 0.25: Aceptable
  - < 0.25: Mejorable

### Calinski-Harabasz Score
- **Rango**: [0, ∞)
- **Interpretación**: Mayor es mejor (separación inter-cluster vs intra-cluster)

### Davies-Bouldin Score
- **Rango**: [0, ∞)
- **Interpretación**: Menor es mejor (compacidad de clusters)

## 🔍 Análisis de Resultados

### Distribución de Clusters (Method 1)
- **Cluster 0**: 866 canciones (género/estilo específico)
- **Cluster 1**: 3,648 canciones (mainstream/popular)
- **Cluster 2**: 5,163 canciones (diverso/experimental)

### Características Distintivas
Cada cluster presenta patrones únicos en:
- **Energía y Danceability**: Clusters de alta/baja energía
- **Acousticness vs Instrumentalness**: Música acústica vs electrónica
- **Valence**: Canciones positivas vs melancólicas
- **Tempo**: Ritmos lentos, medios y rápidos

## 🧪 Pruebas y Validación

### Ejecutar Pruebas
```bash
python tests/test_compatibility.py
```

### Validación de Modelos
Los modelos han sido validation con:
- **Dataset**: 9,677 canciones representativas
- **Métricas cruzadas**: Silhouette, Calinski-Harabasz, Davies-Bouldin
- **Tiempo de entrenamiento**: 3-6 segundos por método
- **Reproducibilidad**: Semilla aleatoria fija (random_state=42)

## 🎵 Casos de Uso

### 1. Sistema de Recomendación Musical
- Encontrar canciones similares por características musicales
- Descubrimiento de música basado en clusters
- Análisis de preferencias musicales

### 2. Análisis Musical
- Segmentación automática de géneros musicales
- Identificación de patrones en características de audio
- Clustering de playlists y colecciones

### 3. Investigación Académica
- Estudio de patrones musicales a gran escala
- Análisis de evolución musical temporal
- Clasificación automática de géneros

## 🛠️ Dependencias

### Librerías Principales
- **pandas** >= 1.5.0: Manipulación de datos
- **numpy** >= 1.20.0: Computación numérica
- **scikit-learn** >= 1.0.0: Algoritmos de machine learning
- **matplotlib** >= 3.5.0: Visualización
- **seaborn** >= 0.11.0: Visualización estadística

### Librerías Auxiliares
- **tqdm** >= 4.64.0: Barras de progreso
- **joblib** >= 1.1.0: Persistencia de modelos

## 📚 Referencias Técnicas

### Algoritmos Implementados
- **K-Means**: Clustering particional con optimización Lloyd
- **PCA**: Análisis de componentes principales para reducción dimensional
- **StandardScaler**: Normalización Z-score para características

### Métricas de Evaluación
- **Silhouette Analysis**: Calidad de separación de clusters
- **Elbow Method**: Determinación automática de K óptimo
- **Multiple Metrics**: Criterio combinado para selección de modelos

## 🤝 Contribución

Para contribuir al proyecto:
1. Seguir la estructura de carpetas establecida
2. Documentar nuevas funcionalidades
3. Incluir pruebas para nuevos algoritmos
4. Mantener compatibilidad con el formato de datos establecido

## 📄 Licencia

Este módulo es parte del sistema de recomendación musical multimodal para investigación académica en análisis de características musicales.

---

**Estado**: ✅ VALIDADO Y LISTO PARA PRODUCCIÓN  
**Última actualización**: 2025-01-30  
**Dataset**: 9,677 canciones representativas  
**Mejor modelo**: Method 1 (PCA 5 componentes, Silhouette 0.314)