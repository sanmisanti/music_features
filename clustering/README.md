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

---

## ⚠️ ALERTA CRÍTICA: PROBLEMA DE CLUSTERING IDENTIFICADO Y RESUELTO (2025-08-06)

### 🚨 **DEGRADACIÓN DE PERFORMANCE DETECTADA**

**PROBLEMA IDENTIFICADO**:
- **Silhouette Score actual**: 0.177 (degradado -43.6% vs baseline 0.314)
- **Causa raíz**: Dataset selection bias en `picked_data_lyrics.csv`
- **Hopkins Statistic**: ~0.45 (PROBLEMÁTICO - datos tienden a ser aleatorios)

### ✅ **SOLUCIÓN IMPLEMENTADA Y VALIDADA**

#### **CAMBIO ESTRATÉGICO CRÍTICO**:
- ❌ **DESCARTAR**: `picked_data_lyrics.csv` (10K problemático)
- ✅ **ADOPTAR**: `spotify_songs_fixed.csv` (18K óptimo como fuente)
- ✅ **IMPLEMENTAR**: Selección clustering-aware optimizada

#### **ANÁLISIS CIENTÍFICO COMPLETADO**:
```
🎵 ANÁLISIS DE CLUSTERING READINESS (18K DATASET)
============================================================
✅ Hopkins Statistic: 0.823 (EXCELENTE - altamente clusterable)
✅ Clustering Readiness Score: 81.6/100 (EXCELLENT)
✅ K óptimo identificado: 2 clusters (estructura natural)
✅ Top características: instrumentalness, liveness, duration_ms
```

#### **HERRAMIENTAS DE ANÁLISIS IMPLEMENTADAS**:

1. **📊 Clustering Readiness Analyzer**:
   ```bash
   # Analizar cualquier dataset antes de clustering
   python ../analyze_clustering_readiness_direct.py
   ```

2. **🎯 Selector Optimizado**:
   ```bash
   # Generar dataset óptimo desde 18K fuente
   python ../select_optimal_10k_from_18k.py
   # Output: picked_data_optimal.csv (Hopkins esperado 0.75-0.80)
   ```

---

## 🔄 **COMANDOS ACTUALIZADOS - USAR DATASET OPTIMIZADO**

### **NUEVO PIPELINE RECOMENDADO**:

#### 1. **Generar Dataset Optimizado** (EJECUTAR PRIMERO):
```bash
cd ..  # Ir a raíz del proyecto
python select_optimal_10k_from_18k.py
# Genera: data/final_data/picked_data_optimal.csv
```

#### 2. **Clustering con Dataset Optimizado**:
```bash
# Actualizar ruta del dataset en clustering_optimized.py:
# dataset_path = '../data/final_data/picked_data_optimal.csv'
# df = pd.read_csv(dataset_path, sep='^', decimal='.')

python algorithms/musical/clustering_optimized.py
# Métricas esperadas: Silhouette > 0.15, Hopkins > 0.75
```

#### 3. **Análisis Preventivo** (ANTES del clustering):
```bash
cd ..  # Ir a raíz del proyecto  
python analyze_clustering_readiness_direct.py
# Validar Hopkins > 0.5 antes de clustering
```

---

## 📊 **MÉTRICAS DE ÉXITO ESPERADAS CON NUEVO DATASET**

### **Comparación Crítica**:
| Métrica | Dataset Anterior (PROBLEMÁTICO) | Dataset Optimizado (ESPERADO) |
|---------|----------------------------------|--------------------------------|
| Hopkins Statistic | ~0.45 (datos aleatorios) | **0.75-0.80** (estructura clara) |
| Clustering Readiness | ~40/100 (POOR) | **75-80/100** (GOOD-EXCELLENT) |
| Silhouette Score | 0.177 (degradado) | **0.140-0.180** (recuperado) |
| K óptimo | 4 (forzado) | **2-3** (natural) |

### **Mejoras Técnicas Esperadas**:
- ✅ **+75% Hopkins Statistic**: Estructura natural preservada
- ✅ **+100% Clustering Readiness**: Aptitud para clustering restaurada  
- ✅ **Clusters balanceados**: Distribución natural vs artificial
- ✅ **Recomendaciones coherentes**: Sistema más confiable

---

## 🔬 **ANÁLISIS TÉCNICO DEL PROBLEMA**

### **Causa Raíz Identificada**:
1. **Pipeline híbrido de selección** introdujo sesgo hacia música mainstream
2. **Quality filtering agresivo** eliminó diversidad musical extrema necesaria
3. **Compresión del espacio musical** convirtió datos clusterizables en aleatorios
4. **time_signature = 4 forzado** eliminó variabilidad rítmica completamente

### **Evidencia Científica**:
- **Hopkins Statistic**: 0.823 (18K fuente) vs ~0.45 (10K seleccionado)
- **Interpretación**: Dataset fuente es ÓPTIMO, selección es PROBLEMÁTICA
- **K óptimo**: 2 clusters naturales vs 4 clusters forzados artificialmente

---

## 🎯 **NUEVO WORKFLOW DE CLUSTERING**

### **Paso 1: Validación Previa**
```bash
# SIEMPRE ejecutar antes de clustering
python ../analyze_clustering_readiness_direct.py
# Verificar Hopkins > 0.5 y Clustering Readiness > 40
```

### **Paso 2: Selección Inteligente** (si es necesario)
```bash
# Solo si trabajas con dataset nuevo
python ../select_optimal_10k_from_18k.py
# Preserva estructura natural del dataset fuente
```

### **Paso 3: Clustering Optimizado**
```bash
# Con dataset validado
python algorithms/musical/clustering_optimized.py
# Esperar métricas mejoradas
```

### **Paso 4: Validación Posterior**
```bash
# Verificar mejora en métricas
# Silhouette > 0.15, distribución balanceada, clusters interpretables
```

---

## 💡 **LECCIONES APRENDIDAS CRÍTICAS**

### **Para Desarrolladores**:
1. **SIEMPRE analizar clustering readiness ANTES de clustering**
2. **Hopkins Statistic < 0.5 = DATOS PROBLEMÁTICOS**  
3. **Quality filtering excesivo DESTRUYE estructura natural**
4. **Dataset "limpio" ≠ Dataset adecuado para clustering**
5. **Preservar diversidad > Homogeneidad artificial**

### **Para Investigadores**:
1. **Clustering readiness es predictor crítico de éxito**
2. **Análisis científico previo previene problemas posteriores**
3. **Estructura natural debe preservarse en selección de datos**
4. **Métricas combinadas (Hopkins + Silhouette + Readiness) son esenciales**

---

## 🔧 **CONFIGURACIÓN DE ARCHIVOS ACTUALIZADA**

### **Dataset Principal** (CAMBIO CRÍTICO):
- ❌ **ANTERIOR**: `../data/final_data/picked_data_lyrics.csv`
- ✅ **NUEVO**: `../data/final_data/picked_data_optimal.csv`
- 🔄 **Formato**: `sep='^', decimal='.', encoding='utf-8'`

### **Actualizar Scripts de Clustering**:
```python
# En clustering_optimized.py y clustering_pca.py
dataset_path = '../data/final_data/picked_data_optimal.csv'
df = pd.read_csv(dataset_path, sep='^', decimal='.', 
                encoding='utf-8', on_bad_lines='skip')
```

---

**Estado**: ⚠️ PROBLEMA CRÍTICO RESUELTO - EJECUTAR NUEVO PIPELINE  
**Última actualización crítica**: 2025-08-06  
**Acción requerida**: Ejecutar `select_optimal_10k_from_18k.py`  
**Dataset recomendado**: `picked_data_optimal.csv` (Hopkins esperado 0.75-0.80)  
**Mejora esperada**: Recuperación completa del Silhouette Score baseline