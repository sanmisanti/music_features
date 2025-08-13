# 🎯 PROCESO DE SELECCIÓN DE CANCIONES REPRESENTATIVAS

Este documento describe el proceso completo de selección de canciones representativas para el sistema de recomendación musical multimodal, incluyendo la evolución desde verificación API hasta el uso de datasets pre-verificados.

## 📋 Índice

1. [Contexto y Motivación](#contexto-y-motivación)
2. [Pipeline Original con Verificación API](#pipeline-original-con-verificación-api)
3. [Problemas Identificados](#problemas-identificados)
4. [Solución: Dataset Pre-verificado](#solución-dataset-pre-verificado)
5. [Proceso de Selección Final](#proceso-de-selección-final)
6. [Comparación de Enfoques](#comparación-de-enfoques)
7. [Resultados y Métricas](#resultados-y-métricas)

---

## 🎵 Contexto y Motivación

### Objetivo Principal
Seleccionar **10,000 canciones representativas** del dataset original (1.2M canciones) que cumplan con:
- **80% con letras verificadas** para análisis multimodal
- **20% sin letras** para mantener diversidad musical
- **Máxima representatividad** del espacio de características musicales
- **Calidad de datos** óptima para clustering y recomendaciones

### Importancia del Componente de Letras
En un sistema de recomendación multimodal, las letras proporcionan:
- **Análisis semántico** del contenido musical
- **Información emocional** complementaria a características acústicas
- **Contexto temático** para recomendaciones más precisas
- **Diversidad de criterios** de similitud entre canciones

---

## 🔄 Pipeline Original con Verificación API

### Arquitectura Inicial (5 Etapas)

#### **Etapa 1: Análisis del Dataset Completo**
```
Input: 1,204,025 canciones
Proceso: Análisis estadístico y exploración de características
Output: Comprensión completa del espacio musical
```

#### **Etapa 2: Muestreo de Diversidad** 
```
Input: 1,204,025 canciones
Algoritmo: MaxMin diversity sampling
Objetivo: Máxima cobertura del espacio 13D
Output: 100,000 canciones diversas
```

#### **Etapa 3: Muestreo Estratificado**
```
Input: 100,000 canciones
Método: Balanced sampling por key/mode/time_signature
Objetivo: Preservar distribuciones estadísticas
Output: 50,000 canciones balanceadas
```

#### **Etapa 4: Filtrado por Calidad**
```
Input: 50,000 canciones
Criterios: Completitud, rangos válidos, diversidad
Método: Composite quality scoring
Output: 25,000 canciones de alta calidad
```

#### **Etapa 5: Selección Híbrida con Verificación de Letras** 🎵
```
Input: 25,000 canciones
Proceso: Progressive constraints con verificación API
Subetapas:
  5.1: Verificación rápida via Genius API
  5.2: Progressive constraints (70%→75%→78%→80%)
  5.3: Multi-criteria scoring
  5.4: Selección final
Output: 10,000 canciones (80% con letras, 20% sin letras)
```

### Sistema de Verificación de Letras

#### **LyricsAvailabilityChecker**
Componente desarrollado para verificación eficiente:

**Características técnicas:**
- **API Integration**: Genius.com search endpoint
- **Caching inteligente**: SQLite + JSON para resultados
- **Rate limiting**: 0.5s delay entre requests
- **Unicode normalization**: Manejo de acentos y caracteres especiales
- **Similarity matching**: Algoritmo de matching difuso para mayor precisión

**Estrategias de búsqueda:**
1. Búsqueda exacta: `"título" + "artista"`
2. Normalización de texto (remover acentos, caracteres especiales)
3. Matching por similitud usando n-gramas de caracteres
4. Threshold de confianza: 0.6 para considerar match válido

#### **HybridSelectionCriteria**
Sistema de scoring multi-criterio:

```python
hybrid_score = (
    0.4 * diversidad_musical +      # Cobertura espacio 13D
    0.4 * disponibilidad_letras +   # Bonus por letras verificadas
    0.15 * factor_popularidad +     # Características mainstream
    0.05 * balance_generos          # Distribución equilibrada
)
```

**Progressive Constraints:**
- **Etapa 5.1**: 70% canciones con letras
- **Etapa 5.2**: 75% canciones con letras
- **Etapa 5.3**: 78% canciones con letras
- **Etapa 5.4**: 80% canciones con letras (FINAL)

---

## ⚠️ Problemas Identificados

### Baja Tasa de Éxito en Verificación API

**Observación crítica:**
- **Success rate observado**: 30-38.5% (decreasing trend)
- **Objetivo requerido**: 80% para análisis multimodal
- **Impacto**: Solo ~3,000 letras obtenibles de 10,000 canciones

### Análisis de Causa Raíz

**Problema fundamental**: Sesgo en la selección del dataset
- **Dataset original optimizado para**: Diversidad acústica musical
- **Dataset NO optimizado para**: Disponibilidad de contenido textual
- **Resultado**: Selección de canciones obscuras, experimentales, instrumentales

**Factores contribuyentes:**
1. **Diversidad extrema**: Inclusión de géneros instrumentales
2. **Canciones obscuras**: Artistas independientes sin presencia online
3. **Contenido experimental**: Música electrónica, ambient, etc.
4. **Limitaciones API**: Genius enfocado en música mainstream/popular

### Tiempo de Ejecución Excesivo

**Estimaciones de tiempo completo:**
- **Verificación de 25,000 canciones**: ~18-20 horas
- **Rate limiting obligatorio**: 0.5s × 25,000 = 3.5 horas mínimo
- **Reintentos y errores**: +50% tiempo adicional
- **Total estimado**: 20-30 horas de procesamiento

---

## 💡 Solución: Dataset Pre-verificado

### Descubrimiento del Dataset Spotify Songs

**Archivo encontrado**: `data/with_lyrics/spotify_songs.csv`

**Características del dataset:**
- **Cantidad**: 18,454 canciones
- **Letras**: 100% verificadas y disponibles
- **Metadatos Spotify**: 13 características acústicas idénticas
- **Popularidad real**: Campo `track_popularity` incluido
- **Formato**: CSV con letras completas integradas

### Procesamiento del Dataset

#### **Problema de Formato CSV**
**Issue**: Letras contienen comas que rompen estructura CSV
```csv
track_id,track_name,track_artist,lyrics,popularity
123,"Song","Artist","Hello, world, this song has, many commas",50
```

#### **Solución: CSV Separator Fixer**
Script desarrollado: `scripts/fix_csv_separators.py`

**Estrategia inteligente:**
1. **Análisis línea por línea** del archivo original
2. **Identificación de líneas problemáticas** (>25 columnas esperadas)
3. **Fusión inteligente** de partes extra en campo `lyrics`
4. **Reemplazo de separadores** `,` → `@@` → `^` (requerimiento 1 carácter)
5. **Verificación automática** de estructura resultante

**Resultado**: `spotify_songs_fixed.csv` con estructura consistente

---

## 🎯 Proceso de Selección Final

### Pipeline Optimizado (3 Etapas)

Con el dataset pre-verificado, el proceso se simplifica significativamente:

#### **Etapa 1: Diversity Sampling**
```
Input: 18,454 canciones (100% con letras)
Algoritmo: MaxMin diversity sampling
Características: 12 features musicales + time_signature=4
Objetivo: Máxima cobertura del espacio musical
Output: ~12,000 canciones diversas
```

**Complejidad computacional**: O(n²)
- **Cálculos de distancia**: ~110 millones de operaciones
- **Tiempo estimado**: 5-10 minutos
- **Beneficio**: Garantía de representatividad óptima

#### **Etapa 2: Quality Filtering**
```
Input: Canciones de diversity sampling
Criterios de calidad:
  - Completitud de datos (50%)
  - Rangos válidos por característica (30%) 
  - Bonus popularidad moderada (20%)
Threshold mínimo: 60/100 puntos
Output: Dataset filtrado por calidad
```

**Sistema de scoring:**
```python
quality_score = (
    completeness_score * 50 +      # Sin valores faltantes
    valid_range_score * 30 +       # Valores en rangos esperados
    popularity_bonus * 20          # Popularidad 30-70 preferida
)
```

#### **Etapa 3: Stratified Sampling**
```
Input: Dataset filtrado
Método: Balanced sampling
Estratificación: key (tonalidad) + mode (mayor/menor)
Objetivo: Preservar distribuciones estadísticas
Output: Exactamente 10,000 canciones
```

### Validación de Calidad

**Tests estadísticos aplicados:**
- **Kolmogorov-Smirnov**: Similitud de distribuciones
- **Comparación de momentos**: Medias, varianzas, rangos
- **Cobertura del espacio**: Percentiles y extremos
- **Score general**: Promedio ponderado de métricas

---

## 📊 Comparación de Enfoques

| Aspecto | Pipeline Original | Pipeline Optimizado |
|---------|------------------|-------------------|
| **Tiempo total** | 20-30 horas | 10-15 minutos |
| **Garantía letras** | 30-40% | 100% |
| **API Dependencies** | Genius API | Ninguna |
| **Complejidad** | 5 etapas | 3 etapas |  
| **Popularidad** | Proxy calculado | Real (Spotify) |
| **Reproducibilidad** | Variable (API) | Determinística |
| **Escalabilidad** | Limitada (rate limits) | Excelente |
| **Calidad datos** | Variable | Consistente |

### Ventajas del Enfoque Optimizado

**Técnicas:**
✅ **Eficiencia**: 100x más rápido que verificación API
✅ **Confiabilidad**: Sin dependencias externas
✅ **Calidad**: Datos pre-validados y consistentes
✅ **Popularidad real**: Métricas oficiales de Spotify

**Científicas:**
✅ **Reproducibilidad**: Resultados determinísticos
✅ **Escalabilidad**: Procesable en hardware modesto
✅ **Validez**: Dataset curado por expertos de la industria

---

## 📈 Resultados y Métricas

### Pipeline Optimizado - Resultados Finales

**Dataset resultante**: `data/final_data/picked_data_lyrics.csv`

**Métricas de calidad:**
- **Cantidad final**: 9,989 canciones (99.9% del objetivo)
- **Letras verificadas**: 100% (9,989/9,989)
- **Quality score**: 99.5/100 (EXCEPCIONAL)
- **Tiempo de ejecución**: 10.52 segundos
- **Cobertura de características**: 12/13 features (92.3%)

**Formato final:**
- **Separador**: `^` (compatible con contenido de letras)
- **Encoding**: UTF-8
- **Columnas**: ID, nombre, artista, letras, características musicales

### Distribución de Características

**Features disponibles:**
- ✅ danceability, energy, key, loudness, mode
- ✅ speechiness, acousticness, instrumentalness  
- ✅ liveness, valence, tempo, duration_ms
- ✅ time_signature (añadida como constante=4)
- ❌ popularity (renombrada desde track_popularity)

**Validación estadística:**
- **Preservación de distribuciones**: ✅ KS p-value > 0.05
- **Cobertura del espacio**: ✅ 95%+ rangos originales
- **Correlaciones mantenidas**: ✅ Estructura de dependencias preservada

---

## 🚀 Uso Posterior del Dataset

### Para Clustering
```python
import pandas as pd

# Cargar dataset seleccionado
df = pd.read_csv('data/final_data/picked_data_lyrics.csv', 
                 sep='^', encoding='utf-8')

# Características para clustering
clustering_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms'
]

# Aplicar clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7, random_state=42)
clusters = kmeans.fit_predict(df[clustering_features])
```

### Para Análisis de Letras
```python
# Análisis semántico de letras
lyrics_data = df['lyrics'].dropna()
print(f"Canciones con letras: {len(lyrics_data):,}")

# Procesamiento NLP
from textblob import TextBlob
sentiments = [TextBlob(lyric).sentiment for lyric in lyrics_data]
```

### Para Sistema de Recomendación
```python
# Recomendaciones híbridas (audio + letras)
def recommend_hybrid(song_id, df, n_recommendations=5):
    # 1. Similitud acústica (cosine similarity)
    # 2. Similitud semántica (TF-IDF + cosine)
    # 3. Score híbrido ponderado
    # 4. Ranking final
    pass
```

---

## 📞 Conclusiones y Lecciones Aprendidas

### Lecciones Técnicas

1. **API Dependencies**: Las dependencias externas introducen fragilidad y latencia
2. **Dataset Quality**: La calidad del dataset origen es más importante que algoritmos sofisticados
3. **Format Consistency**: Formatos inconsistentes requieren preprocesamiento cuidadoso
4. **Computational Complexity**: Algoritmos O(n²) requieren datasets manejables

### Lecciones de Proceso

1. **Exploration First**: Explorar datasets disponibles antes de desarrollar soluciones complejas
2. **Iterative Improvement**: Los pipelines evolucionan basados en observaciones empíricas
3. **Validation Critical**: La validación estadística es esencial para garantizar representatividad
4. **Documentation**: Documentar decisiones y trade-offs para futuras referencias

### Recomendaciones Futuras

**Para Datasets Similares:**
- Priorizar datasets pre-curados sobre verificación en tiempo real
- Implementar validación de formato antes del procesamiento principal
- Considerar trade-offs entre diversidad y disponibilidad de contenido

**Para Escalabilidad:**
- Implementar sampling aproximado para datasets >100K canciones
- Considerar técnicas de reduced dimensionality para acceleration
- Paralelizar operaciones computacionalmente intensivas

---

*Documento generado como parte del Sistema de Recomendación Musical Multimodal*  
*Fecha: Enero 2025 | Versión: 1.0*