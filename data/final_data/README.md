# 📊 DATASET FINAL: CANCIONES REPRESENTATIVAS CON LETRAS

Este documento describe la estructura y especificaciones del dataset final `picked_data_lyrics.csv`, incluyendo el proceso de selección utilizado y las características técnicas del archivo.

## 🎯 Información General

**Archivo**: `picked_data_lyrics.csv`  
**Fecha de generación**: Enero 2025  
**Método de selección**: Pipeline de 3 etapas con diversity sampling  
**Propósito**: Dataset representativo para clustering y análisis multimodal (audio + letras)

---

## 🔄 Proceso de Selección

### Dataset Origen
- **Fuente**: `data/with_lyrics/spotify_songs_fixed.csv`
- **Cantidad inicial**: 18,454 canciones con letras verificadas
- **Características**: 25 columnas con metadatos de Spotify y letras completas

### Pipeline de Selección (3 Etapas)

#### **Etapa 1: Diversity Sampling**
- **Algoritmo**: MaxMin diversity sampling
- **Objetivo**: Máxima cobertura del espacio musical 12-dimensional
- **Criterio**: Maximizar distancias euclidianas entre canciones seleccionadas
- **Resultado**: Selección de canciones que representan toda la diversidad musical

#### **Etapa 2: Quality Filtering**
- **Criterios**: Completitud de datos, rangos válidos, popularidad moderada
- **Objetivo**: Eliminar canciones con datos inconsistentes o incompletos
- **Método**: Sistema de scoring multi-criterio

#### **Etapa 3: Stratified Sampling**
- **Método**: Balanced sampling por tonalidad (`key`) y modalidad (`mode`)
- **Objetivo**: Preservar distribuciones estadísticas del dataset original
- **Resultado**: Selección final de exactamente 9,987 canciones

---

## 💾 Especificaciones Técnicas del Archivo

### Formato General
- **Separador de columnas**: `^` (caret - ASCII 94)
- **Encoding**: UTF-8
- **Formato**: CSV con headers
- **Filas totales**: 9,988 (9,987 datos + 1 header)
- **Tamaño aproximado**: 15-20 MB

### Estructura del Archivo
```
track_id^track_name^track_artist^lyrics^track_popularity^track_album_id^...
```

### Comando de Carga
```python
import pandas as pd
df = pd.read_csv('data/final_data/picked_data_lyrics.csv', sep='^', encoding='utf-8')
```

---

## 📋 Estructura de Columnas

### Metadatos de Identificación
| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| `track_id` | str | ID único de Spotify | "4iV5W9uYEdYUVa79Axb7Rh" |
| `track_name` | str | Nombre de la canción | "Bohemian Rhapsody" |
| `track_artist` | str | Nombre del artista | "Queen" |

### Contenido Textual
| Columna | Tipo | Descripción | Características |
|---------|------|-------------|-----------------|
| `lyrics` | str | Letras completas de la canción | Texto multilinea, puede contener caracteres especiales |

### Metadatos Adicionales
| Columna | Tipo | Descripción | Rango/Formato |
|---------|------|-------------|---------------|
| `track_popularity` | int | Popularidad en Spotify | 0-100 |
| `track_album_id` | str | ID del álbum | ID de Spotify |
| `track_album_name` | str | Nombre del álbum | Texto |
| `track_album_release_date` | str | Fecha de lanzamiento | YYYY-MM-DD |
| `playlist_name` | str | Nombre de playlist origen | Texto |
| `playlist_id` | str | ID de playlist | ID de Spotify |
| `playlist_genre` | str | Género principal | "rock", "pop", "hip-hop", etc. |
| `playlist_subgenre` | str | Subgénero específico | "classic rock", "indie pop", etc. |
| `language` | str | Idioma de la canción | Código ISO ("en", "es", "fr", etc.) |

### Características Musicales de Spotify (Audio Features)
| Columna | Tipo | Descripción | Rango |
|---------|------|-------------|-------|
| `danceability` | float | Qué tan bailable es la canción | 0.0 - 1.0 |
| `energy` | float | Intensidad y actividad percibida | 0.0 - 1.0 |
| `key` | int | Tonalidad musical | 0-11 (C, C#, D, ..., B) |
| `loudness` | float | Volumen general en decibelios | -60.0 - 0.0 |
| `mode` | int | Modalidad musical | 0 (menor), 1 (mayor) |
| `speechiness` | float | Presencia de palabras habladas | 0.0 - 1.0 |
| `acousticness` | float | Medida de acústico vs eléctrico | 0.0 - 1.0 |
| `instrumentalness` | float | Probabilidad de ser instrumental | 0.0 - 1.0 |
| `liveness` | float | Presencia de audiencia en vivo | 0.0 - 1.0 |
| `valence` | float | Positividad musical | 0.0 - 1.0 |
| `tempo` | float | Tempo en BPM | ~50-250 BPM |
| `duration_ms` | int | Duración en milisegundos | Milisegundos |
| `time_signature` | int | Compás musical | 3, 4, 5, etc. (predomina 4) |

---

## 🎵 Características Clave del Dataset

### Features para Clustering
Las siguientes 12 columnas forman el espacio de características para clustering:
```python
clustering_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms'
]
```

### Campos de Texto para NLP
- **`lyrics`**: Contenido principal para análisis semántico
- **`track_name`**: Útil para análisis de títulos
- **`track_artist`**: Para análisis por artista
- **`playlist_genre`/`playlist_subgenre`**: Para análisis por género

### Metadatos Contextuales
- **`track_popularity`**: Para análisis de mainstream vs underground
- **`track_album_release_date`**: Para análisis temporal
- **`language`**: Para análisis multiidioma

---

## 🔧 Consideraciones de Implementación

### Carga del Dataset
```python
import pandas as pd

# Carga básica
df = pd.read_csv('data/final_data/picked_data_lyrics.csv', sep='^', encoding='utf-8')

# Verificación de estructura
print(f"Shape: {df.shape}")  # Debería ser (9987, 25)
print(f"Columnas: {len(df.columns)}")  # Debería ser 25
```

### Manejo de Tipos de Datos
```python
# Conversiones recomendadas
df['track_popularity'] = df['track_popularity'].astype(int)
df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'])

# Features numéricas para clustering
numeric_features = df[clustering_features].select_dtypes(include=[np.number])
```

### Separador de Columnas
**⚠️ Importante**: El separador `^` fue elegido específicamente porque:
- No aparece en las letras de canciones
- No interfiere con puntuación musical común (comas, puntos, comillas)
- Es compatible con CSV estándar
- Evita conflictos con caracteres especiales en títulos/artistas

---

## 🎯 Casos de Uso Recomendados

### 1. Análisis de Clustering
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Preparar features
features = df[clustering_features]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicar clustering
kmeans = KMeans(n_clusters=7, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
df['cluster'] = clusters
```

### 2. Análisis de Letras
```python
# Filtrar canciones con letras
songs_with_lyrics = df[df['lyrics'].notna()]

# Análisis básico de texto
songs_with_lyrics['lyrics_length'] = songs_with_lyrics['lyrics'].str.len()
songs_with_lyrics['word_count'] = songs_with_lyrics['lyrics'].str.split().str.len()
```

### 3. Análisis por Género
```python
# Distribución por géneros
genre_counts = df['playlist_genre'].value_counts()

# Features promedio por género
genre_features = df.groupby('playlist_genre')[clustering_features].mean()
```

---

## 📝 Notas Técnicas

### Integridad de Datos
- Todas las filas contienen letras completas verificadas
- Los metadatos provienen directamente de Spotify API
- El proceso de selección garantiza representatividad musical

### Reproducibilidad
- Pipeline determinístico con seeds fijos
- Proceso completamente documentado en `PROCESO_SELECCION.md`
- Scripts de generación disponibles en `scripts/select_from_lyrics_dataset.py`

### Escalabilidad
- Tamaño optimizado para análisis en memoria (~10K canciones)
- Estructura compatible con bibliotecas estándar (pandas, scikit-learn)
- Formato eficiente para carga rápida

---

*Dataset generado para el Sistema de Recomendación Musical Multimodal*  
*Estructura optimizada para clustering y análisis híbrido audio-texto*