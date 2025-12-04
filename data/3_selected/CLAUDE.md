# 3_selected - Dataset Musical de Produccion

## Archivo Principal

**picked_data_optimal.csv** - 10,000 canciones seleccionadas para analisis musical

## Carga Rapida

```python
import pandas as pd
df = pd.read_csv('data/3_selected/picked_data_optimal.csv', sep='^', encoding='utf-8')
```

## Especificaciones

| Atributo | Valor |
|----------|-------|
| Registros | 10,000 |
| Columnas | 26 |
| Separador | `^` (caret) |
| Encoding | UTF-8 |
| Hopkins Statistic | 0.823 (EXCELENTE) |
| Silhouette Score | 0.289 (+86% vs baseline) |
| Tamano | ~22 MB |

## Columnas Principales

### Metadatos
- `track_id` - ID unico Spotify
- `track_name` - Nombre de la cancion
- `track_artist` - Artista
- `lyrics` - Letra completa
- `playlist_genre` - Genero (rock, pop, rap, latin, r&b, edm)
- `playlist_subgenre` - Subgenero
- `natural_cluster` - Cluster pre-asignado

### Audio Features (12 caracteristicas)
```python
features = ['danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_ms']
```

## Origen

| Atributo | Valor |
|----------|-------|
| Fuente | `2_with_lyrics/spotify_songs_fixed.csv` (18,454 canciones) |
| Seleccion | 54.2% del dataset original |
| Metodo | Clustering-aware con validacion Hopkins |
| Generador | `scripts/generation/generate_optimal_dataset.py` |
| Fecha | 2025-08-12 |

## Archivos en esta Carpeta

| Archivo | Proposito |
|---------|-----------|
| picked_data_optimal.csv | Dataset PRODUCTION |
| optimization_report_20250812_185734.json | Metricas de generacion |
| CLAUDE.md | Este archivo |

## Datasets Derivados

Este dataset es la fuente para:

1. **`4_vectorized/`** - Embeddings BERT de las letras
   - 9,753 letras procesadas -> 8,567 embeddings validos
   - Perdida: 1,433 canciones (14.3%)

2. **`5_unified/`** - Dataset multimodal final
   - 7,811 canciones con embeddings + features musicales
   - Perdida adicional: 756 canciones (8.8%)

## Metricas de Calidad

### Hopkins Statistic: 0.823 (normalizado) / 0.788 (raw)
- Interpretacion: EXCELENTE clusterabilidad
- Rango: >0.75 indica estructura natural de clusters
- Nota: El valor varia segun normalizacion de features

### Silhouette Score: 0.289
- Mejora: +86.1% vs baseline (0.1554)
- Obtenido con: Hierarchical Clustering, K=3

### Diversidad Musical
- Promedio: 1.109 (superior al original)
- Caracteristica mas discriminativa: instrumentalness (ratio 1.33)

## Directivas de Uso

1. **Para analisis SOLO musical**: Usar este dataset directamente
2. **Para analisis multimodal**: Usar `5_unified/unified_multimodal_7811.pkl`
3. **Separador**: SIEMPRE especificar `sep='^'`
4. **12 audio features**: Disponibles para clustering/analisis
5. **Columna lyrics**: Disponible pero usar embeddings de `4_vectorized/` para NLP

## Limitaciones

- No todos los registros tienen embeddings BERT validos
- Para analisis multimodal, usar el dataset unificado de `5_unified/`
- La columna `natural_cluster` es pre-asignada, puede no ser optima para todos los usos
