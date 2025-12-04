# 3_selected - Dataset PRODUCTION

## Archivo Principal

**picked_data_optimal.csv** - El unico dataset que debe usarse en produccion.

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
| Separador | ^ |
| Encoding | UTF-8 |
| Hopkins | 0.823 |
| Silhouette | 0.289 |

## Columnas Importantes

### Metadatos
- `track_id` - ID unico Spotify
- `track_name` - Nombre cancion
- `track_artist` - Artista
- `lyrics` - Letra completa
- `playlist_genre` - Genero (rock, pop, rap, latin, r&b, edm)
- `natural_cluster` - Cluster pre-asignado

### Audio Features (usar para clustering)
```python
features = ['danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_ms']
```

## Origen

- **Fuente**: `2_with_lyrics/spotify_songs_fixed.csv` (18,454 canciones)
- **Generador**: `scripts/generation/generate_optimal_dataset.py`
- **Algoritmo**: Clustering-aware con validacion Hopkins
- **Reporte**: `optimization_report_20250812_185734.json`

## Archivos en esta Carpeta

| Archivo | Proposito |
|---------|-----------|
| picked_data_optimal.csv | Dataset PRODUCTION |
| optimization_report_*.json | Metricas de generacion |
| CLAUDE.md | Este archivo |

## Directivas

1. **Este es el dataset principal** - No usar otros datasets para produccion
2. **Separador ^** - Siempre especificar `sep='^'` al cargar
3. **Hopkins 0.823** - Dataset optimizado para clustering
4. **12 audio features** - Usar para analisis musical
5. **Columna lyrics** - Disponible para analisis semantico/NLP
