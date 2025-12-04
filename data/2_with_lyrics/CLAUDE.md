# 2_with_lyrics - Dataset Kaggle con Letras (FUENTE PRINCIPAL)

## Archivo Principal

**spotify_songs_fixed.csv** - 18,454 canciones con letras de Genius.com

## Especificaciones

| Atributo | Valor |
|----------|-------|
| Registros | 18,454 |
| Columnas | 25 |
| Separador | `@@` |
| Engine | python (requerido) |
| Encoding | UTF-8 |
| Hopkins Statistic | 0.823 normalizado / 0.788 raw (EXCELENTE) |
| Tiene letras | SI |
| Tamano | ~43 MB |

## Carga

```python
import pandas as pd
df = pd.read_csv('data/2_with_lyrics/spotify_songs_fixed.csv',
                 sep='@@',
                 engine='python',
                 encoding='utf-8')
```

## Columnas Principales

### Metadatos
- `track_id` - ID unico Spotify
- `track_name` - Nombre de la cancion
- `track_artist` - Artista
- `lyrics` - Letra completa
- `track_popularity` - Popularidad (0-100)
- `playlist_genre` - Genero (rock, pop, rap, latin, r&b, edm)
- `playlist_subgenre` - Subgenero
- `language` - Idioma detectado

### Audio Features (12)
```python
features = ['danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_ms']
```

## Importancia

**FUENTE PRINCIPAL** para todos los datasets derivados del proyecto:
- `3_selected/picked_data_optimal.csv` (10,000 canciones)
- `4_vectorized/` (embeddings BERT)
- `5_unified/` (dataset multimodal final)

## Archivos en esta Carpeta

| Archivo | Estado | Descripcion |
|---------|--------|-------------|
| spotify_songs_fixed.csv | USAR | Dataset corregido con separador @@ |
| spotify_songs.csv | NO USAR | Version original con problemas de formato |

## Metricas de Calidad

- **Hopkins Statistic**: 0.823 - Indica excelente clusterabilidad
- **Distribucion de generos**: 6 generos balanceados
- **Cobertura de letras**: 100% de registros tienen letras

## Origen

Dataset de Kaggle enriquecido con letras de Genius.com. Independiente del dataset de Spotify API (0_raw/).
