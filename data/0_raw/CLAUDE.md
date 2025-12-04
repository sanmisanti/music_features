# 0_raw - Dataset Original Sin Procesar

## Archivo Principal

**tracks_features.csv** - Dataset original del Spotify Million Playlist Dataset

## Especificaciones

| Atributo | Valor |
|----------|-------|
| Registros | 1,204,025 |
| Columnas | 24 |
| Separador | `,` |
| Encoding | UTF-8 |
| Tiene letras | NO |
| Tamano | ~330 MB |

## Columnas

```
id, name, album, album_id, artists, artist_ids, track_number, disc_number,
explicit, danceability, energy, key, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, duration_ms,
time_signature, year, release_date
```

## Carga

```python
import pandas as pd
df = pd.read_csv('data/0_raw/tracks_features.csv', sep=',', encoding='utf-8')
```

## Uso

**NO USAR EN PRODUCCION**

Este dataset es solo para referencia historica. Proviene de la API de Spotify y no contiene letras, lo cual lo hace inadecuado para analisis multimodal.

## Problema Principal

- Sin letras de canciones
- Demasiado grande para procesamiento rapido
- No tiene informacion de generos estructurada

## Dataset Derivado

Este dataset fue limpiado y guardado en `1_cleaned/` con separador `;`.
