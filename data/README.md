# DATA - Estructura Jerárquica de Datasets

Este directorio organiza los datasets del proyecto según su nivel de madurez y procesamiento, desde datos crudos (nivel 0) hasta datasets production-ready (nivel 3).

---

## Estructura de Carpetas

```
data/
├── 0_raw/              # Nivel 0: Datos originales sin procesar (Spotify API)
├── 1_cleaned/          # Nivel 1: Datos con formato corregido
├── 2_with_lyrics/      # Nivel 2: Datos con letras (Kaggle - fuente independiente)
├── 3_selected/         # Nivel 3: Datasets seleccionados (production-ready)
└── auxiliary/          # Archivos auxiliares (cache, bases de datos)
```

---

## Origen de los Datos

Este proyecto utiliza **dos fuentes de datos independientes** que se complementan:

```
┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐
│   FUENTE 1: Spotify API             │    │   FUENTE 2: Kaggle                  │
│   (Million Playlist Dataset)        │    │   (Dataset enriquecido con letras)  │
├─────────────────────────────────────┤    ├─────────────────────────────────────┤
│ • 1,204,025 canciones               │    │ • 18,454 canciones                  │
│ • 24 columnas técnicas              │    │ • 25 columnas (incluye lyrics)      │
│ • Sin letras                        │    │ • Letras de Genius.com              │
│ • Máxima diversidad musical         │    │ • Metadatos de playlist             │
├─────────────────────────────────────┤    ├─────────────────────────────────────┤
│         0_raw/ y 1_cleaned/         │    │         2_with_lyrics/              │
└─────────────────────────────────────┘    └─────────────────────────────────────┘
                                                        │
                                                        ▼
                                                  3_selected/
                                            (datasets production-ready)
```

**Intersección entre fuentes**: Solo 2,367 canciones (12.8%) aparecen en ambos datasets, ya que Kaggle utilizó un subset diferente del universo Spotify para crear su dataset enriquecido.

---

## 0_raw/ - Datos Originales (Fuente 1: Spotify API)

Dataset en su estado original, directamente del Spotify Million Playlist Dataset.

| Archivo | Registros | Tamaño | Separador | Encoding | Line Endings |
|---------|-----------|--------|-----------|----------|--------------|
| `tracks_features.csv` | 1,204,025 | 330MB | `,` (coma) | UTF-8 | LF (Unix) |

**Columnas (24)**:
```
id, name, album, album_id, artists, artist_ids, track_number,
disc_number, explicit, danceability, energy, key, loudness, mode,
speechiness, acousticness, instrumentalness, liveness, valence,
tempo, duration_ms, time_signature, year, release_date
```

**Origen**: Spotify API - Million Playlist Dataset
**Fecha**: Diciembre 2020

**Comando de carga**:
```python
import pandas as pd
df = pd.read_csv('data/0_raw/tracks_features.csv', sep=',', encoding='utf-8')
```

---

## 1_cleaned/ - Datos con Formato Corregido (Fuente 1)

Datasets derivados de `0_raw/` con separador modificado de `,` a `;` para evitar conflictos con comas internas en campos de texto.

| Archivo | Registros | Tamaño | Separador | Encoding | Line Endings | Propósito |
|---------|-----------|--------|-----------|----------|--------------|-----------|
| `tracks_features_clean.csv` | 1,204,025 | 317MB | `;` | ASCII | CRLF (Windows) | Dataset completo |
| `tracks_features_500.csv` | 500 | 120KB | `;` | ASCII | CRLF (Windows) | Testing rápido |

**Columnas (24)**: Idénticas a `0_raw/tracks_features.csv`

**Transformación aplicada**:
- Cambio de separador: `,` → `;` (para evitar conflictos con comas en campos de texto)
- Conversión de line endings: LF → CRLF
- **Nota**: No se eliminaron registros. Ambos datasets mantienen 1,204,025 filas.

**Comando de carga**:
```python
import pandas as pd
df = pd.read_csv('data/1_cleaned/tracks_features_clean.csv', sep=';', encoding='utf-8')
df_sample = pd.read_csv('data/1_cleaned/tracks_features_500.csv', sep=';', encoding='utf-8')
```

**Relación entre archivos**:
```
0_raw/tracks_features.csv (1.2M, sep=',')
    │
    └──> [cambio separador] ──> 1_cleaned/tracks_features_clean.csv (1.2M, sep=';')
                                    │
                                    └──> [muestreo] ──> 1_cleaned/tracks_features_500.csv (500, sep=';')
```

---

## 2_with_lyrics/ - Datos con Letras (Fuente 2: Kaggle)

Dataset independiente obtenido de Kaggle, donde la comunidad enriqueció un subset de canciones de Spotify con letras extraídas de Genius.com y metadatos adicionales de playlist.

| Archivo | Registros | Tamaño | Separador | Encoding | Line Endings |
|---------|-----------|--------|-----------|----------|--------------|
| `spotify_songs.csv` | 18,454 | 42MB | `,` | UTF-8 | LF (Unix) |
| `spotify_songs_fixed.csv` | 18,454 | 43MB | `@@` | UTF-8 | CRLF (Windows) |

**Columnas (25)**:
```
track_id, track_name, track_artist, lyrics, track_popularity,
track_album_id, track_album_name, track_album_release_date,
playlist_name, playlist_id, playlist_genre, playlist_subgenre,
danceability, energy, key, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo,
duration_ms, language
```

### Comparación de Columnas: Fuente 1 vs Fuente 2

| Atributo | Fuente 1 (0_raw, 1_cleaned) | Fuente 2 (2_with_lyrics) |
|----------|----------------------------|--------------------------|
| ID de track | `id` | `track_id` |
| Nombre | `name` | `track_name` |
| Artista | `artists` | `track_artist` |
| Album | `album` | `track_album_name` |
| Album ID | `album_id` | `track_album_id` |
| Fecha | `release_date` | `track_album_release_date` |
| **Letras** | - | `lyrics` |
| **Idioma** | - | `language` |
| **Popularidad** | - | `track_popularity` |
| **Playlist info** | - | `playlist_name`, `playlist_id`, `playlist_genre`, `playlist_subgenre` |
| IDs de artistas | `artist_ids` | - |
| Numero de track | `track_number` | - |
| Numero de disco | `disc_number` | - |
| Explícito | `explicit` | - |
| Time signature | `time_signature` | - |
| Año | `year` | - |
| **Audio features** | 12 features | 12 features (idénticas) |

**Audio features compartidas (12)**: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`

**Métricas de calidad**:
- Hopkins Statistic: 0.823 (EXCELENTE para clustering)
- Letras pre-verificadas de Genius.com

**Comando de carga**:
```python
import pandas as pd
# Versión con separador corregido (recomendada)
df = pd.read_csv('data/2_with_lyrics/spotify_songs_fixed.csv', sep='@@', engine='python', encoding='utf-8')

# Versión original (puede tener problemas con comas en letras)
df = pd.read_csv('data/2_with_lyrics/spotify_songs.csv', sep=',', encoding='utf-8')
```

**Nota**: Este dataset es la fuente principal para el proyecto ya que contiene tanto datos musicales como letras, permitiendo análisis multimodal.

---

## 3_selected/ - Datasets Seleccionados (Production-Ready)

Datasets finales optimizados para clustering y análisis multimodal. Generados mediante pipelines de selección desde `2_with_lyrics/`.

| Archivo | Registros | Tamaño | Separador | Hopkins | Silhouette | Estado |
|---------|-----------|--------|-----------|---------|------------|--------|
| `picked_data_optimal.csv` | 10,000 | 22MB | `^` | 0.823 | 0.289 | **PRODUCTION** |
| `picked_data_lyrics.csv` | 9,987 | 23MB | `^` | ~0.45 | 0.177 | LEGACY (problemático) |
| `picked_data_0.csv` | 9,677 | 2.6MB | `;` | ~0.75 | 0.314 | LEGACY (baseline) |

### Dataset Recomendado: `picked_data_optimal.csv`

```python
import pandas as pd
df = pd.read_csv('data/3_selected/picked_data_optimal.csv', sep='^', encoding='utf-8')
```

**Documentación adicional**:
- `README.md` - Especificaciones técnicas detalladas
- `PICKED_DATA_LYRICS_ANALYSIS.md` - Análisis del dataset lyrics

---

## auxiliary/ - Archivos Auxiliares

Archivos de soporte para el sistema de extracción de letras.

| Archivo | Tamaño | Descripción |
|---------|--------|-------------|
| `lyrics.db` | 152KB | Base de datos SQLite con letras extraídas |
| `lyrics_availability_cache.json` | 460KB | Cache de disponibilidad de letras en Genius |

---

## Flujo de Datos del Proyecto

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FUENTES DE DATOS                                │
└─────────────────────────────────────────────────────────────────────────────┘

    FUENTE 1: Spotify API                    FUENTE 2: Kaggle
    (sin letras, 1.2M canciones)             (con letras, 18K canciones)
              │                                        │
              ▼                                        ▼
    0_raw/tracks_features.csv              2_with_lyrics/spotify_songs.csv
              │                                        │
              ▼                                        ▼
    1_cleaned/tracks_features_clean.csv    2_with_lyrics/spotify_songs_fixed.csv
              │                                        │
              ▼                                        │
    1_cleaned/tracks_features_500.csv                  │
    (testing)                                          │
                                                       │
                      ┌────────────────────────────────┘
                      │
                      ▼
              [Pipeline de selección clustering-aware]
                      │
                      ▼
              3_selected/picked_data_optimal.csv
              (10K canciones, production-ready)
```

**Decisión de diseño**: Se eligió la Fuente 2 (Kaggle) como base para el sistema de producción porque:
1. Incluye letras necesarias para análisis multimodal
2. Tiene Hopkins Statistic excelente (0.823) para clustering
3. Tamaño manejable para experimentación (18K vs 1.2M)

---

## Resumen de Separadores por Nivel

| Nivel | Carpeta | Separador | Razón |
|-------|---------|-----------|-------|
| 0 | `0_raw/` | `,` | Original de Spotify API |
| 1 | `1_cleaned/` | `;` | Evitar conflictos con comas en texto |
| 2 | `2_with_lyrics/` | `,` / `@@` | Original Kaggle / Corregido para letras |
| 3 | `3_selected/` | `^` | Carácter ASCII poco común, máxima compatibilidad |

---

## Notas Importantes

1. **Fuentes independientes**: Los niveles 0-1 y nivel 2 provienen de fuentes diferentes. Solo 12.8% de los IDs son comunes.
2. **Separadores**: Cada nivel usa diferentes separadores. Verificar siempre antes de cargar.
3. **Dataset problemático**: `picked_data_lyrics.csv` tiene problemas de clustering documentados (Hopkins ~0.45). Usar `picked_data_optimal.csv` en su lugar.
4. **Encoding**: Todos los archivos usan UTF-8 o ASCII compatible.
5. **Testing**: Usar `tracks_features_500.csv` para desarrollo y testing rápido de algoritmos.
6. **Production**: El flujo de producción usa exclusivamente la rama de Kaggle (`2_with_lyrics/` → `3_selected/`).

---

**Última actualización**: Diciembre 2025
