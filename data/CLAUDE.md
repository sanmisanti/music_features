# DATA - Contexto para Claude

## Historia de los Datos

### Fuente 1: Spotify API (sin letras)
Dataset original del Spotify Million Playlist Dataset con 1.2M canciones.
- **Problema**: No contiene letras, solo metadatos y audio features
- **Uso actual**: Solo para testing rapido (500 registros)
- **Ubicacion**: `0_raw/` y `1_cleaned/`

### Fuente 2: Kaggle (con letras)
Dataset reducido pero enriquecido con letras de Genius.com.
- **Ventaja**: Contiene letras completas para analisis multimodal
- **Tamano**: 18,454 canciones (vs 1.2M del original)
- **Hopkins**: 0.823 (excelente para clustering)
- **Ubicacion**: `2_with_lyrics/`

### Interseccion
Solo 12.8% de IDs son comunes entre ambas fuentes. Son datasets independientes.

---

## Estructura de Carpetas

```
data/
├── 0_raw/           -> Spotify API original (1.2M, sep=',') SIN LETRAS
├── 1_cleaned/       -> Formato corregido (sep=';') SIN LETRAS
├── 2_with_lyrics/   -> Kaggle con letras (18K, sep='@@') CON LETRAS
├── 3_selected/      -> PRODUCTION (10K, sep='^') CON LETRAS
└── auxiliary/       -> Cache SQLite y JSON para extraccion de letras
```

---

## Dataset PRODUCTION

**SIEMPRE usar este dataset para cualquier operacion:**

```python
import pandas as pd
df = pd.read_csv('data/3_selected/picked_data_optimal.csv', sep='^', encoding='utf-8')
```

| Atributo | Valor |
|----------|-------|
| Archivo | `data/3_selected/picked_data_optimal.csv` |
| Registros | 10,000 canciones |
| Separador | `^` (caret) |
| Hopkins | 0.823 (excelente) |
| Silhouette | 0.289 (+86% vs baseline) |
| Tiene letras | Si |

---

## Separadores por Carpeta

| Carpeta | Separador | Carga |
|---------|-----------|-------|
| 0_raw/ | `,` | `sep=','` |
| 1_cleaned/ | `;` | `sep=';'` |
| 2_with_lyrics/ | `@@` | `sep='@@', engine='python'` |
| 3_selected/ | `^` | `sep='^'` |

---

## Flujo de Datos

```
FUENTE 1 (sin letras)              FUENTE 2 (con letras)
Spotify API 1.2M                   Kaggle 18K + Genius
       |                                  |
       v                                  v
    0_raw/                         2_with_lyrics/
       |                           spotify_songs_fixed.csv
       v                                  |
    1_cleaned/                            v
    (solo testing)              [Pipeline clustering-aware]
                                          |
                                          v
                                    3_selected/
                               picked_data_optimal.csv
                                    (PRODUCTION)
```

---

## Audio Features para Clustering (12)

```python
clustering_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms'
]
```

---

## Datasets Legacy (NO USAR)

Movidos a `archive/legacy_data/`. NO referenciar en codigo nuevo.

### picked_data_lyrics.csv
- **Que era**: Seleccion de 9,987 canciones con letras
- **Por que se descarto**: Hopkins ~0.45 (muy bajo para clustering)
- **Problema**: Pipeline de seleccion priorizaba letras sobre calidad de clustering
- **Silhouette**: 0.177 (43% peor que baseline)

### picked_data_0.csv
- **Que era**: Seleccion de 9,677 canciones del dataset Spotify API
- **Por que se descarto**: Proviene de Fuente 1 (sin letras)
- **Problema**: No sirve para analisis multimodal (audio + texto)

### PICKED_DATA_LYRICS_ANALYSIS.md
- **Que era**: Documentacion del dataset picked_data_lyrics.csv
- **Por que se descarto**: Documenta dataset obsoleto

---

## Directivas

1. **Dataset principal**: Siempre usar `3_selected/picked_data_optimal.csv`
2. **Testing rapido**: Usar `1_cleaned/tracks_features_500.csv` (500 registros, sin letras)
3. **Separadores**: Verificar separador segun carpeta ANTES de cargar
4. **Rutas en codigo**: Usar `data/3_selected/picked_data_optimal.csv`
5. **Legacy**: NUNCA referenciar archivos en `archive/legacy_data/`
6. **Multimodal**: El dataset production tiene audio features + letras
7. **Clustering**: Hopkins 0.823 garantiza buena separabilidad

---

## Archivos Auxiliares

| Archivo | Proposito |
|---------|-----------|
| `auxiliary/lyrics.db` | Cache SQLite de letras extraidas de Genius |
| `auxiliary/lyrics_availability_cache.json` | Cache de disponibilidad de letras |

Estos archivos se usaron para enriquecer el dataset de Kaggle con letras.
