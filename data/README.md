# DATA - Estructura Jerárquica de Datasets

Este directorio organiza los datasets del proyecto según su nivel de madurez y procesamiento, desde datos crudos (nivel 0) hasta datasets production-ready (nivel 3).

---

## Estructura de Carpetas

```
data/
├── 0_raw/              # Nivel 0: Datos originales sin procesar
├── 1_cleaned/          # Nivel 1: Datos limpios y normalizados
├── 2_with_lyrics/      # Nivel 2: Datos con letras integradas
├── 3_selected/         # Nivel 3: Datasets seleccionados (production-ready)
└── auxiliary/          # Archivos auxiliares (cache, bases de datos)
```

---

## 0_raw/ - Datos Originales

Datasets en su estado original, sin ningún procesamiento.

| Archivo | Registros | Tamaño | Separador | Descripción |
|---------|-----------|--------|-----------|-------------|
| `tracks_features.csv` | 1,204,025 | 330MB | `;` | Dataset completo Spotify Million Playlist |

**Origen**: Spotify API - Million Playlist Dataset
**Fecha**: Diciembre 2020

---

## 1_cleaned/ - Datos Limpios

Datasets procesados con limpieza, normalización y validación de rangos.

| Archivo | Registros | Tamaño | Separador | Descripción |
|---------|-----------|--------|-----------|-------------|
| `tracks_features_clean.csv` | 1,204,025 | 317MB | `;` | Dataset completo limpio y normalizado |
| `tracks_features_500.csv` | 500 | 120KB | `;` | Subset para testing rápido de algoritmos |

**Procesamiento aplicado**:
- Eliminación de valores nulos
- Normalización de rangos [0,1] para features
- Validación de tipos de datos
- Limpieza de caracteres especiales

---

## 2_with_lyrics/ - Datos con Letras

Datasets que incluyen letras de canciones verificadas.

| Archivo | Registros | Tamaño | Separador | Descripción |
|---------|-----------|--------|-----------|-------------|
| `spotify_songs.csv` | 18,454 | 42MB | `@@` | Subset con letras (separador original) |
| `spotify_songs_fixed.csv` | 18,454 | 43MB | `@@` | Versión con separadores corregidos |

**Características**:
- Hopkins Statistic: 0.823 (EXCELENTE para clustering)
- Letras pre-verificadas de Genius.com
- 25 columnas: metadatos Spotify + letras completas

**Nota**: Este es el dataset fuente recomendado para generar selecciones optimizadas.

---

## 3_selected/ - Datasets Seleccionados (Production-Ready)

Datasets finales optimizados para clustering y análisis multimodal.

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

## Flujo de Transformación

```
0_raw/tracks_features.csv (1.2M)
    │
    └──> [limpieza] ──> 1_cleaned/tracks_features_clean.csv
                            │
                            └──> [subset con letras] ──> 2_with_lyrics/spotify_songs_fixed.csv
                                                              │
                                                              └──> [selección optimizada]
                                                                        │
                                                                        ▼
                                                        3_selected/picked_data_optimal.csv
```

---

## Notas Importantes

1. **Separadores**: Cada nivel puede usar diferentes separadores. Verificar antes de cargar.
2. **Dataset problemático**: `picked_data_lyrics.csv` tiene problemas de clustering documentados. Usar `picked_data_optimal.csv` en su lugar.
3. **Encoding**: Todos los archivos usan UTF-8.

---

**Última actualización**: Diciembre 2025
