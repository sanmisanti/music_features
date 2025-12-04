# DATA - Guia de Datasets

## Dataset de Produccion

### Para analisis multimodal completo (RECOMENDADO)

```python
import pickle

with open('data/5_unified/unified_multimodal_7811.pkl', 'rb') as f:
    dataset = pickle.load(f)

track_ids = dataset['data']['track_ids']                    # [7811]
semantic = dataset['data']['semantic_embeddings']           # [7811, 384]
musical_norm = dataset['data']['musical_features_normalized']  # [7811, 12]
metadata = dataset['data']['track_metadata']                # DataFrame
```

- 7,811 canciones con embeddings BERT + caracteristicas musicales
- Ver: [5_unified/CLAUDE.md](5_unified/CLAUDE.md)

### Para analisis solo musical

```python
import pandas as pd
df = pd.read_csv('data/3_selected/picked_data_optimal.csv', sep='^', encoding='utf-8')
```

- 10,000 canciones con caracteristicas Spotify
- Ver: [3_selected/CLAUDE.md](3_selected/CLAUDE.md)

---

## Estructura de Carpetas

| Carpeta | Contenido | Registros | Separador | Docs |
|---------|-----------|-----------|-----------|------|
| 0_raw/ | Spotify original | 1,204,025 | `,` | [CLAUDE.md](0_raw/CLAUDE.md) |
| 1_cleaned/ | Testing rapido | 500 | `;` | [CLAUDE.md](1_cleaned/CLAUDE.md) |
| 2_with_lyrics/ | Fuente con letras | 18,454 | `@@` | [CLAUDE.md](2_with_lyrics/CLAUDE.md) |
| 3_selected/ | Produccion musical | 10,000 | `^` | [CLAUDE.md](3_selected/CLAUDE.md) |
| 4_vectorized/ | Embeddings BERT | 9,753 | N/A | [CLAUDE.md](4_vectorized/CLAUDE.md) |
| 5_unified/ | Multimodal final | 7,811 | N/A | [CLAUDE.md](5_unified/CLAUDE.md) |
| auxiliary/ | Cache sistema | N/A | N/A | [CLAUDE.md](auxiliary/CLAUDE.md) |

---

## Flujo de Datos

```
2_with_lyrics/spotify_songs_fixed.csv (18,454)
    |
    v  Seleccion clustering-aware
3_selected/picked_data_optimal.csv (10,000)
    |
    v  Vectorizacion BERT
4_vectorized/embeddings_bert_9753x384.npy (9,753 -> 8,567 validos)
    |
    v  Unificacion + deduplicacion
5_unified/unified_multimodal_7811.pkl (7,811)
```

---

## Problemas Metodologicos Identificados

### Dataset Multimodal (5_unified/)

El dataset de 7,811 canciones **NO fue disenado intencionalmente**. Es el resultado residual de filtros tecnicos:

| Etapa | Registros | Perdida |
|-------|-----------|---------|
| Seleccion inicial | 10,000 | - |
| Con letras validas | 9,753 | -247 |
| BERT exitoso (no-zero) | 8,567 | -1,186 |
| Post-deduplicacion | 7,811 | -756 |
| **Perdida total** | | **-21.9%** |

### Validaciones Faltantes

- [ ] Hopkins Statistic post-unificacion
- [ ] Analisis de sesgo por disponibilidad de letras
- [ ] Validacion de representatividad vs dataset original
- [ ] Test de estabilidad de clustering (multiples seeds)
- [ ] Analisis de caracteristicas de canciones excluidas

### Sesgos Potenciales No Caracterizados

1. Canciones con letras "faciles" de procesar sobrerrepresentadas
2. Posible sesgo hacia ciertos idiomas/generos
3. Caracteristicas musicales de excluidas vs incluidas no comparadas

---

## Plan de Mejora

Ver: [DATASET_IMPROVEMENT_PLAN.md](DATASET_IMPROVEMENT_PLAN.md)

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

## Datasets Legacy

Archivos obsoletos movidos a `archive/legacy_data/`. NO usar en codigo nuevo.

| Archivo | Razon de descarte |
|---------|-------------------|
| picked_data_0.csv | Sin letras (Fuente 1) |
| picked_data_lyrics.csv | Hopkins ~0.45 (clustering deficiente) |

---

## Aclaracion Historica: 16,081 vs 10,000

### Discrepancia en Documentacion

Algunos documentos del proyecto mencionan **16,081 canciones** mientras que `picked_data_optimal.csv` contiene **10,000**. Esta discrepancia se debe a dos procesos DISTINTOS:

| Proceso | Fecha | Fuente | Resultado | Archivo Generado |
|---------|-------|--------|-----------|------------------|
| **FASE 1.4 - Seleccion** | 2025-08-12 | 18,454 | 10,000 | `picked_data_optimal.csv` |
| **FASE 4 - Purificacion** | 2025-08-12 | 18,454 | 16,081 | Solo analisis (JSON) |

### Explicacion Tecnica

**El valor 16,081** representa el resultado del algoritmo **Hybrid Purification Strategy**:
- 18,454 canciones originales
- 87.1% retencion tras filtros de calidad
- 18,454 x 0.871 = 16,081 canciones

Este resultado se documento en `outputs/fase4_purification/purification_results_*.json` pero **nunca se materializo en un archivo CSV separado**. Fue un analisis experimental que valido la metodologia (+86.1% Silhouette Score).

**El valor 10,000** es el contenido REAL del archivo `picked_data_optimal.csv`:
- Generado por proceso de SELECCION (no purificacion)
- 10,000 canciones seleccionadas de 18,454 (54.2%)
- Optimizado para diversidad musical clustering-aware

### Conclusion

- **Usar 10,000** cuando se refiera al archivo `picked_data_optimal.csv`
- **Usar 16,081** solo en contexto del analisis de Hybrid Purification Strategy
- La mejora +86.1% Silhouette Score se logro sobre el dataset de 18,454, no sobre 10,000

---

## Nota sobre Reportes JSON

Los archivos `*_report.json` en subcarpetas contienen **rutas absolutas legacy** que referencian ubicaciones anteriores a la reorganizacion:

| Ruta en JSON | Ubicacion Actual |
|--------------|------------------|
| `data/with_lyrics/` | `data/2_with_lyrics/` |
| `data/final_data/` | `data/3_selected/` |

Estos reportes son **artefactos historicos** del proceso de generacion. Las rutas correctas son las documentadas en cada CLAUDE.md de subcarpeta.
