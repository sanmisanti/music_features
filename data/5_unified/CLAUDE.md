# 5_unified - Dataset Multimodal Unificado (FINAL)

## ESTE ES EL DATASET PRINCIPAL PARA RECOMENDACIONES MULTIMODALES

## Archivo Principal

**unified_multimodal_7811.pkl** - Dataset completo con embeddings BERT + caracteristicas musicales

## Estadisticas

| Metrica | Valor |
|---------|-------|
| Canciones | 7,811 |
| Embeddings semanticos | [7811, 384] BERT L2-normalized |
| Caracteristicas musicales | [7811, 12] StandardScaler |
| Generos | 6 (rock, r&b, pop, rap, edm, latin) |
| Tamano total | ~39 MB |

## Carga Rapida

```python
import pickle

with open('data/5_unified/unified_multimodal_7811.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Contenido del dataset
track_ids = dataset['data']['track_ids']                    # [7811]
semantic = dataset['data']['semantic_embeddings']           # [7811, 384]
musical_raw = dataset['data']['musical_features_raw']       # [7811, 12]
musical_norm = dataset['data']['musical_features_normalized']  # [7811, 12]
metadata = dataset['data']['track_metadata']                # DataFrame
```

## Carga de Arrays Individuales

```python
import numpy as np

# Arrays pre-extraidos (mas rapido para acceso parcial)
semantic = np.load('data/5_unified/arrays/semantic_embeddings.npy')  # [7811, 384]
musical_raw = np.load('data/5_unified/arrays/musical_features_raw.npy')  # [7811, 12]
musical_norm = np.load('data/5_unified/arrays/musical_features_normalized.npy')  # [7811, 12]
track_ids = np.load('data/5_unified/arrays/track_ids.npy', allow_pickle=True)  # [7811]
```

## Distribucion de Generos

| Genero | Cantidad | Porcentaje |
|--------|----------|------------|
| rock | 1,927 | 24.7% |
| r&b | 1,555 | 19.9% |
| pop | 1,418 | 18.2% |
| rap | 1,372 | 17.6% |
| edm | 782 | 10.0% |
| latin | 757 | 9.7% |

## Archivos en esta Carpeta

| Archivo | Descripcion |
|---------|-------------|
| unified_multimodal_7811.pkl | Dataset completo serializado |
| arrays/ | Arrays numpy individuales |
| aligned_songs.csv | Metadatos exportados en CSV |
| unification_report.json | Estadisticas del dataset |
| intersection_report.json | Reporte de auditoria de interseccion |
| valid_track_ids_7811.npy | IDs de tracks validos |
| aligned_songs_summary.json | Resumen de alineacion |

## LIMITACIONES METODOLOGICAS CRITICAS

### Origen del Dataset

Este dataset **NO fue disenado intencionalmente**. Es el resultado residual de filtros tecnicos:

```
picked_data_optimal.csv (10,000)
    |
    v  Filtro: letras disponibles
9,753 con letras
    |
    v  Filtro: vectorizacion BERT exitosa
8,567 embeddings validos
    |
    v  Filtro: deduplicacion + interseccion
7,811 canciones finales
```

### Perdida de Datos

- **Perdida total**: 21.9% (10,000 -> 7,811)
- **Causa 1**: 247 canciones sin letras validas
- **Causa 2**: 1,186 fallos de vectorizacion BERT
- **Causa 3**: 756 por deduplicacion/no-match

### Validaciones FALTANTES

- [ ] Hopkins Statistic post-unificacion (NO CALCULADO)
- [ ] Analisis de sesgo por disponibilidad de letras
- [ ] Validacion de representatividad vs dataset original
- [ ] Test de estabilidad de clustering (multiples seeds)
- [ ] Analisis de caracteristicas de canciones excluidas

### Sesgos Potenciales No Caracterizados

1. Canciones con letras "faciles" de procesar sobrerrepresentadas
2. Posible sesgo hacia ciertos idiomas/generos
3. Caracteristicas musicales de excluidas vs incluidas no comparadas

## Ver Tambien

- **Plan de mejora**: `data/DATASET_IMPROVEMENT_PLAN.md`
- **Origen de embeddings**: `data/4_vectorized/CLAUDE.md`
- **Dataset fuente**: `data/3_selected/CLAUDE.md`
