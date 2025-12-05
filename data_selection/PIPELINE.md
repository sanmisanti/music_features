# Pipeline de Seleccion de Datos - Clustering-Aware

**Status**: COMPLETADO - Dataset generado el 2025-08-12
**Output**: `data/3_selected/picked_data_optimal.csv` (10,000 canciones)

---

## Resumen Ejecutivo

Este pipeline selecciona 10,000 canciones del dataset fuente de 18,454 preservando la estructura natural de clustering identificada mediante Hopkins Statistic.

### Resultados Obtenidos

| Metrica | Valor |
|---------|-------|
| Dataset fuente | 18,454 canciones |
| Dataset generado | 10,000 canciones |
| Hopkins fuente | 0.823 (EXCELENTE) |
| Silhouette final | 0.289 (+86% vs baseline) |
| Tiempo ejecucion | 239.7 segundos |

---

## Arquitectura

```
data_selection/
├── clustering_aware/
│   ├── select_optimal_10k_from_18k.py   # Script principal (878 lineas)
│   ├── hopkins_validator.py              # Validador Hopkins (561 lineas)
│   └── __init__.py
├── PIPELINE.md                           # Este documento
└── __init__.py
```

### Componentes Legacy

Movidos a `scripts/legacy/data_selection/`:
- `pipeline/` - Pipeline 1.2M nunca ejecutado
- `config/` - Configuracion obsoleta
- `sampling/` - Reimplementado en clustering_aware

---

## Fundamento Cientifico

### Hopkins Statistic

Metrica que mide tendencia natural al clustering [0,1]:
- **> 0.75**: Altamente clusterizable (estructura clara)
- **0.5-0.75**: Moderadamente clusterizable
- **< 0.5**: Tiende a aleatorio (problematico)

### Descubrimiento Critico

El pipeline hibrido anterior (1.2M canciones) produjo Hopkins ~0.45, destruyendo la estructura natural. La solucion fue usar el dataset de 18K con Hopkins 0.823 como fuente.

---

## Proceso de Seleccion

### Paso 1: Pre-Clustering Natural

```python
# K=2 identificado como optimo natural
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_scaled)

# Resultado:
# Cluster 0: ~60% canciones (vocal/mainstream)
# Cluster 1: ~40% canciones (instrumental/experimental)
```

### Paso 2: Seleccion Proporcional

Mantiene proporcion natural de cada cluster:
- Cluster 0: 6,000 canciones (60%)
- Cluster 1: 4,000 canciones (40%)

### Paso 3: MaxMin Sampling con KD-Tree

Dentro de cada cluster, maximiza diversidad usando top 5 caracteristicas:

```python
top_features = ['instrumentalness', 'liveness', 'duration_ms', 'energy', 'danceability']
```

Optimizacion: KD-Tree reduce complejidad O(n^2) a O(n log n).

### Paso 4: Validacion Hopkins Continua

Durante seleccion, valida que Hopkins se mantenga > 0.70:
- Si Hopkins cae, activa fallback de diversidad
- Re-valida post-fallback

---

## Ejecucion

### Comando (ya ejecutado 2025-08-12)

```bash
python data_selection/clustering_aware/select_optimal_10k_from_18k.py
```

### Rutas en el Script (legacy)

El script referencia rutas anteriores a la reorganizacion:
- Input: `data/with_lyrics/spotify_songs_fixed.csv` -> Ahora: `data/2_with_lyrics/`
- Output: `data/final_data/picked_data_optimal.csv` -> Ahora: `data/3_selected/`

El dataset ya fue generado, no es necesario re-ejecutar.

---

## Caracteristicas Musicales (12)

```python
musical_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms'
]
```

Nota: `time_signature` no esta disponible en el dataset fuente de 18K.

---

## Validacion de Resultados

### Metricas Obtenidas (optimization_report_20250812_185734.json)

```json
{
  "source_dataset": {
    "total_songs": 18454,
    "hopkins_baseline": 0.788
  },
  "selection_process": {
    "target_size": 10000,
    "selected_size": 10000,
    "execution_time_seconds": 239.73
  },
  "quality_metrics": {
    "average_musical_diversity": 1.109
  }
}
```

### Diversidad por Caracteristica

| Feature | Original STD | Selected STD | Ratio |
|---------|-------------|--------------|-------|
| instrumentalness | 0.168 | 0.223 | 1.33 |
| liveness | 0.154 | 0.184 | 1.20 |
| energy | 0.181 | 0.198 | 1.10 |

Ratios > 1.0 indican mayor diversidad en seleccion vs original.

---

## Referencias

- **Hopkins Statistic**: Lawson & Jurs (1990)
- **MaxMin Sampling**: Gonzalez (1985)
- **Silhouette Analysis**: Rousseeuw (1987)

---

## Documentacion Relacionada

- Dataset: `data/3_selected/CLAUDE.md`
- Resultados clustering: `outputs/fase4_purification/`
- Legacy pipeline: `scripts/legacy/data_selection/README.md`
