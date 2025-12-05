# DATA_SELECTION - Modulo de Seleccion de Datos

## Resumen Ejecutivo

Este modulo implementa el pipeline **clustering-aware** que genero el dataset de produccion `picked_data_optimal.csv` (10,000 canciones) a partir del dataset fuente de 18,454 canciones.

| Metrica | Valor |
|---------|-------|
| Dataset fuente | 18,454 canciones |
| Dataset generado | 10,000 canciones (54.2%) |
| Hopkins Statistic | 0.823 (EXCELENTE) |
| Silhouette Score | 0.289 (+86% vs baseline) |
| Fecha de generacion | 2025-08-12 |
| Tiempo de ejecucion | 239.7 segundos |

---

## Estado del Modulo

**STATUS: COMPLETADO** - El dataset ya fue generado y validado. No es necesario re-ejecutar.

**Output principal**: `data/3_selected/picked_data_optimal.csv`

---

## Arquitectura

### Componentes Activos

```
data_selection/
├── clustering_aware/
│   ├── select_optimal_10k_from_18k.py   # Script principal (878 lineas)
│   ├── hopkins_validator.py              # Validador Hopkins (561 lineas)
│   └── __init__.py
├── PIPELINE.md                           # Documentacion del proceso
├── CLAUDE.md                             # Este archivo
└── __init__.py                           # Exports: HopkinsValidator
```

### Componentes Legacy

Movidos a `scripts/legacy/data_selection/` (solo referencia historica):

| Componente | Razon de Archivado |
|------------|-------------------|
| pipeline/ | Nunca ejecutado, dependencias rotas |
| config/ | Solo usado por pipeline/ |
| sampling/ | Reimplementado en clustering_aware/ |
| PIPELINE_ANALYSIS.md | Documentaba pipeline 1.2M, obsoleto |

---

## Metodologia Implementada

### Hopkins Statistic

Metrica que mide tendencia natural al clustering [0,1]:
- **> 0.75**: Altamente clusterizable (estructura clara)
- **0.5-0.75**: Moderadamente clusterizable
- **< 0.5**: Tiende a aleatorio (problematico)

El dataset fuente de 18,454 tiene Hopkins 0.823, indicando estructura natural optima.

### Proceso de Seleccion (4 pasos)

1. **Pre-Clustering Natural (K=2)**
   - Cluster 0: ~60% canciones (vocal/mainstream)
   - Cluster 1: ~40% canciones (instrumental/experimental)

2. **Seleccion Proporcional**
   - 6,000 canciones de Cluster 0 (60%)
   - 4,000 canciones de Cluster 1 (40%)

3. **MaxMin Sampling con KD-Tree**
   - Maximiza diversidad dentro de cada cluster
   - Complejidad O(n log n) vs O(n^2) naive
   - Features utilizadas: instrumentalness, liveness, duration_ms, energy, danceability

4. **Validacion Hopkins Continua**
   - Durante seleccion, valida Hopkins > 0.70
   - Fallback de diversidad si Hopkins degrada

---

## Dataset Generado

### Archivo Principal

**`data/3_selected/picked_data_optimal.csv`** - 10,000 canciones

### Especificaciones

| Atributo | Valor |
|----------|-------|
| Registros | 10,000 |
| Columnas | 26 |
| Separador | `^` (caret) |
| Encoding | UTF-8 |

### Carga Rapida

```python
import pandas as pd
df = pd.read_csv('data/3_selected/picked_data_optimal.csv', sep='^', encoding='utf-8')
```

### Caracteristicas Musicales (12)

```python
musical_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms'
]
```

Nota: `time_signature` no esta disponible en el dataset fuente de 18K.

---

## Metricas de Calidad

### Diversidad por Caracteristica

| Feature | Original STD | Selected STD | Ratio |
|---------|-------------|--------------|-------|
| instrumentalness | 0.168 | 0.223 | 1.33 |
| liveness | 0.154 | 0.184 | 1.20 |
| energy | 0.181 | 0.198 | 1.10 |

Ratios > 1.0 indican mayor diversidad en seleccion vs original.

### Reporte de Optimizacion

Archivo: `data/3_selected/optimization_report_20250812_185734.json`

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

---

## Nota Historica: Pipeline Hibrido 1.2M

El proyecto originalmente planeaba un pipeline que procesaba 1.2M canciones con verificacion de letras via API Genius. Este enfoque fue **abandonado** porque:

1. Producia Hopkins ~0.45 (destruia estructura de clustering)
2. Requeria 4-6 horas de ejecucion
3. Tenia dependencias rotas (modulos faltantes)

La solucion fue cambiar al dataset de 18K canciones con Hopkins 0.823 como fuente, resultando en el pipeline clustering-aware actual que ejecuta en ~4 minutos.

---

## Nota Historica: 16,081 vs 10,000

Algunos documentos mencionan 16,081 canciones. Esta discrepancia se explica por dos procesos DISTINTOS:

| Proceso | Resultado | Archivo |
|---------|-----------|---------|
| Seleccion (FASE 1.4) | 10,000 | `picked_data_optimal.csv` |
| Purificacion (FASE 4) | 16,081 | Solo analisis JSON |

El valor 16,081 fue un analisis experimental (18,454 x 87.1% retencion) que nunca se materializo en CSV. El archivo de produccion contiene 10,000 canciones.

---

## API Exportada

```python
from data_selection import HopkinsValidator, quick_hopkins_check

# Validacion rapida
score = quick_hopkins_check(X, n_samples=100)

# Validacion completa
validator = HopkinsValidator()
result = validator.validate(X, n_iterations=10)
```

---

## Ejecucion (ya completada)

```bash
# NO ES NECESARIO RE-EJECUTAR - Dataset ya generado
python data_selection/clustering_aware/select_optimal_10k_from_18k.py
```

**Rutas en el script** (legacy, pre-reorganizacion):
- Input esperado: `data/with_lyrics/spotify_songs_fixed.csv`
- Input actual: `data/2_with_lyrics/spotify_songs_fixed.csv`

---

## Referencias

### Documentacion Interna

- **PIPELINE.md** - Proceso detallado de seleccion
- **data/3_selected/CLAUDE.md** - Dataset generado
- **scripts/legacy/data_selection/README.md** - Componentes archivados

### Referencias Academicas

- Hopkins Statistic: Lawson & Jurs (1990)
- MaxMin Sampling: Gonzalez (1985)
- Silhouette Analysis: Rousseeuw (1987)
