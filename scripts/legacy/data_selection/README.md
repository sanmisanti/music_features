# data_selection - Componentes Legacy

**Fecha de archivado**: 2025-12

## Razon de Archivado

Estos componentes fueron desarrollados para un pipeline que procesaba 1.2M canciones con verificacion de letras via API Genius. Este enfoque fue **abandonado** en favor del pipeline clustering-aware que:

1. Usa dataset de 18,454 canciones como fuente (Hopkins 0.823)
2. Preserva estructura natural de clustering
3. No requiere llamadas a API externa
4. Ejecuta en ~4 minutos vs 4-6 horas

## Contenido Archivado

| Directorio | Proposito Original | Razon de Descarte |
|------------|-------------------|-------------------|
| `pipeline/` | Orquestacion 1.2M canciones | Nunca ejecutado, dependencias rotas |
| `config/` | Configuracion pipeline hibrido | Solo usado por pipeline/ |
| `sampling/` | Estrategias de muestreo | Reimplementado en clustering_aware/ |
| `PIPELINE_ANALYSIS.md` | Documentacion pipeline 1.2M | Obsoleto, reemplazado por PIPELINE.md |

## Dependencias Rotas en pipeline/

```python
# representative_selector.py linea 40-41:
from lyrics_availability_checker import LyricsAvailabilityChecker  # NO EXISTE
from hybrid_selection_criteria import HybridSelectionCriteria      # NO EXISTE
```

## Pipeline Actual

El sistema de seleccion activo esta en:
```
data_selection/
├── clustering_aware/
│   ├── select_optimal_10k_from_18k.py  # Generador principal
│   └── hopkins_validator.py            # Validador Hopkins
└── PIPELINE.md                         # Documentacion actual
```

## NO USAR

Estos archivos son **solo referencia historica**. Para seleccion de datos, usar:

```bash
python data_selection/clustering_aware/select_optimal_10k_from_18k.py
```

## Dataset Generado

El archivo `picked_data_optimal.csv` fue generado por `clustering_aware/` el 2025-08-12.
Ubicacion actual: `data/3_selected/picked_data_optimal.csv`
