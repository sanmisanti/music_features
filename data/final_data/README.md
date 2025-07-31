# 🎵 Final Data - Selected Datasets

Este directorio contiene los datasets finales seleccionados para el sistema de recomendación multimodal.

## 📁 Contenido

### `picked_data_0.csv` (ARCHIVADO)
- **Origen**: Selección manual previa
- **Cantidad**: Variable
- **Método**: Selección tradicional sin verificación de letras
- **Estado**: Archivado para referencia histórica

### `picked_data_1.csv` (ACTUAL) 🎯
- **Origen**: Pipeline Híbrido con Verificación de Letras
- **Cantidad**: 10,000 canciones representativas
- **Composición**:
  - 8,000 canciones con letras verificadas (80%)
  - 2,000 canciones sin letras (20%)
- **Método**: 5-stage pipeline con progressive constraints
- **Características**:
  - Diversidad musical preservada
  - Distribuciones estadísticas mantenidas
  - Calidad de datos validada
  - Optimizado para análisis multimodal

## 🔧 Generación

### Pipeline Híbrido Utilizado
```bash
python scripts/run_hybrid_selection_pipeline.py --target-size 10000
```

### Stages del Pipeline
1. **Diversity Sampling**: 1.2M → 100K (MaxMin algorithm)
2. **Stratified Sampling**: 100K → 50K (Distribution preservation)
3. **Quality Filtering**: 50K → 25K (Composite scoring)
4. **Hybrid Selection**: 25K → 10K (Lyrics verification + Progressive constraints)

### Progressive Constraints
- Stage 4.1: 70% con letras
- Stage 4.2: 75% con letras  
- Stage 4.3: 78% con letras
- Stage 4.4: 80% con letras (FINAL)

## 📊 Formato de Datos

### Estructura CSV
```
id,name,artists,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature
```

### Encoding
- **Separador**: `;` (punto y coma)
- **Decimal**: `,` (coma - formato español)
- **Codificación**: UTF-8
- **Header**: Incluido

### Lectura Recomendada
```python
import pandas as pd

# Cargar dataset híbrido actual
df = pd.read_csv('data/final_data/picked_data_1.csv', 
                 sep=';', decimal=',', encoding='utf-8')
```

## 🎯 Uso Posterior

### Para Clustering
```python
# Usar picked_data_1.csv como input para clustering
python clustering/clustering.py --input data/final_data/picked_data_1.csv
```

### Para Extracción de Letras
```python
# Extraer letras de las 8,000 canciones verificadas
python lyrics_extractor/genius_lyrics_extractor.py --input data/final_data/picked_data_1.csv
```

### Para Recomendaciones
```python
# Sistema de recomendación con datos híbridos
jupyter notebook pred.ipynb
```

## 📈 Métricas de Calidad

### Validación Esperada
- **Quality Score**: ≥ 85/100
- **Distributional Similarity**: KS-test p-value > 0.05
- **Feature Coverage**: 95%+ del espacio original
- **Lyrics Availability**: 80% ± 2%

### Archivos de Validación
Los reportes de calidad se generan en:
```
outputs/selection_pipeline_[TIMESTAMP]/
├── validation/
│   ├── selection_validation_report.md
│   └── quality_metrics.json
└── reports/
    └── final_selection_report.md
```

## 🔄 Versionado

- **v0**: `picked_data_0.csv` - Selección manual/tradicional
- **v1**: `picked_data_1.csv` - Pipeline híbrido con letras ← **ACTUAL**

## 📞 Información Técnica

- **Generado por**: Exploratory Analysis Selection Pipeline v1.0
- **Fecha**: 2025-01-28
- **Algoritmo**: Hybrid Multi-Stage Selection with Lyrics Verification
- **API utilizada**: Genius.com para verificación de letras

---

*Datasets optimizados para el Sistema de Recomendación Musical Multimodal*