# 🎯 Scripts Directory - Music Selection Pipeline

Este directorio contiene los scripts ejecutables principales para el pipeline completo de selección inteligente de canciones. El objetivo es procesar el dataset completo de 1.2M canciones y seleccionar inteligentemente 10,000 canciones representativas para el modelo final.

## 📋 Descripción General

El pipeline implementa una estrategia multi-etapa para garantizar que el subset seleccionado mantenga todas las propiedades estadísticas y de diversidad del dataset original:

1. **Análisis Completo** (1.2M → análisis estadístico)
2. **Muestreo Diverso** (1.2M → 100K por diversidad)  
3. **Filtrado por Calidad** (100K → 50K por completitud)
4. **Selección Final** (50K → 10K por representatividad)

## 🔧 Scripts Principales

### 1. `large_dataset_processor.py`
**Propósito**: Procesa y analiza el dataset completo de 1.2M canciones.

```bash
# Uso básico
python scripts/large_dataset_processor.py

# Con parámetros personalizados
python scripts/large_dataset_processor.py --sample-size 50000 --output-dir outputs/analysis --verbose

# Especificar dataset
python scripts/large_dataset_processor.py --dataset-path path/to/tracks_features.csv
```

**Funcionalidades**:
- Análisis estadístico completo de las 13 características musicales
- Reducción dimensional con PCA y t-SNE
- Análisis de correlaciones y importancia de features
- Generación de reportes comprensivos
- Optimizado para datasets grandes (>1GB)

**Salidas**:
- Análisis estadístico detallado
- Visualizaciones de distribuciones
- Reportes en formato Markdown y JSON
- Logs de procesamiento

### 2. `representative_selector.py`  
**Propósito**: Selecciona inteligentemente N canciones representativas del dataset completo.

```bash
# Seleccionar 10,000 canciones (default)
python scripts/representative_selector.py

# Seleccionar cantidad personalizada
python scripts/representative_selector.py --target-size 5000

# Con directorio de salida personalizado
python scripts/representative_selector.py --target-size 10000 --output-dir outputs/selection_10k
```

**Estrategia de Selección Multi-Etapa**:
1. **Diversidad**: Muestreo max-min para cobertura del espacio de features
2. **Estratificado**: Preservación de distribuciones estadísticas
3. **Calidad**: Filtrado por completitud y validez de datos
4. **Clustering**: Selección final basada en representatividad de clusters

**Salidas**:
- Dataset seleccionado en formato CSV
- Reporte de calidad de selección
- Métricas de representatividad
- Logs detallados del proceso

### 3. `selection_validator.py`
**Propósito**: Valida la calidad y representatividad de la selección realizada.

```bash
# Validar selección
python scripts/selection_validator.py \
  --original-path data/original_data/tracks_features.csv \
  --selected-path outputs/selected_songs_10000.csv

# Con directorio de salida personalizado
python scripts/selection_validator.py \
  --original-path data/original_data/tracks_features.csv \
  --selected-path outputs/selected_songs_10000.csv \
  --output-dir outputs/validation
```

**Validaciones Realizadas**:
- ✅ **Distribuciones Estadísticas**: Comparación de medias, deviaciones, tests KS
- ✅ **Cobertura del Espacio**: Análisis de rangos y percentiles  
- ✅ **Preservación de Diversidad**: Métricas de distancias y variabilidad
- ✅ **Correlaciones**: Conservación de relaciones entre features

**Salidas**:
- Reporte de validación detallado
- Visualizaciones comparativas
- Puntuación de calidad (0-100)
- Recomendaciones de uso

### 4. `main_selection_pipeline.py` ⭐
**Propósito**: Orquesta el pipeline completo de principio a fin.

```bash
# Pipeline completo con configuración por defecto
python scripts/main_selection_pipeline.py

# Pipeline con parámetros personalizados
python scripts/main_selection_pipeline.py \
  --target-size 10000 \
  --output-dir outputs/pipeline_results \
  --verbose

# Saltar análisis si ya fue ejecutado
python scripts/main_selection_pipeline.py \
  --skip-analysis \
  --target-size 15000
```

**Pipeline Integrado**:
1. 🔍 **Análisis del Dataset Grande** (opcional, se puede saltar)
2. 🎯 **Selección Representativa** 
3. 🔍 **Validación de Calidad**
4. 📄 **Reporte Final Comprensivo**

**Salidas Organizadas**:
```
outputs/selection_pipeline_TIMESTAMP/
├── analysis/          # Resultados del análisis del dataset completo
├── selection/         # Dataset seleccionado y métricas
├── validation/        # Reportes de validación y visualizaciones
└── reports/           # Reporte final integrado
```

## 📊 Configuración y Dependencias

### Configuración Automática
Los scripts utilizan la configuración centralizada del sistema:

```python
# Configuración automática para datasets grandes
configure_for_large_dataset()  # Optimiza memoria y performance

# Configuraciones clave:
- chunk_size: 50,000 registros
- memory_management: activado
- parallel_processing: todos los cores disponibles
- caching: habilitado
```

### Dependencias Requeridas
Todas las dependencias están incluidas en el sistema exploratorio existente:

- **Core**: pandas, numpy, scikit-learn
- **Visualización**: matplotlib, seaborn  
- **Estadística**: scipy.stats
- **ML**: sklearn (PCA, clustering, preprocessing)

## 🎯 Casos de Uso Comunes

### Caso 1: Selección Rápida de 10K Canciones
```bash
# Un solo comando para todo el pipeline
python scripts/main_selection_pipeline.py --target-size 10000
```

### Caso 2: Análisis Detallado + Selección Personalizada
```bash
# 1. Análisis completo del dataset
python scripts/large_dataset_processor.py --verbose

# 2. Selección de 5K canciones
python scripts/representative_selector.py --target-size 5000

# 3. Validación detallada
python scripts/selection_validator.py \
  --original-path data/original_data/tracks_features.csv \
  --selected-path outputs/selected_songs_5000.csv
```

### Caso 3: Selección Solo (sin análisis previo)
```bash
# Pipeline saltando el análisis (más rápido)
python scripts/main_selection_pipeline.py --skip-analysis --target-size 15000
```

### Caso 4: Validación de Selección Existente
```bash
# Solo validar una selección ya realizada
python scripts/selection_validator.py \
  --original-path data/original_data/tracks_features.csv \
  --selected-path mi_seleccion.csv \
  --output-dir validacion_mi_seleccion
```

## 📈 Interpretación de Resultados

### Métricas de Calidad Clave

**Score de Selección (0-100)**:
- **90-100**: Excelente representatividad, listo para usar
- **80-89**: Buena calidad, apto para la mayoría de casos
- **70-79**: Calidad moderada, revisar detalles
- **<70**: Calidad insuficiente, revisar proceso

**Tests de Validación**:
- ✅ **Distribuciones**: Preservación de medias y varianzas
- ✅ **Cobertura**: Representación de todo el espacio de features
- ✅ **Diversidad**: Mantenimiento de variabilidad musical
- ✅ **Correlaciones**: Conservación de relaciones entre características

### Archivos de Salida Importantes

**Dataset Seleccionado**:
```
selected_songs_N_TIMESTAMP.csv  # N canciones seleccionadas
```

**Reportes de Calidad**:
```
selection_validation_report_TIMESTAMP.md     # Reporte legible
pipeline_summary_TIMESTAMP.json              # Datos estructurados
```

**Visualizaciones**:
```
distribution_comparison.png      # Comparación de distribuciones
correlation_comparison.png       # Mapas de calor de correlaciones
boxplot_comparison.png          # Comparación de rangos
```

## 🔧 Troubleshooting

### Problemas Comunes

**Error de Memoria Insuficiente**:
```bash
# Reducir el tamaño de muestra para análisis inicial
python scripts/large_dataset_processor.py --sample-size 100000
```

**Dataset No Encontrado**:
```bash
# Especificar ruta explícita
python scripts/main_selection_pipeline.py --dataset-path ruta/completa/al/dataset.csv
```

**Selección de Baja Calidad**:
- Verificar calidad del dataset original
- Aumentar el tamaño de muestra en etapas intermedias
- Revisar configuración de features en `config/features_config.py`

### Logs y Debugging

Todos los scripts generan logs detallados:
```
outputs/main_selection_pipeline.log      # Pipeline principal
outputs/large_dataset_processing.log     # Análisis
outputs/representative_selection.log     # Selección
outputs/selection_validation.log         # Validación
```

Usar `--verbose` para información adicional de debugging.

## 🚀 Próximos Pasos

Una vez completada la selección:

1. **Verificar Calidad**: Score de validación ≥ 80
2. **Usar en Clustering**: Aplicar K-means con las canciones seleccionadas
3. **Desarrollar Modelo**: Entrenar sistema de recomendación
4. **Integrar con Letras**: Combinar con análisis semántico

## 📞 Soporte

Para problemas o dudas:
1. Revisar logs de error en `outputs/`
2. Verificar configuración en `exploratory_analysis/config/`
3. Consultar documentación en `DOCS.md` y `ANALYSIS_RESULTS.md`

---

*Scripts desarrollados como parte del Sistema de Recomendación Musical Multimodal*  
*Versión: 1.0 | Fecha: Enero 2025*