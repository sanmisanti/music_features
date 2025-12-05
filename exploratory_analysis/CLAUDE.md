# MODULO EXPLORATORY_ANALYSIS

**Ultima actualizacion**: Diciembre 2025
**Estado**: SISTEMA FUNCIONAL - Componentes activos validados

---

## OBJETIVO DEL MODULO

Proporcionar analisis exploratorio de datasets musicales con enfoque en optimizar la seleccion de datos para clustering. El modulo evalua calidad tecnica de datos e idoneidad para clustering efectivo.

### Capacidades Principales
1. Analisis exploratorio basico: estadisticas descriptivas, distribuciones, correlaciones
2. Evaluacion de clustering readiness: Hopkins Statistic, K optimo, separabilidad
3. Reduccion de dimensionalidad: PCA, t-SNE, UMAP
4. Generacion automatica de reportes: JSON, Markdown, HTML
5. Visualizaciones profesionales: distribuciones, mapas de calor de correlacion

---

## ARQUITECTURA DEL MODULO

### Estructura de Directorios

```
exploratory_analysis/
├── __init__.py
├── CLAUDE.md                           # Esta documentacion
├── run_full_analysis.py                # Script principal de analisis
├── analyze_clustering_readiness.py     # Script Hopkins/clustering
├── results/                            # Resultados de analisis
│   └── clustering_readiness_*.json
├── config/
│   ├── __init__.py
│   ├── analysis_config.py              # Configuracion centralizada (281 lineas)
│   └── features_config.py              # Definiciones caracteristicas Spotify
├── data_loading/
│   ├── __init__.py
│   ├── data_loader.py                  # Carga inteligente con chunking
│   └── data_validator.py               # Validacion de calidad de datos
├── statistical_analysis/
│   ├── __init__.py
│   └── descriptive_stats.py            # Estadisticas descriptivas completas
├── visualization/
│   ├── __init__.py
│   ├── distribution_plots.py           # Histogramas, boxplots, distribuciones
│   └── correlation_heatmaps.py         # Mapas de calor de correlacion
├── feature_analysis/
│   ├── __init__.py
│   ├── dimensionality_reduction.py     # PCA, t-SNE, UMAP (602 lineas)
│   └── clustering_readiness.py         # Hopkins, K optimo (662 lineas)
├── reporting/
│   ├── __init__.py
│   └── report_generator.py             # Generacion reportes JSON/MD/HTML
└── utils/
    ├── __init__.py
    └── file_utils.py                   # Utilidades de archivo y memoria
```

---

## COMPONENTES IMPLEMENTADOS

### 1. Config (Configuracion)

**analysis_config.py** - Configuracion centralizada con dataclasses:
- DataConfig: parametros CSV, sampling, memoria
- StatsConfig: niveles de significancia, tests
- PlotConfig: estilos, colores, formatos
- FeatureConfig: seleccion, reduccion dimensional
- ClusteringConfig: parametros clustering
- ReportConfig: formatos de salida

**features_config.py** - Definiciones de 13 caracteristicas musicales Spotify con tipos, rangos, y metodos de normalizacion.

### 2. Data Loading (Carga de Datos)

**data_loader.py** - Carga optimizada de datasets:
- Soporte para multiples formatos CSV (separadores @@, ^, ,)
- Chunking para datasets grandes
- Sampling configurable
- Validacion automatica opcional

**data_validator.py** - Validacion de calidad:
- Verificacion de rangos por caracteristica
- Deteccion de valores nulos
- Score de calidad 0-100
- Recomendaciones automaticas

### 3. Statistical Analysis (Analisis Estadistico)

**descriptive_stats.py** - Estadisticas descriptivas completas:
- Medidas de tendencia central: media, mediana, moda
- Medidas de dispersion: std, varianza, rango, IQR
- Forma de distribucion: asimetria, curtosis
- Deteccion de outliers: IQR, Z-score
- Dataclasses FeatureStats y DatasetStats

### 4. Visualization (Visualizacion)

**distribution_plots.py** - Graficos de distribucion:
- Histogramas con KDE
- Boxplots y violin plots
- Q-Q plots para normalidad
- Comparacion entre caracteristicas

**correlation_heatmaps.py** - Analisis de correlacion:
- Matrices de correlacion Pearson/Spearman
- Heatmaps con clustering jerarquico
- Identificacion de correlaciones altas
- Exportacion PNG de alta resolucion

### 5. Feature Analysis (Analisis de Caracteristicas)

**dimensionality_reduction.py** (602 lineas) - Reduccion dimensional:
- PCA con analisis de componentes e interpretacion
- t-SNE para visualizacion no lineal
- UMAP (opcional, requiere libreria)
- Feature selection por varianza y mutual information
- Comparacion visual de metodos

**clustering_readiness.py** (662 lineas) - Evaluacion para clustering:
- Hopkins Statistic: tendencia de clustering (>0.5 = clusterable)
- K optimo: Elbow, Silhouette, Calinski-Harabasz
- Analisis de separabilidad: distancias, densidad
- Ranking de caracteristicas: poder discriminativo
- Score de clustering readiness 0-100

### 6. Reporting (Reportes)

**report_generator.py** - Generacion automatica de reportes:
- Integracion de todos los modulos de analisis
- Formatos: JSON (datos), Markdown (legible), HTML (web)
- Visualizaciones embebidas
- Resumen ejecutivo automatico

### 7. Utils (Utilidades)

**file_utils.py** - Utilidades de sistema:
- Formateo de tamanos de archivo
- Monitoreo de memoria del proceso
- Verificacion de existencia de archivos
- Creacion segura de directorios

---

## SCRIPTS PRINCIPALES

### run_full_analysis.py

Ejecuta analisis exploratorio completo del dataset:

```bash
python exploratory_analysis/run_full_analysis.py
```

**Pipeline de ejecucion**:
1. Carga y validacion de datos
2. Analisis estadistico descriptivo
3. Generacion de visualizaciones
4. Analisis PCA y t-SNE
5. Generacion de reporte comprensivo

**Tiempo estimado**: 75 segundos para dataset completo

**Salidas**:
- Reportes en `outputs/reports/`
- Visualizaciones en `outputs/reports/visualizations/`

### analyze_clustering_readiness.py

Evalua aptitud del dataset para clustering:

```bash
python exploratory_analysis/analyze_clustering_readiness.py
```

**Metricas calculadas**:
- Hopkins Statistic
- K optimo recomendado
- Score de separabilidad
- Ranking de caracteristicas
- Clustering Readiness Score (0-100)

**Salidas**: `outputs/clustering_readiness/`

---

## DATASET PRINCIPAL

**Ubicacion**: `data/3_selected/picked_data_optimal.csv`
- Registros: 10,000 canciones
- Caracteristicas: 13 musicales Spotify + metadatos
- Separador: `^`
- Hopkins Statistic: 0.823 (excelente)

---

## RESULTADOS DE ANALISIS

La carpeta `results/` contiene outputs de analisis ejecutados:

| Archivo | Contenido |
|---------|-----------|
| clustering_readiness_direct_*.json | Hopkins 0.823, K optimo, feature ranking |

---

## USO PROGRAMATICO

### Clustering Readiness Assessment

```python
from exploratory_analysis.feature_analysis import ClusteringReadiness

analyzer = ClusteringReadiness()
results = analyzer.calculate_clustering_readiness_score(df)

print(f"Score: {results['readiness_score']}/100")
print(f"Hopkins: {results['component_analysis']['clustering_tendency']['hopkins_statistic']}")
```

### Analisis PCA

```python
from exploratory_analysis.feature_analysis import DimensionalityReducer

reducer = DimensionalityReducer()
pca_results = reducer.fit_pca(df, variance_threshold=0.90)

print(f"Componentes: {pca_results['n_components']}")
print(f"Varianza explicada: {pca_results['total_variance_explained']:.1%}")
```

### Estadisticas Descriptivas

```python
from exploratory_analysis.statistical_analysis import DescriptiveStats

stats = DescriptiveStats()
results = stats.analyze_dataset(df)

for feature, stats_obj in results['feature_stats'].items():
    print(f"{feature}: mean={stats_obj.mean:.3f}, std={stats_obj.std:.3f}")
```

---

## DEPENDENCIAS

**Core**:
- pandas, numpy
- scikit-learn (PCA, t-SNE, KMeans, StandardScaler)
- scipy (stats, spatial.distance)
- matplotlib, seaborn

**Opcional**:
- umap-learn (para UMAP)
- plotly (visualizaciones interactivas)
- psutil (monitoreo de memoria)

---

## NOTAS TECNICAS

1. El modulo utiliza separador `^` por defecto para datasets de `data/3_selected/`
2. Hopkins Statistic requiere minimo 100 muestras para resultados significativos
3. t-SNE tiene complejidad O(n^2), se recomienda sampling para datasets >5000
4. Los reportes HTML incluyen CSS embebido para portabilidad

---

*Modulo estado: FUNCIONAL - Componentes validados*
*Ultima limpieza: Diciembre 2025 - Eliminados 13 stubs vacios*
