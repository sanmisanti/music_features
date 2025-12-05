# FASE 3: Clustering Multimodal Exhaustivo

## Descripción

Sistema completo de experimentación algorítmica exhaustiva para clustering multimodal con prioridad en interpretabilidad. Evalúa sistemáticamente configuraciones de clustering en espacios vectoriales musical (12D) y semántico (384D) con función objetivo multi-criterio balanceada.

## Objetivos Científicos

- **Evaluación comparativa** de clustering musical vs semántico para recomendaciones interpretables
- **Análisis de correspondencias cross-modales** con NMI ≥ 0.60 
- **Optimización función objetivo multi-criterio** balanceando calidad técnica e interpretabilidad
- **Validación automática de interpretabilidad** con etiquetado coherente de clusters
- **Determinación de arquitectura óptima** para sistema de recomendaciones multimodal

## Arquitectura del Sistema

### Módulos Principales

- **`run_multimodal_clustering_evaluation.py`**: CLI principal con interfaz completa
- **`multimodal_clustering_experimenter.py`**: Orquestador de experimentación
- **`algorithm_evaluator.py`**: Evaluador especializado por dominio
- **`interpretability_validator.py`**: Sistema de interpretabilidad automática
- **`cross_modal_analyzer.py`**: Analizador de correspondencias cross-modales
- **`config/`**: Configuraciones algorítmicas especializadas

### Configuración Algorítmica

#### Dominio Musical (12D)
- **Algoritmos**: Hierarchical (Ward/Complete/Average), K-Means++, GMM-Full, DBSCAN euclidiano
- **Rango K**: [5, 6, 7, 8, 9, 10] 
- **Total configuraciones**: ~36

#### Dominio Semántico (384D)  
- **Algoritmos**: Hierarchical (Ward/Average), K-Means++, GMM-Tied, DBSCAN coseno
- **Rango K**: [5, 6, 7, 8]
- **Optimizaciones**: Métricas coseno, iteraciones reducidas, estabilidad numérica
- **Total configuraciones**: ~22

## Función Objetivo Multi-Criterio

```
Score = 0.3 × Silhouette_norm + 0.3 × Balance + 0.2 × Interpretabilidad + 0.1 × Cross_Modal + 0.1 × Granularidad
```

### Criterios de Evaluación

1. **Silhouette Score** (30%): Calidad técnica de clustering
2. **Balance Distribution** (30%): Evitar dominancia/fragmentación excesiva
3. **Interpretability Score** (20%): Coherencia automática de etiquetas
4. **Cross-Modal Correspondence** (10%): NMI entre dominios
5. **Granularity Bonus** (10%): Incentivo para K≥5 (interpretabilidad práctica)

## Uso del Sistema

### Instalación de Dependencias

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Comandos Principales

#### Evaluación Completa
```bash
python run_multimodal_clustering_evaluation.py \
  --dataset ../phase1_dataset_unification/unified_multimodal_dataset_20250822_004929.pkl \
  --output ./results \
  --verbose
```

#### Evaluación Rápida (sin cross-modal)
```bash
python run_multimodal_clustering_evaluation.py \
  --dataset dataset.pkl \
  --output ./results \
  --no-cross-modal
```

#### Mostrar Configuración
```bash
python run_multimodal_clustering_evaluation.py --show-config
```

#### Validar Dataset
```bash
python run_multimodal_clustering_evaluation.py --validate-dataset dataset.pkl
```

### Opciones CLI

- `--dataset, -d`: Ruta al dataset multimodal unificado (.pkl)
- `--output, -o`: Directorio de salida (default: ./results)
- `--no-cross-modal`: Omitir análisis cross-modal (más rápido)
- `--top-n-cross-modal`: N configuraciones para cross-modal (default: 3)
- `--verbose, -v`: Progreso detallado
- `--quiet, -q`: Solo resultados finales

## Outputs del Sistema

### Resultados por Dominio
- `musical_clustering_results_TIMESTAMP.csv`: Todas las configuraciones musicales evaluadas
- `musical_top5_configurations_TIMESTAMP.csv`: Mejores 5 configuraciones musicales
- `semantic_clustering_results_TIMESTAMP.csv`: Todas las configuraciones semánticas
- `semantic_top5_configurations_TIMESTAMP.csv`: Mejores 5 configuraciones semánticas

### Análisis Cross-Modal
- `cross_modal_analysis_TIMESTAMP.json`: Correspondencias entre dominios
- `contingency_matrix_TIMESTAMP.png`: Visualización de correspondencias

### Reporte Científico
- `comprehensive_report_TIMESTAMP.json`: Reporte científico completo con:
  - Mejores configuraciones por dominio
  - Análisis cross-modal exhaustivo  
  - Validación de interpretabilidad
  - Conclusiones científicas fundamentadas

## Criterios de Éxito

### Técnicos
- **Silhouette Score** ≥ 0.15
- **Balance de clusters** ≥ 0.6
- **Granularidad** K ≥ 5 en ambos dominios

### Interpretabilidad
- **100% clusters etiquetables** automáticamente
- **Coherencia interna** validada por dominio
- **NMI cross-modal** ≥ 0.60 para coherencia multimodal

### Científicos
- **Reproducibilidad** completa con mismos parámetros
- **Justificación estadística** de configuraciones óptimas
- **Metodología publicable** en Music Information Retrieval

## Interpretabilidad Automática

### Musical (12D)
- **Etiquetas automáticas** basadas en características dominantes
- **Ejemplos**: "Alta Energía & Positivo", "Acústico & Melancólico"
- **Validación**: Consistencia interna por característica

### Semántico (384D)
- **Etiquetas automáticas** basadas en coherencia coseno
- **Ejemplos**: "Tema Principal Muy Coherente", "Subtema Moderadamente Coherente"  
- **Validación**: Similitud coseno interna promedio

## Análisis Cross-Modal

### Correspondencias
- **Fuertes** (≥30% overlap): Clusters altamente alineados
- **Débiles** (10-30% overlap): Alineación parcial
- **Divergentes**: Casos de fragmentación cross-modal

### Métricas
- **NMI (Normalized Mutual Information)**: Correspondencia global
- **ARI (Adjusted Rand Index)**: Concordancia de clustering
- **Cobertura**: % muestras en correspondencias fuertes

## Contribuciones Metodológicas

1. **Función objetivo multi-criterio** balanceada para clustering orientado a recomendaciones
2. **Sistema de interpretabilidad automática** especializado por dimensionalidad
3. **Protocolo de evaluación cross-modal** para coherencia multimodal
4. **Configuraciones algorítmicas optimizadas** para espacios 12D vs 384D

## Limitaciones y Consideraciones

- **Tiempo de ejecución**: ~10-15 minutos para evaluación completa
- **Memoria requerida**: ~4GB RAM para dataset de 7,811 canciones
- **Dependencias**: Requiere scikit-learn actualizado para algunos algoritmos
- **Interpretabilidad semántica**: Limitada a métricas de coherencia interna

## Extensiones Futuras

- **Clustering ensemble** combinando mejores configuraciones
- **Interpretabilidad semántica mejorada** con análisis de contenido textual
- **Optimización hiperparámetros** con búsqueda bayesiana
- **Validación cualitativa** con expertos musicales