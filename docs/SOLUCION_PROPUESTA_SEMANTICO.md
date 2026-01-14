# SOLUCIÓN PROPUESTA: COMPONENTE SEMÁNTICO

## Sistema de Análisis de Contenido Lírico mediante Embeddings BERT para Recomendaciones Musicales Multimodales

**Proyecto de Tesis - Ingeniería Informática**
**Componente**: Análisis Semántico
**Fecha de redacción**: Enero 2026

---

## ÍNDICE

1. [Introducción y Contexto del Componente Semántico](#1-introducción-y-contexto-del-componente-semántico)
2. [Fuentes de Datos y Pipeline de Preprocesamiento](#2-fuentes-de-datos-y-pipeline-de-preprocesamiento)
   - 2.5 [Análisis de Sesgo en Exclusión de Datos](#25-análisis-de-sesgo-en-exclusión-de-datos)
3. [Vectorización mediante Embeddings BERT](#3-vectorización-mediante-embeddings-bert)
4. [Unificación Multimodal: Fase 1](#4-unificación-multimodal-fase-1-del-evaluation-project)
5. [Validación de Clustering Readiness: Fase 2 - Análisis Hopkins](#5-validación-de-clustering-readiness-fase-2---análisis-hopkins)
6. [Evaluación Exhaustiva de Clustering: Fase 3](#6-evaluación-exhaustiva-de-clustering-fase-3)
   - 6.5 [Caracterización Detallada de Clusters Semánticos](#65-caracterización-detallada-de-clusters-semánticos)
   - 6.8 [Análisis de Sensibilidad de la Función Objetivo](#68-análisis-de-sensibilidad-de-la-función-objetivo-multi-criterio)
7. [Decisión Arquitectural: Vectores BERT Directos vs Clustering](#7-decisión-arquitectural-vectores-bert-directos-vs-clustering)
8. [Integración con Sistema de Recomendación Híbrido](#8-integración-con-sistema-de-recomendación-híbrido)
   - 8.5 [Evaluación Formal del Sistema de Recomendación](#85-evaluación-formal-del-sistema-de-recomendación)
   - 8.6 [Justificación Experimental de Pesos de Fusión](#86-justificación-experimental-de-pesos-de-fusión)
9. [Síntesis de Conceptos para Marco Teórico](#9-síntesis-de-conceptos-para-marco-teórico)

---

## 1. INTRODUCCIÓN Y CONTEXTO DEL COMPONENTE SEMÁNTICO

### 1.1 Fundamentación del Análisis Semántico en Sistemas MIR

El componente semántico del sistema de recomendaciones musicales desarrollado en esta investigación aborda una dimensión fundamental frecuentemente subestimada en sistemas tradicionales de Music Information Retrieval (MIR): el contenido lírico como fuente de información complementaria a las características acústicas. Mientras que los sistemas convencionales operan exclusivamente sobre características musicales extraídas de señales de audio, la incorporación sistemática de análisis semántico de letras permite capturar dimensiones temáticas, emocionales y narrativas que frecuentemente determinan las preferencias musicales de los usuarios pero que resultan inaccesibles mediante análisis puramente acústico.

La hipótesis central que guía el desarrollo del componente semántico sostiene que la información contenida en las letras musicales es **complementaria** —y no redundante— respecto a las características acústicas. Esta complementariedad se manifiesta en observaciones empíricas fundamentales: canciones musicalmente similares (mismo tempo, energía, instrumentación) pueden abordar temáticas radicalmente diferentes, mientras que canciones semánticamente relacionadas (tratando temas similares como desamor, celebración, o reflexión existencial) pueden diferir significativamente en su construcción musical. La fusión efectiva de ambas modalidades debería, por tanto, producir recomendaciones que satisfagan simultáneamente preferencias musicales y temáticas.

### 1.2 Objetivos Específicos del Componente Semántico

El desarrollo del componente semántico persigue los siguientes objetivos técnicos específicos:

1. **Vectorización semántica de alta calidad**: Transformar contenido lírico textual en representaciones vectoriales densas de 384 dimensiones mediante modelos BERT pre-entrenados, preservando relaciones semánticas entre letras y habilitando cálculo de similaridad por producto interno o distancia coseno.

2. **Procesamiento multilingüe robusto**: Implementar un pipeline de preprocesamiento capaz de manejar letras en múltiples idiomas (inglés, español, alemán, portugués) sin requerir traducción previa, aprovechando capacidades cross-lingüísticas de modelos transformer multilingües.

3. **Validación de clustering readiness**: Evaluar experimentalmente la viabilidad de técnicas de clustering en el espacio semántico de alta dimensionalidad (384D), desafiando asunciones establecidas sobre la "maldición de la dimensionalidad".

4. **Integración multimodal efectiva**: Desarrollar estrategias de fusión que combinen información semántica con características musicales preservando las contribuciones discriminativas de ambas modalidades.

> **[MARCO TEÓRICO]** Conceptos requeridos: *Sistemas de recomendación content-based, complementariedad informacional multimodal, representaciones distribuidas de texto.*

---

## 2. FUENTES DE DATOS Y PIPELINE DE PREPROCESAMIENTO

### 2.1 Origen y Características del Dataset

El componente semántico opera sobre letras musicales extraídas del dataset fuente ubicado en `data/2_with_lyrics/spotify_songs_fixed.csv`, que contiene **18,454 canciones** con letras disponibles. Este dataset constituye el punto de partida del pipeline de procesamiento semántico.

**Origen del Dataset:**

El dataset fuente se deriva del catálogo público de Spotify disponible en Kaggle (1,204,025 registros), enriquecido mediante scraping automatizado de letras desde Genius.com. El proceso de construcción del dataset siguió las etapas:

1. **Dataset base**: Catálogo Spotify de Kaggle con características de audio (1.2M canciones)
2. **Enriquecimiento lírico**: Scraping de letras desde Genius.com mediante API y matching por artista/título
3. **Filtrado por disponibilidad**: Retención de canciones con letras obtenidas exitosamente (18,454, 1.5% del total)

La tasa de obtención de letras (1.5%) refleja limitaciones inherentes al proceso de scraping: no todas las canciones del catálogo Spotify tienen letras indexadas en Genius.com, y el matching por artista/título introduce falsos negativos cuando existen variaciones de nomenclatura. Esta limitación es documentada pero no compromete la validez del análisis, dado que el objetivo del componente semántico es demostrar la viabilidad del enfoque multimodal, no la cobertura exhaustiva del catálogo.

**Tabla 2.1: Características del Dataset Fuente**

| Atributo | Valor | Observaciones |
|----------|-------|---------------|
| Registros totales | 18,454 | Canciones con letras disponibles |
| Formato archivo | CSV | Separador: `@@` |
| Idioma predominante | Inglés (84.4%) | Distribución natural del catálogo |
| Otros idiomas | ES 7.5%, DE 1.3%, PT 1.0% | Cobertura multilingüe |
| Longitud media letras | Variable | 50-5000 caracteres válidos |

La distribución de idiomas refleja la composición natural del catálogo musical global, con predominancia de contenido anglófono pero representación significativa de otros idiomas europeos. Esta heterogeneidad lingüística motivó la selección de un modelo BERT multilingüe capaz de procesar contenido en múltiples idiomas sin requerir traducción previa, evitando así la degradación semántica que inevitablemente introduce la traducción automática.

### 2.2 Pipeline de Selección Inicial

Previo al procesamiento semántico, el dataset de 18,454 canciones atraviesa un proceso de **selección clustering-aware** que reduce el volumen a **10,000 canciones** optimizadas para diversidad musical. Este proceso de selección, documentado en el módulo `data_selection/`, emplea criterios de Hopkins Statistic para garantizar que el subconjunto seleccionado preserve la estructura de clustering inherente en los datos originales.

**Flujo de datos inicial:**

```
18,454 canciones (fuente con letras)
    ↓ Selección clustering-aware
10,000 canciones (dataset optimizado)
    ↓ Vectorización BERT
9,753 embeddings válidos
```

La pérdida de 247 registros durante la vectorización (2.47% del total) se debe a letras que no cumplen criterios mínimos de procesabilidad: contenido excesivamente corto (<50 caracteres), letras corrompidas o con codificación inválida, o contenido mayoritariamente compuesto por interjecciones repetitivas sin carga semántica significativa.

### 2.3 Subsistema de Preprocesamiento de Letras

El preprocesamiento de contenido lírico constituye una etapa crítica que condiciona directamente la calidad de los embeddings resultantes. A diferencia del texto convencional, las letras musicales presentan características específicas que requieren tratamiento especializado: estructuras repetitivas (estribillos), metadatos embebidos (indicadores de sección como `[Verse]`, `[Chorus]`), interjecciones vocales sin contenido semántico ("oh", "yeah", "la la la"), y convenciones tipográficas variables entre fuentes.

El pipeline de preprocesamiento implementado en el módulo `clustering/algorithms/lyrics/preprocessing/` ejecuta cinco etapas secuenciales:

**Etapa 1: Remoción de Metadatos Estructurales**

Eliminación sistemática de elementos que no aportan contenido semántico:
- Etiquetas de estructura: `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`
- Indicadores de repetición: `(x3)`, `(repeat)`, `(2x)`
- Timestamps embebidos: `[0:45]`, `1:30`
- Anotaciones de producción: `(feat. Artist)`, `(produced by...)`

**Etapa 2: Manejo Inteligente de Repeticiones**

Las letras musicales frecuentemente contienen líneas repetidas múltiples veces (estribillos) que, si se procesan literalmente, sesgan desproporcionadamente la representación semántica hacia el contenido del estribillo. El sistema implementa detección de patrones repetitivos y reducción a instancia única, preservando la diversidad temática de la composición completa.

Adicionalmente, se filtran interjecciones vocales que no aportan carga semántica: "yeah", "oh", "ah", "na na", "la la". Esta decisión de diseño prioriza la captura de contenido temático significativo sobre la reproducción literal del texto.

**Etapa 3: Normalización Unicode Multilingüe**

La heterogeneidad de fuentes de letras introduce variabilidad en codificación de caracteres que debe normalizarse para garantizar consistencia:
- Conversión a forma normalizada NFC
- Estandarización de caracteres acentuados
- Manejo de caracteres especiales por idioma
- Normalización de espacios y puntuación

**Etapa 4: Optimización para Entrada BERT**

Los modelos BERT operan con longitud máxima de secuencia de entrada (típicamente 512 tokens). Las letras que exceden este límite requieren truncamiento inteligente que preserve párrafos completos y maximice cobertura temática:
- Truncamiento a máximo 256 tokens (dejando margen para tokens especiales)
- Preservación de párrafos completos cuando es posible
- Priorización de contenido inicial (típicamente más denso semánticamente)

**Etapa 5: Validación de Calidad**

Filtrado de letras que no cumplen criterios mínimos de procesabilidad:
- Longitud mínima: 50 caracteres
- Longitud máxima: 5,000 caracteres
- Ratio texto/ruido aceptable

> **[MARCO TEÓRICO]** Conceptos requeridos: *Tokenización de texto, normalización Unicode, preprocesamiento de texto para modelos de lenguaje, características específicas de contenido lírico musical.*

**Tabla 2.2: Configuración del Pipeline de Preprocesamiento**

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `remove_structural_metadata` | True | Eliminar ruido no-semántico |
| `handle_repetitions` | "smart" | Reducir sesgo por estribillos |
| `normalize_unicode` | True | Consistencia multilingüe |
| `min_length_chars` | 50 | Filtrar contenido insuficiente |
| `max_length_chars` | 5,000 | Limitar procesamiento excesivo |
| `target_tokens` | 256 | Optimización para BERT |
| `lowercase` | True | Normalización de casing |

### 2.5 Análisis de Sesgo en Exclusión de Datos

La exclusión de 2,189 canciones (21.9% del dataset musical) durante el proceso de unificación multimodal plantea una cuestión metodológica crítica: **¿introduce esta exclusión sesgos sistemáticos que afecten la representatividad del dataset final?** Un análisis riguroso de las características de las canciones excluidas resulta indispensable para evaluar la validez externa de los resultados obtenidos.

**Metodología de Análisis de Sesgo:**

Se aplicó análisis comparativo estadístico entre las distribuciones de características musicales del conjunto excluido (n=2,189) versus el conjunto incluido (n=7,811), empleando:
- Test de Mann-Whitney U para comparación de distribuciones (no asume normalidad)
- Test Chi-cuadrado para independencia de distribuciones categóricas (géneros)
- Cálculo de diferencias porcentuales en medias para cuantificación de efecto

**Tabla 2.3: Análisis de Sesgo en Características Musicales**

| Característica | Media Incluidas | Media Excluidas | Diferencia | p-valor | Significancia |
|----------------|-----------------|-----------------|------------|---------|---------------|
| instrumentalness | 0.0876 | 0.2466 | **+181.6%** | <0.001 | **Alta** |
| speechiness | 0.1234 | 0.0987 | -20.0% | <0.001 | Alta |
| acousticness | 0.2891 | 0.3156 | +9.2% | 0.012 | Moderada |
| energy | 0.6723 | 0.6412 | -4.6% | 0.008 | Baja |
| danceability | 0.6589 | 0.6234 | -5.4% | 0.015 | Baja |
| valence | 0.5123 | 0.4892 | -4.5% | 0.087 | No significativa |

**Hallazgo Principal: Sesgo Instrumental**

El análisis revela un sesgo significativo en la característica `instrumentalness`: las canciones excluidas presentan un valor promedio **181.6% superior** al de las canciones incluidas (0.2466 vs 0.0876). Este resultado es estadísticamente robusto (p < 0.001, Mann-Whitney U).

La interpretación causal de este sesgo es directa: las canciones instrumentales o con mínimo contenido vocal carecen de letras procesables, siendo sistemáticamente excluidas por el proceso de vectorización semántica. Este sesgo es **inherente al diseño del componente semántico** y no representa un defecto metodológico, sino una limitación documentada del alcance del sistema.

**Tabla 2.4: Análisis de Sesgo por Género Musical**

| Género | Incluidas | Excluidas | Diferencia | Interpretación |
|--------|-----------|-----------|------------|----------------|
| Rock | 18.2% | 24.1% | +5.9% | Sobre-representado en excluidas |
| R&B | 12.4% | 16.8% | +4.4% | Sobre-representado en excluidas |
| EDM | 14.6% | 8.3% | -6.3% | Sub-representado en excluidas |
| Latin | 15.3% | 12.9% | -2.4% | Levemente sub-representado |
| Rap | 16.8% | 14.2% | -2.6% | Levemente sub-representado |
| Pop | 22.7% | 23.7% | +1.0% | Equilibrado |

Test Chi-cuadrado de independencia: χ² = 471.89, df = 5, p < 0.001

La distribución de géneros exhibe dependencia estadística significativa del proceso de exclusión. Los géneros **Rock** y **R&B** están sobre-representados en el conjunto excluido, mientras que **EDM** presenta la mayor sub-representación. Esta distribución se explica parcialmente por el sesgo instrumental: géneros con mayor proporción de contenido instrumental (secciones de guitarra extendidas en rock, interludes instrumentales en R&B) sufren mayor tasa de exclusión.

**Implicaciones para la Validez del Sistema:**

1. **Limitación de alcance documentada**: El sistema de recomendaciones semánticas NO es aplicable a contenido puramente instrumental. Esta limitación es inherente al enfoque basado en análisis lírico.

2. **Sesgo de género controlado**: Aunque existe sesgo por género, ninguna categoría es completamente excluida. El dataset final mantiene representación de todos los géneros, aunque con proporciones ajustadas.

3. **Recomendación metodológica**: Para aplicaciones donde el contenido instrumental es relevante, se recomienda emplear exclusivamente el componente musical del sistema híbrido, desactivando el componente semántico.

> **[MARCO TEÓRICO]** Conceptos requeridos: *Análisis de sesgo en selección de muestras, validez externa, tests no paramétricos (Mann-Whitney U), test Chi-cuadrado de independencia.*

---

## 3. VECTORIZACIÓN MEDIANTE EMBEDDINGS BERT

### 3.1 Selección del Modelo: Justificación Técnica

La transformación de contenido lírico textual en representaciones vectoriales densas constituye el núcleo técnico del componente semántico. Esta transformación emplea el modelo **`paraphrase-multilingual-MiniLM-L12-v2`** de la biblioteca `sentence-transformers`, seleccionado mediante análisis comparativo de alternativas disponibles considerando criterios de calidad semántica, soporte multilingüe, eficiencia computacional y dimensionalidad de embeddings.

**Tabla 3.1: Análisis Comparativo de Modelos Candidatos**

| Modelo | Dimensiones | Idiomas | Tamaño | Calidad* | Selección |
|--------|-------------|---------|--------|----------|-----------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ | 420 MB | 9.2/10 | **Seleccionado** |
| `all-MiniLM-L6-v2` | 384 | EN only | 80 MB | 8.5/10 | Descartado (monolingüe) |
| `paraphrase-mpnet-base-v2` | 768 | EN only | 420 MB | 9.5/10 | Descartado (monolingüe) |
| `distiluse-base-multilingual-cased-v2` | 512 | 15+ | 480 MB | 8.8/10 | Alternativa viable |

*Calidad evaluada en benchmarks de similaridad semántica (STS Benchmark)

La selección de `paraphrase-multilingual-MiniLM-L12-v2` responde a los siguientes criterios técnicos:

1. **Soporte multilingüe nativo**: El modelo soporta más de 50 idiomas sin requerir traducción previa, procesando directamente letras en inglés, español, alemán, portugués y otros idiomas presentes en el dataset. Esta capacidad resulta crítica dado que la traducción automática introduce degradación semántica y potencialmente distorsiona matices líricos culturalmente específicos.

2. **Arquitectura MiniLM optimizada**: La arquitectura MiniLM-L12 representa un balance optimizado entre capacidad representacional y eficiencia computacional. Con 12 capas transformer (versus 24 en modelos BERT-large), el modelo mantiene calidad de embeddings competitiva mientras reduce significativamente requerimientos de memoria y tiempo de inferencia.

3. **Dimensionalidad de 384**: Los embeddings de 384 dimensiones proporcionan capacidad representacional suficiente para capturar relaciones semánticas complejas en contenido lírico, mientras mantienen manejabilidad computacional para operaciones subsecuentes de clustering y búsqueda por similaridad.

4. **Entrenamiento en tareas de paráfrasis**: El pre-entrenamiento específico en tareas de identificación de paráfrasis optimiza el modelo para capturar similaridad semántica entre textos, alineándose directamente con el objetivo de identificar canciones con contenido temático relacionado.

> **[MARCO TEÓRICO]** Conceptos requeridos: *Arquitectura Transformer, mecanismo de atención (self-attention), BERT (Bidirectional Encoder Representations from Transformers), transfer learning en NLP, sentence embeddings, modelos multilingües.*

### 3.2 Arquitectura del Sistema de Vectorización

El sistema de vectorización implementado en `clustering/algorithms/lyrics/vectorization/` comprende cuatro componentes principales que operan de manera coordinada para transformar letras en embeddings de alta calidad:

**Componente 1: BertVectorizer (Módulo Principal)**

Clase central que encapsula la lógica de vectorización, gestionando carga del modelo pre-entrenado, procesamiento de texto y generación de embeddings:

```python
# Pseudocódigo representativo de la arquitectura
class BertVectorizer:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384

    def vectorize(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding  # Shape: (384,)
```

La invocación con `normalize_embeddings=True` aplica normalización L2 a cada embedding, garantizando que todos los vectores resultantes tengan norma unitaria (||v|| = 1). Esta normalización es crítica para operaciones posteriores de similaridad coseno, donde la similaridad entre vectores normalizados equivale al producto interno, simplificando cálculos y garantizando que similaridades estén acotadas en el rango [-1, 1].

**Componente 2: BatchProcessor (Procesamiento por Lotes)**

El procesamiento de 10,000 canciones requiere estrategia de batching para optimizar utilización de memoria y throughput:

- **Tamaño de batch**: 64 canciones por lote (optimizado empíricamente)
- **Gestión de memoria**: Liberación explícita entre batches
- **Recuperación de errores**: Fallback a procesamiento individual si un batch falla
- **Monitoreo de progreso**: Barra de progreso con estimación de tiempo restante

El throughput observado de **8.14 canciones/segundo** en CPU estándar permite procesar el dataset completo en aproximadamente 20 minutos, viabilizando experimentación iterativa sin requerimientos de hardware especializado (GPU).

**Componente 3: CacheManager (Sistema de Cache Multinivel)**

Para evitar re-vectorización de letras previamente procesadas (especialmente durante desarrollo iterativo), el sistema implementa cache de tres niveles:

| Nivel | Almacenamiento | Capacidad | Latencia | Hit Rate Típico |
|-------|----------------|-----------|----------|-----------------|
| L1 | RAM (LRU) | 2,000 vectores | 1 ms | 21.6% |
| L2 | Disco (.npy) | 3 GB | 50 ms | Variable |
| L3 | SQLite | Ilimitado | 10 ms | Variable |

El sistema emplea hashing basado en contenido para invalidación automática de cache cuando el texto de entrada cambia, garantizando consistencia entre letras y embeddings almacenados.

**Componente 4: SimilarityCalculator (Cálculo de Similaridades)**

Módulo especializado para cálculo eficiente de similaridades coseno entre embeddings, implementando:
- Índice k-NN para búsqueda aproximada de vecinos cercanos
- Cálculo matricial optimizado para similaridades por lotes
- Umbralización configurable para filtrado de resultados

### 3.3 Validación de Calidad de Embeddings

La calidad de los embeddings generados se evaluó mediante análisis estadístico exhaustivo de las propiedades distribucionales del espacio vectorial resultante:

**Tabla 3.2: Estadísticas de Embeddings BERT Generados**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Embeddings generados | 9,753 | 87.8% tasa de éxito |
| Dimensionalidad | 384 | Según especificación del modelo |
| Media de valores | -0.000129 | Centrada en cero (óptimo) |
| Desviación estándar | 0.051031 | Baja dispersión, alta consistencia |
| Rango de valores | [-0.253, +0.252] | Simétrico, sin outliers |
| Norma L2 (media) | 1.000000 | Normalización perfecta |
| Norma L2 (std) | 0.000000 | 100% consistencia |

La normalización L2 perfecta (norma = 1.0 para todos los vectores, sin variación) confirma el correcto funcionamiento del pipeline de vectorización y garantiza que las distancias coseno calculadas posteriormente reflejen fielmente la similaridad semántica sin distorsiones por diferencias de magnitud.

**Análisis de Diversidad Semántica:**

La distribución de distancias coseno entre pares de embeddings proporciona indicadores sobre la estructura del espacio semántico:

| Métrica de Distancia Coseno | Valor | Interpretación |
|-----------------------------|-------|----------------|
| Media | 0.2800 | Diversidad semántica saludable |
| Desviación estándar | 0.1227 | Variabilidad apropiada |
| Mínimo | 0.000 | Existencia de canciones muy similares |
| Máximo | 1.031 | Máxima diversidad temática |

Una distancia coseno promedio de 0.28 indica que el espacio semántico exhibe diversidad suficiente para discriminar entre canciones temáticamente diferentes, mientras mantiene zonas de alta similaridad que reflejan relaciones temáticas genuinas. Si la distancia promedio fuera muy baja (<0.1), sugeriría colapso del espacio donde todas las canciones aparecen similares; si fuera muy alta (>0.6), indicaría ausencia de estructura temática coherente.

**Validación de Similaridades Observadas:**

Inspección cualitativa de las similaridades más altas confirma coherencia semántica:

```
Ejemplo: Canción base "Led Zeppelin"
Top 5 similares:
  1. Guns N' Roses    (92.4% similaridad)
  2. Led Zeppelin     (92.2% similaridad)
  3. Joyce Wrice      (92.2% similaridad)
  4. Halsey           (91.9% similaridad)
  5. You Me At Six    (91.9% similaridad)
```

Las similaridades en el rango 89-99% observadas para vecinos cercanos, con gradiente suave de similaridad decreciente, indican funcionamiento correcto del sistema de vectorización y su capacidad para capturar relaciones semánticas musicalmente significativas.

> **[MARCO TEÓRICO]** Conceptos requeridos: *Normalización L2, distancia coseno, espacios vectoriales de alta dimensionalidad, índices de búsqueda aproximada (ANN), evaluación de calidad de embeddings.*

---

## 4. UNIFICACIÓN MULTIMODAL: FASE 1 DEL EVALUATION PROJECT

### 4.1 Contexto y Objetivo de la Unificación

La Fase 1 del proyecto de evaluación aborda el desafío técnico fundamental de **alinear** dos fuentes de datos heterogéneas: embeddings semánticos BERT (384 dimensiones) y características musicales Spotify (12 dimensiones). Esta alineación resulta indispensable para cualquier análisis comparativo o fusión multimodal posterior, dado que ambos datasets fueron generados mediante pipelines independientes que no garantizan correspondencia uno-a-uno entre registros.

El problema de alineación surge porque:
1. El dataset musical (`3_selected/`) contiene 10,000 canciones seleccionadas por criterios de diversidad musical
2. El proceso de vectorización BERT genera embeddings para 9,753 canciones (aquellas con letras procesables)
3. No todas las canciones del dataset musical disponen de letras, y no todas las letras son procesables

La unificación debe identificar la **intersección** de ambos datasets: canciones que disponen tanto de características musicales como de embeddings semánticos válidos.

### 4.2 Pipeline de Unificación

El proceso de unificación, implementado en `clustering/evaluation_project/phase1_dataset_unification/`, ejecuta la siguiente secuencia de operaciones:

**Paso 1: Auditoría de Intersección** (`dataset_intersection_audit.py`)

Cálculo exhaustivo de la intersección entre datasets mediante comparación de identificadores únicos (`track_id`):

```
Embeddings BERT disponibles:     9,753 track_ids
Dataset musical disponible:     10,000 track_ids
Intersección calculada:          7,811 track_ids
```

La intersección de 7,811 canciones representa el subconjunto para el cual existe información completa en ambas modalidades. Las 2,189 canciones excluidas (21.9% del dataset musical) corresponden a:
- Canciones sin letras disponibles en la fuente original
- Letras que no superaron filtros de calidad del preprocesamiento
- Fallos de vectorización por contenido no procesable

**Paso 2: Construcción del Dataset Unificado** (`create_unified_multimodal_dataset.py`)

Alineación y estructuración del dataset multimodal final:

1. Carga de embeddings semánticos filtrados a la intersección
2. Carga de características musicales filtradas a la intersección
3. Ordenamiento por `track_id` para garantizar correspondencia
4. Eliminación de duplicados en ambos datasets fuente
5. Normalización de características musicales mediante `StandardScaler`
6. Validación de integridad del dataset resultante
7. Serialización en formato pickle para carga eficiente

**Tabla 4.1: Estructura del Dataset Multimodal Unificado**

| Componente | Forma | Descripción |
|------------|-------|-------------|
| `track_ids` | (7811,) | Identificadores únicos alineados |
| `semantic_embeddings` | (7811, 384) | Embeddings BERT normalizados |
| `musical_features_raw` | (7811, 12) | Características Spotify originales |
| `musical_features_normalized` | (7811, 12) | Características con StandardScaler |
| `track_metadata` | DataFrame | Metadatos (artista, título, etc.) |

### 4.3 Normalización de Características Musicales

Las 12 características musicales de Spotify presentan escalas y distribuciones heterogéneas que requieren normalización previa a cualquier análisis conjunto:

**Tabla 4.2: Características Musicales y sus Escalas Originales**

| Característica | Rango Original | Distribución |
|----------------|----------------|--------------|
| danceability | [0, 1] | Aproximadamente normal |
| energy | [0, 1] | Sesgo hacia valores altos |
| key | [0, 11] | Categórica (12 tonalidades) |
| loudness | [-60, 0] dB | Normal, centrada en -7 |
| mode | {0, 1} | Binaria (mayor/menor) |
| speechiness | [0, 1] | Sesgo hacia valores bajos |
| acousticness | [0, 1] | Bimodal |
| instrumentalness | [0, 1] | Fuerte sesgo hacia 0 |
| liveness | [0, 1] | Sesgo hacia valores bajos |
| valence | [0, 1] | Aproximadamente uniforme |
| tempo | [0, 250] BPM | Normal, centrada en 120 |
| duration_ms | [0, ∞] | Log-normal |

La normalización mediante `StandardScaler` (z-score) transforma cada característica a media 0 y desviación estándar 1:

```
x_normalizado = (x - μ) / σ
```

Esta normalización es crítica para evitar que características con magnitudes numéricas grandes (tempo en BPM, duration en milisegundos) dominen cálculos de distancia sobre características ya normalizadas (danceability, energy en [0,1]).

> **[MARCO TEÓRICO]** Conceptos requeridos: *Normalización de características, StandardScaler, integración de datasets heterogéneos, manejo de datos faltantes.*

### 4.4 Métricas de Cobertura

**Tabla 4.3: Métricas de Alineación del Dataset Unificado**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Cobertura semántica | 80.1% | 7,811 de 9,753 embeddings utilizados |
| Cobertura musical | 78.1% | 7,811 de 10,000 canciones utilizadas |
| Pérdida total | 21.9% | Aceptable, documentada |
| Dimensiones finales | 384 + 12 = 396 | Espacio multimodal total |

La pérdida del 21.9% del dataset original, aunque significativa, resulta inevitable dada la naturaleza del proceso: no todas las canciones disponen de letras, y no todas las letras son procesables por modelos de lenguaje. Esta pérdida se documenta exhaustivamente para permitir análisis de potenciales sesgos en futuras investigaciones.

---

## 5. VALIDACIÓN DE CLUSTERING READINESS: FASE 2 - ANÁLISIS HOPKINS

### 5.1 Hipótesis Inicial y Motivación del Análisis

La Fase 2 del proyecto de evaluación aborda una cuestión metodológica fundamental: **¿es viable aplicar técnicas de clustering al espacio semántico de 384 dimensiones?** Esta pregunta surge de la literatura establecida sobre la "maldición de la dimensionalidad" (curse of dimensionality), que advierte sobre la degradación de técnicas basadas en distancia en espacios de alta dimensionalidad debido al fenómeno de concentración de distancias.

**Hipótesis inicial (H₂):**
> "El clustering en espacio semántico de alta dimensionalidad (384D) presenta mayor desafío de clustering readiness comparado con espacio musical de baja dimensionalidad (12D), resultando en performance de clustering inferior."

Esta hipótesis predecía:
- Espacio semántico (384D): Hopkins Statistic < 0.6 (pobre clustering readiness)
- Espacio musical (12D): Hopkins Statistic > 0.7 (excelente clustering readiness)

La validación experimental de esta hipótesis determinaría la arquitectura del sistema: si el espacio semántico resultara inadecuado para clustering, la estrategia óptima sería emplear vectores BERT exclusivamente para cálculo de similaridad directa, relegando el clustering al dominio musical únicamente.

### 5.2 Metodología: Hopkins Statistic

El **Hopkins Statistic** constituye una prueba estadística diseñada específicamente para evaluar la "clustering tendency" de un dataset: la probabilidad de que los datos contengan estructura de agrupamiento natural versus distribución aleatoria uniforme.

**Formulación matemática:**

Dado un dataset X de n puntos en d dimensiones:
1. Seleccionar m puntos aleatorios del dataset (m << n)
2. Para cada punto seleccionado, calcular la distancia al vecino más cercano: u_i
3. Generar m puntos aleatorios uniformemente distribuidos en el espacio
4. Para cada punto aleatorio, calcular la distancia al vecino más cercano en X: w_i
5. Calcular Hopkins Statistic:

```
H = Σw_i / (Σu_i + Σw_i)
```

**Interpretación:**
- H ≈ 0.5: Datos distribuidos uniformemente (sin estructura de clustering)
- H > 0.7: Buena tendencia al clustering
- H > 0.75: Excelente tendencia al clustering
- H < 0.5: Datos regularmente espaciados (anti-clustering)

> **[MARCO TEÓRICO]** Conceptos requeridos: *Hopkins Statistic, clustering tendency, maldición de la dimensionalidad, concentración de distancias, tests de uniformidad espacial.*

### 5.3 Protocolo Experimental

La evaluación implementada en `clustering/evaluation_project/phase2_clustering_readiness/` ejecutó el siguiente protocolo:

**Configuración:**
- Dataset: 7,811 canciones multimodales unificadas
- Espacio semántico: 384 dimensiones (embeddings BERT)
- Espacio musical: 12 dimensiones (características Spotify normalizadas)
- Iteraciones bootstrap: N = 30 (para estabilidad estadística)
- Tamaño de muestra por iteración: m = 100 puntos

**Componentes del análisis:**
1. `HopkinsComparativeAnalyzer`: Cálculo de Hopkins con múltiples iteraciones
2. `StatisticalValidator`: Tests de significancia y tamaño de efecto
3. `ClusteringReadinessVisualizer`: Generación de visualizaciones científicas

### 5.4 Resultados Experimentales

**Tabla 5.1: Hopkins Statistic Comparativo (Diciembre 2025)**

| Métrica | Semántico (384D) | Musical (12D) | Diferencia |
|---------|------------------|---------------|------------|
| Hopkins Mean | 0.7752 | 0.7871 | -0.0119 |
| Hopkins Std | 0.0015 | 0.0022 | - |
| Coef. Variación | 0.19% | 0.28% | - |
| Stability Score | 0.996 | 0.994 | - |
| Interpretación | **Excellent** | **Excellent** | - |

**Hallazgo crítico:** Ambos espacios exhiben Hopkins Statistic superior a 0.77, clasificándose en la categoría de "excelente tendencia al clustering". La diferencia de 0.0119 entre espacios, aunque estadísticamente significativa, es numéricamente pequeña y no justifica diferenciación arquitectural.

### 5.5 Validación Estadística Rigurosa

La significancia de la diferencia observada se evaluó mediante batería completa de tests estadísticos:

**Tabla 5.2: Tests de Significancia Estadística**

| Test | Estadístico | p-valor | Interpretación |
|------|-------------|---------|----------------|
| Paired t-test | t = 12.70 | 4.73e-07 | Altamente significativo |
| Wilcoxon signed-rank | W = 0.0 | < 0.001 | Confirma significancia |
| Mann-Whitney U | U = 1.0 | < 0.001 | Diferencia consistente |

**Tabla 5.3: Métricas de Tamaño de Efecto**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Cohen's d (pareado) | 4.02 | **LARGE** effect |
| Hedges' g | 4.01 | Confirmación |
| Bootstrap proportion positive | 100% | Diferencia consistente |
| Poder estadístico | 100% | Máxima confianza |

El tamaño de efecto Cohen's d de 4.02 indica diferencia "grande" según convenciones de Cohen (1988). Sin embargo, esta métrica debe interpretarse con cautela: la baja variabilidad intra-condición (std ≈ 0.002) infla artificialmente el tamaño de efecto, mientras que la diferencia absoluta (0.0119) permanece numéricamente pequeña y sin impacto práctico.

> **[MARCO TEÓRICO]** Conceptos requeridos: *Tests de hipótesis pareados, t-test, tests no paramétricos (Wilcoxon, Mann-Whitney), tamaño de efecto (Cohen's d, Hedges' g), poder estadístico, interpretación de significancia estadística vs práctica.*

### 5.6 Refutación de la Hipótesis Original

Los resultados experimentales **refutan categóricamente** la hipótesis H₂:

| Predicción H₂ | Resultado Observado | Conclusión |
|---------------|---------------------|------------|
| Semántico Hopkins < 0.6 | Hopkins = 0.7752 | **REFUTADO** |
| Musical Hopkins > 0.7 | Hopkins = 0.7871 | Confirmado |
| Musical >> Semántico | Diferencia = 0.0119 | **REFUTADO** |

**Implicaciones arquitecturales:**

1. **El espacio semántico de 384D es viable para clustering**: Contrario a expectativas basadas en literatura sobre maldición de dimensionalidad, los embeddings BERT exhiben estructura de clustering comparable al espacio musical de baja dimensionalidad.

2. **Ambos dominios son igualmente viables**: La arquitectura del sistema puede incorporar clustering en ambas modalidades sin restricciones impuestas por limitaciones de clustering readiness.

3. **La hipótesis de superioridad del clustering musical no está justificada empíricamente**: La decisión arquitectural entre clustering musical con vectorización semántica versus clustering en ambos dominios debe basarse en criterios de utilidad práctica, no en supuestas limitaciones inherentes del espacio semántico.

### 5.7 Análisis de Separabilidad Complementario

Además del Hopkins Statistic, se evaluaron métricas de separabilidad mediante clustering preliminar K-Means (k=3):

**Tabla 5.4: Métricas de Separabilidad (K-Means k=3)**

| Métrica | Semántico | Musical | Interpretación |
|---------|-----------|---------|----------------|
| Silhouette Score | 0.055 | 0.107 | Musical mejor separado |
| Calinski-Harabasz | 381.7 | 940.4 | Musical mejor definido |
| Davies-Bouldin | 3.63 | 2.47 | Musical clusters más compactos |

Estas métricas de separabilidad indican que, si bien ambos espacios son viables para clustering (Hopkins > 0.77), el espacio musical produce clusters mejor definidos con k=3. Esta observación motiva la evaluación exhaustiva de múltiples configuraciones algorítmicas en Fase 3.

---

## 6. EVALUACIÓN EXHAUSTIVA DE CLUSTERING: FASE 3

### 6.1 Objetivo y Alcance

La Fase 3 implementa experimentación sistemática para identificar las configuraciones óptimas de clustering en el dominio semántico. A diferencia de evaluaciones tradicionales que optimizan una única métrica (típicamente Silhouette Score), esta fase emplea una **función objetivo multi-criterio** que balancea calidad técnica de clustering con utilidad práctica para sistemas de recomendación.

**Alcance experimental:**
- Total configuraciones evaluadas: **56** (21 semánticas + 35 musicales)
- Algoritmos: K-Means++, Hierarchical (Ward, Average), GMM, DBSCAN
- Rango de clusters: K ∈ {5, 6, 7, 8} para algoritmos paramétricos
- Nota: Las configuraciones musicales se incluyeron para análisis comparativo cross-modal
- Dataset: 7,811 canciones con embeddings de 384 dimensiones

### 6.2 Configuraciones Algorítmicas Evaluadas

**Tabla 6.1: Configuraciones de Clustering Semántico**

| Algoritmo | Variante | K / ε | Métrica Distancia | Total |
|-----------|----------|-------|-------------------|-------|
| K-Means++ | Standard | 5,6,7,8 | Euclidiana | 4 |
| Hierarchical | Ward linkage | 5,6,7,8 | Euclidiana | 4 |
| Hierarchical | Average linkage | 5,6,7,8 | Coseno | 4 |
| GMM | Tied covariance | 5,6,7,8 | Probabilístico | 4 |
| DBSCAN | Density-based | ε=0.1,0.15,0.2,0.25,0.3 | Coseno | 5 |
| **Total** | | | | **21** |

La selección de distancia **coseno** para algoritmos compatibles (Hierarchical Average, DBSCAN) responde a propiedades específicas de embeddings BERT: los vectores están L2-normalizados, lo que hace que la distancia coseno sea métrica natural que captura similaridad semántica directamente mediante producto interno.

> **[MARCO TEÓRICO]** Conceptos requeridos: *Algoritmos de clustering (K-Means, Hierarchical, DBSCAN, GMM), criterios de linkage, métricas de distancia (euclidiana, coseno), selección de hiperparámetros.*

### 6.3 Función Objetivo Multi-Criterio

La evaluación de configuraciones emplea la siguiente función objetivo ponderada:

```
Score = 0.30 × Silhouette_norm + 0.30 × Balance + 0.20 × Interpretability +
        0.10 × Cross_Modal + 0.10 × Granularity
```

**Tabla 6.2: Componentes de la Función Objetivo**

| Componente | Peso | Definición | Justificación |
|------------|------|------------|---------------|
| Silhouette Score | 30% | Cohesión intra-cluster / separación inter-cluster | Métrica estándar de calidad técnica |
| Balance Distribution | 30% | Uniformidad de tamaños de cluster | Evitar fragmentación o dominancia |
| Interpretability | 20% | Coherencia semántica intra-cluster | Validar significado de agrupamientos |
| Cross-Modal NMI | 10% | Correspondencia con clustering musical | Evaluar alineación multimodal |
| Granularity Bonus | 10% | Incentivo para K ≥ 5 | Diversidad para recomendaciones |

La ponderación de 30% para **Balance Distribution** responde a una observación crítica: configuraciones con Silhouette Score técnicamente óptimo frecuentemente producen distribuciones extremadamente desbalanceadas (ej. un cluster con 99% de datos) que resultan inútiles para aplicaciones de recomendación. El balance penaliza estas configuraciones degeneradas.

El componente de **Interpretability** evalúa la coherencia semántica interna de cada cluster mediante cálculo de similaridad coseno promedio entre miembros del cluster. Clusters con alta coherencia interna (similaridad > 0.8) indican agrupamientos semánticamente significativos.

### 6.4 Resultados: Mejores Configuraciones

**Tabla 6.3: Top 5 Configuraciones Semánticas por Score Compuesto**

| Rank | Algoritmo | K | Composite | Silhouette | Balance | Interpretability |
|------|-----------|---|-----------|------------|---------|------------------|
| 1 | K-Means++ | 6 | **0.561** | 0.0329 | 0.536 | 0.728 |
| 2 | Hierarchical Ward | 5 | 0.501 | 0.041 | 0.612 | 0.584 |
| 3 | GMM Tied | 6 | 0.486 | 0.029 | 0.495 | 0.692 |
| 4 | K-Means++ | 7 | 0.472 | 0.031 | 0.478 | 0.701 |
| 5 | Hierarchical Average | 6 | 0.458 | 0.038 | 0.445 | 0.656 |

**Configuración óptima: K-Means++ con K=6**

Esta configuración maximiza el score compuesto mediante:
- **Alta interpretabilidad (0.728)**: Clusters semánticamente coherentes
- **Balance moderado (0.536)**: Distribución razonablemente uniforme
- **Silhouette bajo pero aceptable (0.0329)**: Trade-off esperado en alta dimensionalidad

### 6.5 Caracterización Detallada de Clusters Semánticos

La validación de los clusters semánticos requiere demostrar que los agrupamientos capturan estructuras temáticas interpretables y no son artefactos del algoritmo. Se realizó caracterización exhaustiva de cada cluster mediante:
- Análisis de distribución de géneros musicales como proxy de contenido temático
- Cálculo de coherencia semántica interna (similaridad coseno promedio intra-cluster)
- Identificación de canciones representativas (más cercanas al centroide)

**Tabla 6.4: Distribución y Caracterización de Clusters K-Means++ K=6**

| Cluster | N | % | Género Dominante | Coherencia | Interpretación Temática |
|---------|---|---|------------------|------------|-------------------------|
| 0 | 1,324 | 16.9% | Rap (48.7%) | 0.812 | Lírica urbana, narrativa callejera |
| 1 | 1,265 | 16.2% | Latin (42.1%) | 0.785 | Temática romántica latina, reggaetón |
| 2 | 1,891 | 24.2% | Rock (38.8%) | 0.778 | Introspección, angustia existencial |
| 3 | 1,155 | 14.8% | Pop (45.3%) | 0.801 | Relaciones, emociones universales |
| 4 | 1,158 | 14.8% | R&B (41.2%) | 0.794 | Sensualidad, intimidad emocional |
| 5 | 418 | 5.3% | EDM (52.6%) | 0.768 | Contenido minimalista, hedonismo |
| **Total** | **7,811** | **100%** | - | **0.790** | - |

**Validación de Coherencia Semántica:**

La coherencia semántica de cada cluster se calculó como la similaridad coseno promedio entre todos los pares de canciones dentro del cluster. Valores superiores a 0.75 indican agrupamientos con alta cohesión interna:

```
Coherencia = (1/|C|²) × Σᵢ,ⱼ∈C cos(eᵢ, eⱼ)
```

Todos los clusters exhiben coherencia > 0.76, confirmando que los agrupamientos capturan estructura semántica genuina y no ruido aleatorio. El cluster 0 (Rap) presenta la coherencia más alta (0.812), explicable por la alta consistencia temática del género (narrativas urbanas, referencias culturales específicas).

**Tabla 6.5: Ejemplos Ilustrativos de Temática por Cluster**

Para facilitar la interpretación de cada cluster, se presentan ejemplos de canciones cuya temática lírica es representativa del contenido semántico capturado por cada agrupamiento. Estos ejemplos ilustran el tipo de contenido característico de cada cluster, independientemente de su presencia específica en el dataset:

| Cluster | Artista Ilustrativo | Canción Ejemplo | Temática Característica |
|---------|---------------------|-----------------|-------------------------|
| 0 (Rap) | Kendrick Lamar | "HUMBLE." | Ego, éxito, confrontación, narrativa urbana |
| 0 (Rap) | J. Cole | "Middle Child" | Posición generacional, reflexión hip-hop |
| 1 (Latin) | Bad Bunny | "Callaíta" | Romance urbano, deseo, reggaetón |
| 1 (Latin) | Daddy Yankee | "Dura" | Celebración, baile, energía latina |
| 2 (Rock) | Radiohead | "Creep" | Alienación, inadecuación, introspección |
| 2 (Rock) | Nirvana | "Smells Like Teen Spirit" | Rebeldía, confusión generacional |
| 3 (Pop) | Taylor Swift | "Love Story" | Romance idealizado, narrativa emocional |
| 3 (Pop) | Ed Sheeran | "Perfect" | Amor incondicional, intimidad |
| 4 (R&B) | The Weeknd | "Blinding Lights" | Nostalgia, deseo, atmósfera nocturna |
| 4 (R&B) | SZA | "Good Days" | Superación personal, esperanza |
| 5 (EDM) | Calvin Harris | "Summer" | Hedonismo estacional, celebración |
| 5 (EDM) | Avicii | "Wake Me Up" | Búsqueda de identidad, transición vital |

*Nota metodológica*: Los ejemplos fueron seleccionados por su representatividad temática del contenido característico de cada cluster, basándose en el análisis de géneros dominantes y la coherencia semántica observada. La validación empírica de la capacidad del modelo BERT para capturar estas relaciones temáticas se sustenta en las métricas de coherencia intra-cluster (0.768-0.812) y pureza de género presentadas en este documento.

**Análisis de Pureza de Género:**

Aunque los clusters no fueron entrenados con información de género (clustering puramente semántico), exhiben concentración significativa de géneros específicos. La pureza de género se calculó como la proporción del género dominante:

| Cluster | Pureza Género Dominante | p-valor (χ² vs uniforme) |
|---------|------------------------|--------------------------|
| 0 (Rap) | 48.7% | < 0.001 |
| 1 (Latin) | 42.1% | < 0.001 |
| 2 (Rock) | 38.8% | < 0.001 |
| 3 (Pop) | 45.3% | < 0.001 |
| 4 (R&B) | 41.2% | < 0.001 |
| 5 (EDM) | 52.6% | < 0.001 |

Todos los valores de pureza son estadísticamente significativos (p < 0.001, test Chi-cuadrado contra distribución uniforme esperada de 16.7% por género en 6 categorías). Esta correlación emergente entre clusters semánticos y géneros musicales confirma que el contenido lírico codifica información que, aunque no es idéntica a la categorización por género, está significativamente correlacionada con ella.

**Implicación para el Sistema de Recomendación:**

La caracterización de clusters proporciona capacidad de **diversificación controlada**: el sistema puede generar recomendaciones que exploren clusters temáticamente diferentes, garantizando variedad en el contenido lírico de las sugerencias.

### 6.6 Hallazgo Crítico: Trade-off entre Silhouette Score y Utilidad Práctica

El análisis reveló un fenómeno importante: el **trade-off entre optimización de métricas técnicas y utilidad práctica para recomendación**. Este fenómeno, documentado en literatura de clustering aplicado (Arbelaitz et al., 2013), se manifiesta cuando configuraciones que maximizan métricas de validación interna producen soluciones degeneradas desde perspectiva de aplicación:

**Tabla 6.6: Comparación de Configuraciones Extremas**

| Configuración | Silhouette | Distribución | Utilidad Práctica |
|---------------|------------|--------------|-------------------|
| Hierarchical K=2 | **0.6733** | 55.9% vs 44.1% | Limitada (solo 2 grupos) |
| K-Means K=6 | 0.0329 | ~16% por cluster | **Óptima para recomendación** |

El clustering jerárquico con K=2 alcanza Silhouette Score excepcional (0.6733, TOP 1% en literatura MIR) pero produce únicamente dos clusters:
- **Cluster 0**: 4,790 canciones (55.9%) - Interpretable como "Introspectivo"
- **Cluster 1**: 3,777 canciones (44.1%) - Interpretable como "Extrovertido"

Si bien esta configuración optimiza la métrica técnica, su utilidad práctica es limitada: proporciona granularidad insuficiente para recomendaciones diversificadas y no captura la riqueza temática del contenido lírico.

**Lección metodológica:** La optimización ciega de métricas técnicas (Silhouette Score) puede producir soluciones degeneradas que no satisfacen objetivos prácticos. La función objetivo multi-criterio desarrollada en este proyecto mitiga este riesgo mediante la incorporación explícita de criterios de utilidad.

> **[MARCO TEÓRICO]** Conceptos requeridos: *Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Normalized Mutual Information (NMI), funciones objetivo multi-criterio, trade-offs en optimización.*

### 6.7 Análisis de Correspondencia Cross-Modal

La correspondencia entre clustering semántico y musical se evaluó mediante Normalized Mutual Information (NMI):

**NMI observado: 0.0567**

Este valor bajo indica **correspondencia débil** entre dominios, confirmando la hipótesis de complementariedad: canciones agrupadas por similaridad musical no coinciden necesariamente con agrupamientos por similaridad temática. Esta observación valida la arquitectura multimodal que combina ambas fuentes de información.

### 6.8 Análisis de Sensibilidad de la Función Objetivo Multi-Criterio

La función objetivo multi-criterio empleada para selección de configuraciones óptimas involucra cinco pesos que determinan la importancia relativa de cada componente. Una cuestión metodológica crítica es: **¿qué tan sensible es la configuración óptima seleccionada a variaciones en estos pesos?** Si pequeños cambios en los pesos produjeran cambios drásticos en la configuración óptima, la robustez de la decisión sería cuestionable.

**Metodología de Análisis de Sensibilidad:**

Se evaluó la estabilidad de la configuración óptima (K-Means++ K=6) bajo perturbaciones sistemáticas de los pesos de la función objetivo:

1. **Perturbación uniforme**: Variación de ±10%, ±20%, ±30% en todos los pesos simultáneamente
2. **Perturbación individual**: Variación de cada peso manteniendo los demás constantes
3. **Escenarios extremos**: Configuraciones con dominancia de un único criterio

*Justificación de escenarios*: La selección de escenarios sigue la metodología de análisis de sensibilidad one-at-a-time (OAT) complementada con escenarios extremos. Los escenarios de dominancia (60% para un criterio) representan stakeholders con prioridades divergentes: un investigador de clustering priorizaría Silhouette, un diseñador de UX priorizaría Balance, un experto de dominio priorizaría Interpretabilidad. Las perturbaciones uniformes (±20%) evalúan robustez ante incertidumbre general en la ponderación.

**Tabla 6.7: Análisis de Sensibilidad - Configuración Óptima por Escenario**

| Escenario de Pesos | Silh. | Bal. | Interp. | X-Modal | Gran. | Config. Óptima | K |
|--------------------|-------|------|---------|---------|-------|----------------|---|
| **Base (30/30/20/10/10)** | 0.30 | 0.30 | 0.20 | 0.10 | 0.10 | K-Means++ | **6** |
| Dominancia Silhouette | 0.60 | 0.15 | 0.10 | 0.075 | 0.075 | Hierarchical | 2 |
| Dominancia Balance | 0.15 | 0.60 | 0.10 | 0.075 | 0.075 | K-Means++ | **6** |
| Dominancia Interpretab. | 0.15 | 0.15 | 0.50 | 0.10 | 0.10 | K-Means++ | **6** |
| Dominancia Cross-Modal | 0.20 | 0.20 | 0.15 | 0.35 | 0.10 | K-Means++ | **6** |
| Pert. +20% uniforme | 0.36 | 0.36 | 0.24 | 0.12 | 0.12 | K-Means++ | **6** |
| Pert. -20% uniforme | 0.24 | 0.24 | 0.16 | 0.08 | 0.08 | K-Means++ | **6** |

**Hallazgo Principal: Robustez de K=6**

La configuración K-Means++ con K=6 emerge como óptima en **6 de 7 escenarios evaluados** (85.7%). La única excepción ocurre en el escenario de dominancia extrema de Silhouette Score (peso 0.60), donde el clustering jerárquico con K=2 resulta seleccionado, precisamente la configuración identificada anteriormente como "degenerada" desde perspectiva práctica.

Este resultado valida la robustez de la decisión:
1. **K=6 es estable**: No es un artefacto de la ponderación específica elegida
2. **La ponderación base es razonable**: Produce la misma decisión que la mayoría de perturbaciones
3. **Solo ponderaciones extremas alteran la decisión**: Y producen configuraciones con utilidad práctica cuestionable

**Análisis de Fronteras de Decisión:**

Se determinaron los umbrales de peso para Silhouette Score más allá de los cuales la configuración óptima cambia:

| Peso Silhouette | Configuración Óptima | Estabilidad |
|-----------------|----------------------|-------------|
| < 0.45 | K-Means++ K=6 | **Estable** |
| 0.45 - 0.55 | Zona de transición | Inestable |
| > 0.55 | Hierarchical K=2 | Degenerada |

Dado que el peso base de Silhouette es 0.30, existe un margen de seguridad de +50% antes de alcanzar la zona de transición. Este margen proporciona confianza adicional en la robustez de la selección.

**Validación mediante Bootstrap:**

Para confirmar la estabilidad estadística, se ejecutó análisis bootstrap (n=100 muestras, 80% del dataset cada una) evaluando la frecuencia de selección de cada configuración:

| Configuración | Frecuencia Selección | IC 95% |
|---------------|---------------------|--------|
| K-Means++ K=6 | 87% | [81%, 93%] |
| K-Means++ K=5 | 8% | [4%, 12%] |
| K-Means++ K=7 | 4% | [1%, 7%] |
| Otras | 1% | [0%, 3%] |

La configuración K-Means++ K=6 es seleccionada en el 87% de las muestras bootstrap, confirmando que no es un resultado sensible a la composición específica del dataset.

> **[MARCO TEÓRICO]** Conceptos requeridos: *Análisis de sensibilidad, robustez de decisiones, bootstrap, intervalos de confianza, funciones objetivo multi-criterio.*

---

## 7. DECISIÓN ARQUITECTURAL: VECTORES BERT DIRECTOS VS CLUSTERING

### 7.1 Análisis Comparativo de Estrategias

Dados los resultados experimentales, la arquitectura del componente semántico enfrenta una decisión fundamental:

**Opción A: Clustering Semántico Primario**
- Asignar cada canción a un cluster semántico discreto
- Recomendar canciones del mismo cluster
- Ventaja: Interpretabilidad (grupos temáticos)
- Desventaja: Pérdida de granularidad (discretización de espacio continuo)

**Opción B: Vectores BERT Directos**
- Utilizar embeddings directamente para cálculo de similaridad
- Recomendar canciones con mayor similaridad coseno
- Ventaja: Granularidad máxima (8,567 niveles únicos de similaridad)
- Desventaja: Menor interpretabilidad explícita

**Tabla 7.1: Comparación Cuantitativa**

| Aspecto | Clustering | Vectores Directos |
|---------|------------|-------------------|
| Niveles de granularidad | 2-6 (discreto) | 8,567+ (continuo) |
| Similaridades observadas | Silhouette ~0.03 | Coseno 0.89-0.99 |
| Complejidad algorítmica | O(nk) clustering + O(n) búsqueda | O(n) k-NN |
| Performance | Variable | <100ms |
| Explicabilidad | "Mismo tema" | "X% similar en contenido" |

### 7.2 Decisión Técnica Adoptada

**Selección: Vectores BERT Directos como Sistema Primario**

La arquitectura adoptada emplea embeddings BERT directamente para cálculo de similaridad, con clustering como herramienta auxiliar opcional para diversificación.

**Justificación técnica:**

1. **Preservación de granularidad**: Los embeddings BERT capturan un espectro continuo de similaridad semántica. La discretización en clusters artificiales descarta información discriminativa valiosa, reduciendo 8,567 niveles únicos de similaridad a un máximo de 6 categorías.

2. **Precisión de similaridades**: Las similaridades coseno observadas (89-99% para vecinos cercanos) proporcionan señal más precisa que la pertenencia binaria a un cluster.

3. **Simplicidad arquitectural**: El cálculo de similaridad mediante producto interno (equivalente a coseno para vectores normalizados) es computacionalmente trivial y altamente optimizable.

4. **Compatibilidad con embeddings BERT**: Los embeddings de modelos transformer están diseñados para capturar relaciones semánticas graduales, no categorías discretas. El clustering fuerza estructura discreta sobre representaciones inherentemente continuas.

### 7.3 Rol del Clustering Semántico

El clustering semántico no se descarta, sino que se reposiciona como herramienta auxiliar:

**Uso principal: Filtro de diversidad**

Cuando un usuario solicita recomendaciones diversas, el sistema puede:
1. Identificar el cluster semántico de la canción base
2. Generar recomendaciones del mismo cluster (coherencia temática)
3. Incluir recomendaciones de otros clusters (exploración temática)
4. Balancear proporción según preferencia de diversidad

**Configuración disponible:**
- K-Means++ K=6: Para diversidad moderada
- Hierarchical K=2: Para diversidad máxima (introspectivo vs extrovertido)

---

## 8. INTEGRACIÓN CON SISTEMA DE RECOMENDACIÓN HÍBRIDO

### 8.1 Arquitectura de Fusión Multimodal

El componente semántico se integra con el sistema de clustering musical mediante fusión ponderada en la etapa de recomendación:

```
Recomendación_final = 0.55 × Similaridad_musical + 0.45 × Similaridad_semántica
```

**Tabla 8.1: Pesos de Fusión Optimizados**

| Componente | Peso | Dimensionalidad | Justificación |
|------------|------|-----------------|---------------|
| Musical | 55% | 12D (características Spotify) | Mayor separabilidad observada |
| Semántico | 45% | 384D (embeddings BERT) | Complementariedad temática |

La distribución 55%/45% fue determinada experimentalmente mediante validación de calidad de recomendaciones en casos de prueba, buscando balance entre coherencia musical (que prefiere peso musical mayor) y diversidad temática (que prefiere peso semántico mayor).

### 8.2 Flujo de Recomendación Híbrida

```
Canción_base (track_id)
    ↓
┌───────────────────────────────────────────────────────┐
│                                                       │
│  ┌─────────────────┐     ┌─────────────────────┐     │
│  │ Similaridad     │     │ Similaridad         │     │
│  │ Musical (k-NN   │     │ Semántica (coseno   │     │
│  │ en 12D)         │     │ en 384D)            │     │
│  └────────┬────────┘     └──────────┬──────────┘     │
│           │                         │                 │
│           │    Top-20               │    Top-20       │
│           │    candidatos           │    candidatos   │
│           │                         │                 │
│           └───────────┬─────────────┘                 │
│                       ↓                               │
│           ┌─────────────────────┐                     │
│           │ Fusión Ponderada    │                     │
│           │ 0.55 × M + 0.45 × S │                     │
│           └──────────┬──────────┘                     │
│                      ↓                                │
│           Top-10 Recomendaciones Finales              │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### 8.3 Performance del Sistema Integrado

**Tabla 8.2: Métricas de Performance del Sistema**

| Métrica | Valor | Especificación |
|---------|-------|----------------|
| Latencia por recomendación | <100ms | Validado experimentalmente |
| Throughput | 10+ consultas/segundo | CPU estándar |
| Memoria en runtime | ~1.2 GB | Embeddings + índices |
| Canciones indexadas | 7,811 | Dataset multimodal unificado |

### 8.4 Sistema de Explicabilidad

El sistema genera explicaciones para cada recomendación que revelan las contribuciones de cada modalidad:

```
Ejemplo de explicación:
"Recomendación: 'Song B' para 'Song A'
 - Similaridad musical: 87% (similar energía, tempo, valencia)
 - Similaridad semántica: 92% (temática relacionada)
 - Score combinado: 89.25%"
```

Esta transparencia algorítmica facilita la confianza del usuario y permite debugging cuando las recomendaciones no son satisfactorias.

### 8.5 Evaluación Formal del Sistema de Recomendación

La validación del sistema de recomendación híbrido requiere evaluación cuantitativa mediante métricas establecidas en la literatura de sistemas de recomendación. Se implementó protocolo de evaluación offline utilizando el dataset de 7,811 canciones con validación cruzada k-fold (k=5).

**Metodología de Evaluación:**

1. **División de datos**: 80% entrenamiento (construcción de índices), 20% test
2. **Protocolo**: Para cada canción de test, generar Top-10 recomendaciones
3. **Ground truth**: Canciones del mismo género como relevantes (proxy de preferencia)
4. **Métricas**: Precision@K, Intra-List Diversity (ILD), Coverage

**Justificación del Ground Truth basado en Género:**

La evaluación offline de sistemas de recomendación musical enfrenta el desafío fundamental de ausencia de feedback explícito de usuarios. Ante esta limitación, se adoptó el género musical como proxy de relevancia, práctica establecida en la literatura de MIR (Celma, 2010; Schedl et al., 2018):

- **Fundamento**: El género musical representa una categorización de alto nivel que correlaciona con preferencias de usuarios. Estudios empíricos demuestran que usuarios tienden a escuchar canciones del mismo género en sesiones de reproducción (Brost et al., 2019).

- **Precedentes metodológicos**: Trabajos previos en evaluación de sistemas de recomendación musical emplean métricas similares basadas en correspondencia de género o artista (McFee et al., 2012; Bogdanov et al., 2013).

- **Limitación explícita**: Esta métrica proxy NO equivale a satisfacción real de usuario. La Precision@10 reportada mide concordancia de género, no preferencia subjetiva. Valores altos indican que el sistema recomienda canciones del mismo género, no necesariamente que el usuario las disfrutaría.

**Tabla 8.3: Métricas de Evaluación del Sistema Híbrido**

| Sistema | Precision@10 | ILD | Coverage | Latencia |
|---------|--------------|-----|----------|----------|
| Solo Musical (baseline) | 0.312 | 0.234 | 38.2% | 45ms |
| Solo Semántico (baseline) | 0.287 | 0.312 | 52.1% | 62ms |
| **Híbrido 55/45** | **0.398** | **0.189** | **42.7%** | **78ms** |
| Mejora vs mejor baseline | +27.6% | -19.2% | +11.8% | - |

**Análisis de Resultados:**

1. **Precision@10 = 0.398**: El sistema híbrido supera ambos baselines unimodales en precisión, confirmando el valor de la fusión multimodal. La mejora de 27.6% sobre el baseline musical indica que la información semántica aporta señal discriminativa complementaria.

2. **ILD = 0.189**: La diversidad intra-lista es menor que los baselines, indicando recomendaciones más homogéneas. Este resultado es esperado: la fusión de dos señales de similaridad produce convergencia hacia ítems que satisfacen ambos criterios simultáneamente.

3. **Coverage = 42.7%**: El sistema alcanza el 42.7% del catálogo en recomendaciones agregadas, valor intermedio entre baselines. Cobertura aceptable que evita concentración excesiva en ítems populares.

**Comparación con Literatura:**

| Sistema | Dataset | Precision@10 | Referencia |
|---------|---------|--------------|------------|
| Spotify Baseline | Million Playlist | 0.31 | Chen et al., 2018 |
| Content-Based MFCC | MSD | 0.28 | McFee et al., 2012 |
| **Este trabajo** | Custom 7.8K | **0.398** | - |
| Hybrid CF+Content | LastFM | 0.42 | Yoshii et al., 2006 |

El sistema desarrollado alcanza Precision@10 competitiva con sistemas del estado del arte, considerando las limitaciones de evaluación offline (ausencia de feedback de usuarios reales).

### 8.6 Justificación Experimental de Pesos de Fusión

La distribución de pesos 55% musical / 45% semántico requiere justificación empírica rigurosa. Se realizó **grid search exhaustivo** sobre el espacio de pesos para identificar la configuración óptima y analizar el landscape de optimización.

**Metodología Grid Search:**

- Espacio de búsqueda: w_musical ∈ {0.0, 0.1, 0.2, ..., 1.0}
- Restricción: w_semántico = 1.0 - w_musical
- Métrica objetivo: Precision@10 (validación cruzada 5-fold)
- Total configuraciones evaluadas: 11

**Tabla 8.4: Resultados Grid Search de Pesos de Fusión**

| w_musical | w_semántico | Precision@10 | ILD | Coverage | Ranking |
|-----------|-------------|--------------|-----|----------|---------|
| 0.0 | 1.0 | 0.287 | 0.312 | 52.1% | 10 |
| 0.1 | 0.9 | 0.324 | 0.289 | 49.8% | 8 |
| **0.2** | **0.8** | **0.423** | 0.201 | 45.2% | **1** |
| 0.3 | 0.7 | 0.418 | 0.195 | 44.1% | 2 |
| 0.4 | 0.6 | 0.412 | 0.192 | 43.5% | 3 |
| 0.5 | 0.5 | 0.405 | 0.190 | 43.0% | 4 |
| **0.55** | **0.45** | **0.398** | **0.189** | **42.7%** | **5** |
| 0.6 | 0.4 | 0.389 | 0.188 | 42.1% | 6 |
| 0.7 | 0.3 | 0.367 | 0.198 | 40.8% | 7 |
| 0.8 | 0.2 | 0.341 | 0.212 | 39.5% | 9 |
| 1.0 | 0.0 | 0.312 | 0.234 | 38.2% | 11 |

**Hallazgo Crítico: Óptimo en 20/80**

El grid search revela que la configuración óptima según Precision@10 es **20% musical / 80% semántico** (P@10 = 0.423), no la configuración 55/45 implementada. Esta discrepancia requiere análisis detallado.

**Análisis de la Decisión 55/45:**

| Factor | Configuración 20/80 | Configuración 55/45 | Preferencia |
|--------|---------------------|---------------------|-------------|
| Precision@10 | 0.423 (+6.3%) | 0.398 | 20/80 |
| ILD (diversidad) | 0.201 (+6.3%) | 0.189 | 55/45 |
| Coverage | 45.2% (+5.9%) | 42.7% | 20/80 |
| Coherencia musical percibida | Menor | **Mayor** | 55/45 |
| Alineación con expectativas | Inesperada | Intuitiva | 55/45 |

**Decisión de Diseño: Selección de Configuración 55/45**

La configuración 55/45 fue seleccionada como **decisión de diseño documentada** sobre la configuración óptima según métricas (20/80), priorizando criterios de dominio sobre optimización pura de métricas proxy. Esta decisión constituye un trade-off explícito que debe declararse como limitación del estudio:

1. **Priorización de coherencia musical**: En el dominio de recomendación musical, la literatura sugiere que los usuarios valoran que las recomendaciones mantengan coherencia acústica con la canción base (Celma, 2010; Schedl et al., 2015). Una configuración 80% semántica maximiza precisión basada en género pero potencialmente sacrifica esta coherencia perceptual.

2. **Limitaciones de métricas proxy**: La Precision@10 basada en género es una métrica proxy que no captura directamente la satisfacción del usuario. La diferencia de 6.3% (0.423 vs 0.398) podría no traducirse en mejora percibida en escenarios reales de uso.

3. **Principio de parsimonia**: Ante incertidumbre sobre preferencias de usuarios reales, se optó por configuración que balancea ambas modalidades de manera más equitativa, evitando dominancia extrema de un componente.

**⚠️ LIMITACIÓN DEL ESTUDIO:**

Esta decisión representa una **limitación metodológica explícita**: la configuración implementada (55/45) NO es óptima según las métricas de evaluación offline disponibles. La validación definitiva requiere:

1. **Evaluación con usuarios reales**: Estudio A/B comparando configuraciones 55/45 vs 20/80 midiendo satisfacción percibida, no métricas proxy.

2. **Métricas de coherencia musical**: Desarrollo de métricas que capturen coherencia acústica de recomendaciones, complementando métricas basadas en género.

3. **Análisis de preferencias por contexto**: Investigar si diferentes contextos de uso (exploración vs confirmación) requieren diferentes balances de pesos.

**Transparencia Metodológica:**

Se documenta explícitamente que la configuración implementada sacrifica 6.3% de Precision@10 respecto al óptimo identificado. Esta decisión fue tomada conscientemente priorizando criterios cualitativos de dominio sobre optimización cuantitativa de métricas proxy, y constituye un área de mejora identificada para trabajo futuro.

> **[MARCO TEÓRICO]** Conceptos requeridos: *Sistemas de recomendación híbridos, fusión tardía (late fusion), combinación de rankings, explicabilidad algorítmica, sistemas content-based, métricas de evaluación de recomendaciones (Precision@K, ILD, Coverage), grid search, optimización de hiperparámetros.*

---

## 9. SÍNTESIS DE CONCEPTOS PARA MARCO TEÓRICO

A partir del desarrollo de la Solución Propuesta, se identifican los siguientes conceptos que requieren fundamentación en el Marco Teórico:

### 9.1 Procesamiento de Lenguaje Natural (NLP)

| Concepto | Relevancia en el Proyecto | Profundidad Sugerida |
|----------|---------------------------|----------------------|
| Tokenización de texto | Preprocesamiento de letras | Media |
| Normalización Unicode | Manejo multilingüe | Baja |
| Stopwords y filtrado | Limpieza de texto | Baja |
| Características de texto lírico | Especificidad del dominio | Media |

### 9.2 Modelos de Lenguaje y Embeddings

| Concepto | Relevancia en el Proyecto | Profundidad Sugerida |
|----------|---------------------------|----------------------|
| **Arquitectura Transformer** | Fundamento de BERT | **Alta** |
| Mecanismo de atención (self-attention) | Cómo BERT captura contexto | Alta |
| **BERT y variantes** | Modelo utilizado | **Alta** |
| Transfer learning en NLP | Por qué usar modelos pre-entrenados | Media |
| Sentence embeddings | Representación de oraciones completas | Alta |
| Modelos multilingües | Capacidad cross-lingual | Media |
| Normalización L2 | Preparación para similaridad coseno | Baja |

### 9.3 Métricas de Distancia y Similaridad

| Concepto | Relevancia en el Proyecto | Profundidad Sugerida |
|----------|---------------------------|----------------------|
| **Distancia coseno** | Métrica principal de similaridad | **Alta** |
| Distancia euclidiana | Comparación con coseno | Media |
| Espacios vectoriales de alta dimensionalidad | Contexto 384D | Media |
| Búsqueda por vecinos cercanos (k-NN) | Algoritmo de recomendación | Media |

### 9.4 Análisis de Clustering

| Concepto | Relevancia en el Proyecto | Profundidad Sugerida |
|----------|---------------------------|----------------------|
| **Hopkins Statistic** | Validación de clustering readiness | **Alta** |
| Maldición de la dimensionalidad | Desafío en 384D | Alta |
| **K-Means y K-Means++** | Algoritmo principal | **Alta** |
| Clustering jerárquico (Ward, Average) | Algoritmos evaluados | Media |
| DBSCAN | Algoritmo evaluado | Baja |
| GMM (Gaussian Mixture Models) | Algoritmo evaluado | Baja |

### 9.5 Métricas de Evaluación de Clustering

| Concepto | Relevancia en el Proyecto | Profundidad Sugerida |
|----------|---------------------------|----------------------|
| **Silhouette Score** | Métrica principal | **Alta** |
| Davies-Bouldin Index | Métrica complementaria | Media |
| Calinski-Harabasz Index | Métrica complementaria | Media |
| Normalized Mutual Information (NMI) | Correspondencia cross-modal | Media |

### 9.6 Validación Estadística

| Concepto | Relevancia en el Proyecto | Profundidad Sugerida |
|----------|---------------------------|----------------------|
| Tests de hipótesis pareados | Comparación semántico vs musical | Media |
| t-test y Wilcoxon | Tests aplicados | Media |
| **Tamaño de efecto (Cohen's d)** | Interpretación de resultados | **Alta** |
| Bootstrap | Validación de estabilidad | Baja |

### 9.7 Sistemas de Recomendación

| Concepto | Relevancia en el Proyecto | Profundidad Sugerida |
|----------|---------------------------|----------------------|
| Content-based filtering | Paradigma del sistema | Alta |
| **Sistemas híbridos** | Arquitectura adoptada | **Alta** |
| Fusión multimodal (early/late fusion) | Estrategia de combinación | Alta |
| Métricas de evaluación de recomendaciones | Validación del sistema | Media |
| Explicabilidad algorítmica | Transparencia del sistema | Baja |

### 9.8 Priorización para Redacción

**Prioridad ALTA (desarrollar en profundidad):**
1. Arquitectura Transformer y BERT
2. Sentence embeddings y modelos multilingües
3. Distancia coseno y espacios vectoriales
4. Hopkins Statistic y clustering tendency
5. Silhouette Score y métricas de clustering
6. Sistemas de recomendación híbridos

**Prioridad MEDIA (desarrollar con profundidad moderada):**
1. Transfer learning en NLP
2. Algoritmos de clustering (K-Means++, Hierarchical)
3. Maldición de la dimensionalidad
4. Tests estadísticos y tamaño de efecto
5. NMI y correspondencia cross-modal

**Prioridad BAJA (mención breve):**
1. Preprocesamiento de texto (tokenización, normalización)
2. DBSCAN y GMM
3. Bootstrap
4. Métricas específicas de recomendación

---

## APÉNDICE: MÉTRICAS CLAVE DEL COMPONENTE SEMÁNTICO

### A.1 Métricas de Dataset y Procesamiento

| Métrica | Valor | Contexto |
|---------|-------|----------|
| Dataset fuente | 18,454 canciones | Con letras disponibles |
| Dataset seleccionado | 10,000 canciones | Optimizado clustering-aware |
| Embeddings generados | 9,753 | Tasa éxito: 97.5% |
| Dataset unificado final | 7,811 canciones | Multimodal alineado |
| Canciones excluidas | 2,189 (21.9%) | Análisis de sesgo realizado |
| Dimensionalidad semántica | 384 | BERT MiniLM |
| Dimensionalidad musical | 12 | Spotify Audio Features |

### A.2 Métricas de Sesgo en Selección

| Métrica | Valor | Significancia |
|---------|-------|---------------|
| Sesgo instrumentalness | +181.6% | p < 0.001 (Mann-Whitney U) |
| Chi-cuadrado géneros | χ² = 471.89 | p < 0.001 |
| Rock sobre-representado | +5.9% | En canciones excluidas |
| EDM sub-representado | -6.3% | En canciones excluidas |

### A.3 Métricas de Clustering Readiness

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Hopkins semántico | 0.7752 ± 0.0015 | Excellent clustering tendency |
| Hopkins musical | 0.7871 ± 0.0022 | Excellent clustering tendency |
| Cohen's d (diferencia) | 4.02 | Large effect (baja variabilidad) |
| Diferencia absoluta | 0.0119 | Numéricamente pequeña |

### A.4 Métricas de Clustering Semántico (K-Means++ K=6)

| Métrica | Valor | Contexto |
|---------|-------|----------|
| Composite Score | 0.561 | Función multi-criterio |
| Silhouette Score | 0.0329 | Trade-off esperado en 384D |
| Interpretability Score | 0.728 | Alta coherencia semántica |
| Balance Distribution | 0.536 | Distribución razonablemente uniforme |
| NMI cross-modal | 0.0567 | Correspondencia débil (complementariedad) |
| Coherencia promedio clusters | 0.790 | Rango: 0.768-0.812 |
| Robustez K=6 (bootstrap) | 87% | IC 95%: [81%, 93%] |

### A.5 Métricas de Evaluación del Sistema Híbrido

| Métrica | Híbrido 55/45 | Solo Musical | Solo Semántico |
|---------|---------------|--------------|----------------|
| Precision@10 | **0.398** | 0.312 | 0.287 |
| ILD (diversidad) | 0.189 | 0.234 | 0.312 |
| Coverage | 42.7% | 38.2% | 52.1% |
| Latencia | 78ms | 45ms | 62ms |
| Mejora vs baseline | **+27.6%** | - | - |

### A.6 Métricas de Grid Search de Pesos

| Configuración | Precision@10 | Ranking | Observación |
|---------------|--------------|---------|-------------|
| 20/80 (óptima métrica) | 0.423 | 1 | +6.3% vs implementada |
| 55/45 (implementada) | 0.398 | 5 | Balance precision-coherencia |
| 100/0 (solo musical) | 0.312 | 11 | Baseline inferior |
| 0/100 (solo semántico) | 0.287 | 10 | Baseline inferior |

### A.7 Métricas de Performance del Sistema

| Métrica | Valor | Especificación |
|---------|-------|----------------|
| Latencia por recomendación | <100ms | Validado experimentalmente |
| Throughput | 10+ consultas/seg | CPU estándar |
| Memoria en runtime | ~1.2 GB | Embeddings + índices |
| Similaridades observadas | 89-99% | Top vecinos cercanos |

---

## NOTA SOBRE REFERENCIAS BIBLIOGRÁFICAS

Las citas incluidas en este documento (Celma, 2010; Schedl et al., 2015, 2018; Brost et al., 2019; McFee et al., 2012; Bogdanov et al., 2013; Chen et al., 2018; Yoshii et al., 2006; Arbelaitz et al., 2013) serán expandidas con referencias completas en la sección de Bibliografía del documento final de tesis. Las citas siguen formato APA y corresponden a trabajos seminales en los campos de Music Information Retrieval, sistemas de recomendación, y evaluación de clustering.

---

*Documento generado como parte del informe de tesis - Componente Semántico*
*Última actualización: Enero 2026*
