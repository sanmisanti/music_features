# 📚 DOCUMENTACIÓN TÉCNICA ACADÉMICA
## Módulo de Análisis de Características Musicales
### Sistema de Recomendación Musical Multimodal

---

**Tesis de Ingeniería Informática**  
**Módulo**: Análisis Exploratorio de Características Musicales  
**Contexto**: Sistema de Recomendación Musical Multimodal  
**Última actualización**: 26 de enero de 2025

---

## RESUMEN EJECUTIVO

Este documento presenta la implementación y validación del módulo de análisis exploratorio de características musicales, componente fundamental de un sistema de recomendación musical multimodal. El módulo desarrollado procesa y analiza 13 características acústicas extraídas de la API de Spotify, aplicando técnicas avanzadas de análisis estadístico, visualización de datos y reducción dimensional para identificar patrones intrínsecos en datos musicales.

**Palabras clave**: Análisis exploratorio de datos, características musicales, PCA, t-SNE, sistemas de recomendación, Music Information Retrieval (MIR)

---

## 1. INTRODUCCIÓN

### 1.1 Contexto del Proyecto

Los sistemas de recomendación musical han evolucionado significativamente desde los enfoques basados en filtrado colaborativo hacia sistemas multimodales que integran múltiples fuentes de información musical [McFee et al., 2012]. El presente trabajo forma parte de un sistema de recomendación que combina análisis de características acústicas con procesamiento semántico de letras, siguiendo la tendencia hacia sistemas híbridos más robustos [Schedl et al., 2018].

### 1.2 Definición del Problema

El análisis de características musicales presenta desafíos únicos en el procesamiento de datos multidimensionales donde cada dimensión representa aspectos perceptuales, rítmicos, armónicos o estructurales de la música. La necesidad de comprender la estructura intrínseca de estos datos es fundamental para el diseño de algoritmos de clustering y recomendación efectivos.

### 1.3 Objetivos

**Objetivo General**: Desarrollar y validar un sistema modular de análisis exploratorio para características musicales que soporte la toma de decisiones en el diseño de algoritmos de clustering.

**Objetivos Específicos**:
1. Implementar un pipeline de procesamiento robusto para datasets musicales masivos
2. Aplicar técnicas de análisis estadístico descriptivo para caracterizar el espacio de features
3. Desarrollar herramientas de visualización especializadas para datos musicales multidimensionales
4. Evaluar métodos de reducción dimensional para optimización de clustering

---

## 2. MARCO TEÓRICO

### 2.1 Music Information Retrieval (MIR)

Music Information Retrieval constituye un campo interdisciplinario que combina técnicas de procesamiento de señales, aprendizaje automático y musicología computacional [Müller, 2015]. Las características acústicas utilizadas en este trabajo se fundamentan en modelos perceptuales del Audio Features Framework de Spotify, que traduce propiedades físicas del audio a medidas perceptualmente relevantes.

### 2.2 Características Acústicas de Spotify

El conjunto de características utilizado comprende 13 dimensiones categorizadas según su naturaleza musical:

#### 2.2.1 Features de Audio (7 dimensiones)
- **Danceability** [0,1]: Métrica de aptitud para el baile basada en tempo, estabilidad rítmica, fuerza del beat y regularidad general
- **Energy** [0,1]: Medida perceptual de intensidad y actividad, correlacionada con atributos como rango dinámico, loudness percibido, timbre y entropía del onset
- **Speechiness** [0,1]: Detección de presencia de palabras habladas en una pista
- **Acousticness** [0,1]: Medida de confianza de si la pista es acústica
- **Instrumentalness** [0,1]: Predicción de si una pista no contiene vocales
- **Liveness** [0,1]: Detección de presencia de audiencia en la grabación
- **Valence** [0,1]: Medida de positividad musical transmitida por una pista

#### 2.2.2 Features Rítmicas (2 dimensiones)
- **Tempo** [BPM]: Tempo general estimado de una pista en beats per minute
- **Time Signature** [3-7]: Signatura temporal estimada (compás)

#### 2.2.3 Features Armónicas (3 dimensiones)
- **Key** [0-11]: Tonalidad estimada de la pista (notación Pitch Class)
- **Loudness** [dB]: Loudness general de una pista en decibeles
- **Mode** [0,1]: Modalidad (mayor=1, menor=0)

#### 2.2.4 Features Estructurales (1 dimensión)
- **Duration_ms** [ms]: Duración de la pista en milisegundos

### 2.3 Análisis Estadístico Multivariado

#### 2.3.1 Estadística Descriptiva

La caracterización estadística de datasets musicales requiere consideraciones especiales debido a la naturaleza no-gaussiana de muchas distribuciones musicales [Sturm, 2013]. Se implementaron las siguientes métricas:

**Medidas de Tendencia Central**:
- Media aritmética: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
- Mediana: valor que divide la distribución en dos partes iguales

**Medidas de Dispersión**:
- Desviación estándar: $\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$
- Rango intercuartílico (IQR): $Q_3 - Q_1$

**Medidas de Forma**:
- Skewness (asimetría): $S = \frac{E[(X-\mu)^3]}{\sigma^3}$
- Kurtosis (apuntamiento): $K = \frac{E[(X-\mu)^4]}{\sigma^4}$

#### 2.3.2 Análisis de Correlación

Se implementaron tres métodos de correlación para capturar diferentes tipos de dependencias:

**Correlación de Pearson**:
$$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Mide relaciones lineales entre variables. Asume normalidad y homocedasticidad.

**Correlación de Spearman**:
$$\rho = 1 - \frac{6\sum_{i=1}^{n}d_i^2}{n(n^2-1)}$$

Donde $d_i$ es la diferencia entre rangos. Captura relaciones monótonas no necesariamente lineales.

**Correlación de Kendall**:
$$\tau = \frac{n_c - n_d}{\frac{1}{2}n(n-1)}$$

Donde $n_c$ y $n_d$ son pares concordantes y discordantes respectivamente. Robusto ante outliers.

### 2.4 Reducción Dimensional

#### 2.4.1 Principal Component Analysis (PCA)

PCA realiza una transformación lineal que proyecta los datos a un subespacio de menor dimensión preservando la máxima varianza [Jolliffe, 2002].

**Formulación Matemática**:
Sea $X \in \mathbb{R}^{n \times p}$ la matriz de datos centrada. La descomposición en valores singulares:

$$X = U\Sigma V^T$$

Los componentes principales son las columnas de $V$, ordenadas por los valores singulares en $\Sigma$.

**Criterio de Selección de Componentes**:
Se retienen $k$ componentes tal que:
$$\frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{p}\lambda_i} \geq 0.90$$

Donde $\lambda_i$ son los eigenvalores ordenados decrecientemente.

#### 2.4.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE es una técnica de reducción dimensional no-lineal especialmente efectiva para visualización [van der Maaten & Hinton, 2008].

**Formulación**:
Define probabilidades condicionales en el espacio original:
$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$

Y probabilidades simétricas en el espacio reducido:
$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l}(1 + ||y_k - y_l||^2)^{-1}}$$

Minimiza la divergencia KL:
$$C = KL(P||Q) = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**Parámetro Perplexity**: Se ajusta automáticamente según el tamaño del dataset:
$$\text{perplexity} = \min(30, \frac{n-1}{3})$$

### 2.5 Selección de Features

#### 2.5.1 Variance Threshold

Elimina features con varianza inferior a un umbral, identificando características poco informativas:
$$\text{Var}(X) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2 < \tau$$

Se utiliza $\tau = 0.01$ para eliminar features quasi-constantes.

#### 2.5.2 Mutual Information

Para selección supervisada, utiliza información mutua entre features y target:
$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

---

## 3. METODOLOGÍA

### 3.1 Arquitectura del Sistema

El sistema implementa una arquitectura modular basada en el patrón de separación de responsabilidades, con los siguientes componentes:

```
exploratory_analysis/
├── config/           # Configuraciones centralizadas
├── data_loading/     # Gestión de datos y validación
├── statistical_analysis/  # Análisis estadístico
├── visualization/    # Herramientas de visualización
├── feature_analysis/ # Reducción dimensional y selección
├── reporting/        # Generación de reportes
└── utils/           # Utilidades compartidas
```

### 3.2 Pipeline de Procesamiento

#### 3.2.1 Carga y Validación de Datos

**Algoritmo de Carga Inteligente**:
```python
def intelligent_loading(file_path, sample_size=None):
    file_size = get_file_size(file_path)
    if file_size > MEMORY_THRESHOLD:
        return chunked_loading(file_path, sample_size)
    else:
        return direct_loading(file_path)
```

**Validación Multi-nivel**:
- **BASIC**: Verificación de tipos y rangos
- **STANDARD**: Detección de outliers y missing values
- **STRICT**: Validación de distribuciones y consistencia

#### 3.2.2 Análisis Estadístico

Se implementó un sistema de análisis estadístico robusto que calcula:

1. **Estadísticas Descriptivas**: Para cada feature individual
2. **Matrices de Correlación**: Usando tres métodos (Pearson, Spearman, Kendall)
3. **Detección de Outliers**: Método IQR con factor k=1.5
4. **Clasificación de Distribuciones**: Basada en skewness y kurtosis

#### 3.2.3 Sistema de Visualización

**Componentes Implementados**:
- `DistributionPlotter`: Histogramas, box plots, violin plots
- `CorrelationPlotter`: Heatmaps, clustered heatmaps, network plots

**Consideraciones de Diseño**:
- Uso de paletas de colores perceptualmente uniformes
- Adaptación automática de layouts según número de features
- Exportación en alta resolución (300 DPI)

### 3.3 Metodología de Testing

Se implementó una estrategia de testing exhaustiva con tres niveles:

1. **Tests Unitarios**: Para cada módulo individual
2. **Tests de Integración**: Verificando interoperabilidad
3. **Tests de Performance**: Medición de tiempos y memoria

**Métricas de Calidad**:
- Completitud de datos (% missing values)
- Unicidad (% duplicados)
- Consistencia (encoding y formatos)
- Validez (rangos esperados)

---

## 4. IMPLEMENTACIÓN

### 4.1 Stack Tecnológico

**Lenguaje**: Python 3.8+  
**Librerías Core**:
- `pandas` (1.5+): Manipulación de datos
- `numpy` (1.21+): Operaciones numéricas
- `scikit-learn` (1.1+): Machine learning y estadística
- `matplotlib` (3.5+): Visualización base
- `seaborn` (0.11+): Visualización estadística avanzada

**Librerías Opcionales**:
- `umap-learn`: Reducción dimensional no-lineal
- `networkx`: Análisis de redes de correlación

### 4.2 Configuración Centralizada

Se implementó un sistema de configuración basado en dataclasses para garantizar consistencia:

```python
@dataclass
class DataConfig:
    separator: str = ';'
    decimal: str = ','
    encoding: str = 'utf-8'
    default_sample_size: int = 10000
    random_state: int = 42
```

### 4.3 Gestión de Memoria

Para datasets grandes (>1.2M registros), se implementaron estrategias de optimización:

- **Chunked Loading**: Carga por fragmentos
- **Lazy Evaluation**: Cálculos bajo demanda
- **Memory Mapping**: Para archivos muy grandes
- **Automatic Sampling**: Reducción inteligente de muestras

### 4.4 Manejo de Errores y Logging

Sistema robusto de logging con niveles diferenciados:
- **INFO**: Operaciones normales y métricas
- **WARNING**: Situaciones recuperables
- **ERROR**: Errores críticos que detienen procesamiento

---

## 5. RESULTADOS Y ANÁLISIS

### 5.1 Validación del Sistema

#### 5.1.1 Calidad de Datos

**Dataset de Validación**: 500 muestras de 1.2M canciones totales

**Métricas Obtenidas**:
- **Completitud**: 100% (0 valores faltantes)
- **Unicidad**: 100% (0 duplicados)
- **Consistencia**: Excelente (encoding automático correcto)
- **Puntuación General**: 99.5-100/100 (EXCELLENT)

#### 5.1.2 Distribuciones de Features

**Análisis de Normalidad**:
```
Features aproximadamente normales (|skewness| < 0.5):
- energy: skewness = -0.297
- valence: skewness = 0.062
- danceability: skewness = -0.184

Features moderadamente sesgadas (0.5 ≤ |skewness| < 1.0):
- loudness: skewness = -0.569
- acousticness: skewness = 0.663

Features altamente sesgadas (|skewness| ≥ 1.0):
- speechiness: skewness = 6.194
- instrumentalness: skewness = 2.505
```

**Implicaciones**: La presencia de distribuciones sesgadas es esperada en datos musicales y justifica el uso de métodos robustos como correlación de Spearman.

### 5.2 Análisis de Correlaciones

#### 5.2.1 Correlaciones Significativas Identificadas

**Correlaciones Altas (|r| ≥ 0.7)**:
1. `energy ↔ loudness`: r = 0.753
   - **Interpretación**: Relación física esperada entre energía percibida y amplitud
   - **Implicación**: Posible redundancia para clustering

2. `energy ↔ acousticness`: r = -0.711
   - **Interpretación**: Dicotomía entre música electrónica y acústica
   - **Implicación**: Dimensión fundamental de caracterización musical

**Correlaciones Moderadas (0.3 ≤ |r| < 0.7)**:
- `danceability ↔ valence`: r = 0.456
- `energy ↔ valence`: r = 0.438

#### 5.2.2 Análisis de Red de Correlaciones

Con umbral |r| ≥ 0.3, se identificaron 6 conexiones significativas de 15 posibles (40%), indicando:
- Estructura moderadamente correlacionada
- Información complementaria en la mayoría de features
- Red de dependencias interpretable musicalmente

### 5.3 Reducción Dimensional

#### 5.3.1 Análisis PCA

**Resultados Obtenidos**:
- **Componentes retenidos**: 10 (para 90% de varianza)
- **Varianza explicada total**: 93.6%
- **Distribución de varianza por componente**:
  - PC1: 21.7% (energy-driven)
  - PC2: 15.7% (danceability-driven)
  - PC3: 10.4% (mode-driven)

**Interpretación de Componentes Principales**:

**PC1 (21.7% varianza)**: "Dimensión de Intensidad Musical"
- Features dominantes: energy, loudness, acousticness (negativo)
- Interpretación: Contraste entre música intensa/electrónica vs. suave/acústica

**PC2 (15.7% varianza)**: "Dimensión de Bailabilidad"
- Features dominantes: danceability, valence, tempo
- Interpretación: Características que favorecen el movimiento y positividad

**PC3 (10.4% varianza)**: "Dimensión Armónica"
- Features dominantes: mode, key-related features
- Interpretación: Características tonales y armónicas

#### 5.3.2 Análisis t-SNE

**Parámetros de Convergencia**:
- **KL Divergence**: 0.3701 (< 1.0, calidad aceptable)
- **Iteraciones**: 849 (convergencia exitosa)
- **Perplexity**: Ajustado automáticamente según tamaño del dataset

**Validación de Calidad**:
La divergencia KL obtenida (0.3701) indica una proyección de buena calidad, donde la estructura local del espacio de alta dimensión se preserva adecuadamente en 2D.

### 5.4 Selección de Features

#### 5.4.1 Resultados de Variance Threshold

- **Features retenidas**: 12 de 13 (92.3%)
- **Feature eliminada**: 1 por baja varianza
- **Interpretación**: Alta diversidad en el dataset, pocas features redundantes

#### 5.4.2 Análisis de Importancia

**Ranking de Features por Importancia PCA**:
1. valence: 0.096
2. tempo: 0.088
3. loudness: 0.087
4. energy: 0.083
5. duration_ms: 0.082

Este ranking sugiere que características perceptuales (valence) y rítmicas (tempo) son más discriminativas que características puramente técnicas.

---

## 6. DISCUSIÓN

### 6.1 Validación de Hipótesis

#### 6.1.1 Calidad de Datos
**Hipótesis**: El dataset de Spotify presenta alta calidad para análisis ML.
**Resultado**: CONFIRMADA. Calidad 99.5-100/100 sin requerimientos de limpieza extensiva.

#### 6.1.2 Estructura Correlacional
**Hipótesis**: Las correlaciones reflejan relaciones musicológicas conocidas.
**Resultado**: CONFIRMADA. Correlaciones como energy-loudness y energy-acousticness son musicológicamente coherentes.

#### 6.1.3 Reducción Dimensional
**Hipótesis**: PCA puede reducir efectivamente la dimensionalidad manteniendo información relevante.
**Resultado**: CONFIRMADA. 93.6% de varianza en 10 componentes con interpretabilidad musical clara.

### 6.2 Limitaciones Identificadas

#### 6.2.1 Tamaño de Muestra para t-SNE
El uso de muestras pequeñas (80-200 canciones) para t-SNE puede no capturar completamente la estructura global del espacio musical. Se recomienda validación con muestras ≥1000 canciones.

#### 6.2.2 Dependencia de UMAP
La ausencia de UMAP limita las opciones de reducción dimensional no-lineal. Su instalación mejoraría las capacidades de análisis exploratorio.

#### 6.2.3 Selección Supervisada
La limitación de mutual information para targets continuos restringe las opciones de selección supervisada de features.

### 6.3 Implicaciones para Clustering

#### 6.3.1 Preparación de Datos
Los resultados indican que el dataset está óptimamente preparado para algoritmos de clustering:
- Alta calidad de datos (sin missing values)
- Baja redundancia entre features
- Estructura dimensional clara y interpretable

#### 6.3.2 Selección de Features
Se recomienda el uso de 10-12 features, potencialmente aplicando PCA para datasets muy grandes donde la eficiencia computacional sea crítica.

#### 6.3.3 Normalización
La presencia de diferentes escalas (BPM vs. features [0,1]) justifica el uso de StandardScaler antes del clustering.

---

## 7. CONCLUSIONES

### 7.1 Contribuciones Principales

1. **Sistema Modular Robusto**: Implementación de un framework extensible para análisis exploratorio de datos musicales con arquitectura escalable.

2. **Validación Empírica**: Demostración de la alta calidad del dataset de Spotify y identificación de estructuras correlacionales musicológicamente coherentes.

3. **Reducción Dimensional Interpretable**: Caracterización exitosa del espacio de features mediante PCA con componentes musicalmente interpretables.

4. **Framework de Testing**: Desarrollo de metodología de validación exhaustiva con métricas específicas para datos musicales.

### 7.2 Impacto en el Sistema de Recomendación

Los resultados del análisis exploratorio proporcionan fundamentos sólidos para las siguientes fases:

- **Clustering Optimizado**: Features seleccionadas y validadas para segmentación efectiva
- **Interpretabilidad Musical**: Componentes principales con significado musical claro
- **Escalabilidad**: Pipeline optimizado para datasets de millones de registros

### 7.3 Trabajo Futuro

#### 7.3.1 Extensiones Inmediatas
- Implementación de UMAP para análisis no-lineal completo
- Validación con dataset completo (1.2M canciones)
- Análisis cross-cultural con datasets internacionales

#### 7.3.2 Integraciones Futuras
- Fusión con análisis semántico de letras
- Incorporación de features de audio de bajo nivel
- Desarrollo de métricas de similaridad musical específicas

---

## REFERENCIAS

[1] McFee, B., Bertin-Mahieux, T., Ellis, D. P., & Lanckriet, G. R. (2012). The million song dataset challenge. *Proceedings of the 21st international conference companion on World Wide Web*, 909-916.

[2] Schedl, M., Zamani, H., Chen, C. W., Deldjoo, Y., & Elahi, M. (2018). Current challenges and visions in music recommender systems research. *International journal of multimedia information retrieval*, 7(2), 95-116.

[3] Müller, M. (2015). *Fundamentals of music processing: Audio, analysis, algorithms, applications*. Springer.

[4] Sturm, B. L. (2013). Classification accuracy is not enough: On the evaluation of music genre recognition systems. *Journal of Intelligent Information Systems*, 41(3), 371-406.

[5] Jolliffe, I. T. (2002). *Principal component analysis*. Springer.

[6] van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of machine learning research*, 9(11), 2579-2605.

[7] Spotify. (2023). *Web API Audio Features Documentation*. https://developer.spotify.com/documentation/web-api/reference/get-audio-features

[8] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of machine learning research*, 12, 2825-2830.

---

**Anexos disponibles en repositorio**:
- Anexo A: Código fuente completo
- Anexo B: Resultados de tests detallados
- Anexo C: Visualizaciones generadas
- Anexo D: Configuraciones utilizadas

---

*Documento generado automáticamente por el sistema de documentación académica*  
*Versión: 1.0 | Fecha: 26 de enero de 2025*