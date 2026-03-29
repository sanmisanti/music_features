# CLAUDE_ETAPA4.md — Clustering Multimodal

Documento de planificacion, decisiones y contexto para la Etapa 4 del proyecto.
Referenciado desde el `CLAUDE.md` principal del repositorio.

---

## 1. OBJETIVO DEL CLUSTERING: POR QUE LO HACEMOS

### 1.1 La pregunta que la v1 no respondio

En la primera ejecucion, el clustering tuvo tres roles que se mezclaron sin justificacion:

1. **Seleccion de datos** (clustering-aware sampling 18K -> 10K): introdujo 57.7% de perdida sin justificacion. Ya no aplica en v2 (conservamos 17,964 canciones con filtros de calidad).
2. **Descubrimiento de estructura**: se ejecuto pero con resultados problematicos (Silhouette jerarquico semantico 0.6733 con distribucion 99.98%/0.02% — artefacto inutilizable).
3. **Componente del sistema de recomendacion**: despues de todo el trabajo, la conclusion fue "abandonar clustering semantico, usar vectores BERT directos con k-NN". El clustering se declaro prescindible.

La v1 nunca respondio explicitamente: **para que clusterizamos**.

### 1.2 Roles justificados en v2

El clustering en la re-ejecucion tiene **dos roles separados y explicitamente diferenciados**:

#### Rol 1: Validacion de la calidad de las representaciones (analitico)

Si los embeddings semanticos y las features musicales capturan informacion real, deberian exhibir estructura interna — no ser ruido uniforme. El clustering valida esto:

- **Hopkins** responde: "hay estructura en estos datos, o son indistinguibles de ruido?"
- **Metricas internas** (Silhouette, Davies-Bouldin) responden: "esa estructura es cohesiva y separable, o es difusa?"
- **NMI cross-modal** responde: "los clusters semanticos y musicales capturan informacion diferente?" (NMI bajo = alta complementariedad = la fusion esta justificada)

Este rol es puramente analitico. No produce un componente del sistema; produce **evidencia** de que las features de la Etapa 3 son utiles. Sin esta validacion, la recomendacion y la fusion se construyen sobre una premisa no verificada.

#### Rol 2: Mecanismo de agrupacion para el sistema de recomendacion (funcional)

Las opciones reales son:

- **k-NN directo** (sin clustering): para cada cancion, buscar las K mas similares por distancia coseno. Simple, efectivo, sin perdida de informacion.
- **Clustering + recomendacion intra/inter-cluster**: primero agrupar, luego recomendar dentro del cluster o diversificar entre clusters. Mas complejo, pero permite diversidad controlada e interpretabilidad.

La v1 eligio k-NN directo despues de que el clustering fallara. Eso no invalida el clustering como componente — invalida el clustering **mal ejecutado**. La pregunta no es "clustering si o no", sino "en que condiciones el clustering aporta valor sobre k-NN directo, y como lo medimos".

**Evaluacion del Rol 2 (post Fase B, 2026-03-28)**: Los resultados muestran que la granularidad natural de los datos es baja (k=2 musical, k=3 semantico UMAP). Con clusters de 5,000-9,000 canciones, "pertenecer al mismo cluster" es una senal demasiado debil para recomendacion. **Decision: k-NN directo como mecanismo primario de recomendacion en Etapa 5**. El clustering se conserva como herramienta analitica (Rol 1 cumplido) y como mecanismo auxiliar de diversificacion (los 2-3 clusters pueden garantizar que las recomendaciones no se concentren en un solo grupo tematico/sonoro).

---

## 2. PROBLEMAS TECNICOS Y DECISIONES FUNDAMENTADAS

### 2.1 Maldicion de la dimensionalidad en 384D

**Problema** (Aggarwal, Hinneburg & Keim, 2001): en 384 dimensiones, la diferencia relativa entre la distancia al vecino mas cercano y al mas lejano tiende a cero. Consecuencias:

- K-Means se degrada: centroides pierden significado discriminativo
- Hopkins pierde potencia estadistica (puede dar valores altos sin estructura real)
- HDBSCAN descarta demasiados puntos como ruido
- Distancia euclidiana pierde capacidad discriminativa

**Mitigacion existente**: nuestros embeddings estan L2-normalizados (confinados a la hiperesfera unitaria), lo que preserva discriminabilidad angular (distancia coseno). Esto mitiga, no elimina el problema.

**Mitigacion adicional posible**: UMAP como reduccion dimensional previa al clustering del espacio semantico. Allaoui, Kherfi & Cheriet (2020) reportan mejoras de hasta 60% en calidad de clustering con UMAP como preprocesamiento. Pero introduce un hiperparametro (`n_components`) cuya eleccion requiere justificacion.

**Decision**: Evaluar clustering en espacio original (384D con distancia coseno) Y en proyeccion UMAP. Comparar resultados para decidir con evidencia, no con suposiciones.

### 2.2 Hopkins en alta dimensionalidad

El Hopkins 0.7752 de la v1 en 384D se interpretaba con cautela por la advertencia de Adzhemyan et al. (2018) de que Hopkins pierde potencia en alta dimensionalidad.

**Decision**: Calcular Hopkins sobre 4 espacios con distribuciones de referencia correctas.

**Resultado (ejecutado 2026-03-28)**: El diagnostico de curse-of-dimensionality dio resultado **inverso** al esperado: Hopkins UMAP (0.9923) es mayor que Hopkins 384D (0.9472). Si la alta dimensionalidad inflara Hopkins artificialmente, la proyeccion UMAP deberia dar un valor menor. Lo opuesto ocurre: UMAP concentra la estructura en menos dimensiones y Hopkins la detecta con mas fuerza. Esto confirma que el 0.9472 en 384D es conservador, no inflado.

**Detalle tecnico critico**: Para el espacio semantico (L2-normalizado en hiperesfera unitaria), los puntos uniformes de referencia se generan sobre la hiperesfera (vectores normales multivariados + L2-normalizacion), no en el hipercubo. Implementado en `hopkins.py:_generate_uniform_hypersphere()`.

### 2.3 Silhouette y distribuciones degeneradas

La v1 obtuvo Silhouette 0.6733 con clustering jerarquico semantico — un valor excelente que resulto ser artefacto de distribucion 99.98%/0.02%. La version micro-averaged (estandar scikit-learn) es vulnerable a desbalance extremo.

**Decision**:
- Usar Silhouette **macro-averaged** (promedio por cluster, no por punto) como metrica primaria
- Reportar **ambas versiones** (micro y macro) para transparencia
- **Siempre** reportar distribucion de tamanos de clusters junto con metricas
- Rechazar automaticamente configuraciones con algun cluster < 1% del dataset

### 2.4 Metricas de evaluacion interna

Segun Chicco et al. (2025, PeerJ — estudio empirico sobre 10 datasets):

| Metrica | Consistencia con ARI | Rol en v2 |
|---------|---------------------|-----------|
| Silhouette | 100% (10/10) | **Primaria** |
| Davies-Bouldin | 100% (10/10) | **Primaria** |
| Calinski-Harabasz | 70% (7/10) | Complementaria |
| Dunn | 60% en reales | No usar |
| Gap Statistic | 80% (8/10) | Complementaria |

**Decision**: Triangulacion Silhouette macro + Davies-Bouldin como metricas de seleccion. Calinski-Harabasz como complementaria. No usar Dunn.

### 2.5 Genero como proxy de ground truth

El genero (6 categorias) es la unica validacion externa disponible, pero:

- Las etiquetas son subjetivas (acuerdo inter-evaluador ~70-80%)
- No son mutuamente excluyentes
- Capturan solo una dimension de similitud musical
- Evolucionan temporalmente

**Decision**:
- Declarar **upfront** en el informe que el genero es proxy, no ground truth absoluto
- Usar ARI/NMI contra genero como indicador **complementario**, no como metrica primaria
- Si clusters no correlacionan con genero, eso no implica clusters malos — puede indicar que capturan otra dimension (tematica, energia, mood)
- NO optimizar clustering para maximizar concordancia con genero (eso seria circular)

### 2.6 UMAP: visualizacion vs preprocesamiento

UMAP tiene dos usos distintos que no deben confundirse:

1. **Visualizacion** (n_components=2): para figuras del informe, validacion cualitativa. Siempre se hace.
2. **Preprocesamiento** (n_components=10-50): reduccion dimensional antes de clustering. Solo si la evidencia muestra que clustering en 384D es problematico.

**Decision**: UMAP para visualizacion es obligatorio. UMAP como preprocesamiento se evalua comparando metricas de clustering con y sin reduccion.

---

## 3. PRE-REGISTRO DE HIPOTESIS

Formuladas ANTES de ejecutar experimentos (correccion explicita del problema post-hoc de v1):

- **H1**: Ambos espacios (semantico y musical) exhiben tendencia al agrupamiento significativa (Hopkins > 0.7 con intervalo de confianza bootstrap)
- **H2**: El espacio musical (13D) produce clusters mas cohesivos que el semantico (384D) en dimensionalidad original (Silhouette musical > Silhouette semantico), debido a la maldicion de la dimensionalidad
- **H3**: Los clusters semanticos y musicales capturan informacion complementaria (NMI cross-modal < 0.15), lo que justifica la fusion multimodal
- **H4**: La purificacion post-clustering mejora Silhouette sin eliminar mas del 15% de los datos

Si alguna hipotesis se rechaza, se documenta como resultado negativo en el informe, no se reformula.

---

## 4. PIPELINE — ESTADO DE EJECUCION

### Paso 1: Tendencia al agrupamiento (Hopkins) — COMPLETADO (2026-03-28)

**Modulos implementados:**

| Modulo | Ubicacion | Contenido |
|--------|-----------|-----------|
| Hopkins | `src/clustering/hopkins.py` | `HopkinsReport`, `compute_hopkins_statistic()` (hiperesfera/hipercubo), `compute_hopkins_bootstrap()` |
| Reduccion | `src/clustering/reduction.py` | `UMAPReport`, `reduce_umap()`, `reduce_for_visualization()`, `prepare_spaces_for_hopkins()` |
| Tablas | `src/clustering/tables.py` | `generate_hopkins_table()` |
| Plots | `src/clustering/plots.py` | `plot_hopkins_barplot()` |
| Orquestador | `src/clustering/run_hopkins.py` | `python -m src.clustering.run_hopkins` — pipeline 9 pasos |

**Artefactos generados (ejecucion exitosa, 90.6s):**
- `results/metrics/etapa4_hopkins.json`
- `results/tables/hopkins_results.tex` + `thesis/tables/`
- `results/figures/hopkins_results.pdf` + `thesis/figures/`

**Resultados:**

| Espacio | Dims | Metrica | Ref. uniforme | Hopkins (media +/- DE) | IC 95% |
|---------|------|---------|---------------|------------------------|--------|
| Semantico 384D | 384 | coseno | hiperesfera | **0.9472 +/- 0.0028** | [0.9420, 0.9517] |
| Semantico UMAP | 30 | euclidiana | hipercubo | **0.9923 +/- 0.0005** | [0.9914, 0.9931] |
| Musical 13D | 13 | euclidiana | hipercubo | **0.8298 +/- 0.0103** | [0.8110, 0.8447] |
| Concatenado 397D | 397 | euclidiana | hipercubo | **0.7415 +/- 0.0088** | [0.7295, 0.7554] |

**H1: CONFIRMED** — los cuatro espacios superan 0.7.

**Comparacion con v1:**

| Espacio | v2 | v1 | Diferencia |
|---------|----|----|-----------|
| Semantico | 0.9472 | 0.7752 | +0.1720 (E5 + chunking + dataset mayor) |
| Musical | 0.8298 | 0.7871 | +0.0427 (dataset mayor + key circular) |

**Hallazgos clave:**
1. El espacio semantico tiene estructura mucho mas fuerte que en v1 (E5-small + chunking + 2.3x datos)
2. La curse-of-dimensionality NO infla Hopkins: UMAP 30D (0.9923) > 384D (0.9472), confirmando que el valor en 384D es conservador
3. El espacio concatenado es el mas debil (0.7415) — las 384D semanticas dominan sobre las 13 musicales, diluyendo la contribucion musical
4. **Implicacion**: el clustering debe hacerse por espacio separado, no sobre el concatenado

### Pasos 2-3: Multi-algoritmo y NMI cross-modal — COMPLETADO (2026-03-28)

**Modulos implementados:**

| Modulo | Ubicacion | Contenido |
|--------|-----------|-----------|
| Algoritmos | `src/clustering/algorithms.py` | `ClusteringResult`, `run_kmeans()`, `run_ward()`, `run_hdbscan()`, `run_all_algorithms()` |
| Evaluacion | `src/clustering/evaluation.py` | `ClusteringMetrics`, `evaluate_clustering()` (Sil macro/micro, DB, CH, ARI, NMI), `compute_cross_modal_nmi()`, `select_best_configuration()` |
| Tablas | `src/clustering/tables.py` | +`generate_clustering_comparison_table()`, `generate_best_configurations_table()`, `generate_cross_modal_nmi_table()` |
| Plots | `src/clustering/plots.py` | +`plot_silhouette_comparison()`, `plot_cluster_sizes()` |
| Orquestador | `src/clustering/run_clustering.py` | `python -m src.clustering.run_clustering` — pipeline 12 pasos |

**Artefactos generados (ejecucion exitosa, 2400.8s = ~40 min):**
- `results/metrics/etapa4_clustering.json`
- `results/tables/clustering_comparison_{musical_13d,semantic_384d,semantic_umap}.tex` + `thesis/tables/`
- `results/tables/clustering_best_configs.tex` + `thesis/tables/`
- `results/tables/cross_modal_nmi.tex` + `thesis/tables/`
- `results/figures/silhouette_comparison_{musical_13d,semantic_384d,semantic_umap}.pdf` + `thesis/figures/`
- `results/figures/cluster_sizes_{best_configs}.pdf` + `thesis/figures/`

**Resultados — Mejor configuracion por espacio:**

| Espacio | Algoritmo | K | Sil_macro | Sil_micro | DB | NMI_genero |
|---------|-----------|---|-----------|-----------|-----|-----------|
| Musical 13D | HDBSCAN (min_size=200) | 2 | 0.1991 | 0.1887 | 1.9748 | 0.0277 |
| Semantico 384D | K-Means++ | 5 | 0.0362 | 0.0277 | 5.0523 | 0.1885 |
| Semantico UMAP 30D | HDBSCAN (min_size=200) | 3 | 0.9131 | 0.8237 | 0.1619 | 0.1227 |

**NMI cross-modal: 0.0096** (entre semantico 384D K-Means k=5 y musical HDBSCAN k=2).

**H2: CONFIRMED** — Sil musical (0.1991) > Sil semantico (0.0362).
**H3: CONFIRMED** — NMI cross-modal (0.0096) < 0.15. Complementariedad casi total.

**Hallazgos clave del multi-algoritmo:**

1. **Semantico 384D: clustering muy pobre.** Hopkins fue alto (0.9472) pero Silhouette fue 0.0362 — la estructura existe pero no se traduce en clusters bien separados. La curse-of-dimensionality no afecta la deteccion de estructura (Hopkins) pero si la calidad de particion (Silhouette). HDBSCAN fue completamente inutilizable en 384D (100% noise con min_size >= 100).

2. **Musical 13D: clustering debil.** K-Means fue el mejor algoritmo de particion (Sil_macro=0.1028 con k=5). HDBSCAN encontro 2 regiones densas pero con 84% noise. Los clusters musicales no correlacionan con genero (NMI=0.0277).

3. **Semantico UMAP 30D: resultado mas prometedor.** UMAP es esencial para clustering del espacio semantico. HDBSCAN encontro 3 clusters con solo 3.6% noise y Sil_macro=0.9131. K-Means/Ward dieron Sil_macro 0.5-0.68 pero con k>=6 generaron clusters degenerados. La estructura natural del espacio semantico proyectado es de ~3-5 grupos.

4. **Granularidad natural baja.** El espacio musical tiene ~2 regiones densas, el semantico ~3 grupos. Con clusters de 5,000-9,000 canciones, el clustering no tiene granularidad suficiente para recomendacion fina. Forzar K mas altos produce clusters degenerados o de baja calidad.

5. **NMI cross-modal extremadamente bajo (0.0096).** Los clusters semanticos y musicales son practicamente independientes, incluso mas que en v1 (0.0567). Esto justifica plenamente la fusion multimodal.

6. **Implicacion para Etapa 5**: k-NN directo como mecanismo primario de recomendacion. Clustering como herramienta analitica y auxiliar de diversificacion.

**Comparacion con v1:**

| Metrica | v2 | v1 | Nota |
|---------|----|----|------|
| NMI cross-modal | 0.0096 | 0.0567 | Complementariedad aun mayor en v2 |
| Sil musical (mejor) | 0.1991 (HDBSCAN k=2) | 0.1554 (baseline) | No directamente comparable (HDBSCAN vs K-Means, 84% noise) |
| Sil semantico 384D | 0.0362 | ~0.06-0.11 (K-Means) | v2 ligeramente peor, dataset mas grande |

### Paso 4: Purificacion — PENDIENTE

Estrategias a evaluar sobre las mejores configuraciones:
- Eliminacion de puntos con Silhouette negativo
- Eliminacion de outliers por distancia al centroide (> 2 sigma)
- Combinada con cap en 15% de remocion

Candidatos para purificacion:
- Musical K-Means k=5 (Sil_macro=0.1028) — mas margen de mejora
- Semantico UMAP HDBSCAN k=3 (Sil_macro=0.9131) — ya excelente, mejora marginal esperada

### Paso 5: Visualizacion UMAP 2D — PENDIENTE

- Scatter plot coloreado por cluster assignment
- Scatter plot coloreado por genero (validacion visual contra proxy)
- Figuras para thesis/figures/

---

## 5. LO QUE NO SE DEBE HACER (lecciones de v1)

| Prohibicion | Razon | Referencia v1 |
|-------------|-------|---------------|
| Datos sinteticos para entrenar modelos | Performance predictor con 20 obs. hardcoded daba Silhouette predicho = 1.0 | `performance_predictor.py` |
| Funciones que devuelven constantes | Gap analysis retornaba optimal_k=3 hardcoded, natural structure score retornaba 0.75 | `dimensionality_impact_assessment.py` |
| Hipotesis post-hoc | Se esperaba Hopkins bajo en 384D, resulto alto; se reformulo la hipotesis | FULL_PROJECT.md |
| Metricas infladas por distribuciones degeneradas | Silhouette 0.6733 con 99.98%/0.02% es artefacto | Clustering jerarquico semantico |
| Limitaciones presentadas como decisiones positivas | Pesos 55/45 vs optimo 20/80 presentados como "decision de diseno" | Sistema de recomendacion |
| Seleccion de datos por conveniencia del clustering | 18K -> 7,811 (57.7% perdida) sin justificacion | `select_optimal_10k_from_18k.py` |
| Toda funcion devuelve resultados reales o se elimina | No existen funciones stub/placeholder | Directiva global v2 |

---

## 6. BENCHMARKS DE REFERENCIA (v1)

Valores de la primera ejecucion que sirven como referencia, no como objetivo a alcanzar:

| Metrica | Valor v1 | Dataset v1 | Dataset v2 | Nota |
|---------|----------|-----------|-----------|------|
| Hopkins Semantico | 0.7752 +/- 0.0015 | 7,811 x 384 | 17,964 x 384 | Interpretar con cautela (384D) |
| Hopkins Musical | 0.7871 +/- 0.0022 | 7,811 x 12 | 17,964 x 13 | v2 tiene key circular (12->13D) |
| Silhouette post-purif. | 0.2893 | 7,811 musical | 17,964 | v1 solo musical |
| NMI cross-modal | 0.0567 | 7,811 | 17,964 | Complementariedad alta |
| Precision@10 | 0.398 | 7,811 hibrido | 17,964 | Etapa 5, no Etapa 4 |

Diferencias clave que impiden comparacion directa:
- v2 tiene 2.3x mas datos (17,964 vs 7,811)
- v2 usa E5-small (512 tokens) vs MiniLM (128 tokens) — embeddings distintos
- v2 tiene 13 features musicales (key circular) vs 12 en v1
- v2 aplica chunking (cobertura 100%) vs truncamiento en v1

---

## 7. LITERATURA CLAVE PARA ESTA ETAPA

### Fuentes principales (de thesis/investigacion/)

| Referencia | Aporte relevante |
|-----------|-----------------|
| Aggarwal, Hinneburg & Keim (2001) | Curse of dimensionality: metricas de distancia pierden discriminabilidad |
| Arthur & Vassilvitskii (2007) | K-Means++: inicializacion D2-weighted, garantia O(log k) |
| Campello, Moulavi & Sander (2013) | HDBSCAN: extension jerarquica de DBSCAN sin epsilon fijo |
| McInnes, Healy & Melville (2018) | UMAP: reduccion no lineal, preserva estructura global |
| Allaoui, Kherfi & Cheriet (2020) | UMAP como preprocesamiento mejora clustering hasta 60% |
| Chicco et al. (2025) | Silhouette + Davies-Bouldin: 100% consistencia con ARI |
| Murtagh & Legendre (2014) | Ward tiene dos implementaciones distintas (Ward.D vs Ward.D2) |
| Rousseeuw (1987) | Silhouette coefficient: definicion y propiedades |
| Dinh & Huynh (2021) | Refinamiento iterativo post-clustering |
| Tibshirani, Walther & Hastie (2001) | Gap Statistic: seleccion de K con fundamentacion estadistica |

### Hallazgo transversal

La purificacion post-clustering **no tiene framework formal en la literatura** (Gan & Ng 2017, Dinh & Huynh 2021). Esto posiciona la purificacion hibrida como contribucion metodologica original del proyecto.

---

## 8. ENTREGABLES LaTeX DE ESTA ETAPA

### Marco Teorico (Cap. 4)

Subsecciones de §4.5 a redactar:

| Seccion | Contenido | Estado |
|---------|-----------|--------|
| §4.5.1 Hopkins Statistic | Formula, interpretacion, limitaciones en alta D | Esqueleto |
| §4.5.2 K-Means y K-Means++ | WCSS, inicializacion D2, convergencia | Esqueleto |
| §4.5.3 Ward | Criterio de minima varianza, dendrograma, Ward.D vs Ward.D2 | Esqueleto |
| §4.5.4 HDBSCAN | Jerarquia de densidades, min_cluster_size, noise | Esqueleto |
| §4.5.5 Evaluacion interna | Silhouette, Davies-Bouldin, Calinski-Harabasz | Esqueleto |
| §4.5.6 Maldicion de la dimensionalidad | Convergencia de distancias, mitigaciones | Esqueleto |
| §4.5.7 UMAP | Grafo de vecinos, optimizacion layout, hiperparametros | Esqueleto |
| §4.5.8 Purificacion | Estrategias, contribucion original, metricas pre/post | Esqueleto |

### Solucion Propuesta (Cap. 5)

| Seccion | Contenido | Estado |
|---------|-----------|--------|
| §5.6.1 Evaluacion Hopkins | Resultados Hopkins en 4 espacios | Esqueleto vacio |
| §5.6.2 Seleccion algoritmo | Comparacion multi-algoritmo, seleccion justificada | Esqueleto vacio |
| §5.6.3 Purificacion | Procedimiento, resultados pre/post | Esqueleto vacio |

### Resultados (Cap. 6)

| Seccion | Contenido | Estado |
|---------|-----------|--------|
| §6.2.1 Tendencia al agrupamiento | Tablas Hopkins, interpretacion | Esqueleto vacio |
| §6.2.2 Comparacion de algoritmos | Tablas metricas, seleccion | Esqueleto vacio |
| §6.2.3 Efecto de purificacion | Pre/post, trade-off pureza/cobertura | Esqueleto vacio |

---

## 9. CONFIGURACION EN src/config.py

Parametros de clustering (todos definidos):

```python
# Hopkins
HOPKINS_ITERATIONS = 30
HOPKINS_SAMPLE_SIZE = 100
HOPKINS_THRESHOLD = 0.7

# Algoritmos
CLUSTERING_K_RANGE = [5, 6, 7, 8]
HDBSCAN_MIN_CLUSTER_SIZES = [50, 100, 200, 300, 500]

# UMAP
UMAP_PREPROCESSING_N_COMPONENTS = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Umbrales de calidad
MIN_CLUSTER_PCT = 0.01           # rechazar clusters < 1% del dataset
PURIFICATION_MAX_REMOVAL_PCT = 0.15
PURIFICATION_SIGMA_THRESHOLD = 2.0

# Paths
CLUSTERED_DIR = DATA_DIR / "6_clustered"
CLUSTERED_DATASET = CLUSTERED_DIR / "clustered_dataset.npz"

# Referencia v1 (no usar para seleccion)
DBSCAN_EPS_RANGE = [0.1, 0.15, 0.2, 0.25, 0.3]
OBJECTIVE_WEIGHTS = { ... }  # legacy, no se usa en v2
```
