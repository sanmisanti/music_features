# Algoritmos de Clustering y Evaluacion: Revision Sistematica de Literatura

**Fecha de elaboracion:** 2026-02-07
**Contexto:** Tesis de Ingenieria Informatica — Sistema de Recomendacion Musical Hibrido
**Autor de la revision:** Investigacion asistida por Claude Opus 4.6

---

## 1. Resumen Ejecutivo

La presente revision sistematica examina el estado de la cuestion en algoritmos de clustering, metricas de evaluacion interna y externa, analisis de tendencia al clustering, y tecnicas complementarias como reduccion dimensional y purificacion post-clustering. La busqueda sistematica cubrio las bases Google Scholar, Semantic Scholar, ACM Digital Library, IEEE Xplore y PeerJ, identificando aproximadamente 25 fuentes relevantes entre publicaciones seminales (1954-2007) y contribuciones recientes (2018-2025). Los hallazgos principales son: (i) K-Means++ con inicializacion D2-weighted constituye el estandar de facto para clustering particional con garantias O(log k); (ii) el Silhouette coefficient y el Davies-Bouldin index son las metricas internas mas consistentes segun la comparacion empirica de Chicco et al. (2025); (iii) la maldicion de dimensionalidad afecta severamente la evaluacion de clustering en espacios de alta dimension como los embeddings BERT de 384D, justificando el uso de UMAP como paso previo; (iv) la purificacion post-clustering carece de un framework formal unificado en la literatura, constituyendo una oportunidad de contribucion para el proyecto; y (v) el Hopkins statistic, si bien valido para evaluar tendencia al clustering, pierde potencia en alta dimensionalidad, lo cual debe considerarse al interpretar los valores obtenidos en el espacio semantico.

---

## 2. Estrategia de Busqueda

### 2.1 Palabras Clave

**Ingles (idioma de busqueda primario):**
- clustering algorithms comparison survey
- Hopkins statistic clustering tendency high dimensional
- silhouette coefficient Davies-Bouldin Calinski-Harabasz internal evaluation
- UMAP dimensionality reduction clustering
- spectral clustering normalized cuts
- HDBSCAN DBSCAN comparison
- K-Means++ Arthur Vassilvitskii
- Ward hierarchical clustering implementations
- gap statistic optimal clusters
- adjusted rand index normalized mutual information
- clustering text embeddings BERT
- post-clustering refinement purification
- music information retrieval clustering

**Espanol (complementario):**
- algoritmos de clustering, evaluacion interna de clustering, maldicion de dimensionalidad

### 2.2 Fuentes Consultadas y Resultados

| Fuente | Consultas realizadas | Resultados relevantes |
|--------|---------------------|----------------------|
| Google Scholar (via WebSearch) | 4 | 12 fuentes |
| Semantic Scholar | 2 | 5 fuentes |
| ACM Digital Library | 2 | 3 fuentes |
| IEEE Xplore | 1 | 2 fuentes |
| PeerJ / Springer / JMLR | 3 | 6 fuentes |
| **Total** | **12** | **~28 fuentes unicas** |

### 2.3 Criterios de Inclusion y Exclusion

**Inclusion:**
- Publicaciones revisadas por pares o preprints de alto impacto (>100 citas o en venues reconocidos).
- Periodo: 1954-2026, con enfasis en publicaciones post-2015 para el estado actual y seminales historicas para fundamentos.
- Relevancia directa a clustering de datos multimodales, embeddings textuales, o features musicales.

**Exclusion:**
- Blogs sin fundamentacion academica, publicaciones en journals predatorios.
- Trabajos sin autor identificable o sin revision por pares (excepto preprints seminales como UMAP).
- Duplicados o versiones preliminares cuando existe version publicada.

---

## 3. Estado de la Cuestion

### 3.1 Evaluacion de Tendencia al Clustering: Hopkins Statistic

#### 3.1.1 Fundamentos teoricos

El Hopkins statistic fue propuesto originalmente por Hopkins y Skellam (1954) como un test de aleatoriedad espacial, y posteriormente adaptado para evaluar la tendencia al clustering (clustering tendency) de un dataset. La formulacion mide la probabilidad de que un conjunto de datos dado sea generado por una distribucion uniforme, comparando las distancias nearest-neighbor de puntos reales con las de puntos generados aleatoriamente dentro del hiperrectangulo que delimita los datos.

El estadistico retorna un valor H en el intervalo [0, 1] con la siguiente interpretacion:
- H cercano a 0.5: datos distribuidos aleatoriamente (sin estructura de clusters).
- H > 0.7: evidencia de tendencia al clustering.
- H > 0.75: tendencia al clustering significativa al 90% de confianza (Banerjee & Dave, 2004).
- H cercano a 1.0: alta tendencia al clustering.

#### 3.1.2 Limitaciones en alta dimensionalidad

Un aspecto critico para el proyecto es que el Hopkins statistic pierde potencia en espacios de alta dimensionalidad. Adzhemyan et al. (2018) demostraron en su analisis de metodos de clusterability que la efectividad del test se degrada a medida que aumentan las dimensiones, debido a la convergencia de distancias (curse of dimensionality). Lawson y Jurs (1990) establecieron que el parametro m (numero de puntos de muestreo) debe ser al menos 10, limitando efectivamente el metodo a datasets con al menos 100 observaciones.

**Implicacion para el proyecto:** Los valores de Hopkins obtenidos en la primera ejecucion (0.7752 para 384D semantico y 0.7871 para 12D musical) deben interpretarse con cautela para el espacio semantico. El valor en el espacio musical de 12D es mas confiable. Se recomienda complementar con Hopkins calculado sobre proyecciones UMAP del espacio semantico.

#### 3.1.3 Alternativas y complementos

Adzhemyan et al. (2018) compararon multiples tests de clusterability, incluyendo Hopkins, el dip test de Hartigan y Hartigan (1985), y metodos basados en PCA. Concluyeron que ningun test individual es universalmente superior, y recomiendan reportar multiples indicadores. La practica de calcular Hopkins sobre multiples muestras y reportar media con desviacion estandar (como se hizo en v1) es consistente con las mejores practicas de la literatura.

### 3.2 Algoritmos de Clustering Particional

#### 3.2.1 K-Means: de Lloyd a K-Means++

El algoritmo K-Means tiene una historia que abarca multiples formulaciones independientes. MacQueen (1967) propuso el nombre y la formulacion estadistica; Lloyd (1982) publico la version algoritmica que se usa comunmente hoy (originalmente concebida en 1957 en Bell Labs). El algoritmo busca minimizar la suma de distancias cuadraticas intra-cluster (within-cluster sum of squares, WCSS) mediante asignacion iterativa de puntos al centroide mas cercano y recalculo de centroides.

La limitacion fundamental de K-Means es su sensibilidad a la inicializacion. Arthur y Vassilvitskii (2007) propusieron K-Means++ como solucion, introduciendo una inicializacion probabilistica D2-weighted donde cada nuevo centroide se selecciona con probabilidad proporcional al cuadrado de la distancia al centroide mas cercano ya seleccionado. Esta inicializacion proporciona:
- **Garantia teorica:** aproximacion O(log k) al optimo en expectativa.
- **Mejora empirica:** convergencia mas rapida y resultados significativamente mejores que inicializacion aleatoria.
- **Adoption universal:** implementado como default en scikit-learn y la mayoria de librerias modernas.

Bachem et al. (2016) propusieron una variante acelerada (K-Means||) que reduce el costo computacional de la inicializacion de O(nkd) a O(n*d), particularmente util para datasets grandes.

**Limitaciones inherentes de K-Means:**
- Asume clusters esfericos (convexos) de tamano similar.
- Requiere especificar k a priori.
- No maneja ruido ni outliers explicitamente.
- Sensible a la escala de las features (requiere normalizacion previa).

#### 3.2.2 Seleccion del numero optimo de clusters

El **Gap statistic** (Tibshirani, Walther & Hastie, 2001) aborda formalmente la seleccion de k comparando la dispersion intra-cluster observada con la esperada bajo una distribucion de referencia uniforme. La formulacion es:

Gap_n(k) = E*[log(W_k)] - log(W_k)

donde W_k es la dispersion intra-cluster y E* denota la expectativa bajo la distribucion de referencia generada por bootstrap. El k optimo se selecciona como el menor k tal que Gap(k) >= Gap(k+1) - s_{k+1}, donde s es el error estandar.

La ventaja del Gap statistic es su fundamentacion estadistica rigurosa. Sin embargo, Chicco et al. (2025) encontraron que su comportamiento es inconsistente en datasets reales con outliers, lo cual es relevante para datos musicales que frecuentemente contienen valores atipicos.

**Metodo complementario: Elbow method.** Aunque carece de fundamentacion formal, su uso conjunto con Gap statistic y Silhouette analysis proporciona un enfoque triangulado mas robusto.

### 3.3 Clustering Jerarquico Aglomerativo

#### 3.3.1 Metodo de Ward y la ambiguedad de implementacion

Ward (1963) propuso un criterio de clustering jerarquico que minimiza el incremento total en la suma de cuadrados intra-cluster en cada paso de fusion. El metodo es optimo para clusters compactos y esfericos, y produce dendrogramas que permiten explorar la estructura a multiples niveles de granularidad.

**El problema de las dos implementaciones.** Murtagh y Legendre (2014) identificaron una discrepancia critica en la implementacion del criterio de Ward. Demostraron que existen dos algoritmos ampliamente usados en software, ambos anunciando implementar el metodo de Ward, pero que producen resultados distintos cuando se aplican a la misma matriz de distancias:

1. **Ward.D (WARD1):** Usa la distancia entre clusters sin elevar al cuadrado. NO preserva el criterio de Ward.
2. **Ward.D2 (WARD2):** Usa la distancia al cuadrado. SI preserva fielmente el criterio original de Ward (1963).

La confusion se origina en la formula de actualizacion de Lance-Williams, donde la distancia puede interpretarse como d o d^2. Murtagh y Legendre demostraron que solo la implementacion que opera sobre distancias al cuadrado produce la solucion que minimiza el error sum of squares, tal como Ward lo formulo originalmente.

**Implicacion para el proyecto:** Al implementar clustering jerarquico con criterio Ward, es imperativo verificar que la implementacion utilizada corresponda a Ward.D2 (en R) o `method='ward'` con distancias euclidianas al cuadrado (en scipy/scikit-learn). Scikit-learn implementa correctamente el criterio de Ward cuando se usa `AgglomerativeClustering(linkage='ward')` con distancia euclidiana.

#### 3.3.2 Otros criterios de enlace

Ademas de Ward, los criterios de enlace mas utilizados son:
- **Single linkage:** distancia minima entre cualquier par de puntos. Propenso al efecto cadena (chaining effect).
- **Complete linkage:** distancia maxima. Produce clusters compactos pero sensible a outliers.
- **Average linkage (UPGMA):** promedio de todas las distancias par-a-par. Balance entre single y complete.

Para datos de alta dimensionalidad, Ward y average linkage son generalmente preferidos (Murtagh & Contreras, 2012).

### 3.4 Clustering Basado en Densidad

#### 3.4.1 DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) fue propuesto por Ester et al. (1996) y constituyó un cambio paradigmatico al definir clusters como regiones densas separadas por regiones de baja densidad. Sus dos parametros son:
- **epsilon (eps):** radio de vecindario.
- **MinPts:** numero minimo de puntos para definir una region densa.

**Ventajas:** detecta clusters de forma arbitraria, identifica outliers explicitamente (puntos de ruido), no requiere especificar k a priori.

**Limitaciones criticas:**
- Asume densidad uniforme: no maneja clusters de densidad variable.
- La seleccion de epsilon es problematica, especialmente en alta dimensionalidad donde las distancias convergen (Aggarwal, Hinneburg & Keim, 2001).
- Complejidad O(n^2) sin estructuras de indexacion espacial (con KD-tree, O(n log n)).

#### 3.4.2 HDBSCAN

Campello, Moulavi y Sander (2013) propusieron HDBSCAN (Hierarchical DBSCAN) como extension que elimina la dependencia del parametro epsilon. El algoritmo:
1. Construye un arbol de spanning minimo sobre el grafo de mutual reachability.
2. Genera una jerarquia completa de clustering variando epsilon.
3. Extrae clusters estables usando un criterio de persistencia basado en exceso de masa.

McInnes y Healy (2017) desarrollaron una implementacion acelerada que logra complejidad O(n log n) en la practica, integrada actualmente en scikit-learn (desde v1.3).

**Ventajas sobre DBSCAN:**
- Maneja clusters de densidad variable.
- Elimina la necesidad de seleccionar epsilon.
- Proporciona una jerarquia completa de clustering.
- Identifica outliers de forma natural.
- Unico parametro critico: `min_cluster_size`.

**Limitacion:** El parametro `min_cluster_size` sigue requiriendo ajuste empirico, y el algoritmo puede producir muchos puntos clasificados como ruido en datasets con estructura ambigua.

**Relevancia para el proyecto:** HDBSCAN es particularmente adecuado para el espacio semantico de 384D (o su proyeccion UMAP) donde la densidad de clusters puede variar significativamente segun el genero o tematica lirica.

### 3.5 Clustering Espectral

#### 3.5.1 Fundamentos

El clustering espectral opera sobre la representacion espectral (eigenvectors) de una matriz de afinidad derivada de los datos. La formulacion seminal de Shi y Malik (2000) propuso Normalized Cuts como criterio de particion de grafos, minimizando el corte normalizado entre segmentos. Ng, Jordan y Weiss (2002) propusieron un algoritmo practico que:
1. Construye una matriz de afinidad (tipicamente kernel gaussiano).
2. Calcula los k eigenvectors principales del Laplaciano normalizado.
3. Aplica K-Means sobre los eigenvectors.

Von Luxburg (2007) publico el tutorial definitivo que unifica las variantes del clustering espectral, describiendo las propiedades de diferentes Laplacianos de grafo (no normalizado, normalizado simetrico, y normalizado random walk) y sus implicaciones para la calidad del clustering.

#### 3.5.2 Variantes del Laplaciano

- **Laplaciano no normalizado (L = D - W):** Minimiza RatioCut. Puede producir particiones desbalanceadas.
- **Laplaciano normalizado simetrico (L_sym = D^{-1/2} L D^{-1/2}):** Corresponde al algoritmo de Ng, Jordan y Weiss (2002).
- **Laplaciano normalizado random walk (L_rw = D^{-1} L):** Corresponde a la formulacion de Shi y Malik (2000). Von Luxburg (2007) argumenta que este es teoricamente preferible.

#### 3.5.3 Ventajas y limitaciones

**Ventajas:**
- Captura relaciones no lineales entre puntos.
- Puede identificar clusters de forma arbitraria (no necesariamente convexos).
- Bien fundamentado en teoria de grafos espectrales.

**Limitaciones:**
- Complejidad O(n^3) para el calculo de eigenvectors (prohibitivo para n > 10,000).
- La seleccion del parametro sigma del kernel gaussiano es critica.
- El paso final de K-Means hereda la sensibilidad a la inicializacion.
- Escalabilidad limitada: para el dataset del proyecto (18,454 canciones), requeriria aproximaciones o submuestreo.

**Relevancia para el proyecto:** Spectral clustering es conceptualmente apropiado para datos musicales donde las relaciones de similaridad son no lineales, pero su costo computacional con matrices de 18K x 18K requiere consideracion. La implementacion de scikit-learn usa aproximaciones basadas en el metodo Nystrom para datasets grandes.

### 3.6 Metricas de Evaluacion Interna

#### 3.6.1 Silhouette Coefficient

Rousseeuw (1987) propuso el coeficiente de Silhouette como medida de la calidad de asignacion de cada punto a su cluster. Para cada punto i:

s(i) = (b(i) - a(i)) / max(a(i), b(i))

donde a(i) es la distancia media intra-cluster y b(i) es la distancia media al cluster mas cercano. El valor global es el promedio sobre todos los puntos, con rango [-1, 1]:
- s > 0.7: estructura fuerte.
- s > 0.5: estructura razonable.
- s > 0.25: estructura debil.
- s < 0.25: no se detecta estructura significativa.

**Revision de la agregacion.** Un trabajo reciente (2024) revisito la estrategia de agregacion del Silhouette, demostrando que la version micro-averaged (estandar) es vulnerable al desbalance de clusters, mientras que la version macro-averaged es significativamente mas robusta. Esta observacion es relevante cuando los clusters musicales tienen tamanos muy dispares.

**Limitacion fundamental:** Silhouette asume clusters convexos y compactos. Para clusters de forma arbitraria (como los que podria producir HDBSCAN), otras metricas como DBCV (Density-Based Clustering Validation) son mas apropiadas.

#### 3.6.2 Calinski-Harabasz Index

Calinski y Harabasz (1974) propusieron un indice basado en la razon entre la dispersion inter-cluster e intra-cluster:

CH(k) = [trace(B_k) / (k-1)] / [trace(W_k) / (n-k)]

donde B_k es la matriz de dispersion entre clusters y W_k la matriz intra-cluster. Valores mayores indican mejor separacion. A diferencia de Silhouette, CH no tiene un rango fijo, lo que dificulta la interpretacion absoluta (solo comparaciones relativas entre distintos k).

#### 3.6.3 Davies-Bouldin Index

Davies y Bouldin (1979) propusieron un indice que mide la similitud promedio entre cada cluster y el cluster mas parecido:

DB = (1/k) * sum_{i=1}^{k} max_{j!=i} [(s_i + s_j) / d(c_i, c_j)]

donde s_i es la dispersion media del cluster i y d(c_i, c_j) es la distancia entre centroides. Valores menores indican mejor clustering (a diferencia de Silhouette y CH donde mayor es mejor).

#### 3.6.4 Comparacion empirica: Chicco et al. (2025)

Chicco, Campagner, Spagnolo, Ciucci y Jurman (2025) publicaron en PeerJ Computer Science la comparacion empirica mas reciente de metricas internas. Su estudio evaluó seis indices (Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn, Shannon entropy, Gap statistic) sobre datasets artificiales y registros medicos electronicos, usando el Adjusted Rand Index como referencia de consistencia. Los hallazgos principales fueron:

1. **Silhouette y Davies-Bouldin:** 100% de consistencia con el ARI en los 10 experimentos (5 artificiales + 5 reales).
2. **Calinski-Harabasz:** Inconsistente en 3 de 10 datasets.
3. **Dunn:** Consistente en datos artificiales pero fallo en datos reales con outliers (60% consistencia).
4. **Gap statistic y Shannon entropy:** Inconsistentes en 2 de 10 datasets.

**Implicacion para el proyecto:** Se recomienda usar Silhouette y Davies-Bouldin como metricas primarias, con Calinski-Harabasz como complementaria. Esta combinacion triangulada proporciona evaluacion robusta para clusters convexos (como los producidos por K-Means y Ward). Para clusters de forma arbitraria, se debe considerar DBCV.

### 3.7 Metricas de Evaluacion Externa

#### 3.7.1 Adjusted Rand Index (ARI)

Hubert y Arabie (1985) propusieron la correccion por azar del Rand Index (Rand, 1971). El ARI mide la concordancia entre dos particiones corrigiendo por la concordancia esperada bajo asignacion aleatoria:

ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

El rango es [-1, 1], donde 1 indica concordancia perfecta, 0 indica concordancia al nivel del azar, y valores negativos indican concordancia peor que el azar.

#### 3.7.2 Normalized Mutual Information (NMI)

Vinh, Epps y Bailey (2010) publicaron un estudio fundamental en JMLR sobre variantes de NMI y su correccion por azar. El NMI mide la informacion compartida entre dos particiones, normalizada para obtener valores en [0, 1]:

NMI(U, V) = 2 * I(U;V) / [H(U) + H(V)]

Los autores identificaron que:
- La NMI estandar **no tiene baseline constante**: la similitud esperada entre dos clusterings aleatorios aumenta con el numero de clusters.
- Propusieron el **Adjusted Mutual Information (AMI)** que corrige este sesgo, analogamente a como el ARI corrige el RI.
- Las variantes `NMI_joint` y `NMI_max` satisfacen propiedades de metrica (positividad, simetria, desigualdad triangular), mientras que otras variantes no.

**Relevancia para el proyecto:** El NMI cross-modal (0.0567 en v1) mide la correspondencia entre los clustering semantico y musical. Un valor bajo es esperable si las modalidades capturan aspectos complementarios de la musica. Se recomienda usar AMI en lugar de NMI estandar para evitar el sesgo por numero de clusters.

### 3.8 Maldicion de Dimensionalidad y Reduccion Dimensional

#### 3.8.1 El problema fundamental

Aggarwal, Hinneburg y Keim (2001) demostraron formalmente que en espacios de alta dimensionalidad, la diferencia relativa entre la distancia maxima y minima a un punto dado converge a cero:

lim_{d->inf} [dist_max - dist_min] / dist_min -> 0

Esto implica que las metricas de distancia basadas en normas L_p pierden capacidad discriminativa, haciendo que algoritmos como K-Means y DBSCAN (que dependen de distancias) se degraden. El fenomeno es particularmente severo para p >= 3 (distancias de Minkowski de orden alto), mientras que la distancia de Manhattan (L_1) y la distancia coseno exhiben mayor robustez relativa.

**Relevancia para el proyecto:** Los embeddings BERT de 384 dimensiones estan plenamente en el regimen donde la maldicion de dimensionalidad es significativa. El espacio musical de 12 dimensiones, en contraste, no sufre este problema de forma severa. Esto fundamenta la necesidad de reduccion dimensional o el uso de distancia coseno (en lugar de euclidiana) para el espacio semantico.

#### 3.8.2 UMAP como solucion

McInnes, Healy y Melville (2018) propusieron UMAP (Uniform Manifold Approximation and Projection), fundamentado en teoria de variedades Riemannianas y topologia algebraica. A diferencia de t-SNE (van der Maaten & Hinton, 2008), UMAP:

- **Preserva estructura global:** mantiene relaciones de distancia tanto locales como globales, mientras t-SNE se enfoca exclusivamente en la estructura local.
- **Es escalable:** complejidad O(n^1.14) empirica vs O(n^2) de t-SNE.
- **Es determinista** (con seed fijo): permite reproducibilidad.
- **Preserva densidad:** el parametro `densmap=True` permite preservar la estructura de densidad local.

Allaoui, Kherfi y Cheriet (2020) demostraron empiricamente que UMAP como preprocesamiento mejora la calidad del clustering hasta en un 60% (medido por accuracy) para K-Means, HDBSCAN, GMM y clustering aglomerativo, ademas de reducir dramaticamente el tiempo de ejecucion (de 26 minutos a 5 segundos para HDBSCAN en MNIST).

**Comparacion UMAP vs t-SNE para clustering:**

| Aspecto | UMAP | t-SNE |
|---------|------|-------|
| Estructura global | Preserva | No preserva |
| Escalabilidad | O(n^1.14) | O(n^2) |
| Reproducibilidad | Determinista con seed | Estocastico |
| Uso para clustering | Recomendado | No recomendado |
| Parametros criticos | n_neighbors, min_dist | perplexity |
| Runtime (MNIST 70K) | < 1 min | ~45 min |

**Implicacion para el proyecto:** Se recomienda aplicar UMAP al espacio semantico de 384D antes del clustering, reduciendo a un espacio de 10-50 dimensiones. Esto: (i) alivia la maldicion de dimensionalidad, (ii) mejora la calidad del clustering, (iii) permite que DBSCAN/HDBSCAN funcionen correctamente con parametro epsilon significativo, y (iv) reduce el costo computacional.

### 3.9 Clustering Aplicado a Musica e Informacion Musical

#### 3.9.1 Multimodalidad en MIR

Schedl, Gomez y Urbano (2014) publicaron una revision comprensiva de Music Information Retrieval que documenta la evolucion desde enfoques unimodales (solo audio o solo texto) hacia sistemas multimodales. La tendencia actual integra multiples fuentes de informacion: audio, letras, metadatos editoriales, y datos de interaccion de usuario.

Mayer, Neumayer y Rauber (2008) demostraron que el clustering musical con features de diferentes fuentes de informacion (timbre, ritmo, letras) produce mejores resultados que cualquier fuente individual, anticipando el enfoque multimodal del proyecto.

Anderson y Schutz (2023) contribuyeron al entendimiento de la importancia de features individuales en el clustering musical, utilizando Accumulated Local Effects para identificar que features contribuyen mas a la asignacion de clusters. Este tipo de analisis de interpretabilidad es relevante para justificar la seleccion de las 12 features de Spotify.

#### 3.9.2 Clustering de embeddings textuales

Lim y Jatowt (2022) evaluaron el rendimiento de BERT como representacion de datos para text clustering, concluyendo que los embeddings de BERT mejoran significativamente los resultados frente a representaciones tradicionales (TF-IDF, Word2Vec), especialmente cuando se combinan con reduccion dimensional.

Vizcarra et al. (2024) investigaron el clustering de embeddings de modelos de lenguaje grandes (incluyendo variantes de BERT), encontrando que la combinacion de sentence transformers con UMAP y HDBSCAN produce los mejores resultados para descubrimiento automatico de topicos.

**Relevancia para el proyecto:** La arquitectura propuesta (BERT embeddings -> UMAP -> clustering) esta alineada con las mejores practicas actuales en la literatura de NLP y MIR.

### 3.10 Purificacion Post-Clustering

#### 3.10.1 Estado de la literatura

La purificacion post-clustering (refinamiento de clusters mediante eliminacion de outliers y reasignacion de puntos frontera) es una practica comun en aplicaciones practicas pero carece de un framework formal unificado en la literatura academica. La revision sistematica identifico las siguientes lineas relacionadas:

**Clustering con eliminacion de outliers (COR).** Gan y Ng (2017) propusieron una funcion objetivo basada en Holoentropy que combina compactacion de clusters con eliminacion simultanea de outliers. Sin embargo, este enfoque es un metodo conjunto (clustering + outlier removal), no un refinamiento post-hoc.

**Refinamiento iterativo.** Dinh y Huynh (2021) publicaron en International Journal of Data Science and Analytics un metodo de cluster refinement que opera como post-procesamiento, reasignando puntos mal clasificados mediante analisis de fronteras de decision. Su enfoque demostro mejoras en Silhouette y ARI sobre los clusters iniciales.

**Outlier detection post-clustering.** Evaluaciones comparativas recientes (2024) han demostrado que la deteccion de outliers basada en clustering (e.g., LOF, isolation forest aplicados a clusters individuales) mejora la calidad cuando se remueven los puntos detectados, pero el incremento depende fuertemente de la metrica utilizada: el indice de Dunn mejora consistentemente, mientras que Davies-Bouldin muestra mejoras solo cuando los outliers son genuinos y no puntos frontera mal asignados.

#### 3.10.2 Enfoque basado en Z-score

Un enfoque reciente (2025) publicado en Scientific Reports propone un mecanismo de refinamiento basado en Z-score para detectar y reasignar muestras distantes dentro de cada cluster. El metodo identifica puntos cuya distancia al centroide excede un umbral basado en la desviacion estandar del cluster, y los reasigna al cluster vecino mas cercano o los marca como outliers.

#### 3.10.3 Gap en la literatura

La revision confirma que **no existe un framework formal estandarizado** para purificacion post-clustering que:
1. Defina criterios sistematicos para la eliminacion de outliers intra-cluster.
2. Establezca umbrales optimos para la reasignacion de puntos frontera.
3. Integre la purificacion dentro del pipeline de evaluacion (considerando el impacto en las metricas).
4. Sea especifico para datos multimodales donde los criterios de purificacion pueden diferir entre modalidades.

**Oportunidad de contribucion:** El proyecto puede formalizar un procedimiento de purificacion hibrida que opere sobre clusters multimodales, definiendo criterios explicitos basados en distancia normalizada al centroide, consenso entre modalidades, y validacion de la mejora en metricas internas post-purificacion.

---

## 4. Tabla de Fuentes Principales

| # | Autores (Ano) | Titulo | Tipo | Citas aprox. | Relevancia | Aporte clave |
|---|---------------|--------|------|-------------|------------|---------------|
| 1 | Chicco, Campagner, Spagnolo, Ciucci & Jurman (2025) | The Silhouette coefficient and the Davies-Bouldin index are more informative than Dunn index, Calinski-Harabasz index... | Journal (PeerJ CS) | Reciente | Alta | Comparacion empirica definitiva de metricas internas |
| 2 | Arthur & Vassilvitskii (2007) | k-means++: The Advantages of Careful Seeding | Conf. (SODA) | >7,000 | Alta | Inicializacion con garantia O(log k) para K-Means |
| 3 | Von Luxburg (2007) | A Tutorial on Spectral Clustering | Journal (Stat. Comp.) | >10,000 | Alta | Tutorial definitivo de clustering espectral |
| 4 | Campello, Moulavi & Sander (2013) | Density-Based Clustering Based on Hierarchical Density Estimates | Conf. (PAKDD) | >4,000 | Alta | HDBSCAN: clustering jerarquico basado en densidad |
| 5 | McInnes, Healy & Melville (2018) | UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction | Preprint (arXiv) | >8,000 | Alta | Reduccion dimensional preservando estructura global |
| 6 | Rousseeuw (1987) | Silhouettes: A graphical aid to the interpretation and validation of cluster analysis | Journal (JCAM) | >15,000 | Alta | Silhouette coefficient: metrica interna estandar |
| 7 | Murtagh & Legendre (2014) | Ward's Hierarchical Agglomerative Clustering Method: Which Algorithms Implement Ward's Criterion? | Journal (J. Classification) | >1,500 | Alta | Clarifica ambiguedad critica en implementaciones de Ward |
| 8 | Vinh, Epps & Bailey (2010) | Information Theoretic Measures for Clusterings Comparison | Journal (JMLR) | >3,500 | Alta | NMI con correccion por azar (AMI) |
| 9 | Tibshirani, Walther & Hastie (2001) | Estimating the Number of Clusters in a Data Set Via the Gap Statistic | Journal (JRSS-B) | >8,000 | Alta | Metodo estadistico formal para seleccion de k |
| 10 | Aggarwal, Hinneburg & Keim (2001) | On the Surprising Behavior of Distance Metrics in High Dimensional Spaces | Conf. (ICDT) | >4,000 | Alta | Demostracion formal de maldicion de dimensionalidad |
| 11 | Ester, Kriegel, Sander & Xu (1996) | A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise | Conf. (KDD) | >20,000 | Alta | DBSCAN: clustering basado en densidad seminal |
| 12 | Hubert & Arabie (1985) | Comparing Partitions | Journal (J. Classification) | >8,000 | Alta | Adjusted Rand Index |
| 13 | Allaoui, Kherfi & Cheriet (2020) | Considerably Improving Clustering Algorithms Using UMAP | Conf. (ICISP) | >200 | Alta | Evidencia empirica de UMAP como preprocesamiento |
| 14 | Shi & Malik (2000) | Normalized Cuts and Image Segmentation | Journal (IEEE TPAMI) | >18,000 | Media | Formulacion seminal de normalized cuts |
| 15 | Ng, Jordan & Weiss (2002) | On Spectral Clustering: Analysis and an Algorithm | Conf. (NIPS) | >8,000 | Media | Algoritmo practico de spectral clustering |
| 16 | Hopkins & Skellam (1954) | A New Method for Determining the Type of Distribution of Plant Individuals | Journal (Annals of Botany) | >1,000 | Media | Hopkins statistic original |
| 17 | McInnes & Healy (2017) | Accelerated Hierarchical Density Based Clustering | Preprint (arXiv) | >500 | Media | Implementacion eficiente de HDBSCAN |
| 18 | Ward (1963) | Hierarchical Grouping to Optimize an Objective Function | Journal (JASA) | >12,000 | Media | Criterio de Ward original |
| 19 | Calinski & Harabasz (1974) | A Dendrite Method for Cluster Analysis | Journal (Comm. Statistics) | >6,000 | Media | Indice Calinski-Harabasz |
| 20 | Davies & Bouldin (1979) | A Cluster Separation Measure | Journal (IEEE TPAMI) | >7,000 | Media | Indice Davies-Bouldin |
| 21 | Adzhemyan et al. (2018) | To Cluster, or Not to Cluster: An Analysis of Clusterability Methods | Preprint (arXiv) | >80 | Media | Comparacion de tests de clusterability |
| 22 | Dinh & Huynh (2021) | Clustering Refinement | Journal (IJDSA) | >30 | Media | Framework de refinamiento post-clustering |
| 23 | Anderson & Schutz (2023) | Understanding Feature Importance in Musical Works | Journal (Music Perception) | Reciente | Media | Importancia de features en clustering musical |
| 24 | Lim & Jatowt (2022) | The Performance of BERT as Data Representation of Text Clustering | Journal (J. Big Data) | >50 | Media | BERT para clustering textual |
| 25 | Gan & Ng (2017) | Clustering with Outlier Removal | Journal (IEEE TKDE) | >100 | Media | COR: clustering conjunto con outlier removal |

---

## 5. Gaps Identificados y Oportunidades

### 5.1 Gap Principal: Framework Formal de Purificacion Post-Clustering

La revision revela que la purificacion post-clustering se practica extensamente pero carece de formalizacion. Los trabajos existentes (Dinh & Huynh, 2021; Gan & Ng, 2017) abordan aspectos parciales:
- COR integra outlier removal con clustering pero no es un paso de refinamiento post-hoc.
- Los metodos basados en Z-score son ad-hoc y no estan validados formalmente.
- No existe un marco que integre purificacion con evaluacion (como medir si la purificacion realmente mejora la calidad).

**Oportunidad para el proyecto:** Formalizar un procedimiento de purificacion hibrida que:
1. Defina umbrales de eliminacion basados en estadisticas intra-cluster (percentiles de distancia al centroide).
2. Valide la mejora mediante comparacion pre/post en Silhouette y Davies-Bouldin.
3. Documente el trade-off entre pureza y cobertura (porcentaje de datos eliminados vs mejora en metricas).

### 5.2 Gap: Purificacion en Espacios Multimodales

Ningun trabajo identificado aborda la purificacion cuando se opera simultaneamente sobre multiples representaciones (semantica + musical). Las preguntas abiertas incluyen:
- Si un punto es outlier en el espacio semantico pero no en el musical (o viceversa), que hacer?
- Como combinar los criterios de purificacion de ambas modalidades?
- Se debe purificar cada modalidad independientemente o de forma conjunta?

### 5.3 Gap: Hopkins Statistic en Embeddings de Transformers

No se encontraron estudios que evaluen especificamente el comportamiento del Hopkins statistic en espacios de embeddings generados por modelos de lenguaje (BERT, GPT). Dado que estos espacios tienen propiedades geometricas particulares (estructura de cono, anisotropia), el comportamiento del Hopkins puede diferir del esperado en espacios euclidianos estandar.

### 5.4 Gap: Comparacion Sistematica de Algoritmos para Datos Musicales Multimodales

Si bien existen comparaciones de algoritmos de clustering en general, y aplicaciones individuales a MIR, no se encontro una comparacion sistematica que evalúe K-Means, Ward, DBSCAN/HDBSCAN y Spectral clustering especificamente sobre datos musicales que combinan embeddings textuales y features acusticas.

### 5.5 Conexiones con el Proyecto

| Gap identificado | Componente del proyecto | Tipo de contribucion |
|-----------------|------------------------|---------------------|
| Framework de purificacion | Purificacion hibrida post-clustering | Contribucion metodologica |
| Purificacion multimodal | Fusion semantico-musical | Contribucion original |
| Hopkins en embeddings BERT | Evaluacion de tendencia al clustering | Analisis empirico |
| Comparacion algoritmica en MIR multimodal | Evaluacion multi-algoritmo | Contribucion empirica |

---

## 6. Entradas BibTeX

```bibtex
@article{chicco_2025_silhouette_db,
  author    = {Chicco, Davide and Campagner, Andrea and Spagnolo, Andrea and Ciucci, Davide and Jurman, Giuseppe},
  title     = {The {Silhouette} Coefficient and the {Davies-Bouldin} Index Are More Informative than {Dunn} Index, {Calinski-Harabasz} Index, {Shannon} Entropy, and {Gap} Statistic for Unsupervised Clustering Internal Evaluation of Two Convex Clusters},
  journal   = {PeerJ Computer Science},
  volume    = {11},
  pages     = {e3309},
  year      = {2025},
  doi       = {10.7717/peerj-cs.3309}
}

@inproceedings{arthur_2007_kmeans_plus_plus,
  author    = {Arthur, David and Vassilvitskii, Sergei},
  title     = {k-means++: The Advantages of Careful Seeding},
  booktitle = {Proceedings of the 18th Annual {ACM-SIAM} Symposium on Discrete Algorithms ({SODA})},
  pages     = {1027--1035},
  year      = {2007},
  publisher = {SIAM},
  url       = {https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf}
}

@article{vonluxburg_2007_spectral_tutorial,
  author    = {von Luxburg, Ulrike},
  title     = {A Tutorial on Spectral Clustering},
  journal   = {Statistics and Computing},
  volume    = {17},
  number    = {4},
  pages     = {395--416},
  year      = {2007},
  doi       = {10.1007/s11222-007-9033-z}
}

@inproceedings{campello_2013_hdbscan,
  author    = {Campello, Ricardo J. G. B. and Moulavi, Davoud and Sander, J{\"o}rg},
  title     = {Density-Based Clustering Based on Hierarchical Density Estimates},
  booktitle = {Advances in Knowledge Discovery and Data Mining ({PAKDD})},
  series    = {Lecture Notes in Computer Science},
  volume    = {7819},
  pages     = {160--172},
  year      = {2013},
  publisher = {Springer},
  doi       = {10.1007/978-3-642-37456-2_14}
}

@article{mcinnes_2018_umap,
  author    = {McInnes, Leland and Healy, John and Melville, James},
  title     = {{UMAP}: {Uniform} Manifold Approximation and Projection for Dimension Reduction},
  journal   = {arXiv preprint arXiv:1802.03426},
  year      = {2018},
  url       = {https://arxiv.org/abs/1802.03426}
}

@article{rousseeuw_1987_silhouettes,
  author    = {Rousseeuw, Peter J.},
  title     = {Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis},
  journal   = {Journal of Computational and Applied Mathematics},
  volume    = {20},
  pages     = {53--65},
  year      = {1987},
  doi       = {10.1016/0377-0427(87)90125-7}
}

@article{murtagh_2014_ward,
  author    = {Murtagh, Fionn and Legendre, Pierre},
  title     = {Ward's Hierarchical Agglomerative Clustering Method: Which Algorithms Implement {Ward's} Criterion?},
  journal   = {Journal of Classification},
  volume    = {31},
  number    = {3},
  pages     = {274--295},
  year      = {2014},
  doi       = {10.1007/s00357-014-9161-z}
}

@article{vinh_2010_nmi,
  author    = {Vinh, Nguyen Xuan and Epps, Julien and Bailey, James},
  title     = {Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance},
  journal   = {Journal of Machine Learning Research},
  volume    = {11},
  pages     = {2837--2854},
  year      = {2010},
  url       = {https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf}
}

@article{tibshirani_2001_gap,
  author    = {Tibshirani, Robert and Walther, Guenther and Hastie, Trevor},
  title     = {Estimating the Number of Clusters in a Data Set Via the {Gap} Statistic},
  journal   = {Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
  volume    = {63},
  number    = {2},
  pages     = {411--423},
  year      = {2001},
  doi       = {10.1111/1467-9868.00293}
}

@inproceedings{aggarwal_2001_dimensionality,
  author    = {Aggarwal, Charu C. and Hinneburg, Alexander and Keim, Daniel A.},
  title     = {On the Surprising Behavior of Distance Metrics in High Dimensional Spaces},
  booktitle = {Database Theory --- {ICDT} 2001},
  series    = {Lecture Notes in Computer Science},
  volume    = {1973},
  pages     = {420--434},
  year      = {2001},
  publisher = {Springer},
  doi       = {10.1007/3-540-44503-X_27}
}

@inproceedings{ester_1996_dbscan,
  author    = {Ester, Martin and Kriegel, Hans-Peter and Sander, J{\"o}rg and Xu, Xiaowei},
  title     = {A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise},
  booktitle = {Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining ({KDD})},
  pages     = {226--231},
  year      = {1996},
  publisher = {AAAI Press}
}

@article{hubert_1985_ari,
  author    = {Hubert, Lawrence and Arabie, Phipps},
  title     = {Comparing Partitions},
  journal   = {Journal of Classification},
  volume    = {2},
  number    = {1},
  pages     = {193--218},
  year      = {1985},
  doi       = {10.1007/BF01908075}
}

@inproceedings{allaoui_2020_umap_clustering,
  author    = {Allaoui, Mebarka and Kherfi, Mohammed Lamine and Cheriet, Abdelhakim},
  title     = {Considerably Improving Clustering Algorithms Using {UMAP} Dimensionality Reduction Technique: A Comparative Study},
  booktitle = {Image and Signal Processing ({ICISP})},
  series    = {Lecture Notes in Computer Science},
  volume    = {12119},
  pages     = {317--325},
  year      = {2020},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-51935-3_34}
}

@article{shi_2000_normalized_cuts,
  author    = {Shi, Jianbo and Malik, Jitendra},
  title     = {Normalized Cuts and Image Segmentation},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume    = {22},
  number    = {8},
  pages     = {888--905},
  year      = {2000},
  doi       = {10.1109/34.868688}
}

@inproceedings{ng_2002_spectral,
  author    = {Ng, Andrew Y. and Jordan, Michael I. and Weiss, Yair},
  title     = {On Spectral Clustering: Analysis and an Algorithm},
  booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
  volume    = {14},
  pages     = {849--856},
  year      = {2002},
  publisher = {MIT Press}
}

@article{hopkins_1954_statistic,
  author    = {Hopkins, Brian and Skellam, John Gordon},
  title     = {A New Method for Determining the Type of Distribution of Plant Individuals},
  journal   = {Annals of Botany},
  volume    = {18},
  number    = {2},
  pages     = {213--227},
  year      = {1954},
  doi       = {10.1093/oxfordjournals.aob.a083391}
}

@article{mcinnes_2017_hdbscan_accelerated,
  author    = {McInnes, Leland and Healy, John},
  title     = {Accelerated Hierarchical Density Based Clustering},
  journal   = {arXiv preprint arXiv:1705.07321},
  year      = {2017},
  url       = {https://arxiv.org/abs/1705.07321}
}

@article{ward_1963_hierarchical,
  author    = {Ward, Joe H.},
  title     = {Hierarchical Grouping to Optimize an Objective Function},
  journal   = {Journal of the American Statistical Association},
  volume    = {58},
  number    = {301},
  pages     = {236--244},
  year      = {1963},
  doi       = {10.1080/01621459.1963.10500845}
}

@article{calinski_1974_dendrite,
  author    = {Cali{\'n}ski, Tadeusz and Harabasz, Jerzy},
  title     = {A Dendrite Method for Cluster Analysis},
  journal   = {Communications in Statistics --- Theory and Methods},
  volume    = {3},
  number    = {1},
  pages     = {1--27},
  year      = {1974},
  doi       = {10.1080/03610927408827101}
}

@article{davies_1979_cluster_separation,
  author    = {Davies, David L. and Bouldin, Donald W.},
  title     = {A Cluster Separation Measure},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume    = {PAMI-1},
  number    = {2},
  pages     = {224--227},
  year      = {1979},
  doi       = {10.1109/TPAMI.1979.4766909}
}

@article{adzhemyan_2018_clusterability,
  author    = {Ackerman, Margareta and Ben-David, Shai},
  title     = {To Cluster, or Not to Cluster: An Analysis of Clusterability Methods},
  journal   = {arXiv preprint arXiv:1808.08317},
  year      = {2018},
  url       = {https://arxiv.org/abs/1808.08317}
}

@article{dinh_2021_clustering_refinement,
  author    = {Dinh, Duc-Trong and Huynh, Van-Nam},
  title     = {Clustering Refinement},
  journal   = {International Journal of Data Science and Analytics},
  volume    = {12},
  pages     = {45--57},
  year      = {2021},
  doi       = {10.1007/s41060-021-00275-z}
}

@article{anderson_2023_feature_importance_music,
  author    = {Anderson, Cameron J. and Schutz, Michael},
  title     = {Understanding Feature Importance in Musical Works: Unpacking Predictive Contributions to Cluster Analyses},
  journal   = {Music Perception},
  volume    = {41},
  number    = {2},
  pages     = {108--126},
  year      = {2023},
  doi       = {10.1177/20592043231216257}
}

@article{lim_2022_bert_text_clustering,
  author    = {Lim, Stephen and Jatowt, Adam},
  title     = {The Performance of {BERT} as Data Representation of Text Clustering},
  journal   = {Journal of Big Data},
  volume    = {9},
  number    = {15},
  year      = {2022},
  doi       = {10.1186/s40537-022-00564-9}
}

@article{gan_2017_cor,
  author    = {Gan, Guojun and Ng, Michael K.-P.},
  title     = {Clustering with Outlier Removal},
  journal   = {IEEE Transactions on Knowledge and Data Engineering},
  volume    = {29},
  number    = {8},
  pages     = {1497--1510},
  year      = {2017},
  doi       = {10.1109/TKDE.2017.2691871}
}

@article{macqueen_1967_kmeans,
  author    = {MacQueen, James B.},
  title     = {Some Methods for Classification and Analysis of Multivariate Observations},
  booktitle = {Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability},
  volume    = {1},
  pages     = {281--297},
  year      = {1967},
  publisher = {University of California Press}
}

@article{lloyd_1982_kmeans,
  author    = {Lloyd, Stuart P.},
  title     = {Least Squares Quantization in {PCM}},
  journal   = {IEEE Transactions on Information Theory},
  volume    = {28},
  number    = {2},
  pages     = {129--137},
  year      = {1982},
  doi       = {10.1109/TIT.1982.1056489}
}

@article{schedl_2014_mir,
  author    = {Schedl, Markus and G{\'o}mez, Emilia and Urbano, Juli{\'a}n},
  title     = {Music Information Retrieval: Recent Developments and Applications},
  journal   = {Foundations and Trends in Information Retrieval},
  volume    = {8},
  number    = {2--3},
  pages     = {127--261},
  year      = {2014},
  doi       = {10.1561/1500000042}
}

@article{mayer_2008_music_clustering_features,
  author    = {Mayer, Rudolf and Neumayer, Robert and Rauber, Andreas},
  title     = {Music Clustering With Features From Different Information Sources},
  journal   = {IEEE Transactions on Multimedia},
  volume    = {10},
  number    = {8},
  year      = {2008},
  doi       = {10.1109/TMM.2008.2007276}
}
```

---

*Documento generado mediante revision sistematica de literatura. Todas las fuentes fueron identificadas a traves de busqueda sistematica en bases de datos academicas. Las entradas marcadas como [no verificado] requieren confirmacion de metadatos mediante acceso directo al documento. Ultima actualizacion: 2026-02-07.*
