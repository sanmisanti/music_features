# Evaluacion Experimental y Metodologia Estadistica

## Investigacion sistematica para tesis de Ingenieria Informatica
**Fecha de elaboracion**: 2026-02-08
**Contexto**: Sistema de recomendacion musical hibrido con clustering multi-modal

---

## 1. Resumen ejecutivo

La evaluacion experimental en sistemas de aprendizaje automatico y recomendacion musical constituye un campo con una base teorica madura pero con deficiencias sistematicas en su aplicacion practica. La revision de la literatura revela tres pilares fundamentales: (1) la comparacion estadistica rigurosa de algoritmos, dominada por el marco no parametrico de Demsar (2006) y su extension por Garcia y Herrera (2008), complementado recientemente por alternativas bayesianas propuestas por Benavoli et al. (2017); (2) la crisis de reproducibilidad documentada empiricamente por Gundersen y Kjensmo (2018), Pineau et al. (2021) y Kapoor y Narayanan (2023), quienes identificaron data leakage en 294 publicaciones; y (3) los marcos de evaluacion especificos para sistemas de recomendacion, particularmente el framework FEVR de Zangerle y Bauer (2022) y las metricas beyond-accuracy. Se identificaron mas de 25 fuentes relevantes publicadas entre 2002 y 2024, con alta concentracion en venues de primer nivel (JMLR, ACM Computing Surveys, IEEE TMM, NeurIPS). Los gaps mas relevantes para el proyecto incluyen la ausencia de marcos integrados que combinen validacion de clustering sin ground truth con evaluacion de recomendacion offline, y la escasa aplicacion de tests de significancia en estudios de MIR con clustering multi-modal.

---

## 2. Estrategia de busqueda

### 2.1 Palabras clave utilizadas

**Ingles (busqueda principal)**:
- "statistical comparison classifiers Friedman Nemenyi"
- "Bayesian analysis comparison classifiers JMLR"
- "reproducibility machine learning experiments"
- "data leakage reproducibility crisis"
- "clustering evaluation without ground truth stability"
- "clustering validation internal external indices"
- "recommender system evaluation framework metrics"
- "beyond accuracy metrics recommender systems"
- "popularity bias fairness music recommendation"
- "multi-objective optimization NSGA-II clustering"
- "pre-registration hypothesis machine learning"
- "horse effect music information retrieval"

**Espanol (busqueda complementaria)**:
- "evaluacion estadistica algoritmos clustering"
- "reproducibilidad experimentos aprendizaje automatico"

### 2.2 Fuentes consultadas y resultados relevantes

| Fuente | Busquedas realizadas | Resultados relevantes |
|--------|---------------------|----------------------|
| Google Scholar (via WebSearch) | 4 | 12 fuentes primarias |
| Semantic Scholar | 3 | 8 fuentes con grafos de citas |
| ACM Digital Library | 2 | 5 fuentes (ACM CSUR, RecSys) |
| IEEE Xplore | 1 | 2 fuentes (IEEE TMM) |
| JMLR directo | 3 | 4 fuentes seminales |
| Wiley Online Library | 2 | 3 fuentes (WIREs) |
| Frontiers | 1 | 2 fuentes |
| arXiv | 2 | 3 preprints relevantes |

**Total de busquedas ejecutadas**: 14
**Total de fuentes relevantes identificadas**: 25+

### 2.3 Criterios de inclusion y exclusion

**Inclusion**:
- Publicaciones revisadas por pares en journals o conferencias indexadas (2002-2026).
- Preprints de alto impacto con mas de 50 citas o de autores reconocidos.
- Trabajos seminales anteriores a 2015 cuando son fundamentales para el campo (Demsar 2006, Deb 2002).
- Surveys y revisiones sistematicas con cobertura comprehensiva.

**Exclusion**:
- Blogs, tutoriales no academicos y fuentes sin autor identificable.
- Publicaciones en revistas sin indexacion reconocida.
- Trabajos duplicados o versiones previas de articulos ya incluidos en su version final.

---

## 3. Estado de la cuestion

### 3.1 Tests de significancia estadistica para comparacion de algoritmos

#### 3.1.1 El marco frecuentista clasico: Demsar (2006)

El trabajo seminal de Demsar (2006), publicado en el Journal of Machine Learning Research, establecio el marco de referencia para la comparacion estadistica de clasificadores sobre multiples conjuntos de datos. Con mas de 15,000 citas, este articulo transformo la practica experimental en machine learning al demostrar que los tests parametricos comunmente utilizados (t-test pareado, ANOVA) violaban supuestos fundamentales cuando se aplicaban a comparaciones de algoritmos.

Demsar propuso un protocolo de dos niveles:

1. **Comparacion de dos algoritmos**: El test de rangos con signo de Wilcoxon, que no asume normalidad en la distribucion de diferencias de rendimiento y es robusto ante outliers.

2. **Comparacion de multiples algoritmos**: El test de Friedman como omnibus test, seguido del test post-hoc de Nemenyi para comparaciones por pares. El test de Friedman opera sobre rangos de rendimiento a traves de datasets, evaluando la hipotesis nula de que todos los algoritmos tienen rendimiento equivalente. Cuando se rechaza, el test de Nemenyi identifica que pares de algoritmos difieren significativamente, utilizando la diferencia critica (CD) como umbral.

La contribucion metodologica central fue demostrar que, en la comparacion de k algoritmos sobre N datasets, los tests no parametricos basados en rangos son preferibles porque: (a) las diferencias de rendimiento rara vez siguen distribuciones normales, (b) los rangos son mas robustos ante escalas heterogeneas entre datasets, y (c) el test de Friedman mantiene un control adecuado del error de tipo I incluso con muestras pequenas.

**Relevancia para el proyecto**: La comparacion de 4 algoritmos de clustering (K-Means, DBSCAN, Agglomerative, Spectral) con multiples configuraciones de K sobre el dataset de 18,454 canciones constituye exactamente el escenario para el cual Demsar diseno este marco. El test de Friedman con post-hoc de Nemenyi permitira determinar si las diferencias observadas en Silhouette, Davies-Bouldin y Calinski-Harabasz son estadisticamente significativas.

#### 3.1.2 Extension para comparaciones por pares: Garcia y Herrera (2008)

Garcia y Herrera (2008) extendieron el trabajo de Demsar proponiendo procedimientos post-hoc mas potentes para comparaciones nxn (todos contra todos). Mientras que Nemenyi es conservador por controlar el family-wise error rate (FWER) de manera estricta, Garcia y Herrera demostraron que tests como Holm, Shaffer y Bergmann-Hommel ofrecen mayor potencia estadistica sin sacrificar el control del error de tipo I.

Las contribuciones especificas incluyen:

- **Procedimiento de Holm (1979)**: Ajuste secuencial step-down del p-valor que rechaza hipotesis ordenadas por significancia, ofreciendo mayor potencia que Bonferroni.
- **Procedimiento de Shaffer**: Explota las relaciones logicas entre hipotesis para permitir mas rechazos.
- **Procedimiento de Bergmann-Hommel**: El mas potente pero computacionalmente costoso, explota exhaustivamente las relaciones entre hipotesis.

Garcia y Herrera recomendaron el procedimiento de Holm como balance optimo entre potencia y simplicidad, y Bergmann-Hommel cuando el numero de algoritmos es manejable (k <= 5).

**Relevancia para el proyecto**: Con 4 algoritmos de clustering, el numero de comparaciones por pares es C(4,2) = 6, lo que hace viable el uso de Bergmann-Hommel. Sin embargo, considerando multiples valores de K y multiples seeds, el procedimiento de Holm ofrece un equilibrio practico.

#### 3.1.3 Tests para comparacion pareada: Dietterich (1998)

Dietterich (1998) abordo un problema anterior y complementario: la comparacion de dos algoritmos sobre un unico dataset. Su analisis revelo que el test t pareado convencional (basado en 10-fold cross-validation) tiene una probabilidad de error de tipo I inaceptablemente elevada debido a la dependencia entre los folds.

Propuso el test 5x2cv paired t-test, que realiza 5 repeticiones de 2-fold cross-validation, produciendo 10 estimaciones independientes de la diferencia de rendimiento. Este test mantiene el error de tipo I cercano al nivel nominal (0.05) y ofrece potencia razonable para detectar diferencias genuinas.

Alpaydin (1999) extendio este trabajo proponiendo el 5x2cv F-test (combined 5x2cv F-test), que utiliza un estadistico F con 10 y 5 grados de libertad, ofreciendo mayor potencia que el test de Dietterich manteniendo el control del error de tipo I.

**Relevancia para el proyecto**: Para comparaciones directas entre el componente semantico (BERT 384D) y el componente musical (Spotify 12D) dentro de un mismo algoritmo de clustering, el test 5x2cv resulta apropiado dado que opera sobre un unico dataset. No obstante, la adaptacion a clustering (unsupervised) requiere sustituir la metrica de accuracy por indices internos de validacion.

#### 3.1.4 Alternativas bayesianas: Benavoli et al. (2017)

Benavoli, Corani, Demsar y Zaffalon (2017) propusieron un cambio de paradigma: abandonar los tests de hipotesis nula (NHST) en favor de analisis bayesianos. Su articulo en JMLR argumenta que los NHST presentan falacias fundamentales, entre ellas:

- La probabilidad de rechazar H0 depende del tamano muestral, no del tamano del efecto.
- La dicotomia significativo/no-significativo es artificial y no informa sobre la magnitud practica de las diferencias.
- El p-valor no es la probabilidad de que H0 sea verdadera dado los datos.

Como alternativa, propusieron tests bayesianos que producen tres probabilidades posteriores: P(algoritmo A > B), P(algoritmo B > A) y P(equivalencia practica), donde la equivalencia se define mediante una "region de equivalencia practica" (ROPE). Este enfoque permite distinguir entre "no hay diferencia" y "no tenemos suficiente evidencia para detectar la diferencia".

Para la comparacion de multiples clasificadores, propusieron el test bayesiano de rangos con signo y la version bayesiana del test de Friedman, implementados en la libreria Python `baycomp`.

**Relevancia para el proyecto**: El enfoque bayesiano es particularmente valioso para el proyecto porque: (1) con 4 algoritmos y multiples configuraciones, poder cuantificar la probabilidad de equivalencia practica es mas informativo que un simple rechazo/no-rechazo; (2) la ROPE permite definir que significa una diferencia "practica" en Silhouette (por ejemplo, delta < 0.02 como equivalente); (3) reportar distribuciones posteriores complementa los diagramas CD de Demsar con informacion sobre incertidumbre.

### 3.2 Reproducibilidad en Machine Learning

#### 3.2.1 Diagnostico del estado actual

La reproducibilidad en investigacion basada en machine learning ha sido diagnosticada como deficiente por multiples estudios independientes. Gundersen y Kjensmo (2018) analizaron 400 publicaciones de AAAI e IJCAI y encontraron que solo el 25% de las variables necesarias para reproduccion estaban adecuadamente documentadas. Mas alarmante, solo el 6% de los trabajos declaraban explicitamente las preguntas de investigacion, y apenas el 5% formulaban hipotesis testables.

Su marco taxonomico establece tres niveles de reproducibilidad:
- **R1 (Experiment Reproducible)**: El experimento puede reproducirse con el mismo metodo y datos.
- **R2 (Data Reproducible)**: Los resultados se pueden obtener con los mismos datos pero diferente implementacion.
- **R3 (Method Reproducible)**: El metodo se puede aplicar a nuevos datos produciendo conclusiones consistentes.

La documentacion se organiza en tres factores: metodo (especificacion del algoritmo y problema), datos (descripcion de los datos utilizados), y experimento (como se condujo el experimento).

#### 3.2.2 El Reproducibility Checklist de NeurIPS

Pineau et al. (2021) documentaron el programa de reproducibilidad implementado en NeurIPS desde 2019, que incluye tres componentes: (1) politica de envio de codigo, (2) challenge de reproducibilidad comunitario, y (3) checklist de reproducibilidad como requisito de envio.

El checklist exige documentar:
- Descripcion matematica completa del modelo/algoritmo.
- Supuestos explicitados y justificados.
- Analisis de complejidad temporal y espacial.
- Detalles de splits train/validation/test.
- Explicacion de datos excluidos y pasos de preprocesamiento.
- Enlace a codigo y datos descargables.
- Seeds aleatorias utilizadas.
- Numero de ejecuciones y variabilidad reportada.

El impacto fue significativo: la disponibilidad publica de codigo aumento sustancialmente tras la adopcion del checklist en ICML y NeurIPS.

#### 3.2.3 Data leakage como amenaza sistematica

Kapoor y Narayanan (2023) condujeron la investigacion mas comprensiva sobre data leakage en ciencia basada en ML, publicada en Patterns (Cell Press). Identificaron 294 publicaciones afectadas por data leakage en 17 campos cientificos, produciendo conclusiones "desmedidamente optimistas" en todos los casos.

Propusieron una taxonomia de 8 tipos de leakage:
1. **No independence between training and test**: Duplicados o datos relacionados en ambos splits.
2. **Pre-processing on entire dataset**: Normalizacion, feature selection o PCA antes de separar train/test.
3. **Feature leakage**: Variables que codifican informacion del target no disponible en produccion.
4. **Temporal leakage**: Uso de datos futuros para predecir eventos pasados.
5. **Non-independence in resampling**: Correlacion entre muestras en cross-validation.
6. **Sampling bias not accounted for**: Sesgo de seleccion no corregido.
7. **Leakage from test set reuse**: Optimizacion iterativa sobre el test set.
8. **Benchmark contamination**: Inclusion de datos de benchmark en pre-entrenamiento.

Como mitigacion, propusieron "model info sheets": documentos estandarizados que describen las decisiones de pipeline, fuentes de datos y procedimientos de validacion.

Un caso de estudio emblematico fue la prediccion de guerras civiles, donde modelos ML supuestamente superiores a modelos estadisticos tradicionales perdieron su ventaja cuando se corrigio el data leakage.

**Relevancia para el proyecto**: El pipeline del proyecto presenta riesgos especificos de leakage: (1) la vectorizacion BERT se aplica a todas las letras antes de la separacion en clusters, lo cual es correcto para clustering pero debe documentarse explicitamente; (2) la seleccion de K optimo usando metricas internas sobre todo el dataset es inherente al clustering pero debe distinguirse de evaluacion predictiva; (3) el uso de genero como ground truth proxy requiere verificar que la informacion de genero no influya en las features de entrada.

#### 3.2.4 Pre-registro de hipotesis en ML

La practica de pre-registro, bien establecida en ciencias sociales y medicina, esta siendo adoptada gradualmente en ML. El NeurIPS 2021 Pre-Registration Workshop represento un hito, proponiendo que investigadores documenten sus hipotesis, protocolos experimentales y criterios de decision antes de ejecutar experimentos.

Los beneficios documentados incluyen:
- Separacion explicita entre analisis exploratorio y confirmatorio.
- Reduccion de HARKing (Hypothesizing After Results are Known).
- Prevencion de p-hacking y seleccion selectiva de resultados.
- Mayor credibilidad de los hallazgos reportados.

Para predictive modeling, se ha propuesto un template de pre-registro adaptado que incluye: definicion del problema, hipotesis especificas, metricas de evaluacion predefinidas, protocolo de splitting, y criterios de decision para aceptar/rechazar hipotesis.

**Relevancia para el proyecto**: En la primera ejecucion, las hipotesis se formularon post-hoc, lo que constituye una debilidad metodologica grave. La re-ejecucion debe pre-registrar: (H1) que el clustering multi-modal produce agrupaciones mas coherentes que los componentes individuales, (H2) que la fusion hibrida supera los componentes individuales en Precision@K, y (H3) que la optimizacion de pesos mejora significativamente sobre la ponderacion uniforme.

### 3.3 Evaluacion de clustering sin ground truth

#### 3.3.1 Indices de validacion interna

La evaluacion de clustering en ausencia de etiquetas verdaderas constituye un problema fundamental en aprendizaje no supervisado. La literatura distingue entre indices geometricos (basados en distancias) y no geometricos (basados en distribucion de variables).

Los indices internos mas utilizados y sus propiedades:

- **Silhouette (Rousseeuw, 1987)**: Mide la cohesion intra-cluster y separacion inter-cluster en rango [-1, 1]. Robusto para clusters convexos pero sesgado contra formas arbitrarias.
- **Davies-Bouldin (1979)**: Ratio de dispersion intra-cluster a separacion inter-cluster. Valores menores son preferibles. Sensible a clusters de tamano desigual.
- **Calinski-Harabasz (1974)**: Ratio de varianza entre-clusters a varianza intra-cluster. Favorece clusters compactos y bien separados.
- **Hopkins statistic**: Mide la tendencia de clustering (clusterability) de los datos. Valores > 0.7 indican estructura de clusters.

Un estudio comprehensivo de 2024 comparo 27 CVIs (17 geometricos y 10 no geometricos) en datos binarios, revelando que no existe un indice universalmente superior y que la seleccion debe considerar las caracteristicas del dataset y el algoritmo de clustering.

#### 3.3.2 Estabilidad como criterio de validacion

Liu et al. (2022) publicaron una revision comprehensiva sobre estimacion de estabilidad en clustering en WIREs Computational Statistics. El concepto central es que un clustering valido debe ser estable ante perturbaciones de los datos: si se producen datasets perturbados cercanos al original y se aplica el mismo algoritmo, los clusters resultantes deberian ser similares.

Los enfoques principales incluyen:

- **Bootstrap stability**: Remuestreo con reemplazo y comparacion de clusterings. Liu et al. identificaron que el bootstrap puede sobreajustar en analogia con la mala separacion train/test en aprendizaje supervisado.
- **Out-of-bag (OOB) stability**: Los items no seleccionados en cada bootstrap sirven como conjunto de validacion, evitando el sesgo del bootstrap clasico.
- **Subsampling stability**: Multiples submuestras sin reemplazo, comparando clusterings via indices externos (Adjusted Rand Index, NMI).
- **Cross-validation para clustering**: Adaptacion de k-fold CV donde se entrena el clustering en k-1 folds y se evalua la asignacion en el fold restante.

**Relevancia para el proyecto**: Con multiples seeds (configuracion del proyecto), la estabilidad se puede evaluar comparando clusterings producidos por diferentes inicializaciones de K-Means o diferentes muestras de DBSCAN. El ARI entre ejecuciones es un indicador directo de estabilidad.

#### 3.3.3 Validacion sistematica: el marco de Ullmann (2022)

Ullmann (2022) publico en WIREs Data Mining and Knowledge Discovery un marco sistematico para validacion de clustering que integra cuatro aproximaciones:

1. **Comparacion con etiquetas verdaderas** (cuando existen): ARI, NMI, V-measure.
2. **Indices internos y externos**: Silhouette, Davies-Bouldin, Calinski-Harabasz.
3. **Analisis de estabilidad**: Bootstrap, subsampling, perturbaciones.
4. **Validacion visual**: t-SNE, UMAP para inspeccion humana de la estructura.

El marco propone una separacion explicita entre "discovery data" (usada para seleccionar el metodo de clustering) y "validation data" (usada para validar los resultados), en analogia con la separacion train/test de aprendizaje supervisado.

**Relevancia para el proyecto**: Este marco es directamente aplicable: (1) usar indices internos (Silhouette, Hopkins) para seleccionar K y algoritmo; (2) validar con estabilidad (multiples seeds); (3) usar genero como ground truth proxy con ARI/NMI, declarando explicitamente su naturaleza aproximada; (4) visualizar con UMAP/t-SNE para validacion cualitativa.

#### 3.3.4 El problema del ground truth proxy

El uso de genero musical como ground truth para evaluar clustering es una practica comun pero problematica en MIR. La literatura reconoce varias limitaciones:

- **Subjetividad**: Las etiquetas de genero son asignaciones culturales y subjetivas, no categorias naturales. Dos evaluadores humanos coinciden en genero en aproximadamente 70-80% de los casos.
- **Granularidad variable**: Un artista puede pertenecer a multiples generos, y los generos tienen jerarquias (rock -> alternative rock -> indie rock).
- **Dependencia temporal**: Los generos evolucionan y las fronteras cambian con el tiempo.
- **No exhaustividad**: El genero captura solo una dimension de la similitud musical; timbre, ritmo, emocion y estructura son dimensiones ortogonales.

La recomendacion de la literatura es tratar el genero como una "aproximacion informativa pero incompleta" y reportar multiples metricas complementarias, declarando explicitamente las limitaciones del proxy.

### 3.4 Evaluacion de sistemas de recomendacion

#### 3.4.1 El framework FEVR: evaluacion sistematica

Zangerle y Bauer (2022) publicaron en ACM Computing Surveys el Framework for EValuating Recommender systems (FEVR), que sistematiza las decisiones de diseno necesarias para una evaluacion comprehensiva. FEVR organiza la evaluacion en tres bloques:

1. **Principios de evaluacion**: Definicion de hipotesis, seleccion de metricas, consideracion de generalizabilidad.
2. **Metodos de evaluacion**: Offline (historicos), online (A/B testing), user studies.
3. **Diseno experimental**: Datasets, splits, baselines, condiciones de comparacion.

El framework enfatiza que la evaluacion debe ser guiada por hipotesis predefinidas, no por la exploracion post-hoc de resultados. Ademas, destaca la importancia de la repetibilidad (mismo equipo, mismo setup) y reproducibilidad (diferente equipo, mismo metodo).

**Relevancia para el proyecto**: FEVR proporciona una estructura para organizar la evaluacion del sistema hibrido: definir hipotesis pre-registradas, seleccionar metricas (Precision@K, nDCG, diversidad), disenar el protocolo de evaluacion offline, y documentar todas las decisiones de diseno.

#### 3.4.2 Evaluacion offline: desafios y direcciones

Castells y Moffat (2022) publicaron en AI Magazine un analisis de los desafios de la evaluacion offline de sistemas de recomendacion. Los hallazgos principales:

- **Correlacion debil offline-online**: Los resultados de evaluacion offline frecuentemente no correlacionan con la satisfaccion de usuarios en evaluacion online.
- **Sensibilidad al protocolo**: Diferentes configuraciones de evaluacion offline (splits temporales vs aleatorios, metricas, K en top-K) pueden producir rankings de sistemas contradictories.
- **Bias de seleccion**: Los datos historicos estan sesgados por el sistema de recomendacion que los genero (feedback loop).

Los autores recomiendan: (1) reportar multiples metricas complementarias, (2) usar splits temporales cuando sea posible, (3) documentar explicitamente las limitaciones de la evaluacion offline, (4) considerar el evaluation as information retrieval, adaptando metricas IR con cuidado.

**Relevancia para el proyecto**: La evaluacion del sistema de recomendacion musical es exclusivamente offline (sin interacciones de usuario reales), lo que requiere documentar explicitamente esta limitacion. El uso de genero como proxy de relevancia agrega una capa adicional de aproximacion que debe ser transparente.

#### 3.4.3 Metricas beyond-accuracy

La investigacion reciente ha expandido significativamente el espectro de metricas mas alla de la accuracy/precision tradicional. Kaminskas y Bridge (2016) publicaron un survey seminal en ACM TIIS identificando cuatro dimensiones:

- **Diversidad**: Mide la variedad intra-lista de recomendaciones. Se operacionaliza como 1 - similitud media entre pares de items recomendados. Lista recomendaciones excesivamente similares son insatisfactorias (efecto "filter bubble").
- **Novedad**: Cuantifica que tan nuevos o desconocidos son los items recomendados para el usuario. Relacionada inversamente con la popularidad del item.
- **Serendipity**: Combina relevancia con sorpresa. Un item serendipitoso es tanto relevante como inesperado. Mas dificil de operacionalizar que diversidad y novedad.
- **Cobertura (Coverage)**: Catalogo coverage mide la fraccion de items que el sistema recomienda al menos una vez; user coverage mide la fraccion de usuarios para quienes el sistema puede generar recomendaciones.

Un estudio de 2024 encontro que novedad y serendipity tienen impacto positivo en engagement del usuario, mientras que mayor diversidad puede perjudicar el engagement en ciertos contextos, revelando tensiones entre diferentes dimensiones de calidad.

**Relevancia para el proyecto**: La evaluacion del sistema hibrido debe incluir al menos diversidad y cobertura ademas de Precision@K. La diversidad es especialmente relevante porque el clustering multi-modal podria producir recomendaciones excesivamente homogeneas dentro de un cluster.

### 3.5 Fairness y bias en recomendacion musical

#### 3.5.1 Popularity bias

Klimashevskaia et al. (2024) publicaron en User Modeling and User-Adapted Interaction un survey comprehensivo sobre popularity bias en sistemas de recomendacion. El popularity bias se manifiesta cuando los algoritmos sobre-recomiendan items populares en detrimento de items de cola larga (long tail), amplificando desigualdades preexistentes.

El survey reviso publicaciones entre 2000 y 2024 que abordan "popularity bias" en sistemas de recomendacion, identificando que:

- Los algoritmos de filtrado colaborativo son particularmente susceptibles al popularity bias.
- Los enfoques content-based son menos afectados pero no inmunes.
- La reduccion de popularity bias frecuentemente se equipara con aumento de fairness, pero esta equivalencia es cuestionable.
- Las metricas de evaluacion convencionales (precision, recall) pueden ocultar el popularity bias al favorecer items populares que los usuarios ya conocen.

#### 3.5.2 Fairness multi-stakeholder en musica

Dinnissen y Bauer (2022) publicaron en Frontiers in Big Data una revision sobre fairness en sistemas de recomendacion musical desde la perspectiva de multiples stakeholders:

- **Usuarios/listeners**: Justicia en la exposicion a diversidad de generos, artistas y culturas.
- **Artistas/creadores**: Justicia en la visibilidad y oportunidades de descubrimiento.
- **Plataformas**: Balance entre satisfaccion del usuario y objetivos comerciales.

La revision encontro que la gran mayoria de los trabajos analizan el estado actual de fairness pero pocos proponen soluciones concretas, identificando un gap significativo entre diagnostico y accion.

**Relevancia para el proyecto**: El dataset de 18,454 canciones tiene distribuciones desiguales de generos y popularidad. La evaluacion debe medir si el sistema hibrido exhibe popularity bias y si las recomendaciones cubren adecuadamente la cola larga del catalogo.

### 3.6 El efecto "Horse" en MIR

Sturm (2014) publico en IEEE Transactions on Multimedia un articulo influyente que introdujo el concepto de "horse" en Music Information Retrieval, inspirado en el caballo "Clever Hans" que aparentaba resolver problemas matematicos pero en realidad respondia a senales involuntarias de su entrenador.

Un sistema MIR es un "horse" cuando aparenta resolver una tarea (e.g., reconocimiento de genero) pero en realidad explota factores irrelevantes confundidos con las etiquetas del dataset (e.g., condiciones de grabacion, duracion de los clips, calidad de audio). Sturm demostro mediante experimentos controlados que tres sistemas state-of-the-art de reconocimiento de genero y emocion musical eran "horses".

Las implicaciones son profundas:
- Las metricas de evaluacion estandar (accuracy, precision) no detectan el efecto horse.
- Se requieren "ablation tests" y experimentos controlados para verificar que el sistema utiliza las senales correctas.
- Los datasets de benchmark pueden contener confounders sistematicos que invalidan las evaluaciones.

**Relevancia para el proyecto**: Dos riesgos de "horse" existen en el proyecto: (1) la vectorizacion BERT podria capturar patrones superficiales de las letras (longitud, vocabulario) en lugar de semantica; (2) las features musicales de Spotify podrian correlacionar con metadatos (ano de publicacion, popularidad) mas que con contenido musical. Experimentos de ablacion son necesarios para verificar que cada componente contribuye por las razones correctas.

### 3.7 Optimizacion multi-objetivo: NSGA-II

Deb et al. (2002) publicaron el algoritmo NSGA-II (Non-dominated Sorting Genetic Algorithm II), que se ha convertido en el estandar de facto para optimizacion multi-objetivo en ML. Con complejidad O(MN^2) donde M es el numero de objetivos y N el tamano de la poblacion, NSGA-II utiliza:

- **Non-dominated sorting**: Clasificacion de soluciones en frentes de Pareto.
- **Crowding distance**: Mantenimiento de diversidad en el frente de Pareto.
- **Elitismo**: Preservacion de las mejores soluciones entre generaciones.

En el contexto de ML y clustering, NSGA-II se ha aplicado para:
- Optimizacion simultanea de multiples metricas de clustering.
- Seleccion de features y hyperparametros con objetivos en conflicto (accuracy vs complejidad).
- Busqueda de pesos optimos en sistemas hibridos con multiples componentes.

**Relevancia para el proyecto**: La optimizacion de pesos de fusion entre el componente semantico y musical es inherentemente un problema multi-objetivo: maximizar precision vs diversidad vs cobertura. NSGA-II podria explorar el frente de Pareto de pesos, en lugar de la busqueda grid utilizada en v1 que produjo pesos suboptimos (55/45 vs 20/80).

### 3.8 Tamanos de efecto e intervalos de confianza

La comunidad de ML ha adoptado progresivamente la recomendacion de reportar tamanos de efecto e intervalos de confianza ademas de (o en lugar de) p-valores. Los metodos bootstrap son particularmente apropiados para metricas de ML porque:

- No asumen distribucion normal de las metricas.
- Son aplicables a cualquier estadistico (Silhouette, Precision@K, NMI).
- El bootstrap BCa (bias-corrected and accelerated) corrige sesgo y asimetria.

Para clustering, los intervalos de confianza se pueden construir sobre indices internos (Silhouette) mediante bootstrap de observaciones, y sobre indices externos (ARI, NMI) mediante bootstrap de asignaciones.

**Relevancia para el proyecto**: La primera ejecucion no reporto intervalos de confianza para ninguna metrica. La re-ejecucion debe incluir: IC 95% via bootstrap para todas las metricas principales, con al menos 1000 repeticiones de bootstrap.

---

## 4. Tabla de fuentes principales

| # | Autores (Ano) | Titulo | Tipo | Citas aprox. | Relevancia | Aporte clave |
|---|---------------|--------|------|-------------|------------|---------------|
| 1 | Demsar (2006) | Statistical Comparisons of Classifiers over Multiple Data Sets | Journal (JMLR) | 15,000+ | Alta | Marco Friedman + Nemenyi para comparacion de multiples algoritmos |
| 2 | Garcia y Herrera (2008) | An Extension on "Statistical Comparisons of Classifiers over Multiple Data Sets" for all Pairwise Comparisons | Journal (JMLR) | 5,000+ | Alta | Tests post-hoc mas potentes: Holm, Shaffer, Bergmann-Hommel |
| 3 | Benavoli et al. (2017) | Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis | Journal (JMLR) | 1,500+ | Alta | Alternativa bayesiana a NHST con ROPE y distribuciones posteriores |
| 4 | Dietterich (1998) | Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms | Journal (Neural Computation) | 8,000+ | Alta | Test 5x2cv para comparacion pareada con control de tipo I |
| 5 | Kapoor y Narayanan (2023) | Leakage and the Reproducibility Crisis in ML-based Science | Journal (Patterns) | 500+ | Alta | Taxonomia de 8 tipos de data leakage; 294 papers afectados |
| 6 | Pineau et al. (2021) | Improving Reproducibility in Machine Learning Research | Journal (JMLR) | 1,000+ | Alta | NeurIPS Reproducibility Checklist y programa de reproducibilidad |
| 7 | Gundersen y Kjensmo (2018) | State of the Art: Reproducibility in Artificial Intelligence | Conferencia (AAAI) | 800+ | Alta | Solo 25% de variables documentadas; 3 niveles de reproducibilidad |
| 8 | Zangerle y Bauer (2022) | Evaluating Recommender Systems: Survey and Framework | Journal (ACM CSUR) | 200+ | Alta | Framework FEVR para evaluacion sistematica de RS |
| 9 | Castells y Moffat (2022) | Offline Recommender System Evaluation: Challenges and New Directions | Journal (AI Magazine) | 100+ | Alta | Desafios de evaluacion offline; correlacion debil offline-online |
| 10 | Liu et al. (2022) | Stability Estimation for Unsupervised Clustering: A Review | Journal (WIREs Comp Stats) | 100+ | Alta | Revision comprensiva de metodos de estabilidad en clustering |
| 11 | Ullmann (2022) | Validation of Cluster Analysis Results on Validation Data: A Systematic Framework | Journal (WIREs DMKD) | 50+ | Alta | Marco 4-niveles: indices, estabilidad, external, visual |
| 12 | Sturm (2014) | A Simple Method to Determine if a MIR System is a "Horse" | Journal (IEEE TMM) | 300+ | Alta | Efecto horse en MIR; necesidad de ablation tests |
| 13 | Klimashevskaia et al. (2024) | A Survey on Popularity Bias in Recommender Systems | Journal (UMUAI) | 50+ | Media | Taxonomy de popularity bias; impacto en fairness |
| 14 | Dinnissen y Bauer (2022) | Fairness in Music Recommender Systems: A Stakeholder-Centered Mini Review | Journal (Frontiers in Big Data) | 80+ | Media | Fairness multi-stakeholder en musica; gap diagnostico-accion |
| 15 | Deb et al. (2002) | A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II | Journal (IEEE TEC) | 45,000+ | Media | NSGA-II para optimizacion multi-objetivo |
| 16 | Kaminskas y Bridge (2016) | Diversity, Serendipity, Novelty, and Coverage: A Survey | Journal (ACM TIIS) | 500+ | Media | Taxonomia de metricas beyond-accuracy |
| 17 | Alpaydin (1999) | Combined 5x2cv F Test for Comparing Supervised Classification Learning Algorithms | Journal (Neural Computation) | 500+ | Media | Extension del test de Dietterich con mayor potencia |
| 18 | Rousseeuw (1987) | Silhouettes: A Graphical Aid to the Interpretation of Cluster Analysis | Journal (JCAM) | 20,000+ | Media | Indice Silhouette para validacion interna de clustering |
| 19 | Semmelrock et al. (2025) | Reproducibility in Machine-Learning-Based Research: Overview, Barriers, and Drivers | Journal (AI Magazine) | [reciente] | Media | Revision actualizada de barreras para reproducibilidad |
| 20 | Urbano et al. (2019) | Statistical Analysis of Results in MIR: Why and How | Conferencia (ISMIR) | 50+ [no verificado] | Media | Tests estadisticos adaptados al contexto MIR |

---

## 5. Gaps identificados y oportunidades

### 5.1 Gaps en la literatura

#### Gap 1: Ausencia de marco integrado clustering-recomendacion
La literatura trata la evaluacion de clustering (indices internos, estabilidad) y la evaluacion de recomendacion (precision, diversidad) como problemas separados. No se identifico un marco que integre ambas evaluaciones para sistemas basados en clustering como backbone de recomendacion. Esta desconexion es problematica porque la calidad del clustering impacta directamente la calidad de la recomendacion, pero la relacion no esta formalizada.

#### Gap 2: Tests estadisticos en MIR con clustering multi-modal
La aplicacion del marco de Demsar (2006) esta bien documentada para clasificadores supervisados, pero su adaptacion a clustering multi-modal (combinando espacios semanticos y musicales) es escasa. Las preguntas abiertas incluyen: como comparar estadisticamente algoritmos de clustering cuando los indices internos son las unicas metricas, y como manejar la dependencia entre metricas (Silhouette y Davies-Bouldin estan correlacionadas).

#### Gap 3: Evaluacion de fusion de componentes heterogeneos
La optimizacion de pesos de fusion entre componentes semanticos y musicales se ha abordado principalmente con busqueda grid o heuristica. No se identificaron trabajos que apliquen NSGA-II especificamente a la fusion de espacios de embeddings para recomendacion musical con evaluacion estadistica rigurosa de los frentes de Pareto resultantes.

#### Gap 4: Pre-registro en MIR experimental
Pese a la creciente adopcion de pre-registro en ML general (NeurIPS Workshop 2021), su aplicacion en MIR es practicamente inexistente. Los estudios de clustering y recomendacion musical tipicamente no distinguen entre analisis exploratorio y confirmatorio.

#### Gap 5: Evaluacion de ground truth proxy con cuantificacion de incertidumbre
El uso de genero como proxy es ubicuo pero la cuantificacion de "cuanto informacion pierde" esta poco estudiada. Seria valioso un analisis de sensibilidad que mida como varian los resultados de evaluacion bajo diferentes granularidades de genero (e.g., 6 generos vs 12 sub-generos vs tags libres).

### 5.2 Oportunidades para el proyecto de tesis

1. **Protocolo de evaluacion pre-registrado**: La re-ejecucion puede ser uno de los pocos proyectos de MIR con hipotesis pre-registradas, lo que fortalece significativamente la validez de las conclusiones.

2. **Aplicacion del marco completo de Demsar + Benavoli**: Combinar el diagrama CD de Demsar con las probabilidades posteriores bayesianas de Benavoli proporciona una evaluacion estadistica state-of-the-art para la comparacion de algoritmos de clustering.

3. **Evaluacion multi-nivel**: Implementar el marco de Ullmann (indices internos + estabilidad + ground truth proxy + visualizacion) constituiria una evaluacion de clustering mas rigurosa que la mayoria de los trabajos publicados en MIR.

4. **Documentacion explicita de limitaciones**: Siguiendo las recomendaciones de Kapoor y Narayanan (2023) y el checklist de Pineau et al. (2021), documentar todas las decisiones de pipeline, seeds, y posibles fuentes de leakage.

5. **Metricas beyond-accuracy**: Incluir diversidad, cobertura y analysis de popularity bias en la evaluacion del sistema de recomendacion, siguiendo FEVR y los surveys de Kaminskas y Klimashevskaia.

6. **Ablation tests anti-horse**: Disenar experimentos controlados para verificar que el componente semantico captura semantica (no longitud de letras) y que el componente musical captura contenido musical (no popularidad).

---

## 6. Entradas BibTeX

```bibtex
@article{demsar_2006_statistical,
  author    = {Dem{\v{s}}ar, Janez},
  title     = {Statistical Comparisons of Classifiers over Multiple Data Sets},
  journal   = {Journal of Machine Learning Research},
  volume    = {7},
  pages     = {1--30},
  year      = {2006},
  url       = {https://jmlr.org/papers/v7/demsar06a.html}
}

@article{garcia_2008_extension,
  author    = {Garc{\'\i}a, Salvador and Herrera, Francisco},
  title     = {An Extension on ``Statistical Comparisons of Classifiers over Multiple Data Sets'' for all Pairwise Comparisons},
  journal   = {Journal of Machine Learning Research},
  volume    = {9},
  pages     = {2677--2694},
  year      = {2008},
  url       = {https://jmlr.org/papers/v9/garcia08a.html}
}

@article{benavoli_2017_bayesian,
  author    = {Benavoli, Alessio and Corani, Giorgio and Dem{\v{s}}ar, Janez and Zaffalon, Marco},
  title     = {Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis},
  journal   = {Journal of Machine Learning Research},
  volume    = {18},
  number    = {77},
  pages     = {1--36},
  year      = {2017},
  url       = {https://jmlr.org/papers/v18/16-305.html}
}

@article{dietterich_1998_approximate,
  author    = {Dietterich, Thomas G.},
  title     = {Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms},
  journal   = {Neural Computation},
  volume    = {10},
  number    = {7},
  pages     = {1895--1923},
  year      = {1998},
  doi       = {10.1162/089976698300017197}
}

@article{alpaydin_1999_5x2cv,
  author    = {Alpayd{\i}n, Ethem},
  title     = {Combined 5x2cv {F} Test for Comparing Supervised Classification Learning Algorithms},
  journal   = {Neural Computation},
  volume    = {11},
  number    = {8},
  pages     = {1885--1892},
  year      = {1999},
  doi       = {10.1162/089976699300016007}
}

@article{kapoor_2023_leakage,
  author    = {Kapoor, Sayash and Narayanan, Arvind},
  title     = {Leakage and the Reproducibility Crisis in Machine-Learning-Based Science},
  journal   = {Patterns},
  volume    = {4},
  number    = {9},
  pages     = {100804},
  year      = {2023},
  doi       = {10.1016/j.patter.2023.100804},
  publisher = {Cell Press}
}

@article{pineau_2021_reproducibility,
  author    = {Pineau, Joelle and Vincent-Lamarre, Philippe and Sinha, Koustuv and Larivi{\`e}re, Vincent and Beygelzimer, Alina and d'Alch{\'e}-Buc, Florence and Fox, Emily and Larochelle, Hugo},
  title     = {Improving Reproducibility in Machine Learning Research (A Report from the {NeurIPS} 2019 Reproducibility Program)},
  journal   = {Journal of Machine Learning Research},
  volume    = {22},
  pages     = {1--20},
  year      = {2021},
  url       = {https://jmlr.org/papers/v22/20-303.html}
}

@inproceedings{gundersen_2018_reproducibility,
  author    = {Gundersen, Odd Erik and Kjensmo, Sigbj{\o}rn},
  title     = {State of the Art: Reproducibility in Artificial Intelligence},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2018},
  volume    = {32},
  number    = {1},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/11503}
}

@article{zangerle_2022_fevr,
  author    = {Zangerle, Eva and Bauer, Christine},
  title     = {Evaluating Recommender Systems: Survey and Framework},
  journal   = {ACM Computing Surveys},
  volume    = {55},
  number    = {8},
  articleno = {170},
  year      = {2022},
  doi       = {10.1145/3556536}
}

@article{castells_2022_offline,
  author    = {Castells, Pablo and Moffat, Alistair},
  title     = {Offline Recommender System Evaluation: Challenges and New Directions},
  journal   = {AI Magazine},
  volume    = {43},
  number    = {2},
  pages     = {225--238},
  year      = {2022},
  doi       = {10.1002/aaai.12051}
}

@article{liu_2022_stability,
  author    = {Liu, Tianyi and Dalmia, Arjun and Bhatt, Rashmi},
  title     = {Stability Estimation for Unsupervised Clustering: A Review},
  journal   = {WIREs Computational Statistics},
  volume    = {14},
  number    = {6},
  pages     = {e1575},
  year      = {2022},
  doi       = {10.1002/wics.1575}
}

@article{ullmann_2022_validation,
  author    = {Ullmann, Thomas and Hennig, Christian and Boulesteix, Anne-Laure},
  title     = {Validation of Cluster Analysis Results on Validation Data: A Systematic Framework},
  journal   = {WIREs Data Mining and Knowledge Discovery},
  volume    = {12},
  number    = {3},
  pages     = {e1444},
  year      = {2022},
  doi       = {10.1002/widm.1444}
}

@article{sturm_2014_horse,
  author    = {Sturm, Bob L.},
  title     = {A Simple Method to Determine if a Music Information Retrieval System is a ``Horse''},
  journal   = {IEEE Transactions on Multimedia},
  volume    = {16},
  number    = {6},
  pages     = {1636--1644},
  year      = {2014},
  doi       = {10.1109/TMM.2014.2330697}
}

@article{klimashevskaia_2024_popularity,
  author    = {Klimashevskaia, Anastasiia and Jannach, Dietmar and Elahi, Mehdi and Trattner, Christoph},
  title     = {A Survey on Popularity Bias in Recommender Systems},
  journal   = {User Modeling and User-Adapted Interaction},
  volume    = {34},
  number    = {5},
  pages     = {1777--1834},
  year      = {2024},
  doi       = {10.1007/s11257-024-09406-0}
}

@article{dinnissen_2022_fairness,
  author    = {Dinnissen, Karlijn and Bauer, Christine},
  title     = {Fairness in Music Recommender Systems: A Stakeholder-Centered Mini Review},
  journal   = {Frontiers in Big Data},
  volume    = {5},
  pages     = {913608},
  year      = {2022},
  doi       = {10.3389/fdata.2022.913608}
}

@article{deb_2002_nsga2,
  author    = {Deb, Kalyanmoy and Pratap, Amrit and Agarwal, Sameer and Meyarivan, T.},
  title     = {A Fast and Elitist Multiobjective Genetic Algorithm: {NSGA-II}},
  journal   = {IEEE Transactions on Evolutionary Computation},
  volume    = {6},
  number    = {2},
  pages     = {182--197},
  year      = {2002},
  doi       = {10.1109/4235.996017}
}

@article{kaminskas_2016_beyond,
  author    = {Kaminskas, Marius and Bridge, Derek},
  title     = {Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems},
  journal   = {ACM Transactions on Interactive Intelligent Systems},
  volume    = {7},
  number    = {1},
  articleno = {2},
  year      = {2016},
  doi       = {10.1145/2926720}
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

@article{semmelrock_2025_reproducibility,
  author    = {Semmelrock, Lukas and others},
  title     = {Reproducibility in Machine-Learning-Based Research: Overview, Barriers, and Drivers},
  journal   = {AI Magazine},
  year      = {2025},
  doi       = {10.1002/aaai.70002},
  note      = {[no verificado -- datos parciales de WebSearch]}
}

@inproceedings{urbano_2019_statistical_mir,
  author    = {Urbano, Juli{\'a}n and Schedl, Markus and Serra, Xavier},
  title     = {Statistical Analysis of Results in Music Information Retrieval: Why and How},
  booktitle = {Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2019},
  url       = {https://julian-urbano.info/files/publications/067-statistical-analysis-results-music-information-retrieval-why-how.pdf},
  note      = {[ano y detalles no verificados -- datos de WebSearch]}
}
```

---

## Referencias de acceso rapido

- Demsar (2006): https://jmlr.org/papers/v7/demsar06a.html
- Garcia y Herrera (2008): https://jmlr.org/papers/v9/garcia08a.html
- Benavoli et al. (2017): https://jmlr.org/papers/v18/16-305.html
- Dietterich (1998): https://doi.org/10.1162/089976698300017197
- Kapoor y Narayanan (2023): https://doi.org/10.1016/j.patter.2023.100804
- Pineau et al. (2021): https://jmlr.org/papers/v22/20-303.html
- Zangerle y Bauer (2022): https://doi.org/10.1145/3556536
- Sturm (2014): https://doi.org/10.1109/TMM.2014.2330697
- Liu et al. (2022): https://doi.org/10.1002/wics.1575
- Ullmann et al. (2022): https://doi.org/10.1002/widm.1444
- baycomp (Python): https://github.com/BayesianTestsML/tutorial/
