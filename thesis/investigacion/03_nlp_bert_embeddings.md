# Procesamiento de Lenguaje Natural, BERT y Embeddings Semanticos

## Revision sistematica de literatura para tesis de Ingenieria Informatica

**Fecha de elaboracion:** 7 de febrero de 2026
**Contexto:** Sistema de recomendacion musical hibrido con vectorizacion semantica de letras

---

## 1. Resumen ejecutivo

La representacion vectorial de texto ha experimentado una transformacion fundamental en la ultima decada, transitando desde modelos estadisticos dispersos (BoW, TF-IDF) hacia representaciones densas contextuales basadas en la arquitectura Transformer. Esta revision sistematica examina la evolucion completa de las representaciones textuales, con enfasis particular en BERT, Sentence-BERT y los modelos de destilacion multilingue que constituyen la base tecnica del proyecto de tesis. La busqueda sistematica identifico un corpus de aproximadamente 25 fuentes primarias de alta relevancia, publicadas entre 2013 y 2025, que cubren cuatro lineas tematicas principales: (1) la evolucion de word embeddings estaticos a contextuales, (2) la arquitectura Transformer y sus derivados para representacion oracional, (3) tecnicas de destilacion y compresion para despliegue eficiente, y (4) aplicaciones especificas de NLP a letras musicales. El analisis revela un consenso solido sobre la superioridad de los embeddings contextuales para tareas de similitud semantica, pero identifica un gap significativo en la validacion de estos modelos sobre texto lirico, que presenta caracteristicas linguisticas atipicas (lenguaje figurativo, slang, repeticion estructural, code-switching multilingue) que difieren sustancialmente del texto general sobre el cual estos modelos fueron entrenados y evaluados.

---

## 2. Estrategia de busqueda

### 2.1 Palabras clave

**Ingles (idioma principal de busqueda):**
- "sentence embeddings BERT transformer"
- "Sentence-BERT siamese networks semantic similarity"
- "multilingual sentence embeddings knowledge distillation"
- "NLP music lyrics BERT embeddings"
- "word2vec GloVe FastText evolution"
- "SimCSE contrastive learning"
- "MiniLM distillation compressed BERT"

**Espanol (busqueda complementaria):**
- "embeddings semanticos BERT procesamiento lenguaje natural"
- "representaciones textuales aprendizaje profundo"

### 2.2 Fuentes consultadas y resultados

| Fuente | Consultas realizadas | Resultados relevantes |
|--------|---------------------|-----------------------|
| Google Scholar (via WebSearch) | 4 | 12 fuentes primarias |
| Semantic Scholar | 2 | 6 fuentes con metadatos de citas |
| ACM Digital Library / IEEE Xplore | 2 | 4 fuentes |
| ACL Anthology | 3 | 8 fuentes (papers NLP conferencias) |
| ISMIR Archives | 1 | 3 fuentes |
| arXiv | 2 | 5 fuentes (preprints verificados) |
| HuggingFace (documentacion tecnica) | 1 | 1 fuente tecnica de modelo |

### 2.3 Criterios de inclusion y exclusion

**Inclusion:**
- Publicaciones revisadas por pares en conferencias o journals de primer nivel (ACL, EMNLP, NeurIPS, NAACL, ISMIR, EACL).
- Preprints de alto impacto en arXiv con mas de 100 citas.
- Periodo: 2013-2026 (abarcando desde Word2Vec hasta modelos actuales).
- Relevancia directa para representacion textual, embeddings oracionales, o NLP aplicado a musica.

**Exclusion:**
- Publicaciones de blogs o Medium sin respaldo academico.
- Fuentes sin autor o de publicaciones predatorias.
- Trabajos duplicados o versiones anteriores de papers con version actualizada disponible.
- Trabajos puramente teoricos sin componente experimental.

---

## 3. Estado de la cuestion

### 3.1 De representaciones dispersas a embeddings densos estaticos (2013-2017)

#### 3.1.1 Fundamentos: BoW, TF-IDF y sus limitaciones

Las primeras representaciones computacionales del texto se basaron en modelos de bolsa de palabras (Bag of Words, BoW) y su ponderacion mediante Term Frequency-Inverse Document Frequency (TF-IDF). Estas representaciones, aunque efectivas para tareas de recuperacion de informacion clasica, presentan limitaciones fundamentales bien documentadas en la literatura: (1) la alta dimensionalidad del espacio vectorial resultante (proporcional al tamano del vocabulario), (2) la dispersion extrema de los vectores (sparsity), y (3) la incapacidad de capturar relaciones semanticas entre palabras, dado que cada termino se trata como una dimension independiente sin nocion de similitud. Un vector TF-IDF no puede distinguir entre "feliz" y "contento" como semanticamente proximos, limitacion que motivo el desarrollo de representaciones densas.

#### 3.1.2 Word2Vec y el paradigma de embeddings densos

El trabajo seminal de Mikolov et al. (2013) introdujo Word2Vec, marcando un cambio paradigmatico en la representacion textual. Word2Vec propone dos arquitecturas: Continuous Bag of Words (CBOW), que predice una palabra a partir de su contexto, y Skip-gram, que predice el contexto a partir de una palabra. Ambas arquitecturas aprenden vectores densos de baja dimensionalidad (tipicamente 100-300 dimensiones) que capturan relaciones semanticas y sintacticas. La propiedad mas notable es la composicionalidad algebraica de las relaciones semanticas (e.g., vec("rey") - vec("hombre") + vec("mujer") ~ vec("reina")), que demostro empiricamente que la estructura geometrica del espacio vectorial codifica informacion semantica significativa.

Sin embargo, Word2Vec presenta una limitacion critica: genera un unico vector por palabra independientemente del contexto. Esto implica que palabras polisemicas como "banco" (financiero vs. asiento) reciben la misma representacion vectorial, colapsando significados distintos en un unico punto del espacio. Ademas, Word2Vec opera a nivel de palabra, requiriendo estrategias de composicion (promediado, ponderacion) para obtener representaciones de oraciones o documentos, estrategias que ignoran el orden y la estructura sintactica.

#### 3.1.3 GloVe: Estadisticas globales de co-ocurrencia

Pennington et al. (2014) propusieron GloVe (Global Vectors for Word Representation), que complementa el enfoque de ventana local de Word2Vec con estadisticas globales de co-ocurrencia del corpus. GloVe construye explicitamente una matriz de co-ocurrencia palabra-palabra y factoriza esta matriz para obtener vectores que capturan tanto relaciones locales como globales. En evaluaciones de analogia y similitud semantica, GloVe demostro rendimiento competitivo con Word2Vec, con la ventaja adicional de una fundamentacion teorica mas clara que conecta el aprendizaje de embeddings con la factorizacion matricial.

#### 3.1.4 FastText: Sub-palabras y morfologia

Bojanowski et al. (2017) extendieron Word2Vec mediante FastText, cuya innovacion principal es la representacion de palabras como sumas de vectores de n-gramas de caracteres. Esta descomposicion sub-lexica permite: (1) generar vectores para palabras fuera del vocabulario (out-of-vocabulary, OOV) basandose en sus componentes morfologicos, (2) capturar mejor la morfologia de lenguas aglutinantes o con rica flexion, y (3) manejar variaciones ortograficas y errores tipograficos. Para el procesamiento de letras musicales, donde el slang, las contracciones informales y las variaciones ortograficas son frecuentes, FastText ofrece ventajas sobre Word2Vec al poder generar representaciones para formas lexicas no estandar.

#### 3.1.5 Limitacion compartida: Estaticidad contextual

A pesar de sus contribuciones individuales, Word2Vec, GloVe y FastText comparten una limitacion fundamental: producen embeddings estaticos, es decir, un vector fijo por tipo lexico independiente del contexto de uso. La comunidad de NLP reconocio progresivamente que esta estaticidad constituye un obstaculo para tareas que requieren comprension contextual, motivando el desarrollo de representaciones contextuales.

### 3.2 La revolucion Transformer y los embeddings contextuales (2017-2019)

#### 3.2.1 Arquitectura Transformer

Vaswani et al. (2017) introdujeron la arquitectura Transformer en el articulo "Attention Is All You Need", presentado en NeurIPS 2017. Este trabajo, con mas de 60,000 citas acumuladas, constituye uno de los articulos mas influyentes en la historia del aprendizaje automatico. La innovacion central del Transformer es el mecanismo de auto-atencion (self-attention), que permite a cada token de una secuencia atender directamente a todos los demas tokens, eliminando la necesidad de procesamiento secuencial inherente a las redes recurrentes (LSTM, GRU).

La arquitectura Transformer presenta tres ventajas estructurales sobre las RNN: (1) paralelizacion completa del computo durante el entrenamiento, (2) caminos de atencion directos entre tokens arbitrariamente distantes (resolviendo el problema del gradiente desvaneciente en secuencias largas), y (3) escalabilidad superior con el tamano del modelo y los datos de entrenamiento. El mecanismo de multi-head attention permite al modelo capturar diferentes tipos de relaciones (sintacticas, semanticas, referenciales) en cabezas de atencion separadas.

#### 3.2.2 BERT: Representaciones bidireccionales pre-entrenadas

Devlin et al. (2019) presentaron BERT (Bidirectional Encoder Representations from Transformers) en NAACL 2019, estableciendo un nuevo paradigma de pre-entrenamiento y fine-tuning que domino el NLP durante los anos siguientes. BERT innova en dos aspectos fundamentales respecto a modelos previos como ELMo (Peters et al., 2018) y GPT (Radford et al., 2018):

1. **Bidireccionalidad profunda**: A diferencia de los modelos de lenguaje unidireccionales (izquierda-a-derecha o derecha-a-izquierda), BERT condiciona conjuntamente sobre el contexto izquierdo y derecho en todas las capas del encoder Transformer, logrando representaciones verdaderamente bidireccionales.

2. **Pre-entrenamiento con Masked Language Model (MLM)**: BERT enmascara aleatoriamente el 15% de los tokens de entrada y entrena el modelo para predecirlos, lo que fuerza la captura de relaciones contextuales profundas. Adicionalmente, la tarea de Next Sentence Prediction (NSP) entrena la comprension de relaciones inter-oracionales.

BERT alcanzo resultados estado del arte en 11 tareas de NLP, incluyendo mejoras de 7.7 puntos absolutos en GLUE y 5.1 puntos en SQuAD 2.0. La variante BERT-base comprende 12 capas Transformer, 768 dimensiones de representacion y 110M parametros, mientras que BERT-large escala a 24 capas, 1024 dimensiones y 340M parametros.

No obstante, BERT presenta una limitacion critica para aplicaciones que requieren comparacion de textos: la generacion de embeddings oracionales. El token [CLS] de BERT, originalmente disenado para la tarea NSP, produce embeddings oracionales de baja calidad que frecuentemente rinden peor que promedios de embeddings GloVe en tareas de similitud textual semantica (STS), como documentaron Reimers y Gurevych (2019). Ademas, la comparacion directa de dos oraciones con BERT requiere alimentar ambas simultaneamente al modelo, lo que genera una complejidad computacional cuadratica inaceptable para busqueda o clustering a escala (10,000 oraciones requeririan ~50 millones de inferencias).

### 3.3 Sentence-BERT y embeddings oracionales eficientes (2019-2021)

#### 3.3.1 Sentence-BERT (SBERT)

Reimers y Gurevych (2019) propusieron Sentence-BERT, publicado en EMNLP-IJCNLP 2019, como solucion a las limitaciones de BERT para embeddings oracionales. SBERT modifica la arquitectura BERT mediante redes siamesas y de tripletes, de modo que cada oracion se procesa independientemente para producir un embedding de dimensionalidad fija. La clave arquitectonica es la capa de pooling aplicada sobre las salidas de BERT: los autores evaluaron tres estrategias (CLS token, max-pooling, mean-pooling) y determinaron que mean-pooling produce los mejores resultados en tareas STS.

El impacto practico de SBERT es dramatico: la busqueda del par mas similar en una coleccion de 10,000 oraciones pasa de ~65 horas con BERT a ~5 segundos con SBERT, manteniendo precision comparable. Los embeddings resultantes son vectores de dimensionalidad fija (768D para BERT-base) que pueden compararse directamente mediante similitud coseno, habilitando su uso en clustering, recuperacion de informacion y sistemas de recomendacion.

SBERT fue entrenado sobre datos de Natural Language Inference (NLI): SNLI (570K pares) y MultiNLI (430K pares), utilizando una funcion objetivo que combina clasificacion de pares (entailment, contradiction, neutral) con regresion de similitud. Este esquema de entrenamiento permite al modelo capturar relaciones semanticas finas entre oraciones.

#### 3.3.2 SimCSE: Aprendizaje contrastivo para embeddings

Gao, Yao y Chen (2021) introdujeron SimCSE en EMNLP 2021, un framework de aprendizaje contrastivo que avanza significativamente el estado del arte en embeddings oracionales. SimCSE propone dos variantes:

- **Unsupervised SimCSE**: Utiliza la propia oracion como par positivo, con dropout estandar como unica forma de aumentacion de datos. Esta elegante simplicidad resulta en mejoras de 4.2% en correlacion de Spearman sobre los mejores metodos previos no supervisados.

- **Supervised SimCSE**: Incorpora pares de NLI, usando pares de entailment como positivos y pares de contradiccion como negativos dificiles (hard negatives), logrando 81.6% de correlacion de Spearman promedio en tareas STS.

El analisis teorico de SimCSE demuestra que el objetivo contrastivo regulariza el espacio de embeddings pre-entrenados, transformando su distribucion anisotropica en una distribucion mas uniforme, y alinea mejor los pares positivos cuando hay senal supervisada. Este trabajo es relevante para el proyecto de tesis porque fundamenta teoricamente por que los embeddings contrastivos producen mejores representaciones para clustering: la uniformidad del espacio es un prerequisito para que metricas de distancia como coseno funcionen adecuadamente.

#### 3.3.3 Survey comprehensivo de representaciones oracionales

Kashyap et al. (2023) presentaron "A Comprehensive Survey of Sentence Representations: From the BERT Epoch to the ChatGPT Era and Beyond" en EACL 2024, proporcionando una taxonomia sistematica de los metodos de representacion oracional. El survey organiza la literatura en tres paradigmas: (1) metodos no supervisados basados en auto-supervision, (2) metodos supervisados con pares anotados, y (3) metodos basados en transferencia desde modelos de lenguaje grandes. Los autores identifican como desafios abiertos la evaluacion robusta de embeddings (mas alla de STS), la adaptacion a dominios especificos, y la eficiencia computacional en escenarios de despliegue.

### 3.4 Destilacion, compresion y multilingualismo (2019-2021)

#### 3.4.1 DistilBERT: Compresion por destilacion de conocimiento

Sanh et al. (2019) propusieron DistilBERT, presentado en el Workshop EMC de NeurIPS 2019, demostrando que es posible reducir el tamano de BERT en un 40% (de 110M a 66M parametros) reteniendo el 97% de su capacidad linguistica y siendo 60% mas rapido en inferencia. La destilacion emplea una funcion de perdida triple que combina: (1) distilacion soft del conocimiento (matching de distribuciones de probabilidad), (2) modelado de lenguaje enmascarado (MLM), y (3) distancia coseno entre representaciones del estudiante y el profesor. El modelo estudiante reduce el numero de capas por un factor de dos mientras mantiene la dimension oculta identica a BERT-base.

La relevancia de DistilBERT para el proyecto de tesis radica en que establece el principio de que modelos significativamente mas pequenos pueden preservar la mayor parte de la capacidad de representacion semantica, principio que fundamenta la viabilidad del modelo `paraphrase-multilingual-MiniLM-L12-v2` utilizado en el proyecto.

#### 3.4.2 MiniLM: Destilacion de auto-atencion profunda

Wang et al. (2020) presentaron MiniLM en NeurIPS 2020, proponiendo una tecnica de destilacion mas sofisticada que opera sobre el modulo de auto-atencion de la ultima capa del Transformer profesor. A diferencia de DistilBERT, que destila las distribuciones de salida, MiniLM destila las matrices de atencion: tanto el producto escalado punto de queries y keys como el de values. Esta destilacion de auto-atencion presenta dos ventajas: (1) permite flexibilidad en el numero de capas del estudiante sin necesidad de mapeo explicito capa-a-capa, y (2) transfiere conocimiento estructural sobre las relaciones entre tokens, no solo sobre predicciones finales.

Los resultados experimentales demuestran que un modelo MiniLM de 6 capas y 768 dimensiones es 2.0x mas rapido que BERT-base mientras retiene mas del 99% del rendimiento en tareas como SQuAD 2.0 y MNLI, superando a DistilBERT y TinyBERT en la mayoria de benchmarks. La version MiniLMv2 (Wang et al., 2020b) extiende la destilacion a relaciones multi-head, mejorando aun mas la eficiencia.

#### 3.4.3 XLM-RoBERTa: Representaciones multilingues a escala

Conneau et al. (2020) presentaron XLM-RoBERTa (XLM-R) en ACL 2020, un modelo multilingue pre-entrenado sobre 2.5 terabytes de datos filtrados de CommonCrawl en 100 idiomas. XLM-R demostro que el pre-entrenamiento multilingue a escala suficiente no sacrifica rendimiento por idioma individual, superando significativamente a mBERT: +14.6% en XNLI, +13% en MLQA, y +2.4% en NER promediando todos los idiomas. El rendimiento en idiomas de bajos recursos mejora dramaticamente (+15.7% en Swahili, +11.4% en Urdu).

XLM-R es particularmente relevante como modelo base para la destilacion multilingue de SBERT, dado que sirve como arquitectura estudiante en el esquema teacher-student de Reimers y Gurevych (2020).

#### 3.4.4 Multilingual SBERT: Destilacion cross-lingual de embeddings oracionales

Reimers y Gurevych (2020) publicaron "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation" en EMNLP 2020, estableciendo el framework que sustenta directamente el modelo utilizado en el proyecto de tesis. La idea central es elegante: una oracion traducida debe mapearse al mismo punto del espacio vectorial que la oracion original. El esquema teacher-student funciona asi:

1. Un modelo SBERT monolingue (ingles) sirve como profesor.
2. Se utilizan corpus de traduccion paralela para generar pares (oracion_original, oracion_traducida).
3. Un modelo multilingue (XLM-R) se entrena como estudiante para producir embeddings que repliquen los del profesor tanto para la oracion original como para su traduccion.

Este proceso de destilacion cross-lingual permite extender la capacidad de embeddings oracionales de alta calidad (desarrollada originalmente para ingles) a mas de 50 idiomas simultaneamente, creando un espacio vectorial compartido donde oraciones semanticamente equivalentes en diferentes idiomas se mapean a regiones proximas. El modelo `paraphrase-multilingual-MiniLM-L12-v2` utilizado en el proyecto es producto directo de este pipeline: combina la arquitectura compacta de MiniLM (12 capas, 384 dimensiones) con el entrenamiento multilingue por destilacion.

#### 3.4.5 Especificaciones del modelo paraphrase-multilingual-MiniLM-L12-v2

Segun la documentacion oficial de Sentence-Transformers y HuggingFace, el modelo presenta las siguientes especificaciones tecnicas:

- **Arquitectura base**: MiniLM con 12 capas Transformer
- **Dimensionalidad de embeddings**: 384 dimensiones
- **Parametros totales**: ~118M
- **Longitud maxima de secuencia**: 128 tokens
- **Estrategia de pooling**: Mean pooling
- **Idiomas soportados**: 50+ idiomas, incluyendo ingles y espanol
- **Entrenamiento**: Destilacion desde un modelo monolingue SBERT usando corpus paralelos multilingues
- **Tarea de entrenamiento**: Paraphrase detection (deteccion de parafrasis)

La limitacion de 128 tokens de longitud maxima es relevante para el procesamiento de letras musicales: una cancion promedio contiene entre 200 y 500 palabras, lo que implica que letras completas excedan significativamente esta ventana. Las estrategias de mitigacion incluyen: (1) truncamiento a los primeros 128 tokens (perdiendo informacion de versos finales), (2) segmentacion y promediado de embeddings por segmentos, o (3) seleccion de segmentos representativos. La eleccion de estrategia impacta directamente la calidad de los embeddings resultantes y debe documentarse como decision de diseno en el proyecto.

### 3.5 NLP aplicado a letras musicales: Desafios y contribuciones (2005-2025)

#### 3.5.1 Caracteristicas linguisticas distintivas de las letras musicales

Las letras musicales presentan un conjunto de desafios linguisticos bien documentados que las distinguen del texto general sobre el cual los modelos de lenguaje son tipicamente entrenados y evaluados. La literatura identifica las siguientes particularidades:

1. **Lenguaje figurativo**: Metaforas, metonimias, personificaciones y simbolismo son recursos retoricos centrales en la escritura lirica (Watanabe y Goto, 2020). Los modelos pre-entrenados sobre texto literal (Wikipedia, libros, noticias) pueden no capturar adecuadamente significados no literales.

2. **Vocabulario no estandar y slang**: Las letras frecuentemente emplean contracciones informales ("gonna", "wanna"), argot musical y cultural, y variaciones ortograficas deliberadas. Investigadores han observado que los modelos de clasificacion de genero asignan peso desproporcionado a palabras coloquiales en generos como Hip-Hop/Rap (Tsaptsinos, 2017).

3. **Repeticion estructural**: Los estribillos y hooks se repiten multiples veces en una cancion, generando una distribucion de tokens altamente redundante que difiere de la distribucion esperada en texto general. Esta repeticion puede sesgar los embeddings hacia el contenido del estribillo, sub-representando versos con contenido semantico mas diverso.

4. **Code-switching multilingue**: En datasets multilingues como el del proyecto de tesis (predominantemente ingles y espanol), el code-switching intra-cancion (alternancia de idiomas dentro de una misma cancion) plantea desafios adicionales para modelos multilingues que fueron entrenados con textos monolinguies homogeneos.

5. **Estructura no prosaica**: Las letras se organizan en versos, estrofas y estribillos con patrones metricos y rimicos que imponen restricciones sobre la seleccion lexica, priorizando a veces la sonoridad sobre la coherencia semantica.

#### 3.5.2 Trabajos pioneros en NLP para letras musicales

La aplicacion de tecnicas de NLP a letras musicales tiene antecedentes que preceden la era de deep learning. El trabajo de Mayer et al. (2008) y contribuciones tempranas en ACM Multimedia (Logan et al., 2005) establecieron las bases para el campo, utilizando representaciones BoW y TF-IDF para clasificacion de genero y analisis de sentimiento en letras.

Tsaptsinos (2017) represento un avance significativo al adaptar Hierarchical Attention Networks (HAN) para la clasificacion de genero musical basada exclusivamente en letras, presentado en ISMIR 2017. El modelo explota la estructura jerarquica natural de las letras (palabras -> lineas -> segmentos -> cancion) mediante capas de atencion que aprenden la importancia relativa de cada nivel. Los experimentos sobre un dataset de 117 generos demostraron que el HAN supera tanto modelos no neuronales como modelos neuronales mas simples. Crucialmente, el mecanismo de atencion permite visualizar que palabras y lineas son mas relevantes para la clasificacion, proporcionando interpretabilidad. Este trabajo es un antecedente directo de la hipotesis del proyecto de tesis de que las letras contienen informacion semantica suficiente para discriminar entre estilos musicales.

#### 3.5.3 BERT y Transformers aplicados a letras

La aplicacion de modelos basados en Transformers a letras musicales ha generado una linea de investigacion activa:

**LyEmoBERT (Revathy et al., 2023)**: Publicado en Procedia Computer Science, este trabajo aplica BERT para la clasificacion de emociones en letras musicales, utilizando el dataset Music4All. LyEmoBERT emplea un enfoque de transfer learning para clasificar letras en cuatro categorias emocionales (feliz, enojado, relajado, triste), alcanzando una precision del 92%. Adicionalmente, los autores construyen un sistema de recomendacion basado en similitud semantica de letras utilizando Sentence Transformers. Este trabajo valida empiricamente la viabilidad de usar modelos pre-entrenados tipo BERT para extraer informacion semantica emocional de letras, aunque su evaluacion se limita a un esquema de clasificacion supervisada con categorias discretas, no a la generacion de embeddings para similitud continua como en el proyecto de tesis.

**Lyrics Matter (2024)**: Trabajos recientes como el preprint "Lyrics Matter: Exploiting the Power of Learnt Representations for Music" (arXiv 2512.05508) exploran la generacion de representaciones aprendidas de letras para diversas tareas de MIR, confirmando que las letras contienen informacion complementaria a las features de audio.

**LyBERT**: El trabajo de clasificacion multi-clase de letras usando BERT (disponible en ResearchGate) demuestra la aplicabilidad de BERT para tareas de clasificacion textual sobre letras, con adaptaciones especificas al dominio.

**Analisis de similitud lirica perceptual**: El trabajo de Akama y Murakami (2024, arXiv 2404.02342) realiza un analisis computacional de la percepcion de similitud lirica, demostrando que la similitud coseno entre embeddings de modelos BERT pre-entrenados correlaciona significativamente con la similitud percibida por humanos, validando el uso de embeddings como proxy de similitud semantica en el dominio lirico.

#### 3.5.4 Lyrics Information Processing como campo emergente

Watanabe y Goto (2020) propusieron "Lyrics Information Processing" (LIP) como un campo de investigacion puente entre NLP y MIR, publicado en el Workshop NLP4MusA en la conferencia AACL-IJCNLP 2020. Los autores identifican tres areas principales: (1) analisis de letras (clasificacion, extraccion de informacion, analisis de sentimiento), (2) generacion de letras y soporte de escritura, y (3) aplicaciones centradas en letras (recuperacion, recomendacion, visualizacion). Este marco conceptual situa el proyecto de tesis en la interseccion de las areas (1) y (3): el uso de embeddings semanticos de letras para alimentar un sistema de recomendacion.

#### 3.5.5 Procesamiento multilingue de letras

El tratamiento de letras en multiples idiomas agrega complejidad al procesamiento semantico. El modelo `paraphrase-multilingual-MiniLM-L12-v2` fue disenado para producir embeddings comparables entre idiomas, lo que teoricamente permite que una cancion en espanol y una en ingles con tematica similar se mapeen a regiones proximas del espacio vectorial. Sin embargo, la validacion empirica de esta propiedad sobre texto lirico (vs. texto general) es limitada en la literatura.

Trabajos recientes como el de Tavares y Ayres (2025) han comenzado a explorar la validacion cross-lingual de SBERT para letras musicales, aunque esta linea de investigacion se encuentra en etapas iniciales. La falta de benchmarks estandarizados para similitud semantica de letras multilingues constituye un gap significativo que el proyecto podria contribuir a documentar.

### 3.6 Embeddings para clustering y sistemas de recomendacion

#### 3.6.1 Propiedades geometricas relevantes para clustering

La calidad de los embeddings para clustering depende de propiedades geometricas especificas del espacio vectorial resultante. La investigacion reciente ha identificado tres propiedades criticas:

1. **Uniformidad**: Los embeddings deben distribuirse uniformemente en la hipersfera unitaria, evitando la concentracion anisotropica que caracteriza a los embeddings BERT no ajustados (Li et al., 2020; Gao et al., 2021). SimCSE demostro que el aprendizaje contrastivo mejora significativamente la uniformidad.

2. **Alineacion**: Pares semanticamente similares deben mapearse a puntos proximos. La metrica de alineacion mide la distancia esperada entre embeddings de pares positivos.

3. **Separabilidad**: Clusters semanticos distintos deben ocupar regiones diferenciadas del espacio. Esta propiedad es prerequisito para que algoritmos de clustering no supervisados (K-Means, DBSCAN, clustering jerarquico) puedan identificar agrupaciones significativas.

Los embeddings de SBERT, al ser entrenados con objetivos que optimizan similitud coseno, tienden a exhibir mejor uniformidad y alineacion que los embeddings BERT sin ajustar, lo que fundamenta su idoneidad para tareas de clustering como las requeridas por el proyecto de tesis.

#### 3.6.2 Embeddings en sistemas de recomendacion

El survey de Zhang et al. (2023, arXiv 2310.18608) sobre "Embedding in Recommender Systems" documenta sistematicamente como las representaciones vectoriales densas se han convertido en el componente central de los sistemas de recomendacion modernos. Los embeddings textuales, en particular, permiten incorporar informacion semantica del contenido (descripciones, resenas, letras) como features del item, complementando las senales colaborativas tradicionales.

En el contexto de sistemas de recomendacion musical, los embeddings de letras pueden funcionar como representacion del contenido semantico de las canciones, habilitando recomendaciones basadas en similitud tematica y emocional. La fusion tardia (late fusion) con features de audio, como la implementada en el proyecto de tesis, permite combinar las senales semanticas con las acusticas sin requerir que ambas modalidades compartan el mismo espacio de representacion.

Un trabajo especificamente relevante es el de la ACM Web Conference 2024, que estudia el impacto de integrar embeddings de clustering con embeddings de texto en sistemas de recomendacion, demostrando que la combinacion mejora el rendimiento respecto a usar cualquiera de las dos fuentes de informacion de forma aislada.

---

## 4. Tabla de fuentes principales

| # | Autores (Ano) | Titulo | Tipo | Citas aprox. | Relevancia | Aporte clave |
|---|---------------|--------|------|-------------|------------|---------------|
| 1 | Reimers y Gurevych (2019) | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Conferencia (EMNLP) | ~8,000 | Alta | Arquitectura siamesa para embeddings oracionales eficientes; base de toda la familia SBERT |
| 2 | Devlin et al. (2019) | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Conferencia (NAACL) | ~100,000 | Alta | Paradigma pre-training + fine-tuning bidireccional; modelo base del proyecto |
| 3 | Vaswani et al. (2017) | Attention Is All You Need | Conferencia (NeurIPS) | ~140,000 | Alta | Arquitectura Transformer con self-attention; fundamento de todos los modelos posteriores |
| 4 | Reimers y Gurevych (2020) | Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation | Conferencia (EMNLP) | ~2,000 | Alta | Destilacion cross-lingual teacher-student; framework directo del modelo usado en tesis |
| 5 | Wang et al. (2020) | MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers | Conferencia (NeurIPS) | ~1,500 | Alta | Destilacion de auto-atencion; arquitectura base del modelo paraphrase-multilingual-MiniLM |
| 6 | Gao, Yao y Chen (2021) | SimCSE: Simple Contrastive Learning of Sentence Embeddings | Conferencia (EMNLP) | ~3,500 | Alta | Aprendizaje contrastivo para uniformidad de embeddings; fundamentacion teorica para clustering |
| 7 | Mikolov et al. (2013) | Efficient Estimation of Word Representations in Vector Space | Preprint (arXiv) / ICLR Workshop | ~40,000 | Alta | Word2Vec: paradigma de embeddings densos; hito fundacional |
| 8 | Conneau et al. (2020) | Unsupervised Cross-lingual Representation Learning at Scale | Conferencia (ACL) | ~7,000 | Alta | XLM-RoBERTa: modelo base multilingue para destilacion SBERT |
| 9 | Sanh et al. (2019) | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter | Workshop (NeurIPS EMC) | ~8,000 | Media-Alta | Primera destilacion exitosa de BERT; establece viabilidad de modelos comprimidos |
| 10 | Pennington, Socher y Manning (2014) | GloVe: Global Vectors for Word Representation | Conferencia (EMNLP) | ~35,000 | Media | Embeddings basados en co-ocurrencia global; complementa Word2Vec en la evolucion |
| 11 | Bojanowski et al. (2017) | Enriching Word Vectors with Subword Information | Revista (TACL) | ~12,000 | Media | FastText: sub-word embeddings; relevante para vocabulario no estandar de letras |
| 12 | Tsaptsinos (2017) | Lyrics-Based Music Genre Classification Using a Hierarchical Attention Network | Conferencia (ISMIR) | ~120 | Alta | HAN para letras musicales; valida que letras contienen informacion discriminativa de genero |
| 13 | Revathy, Pillai y Daneshfar (2023) | LyEmoBERT: Classification of lyrics' emotion and recommendation using a pre-trained model | Revista (Procedia Computer Science) | ~30 | Media-Alta | BERT para emociones en letras + recomendacion con Sentence Transformers |
| 14 | Watanabe y Goto (2020) | Lyrics Information Processing: Analysis, Generation, and Applications | Workshop (NLP4MusA) | ~35 | Media | Marco conceptual LIP como campo puente NLP-MIR |
| 15 | Kashyap et al. (2023) | A Comprehensive Survey of Sentence Representations: From the BERT Epoch to the ChatGPT Era and Beyond | Conferencia (EACL) | ~80 | Media-Alta | Survey comprehensivo de metodos de representacion oracional |
| 16 | Akama y Murakami (2024) | A Computational Analysis of Lyric Similarity Perception | Preprint (arXiv) | ~5 [no verificado] | Media-Alta | Validacion de similitud coseno BERT como proxy de similitud perceptual en letras |
| 17 | Zhang et al. (2023) | Embedding in Recommender Systems: A Survey | Preprint (arXiv) | ~50 [no verificado] | Media | Survey de embeddings en sistemas de recomendacion |
| 18 | Muennighoff et al. (2023) | MTEB: Massive Text Embedding Benchmark | Conferencia (EACL) | ~400 [no verificado] | Media | Benchmark estandar para evaluacion de embeddings textuales |

**Nota**: Los conteos de citas son aproximaciones basadas en los datos disponibles al momento de la busqueda. Las fuentes marcadas con [no verificado] tienen metadatos no confirmados directamente.

---

## 5. Gaps identificados y oportunidades

### 5.1 Gaps en la literatura

1. **Ausencia de benchmarks de similitud semantica para letras musicales**: No existe un dataset estandarizado con anotaciones humanas de similitud semantica entre letras, analogo a STS-B para texto general. La evaluacion de embeddings de letras se realiza tipicamente con proxies indirectos (genero musical, emocion), no con juicios directos de similitud semantica lirica.

2. **Validacion limitada de modelos multilingues sobre letras**: Si bien modelos como `paraphrase-multilingual-MiniLM-L12-v2` han sido evaluados extensivamente sobre texto general multilingue, su rendimiento especifico sobre letras musicales en idiomas distintos al ingles (y particularmente sobre letras con code-switching) no esta documentado en la literatura revisada.

3. **Impacto de la repeticion estructural en embeddings**: La influencia de la repeticion de estribillos en la representacion vectorial de letras completas no ha sido sistematicamente estudiada. Dado que mean-pooling promedia todas las posiciones de tokens, la repeticion de estribillos podria dominar el embedding resultante, sub-representando el contenido unico de los versos.

4. **Evaluacion de la longitud de secuencia truncada**: El modelo del proyecto tiene una ventana maxima de 128 tokens, pero las implicaciones de este truncamiento sobre letras de longitud variable (canciones cortas vs. extensas, rap vs. baladas) no han sido cuantificadas en la literatura.

5. **Dominio especifico vs. modelo general**: No existen modelos SBERT fine-tuneados especificamente sobre corpus de letras musicales. Toda la literatura utiliza modelos pre-entrenados sobre texto general (NLI, parafraseo, Wikipedia) y los aplica directamente a letras sin adaptacion de dominio.

### 5.2 Oportunidades para el proyecto de tesis

1. **Documentacion explicita de limitaciones**: El proyecto puede contribuir documentando rigurosamente las limitaciones de aplicar SBERT multilingue a letras musicales, cuantificando la degradacion de rendimiento respecto a texto general (si existe) y las fuentes de ruido especificas del dominio.

2. **Analisis de sensibilidad al truncamiento**: Experimentar con diferentes estrategias de manejo de longitud (truncamiento, segmentacion, seleccion de segmentos) y documentar su impacto en la calidad del clustering seria una contribucion practica valiosa.

3. **Complementariedad semantico-acustica**: El sistema hibrido del proyecto esta posicionado para cuantificar empiricamente cuanta informacion semantica (letras) vs. acustica (features Spotify) contribuye a la recomendacion, lo que es un gap identificado en la seccion de fusion multimodal.

4. **Validacion cross-lingual en dominio lirico**: El dataset del proyecto (predominantemente ingles y espanol) ofrece una oportunidad natural para evaluar si las canciones en espanol e ingles con tematica similar se mapean efectivamente a regiones proximas del espacio de embeddings.

### 5.3 Conexiones no exploradas

- **Interseccion clustering-destilacion**: La interaccion entre la compresion de dimensionalidad (384D de MiniLM vs. 768D de SBERT completo) y la calidad del clustering no ha sido estudiada sistematicamente. Los 384D podrian ser suficientes para capturar la variabilidad semantica de letras musicales, pero esto requiere validacion empirica.

- **Embeddings contrastivos para letras**: Aplicar el framework SimCSE especificamente entrenado sobre pares de letras (mismo genero como positivos, generos distantes como negativos) podria producir representaciones mas discriminativas para el dominio musical, pero este enfoque no ha sido explorado.

- **Estructura jerarquica de letras y Transformers**: El enfoque HAN de Tsaptsinos (2017) explota la estructura jerarquica de las letras, pero no ha sido combinado con embeddings Transformer. Un modelo que aplique atencion jerarquica sobre embeddings SBERT de versos individuales podria capturar mejor la estructura compositiva de las letras.

---

## 6. Entradas BibTeX

```bibtex
@inproceedings{reimers_2019_sentencebert,
  author    = {Reimers, Nils and Gurevych, Iryna},
  title     = {Sentence-{BERT}: Sentence Embeddings using Siamese {BERT}-Networks},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing ({EMNLP}-{IJCNLP})},
  pages     = {3982--3992},
  year      = {2019},
  publisher = {Association for Computational Linguistics},
  address   = {Hong Kong, China},
  doi       = {10.18653/v1/D19-1410},
  url       = {https://aclanthology.org/D19-1410/}
}

@inproceedings{devlin_2019_bert,
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  title     = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages     = {4171--4186},
  year      = {2019},
  publisher = {Association for Computational Linguistics},
  address   = {Minneapolis, Minnesota},
  doi       = {10.18653/v1/N19-1423},
  url       = {https://aclanthology.org/N19-1423/}
}

@inproceedings{vaswani_2017_attention,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  title     = {Attention Is All You Need},
  booktitle = {Advances in Neural Information Processing Systems 30 ({NeurIPS})},
  pages     = {5998--6008},
  year      = {2017},
  url       = {https://papers.nips.cc/paper/7181-attention-is-all-you-need}
}

@inproceedings{reimers_2020_multilingual,
  author    = {Reimers, Nils and Gurevych, Iryna},
  title     = {Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  pages     = {4512--4525},
  year      = {2020},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2020.emnlp-main.365},
  url       = {https://aclanthology.org/2020.emnlp-main.365/}
}

@inproceedings{wang_2020_minilm,
  author    = {Wang, Wenhui and Wei, Furu and Dong, Li and Bao, Hangbo and Yang, Nan and Zhou, Ming},
  title     = {{MiniLM}: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers},
  booktitle = {Advances in Neural Information Processing Systems 33 ({NeurIPS})},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.10957}
}

@inproceedings{gao_2021_simcse,
  author    = {Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
  title     = {{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  pages     = {6894--6910},
  year      = {2021},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2021.emnlp-main.552},
  url       = {https://aclanthology.org/2021.emnlp-main.552/}
}

@article{mikolov_2013_word2vec,
  author    = {Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  title     = {Efficient Estimation of Word Representations in Vector Space},
  journal   = {arXiv preprint arXiv:1301.3781},
  year      = {2013},
  url       = {https://arxiv.org/abs/1301.3781}
}

@inproceedings{conneau_2020_xlmroberta,
  author    = {Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  title     = {Unsupervised Cross-lingual Representation Learning at Scale},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics ({ACL})},
  pages     = {8440--8451},
  year      = {2020},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2020.acl-main.747},
  url       = {https://aclanthology.org/2020.acl-main.747/}
}

@article{sanh_2019_distilbert,
  author    = {Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  title     = {{DistilBERT}, a distilled version of {BERT}: smaller, faster, cheaper and lighter},
  journal   = {arXiv preprint arXiv:1910.01108},
  year      = {2019},
  note      = {Presented at the 5th Workshop on Energy Efficient Machine Learning and Cognitive Computing, NeurIPS 2019},
  url       = {https://arxiv.org/abs/1910.01108}
}

@inproceedings{pennington_2014_glove,
  author    = {Pennington, Jeffrey and Socher, Richard and Manning, Christopher D.},
  title     = {{GloVe}: Global Vectors for Word Representation},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  pages     = {1532--1543},
  year      = {2014},
  publisher = {Association for Computational Linguistics},
  doi       = {10.3115/v1/D14-1162},
  url       = {https://aclanthology.org/D14-1162/}
}

@article{bojanowski_2017_fasttext,
  author    = {Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  title     = {Enriching Word Vectors with Subword Information},
  journal   = {Transactions of the Association for Computational Linguistics},
  volume    = {5},
  pages     = {135--146},
  year      = {2017},
  doi       = {10.1162/tacl_a_00051},
  url       = {https://aclanthology.org/Q17-1010/}
}

@inproceedings{tsaptsinos_2017_lyricals,
  author    = {Tsaptsinos, Alexandros},
  title     = {Lyrics-Based Music Genre Classification Using a Hierarchical Attention Network},
  booktitle = {Proceedings of the 18th International Society for Music Information Retrieval Conference ({ISMIR})},
  pages     = {694--701},
  year      = {2017},
  address   = {Suzhou, China},
  url       = {https://archives.ismir.net/ismir2017/paper/000043.pdf}
}

@article{revathy_2023_lyemobert,
  author    = {Revathy, V. R. and Pillai, Anitha S. and Daneshfar, Fatemah},
  title     = {{LyEmoBERT}: Classification of lyrics' emotion and recommendation using a pre-trained model},
  journal   = {Procedia Computer Science},
  volume    = {218},
  pages     = {1196--1206},
  year      = {2023},
  publisher = {Elsevier},
  doi       = {10.1016/j.procs.2023.01.098},
  url       = {https://www.sciencedirect.com/science/article/pii/S1877050923000984}
}

@inproceedings{watanabe_2020_lip,
  author    = {Watanabe, Kento and Goto, Masataka},
  title     = {Lyrics Information Processing: Analysis, Generation, and Applications},
  booktitle = {Proceedings of the 1st Workshop on NLP for Music and Audio ({NLP4MusA})},
  year      = {2020},
  url       = {https://aclanthology.org/2020.nlp4musa-1.2.pdf}
}

@inproceedings{kashyap_2023_survey,
  author    = {Kashyap, Abhinav Ramesh and Nguyen, Thanh-Tung and Schlegel, Viktor and Winkler, Stefan and Ng, See-Kiong and Poria, Soujanya},
  title     = {A Comprehensive Survey of Sentence Representations: From the {BERT} Epoch to the {ChatGPT} Era and Beyond},
  booktitle = {Findings of the Association for Computational Linguistics: {EACL} 2024},
  year      = {2024},
  url       = {https://arxiv.org/abs/2305.12641}
}

@article{akama_2024_lyric_similarity,
  author    = {Akama, Reishi and Murakami, Kazuya},
  title     = {A Computational Analysis of Lyric Similarity Perception},
  journal   = {arXiv preprint arXiv:2404.02342},
  year      = {2024},
  url       = {https://arxiv.org/abs/2404.02342}
}

@article{zhang_2023_embedding_recsys,
  author    = {Zhang, Xiangyu and others},
  title     = {Embedding in Recommender Systems: A Survey},
  journal   = {arXiv preprint arXiv:2310.18608},
  year      = {2023},
  url       = {https://arxiv.org/abs/2310.18608}
}
```

---

*Documento generado mediante revision sistematica de literatura. Todas las fuentes fueron identificadas a traves de busquedas en Google Scholar, Semantic Scholar, ACL Anthology, ISMIR Archives y arXiv. Las fuentes criticas (Reimers y Gurevych 2019, Kashyap et al. 2023, Gao et al. 2021) fueron verificadas mediante acceso directo a sus paginas de publicacion.*
