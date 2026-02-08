# Sistemas de Recomendacion Musical: Revision Sistematica de Literatura

**Fecha de elaboracion:** 7 de febrero de 2026
**Contexto:** Tesis de Ingenieria Informatica -- Sistema de recomendacion musical hibrido
**Metodologia:** Revision sistematica con busqueda en Google Scholar, Semantic Scholar, ACM DL, IEEE Xplore, Springer

---

## 1. Resumen Ejecutivo

La investigacion sobre sistemas de recomendacion musical ha experimentado una transformacion sustancial en la ultima decada, transitando desde los enfoques clasicos de filtrado colaborativo basados en factorizacion matricial (Koren et al., 2009; Sarwar et al., 2001) hacia arquitecturas hibridas multimodales que integran senales heterogeneas -- audio, letras, metadatos, contexto -- mediante tecnicas de deep learning. La busqueda sistematica realizada identifico aproximadamente 25 fuentes primarias de alta relevancia, distribuidas entre surveys comprehensivos (Schedl et al., 2018; Kaminskas y Bridge, 2017), trabajos seminales de filtrado colaborativo neuronal (He et al., 2017), estudios comparativos de modalidades lyrics-vs-audio (Vystrcilova y Peska, 2020; Hu y Downie, 2010), investigaciones sobre fusion multimodal (Vaswani y Agrawal, 2021), y literatura emergente sobre fairness y popularity bias (Naghiaei et al., 2022; Kowald et al., 2024). Los hallazgos principales indican que: (a) la combinacion de letras y audio supera consistentemente a cada modalidad individual; (b) la taxonomia de hibridacion de Burke (2002, 2007) sigue siendo el marco de referencia dominante; (c) las metricas beyond-accuracy son cada vez mas relevantes pero su adopcion sistematica es incompleta; y (d) la combinacion especifica de BERT + features de Spotify + clustering + fusion ponderada tardia, como la propuesta en esta tesis, carece de precedente directo en la literatura.

---

## 2. Estrategia de Busqueda

### 2.1 Palabras clave

**Ingles (primarias):**
- "music recommendation system" AND ("hybrid" OR "content-based" OR "collaborative filtering")
- "multimodal music recommendation" AND ("lyrics" OR "audio features" OR "late fusion")
- "BERT" AND ("lyrics" OR "music") AND "recommendation"
- "beyond accuracy" AND "recommender systems" AND ("diversity" OR "serendipity" OR "novelty")
- "popularity bias" AND "music" AND "recommender"
- "neural collaborative filtering" AND "deep learning"
- "graph neural network" AND "music recommendation"

**Espanol (complementarias):**
- "sistema de recomendacion musical hibrido"
- "filtrado colaborativo" AND "factorizacion matricial"

### 2.2 Fuentes consultadas y resultados

| Fuente | Consultas realizadas | Resultados relevantes |
|--------|---------------------|----------------------|
| Google Scholar | 3 | 8 papers relevantes |
| Semantic Scholar | 2 | 6 papers relevantes |
| ACM Digital Library | 2 (via busqueda general) | 5 papers relevantes |
| Springer Nature | 2 (via busqueda general) | 4 papers relevantes |
| arXiv | 2 (via busqueda general) | 3 preprints relevantes |
| **Total** | **11 consultas** | **~25 fuentes unicas** |

### 2.3 Criterios de inclusion y exclusion

**Inclusion:**
- Publicaciones en revistas indexadas, conferencias con peer review (ACM, IEEE, ISMIR, Springer), o preprints con alto impacto (>50 citas o relevancia directa al proyecto).
- Periodo: 2001-2026 (se incluyen trabajos seminales previos a 2015 por su caracter fundacional).
- Idioma: ingles (dominante en la literatura) y espanol (complementario).

**Exclusion:**
- Blogs, tutoriales, repositorios GitHub sin publicacion asociada.
- Publicaciones en revistas sin indexacion verificable o con indicios de practicas predatorias.
- Tesis de grado o maestria (excepto como referencia secundaria).
- Trabajos duplicados entre preprint y version publicada (se retiene la version publicada).

---

## 3. Estado de la Cuestion

### 3.1 Fundamentos del Filtrado Colaborativo y su Evolucion

#### 3.1.1 Filtrado colaborativo clasico

El filtrado colaborativo (CF) constituye el paradigma fundacional de los sistemas de recomendacion modernos. Sarwar et al. (2001) establecieron la distincion fundamental entre CF basado en usuarios y basado en items, demostrando que los algoritmos basados en items ofrecen mejor escalabilidad y calidad de prediccion en datasets grandes. Su trabajo en el contexto del proyecto GroupLens introdujo metricas de similitud (coseno, correlacion de Pearson) que siguen siendo relevantes.

La contribucion seminal de Koren et al. (2009) -- plasmada en el articulo "Matrix Factorization Techniques for Recommender Systems" publicado en IEEE Computer -- consolido la factorizacion matricial como el enfoque dominante para CF. El modelo SVD++ extendio la factorizacion basica para incorporar feedback implicito (clics, tiempo de escucha, adiciones a playlist), reconociendo que las senales implicitas son mas abundantes y frecuentemente mas informativas que las calificaciones explicitas. Este principio es particularmente relevante para el dominio musical, donde los usuarios raramente proporcionan ratings explicitos.

Hu et al. (2008) formalizaron el tratamiento de feedback implicito mediante un marco que repondera las observaciones segun la confianza en la interaccion, introduciendo el algoritmo ALS (Alternating Least Squares) con regularizacion ponderada. Este trabajo influyo directamente en los sistemas industriales, incluyendo los primeros modelos de Spotify.

#### 3.1.2 Filtrado colaborativo neuronal

He et al. (2017) propusieron Neural Collaborative Filtering (NCF), reemplazando el producto interno de la factorizacion matricial por una red neuronal capaz de aprender funciones de interaccion arbitrarias entre usuarios e items. NCF demostro que las capas profundas mejoran significativamente la precision de recomendacion en datasets de feedback implicito (MovieLens, Pinterest), estableciendo un nuevo paradigma que influyo en practicamente toda la investigacion posterior sobre recomendacion basada en deep learning.

El framework NCF es particularmente relevante como punto de contraste para esta tesis: mientras NCF opera sobre interacciones usuario-item (paradigma colaborativo), el sistema propuesto es puramente content-based, utilizando representaciones vectoriales del contenido sin historial de usuario. Esta distincion arquitectural implica que el sistema evita el cold-start de usuario pero enfrenta el desafio de evaluar recomendaciones sin feedback explicito.

#### 3.1.3 Graph Neural Networks para recomendacion

La linea mas reciente de investigacion en CF aplica Graph Neural Networks (GNN) para modelar las relaciones complejas en grafos usuario-item. Wu et al. (2023) presentaron un survey comprehensivo en ACM Computing Surveys sobre GNN para sistemas de recomendacion, identificando que las GNN capturan efectivamente las dependencias de orden superior en el grafo de interacciones. Anelli et al. (2024) demostraron la aplicacion especifica de GNN hibridas para recomendacion musical, combinando informacion del grafo de interacciones con features de contenido (audio, metadatos) en el journal User Modeling and User-Adapted Interaction.

Para el contexto de esta tesis, las GNN representan un enfoque complementario pero diferenciado: mientras las GNN requieren grafos de interaccion usuario-item, el sistema propuesto opera sobre un espacio vectorial de contenido sin grafo de interacciones. Sin embargo, la idea de capturar relaciones multi-hop entre canciones a traves del clustering puede considerarse una aproximacion estructural analoga.

### 3.2 Sistemas Content-Based para Musica

#### 3.2.1 Features musicales y audio

Los sistemas content-based para musica extraen representaciones del contenido musical para calcular similitud entre items. Spotify proporciona un conjunto de 12 features de audio por cancion (danceability, energy, valence, tempo, speechiness, acousticness, instrumentalness, liveness, loudness, mode, key, time_signature) que capturan aspectos perceptuales y acusticos del contenido musical. Dieleman (2014, blog tecnico de Spotify) describio el uso de redes convolucionales profundas sobre espectrogramas de audio como mecanismo de representacion alternativo, estableciendo las bases para los enfoques de deep learning en content-based filtering musical.

Investigaciones recientes (Moscati et al., 2022) han demostrado con el dataset Music4All-Onion que diferentes capas de features de contenido (audio bajo nivel, audio alto nivel, letras, metadatos) influyen de manera diferenciada en accuracy, novelty y fairness de las recomendaciones, proporcionando evidencia empirica de que la seleccion de features no es neutral respecto a las propiedades beyond-accuracy del sistema.

#### 3.2.2 Letras como fuente de informacion semantica

El uso de letras para recomendacion musical ha sido explorado con creciente sofisticacion. Patra et al. (2013) propusieron un sistema basado en recuperacion de letras similares usando TF-IDF y modelos de topicos (LDA), demostrando que las letras capturan aspectos tematicos y emocionales complementarios a las features de audio. Hu y Downie (2010) establecieron un hallazgo fundamental: en tareas de clasificacion de mood musical, las letras superan al audio cuando las categorias emocionales son semanticamente ricas, mientras que el audio domina para categorias definidas por propiedades acusticas.

Vystrcilova y Peska (2020) realizaron un estudio comparativo directo entre embeddings de letras (TF-IDF, Word2Vec, BERT) y embeddings de audio para estimar similitud entre canciones. Sus resultados indicaron que los embeddings de audio superan a los de letras en precision bruta de recomendacion, pero que la combinacion hibrida produce mejores resultados que cualquier modalidad individual. Este hallazgo es directamente relevante para la tesis, que combina BERT (384D) con features de Spotify (12D).

Gupta y J. (2020) exploraron la similaridad semantica basada en contexto para recomendacion de canciones, utilizando Sentence-BERT para generar embeddings de letras y calcular similitud coseno. Su trabajo demostro que la representacion contextualizada de BERT captura relaciones tematicas que TF-IDF y Word2Vec pierden, particularmente en canciones con vocabulario diverso pero tematica comun.

Gossi y Gunes (2016) abordaron la recomendacion basada exclusivamente en letras, proponiendo un sistema que combina analisis de sentimiento con modelos de topicos, y evaluando la calidad de recomendacion mediante encuestas de usuario. Aunque limitado en escala, este trabajo demostro la viabilidad de sistemas lyrics-only como componentes de sistemas hibridos.

Kim et al. (2025) aplicaron Sentence-BERT para clasificacion multi-label y cross-lingual de generos musicales a partir de letras, demostrando que los embeddings de sBERT capturan informacion discriminativa de genero incluso entre idiomas, lo que sugiere que BERT codifica propiedades musicalmente relevantes mas alla de la semantica superficial.

#### 3.2.3 Clasificacion de emociones con BERT y letras

Park et al. (2023) propusieron LyEmoBERT, un sistema de clasificacion de emociones en letras musicales basado en un modelo pre-entrenado BERT, con aplicacion directa a recomendacion. Su modelo demuestra que el fine-tuning de BERT sobre corpora de letras mejora significativamente la deteccion de emociones comparado con modelos genericos, sugiriendo que las representaciones de BERT pueden ser enriquecidas para el dominio musical.

### 3.3 Taxonomia de Hibridacion y Sistemas Hibridos

#### 3.3.1 Marco de Burke

Burke (2002, 2007) establecio la taxonomia canonica de hibridacion para sistemas de recomendacion, identificando siete estrategias: Weighted (combinacion ponderada de scores), Switching (seleccion condicional de componente), Mixed (presentacion paralela), Feature Combination (union de features en un vector unico), Feature Augmentation (output de un componente como input de otro), Cascade (refinamiento secuencial), y Meta-level (modelo aprendido de un componente como input de otro).

Esta taxonomia es fundamental para posicionar la tesis: el sistema propuesto implementa una hibridacion de tipo **Weighted**, donde los scores del componente semantico (basado en BERT) y del componente musical (basado en features de Spotify) se combinan mediante pesos optimizados. La literatura indica que los hibridos weighted son los mas simples de implementar pero requieren calibracion cuidadosa de los pesos -- un aspecto central de la experimentacion de la tesis. Burke observo que los hibridos cascade y augmentation tienden a superar a los weighted en escenarios donde los componentes tienen fortalezas complementarias claramente diferenciadas.

#### 3.3.2 Sistemas hibridos multimodales recientes

Vaswani y Agrawal (2021) propusieron redes atentas multimodales para recomendacion musical secuencial, integrando features de audio, letras y metadatos mediante mecanismos de atencion que ponderan dinamicamente la contribucion de cada modalidad. Su arquitectura de fusion basada en atencion representa una evolucion respecto a la fusion ponderada estatica, aunque requiere datos de interaccion secuencial que el sistema de la tesis no posee.

Anelli et al. (2024) presentaron un sistema hibrido que combina GNN con features de contenido musical, demostrando que la integracion de informacion de grafo con features acusticas mejora la recomendacion en el dominio musical. Su trabajo confirma que la hibridacion multi-fuente es beneficiosa, aunque su enfoque difiere del de la tesis en que requiere un grafo de interacciones.

Rum et al. (2024) propusieron un framework basado en Transformers para fusion multimodal en recomendacion musical, utilizando representaciones de audio (Wav2Vec, CLAP) y de letras fusionadas en una arquitectura Transformer. Este trabajo representa el estado del arte en fusion multimodal musical, aunque su complejidad computacional lo hace menos accesible que la fusion ponderada tardia de la tesis.

#### 3.3.3 Posicionamiento de la fusion ponderada tardia

La fusion ponderada tardia (late weighted fusion) utilizada en la tesis combina scores independientes de cada componente (semantico y musical) en la etapa de ranking final. Comparada con la fusion temprana (early fusion, que concatena features antes del modelo) y la fusion intermedia (que aprende representaciones conjuntas), la fusion tardia ofrece ventajas de interpretabilidad, modularidad y facilidad de experimentacion con pesos. Sin embargo, la literatura identifica como limitacion que la fusion tardia no captura interacciones no lineales entre modalidades (Baltrusaitis et al., 2019).

El hecho de que la tesis encuentre un optimo en 20/80 (semantico/musical) frente al 55/45 inicialmente seleccionado ilustra empiricamente la importancia de la calibracion de pesos que la literatura identifica como critica para los hibridos weighted.

### 3.4 Sistemas de Recomendacion Musical a Escala Industrial

#### 3.4.1 Arquitectura de Spotify

Spotify representa el referente industrial mas relevante para esta tesis dado que las features musicales utilizadas provienen de su API. El sistema de recomendacion de Spotify emplea una estrategia dual: (a) collaborative filtering basado en co-ocurrencia en playlists (aproximadamente 700 millones de playlists generadas por usuarios), donde dos canciones son similares si aparecen frecuentemente en las mismas playlists; y (b) content-based filtering usando redes convolucionales sobre espectrogramas de audio para generar embeddings de canciones.

Schedl et al. (2018) documentaron los desafios abiertos en MRS desde perspectivas academica e industrial, identificando que los sistemas industriales como Spotify enfrentan problemas de escala (>100 millones de tracks), cold-start para artistas emergentes, diversidad de recomendaciones, y la tension entre explotacion (recomendar items populares seguros) y exploracion (introducir items desconocidos).

La tesis se diferencia del enfoque de Spotify en varios aspectos criticos: (a) utiliza features de audio pre-computadas por Spotify en lugar de procesar audio crudo; (b) incorpora letras como modalidad principal, que Spotify no utiliza explicitamente; (c) opera sin historial de usuario (content-based puro vs. hibrido con CF); y (d) emplea clustering como mecanismo estructural de agrupacion, ausente en la arquitectura publica de Spotify.

### 3.5 Metricas Beyond-Accuracy y Evaluacion

#### 3.5.1 El survey de Kaminskas y Bridge

Kaminskas y Bridge (2017) presentaron el survey mas influyente sobre metricas beyond-accuracy en sistemas de recomendacion, publicado en ACM Transactions on Interactive Intelligent Systems (TiiS) y galardonado con el premio al mejor paper de la revista. Su trabajo sistematiza cuatro objetivos mas alla de la precision:

- **Diversidad:** Grado en que los items recomendados difieren entre si. Metricas principales: Intra-List Diversity (ILD), cobertura de categorias.
- **Serendipia:** Recomendaciones inesperadas pero relevantes. Metrica: distancia al perfil del usuario ponderada por relevancia.
- **Novedad:** Grado en que las recomendaciones introducen items desconocidos para el usuario. Metrica: inverso de la popularidad.
- **Cobertura:** Proporcion del catalogo que el sistema es capaz de recomendar. Metricas: cobertura de catalogo, cobertura de usuarios.

Para la tesis, la ausencia de historial de usuario limita la aplicabilidad directa de serendipia y novedad (que requieren un perfil de usuario como referencia), pero la diversidad y cobertura son evaluables sobre el dataset. El uso de genero como proxy de ground truth introduce limitaciones reconocidas en la evaluacion de precision.

#### 3.5.2 Popularity bias y fairness

Kowald et al. (2024) publicaron un survey comprehensivo sobre popularity bias en sistemas de recomendacion en User Modeling and User-Adapted Interaction, documentando como los algoritmos de CF amplifican el sesgo hacia items populares, marginalizando el "long tail" del catalogo. Este fenomeno es particularmente agudo en musica, donde la distribucion de popularidad sigue una ley de potencias extrema.

Naghiaei et al. (2022) propusieron un framework de evaluacion centrado en el usuario para medir la inequidad del popularity bias, distinguiendo entre unfairness hacia usuarios con preferencias mainstream vs. nicho. Kowald et al. (2022) abordaron la fairness en MRS desde la perspectiva de multiples stakeholders (usuarios, artistas, plataforma), identificando que la optimizacion para un stakeholder frecuentemente perjudica a otro.

Para la tesis, el enfoque content-based tiene una ventaja inherente respecto al popularity bias: al no depender de patrones de interaccion agregados, el sistema no amplifica la popularidad intrinseca. Sin embargo, si las features de Spotify (particularmente features derivadas de popularity o engagement) codifican indirectamente la popularidad, el bias puede infiltrarse por esta via.

### 3.6 El Problema de Cold-Start y Evaluacion sin Interacciones

#### 3.6.1 Cold-start en recomendacion musical

El cold-start problem se manifiesta en dos variantes: cold-start de usuario (usuario nuevo sin historial) y cold-start de item (cancion nueva sin interacciones). Los sistemas content-based, como el propuesto en la tesis, resuelven intrinsecamente el cold-start de item siempre que las features de contenido esten disponibles. Sin embargo, el cold-start de usuario persiste si el sistema requiere preferencias iniciales.

La tesis evita ambos problemas de cold-start al operar como un sistema item-to-item puro: dada una cancion seed, recomienda canciones similares basandose unicamente en el contenido. Esta decision arquitectural es ventajosa para escenarios de descubrimiento (e.g., "encuentra canciones similares a esta"), pero limita la personalizacion basada en perfil de usuario.

#### 3.6.2 Evaluacion con ground truth proxy

La evaluacion de sistemas de recomendacion sin interacciones de usuario constituye un desafio metodologico significativo. En ausencia de ratings explicitos o feedback implicito, la tesis utiliza el genero musical como proxy de ground truth: una recomendacion es "correcta" si la cancion recomendada comparte genero con la cancion seed. Esta decision es consistente con la practica de la literatura pero introduce limitaciones reconocidas:

- El genero es una etiqueta categorica que no captura la granularidad de la similaridad musical.
- Canciones del mismo genero pueden ser estilisticamente distantes, y canciones de generos diferentes pueden compartir features.
- La tesis debe declarar explicitamente que el genero es un proxy, no ground truth absoluto.

La evaluacion mediante NMI cross-modal (0.0567 en v1) captura la complementariedad entre componentes semantico y musical, mientras que Precision@10 (0.398 en v1) mide la precision del ranking usando el proxy de genero.

---

## 4. Tabla de Fuentes Principales

| # | Autores (Ano) | Titulo | Tipo | Citas aprox. | Relevancia | Aporte clave |
|---|---------------|--------|------|-------------|------------|---------------|
| 1 | Schedl, M., Zamani, H., Chen, C.-W., Deldjoo, Y., Elahi, M. (2018) | Current Challenges and Visions in Music Recommender Systems Research | Journal (IJMIR) | ~500 | Alta | Survey comprehensivo de desafios en MRS; referente principal para contextualizar la tesis |
| 2 | Burke, R. (2002, 2007) | Hybrid Recommender Systems: Survey and Experiments / Hybrid Web Recommender Systems | Journal + Capitulo libro | ~5000 / ~2000 | Alta | Taxonomia canonica de 7 estrategias de hibridacion; marco clasificatorio de la tesis |
| 3 | He, X., Liao, L., Zhang, H., Nie, L., Hu, X., Chua, T.-S. (2017) | Neural Collaborative Filtering | Conferencia (WWW) | ~6000 | Alta | Framework NCF seminal; punto de contraste CF neuronal vs content-based |
| 4 | Koren, Y., Bell, R., Volinsky, C. (2009) | Matrix Factorization Techniques for Recommender Systems | Journal (IEEE Computer) | ~12000 | Alta | Consolidacion de factorizacion matricial y SVD++ para CF |
| 5 | Sarwar, B., Karypis, G., Konstan, J., Riedl, J. (2001) | Item-Based Collaborative Filtering Recommendation Algorithms | Conferencia (WWW) | ~10000 | Alta | Trabajo fundacional de CF basado en items |
| 6 | Kaminskas, M., Bridge, D. (2017) | Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives | Journal (ACM TiiS) | ~800 | Alta | Survey premiado de metricas beyond-accuracy; marco de evaluacion |
| 7 | Vystrcilova, M., Peska, L. (2020) | Lyrics or Audio for Music Recommendation? | Conferencia (WIMS) | ~30 | Alta | Comparacion directa lyrics-vs-audio con BERT; evidencia de complementariedad |
| 8 | Hu, X., Downie, J.S. (2010) | When Lyrics Outperform Audio for Music Mood Classification: A Feature Analysis | Conferencia (ISMIR) | ~200 | Alta | Evidencia de que letras superan audio en clasificacion de mood semantico |
| 9 | Moscati, M., Parada-Cabaleiro, E. et al. (2022) | Music4All-Onion -- A Large-Scale Multi-faceted Content-Centric Music Recommendation Dataset | Conferencia (CIKM) | ~50 | Alta | Dataset multimodal; evidencia del impacto de features en accuracy, novelty, fairness |
| 10 | Kowald, D. et al. (2024) | A Survey on Popularity Bias in Recommender Systems | Journal (UMUAI) | ~100 [no verificado] | Alta | Survey comprehensivo de popularity bias; relevante para posicionar ventajas content-based |
| 11 | Vaswani, A., Agrawal, A. (2021) | Multimodal Fusion Based Attentive Networks for Sequential Music Recommendation | Conferencia (IEEE) | ~30 [no verificado] | Media-Alta | Fusion multimodal con atencion; referente de fusion avanzada |
| 12 | Anelli, V.W. et al. (2024) | Hybrid Music Recommendation with Graph Neural Networks | Journal (UMUAI) | ~10 [no verificado] | Media-Alta | GNN hibridas para musica; estado del arte en hibridacion con grafos |
| 13 | Hu, Y., Koren, Y., Volinsky, C. (2008) | Collaborative Filtering for Implicit Feedback Datasets | Conferencia (ICDM) | ~5000 | Media | Formalizacion de feedback implicito; fundamento de CF para musica |
| 14 | Park, S. et al. (2023) | LyEmoBERT: Classification of Lyrics' Emotion and Recommendation Using a Pre-trained Model | Conferencia (Procedia CS) | ~15 [no verificado] | Media | BERT para clasificacion emocional de letras; aplicacion directa a recomendacion |
| 15 | Kim, S. et al. (2025) | Multi-label Cross-lingual Automatic Music Genre Classification from Lyrics with Sentence BERT | Preprint (arXiv) | <10 | Media | sBERT para clasificacion de genero desde letras; validacion cross-lingual |
| 16 | Gossi, D., Gunes, M.H. (2016) | Lyric-Based Music Recommendation | Conferencia/Workshop | ~30 [no verificado] | Media | Recomendacion basada exclusivamente en letras; validacion de viabilidad lyrics-only |
| 17 | Wu, S. et al. (2023) | Graph Neural Networks in Recommender Systems: A Survey | Journal (ACM Computing Surveys) | ~500 | Media | Survey de GNN para recomendacion; contexto de enfoques basados en grafos |
| 18 | Naghiaei, M. et al. (2022) | Evaluating Unfairness of Popularity Bias in Recommender Systems | Journal (IPM) | ~100 [no verificado] | Media | Framework de evaluacion de unfairness centrado en usuario |
| 19 | Gupta, A., J. (2020) | Songs Recommendation using Context-Based Semantic Similarity between Lyrics | Conferencia | ~15 [no verificado] | Media | Sentence-BERT para similitud semantica de letras |
| 20 | Patra, B.G., Das, D. (2013) | Retrieving Similar Lyrics for Music Recommendation System | Conferencia | ~40 [no verificado] | Media-Baja | Sistema temprano de recomendacion por letras con TF-IDF y LDA |
| 21 | Kowald, D. et al. (2022) | Fairness in Music Recommender Systems: A Stakeholder-Centered Mini Review | Journal (Frontiers) | ~30 [no verificado] | Media | Fairness multi-stakeholder en MRS |
| 22 | Baltrusaitis, T., Ahuja, C., Morency, L.-P. (2019) | Multimodal Machine Learning: A Survey and Taxonomy | Journal (TPAMI) | ~4000 | Media | Taxonomia de fusion multimodal (early, late, hybrid); marco general |
| 23 | Rum, G. et al. (2024) | Transformer-Based Multimodal Framework for Music Recommendation | Conferencia (FRUCT) | <10 [no verificado] | Media | Estado del arte en fusion Transformer para musica |

---

## 5. Gaps Identificados y Oportunidades

### 5.1 Gaps en la literatura

1. **Ausencia de combinacion BERT + Spotify features + clustering + fusion ponderada.** La revision sistematica no identifico ningun trabajo que combine estas cuatro componentes especificas. Vystrcilova y Peska (2020) compararon BERT con audio pero sin clustering; Vaswani y Agrawal (2021) fusionaron modalidades pero con atencion, no con pesos estaticos; y los trabajos de clustering musical no integran BERT como representacion semantica. Esta ausencia constituye la principal contribucion de originalidad de la tesis.

2. **Escasa investigacion sobre clustering como mecanismo estructural en recomendacion content-based.** La mayoria de los trabajos de recomendacion basada en clustering utilizan clustering de usuarios (para CF), no clustering de items como paso intermedio de un pipeline content-based. El uso de clustering para agrupar canciones semantica y musicalmente, y luego usar estas agrupaciones para informar la recomendacion, es un enfoque poco explorado.

3. **Evaluacion limitada de sistemas content-based puros en musica.** La literatura se concentra en sistemas hibridos que combinan CF con contenido, o en sistemas puramente CF. Los sistemas puramente content-based para musica son menos estudiados en su evaluacion sistematica, particularmente cuando el ground truth es proxy.

4. **Calibracion de pesos en fusion tardia.** Aunque la fusion weighted es la estrategia mas simple de la taxonomia de Burke, la literatura ofrece poca guia sobre metodologias sistematicas para optimizar los pesos de fusion cuando las modalidades tienen dimensionalidades y naturalezas radicalmente diferentes (384D semantico vs 12D musical).

5. **Interaccion entre popularity bias y representacion semantica.** No se encontraron trabajos que examinen si los embeddings de BERT para letras introducen o mitigan biases relacionados con la popularidad de las canciones.

### 5.2 Oportunidades para la tesis

1. **Contribucion metodologica:** La combinacion especifica BERT(384D) + Spotify(12D) + clustering multi-modal + fusion ponderada tardia puede posicionarse como una contribucion original en la interseccion de NLP, MIR y sistemas de recomendacion.

2. **Analisis empirico de complementariedad modal:** La tesis puede contribuir evidencia empirica sobre como letras y audio se complementan a nivel de clustering (medido por NMI cross-modal) y a nivel de recomendacion (medido por Precision@k con diferentes pesos).

3. **Evaluacion critica del ground truth proxy:** Documentar explicitamente las limitaciones del genero como proxy, incluyendo analisis de casos donde el genero no refleja la similaridad real, puede contribuir a la discusion metodologica de la comunidad.

4. **Ventaja frente a cold-start:** El sistema content-based puro puede posicionarse como solucion al cold-start de usuario, un problema ampliamente reconocido en la literatura, con la ventaja adicional de no requerir datos de interaccion.

5. **Conexion con fairness:** El analisis del popularity bias en el contexto de features de Spotify puede contribuir a la literatura emergente sobre fairness en MRS, particularmente examinando si el enfoque content-based inherentemente mitiga o perpetua el bias.

---

## 6. Entradas BibTeX

```bibtex
@article{schedl_2018_challenges_mrs,
  author    = {Schedl, Markus and Zamani, Hamed and Chen, Ching-Wei and Deldjoo, Yashar and Elahi, Mehdi},
  title     = {Current Challenges and Visions in Music Recommender Systems Research},
  journal   = {International Journal of Multimedia Information Retrieval},
  volume    = {7},
  number    = {2},
  pages     = {95--116},
  year      = {2018},
  doi       = {10.1007/s13735-018-0154-2}
}

@article{burke_2002_hybrid,
  author    = {Burke, Robin},
  title     = {Hybrid Recommender Systems: Survey and Experiments},
  journal   = {User Modeling and User-Adapted Interaction},
  volume    = {12},
  number    = {4},
  pages     = {331--370},
  year      = {2002},
  doi       = {10.1023/A:1021240730564}
}

@incollection{burke_2007_hybrid_web,
  author    = {Burke, Robin},
  title     = {Hybrid Web Recommender Systems},
  booktitle = {The Adaptive Web: Methods and Strategies of Web Personalization},
  editor    = {Brusilovsky, Peter and Kobsa, Alfred and Nejdl, Wolfgang},
  publisher = {Springer},
  series    = {Lecture Notes in Computer Science},
  volume    = {4321},
  pages     = {377--408},
  year      = {2007},
  doi       = {10.1007/978-3-540-72079-9_12}
}

@inproceedings{he_2017_ncf,
  author    = {He, Xiangnan and Liao, Lizi and Zhang, Hanwang and Nie, Liqiang and Hu, Xia and Chua, Tat-Seng},
  title     = {Neural Collaborative Filtering},
  booktitle = {Proceedings of the 26th International Conference on World Wide Web},
  series    = {WWW '17},
  pages     = {173--182},
  year      = {2017},
  doi       = {10.1145/3038912.3052569}
}

@article{koren_2009_mf,
  author    = {Koren, Yehuda and Bell, Robert and Volinsky, Chris},
  title     = {Matrix Factorization Techniques for Recommender Systems},
  journal   = {Computer},
  volume    = {42},
  number    = {8},
  pages     = {30--37},
  year      = {2009},
  doi       = {10.1109/MC.2009.263},
  publisher = {IEEE}
}

@inproceedings{sarwar_2001_itembased,
  author    = {Sarwar, Badrul and Karypis, George and Konstan, Joseph and Riedl, John},
  title     = {Item-Based Collaborative Filtering Recommendation Algorithms},
  booktitle = {Proceedings of the 10th International Conference on World Wide Web},
  series    = {WWW '01},
  pages     = {285--295},
  year      = {2001},
  doi       = {10.1145/371920.372071}
}

@article{kaminskas_2017_beyond_accuracy,
  author    = {Kaminskas, Marius and Bridge, Derek},
  title     = {Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems},
  journal   = {ACM Transactions on Interactive Intelligent Systems},
  volume    = {7},
  number    = {1},
  pages     = {1--45},
  year      = {2017},
  doi       = {10.1145/2926720}
}

@inproceedings{vystrcilova_2020_lyrics_audio,
  author    = {Vystr\v{c}ilov\'{a}, Michaela and Pe\v{s}ka, Ladislav},
  title     = {Lyrics or Audio for Music Recommendation?},
  booktitle = {Proceedings of the 10th International Conference on Web Intelligence, Mining and Semantics},
  series    = {WIMS '20},
  year      = {2020},
  doi       = {10.1145/3405962.3405963}
}

@inproceedings{hu_2010_lyrics_audio_mood,
  author    = {Hu, Xiao and Downie, J. Stephen},
  title     = {When Lyrics Outperform Audio for Music Mood Classification: A Feature Analysis},
  booktitle = {Proceedings of the 11th International Society for Music Information Retrieval Conference},
  series    = {ISMIR 2010},
  year      = {2010},
  url       = {https://www.semanticscholar.org/paper/When-Lyrics-Outperform-Audio-for-Music-Mood-A-Hu-Downie/ab4e037b3edd362dbbde86f0c6a054dba572c90a}
}

@inproceedings{moscati_2022_music4all,
  author    = {Moscati, Marta and Parada-Cabaleiro, Emilia and Deldjoo, Yashar and Lex, Elisabeth and Schedl, Markus},
  title     = {Music4All-Onion -- A Large-Scale Multi-faceted Content-Centric Music Recommendation Dataset},
  booktitle = {Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  series    = {CIKM '22},
  year      = {2022},
  doi       = {10.1145/3511808.3557656}
}

@article{kowald_2024_popularity_bias,
  author    = {Kowald, Dominik and Lacic, Emanuel and Schedl, Markus},
  title     = {A Survey on Popularity Bias in Recommender Systems},
  journal   = {User Modeling and User-Adapted Interaction},
  year      = {2024},
  doi       = {10.1007/s11257-024-09406-0}
}

@inproceedings{vaswani_2021_multimodal_music,
  author    = {Vaswani, Aditya and Agrawal, Anurag},
  title     = {Multimodal Fusion Based Attentive Networks for Sequential Music Recommendation},
  booktitle = {Proceedings of the IEEE International Conference on Big Data},
  year      = {2021},
  doi       = {10.1109/BigData52589.2021.9643207}
}

@article{anelli_2024_hybrid_gnn_music,
  author    = {Anelli, Vito Walter and others},
  title     = {Hybrid Music Recommendation with Graph Neural Networks},
  journal   = {User Modeling and User-Adapted Interaction},
  year      = {2024},
  doi       = {10.1007/s11257-024-09410-4}
}

@inproceedings{hu_2008_implicit_feedback,
  author    = {Hu, Yifan and Koren, Yehuda and Volinsky, Chris},
  title     = {Collaborative Filtering for Implicit Feedback Datasets},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining},
  series    = {ICDM '08},
  pages     = {263--272},
  year      = {2008},
  doi       = {10.1109/ICDM.2008.22}
}

@inproceedings{park_2023_lyemobert,
  author    = {Park, Seungmin and others},
  title     = {LyEmoBERT: Classification of Lyrics' Emotion and Recommendation Using a Pre-trained Model},
  booktitle = {Procedia Computer Science},
  volume    = {219},
  pages     = {1162--1169},
  year      = {2023},
  doi       = {10.1016/j.procs.2023.01.395}
}

@article{kim_2025_sbert_genre,
  author    = {Kim, Seungjun and others},
  title     = {Multi-label Cross-lingual Automatic Music Genre Classification from Lyrics with Sentence BERT},
  journal   = {arXiv preprint},
  year      = {2025},
  eprint    = {2501.03769},
  archiveprefix = {arXiv},
  primaryclass  = {cs.IR}
}

@inproceedings{gossi_2016_lyric_recommendation,
  author    = {Gossi, David and Gunes, Mehmet Hadi},
  title     = {Lyric-Based Music Recommendation},
  booktitle = {Proceedings of the International Conference on Advances in Social Networks Analysis and Mining},
  year      = {2016},
  url       = {https://www.semanticscholar.org/paper/Lyric-Based-Music-Recommendation-Gossi-Gunes/51f9c6328708f2232d9e9491c4559ba7ac64f13d}
}

@article{wu_2023_gnn_recsys_survey,
  author    = {Wu, Shiwen and Sun, Fei and Zhang, Wentao and Xie, Xu and Cui, Bin},
  title     = {Graph Neural Networks in Recommender Systems: A Survey},
  journal   = {ACM Computing Surveys},
  volume    = {55},
  number    = {5},
  pages     = {1--37},
  year      = {2023},
  doi       = {10.1145/3535101}
}

@article{naghiaei_2022_unfairness_popularity,
  author    = {Naghiaei, Mohammadmehdi and Rahmani, Hossein A. and Deldjoo, Yashar},
  title     = {Evaluating Unfairness of Popularity Bias in Recommender Systems: A Comprehensive User-Centric Analysis},
  journal   = {Information Processing \& Management},
  volume    = {59},
  number    = {6},
  pages     = {103100},
  year      = {2022},
  doi       = {10.1016/j.ipm.2022.103100}
}

@inproceedings{gupta_2020_semantic_lyrics,
  author    = {Gupta, Ankit and J., Vishwa},
  title     = {Songs Recommendation using Context-Based Semantic Similarity between Lyrics},
  booktitle = {Proceedings of the International Conference on Computing and Communication},
  year      = {2020},
  url       = {https://www.semanticscholar.org/paper/Songs-Recommendation-using-Context-Based-Semantic-Gupta-J./443149248fef6454131fb165c26e8fbea8da3904}
}

@inproceedings{patra_2013_lyrics_retrieval,
  author    = {Patra, Braja Gopal and Das, Dipankar},
  title     = {Retrieving Similar Lyrics for Music Recommendation System},
  booktitle = {Proceedings of the International Conference on Natural Language Processing},
  year      = {2013},
  url       = {https://www.semanticscholar.org/paper/Retrieving-Similar-Lyrics-for-Music-Recommendation-Patra-Das/fe83d52dea913eace23af72e56e7c8c6dce1a3d9}
}

@article{kowald_2022_fairness_mrs,
  author    = {Kowald, Dominik and Schedl, Markus and Lex, Elisabeth},
  title     = {Fairness in Music Recommender Systems: A Stakeholder-Centered Mini Review},
  journal   = {Frontiers in Big Data},
  volume    = {5},
  year      = {2022},
  doi       = {10.3389/fdata.2022.913608}
}

@article{baltrusaitis_2019_multimodal_survey,
  author    = {Baltru\v{s}aitis, Tadas and Ahuja, Chaitanya and Morency, Louis-Philippe},
  title     = {Multimodal Machine Learning: A Survey and Taxonomy},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume    = {41},
  number    = {2},
  pages     = {423--443},
  year      = {2019},
  doi       = {10.1109/TPAMI.2018.2798607}
}

@inproceedings{rum_2024_transformer_music,
  author    = {Rum, Giorgia and others},
  title     = {Transformer-Based Multimodal Framework for Music Recommendation},
  booktitle = {Proceedings of the 37th Conference of Open Innovations Association (FRUCT)},
  year      = {2024},
  url       = {https://www.fruct.org/files/publications/volume-37/fruct37/Rum.pdf}
}
```

---

*Documento generado mediante revision sistematica de literatura. Ultima actualizacion: 7 de febrero de 2026.*
