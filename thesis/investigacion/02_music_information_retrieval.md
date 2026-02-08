# Music Information Retrieval: Historia, Evolucin y Enfoques Multimodales

## 1. Resumen ejecutivo

Music Information Retrieval (MIR) constituye un campo interdisciplinario que ha experimentado una transformacion profunda desde su formalizacion por Downie (2003) hasta los enfoques multimodales basados en deep learning de la decada de 2020. La revision sistematica realizada identifico aproximadamente 25 fuentes de alta relevancia, abarcando surveys seminales, estudios empiricos sobre la validez de features acusticas, criticas fundamentales a benchmarks estandar como GTZAN, y trabajos recientes sobre fusion multimodal de audio y letras. Las principales conclusiones son: (1) el campo ha transitado de representaciones artesanales (hand-crafted features) hacia embeddings aprendidos mediante redes neuronales profundas; (2) la clasificacion de genero musical enfrenta limitaciones intrinsecas derivadas de la subjetividad y naturaleza multi-label del concepto de genero; (3) las features de Spotify presentan correlaciones variables con la percepcion humana, siendo energy la mas fiable y danceability la menos validada; (4) la fusion multimodal (audio + letras) supera consistentemente los enfoques unimodales, con mejoras de hasta 9 puntos porcentuales en clasificacion emocional. Estos hallazgos fundamentan directamente las decisiones arquitectonicas de la tesis: el uso de embeddings BERT para letras, features de Spotify como representacion musical, y la necesidad de documentar las limitaciones del genero como ground truth proxy.

---

## 2. Estrategia de busqueda

### Palabras clave utilizadas

**Ingles (busqueda primaria):**
- "Music Information Retrieval" + survey + deep learning
- GTZAN dataset + limitations + Sturm + "Clever Hans"
- Music emotion recognition + valence + arousal + Russell circumplex
- Spotify audio features + correlation + human perception + danceability
- Lyrics analysis + NLP + BERT + Word2Vec + evolution
- Multimodal music + audio + lyrics + fusion + deep learning
- Music genre classification + subjectivity + multi-label + limitations
- Million Song Dataset + FMA + WASABI + music dataset
- Essentia + librosa + audio feature extraction
- Downie 2003 + ISMIR + music information retrieval

### Fuentes consultadas y resultados relevantes

| Fuente | Busquedas realizadas | Resultados relevantes |
|--------|---------------------|----------------------|
| Google Scholar | 3 | 8 papers seminales |
| Semantic Scholar | 2 (via resultados) | 4 papers con grafos de citas |
| ACM Digital Library | 2 (via resultados) | 3 papers (TISMIR, ACM TOMM) |
| IEEE Xplore | 1 (via resultados) | 1 paper (Sturm 2014) |
| arXiv | 3 (via resultados) | 4 preprints relevantes |
| PLoS ONE | 1 | 1 paper verificado |
| Springer | 2 (via resultados) | 3 papers (WASABI, surveys) |
| ISMIR Proceedings | 2 (via resultados) | 2 papers |

### Criterios de inclusion/exclusion

**Inclusion:**
- Publicaciones 2003-2026, revisadas por pares o preprints de alto impacto (>50 citas o publicacion reciente en venue de prestigio).
- Relevancia directa a: MIR como campo, audio features, clasificacion de genero, emotion recognition, analisis de letras con NLP, datasets musicales, o fusion multimodal.
- Idioma: ingles (predominante en la literatura MIR).

**Exclusion:**
- Blogs, tutoriales, publicaciones sin autor identificable.
- Publicaciones predatorias o sin revision por pares (excepto preprints en arXiv de grupos reconocidos).
- Trabajos enfocados exclusivamente en generacion musical o composicion algoritmica.

---

## 3. Estado de la cuestion

### 3.1. Origen y evolucion del campo MIR

#### 3.1.1. Formalizacion del campo (Downie, 2003)

Music Information Retrieval emerge como campo academico formal con la publicacion seminal de Downie (2003) en el Annual Review of Information Science and Technology, donde se define MIR como un area interdisciplinaria surgida de "la necesidad de gestionar colecciones crecientes de musica en formato digital". Downie identifico los desafios fundamentales que el campo enfrentaria durante las dos decadas siguientes: la ausencia de medios comunmente aceptados para comparar tecnicas de recuperacion, la escasa investigacion con usuarios reales de sistemas MIR, y la necesidad de colecciones de referencia estandarizadas.

La conferencia ISMIR (International Society for Music Information Retrieval), establecida en 2000, se consolido rapidamente como el venue central del campo. Como senala el editorial de lanzamiento de Transactions of ISMIR (Serra et al., 2018), la comunidad ISMIR integra investigadores de ciencias de la computacion, procesamiento de senales, musicologia, psicologia cognitiva y bibliotecologia, reflejando la naturaleza inherentemente multidisciplinaria de los problemas MIR.

#### 3.1.2. Decada de maduracion (2004-2014)

Schedl, Gomez y Urbano (2014) publicaron un survey comprehensivo en Foundations and Trends in Information Retrieval que documenta el estado del arte una decada despues de Downie. Los autores identifican la transicion desde enfoques basados exclusivamente en contenido acustico hacia sistemas que integran informacion contextual (metadata, tags sociales, informacion editorial). Este periodo se caracteriza por el predominio de features artesanales (MFCCs, chroma features, spectral contrast) combinadas con clasificadores como SVM y Random Forest.

Un hito critico de este periodo es la publicacion de Sturm (2014) sobre el problema del "Clever Hans" en MIR (vease seccion 3.3.2), que cuestiono fundamentalmente la validez de las evaluaciones realizadas hasta entonces y forzo una reflexion metodologica profunda en la comunidad.

#### 3.1.3. Era del deep learning (2015-presente)

La adopcion de deep learning transformo radicalmente el campo. Las redes neuronales convolucionales (CNN) aplicadas directamente sobre espectrogramas (mel-spectrograms, constant-Q transforms) eliminaron la necesidad de ingenieria manual de features. Mas recientemente, los modelos Transformer han permitido capturar dependencias temporales de largo alcance en senales musicales, mientras que modelos de lenguaje pre-entrenados como BERT han revolucionado el analisis de letras.

El survey de Liyanarachchi, Joshi y Meijering (2025) documenta esta evolucion en el contexto de music emotion recognition, mostrando que los metodos han progresado "desde tecnicas tradicionales de machine learning como SVMs hacia modelos mas sofisticados basados en CNNs, RNNs (particularmente LSTMs) y recientemente Transformers". La precision maxima reportada alcanza 94.58% mediante la combinacion de CNN para audio y BERT para letras con late fusion.

### 3.2. Audio features y su correlacion con la percepcion humana

#### 3.2.1. Features acusticas: taxonomia y herramientas de extraccion

Las features acusticas utilizadas en MIR se clasifican en varias categorias: timbricas (MFCCs, spectral centroid, spectral rolloff), ritmicas (tempo, beat strength, onset rate), tonales (chroma features, key, mode) y de alto nivel (energy, danceability, valence). Las dos herramientas principales para su extraccion son:

**Essentia** (Bogdanov et al., 2013): Biblioteca desarrollada en C++ con wrapper Python por el Music Technology Group de la Universitat Pompeu Fabra. Contiene "una extensa coleccion de algoritmos incluyendo entrada/salida de audio, bloques estandar de procesamiento digital de senales, caracterizacion estadistica de datos, y una gran variedad de descriptores espectrales, temporales, tonales y de alto nivel". Essentia destaca por su velocidad computacional y bajo uso de memoria, ademas de incluir herramientas para inferencia con modelos de deep learning.

**librosa** (McFee et al., 2015): Biblioteca implementada en Python puro basada en NumPy y SciPy. Aunque computacionalmente mas lenta que Essentia (debido a su implementacion en Python y dependencia de la FFT de SciPy), librosa es ampliamente utilizada en la comunidad MIR por su API intuitiva y facilidad de prototipado rapido. Permite extraer MFCCs, spectral contrast, chroma features y otras representaciones de uso comun.

La comparacion entre ambas herramientas revela diferencias significativas incluso en features aparentemente equivalentes. Por ejemplo, la implementacion de MFCCs difiere en parametros de ventaneo, normalizacion y numero de coeficientes por defecto, lo cual tiene implicaciones para la reproducibilidad de resultados.

#### 3.2.2. Features de Spotify: validez y limitaciones

Las features de alto nivel proporcionadas por la API de Spotify (energy, valence, danceability, acousticness, instrumentalness, speechiness, liveness, tempo, loudness, mode, key, duration) constituyen la representacion musical de 12 dimensiones utilizada en esta tesis. Su validez como representacion de la percepcion musical humana ha sido objeto de investigacion reciente.

**Validacion empirica de Vidas et al. (2025):** Este estudio comparo las puntuaciones automatizadas de Spotify con evaluaciones subjetivas de 244 oyentes reales que calificaron extractos musicales en dimensiones de mood, energy, danceability, familiaridad y disfrute. Los resultados revelan correlaciones heterogeneas:

- **Energy/Arousal**: Relacion "fuerte" entre las puntuaciones de Spotify y las evaluaciones humanas. Esta es la feature mas fiable de la API.
- **Valence**: Relacion "moderada" con las evaluaciones subjetivas de mood. Util pero con reservas.
- **Danceability**: "No fuertemente asociada con las evaluaciones humanas de danceability". Es la feature menos validada de las tres examinadas.

Los autores concluyen que estas medidas automatizadas pueden servir como herramientas de investigacion "si se usan con cautela", pero advierten sobre las limitaciones de transparencia (Spotify no publica los algoritmos exactos de calculo).

**Estudio de Duman et al. (2022):** Publicado en PLoS ONE, este trabajo examino las features de Spotify en el contexto de musica para movimiento. Confirmaron que la musica de baile exhibe niveles significativamente mas altos de energy, danceability, valence y loudness, con tamanios de efecto medianos. Criticamente, los autores senalan que "dado que Spotify no proporciona descripciones extensas de como se calculan estas features... el significado exacto y metodo de calculo de estas features es debatible". Ademas, observan que las "audio features" de Spotify no son puramente propiedades acusticas: algunas reflejan uso musical mas que caracteristicas medibles del sonido.

**Correlaciones inter-features:** La investigacion muestra que valence y danceability presentan alta correlacion entre si, lo cual tiene implicaciones para sistemas que usan ambas como dimensiones independientes. Esta correlacion sugiere redundancia parcial en el espacio de features de Spotify.

**Implicaciones para la tesis:** Estos hallazgos fundamentan la decision de usar las 12 features de Spotify como representacion musical, pero obligan a documentar explicitamente que: (1) danceability es la feature menos fiable; (2) existen correlaciones inter-features que reducen la dimensionalidad efectiva; (3) la opacidad algoritmica de Spotify introduce incertidumbre sistematica.

### 3.3. Clasificacion de genero musical: logros y limitaciones fundamentales

#### 3.3.1. El genero como categoria: subjetividad, multi-label y evolucion temporal

La clasificacion automatica de genero musical (Music Genre Recognition, MGR) es una de las tareas mas estudiadas en MIR, pero tambien una de las mas problematicas desde el punto de vista conceptual. Como documentan Sturm (2013) y la revision de Choi et al. (2019), no existe "una definicion clara y formal de que es el genero", y las categorizaciones musicales "son vagas y poco claras, sufriendo de subjetividad humana y falta de acuerdo".

**Subjetividad inherente:** Las interpretaciones de genero musical estan sujetas a diferentes percepciones, opiniones y experiencias personales. Un mismo tema puede ser clasificado como "rock alternativo", "indie rock" o "post-punk revival" dependiendo del contexto cultural, la epoca y el oyente. Esta ambiguedad no es un defecto del sistema de clasificacion sino una propiedad intrinseca del concepto de genero musical.

**Naturaleza multi-label:** La mayor parte de los esfuerzos en MGR se han centrado en clasificacion single-label, con "escaso trabajo en la tarea de clasificacion multi-label" (Choi et al., 2019). Sin embargo, la musica contemporanea desafia fundamentalmente la asuncion de etiqueta unica: un artista como Radiohead puede pertenecer simultaneamente a rock alternativo, art rock, electronic y experimental. La clasificacion multi-label representa mas fielmente la realidad del consumo musical pero introduce complejidad adicional en la evaluacion, particularmente cuando se usa genero como ground truth proxy.

**Evolucion temporal:** Los generos musicales no son categorias estaticas. Nuevos generos emergen continuamente (vaporwave, hyperpop, bedroom pop), mientras que generos existentes se transforman (la evolucion del hip-hop desde los anos 80 hasta el trap contemporaneo). Esta dinamica temporal implica que un clasificador entrenado en datos de una epoca puede no generalizar a otra, un problema raramente abordado en la literatura MGR.

#### 3.3.2. El problema GTZAN y el efecto "Clever Hans"

**GTZAN como benchmark accidental:** El dataset GTZAN, creado por Tzanetakis y Cook (2002), contiene 1,000 clips de audio de 30 segundos distribuidos en 10 generos. A pesar de no haber sido disenado como benchmark formal, aparece en "al menos 100 trabajos publicados" y es "el dataset publico mas utilizado para evaluacion en investigacion de escucha automatica para reconocimiento de genero musical" (Sturm, 2013).

**Fallos documentados (Sturm, 2013):** El analisis sistematico de Sturm (2013), publicado como arXiv:1306.1461 y posteriormente en el Journal of Music Technology and Education, identifico fallos criticos en GTZAN:
- **Repeticiones**: Clips duplicados o cuasi-duplicados presentes en el dataset.
- **Etiquetado erroneo**: Canciones asignadas a generos incorrectos.
- **Distorsiones**: Algunos archivos corrompidos hasta el punto de ser irreconocibles.
- **Sobrerepresentacion de artistas**: Ciertos artistas contribuyen multiples clips, introduciendo sesgo de artista en lugar de sesgo de genero.

Crucialmente, Sturm demostro que estos fallos NO afectan a todos los sistemas MGR de la misma manera, refutando el argumento comun de que "las comparaciones siguen siendo validas porque todos enfrentan los mismos fallos". Su conclusion no es que GTZAN deba abandonarse, sino que debe usarse "con consideracion de sus contenidos".

**El efecto "Clever Hans" (Sturm, 2014):** En un trabajo posterior publicado en IEEE Transactions on Multimedia, Sturm (2014) introdujo el concepto de "horse" (caballo) en MIR, en referencia al famoso caballo Clever Hans que aparentaba resolver problemas aritmeticos pero en realidad respondia a senales involuntarias de su entrenador. Sturm define un "horse" como "un sistema que aparenta ser capaz de una hazana humana notable, como el reconocimiento de genero musical a partir de una senal de audio, pero que en realidad funciona utilizando caracteristicas irrelevantes (confounders)".

Mediante experimentos controlados analogos a los disenados para evaluar a Clever Hans, Sturm demostro que sistemas MGR y MER de estado del arte dependian de factores confundidos con las etiquetas de ground truth del dataset (por ejemplo, diferencias en calidad de grabacion entre generos, o la presencia sistematica de ciertos artefactos de produccion en generos especificos). Esto tiene implicaciones devastadoras para la interpretabilidad de resultados: un clasificador puede alcanzar alta precision sin haber "aprendido" nada sobre genero musical per se.

**Implicaciones para la tesis:** El uso de genero como ground truth proxy para evaluar el sistema de recomendacion debe documentarse explicitamente como una limitacion conocida, fundamentada en la critica de Sturm. Se debe reconocer que: (1) el genero es una categoria subjetiva y multi-label; (2) la evaluacion basada en genero mide coherencia tematica aproximada, no calidad de recomendacion percibida; (3) un NMI alto entre clusters y generos no garantiza que el sistema haya capturado propiedades musicales relevantes.

#### 3.3.3. Enfoques multimodales para clasificacion de genero

Oramas, Barbieri, Nieto y Serra (2018) demostraron en Transactions of ISMIR que la fusion de multiples modalidades mejora significativamente la clasificacion de genero. Su trabajo combino tres fuentes de datos:
- **Audio**: Espectrogramas constant-Q procesados con CNN.
- **Visual**: Portadas de albumes analizadas con ResNet-101.
- **Texto**: Resenas de usuarios con semantic entity linking.

Los resultados mostraron que la combinacion audio+visual alcanzo F1=0.427 frente a F1=0.346 del audio solo (mejora del 23%). En clasificacion multi-label, la fusion trimodal alcanzo AUC-ROC=0.936, superando sustancialmente las modalidades individuales (texto: 0.917, audio: 0.888, visual: 0.743). El hallazgo clave es que "diferentes modalidades incorporan informacion complementaria", validando el principio fundamental de la fusion multimodal que subyace a esta tesis.

### 3.4. Music Emotion Recognition (MER)

#### 3.4.1. El modelo circumplejo de Russell

El modelo circumplejo del afecto propuesto por Russell (1980) constituye el marco teorico predominante en Music Emotion Recognition. Este modelo mapea las emociones en un espacio bidimensional continuo definido por:
- **Valence**: Eje horizontal, representando la valencia emocional desde negativa (tristeza, ira) hasta positiva (alegria, euforia).
- **Arousal**: Eje vertical, representando la intensidad de activacion desde baja (calma, relajacion) hasta alta (excitacion, agitacion).

La popularidad del modelo en MIR se atribuye a dos factores: (1) su simplicidad permite integracion directa con features acusticas cuantificables; (2) proporciona un espacio continuo que evita las limitaciones de las taxonomias categoricas discretas (feliz, triste, enojado, etc.). Sin embargo, el modelo asume independencia entre valence y arousal, una suposicion cuestionada por investigaciones que muestran correlaciones significativas entre ambas dimensiones en ciertos contextos musicales.

#### 3.4.2. Estado del arte en MER

El survey de Liyanarachchi, Joshi y Meijering (2025) proporciona una meta-analisis de 553 estudios identificados, de los cuales 34 fueron incluidos en la revision final. Los hallazgos principales son:

**Asimetria valence-arousal:** Los modelos predicen arousal con mayor precision que valence (r=0.81 vs r=0.67 usando los mejores modelos de cada estudio). Esta asimetria es consistente a traves de la literatura y refleja que arousal tiene correlatos acusticos mas directos (loudness, spectral energy, tempo) mientras que valence depende de factores mas complejos (armonia, modo, relaciones intervalicas, letra).

**Ventaja de metodos multimodales:** Los enfoques multimodales alcanzaron 79.2% de precision frente a 70.6% (audio solo) y 62.9% (letras solas), confirmando la complementariedad de las modalidades. La combinacion optima reportada utiliza CNN para audio y BERT para letras con late fusion, alcanzando 94.58%.

**Modelos lineales vs redes neuronales:** Un hallazgo contraintuitivo es que "los metodos lineales y basados en arboles generalmente superaron a las redes neuronales en tareas de regresion, mientras que las redes neuronales y SVMs mostraron mayor rendimiento en tareas de clasificacion". Esto sugiere que la complejidad del modelo debe ajustarse al tipo de tarea.

**Datasets y limitaciones:** Los principales datasets para MER (DEAM, PMEmo, MoodyLyrics) sufren de tamano limitado, restricciones de copyright, y sesgos culturales. La ausencia de un benchmark multimodal estandarizado dificulta la comparabilidad entre estudios.

### 3.5. El rol de las letras en MIR: evolucion de representaciones textuales

#### 3.5.1. De Bag-of-Words a Word2Vec

Las primeras aproximaciones al analisis de letras emplearon representaciones Bag-of-Words (BoW) y TF-IDF, que capturan frecuencias de terminos pero ignoran el orden secuencial y las relaciones semanticas entre palabras. Aunque BoW "puede causar cierta perdida de informacion" (Fell y Sporleder, 2014), constituyo un punto de partida util para tareas como deteccion de contenido explicito y clasificacion tematica basica.

La aparicion de Word2Vec (Mikolov et al., 2013) marco un avance significativo al proporcionar representaciones densas que capturan relaciones semanticas. Word2Vec fue "aplicado exitosamente en el dominio MIR para deteccion de canciones explicitas, clasificacion de genero y recomendacion musical". Sin embargo, Word2Vec presenta una limitacion critica: "aunque el orden de palabras se considera durante el entrenamiento, el algoritmo no proporciona un metodo para representar un documento completo", lo cual es frecuentemente necesario para tareas downstream en analisis de letras.

Doc2Vec (Le y Mikolov, 2014) extendio Word2Vec a nivel de documento, permitiendo generar un unico vector por cancion. Esto facilito aplicaciones de recomendacion basadas en similaridad coseno entre vectores de letras, aunque la calidad de las representaciones seguia siendo limitada por la falta de sensibilidad al contexto.

#### 3.5.2. Modelos contextuales: BERT y Transformers

La introduccion de BERT (Devlin et al., 2019) represento un cambio de paradigma en el analisis de letras. A diferencia de las tecnicas tradicionales como TF-IDF o BoW que dependen de conteos de palabras o features artesanales, "BERT entiende el contexto, lo cual es crucial para analizar letras" donde la ambiguedad, las metaforas y el lenguaje figurativo son prevalentes.

**LyBERT** (Parada-Cabaleiro et al., 2022): Demostro la superioridad de BERT sobre metodos tradicionales para clasificacion multi-clase de letras, observando que "el modelo BERT mejora la precision general al 92%", evidenciando la ventaja significativa de los embeddings contextuales para tareas MER.

**LyEmoBERT** (2023): Combina clasificacion emocional de letras con recomendacion musical usando modelos pre-entrenados, integrando el analisis de sentimiento basado en BERT con features acusticas para generar recomendaciones emocionalmente coherentes.

**Trabajos recientes (2025):** El preprint "Lyrics Matter: Exploiting the Power of Learnt Representations for Music" confirma que las representaciones aprendidas de letras, especialmente mediante Transformers, capturan informacion complementaria al audio y mejoran multiples tareas MIR simultaneamente.

**Sentence-BERT para clasificacion multi-label cross-lingual (2025):** Un trabajo reciente explora el uso de Sentence-BERT para clasificacion automatica de genero musical multi-label a partir de letras en multiples idiomas, demostrando la capacidad de transferencia cross-lingual de los embeddings contextuales.

**Implicaciones para la tesis:** La decision de usar Sentence-BERT (384 dimensiones) para vectorizar letras esta solidamente fundamentada en la evolucion del campo. La tesis se situa en la linea de trabajos que demuestran la superioridad de embeddings contextuales sobre representaciones estaticas para capturar la semantica de letras musicales.

### 3.6. Datasets principales en MIR

#### 3.6.1. Million Song Dataset (MSD)

Creado por Bertin-Mahieux et al. (2011), el MSD fue "el dataset musical por excelencia desde el inicio de la era del deep learning". Contiene features pre-calculadas para un millon de canciones, con metadata a nivel de artista. Sin embargo, presenta una distribucion de tags extremadamente long-tail: 522,366 tags para 505,216 tracks unicos, donde el tag mas popular ('rock') aparece en 101,071 tracks. El MSD no incluye audio completo (solo features pre-extraidas), lo cual limita su utilidad para investigacion con representaciones aprendidas.

#### 3.6.2. Free Music Archive (FMA)

Defferrard, Benzi, Vandergheynst y Frossard (2017) presentaron FMA en ISMIR 2017, proporcionando 917 GiB de audio con licencia Creative Commons: 106,574 tracks de 16,341 artistas organizados en una taxonomia jerarquica de 161 generos. A diferencia del MSD, FMA incluye "audio completo y de alta calidad", features pre-calculadas, y metadata a nivel de track y usuario. Esta riqueza lo posiciona favorablemente frente al MSD, que "solo proporciona metadata a nivel de artista".

#### 3.6.3. WASABI Dataset

El dataset WASABI (Buffa et al., 2021), publicado en The Semantic Web (ESWC), describe "mas de 2 millones de canciones comerciales, 200K albumes y 77K artistas". Su contribucion diferenciadora es la integracion como knowledge graph que vincula metadata recopilada con metadata generada mediante analisis de letras (temas, lugares, emociones, estructura) y senal de audio (acordes, sonido). El proyecto se basa en la Music Ontology y la extiende con vocabulario especifico, proporcionando un endpoint SPARQL, API REST y navegador interactivo. Posteriormente, Meseguer-Brocal et al. (2022) publicaron en Language Resources and Evaluation una version ampliada centrada en el corpus de letras y su knowledge graph asociado.

#### 3.6.4. Spotify Million Playlist Dataset (MPD)

Presentado como parte del RecSys Challenge 2018, el MPD contiene 1,000,000 de playlists con mas de 2 millones de tracks unicos de casi 300,000 artistas, constituyendo "el dataset publico mas grande de playlists musicales del mundo". La tarea asociada es la continuacion automatica de playlists: dado un titulo y/o conjunto inicial de tracks, predecir los tracks subsiguientes. Aunque el dataset ya no esta disponible para descarga publica, su impacto en la investigacion de recomendacion musical ha sido sustancial.

#### 3.6.5. GTZAN

Descrito en la seccion 3.3.2, GTZAN (Tzanetakis y Cook, 2002) contiene 1,000 clips de 30 segundos en 10 generos. A pesar de sus fallos documentados por Sturm (2013, 2014), permanece como el benchmark mas citado en clasificacion de genero, un testimonio tanto de la inercia en la comunidad como de la dificultad de establecer benchmarks alternativos.

### 3.7. Fusion multimodal en MIR

#### 3.7.1. Estrategias de fusion

La literatura identifica cuatro estrategias principales de fusion multimodal:

- **Early fusion (feature-level)**: Concatenacion de vectores de features antes del modelo de prediccion. Simple pero susceptible a representaciones desbalanceadas cuando las modalidades tienen dimensionalidades muy diferentes.
- **Late fusion (decision-level)**: Combinacion de predicciones independientes por modalidad. Permite modelos especializados por modalidad pero pierde interacciones cross-modal.
- **Model-level fusion**: Integracion dentro de la arquitectura del modelo (e.g., capas de atencion cross-modal). Mayor capacidad expresiva pero mayor complejidad.
- **Adaptive fusion**: Mecanismos de gating que ponderan dinamicamente las contribuciones de cada modalidad. Aborda el problema de contribuciones desbalanceadas pero requiere datos suficientes para aprender los pesos.

#### 3.7.2. Evidencia de complementariedad

La evidencia empirica consistentemente demuestra que audio y letras capturan informacion complementaria:

- Oramas et al. (2018): En clasificacion multi-label de genero, la fusion trimodal (audio+texto+visual) alcanzo AUC-ROC=0.936 vs. 0.888 (audio solo).
- Liyanarachchi et al. (2025): En MER, multimodal alcanzo 79.2% vs. 70.6% (audio) y 62.9% (letras).
- Yu et al. (2019): En correlacion cross-modal audio-letras, demostraron que la alineacion semantica entre ambas modalidades mejora la recuperacion musical.

Un hallazgo consistente es que "la concatenacion simple de features lleva a representaciones pobremente alineadas y contribuciones desbalanceadas de modalidades" (Liyanarachchi et al., 2025), lo cual motiva el desarrollo de estrategias de fusion mas sofisticadas.

**Implicaciones para la tesis:** El sistema hibrido propuesto (semantico 384D + musical 12D) enfrenta precisamente el problema de desbalance dimensional. La estrategia de fusion con pesos optimizables esta fundamentada en la literatura, que demuestra que la ponderacion adaptativa supera a la concatenacion simple.

---

## 4. Tabla de fuentes principales

| # | Autores (Ano) | Titulo | Tipo | Citas aprox. | Relevancia | Aporte clave |
|---|---------------|--------|------|-------------|------------|--------------|
| 1 | Sturm (2014) | A Simple Method to Determine if a MIR System is a "Horse" | Journal (IEEE TMM) | 250+ | Alta | Introduce el concepto de "Clever Hans" en MIR; demuestra que sistemas MGR usan confounders |
| 2 | Sturm (2013) | The GTZAN Dataset: Its Contents, Its Faults, Their Effects on Evaluation | Journal (JMTE) | 400+ | Alta | Documenta fallos sistematicos en el benchmark mas usado de MGR |
| 3 | Oramas et al. (2018) | Multimodal Deep Learning for Music Genre Classification | Journal (TISMIR) | 200+ | Alta | Demuestra complementariedad de modalidades (audio+texto+visual) en MGR |
| 4 | Downie (2003) | Music Information Retrieval | Survey (ARIST) | 500+ | Alta | Formalizacion seminal del campo MIR |
| 5 | Liyanarachchi et al. (2025) | A Survey on Multimodal Music Emotion Recognition | Survey (arXiv) | Reciente | Alta | Meta-analisis de 553 estudios; documenta asimetria valence-arousal |
| 6 | Duman et al. (2022) | Music We Move to: Spotify Audio Features and Reasons for Listening | Journal (PLoS ONE) | 30+ | Alta | Analisis de features Spotify en contexto de movimiento; critica a opacidad algoritmica |
| 7 | Vidas et al. (2025) | Validating Spotify's Valence, Energy, and Danceability Audio Features | Preprint (OSF) | Reciente | Alta | Demuestra que danceability NO correlaciona bien con percepcion humana |
| 8 | Schedl, Gomez y Urbano (2014) | Music Information Retrieval: Recent Developments and Applications | Survey (FnTIR) | 300+ | Alta | Survey comprehensivo del estado del arte MIR una decada post-Downie |
| 9 | Defferrard et al. (2017) | FMA: A Dataset for Music Analysis | Conference (ISMIR) | 400+ | Media-Alta | Dataset abierto con audio completo, 106K tracks, 161 generos |
| 10 | Buffa et al. (2021) | The WASABI Dataset: Cultural, Lyrics and Audio Analysis Metadata | Conference (ESWC) | 50+ | Media-Alta | Knowledge graph con 2M+ canciones, integracion lyrics+audio+metadata |
| 11 | Bogdanov et al. (2013) | Essentia: An Audio Analysis Library for MIR | Conference (ISMIR) | 800+ | Media | Herramienta principal de extraccion de features acusticas |
| 12 | McFee et al. (2015) | librosa: Audio and Music Signal Analysis in Python | Conference (SciPy) | 3000+ | Media | Biblioteca Python dominante para analisis de audio musical |
| 13 | Parada-Cabaleiro et al. (2022) | LyBERT: Multi-class Classification of Lyrics Using BERT | Journal | 20+ | Media-Alta | Demuestra superioridad de BERT sobre metodos tradicionales para clasificacion de letras |
| 14 | Bertin-Mahieux et al. (2011) | The Million Song Dataset | Conference (ISMIR) | 1500+ | Media | Dataset seminal de 1M canciones; features pre-calculadas |
| 15 | Russell (1980) | A Circumplex Model of Affect | Journal (JPSP) | 10000+ | Media | Marco teorico fundamental para representacion dimensional de emociones |
| 16 | Choi et al. (2019) | Machine Learning for Music Genre: Multifaceted Review | Journal (JIIS) | 50+ | Media | Revision de MGR con experimentacion en AudioSet; documenta subjetividad del genero |
| 17 | Chen et al. (2018) | RecSys Challenge 2018: Automatic Music Playlist Continuation | Conference (RecSys) | 100+ | Media | Spotify MPD: 1M playlists, benchmark de recomendacion musical |
| 18 | Yu et al. (2019) | Deep Cross-Modal Correlation Learning for Audio and Lyrics | Journal (ACM TOMM) | 60+ | Media | Alineacion semantica cross-modal audio-letras |
| 19 | Meseguer-Brocal et al. (2022) | The WASABI Song Corpus and Knowledge Graph for Music Lyrics Analysis | Journal (LREC) | 20+ | Media | Extension del corpus WASABI centrada en letras y knowledge graph |
| 20 | Tzanetakis y Cook (2002) | Musical Genre Classification of Audio Signals | Journal (IEEE TSAP) | 5000+ | Media | Paper original del dataset GTZAN; establece la tarea MGR |
| 21 | Serra et al. (2018) | Editorial: Introducing TISMIR | Journal (TISMIR) | 10+ | Baja-Media | Formalizacion de Transactions of ISMIR como venue |

---

## 5. Gaps identificados y oportunidades

### 5.1. Problemas abiertos

1. **Validacion de features de Spotify en contextos de recomendacion**: La mayoria de estudios de validacion (Vidas et al., 2025; Duman et al., 2022) se centran en percepcion individual, no en su utilidad para calcular similaridad entre canciones. No se ha investigado sistematicamente si las distancias en el espacio de features de Spotify se correlacionan con juicios humanos de similaridad musical.

2. **Fusion multimodal con dimensionalidades asimetricas**: La literatura sobre fusion multimodal asume tipicamente dimensionalidades comparables entre modalidades. El caso de esta tesis (384D semanticas + 12D musicales) representa un ratio 32:1 que no esta extensamente estudiado. La mayoria de trabajos concatenan features de dimensionalidades similares o usan mecanismos de atencion que requieren datasets grandes para converger.

3. **Genero como ground truth proxy sin calibracion**: Aunque Sturm (2013, 2014) desmonto la asuncion de que genero es un ground truth fiable, no existe un framework establecido para cuantificar el "error de proxy" al usar genero como sustituto de relevancia percibida en evaluacion de recomendacion.

4. **Analisis de letras cross-lingual en datasets multilingues**: Los datasets como el de esta tesis (basado en Genius, predominantemente en ingles) introducen sesgo linguistico. Los trabajos recientes sobre Sentence-BERT cross-lingual (2025) sugieren posibilidades no exploradas para datasets multilingues.

5. **Reproducibilidad en MIR**: La opacidad de las features de Spotify (algoritmos propietarios, posibles cambios silenciosos en la API) y las restricciones de copyright sobre audio dificultan la reproducibilidad. El campo carece de benchmarks multimodales completamente abiertos y reproducibles.

### 5.2. Oportunidades para la tesis

1. **Documentacion explicita de limitaciones de genero como proxy**: La tesis puede contribuir metodologicamente al documentar un framework de evaluacion que reconoce las limitaciones de Sturm y las aborda con analisis de sensibilidad.

2. **Analisis empirico de la complementariedad semantico-musical**: La combinacion especifica de embeddings BERT de letras (384D) con features de Spotify (12D) y su evaluacion mediante clustering multi-modal no esta extensamente reportada en la literatura. La mayoria de trabajos multimodales usan audio crudo (espectrogramas) en lugar de features de alto nivel pre-calculadas.

3. **Cuantificacion del aporte de cada modalidad**: El sistema hibrido con pesos optimizables permite medir empiricamente la contribucion relativa de letras vs. features musicales, contribuyendo a la discusion sobre complementariedad de modalidades.

4. **Bridge entre MIR y sistemas de recomendacion**: Mientras MIR se centra en clasificacion y recuperacion, y RecSys en prediccion de preferencias, la tesis integra ambos campos al usar representaciones MIR (BERT + Spotify features) como base para recomendacion.

### 5.3. Conexiones no exploradas

- La relacion entre la asimetria valence-arousal en MER y la fiabilidad diferencial de features de Spotify (energy fiable, danceability no fiable) sugiere que la calidad de la representacion musical afecta asimetricamente las dimensiones emocionales capturadas.
- La critica de Sturm sobre "horses" podria extenderse a sistemas de recomendacion: un sistema que recomienda canciones del mismo genero podria estar funcionando como un "horse" que usa genero como confound en lugar de capturar preferencias musicales genuinas.

---

## 6. Entrada BibTeX

```bibtex
@article{downie_2003_mir,
  author    = {Downie, J. Stephen},
  title     = {Music Information Retrieval},
  journal   = {Annual Review of Information Science and Technology},
  volume    = {37},
  number    = {1},
  pages     = {295--340},
  year      = {2003},
  doi       = {10.1002/aris.1440370108},
  publisher = {Wiley}
}

@article{sturm_2013_gtzan,
  author    = {Sturm, Bob L.},
  title     = {The {GTZAN} Dataset: Its Contents, Its Faults, Their Effects on Evaluation, and Its Future Use},
  journal   = {Journal of Music Technology and Education},
  year      = {2014},
  note      = {Preprint: arXiv:1306.1461, June 2013},
  doi       = {10.1080/09298215.2014.894533}
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

@article{oramas_2018_multimodal_genre,
  author    = {Oramas, Sergio and Barbieri, Francesco and Nieto, Oriol and Serra, Xavier},
  title     = {Multimodal Deep Learning for Music Genre Classification},
  journal   = {Transactions of the International Society for Music Information Retrieval},
  volume    = {1},
  number    = {1},
  pages     = {4--21},
  year      = {2018},
  doi       = {10.5334/tismir.10}
}

@article{liyanarachchi_2025_multimodal_mer,
  author    = {Liyanarachchi, Rashini and Joshi, Aditya and Meijering, Erik},
  title     = {A Survey on Multimodal Music Emotion Recognition},
  journal   = {arXiv preprint},
  year      = {2025},
  eprint    = {2504.18799},
  archiveprefix = {arXiv},
  primaryclass  = {cs.SD},
  url       = {https://arxiv.org/abs/2504.18799}
}

@article{duman_2022_spotify_movement,
  author    = {Duman, Deniz and Neto, Pedro and Mavrolampados, Anastasios and Toiviainen, Petri and Luck, Geoff},
  title     = {Music We Move to: {Spotify} Audio Features and Reasons for Listening},
  journal   = {PLoS ONE},
  volume    = {17},
  number    = {9},
  pages     = {e0275228},
  year      = {2022},
  doi       = {10.1371/journal.pone.0275228}
}

@article{vidas_2025_validating_spotify,
  author    = {Vidas, Dianna and Nitschinsk, Lewis and Osborne, Margaret and Rickard, Nikki},
  title     = {Validating {Spotify's} `Valence', `Energy', and `Danceability' Audio Features for Music Psychology Research},
  year      = {2025},
  doi       = {10.31234/osf.io/8gfzw},
  note      = {OSF Preprints},
  url       = {https://osf.io/preprints/psyarxiv/8gfzw}
}

@article{schedl_2014_mir_survey,
  author    = {Schedl, Markus and G\'{o}mez, Emilia and Urbano, Juli\'{a}n},
  title     = {Music Information Retrieval: Recent Developments and Applications},
  journal   = {Foundations and Trends in Information Retrieval},
  volume    = {8},
  number    = {2--3},
  pages     = {127--261},
  year      = {2014},
  doi       = {10.1561/1500000042}
}

@inproceedings{defferrard_2017_fma,
  author    = {Defferrard, Micha\"{e}l and Benzi, Kirell and Vandergheynst, Pierre and Frossard, Xavier},
  title     = {{FMA}: A Dataset for Music Analysis},
  booktitle = {Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2017},
  pages     = {316--323},
  url       = {https://arxiv.org/abs/1612.01840}
}

@inproceedings{buffa_2021_wasabi,
  author    = {Buffa, Michel and Cabrio, Elena and Fell, Michael and Gandon, Fabien and Giboin, Alain and Music, Romain and Pareti, Johan and Zucker, Jean-Daniel},
  title     = {The {WASABI} Dataset: Cultural, Lyrics and Audio Analysis Metadata About 2 Million Popular Commercially Released Songs},
  booktitle = {The Semantic Web -- ESWC 2021},
  pages     = {515--531},
  year      = {2021},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-77385-4_31}
}

@inproceedings{bogdanov_2013_essentia,
  author    = {Bogdanov, Dmitry and Wack, Nicolas and G\'{o}mez, Emilia and Gulati, Sankalp and Herrera, Perfecto and Mayor, Oscar and Roma, Gerard and Salamon, Justin and Zapata, Jos\'{e} R. and Serra, Xavier},
  title     = {{ESSENTIA}: An Audio Analysis Library for Music Information Retrieval},
  booktitle = {Proceedings of the 14th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2013},
  pages     = {493--498},
  url       = {https://essentia.upf.edu/}
}

@inproceedings{mcfee_2015_librosa,
  author    = {McFee, Brian and Raffel, Colin and Liang, Dawen and Ellis, Daniel P. W. and McVicar, Matt and Battenberg, Eric and Nieto, Oriol},
  title     = {librosa: Audio and Music Signal Analysis in Python},
  booktitle = {Proceedings of the 14th Python in Science Conference (SciPy)},
  year      = {2015},
  pages     = {18--24},
  doi       = {10.25080/Majora-7b98e3ed-003}
}

@article{parada_cabaleiro_2022_lybert,
  author    = {Parada-Cabaleiro, Emilia and Batliner, Anton and Schuller, Bj\"{o}rn W.},
  title     = {{LyBERT}: Multi-class Classification of Lyrics Using Bidirectional Encoder Representations from Transformers ({BERT})},
  journal   = {Expert Systems with Applications},
  year      = {2022},
  doi       = {10.1016/j.eswa.2022.117538},
  note      = {[no verificado -- metadata parcial de WebSearch]}
}

@inproceedings{bertin_mahieux_2011_msd,
  author    = {Bertin-Mahieux, Thierry and Ellis, Daniel P. W. and Whitman, Brian and Lamere, Paul},
  title     = {The Million Song Dataset},
  booktitle = {Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2011},
  pages     = {591--596},
  url       = {http://millionsongdataset.com/}
}

@article{russell_1980_circumplex,
  author    = {Russell, James A.},
  title     = {A Circumplex Model of Affect},
  journal   = {Journal of Personality and Social Psychology},
  volume    = {39},
  number    = {6},
  pages     = {1161--1178},
  year      = {1980},
  doi       = {10.1037/h0077714}
}

@article{choi_2019_genre_ml,
  author    = {Choi, Keunwoo and Fazekas, Gy\"{o}rgy and Sandler, Mark and Cho, Kyunghyun},
  title     = {Machine Learning for Music Genre: Multifaceted Review and Experimentation with {AudioSet}},
  journal   = {Journal of Intelligent Information Systems},
  volume    = {55},
  pages     = {69--91},
  year      = {2019},
  doi       = {10.1007/s10844-019-00582-9},
  note      = {[no verificado -- autores pueden diferir; articulo encontrado via ar5iv]}
}

@inproceedings{chen_2018_recsys_spotify,
  author    = {Chen, Ching-Wei and Lamere, Paul and Schedl, Markus and Zamani, Hamed},
  title     = {{RecSys} Challenge 2018: Automatic Music Playlist Continuation},
  booktitle = {Proceedings of the 12th ACM Conference on Recommender Systems (RecSys)},
  year      = {2018},
  pages     = {527--528},
  doi       = {10.1145/3240323.3240342}
}

@article{yu_2019_cross_modal,
  author    = {Yu, Yi and Tang, Suhua and Raposo, Francisco and Chen, Lei},
  title     = {Deep Cross-Modal Correlation Learning for Audio and Lyrics in Music Retrieval},
  journal   = {ACM Transactions on Multimedia Computing, Communications, and Applications},
  volume    = {15},
  number    = {1},
  pages     = {1--20},
  year      = {2019},
  doi       = {10.1145/3281746}
}

@article{meseguer_brocal_2022_wasabi_corpus,
  author    = {Meseguer-Brocal, Gabriel and Buffa, Michel and Music, Romain and Pareti, Johan and Zucker, Jean-Daniel},
  title     = {The {WASABI} Song Corpus and Knowledge Graph for Music Lyrics Analysis},
  journal   = {Language Resources and Evaluation},
  year      = {2022},
  doi       = {10.1007/s10579-022-09601-8},
  note      = {[no verificado -- autores parciales de WebSearch]}
}

@article{tzanetakis_2002_gtzan,
  author    = {Tzanetakis, George and Cook, Perry},
  title     = {Musical Genre Classification of Audio Signals},
  journal   = {IEEE Transactions on Speech and Audio Processing},
  volume    = {10},
  number    = {5},
  pages     = {293--302},
  year      = {2002},
  doi       = {10.1109/TSA.2002.800560}
}

@article{schedl_2018_challenges_recsys,
  author    = {Schedl, Markus and Zamani, Hamed and Chen, Ching-Wei and Deldjoo, Yashar and Elahi, Mehdi},
  title     = {Current Challenges and Visions in Music Recommender Systems Research},
  journal   = {International Journal of Multimedia Information Retrieval},
  volume    = {7},
  pages     = {95--116},
  year      = {2018},
  doi       = {10.1007/s13735-018-0154-2}
}
```

---

*Documento generado mediante revision sistematica de literatura. Fecha: 2026-02-07.*
*Busquedas realizadas: 12 consultas WebSearch + 5 verificaciones WebFetch.*
*Fuentes marcadas con [no verificado] requieren confirmacion manual de metadatos antes de inclusion en bibliography.bib.*
