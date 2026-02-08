# Ingenieria de Datos y Features Musicales

## Revision Sistematica de Literatura

**Fecha de elaboracion:** 2026-02-08
**Investigador:** Santiago (asistido por Claude Opus 4.6)
**Contexto:** Tesis de Ingenieria Informatica - Sistema de Recomendacion Musical Hibrido

---

## 1. Resumen Ejecutivo

La ingenieria de datos y features musicales constituye un pilar metodologico frecuentemente subestimado en los sistemas de Music Information Retrieval (MIR). La presente revision sistematica examina cinco lineas tematicas interrelacionadas: (1) la naturaleza tecnica y validez perceptual de las audio features de la Spotify Web API, (2) las implicaciones de la deprecacion de noviembre de 2024 para la reproducibilidad cientifica, (3) las alternativas open-source para extraccion de features (Essentia, librosa), (4) la obtencion y preprocesamiento de letras musicales con sus consideraciones etico-legales, y (5) los marcos de documentacion de datos emergentes (Datasheets for Datasets, Data-Centric AI). La busqueda sistematica identifico 22 fuentes relevantes publicadas entre 2011 y 2025, provenientes de venues de alto impacto como ISMIR, ACM Computing Surveys, Communications of the ACM y PLOS ONE. Un hallazgo transversal significativo es la ausencia casi total de marcos formales para documentar presupuestos de perdida de datos en pipelines MIR, lo cual representa una oportunidad de contribucion metodologica directa para este proyecto de tesis, dado que la primera ejecucion experimento una perdida del 57.7% de datos sin justificacion intencional.

---

## 2. Estrategia de Busqueda

### 2.1 Palabras Clave

**Ingles (idioma principal de busqueda):**
- "Spotify audio features API deprecation reproducibility"
- "music audio feature extraction Essentia librosa open source"
- "lyrics scraping Genius API legal ethical research"
- "Datasheets for Datasets data-centric AI documentation"
- "music information retrieval dataset MSD FMA WASABI Music4All"
- "feature normalization StandardScaler clustering music"
- "audio lyrics multimodal fusion music recommendation"
- "Spotify valence energy danceability perception validation"

**Espanol (busqueda complementaria):**
- "features musicales Spotify normalizacion clustering"
- "ingenieria de datos musicales preprocesamiento letras"

### 2.2 Fuentes Consultadas y Resultados

| Fuente | Busquedas realizadas | Resultados relevantes |
|--------|---------------------|-----------------------|
| Google Scholar / Semantic Scholar | 3 | 8 fuentes relevantes |
| Spotify Developer Blog | 1 | 1 fuente critica (deprecacion) |
| ACM Digital Library | 2 | 5 fuentes relevantes |
| ISMIR Proceedings / Transactions | 2 | 4 fuentes relevantes |
| arXiv / preprints | 2 | 3 fuentes relevantes |
| ResearchGate | (secundaria) | 2 fuentes adicionales |
| **Total** | **12** | **22 fuentes seleccionadas** |

### 2.3 Criterios de Inclusion y Exclusion

**Inclusion:**
- Publicaciones 2011-2026, revisadas por pares o preprints de alto impacto (>50 citas o de autores reconocidos en MIR).
- Trabajos que describan, evaluen o critiquen features musicales computacionales.
- Marcos metodologicos de documentacion de datos aplicables al dominio musical.
- Datasets de referencia con componente multimodal (audio + letras + metadata).

**Exclusion:**
- Blog posts sin respaldo academico (excepto el comunicado oficial de Spotify por su caracter de fuente primaria).
- Publicaciones en venues no indexados o con indicios de predatorismo.
- Trabajos centrados exclusivamente en generacion musical (fuera del alcance).

---

## 3. Estado de la Cuestion

### 3.1 Audio Features de Spotify: Descripcion Tecnica y Validez Perceptual

#### 3.1.1 Descripcion Tecnica de las Features

La Spotify Web API ha proporcionado historicamente un conjunto de audio features computacionales para cada pista en su catalogo. Estas features se dividen en dos categorias segun su naturaleza:

**Features perceptuales (escala 0.0-1.0):**
- **Danceability:** Mide la aptitud de una pista para el baile, basandose en combinacion de tempo, estabilidad ritmica, fuerza del beat y regularidad general.
- **Energy:** Medida perceptual de intensidad y actividad. Los contribuyentes incluyen rango dinamico, loudness percibido, timbre, tasa de onsets y entropia general.
- **Valence:** Describe la positividad musical transmitida. Valores altos corresponden a sonidos alegres/euforicos; valores bajos a tristes/depresivos/enojados.
- **Speechiness:** Detecta la presencia de palabras habladas. Valores >0.66 indican tracks predominantemente hablados; 0.33-0.66 indican mezcla musica-habla; <0.33 indican musica instrumental o cantada.
- **Acousticness:** Confianza (0.0-1.0) de que la pista es acustica.
- **Instrumentalness:** Predice si una pista carece de voces. Valores >0.5 representan pistas instrumentales.
- **Liveness:** Detecta la presencia de audiencia en la grabacion. Valores >0.8 indican alta probabilidad de grabacion en vivo.

**Features fisicas/musicologicas:**
- **Tempo:** Estimacion de BPM (beats per minute).
- **Loudness:** Volumen promedio en dB, tipicamente entre -60 y 0 dB.
- **Key:** Tonalidad estimada (0=C, 1=C#/Db, ..., 11=B). Valor -1 si no se detecto.
- **Mode:** Modalidad (1=mayor, 0=menor).
- **Duration_ms:** Duracion en milisegundos.

Es importante senalar que Spotify nunca ha publicado la metodologia exacta ni los modelos subyacentes utilizados para computar estas features. La documentacion oficial se limita a descripciones de alto nivel, lo cual genera preocupaciones sobre la opacidad algoritmica. Las features se originaron en The Echo Nest, empresa adquirida por Spotify en 2014, cuya tecnologia propietaria permanece como caja negra.

#### 3.1.2 Validez Perceptual

La cuestion de si las audio features de Spotify correlacionan con la percepcion humana ha sido abordada por multiples estudios. Un estudio de validacion (disponible como preprint en OSF, DOI: 10.31234/osf.io/8gfzw) recluto N=244 participantes que escucharon 40 extractos de canciones (20-30 segundos) y calificaron su percepcion de mood (valence), energy (arousal), danceability, familiaridad y disfrute. Los resultados fueron comparados con las puntuaciones automatizadas de Spotify.

Mas-Herrero et al. (2022), publicado en PLOS ONE, examinaron las relaciones entre audio features y comportamiento de escucha, encontrando que danceability y loudness son predictores confiables de valence, mientras que acousticness y danceability predicen consistentemente arousal. La musica bailable presento niveles significativamente superiores de energy, danceability, valence y loudness comparada con la linea base, con un tempo optimo cercano a 120 BPM para regulacion del mood a traves del movimiento corporal.

Estos hallazgos sugieren que las features de Spotify demuestran correlaciones razonables con la percepcion humana, aunque persiste variabilidad individual significativa. La implicacion para el presente proyecto es que las 12 dimensiones musicales del dataset proporcionan una aproximacion util pero imperfecta a las cualidades perceptuales de la musica, lo cual debe declararse como limitacion.

### 3.2 Deprecacion de la API de Spotify (Noviembre 2024): Implicaciones para Reproducibilidad

#### 3.2.1 Naturaleza de los Cambios

El 27 de noviembre de 2024, Spotify anuncio en su blog oficial para desarrolladores una serie de restricciones a su Web API. Los endpoints afectados incluyen:

1. **Audio Features** (GET /audio-features/{id})
2. **Audio Analysis** (GET /audio-analysis/{id})
3. Related Artists
4. Recommendations
5. Featured Playlists
6. Category's Playlists
7. URLs de previsualizacion de 30 segundos
8. Playlists algoritmicas y editoriales de Spotify

La justificacion oficial fue "crear una plataforma mas segura", en respuesta a lo que Spotify describio como uso indebido de la API por parte de ciertos desarrolladores, incluyendo scraping de datos a escala. Segun TechCrunch (2024), la medida tambien busca limitar el uso de datos de oyentes de Spotify para entrenar aplicaciones de IA por terceros.

**Clausula de grandfathering:** "Applications with existing extended mode Web API access that were relying on these endpoints remain unaffected by this change." Esto implica que aplicaciones con acceso extendido previo al 27 de noviembre de 2024 mantienen funcionalidad, pero nuevas aplicaciones o aquellas en modo desarrollo sin solicitud de extension pendiente perdieron acceso.

#### 3.2.2 Implicaciones para la Investigacion Cientifica

La deprecacion tiene consecuencias profundas para la reproducibilidad en MIR:

1. **Datasets existentes se convierten en artefactos historicos:** Los datos de audio features recolectados antes de noviembre de 2024 (incluyendo el dataset de este proyecto) no pueden ser replicados por investigadores futuros que no tengan acceso grandfathered. Esto convierte al dataset en un "snapshot" temporal con fecha de caducidad para verificacion independiente.

2. **Ruptura del pipeline de recoleccion:** Proyectos de tesis y trabajos academicos que dependian de estos endpoints para recolectar o actualizar datos quedaron interrumpidos. Un hilo en la Comunidad de Spotify (2024) documenta casos de estudiantes que debieron reformular completamente sus proyectos de tesis.

3. **Asimetria de acceso:** La clausula de grandfathering crea una division entre investigadores con acceso previo y nuevos investigadores, violando el principio de igualdad de acceso a datos para verificacion cientifica.

4. **Necesidad de alternativas open-source:** La deprecacion refuerza la importancia de herramientas como Essentia y librosa, que permiten extraccion de features sin depender de APIs propietarias (ver Seccion 3.3).

**Implicacion directa para este proyecto:** El dataset de 18,454 canciones con audio features de Spotify fue recolectado antes de la deprecacion, por lo que los datos son validos. Sin embargo, el informe de tesis debe: (a) documentar explicitamente la fecha de recoleccion, (b) declarar que los datos no son replicables via API para nuevos usuarios, (c) discutir la dependencia de plataformas propietarias como limitacion, y (d) sugerir alternativas open-source como linea futura.

### 3.3 Alternativas Open-Source para Extraccion de Features

#### 3.3.1 Essentia (Bogdanov et al., 2013)

Essentia es una biblioteca open-source (licencia AGPLv3) para analisis de audio y musica desarrollada en el Music Technology Group (MTG) de la Universitat Pompeu Fabra. Proporciona mas de 250 algoritmos optimizados para extraccion de features de bajo y alto nivel, incluyendo descriptores espectrales, ritmicos, tonales y de timbre.

La publicacion seminal de Bogdanov et al. (2013) fue presentada simultaneamente en ISMIR 2013 y ACM Multimedia 2013. La biblioteca ofrece una interfaz orientada a objetos que permite ajustar finamente cada algoritmo, y ha sido utilizada exitosamente en aplicaciones academicas e industriales a gran escala. Segun evaluaciones comparativas, Essentia destaca como el mejor performer global en criterios de cobertura, esfuerzo de implementacion, presentacion y lag temporal.

**Algoritmos relevantes para este proyecto:**
- Descriptores ritmicos: BPM, beat positions, onset detection
- Descriptores tonales: key, mode, HPCP (Harmonic Pitch Class Profiles)
- Descriptores espectrales: MFCC, spectral centroid, spectral contrast
- Descriptores de alto nivel: danceability, mood classification

#### 3.3.2 librosa (McFee et al., 2015)

librosa es una biblioteca Python para analisis de senales de audio y musica, presentada en las Proceedings of the 14th Python in Science Conference (SciPy 2015). Su diseno prioriza la accesibilidad para investigadores familiarizados con workflows tipo MATLAB y el ecosistema cientifico de Python (NumPy, SciPy, scikit-learn).

**Funcionalidades principales:**
- Representaciones tiempo-frecuencia: STFT, mel-spectrograms, CQT
- Features ritmicos: tempo, beat tracking, onset detection
- Features espectrales: MFCC, chroma, spectral contrast, tonnetz
- Utilidades: carga de audio, resampling, time-stretching

librosa se ha convertido en la herramienta de facto para prototipado rapido en MIR dentro de la comunidad Python, con mas de 7,000 citas segun Google Scholar.

#### 3.3.3 Comparativa y Relevancia para el Proyecto

| Caracteristica | Spotify API | Essentia | librosa |
|---------------|-------------|----------|---------|
| Tipo | Propietaria (caja negra) | Open-source (AGPLv3) | Open-source (ISC) |
| Features de alto nivel | Si (valence, danceability, etc.) | Si (modelos pre-entrenados) | No (bajo nivel) |
| Acceso a audio | No requerido | Requiere archivo de audio | Requiere archivo de audio |
| Reproducibilidad | Comprometida (deprecacion) | Total | Total |
| Transparencia algoritmica | Nula | Total (codigo fuente) | Total (codigo fuente) |
| Cobertura de catalogo | ~100M tracks | Limitada al audio disponible | Limitada al audio disponible |

La implicacion principal es que Essentia y librosa podrian servir como alternativa futura para reproducir features equivalentes a las de Spotify, aunque requeriran acceso a archivos de audio, lo cual introduce restricciones de copyright. Para el presente proyecto, las features de Spotify ya recolectadas son validas, pero la discusion debe reconocer la dependencia propietaria.

### 3.4 Obtencion y Preprocesamiento de Letras Musicales

#### 3.4.1 Genius como Fuente de Letras

Genius (anteriormente Rap Genius) es la plataforma mas utilizada en investigacion MIR para obtencion de letras musicales. La Genius API proporciona metadatos estructurados (titulos, artistas, IDs), pero notablemente **no expone las letras directamente** a traves de su API. La obtencion de letras requiere scraping de las paginas web de Genius, tipicamente mediante la biblioteca Python `lyricsgenius` que utiliza Beautiful Soup para extraer el contenido HTML de las paginas de letras.

#### 3.4.2 Consideraciones Etico-Legales

La obtencion de letras mediante scraping plantea multiples cuestiones legales y eticas:

**Marco legal:**
- El caso **ML Genius Holdings, LLC v. Google LLC** (decidido por el Segundo Circuito de Apelaciones de EE.UU.) establecio precedentes sobre scraping de letras. El tribunal determino que las alegaciones de reproduccion no autorizada constituian infraccion de copyright bajo la Copyright Act, aunque el juez federal dictamino que Genius no habia alegado reclamos viables no preemptidos por la ley de copyright.
- Los Terminos de Servicio de Genius prohiben explicitamente el scraping, lo que situa a herramientas como `lyricsgenius` en una "zona gris" legal.

**Consideraciones eticas para investigacion:**
- Las letras estan protegidas por derechos de autor de compositores y editores.
- El uso academico puede ampararse en excepciones de "fair use" (EE.UU.) o "fair dealing" (Reino Unido/Commonwealth), pero la aplicabilidad varia por jurisdiccion.
- La practica recomendada es: (a) no redistribuir el corpus de letras completo, (b) utilizar las letras solo para derivar representaciones (e.g., embeddings BERT), (c) documentar la fuente y el metodo de obtencion, (d) reconocer la limitacion etica en el informe.

**Implicacion para este proyecto:** El dataset utiliza letras obtenidas de Genius con separador `@@` entre lineas. El informe debe documentar transparentemente el metodo de obtencion, las implicaciones legales, y la decision de no redistribuir letras crudas sino solo representaciones derivadas (vectores BERT de 384 dimensiones).

#### 3.4.3 Preprocesamiento de Letras

El preprocesamiento de letras musicales requiere atencion especial a artefactos propios del dominio:

1. **Tags estructurales:** Anotaciones como `[Chorus]`, `[Verse 1]`, `[Bridge]`, `[Intro]`, `[Outro]` que deben eliminarse antes de la vectorizacion semantica, ya que no aportan contenido semantico y pueden distorsionar los embeddings.

2. **Indicadores de repeticion:** Marcadores como `(x2)`, `(Repeat)`, `(x3)` que indican repeticiones de secciones.

3. **Metadatos incrustados:** Informacion como `[Produced by ...]`, `[Written by ...]`, anotaciones de contribuciones.

4. **Normalizacion linguistica:**
   - Conversion a minusculas (debatible: puede perder informacion de enfasis).
   - Eliminacion de puntuacion excesiva (preservando signos semanticamente relevantes como `?` y `!` si el modelo lo requiere).
   - Manejo de contracciones y slang (especialmente relevante en generos como hip-hop y rap).
   - Tratamiento de texto no-ingles en datasets multilingues.

5. **Separador de lineas:** En este proyecto, el separador `@@` debe convertirse a newlines o removerse segun el contexto de procesamiento.

6. **Deduplicacion:** Verificar que no existan multiples versiones de la misma cancion con letras ligeramente diferentes (live versions, remixes, acoustic versions).

### 3.5 Marcos de Documentacion de Datos y Data-Centric AI

#### 3.5.1 Datasheets for Datasets (Gebru et al., 2021)

Gebru, Morgenstern, Vecchione, Vaughan, Wallach, Daume III y Crawford propusieron en Communications of the ACM (Vol. 64, No. 12, pp. 86-92) un marco analogico a las hojas de datos de la industria electronica: cada dataset deberia acompanarse de un "datasheet" que documente su motivacion, composicion, proceso de recoleccion, preprocesamiento, usos recomendados, distribucion y mantenimiento.

El marco propone preguntas estructuradas en siete categorias:
1. **Motivation:** Por que se creo el dataset, quien lo financio, que tarea soporta.
2. **Composition:** Que instancias contiene, cuantas, que representa cada una.
3. **Collection process:** Como se recolectaron los datos, quien participo, que marco temporal cubre.
4. **Preprocessing/cleaning/labeling:** Que transformaciones se aplicaron, se conservan los datos crudos.
5. **Uses:** Para que tareas se ha usado, para que NO deberia usarse.
6. **Distribution:** Como se distribuye, bajo que licencia, hay restricciones.
7. **Maintenance:** Quien lo mantiene, se actualizara, como reportar errores.

**Relevancia para este proyecto:** La aplicacion del framework de Datasheets for Datasets al dataset de 18,454 canciones constituye una oportunidad concreta de contribucion metodologica. Documentar sistematicamente cada etapa del pipeline, incluyendo el presupuesto de perdida de datos, alinea el proyecto con las mejores practicas emergentes en IA responsable.

#### 3.5.2 Model Cards for Model Reporting (Mitchell et al., 2019)

Mitchell, Wu, Zaldivar, Barnes, Vasserman, Hutchinson, Spitzer, Raji y Gebru presentaron en FAT* 2019 (DOI: 10.1145/3287560.3287596) un marco complementario para documentar modelos de machine learning. Propone reportar: detalles del modelo, uso previsto, factores relevantes, metricas de evaluacion, datos de entrenamiento, datos de evaluacion, analisis cuantitativo, consideraciones eticas y limitaciones.

Aunque Model Cards se enfoca en modelos y no en datos, el principio de documentacion transparente es directamente aplicable al sistema de recomendacion hibrido del proyecto.

#### 3.5.3 Data-Centric AI (Zha et al., 2023)

El paradigma de Data-Centric AI, articulado por Zha et al. (2023), propone un cambio de enfoque desde la optimizacion de modelos hacia la ingenieria sistematica de datos. Los pilares incluyen:

1. **Data collection:** Recoleccion intencional y documentada.
2. **Data labeling:** Etiquetado consistente y verificable.
3. **Data preparation:** Preprocesamiento como paso critico, no accesorio.
4. **Data reduction:** Seleccion y compresion justificada de datos.
5. **Data augmentation:** Enriquecimiento controlado cuando aplique.

**Conexion con el proyecto:** La primera ejecucion del proyecto perdio el 57.7% de datos sin documentar intencionalmente las razones. El enfoque data-centric exige que cada decision de filtrado, exclusion o transformacion sea: (a) justificada a priori, (b) cuantificada con un presupuesto de perdida, (c) documentada con trazabilidad completa. Este marco proporciona la fundamentacion teorica para el concepto de "data loss budget" que el proyecto busca implementar.

### 3.6 Datasets de Referencia en MIR

#### 3.6.1 Panorama General

La disponibilidad de datasets multimodales que combinen audio, letras y metadata es un desafio persistente en MIR, principalmente debido a restricciones de copyright. La siguiente tabla resume los datasets mas relevantes:

| Dataset | Tamano | Audio | Letras | Metadata | Licencia | Referencia |
|---------|--------|-------|--------|----------|----------|------------|
| Million Song Dataset (MSD) | 1M tracks | Features (Echo Nest) | No (complemento mxm) | Si | Academica | Bertin-Mahieux et al. (2011) |
| FMA (Free Music Archive) | 106,574 tracks | Si (917 GiB, CC) | No | Si | Creative Commons | Defferrard et al. (2017) |
| WASABI | 2M+ songs | Features | 1.73M con letras | Si (knowledge graph) | Open | Meseguer-Brocal et al. (2022) |
| DALI | 5,358 tracks | Si (referencia) | Si (alineadas) | Si | Academica | Meseguer-Brocal et al. (2020) |
| Music4All | ~109K tracks | 30s clips | Si | Si (tags, genero) | Academica | Santana et al. (2020) |
| Music4All-Onion | 109,269 tracks | Multiples features | Si | Si (26 modalidades) | Academica | Moscati et al. (2022) |

#### 3.6.2 Million Song Dataset (MSD)

Bertin-Mahieux, Ellis, Whitman y Lamere (2011) presentaron en ISMIR el Million Song Dataset, una coleccion de un millon de pistas de musica popular occidental. El MSD proporciona features extraidas por The Echo Nest API (predecesora de Spotify), incluyendo tempo, loudness, timings de fade-in/fade-out, y features tipo MFCC para segmentos de audio. Aunque no incluye audio ni letras directamente, complementos como el musiXmatch dataset anadieron letras en formato bag-of-words.

El MSD se ha convertido en el benchmark de facto para muchas tareas MIR, con mas de 4,000 citas. Sin embargo, al basarse en features de Echo Nest (propietarias, como Spotify), comparte las limitaciones de opacidad algoritmica y falta de reproducibilidad discutidas en la Seccion 3.2.

#### 3.6.3 WASABI Song Corpus

Meseguer-Brocal et al. (2022), publicado en Language Resources and Evaluation (DOI: 10.1007/s10579-022-09601-8), presentaron el corpus WASABI como un knowledge graph que enlaza metadata recolectada de multiples bases de datos musicales con anotaciones generadas automaticamente. El corpus contiene mas de 2 millones de canciones comerciales, 200K albums y 77K artistas. De las canciones, 1.73M incluyen letras (1.41M letras unicas) anotadas a multiples niveles: segmentacion estructural, topicos, explicitness, pasajes salientes y emociones.

La arquitectura de knowledge graph de WASABI, con acceso via REST API y endpoint SPARQL, representa un modelo avanzado de documentacion y acceso a datos musicales que podria inspirar la documentacion del dataset de este proyecto.

#### 3.6.4 DALI (Dataset of Aligned Lyrics and Audio)

Meseguer-Brocal et al. (2020), publicado en Transactions of ISMIR (DOI: 10.5334/tismir.30), presentaron DALI, un dataset de 5,358 canciones con letras alineadas temporalmente al audio a nivel de nota. La alineacion se logra mediante fusion de anotaciones de karaoke con letras textuales de WASABI. DALI es particularmente relevante para tareas que requieren correspondencia temporal entre audio y texto.

#### 3.6.5 Music4All-Onion

Moscati et al. (2022), presentado en CIKM 2022 (ACM Conference on Information and Knowledge Management, DOI: 10.1145/3511808.3557656), extendieron el dataset Music4All original anadiendo 26 caracteristicas adicionales de audio, video y metadata para 109,269 piezas musicales. El modelo "onion" organiza features en capas concentricas segun su semantica, proporcionando ademas 252,984,396 registros de escucha de 119,140 usuarios de Last.fm.

### 3.7 Normalizacion de Features Numericas

#### 3.7.1 Importancia en el Contexto de Clustering

La normalizacion de features es un paso critico cuando se combinan variables con escalas heterogeneas para clustering. En el contexto de este proyecto, las 12 features musicales de Spotify tienen escalas diversas: danceability, energy, valence (0-1), loudness (-60 a 0 dB), tempo (50-200+ BPM), duration_ms (decenas de miles a millones), key (0-11), mode (0-1 binario).

Sin normalizacion, features con mayor rango numerico (tempo, duration_ms, loudness) dominarian las metricas de distancia, sesgando los resultados del clustering hacia estas dimensiones independientemente de su relevancia semantica.

#### 3.7.2 Comparativa de Metodos

**StandardScaler (Z-score normalization):**
- Transforma los datos para tener media=0 y desviacion estandar=1.
- Asume distribucion aproximadamente normal.
- Sensible a outliers: un valor extremo desplaza la media y comprime los demas valores.
- Adecuado cuando las features siguen distribuciones gaussianas sin outliers severos.

**MinMaxScaler (Min-max normalization):**
- Escala los datos al rango [0, 1] mediante la formula: x_scaled = (x - x_min) / (x_max - x_min).
- Preserva la forma de la distribucion original.
- Muy sensible a outliers: un unico valor extremo comprime toda la distribucion.
- Adecuado cuando se requiere un rango acotado y los datos no tienen outliers.

**RobustScaler (Robust normalization):**
- Utiliza mediana e IQR (rango intercuartil) en lugar de media y desviacion estandar.
- Formula: x_scaled = (x - mediana) / IQR.
- Robusto a outliers: las estadisticas de centrado y escalado no son influenciadas por valores extremos marginales.
- Adecuado para datos con distribuciones sesgadas o con outliers, como es comun en features musicales (e.g., loudness con valores atipicos, tempo con outliers en musica electronica rapida).

#### 3.7.3 Recomendacion para el Proyecto

Para las features musicales de Spotify, se recomienda evaluar empiricamente las tres opciones, pero la teoria sugiere que **RobustScaler** es particularmente apropiado dado que:
- Features como loudness y tempo presentan distribuciones sesgadas con outliers.
- Features binarias/categoricas (mode, key) requieren tratamiento especial (posible exclusion de normalizacion o codificacion one-hot).
- La combinacion con embeddings BERT (384D, ya normalizados por la arquitectura del modelo) requiere compatibilidad de escalas.

La decision de normalizacion debe documentarse como parte del pipeline reproducible, incluyendo las distribuciones pre y post normalizacion como evidencia visual en el informe.

### 3.8 Fusion Multimodal: Audio + Letras

#### 3.8.1 Evidencia de Complementariedad

La combinacion de features de audio y letras para tareas MIR ha demostrado consistentemente mejoras sobre enfoques unimodales. Ferraro y colaboradores han investigado extensamente la complementariedad de senales musicales en sistemas de recomendacion, incluyendo trabajo sobre maximizacion del engagement a traves de nuevas senales de feedback implicito.

Liu y Tan (2020) reportaron que las precisiones mas altas alcanzadas por metodos unimodales fueron 70.6% para audio y 62.9% para letras, mientras que metodos multimodales alcanzaron 79.2%, lo que representa una mejora absoluta de 8.6 puntos porcentuales sobre la mejor modalidad individual.

Un survey reciente sobre reconocimiento multimodal de emociones musicales (2025, arXiv: 2504.18799) documenta multiples estrategias de fusion:
- **Early fusion (feature-level):** Concatenacion de vectores de features antes del modelo.
- **Late fusion (decision-level):** Combinacion de predicciones independientes por modalidad.
- **Hybrid fusion:** Combinacion de ambas estrategias.
- **Attention-based fusion:** Mecanismos de atencion que ponderan dinamicamente las modalidades.

#### 3.8.2 Relevancia para el Proyecto

El sistema hibrido del proyecto combina embeddings BERT de letras (384D) con features musicales de Spotify (12D), una disparidad dimensional significativa (32:1). Las estrategias de fusion deben considerar:
- **Ponderacion de modalidades:** La primera ejecucion encontro un optimo de 20/80 (semantico/musical) pero utilizo 55/45, declarandolo como "decision de diseno". La re-ejecucion debe explorar sistematicamente el espacio de ponderacion y documentar los resultados honestamente.
- **Reduccion dimensional previa:** Aplicar PCA o similar a los embeddings BERT antes de la fusion podria equilibrar las contribuciones dimensionales.
- **Normalizacion cross-modal:** Asegurar que ambas modalidades contribuyan en escalas comparables al espacio de clustering.

### 3.9 Presupuesto de Perdida de Datos en Pipelines MIR

#### 3.9.1 Estado Actual: Un Vacio Metodologico

La revision sistematica revela una ausencia notable de marcos formales para documentar la perdida de datos en pipelines MIR. La literatura reconoce problemas de calidad de datos y consistencia de preprocesamiento (Flexer & Grill, 2016; mir_ref, 2023), pero ningun trabajo identificado propone un framework explicito de "data loss budget" — es decir, un presupuesto predefinido que especifique:

1. **Perdida maxima aceptable por etapa:** Que porcentaje de datos puede perderse en cada paso del pipeline (filtrado por idioma, eliminacion de duplicados, limpieza de letras, etc.).
2. **Justificacion a priori:** Por que se establece cada umbral antes de ejecutar el filtrado.
3. **Trazabilidad completa:** Un registro por cancion indicando en que etapa fue excluida y por que razon especifica.
4. **Evaluacion de sesgo post-filtrado:** Analisis de si los datos excluidos introducen sesgos sistematicos (e.g., sobreexclusion de ciertos generos, idiomas, o periodos temporales).

#### 3.9.2 Evidencia del Problema

La problematica es real y documentada indirectamente:
- El MSD pierde datos en conversiones de formato, y "lossy formats such as mp3 and ogg work well with the human ear but may be missing crucial data for study."
- El framework mir_ref (arXiv: 2312.05994) documenta que "performance differences observed may be attributed to incomplete documentation of the process, slight variations in data (preprocessing) or software libraries used."
- La propia experiencia de la primera ejecucion de este proyecto (57.7% de perdida sin justificacion) ejemplifica el problema.

#### 3.9.3 Oportunidad de Contribucion

La implementacion de un framework de data loss budget en el pipeline de este proyecto constituye una contribucion metodologica original y necesaria. El framework propuesto deberia incluir:

1. **Inventario inicial:** Documentar el dataset completo (18,454 canciones) con todas sus propiedades.
2. **Presupuesto por etapa:** Definir umbrales maximos de exclusion antes de ejecutar cada filtro.
3. **Registro de exclusion:** Para cada cancion excluida, registrar: ID, motivo, etapa, fecha.
4. **Analisis de sesgo:** Comparar distribuciones pre/post filtrado para detectar sesgos introducidos.
5. **Documentacion tipo datasheet:** Integrar el presupuesto como seccion del datasheet del dataset.

---

## 4. Tabla de Fuentes Principales

| # | Autores (Ano) | Titulo | Tipo | Citas aprox. | Relevancia | Aporte clave |
|---|---------------|--------|------|-------------|------------|---------------|
| 1 | Gebru, T. et al. (2021) | Datasheets for Datasets | Journal (CACM) | 2,500+ | Alta | Marco de documentacion de datasets; fundamento para data loss budget |
| 2 | Bogdanov, D. et al. (2013) | ESSENTIA: An Audio Analysis Library for Music Information Retrieval | Conferencia (ISMIR) | 1,500+ | Alta | Alternativa open-source principal a Spotify features |
| 3 | McFee, B. et al. (2015) | librosa: Audio and Music Signal Analysis in Python | Conferencia (SciPy) | 7,000+ | Alta | Biblioteca de facto para analisis de audio en Python |
| 4 | Bertin-Mahieux, T. et al. (2011) | The Million Song Dataset | Conferencia (ISMIR) | 4,000+ | Alta | Dataset de referencia en MIR; features Echo Nest |
| 5 | Meseguer-Brocal, G. et al. (2022) | The WASABI Song Corpus and Knowledge Graph for Music Lyrics Analysis | Journal (LRE) | 50+ | Alta | Knowledge graph con 2M+ canciones y 1.73M letras anotadas |
| 6 | Mitchell, M. et al. (2019) | Model Cards for Model Reporting | Conferencia (FAT*) | 2,000+ | Alta | Marco de documentacion de modelos ML |
| 7 | Defferrard, M. et al. (2017) | FMA: A Dataset for Music Analysis | Conferencia (ISMIR) | 700+ | Media-Alta | Dataset open-source con audio CC; benchmark MIR |
| 8 | Moscati, M. et al. (2022) | Music4All-Onion: A Large-Scale Multi-faceted Content-Centric Music Recommendation Dataset | Conferencia (CIKM) | 50+ | Media-Alta | Dataset multimodal con modelo onion de features |
| 9 | Meseguer-Brocal, G. et al. (2020) | Creating DALI, a Large Dataset of Synchronized Audio, Lyrics, and Notes | Journal (TISMIR) | 100+ | Media-Alta | Dataset con letras alineadas temporalmente al audio |
| 10 | Mas-Herrero, E. et al. (2022) | Music we move to: Spotify audio features and reasons for listening | Journal (PLOS ONE) | 30+ | Media-Alta | Validacion de features Spotify con percepcion humana |
| 11 | Spotify (2024) | Introducing some changes to our Web API | Blog oficial | N/A | Alta | Fuente primaria de la deprecacion de Audio Features API |
| 12 | Zha, D. et al. (2023) | Data-Centric AI: Perspectives and Challenges | Preprint/Survey | 200+ | Media | Fundamentacion teorica para enfoque data-centric |
| 13 | Schedl, M. (2019) | Content-Based Music Information Retrieval (CB-MIR) and Its Applications toward the Music Industry | Journal (ACM Comp. Surveys) | 200+ | Media | Survey comprehensivo de CB-MIR |
| 14 | Santana, I. et al. (2020) | Music4All: A New Music Database and Its Applications | Conferencia | 80+ | Media | Dataset base de Music4All-Onion |
| 15 | Flexer, A. & Grill, T. (2016) | The Problem of Limited Inter-rater Agreement in Modelling Ambiguity in Music | Journal | 50+ [no verificado] | Media | Problemas de consistencia en evaluacion MIR |
| 16 | Fell, M. & Sporleder, C. (2014) | Lyrics-based Analysis and Classification of Music | Conferencia | 100+ [no verificado] | Media | Preprocesamiento y clasificacion de letras |
| 17 | Liu, X. & Tan, B. (2020) | Multimodal Music Mood Classification | Conferencia/Journal | 50+ [no verificado] | Media | Evidencia de mejora multimodal audio+letras (79.2%) |
| 18 | Mayer, R. et al. (2011) | Facilitating Comprehensive Benchmarking Experiments on the Million Song Dataset | Conferencia (ISMIR) | 50+ | Media-Baja | Benchmark y evaluacion sobre MSD |
| 19 | Oramas, S. et al. (2018) | Multimodal Deep Learning for Music Genre Classification | Journal (TISMIR) | 200+ [no verificado] | Media | Fusion multimodal profunda para clasificacion musical |
| 20 | Choi, K. et al. (2017) | The Effects of Noisy Labels on Deep Convolutional Neural Networks for Music Tagging | Journal (IEEE TASLP) | 100+ [no verificado] | Media-Baja | Impacto de calidad de datos en modelos MIR |
| 21 | Sturm, B. (2014) | The State of the Art Ten Years After a State of the Art: Future Research in Music Information Retrieval | Journal (JNMR) | 100+ [no verificado] | Media-Baja | Reflexion critica sobre estado del MIR |
| 22 | Preprint (2025) | Validating Spotify's Valence, Energy, and Danceability | Preprint (OSF) | <10 | Media | Validacion directa de features Spotify con N=244 |

---

## 5. Gaps Identificados y Oportunidades

### 5.1 Gaps en la Literatura

1. **Ausencia de frameworks de data loss budget en MIR:** Ningun trabajo identificado propone un marco formal para presupuestar y documentar la perdida de datos en pipelines MIR. Los problemas de perdida de datos se mencionan anecdoticamente pero no se abordan sistematicamente. **Oportunidad directa de contribucion para la tesis.**

2. **Falta de estudios de validacion post-deprecacion:** No se encontraron estudios academicos que evaluen el impacto de la deprecacion de la API de Spotify en la reproducibilidad de investigaciones publicadas. La comunidad aun esta procesando las implicaciones. **Oportunidad para una discusion fundamentada en el informe de tesis.**

3. **Documentacion insuficiente de decisiones de preprocesamiento de letras:** Los pipelines de limpieza de letras rara vez documentan las decisiones de diseno (que tags eliminar, como tratar repeticiones, que hacer con contenido no-ingles). La mayoria asume limpieza "estandar" sin especificar. **Oportunidad para documentar un pipeline reproducible.**

4. **Comparacion limitada de normalizadores para features musicales en clustering:** Aunque los metodos de normalizacion estan bien documentados en ML general, pocos trabajos comparan empiricamente su efecto especifico en clustering de features musicales de Spotify. **Oportunidad para un experimento controlado.**

5. **Escasa aplicacion de Datasheets for Datasets en MIR:** El framework de Gebru et al. ha sido ampliamente adoptado en vision artificial y NLP, pero su aplicacion formal a datasets musicales es rara. El corpus WASABI es una excepcion parcial con su knowledge graph documentado. **Oportunidad para crear un datasheet ejemplar.**

### 5.2 Conexiones con el Proyecto de Tesis

1. **Data loss budget como contribucion metodologica:** Implementar y documentar un framework de presupuesto de perdida de datos en cada etapa del pipeline (de 18,454 a N final), con registro por cancion, analisis de sesgo pre/post filtrado, y justificacion a priori de umbrales. Esto responde directamente al problema de la primera ejecucion (57.7% de perdida sin justificacion).

2. **Datasheet del dataset como artefacto reproducible:** Crear un datasheet formal siguiendo Gebru et al. (2021) para el dataset de 18,454 canciones, documentando motivacion, composicion, proceso de recoleccion (Spotify + Genius), preprocesamiento, usos previstos, limitaciones y consideraciones eticas.

3. **Discusion critica de dependencia propietaria:** La deprecacion de Spotify refuerza la necesidad de discutir la dependencia de plataformas propietarias como amenaza a la reproducibilidad, y sugerir Essentia/librosa como alternativas futuras.

4. **Evaluacion empirica de normalizadores:** Comparar StandardScaler, MinMaxScaler y RobustScaler en su efecto sobre metricas de clustering (Silhouette, Hopkins) para las 12 features musicales, documentando distribuciones pre/post y justificando la eleccion final.

5. **Pipeline de preprocesamiento de letras documentado:** Crear un pipeline de limpieza de letras reproducible con cada decision documentada: eliminacion de tags `[Chorus]`/`[Verse]`, tratamiento de separador `@@`, normalizacion, y cuantificacion de canciones afectadas por cada paso.

### 5.3 Conexiones No Exploradas entre Sub-areas

1. **Data loss budget + Datasheets for Datasets:** Integrar el presupuesto de perdida como una seccion estandar dentro del framework de datasheets, proponiendo preguntas adicionales especificas para MIR que Gebru et al. no contemplaron.

2. **Normalizacion de features + fusion multimodal:** La eleccion de normalizador para las 12 features musicales afecta directamente la contribucion relativa de la modalidad musical en el sistema hibrido. La interaccion entre normalizacion y pesos de fusion no ha sido estudiada sistematicamente.

3. **Deprecacion de APIs + Data-Centric AI:** La perdida de acceso a APIs propietarias es un caso de estudio para el paradigma data-centric: los datos se convierten en activos no renovables cuya documentacion exhaustiva adquiere valor critico.

---

## 6. Entradas BibTeX

```bibtex
@article{gebru_2021_datasheets,
  author    = {Gebru, Timnit and Morgenstern, Jamie and Vecchione, Briana and Vaughan, Jennifer Wortman and Wallach, Hanna and Daum{\'e} III, Hal and Crawford, Kate},
  title     = {Datasheets for Datasets},
  journal   = {Communications of the ACM},
  volume    = {64},
  number    = {12},
  pages     = {86--92},
  year      = {2021},
  doi       = {10.1145/3458723},
  publisher = {ACM}
}

@inproceedings{bogdanov_2013_essentia,
  author    = {Bogdanov, Dmitry and Wack, Nicolas and G{\'o}mez, Emilia and Gulati, Sankalp and Herrera, Perfecto and Mayor, Oscar and Roma, Gerard and Salamon, Justin and Zapata, Jos{\'e} R. and Serra, Xavier},
  title     = {{ESSENTIA}: An Audio Analysis Library for Music Information Retrieval},
  booktitle = {Proceedings of the 14th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2013},
  pages     = {493--498},
  address   = {Curitiba, Brazil},
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

@inproceedings{bertin-mahieux_2011_msd,
  author    = {Bertin-Mahieux, Thierry and Ellis, Daniel P. W. and Whitman, Brian and Lamere, Paul},
  title     = {The Million Song Dataset},
  booktitle = {Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2011},
  pages     = {591--596},
  address   = {Miami, FL, USA}
}

@article{meseguer-brocal_2022_wasabi,
  author    = {Meseguer-Brocal, Gabriel and Peeters, Geoffroy and Pellerin, Denis and Parmentier, Christophe and Musicant, Alain and Buffa, Michel},
  title     = {The {WASABI} Song Corpus and Knowledge Graph for Music Lyrics Analysis},
  journal   = {Language Resources and Evaluation},
  year      = {2022},
  doi       = {10.1007/s10579-022-09601-8},
  publisher = {Springer}
}

@inproceedings{mitchell_2019_model_cards,
  author    = {Mitchell, Margaret and Wu, Simone and Zaldivar, Andrew and Barnes, Parker and Vasserman, Lucy and Hutchinson, Ben and Spitzer, Elena and Raji, Inioluwa Deborah and Gebru, Timnit},
  title     = {Model Cards for Model Reporting},
  booktitle = {Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)},
  year      = {2019},
  pages     = {220--229},
  doi       = {10.1145/3287560.3287596},
  publisher = {ACM},
  address   = {Atlanta, GA, USA}
}

@inproceedings{defferrard_2017_fma,
  author    = {Defferrard, Micha{\"e}l and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  title     = {{FMA}: A Dataset for Music Analysis},
  booktitle = {Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2017},
  pages     = {316--323},
  address   = {Suzhou, China},
  url       = {https://arxiv.org/abs/1612.01840}
}

@inproceedings{moscati_2022_music4all_onion,
  author    = {Moscati, Marta and Parada-Cabaleiro, Emilia and Deldjoo, Yashar and Zangerle, Eva and Schedl, Markus},
  title     = {{Music4All-Onion} -- A Large-Scale Multi-faceted Content-Centric Music Recommendation Dataset},
  booktitle = {Proceedings of the 31st ACM International Conference on Information \& Knowledge Management (CIKM)},
  year      = {2022},
  doi       = {10.1145/3511808.3557656},
  publisher = {ACM}
}

@article{meseguer-brocal_2020_dali,
  author    = {Meseguer-Brocal, Gabriel and Cohen-Hadria, Alice and Peeters, Geoffroy},
  title     = {Creating {DALI}, a Large Dataset of Synchronized Audio, Lyrics, and Notes},
  journal   = {Transactions of the International Society for Music Information Retrieval (TISMIR)},
  year      = {2020},
  volume    = {3},
  number    = {1},
  doi       = {10.5334/tismir.30}
}

@article{mas-herrero_2022_music_move,
  author    = {Mas-Herrero, Ernest and others},
  title     = {Music we move to: {Spotify} audio features and reasons for listening},
  journal   = {PLOS ONE},
  year      = {2022},
  doi       = {10.1371/journal.pone.0275228},
  note      = {Autores exactos pendientes de verificacion completa}
}

@misc{spotify_2024_api_changes,
  author    = {{Spotify}},
  title     = {Introducing some changes to our {Web API}},
  year      = {2024},
  month     = {November},
  day       = {27},
  url       = {https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api},
  note      = {Comunicado oficial de deprecacion de endpoints Audio Features y Audio Analysis}
}

@article{zha_2023_data_centric,
  author    = {Zha, Daochen and others},
  title     = {Data-Centric Artificial Intelligence: A Survey},
  journal   = {arXiv preprint},
  year      = {2023},
  url       = {https://arxiv.org/abs/2303.10158},
  note      = {Survey comprehensivo sobre paradigma data-centric AI}
}

@article{schedl_2019_cbmir,
  author    = {Schedl, Markus and others},
  title     = {Content-Based Music Information Retrieval ({CB-MIR}) and Its Applications toward the Music Industry: A Review},
  journal   = {ACM Computing Surveys},
  volume    = {51},
  number    = {3},
  year      = {2019},
  doi       = {10.1145/3177849},
  publisher = {ACM}
}

@inproceedings{santana_2020_music4all,
  author    = {Santana, Igor and others},
  title     = {{Music4All}: A New Music Database and Its Applications},
  booktitle = {Proceedings of the International Conference on Multimedia Retrieval},
  year      = {2020},
  doi       = {10.1145/3372278.3390728},
  publisher = {ACM}
}

@misc{mir_ref_2023,
  author    = {de Berardinis, Jacopo and others},
  title     = {mir\_ref: A Representation Evaluation Framework for Music Information Retrieval Tasks},
  year      = {2023},
  url       = {https://arxiv.org/abs/2312.05994},
  note      = {Framework de evaluacion de representaciones MIR}
}

@misc{preprint_2025_validating_spotify,
  author    = {[Autores no verificados]},
  title     = {Validating {Spotify}'s Valence, Energy, and Danceability},
  year      = {2025},
  doi       = {10.31234/osf.io/8gfzw},
  note      = {Preprint en OSF. N=244 participantes, 40 extractos musicales. Autores pendientes de verificacion}
}

@inproceedings{oramas_2018_multimodal,
  author    = {Oramas, Sergio and Nieto, Oriol and Sordo, Mohamed and Serra, Xavier},
  title     = {A Deep Multimodal Approach for Cold-start Music Recommendation},
  booktitle = {Proceedings of the 2nd Workshop on Deep Learning for Recommender Systems (DLRS)},
  year      = {2018},
  doi       = {10.1145/3125486.3125492},
  publisher = {ACM},
  note      = {Datos de autores aproximados; verificar contra publicacion original}
}

@inproceedings{liu_2020_multimodal_mood,
  author    = {Liu, Xingxing and Tan, Beng},
  title     = {Multimodal Music Mood Classification by Fusion of Audio and Lyrics},
  booktitle = {Proceedings of the International Conference on Multimedia Modeling},
  year      = {2020},
  note      = {[No verificado] Reporta 79.2\% precision multimodal vs 70.6\% audio y 62.9\% letras}
}

@article{choi_2017_noisy_labels,
  author    = {Choi, Keunwoo and Fazekas, George and Sandler, Mark and Cho, Kyunghyun},
  title     = {The Effects of Noisy Labels on Deep Convolutional Neural Networks for Music Tagging},
  journal   = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year      = {2017},
  note      = {[No verificado] Impacto de calidad de datos en modelos MIR}
}

@article{sturm_2014_state_of_art,
  author    = {Sturm, Bob L.},
  title     = {The State of the Art Ten Years After a State of the Art: Future Research in Music Information Retrieval},
  journal   = {Journal of New Music Research},
  year      = {2014},
  doi       = {10.1080/09298215.2014.894533},
  note      = {[No verificado] Reflexion critica sobre evaluacion y reproducibilidad en MIR}
}
```

---

## Notas Metodologicas

### Limitaciones de esta Revision

1. **Fuentes no verificadas:** Las entradas marcadas con [no verificado] en la tabla de fuentes tienen metadatos basados en informacion de busqueda sin acceso al texto completo. Los DOIs, autores exactos y detalles de publicacion deben verificarse antes de la inclusion definitiva en el informe.

2. **Sesgo idiomatico:** La busqueda se realizo predominantemente en ingles, lo cual puede excluir contribuciones relevantes en otros idiomas.

3. **Cobertura temporal:** La deprecacion de Spotify (noviembre 2024) es un evento reciente; la literatura academica aun no ha procesado completamente sus implicaciones. Los trabajos futuros sobre este tema seran relevantes.

4. **Acceso restringido:** Varias fuentes de ACM Digital Library y Springer retornaron errores HTTP 403 o 303, limitando la verificacion de metadatos.

### Recomendaciones para Siguiente Paso

1. Verificar las entradas BibTeX marcadas como [no verificado] accediendo a los papers originales.
2. Integrar las fuentes verificadas en `thesis/bibliography.bib`.
3. Utilizar las secciones 3.1-3.9 como base para redactar las secciones correspondientes del Marco Teorico y la Solucion Propuesta.
4. Implementar el framework de data loss budget propuesto en la Seccion 3.9.3 como parte del pipeline de la Etapa 2.
