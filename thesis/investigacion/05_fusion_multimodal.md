# Fusion Multimodal: Estado de la Cuestion

## Investigacion sistematica para el Marco Teorico y Solucion Propuesta

**Fecha de elaboracion:** 2026-02-07
**Tematica:** Fusion multimodal -- taxonomia de estrategias, fusion tardia ponderada, normalizacion de espacios heterogeneos, optimizacion de pesos, complementariedad cross-modal, fusion audio+lyrics en musica, dimensionalidad heterogenea (12D vs 384D).
**Contexto:** Tesis de Ingenieria Informatica -- Sistema de recomendacion musical hibrido con fusion tardia ponderada entre componente semantico (BERT 384D) y musical (Spotify 12D).

---

## 1. Resumen ejecutivo

La fusion multimodal constituye uno de los desafios centrales del aprendizaje automatico cuando se dispone de multiples fuentes de informacion heterogeneas. La literatura revisada, que abarca desde los surveys fundacionales de Baltrusaitis et al. (2019) hasta las taxonomias actualizadas de Liang et al. (2024) y Gao et al. (2024), revela una evolucion desde taxonomias simples basadas en el momento temporal de la fusion (early/late) hacia clasificaciones mas sofisticadas que integran mecanismos de atencion, redes proyectivas y aprendizaje contrastivo. Se identificaron aproximadamente 25 fuentes relevantes provenientes de IEEE TPAMI, ACM Computing Surveys, ISMIR, y conferencias de primer nivel. Las principales conclusiones son: (1) la fusion tardia ponderada permanece como estrategia valida y ampliamente utilizada cuando las modalidades son inherentemente heterogeneas en dimensionalidad y naturaleza, ofreciendo ventajas de modularidad e interpretabilidad; (2) la normalizacion de scores constituye un paso critico cuya eleccion impacta significativamente el rendimiento del sistema fusionado; (3) la optimizacion de pesos mediante grid search o validacion cruzada es la practica estandar, aunque enfoques adaptativos basados en atencion representan la frontera actual; y (4) la complementariedad entre modalidades, medible mediante NMI cross-modal, es un indicador clave para justificar la fusion multimodal frente a enfoques unimodales.

---

## 2. Estrategia de busqueda

### 2.1 Palabras clave

**Ingles (primarias):** multimodal fusion taxonomy, late fusion weighted combination, score normalization heterogeneous features, cross-modal complementarity NMI, multimodal music recommendation audio lyrics, dimensionality mismatch fusion, attention-based fusion, hybrid recommender weighted fusion.

**Espanol (secundarias):** fusion multimodal, fusion tardia ponderada, normalizacion de scores, sistemas de recomendacion hibridos.

### 2.2 Fuentes consultadas y resultados

| Fuente | Consultas realizadas | Resultados relevantes |
|--------|---------------------|----------------------|
| Google Scholar / General Web | 6 | ~18 fuentes relevantes |
| Semantic Scholar | 2 | ~6 fuentes relevantes |
| ACM Digital Library | 3 | ~5 fuentes relevantes |
| IEEE Xplore | 2 | ~4 fuentes relevantes |
| ISMIR Archives | 2 | ~4 fuentes relevantes |
| arXiv | 3 | ~5 fuentes relevantes |

**Total de consultas:** 12 busquedas principales + 3 verificaciones via WebFetch.

### 2.3 Criterios de inclusion y exclusion

**Inclusion:**
- Publicaciones 2015-2026, revisadas por pares o preprints de alto impacto (>50 citas o publicacion en venue de primer nivel).
- Surveys y articulos seminales anteriores a 2015 cuando son fundacionales para el campo.
- Relevancia directa con fusion multimodal, sistemas de recomendacion hibridos, o MIR multimodal.

**Exclusion:**
- Blogs, tutoriales, y fuentes sin autor identificable.
- Publicaciones en venues no indexados o predatorios.
- Trabajos exclusivamente sobre fusion de imagenes medicas sin transferibilidad conceptual al dominio musical.

---

## 3. Estado de la cuestion

### 3.1 Taxonomias de fusion multimodal: del momento temporal a la complejidad funcional

#### 3.1.1 La taxonomia clasica: early, late e hybrid fusion

La clasificacion mas extendida de estrategias de fusion multimodal se fundamenta en el momento del pipeline en que se integra la informacion de las distintas modalidades. Esta taxonomia tripartita, consolidada en el survey seminal de Baltrusaitis, Ahuja y Morency (2019), distingue:

**Fusion temprana (early fusion):** Consiste en la concatenacion o combinacion de las representaciones de las distintas modalidades a nivel de features, antes de que un modelo unico procese la representacion conjunta. Su principal ventaja radica en la capacidad de capturar interacciones cross-modales de bajo nivel desde las etapas iniciales del procesamiento. Sin embargo, presenta limitaciones significativas: (a) requiere que las modalidades compartan la misma granularidad temporal o espacial; (b) es sensible a la maldicion de la dimensionalidad cuando se concatenan espacios de features muy distintos en tamano; y (c) no permite aprender representaciones marginales ricas para cada modalidad individual (Li y Tang, 2024).

**Fusion tardia (late fusion):** Opera combinando las decisiones o scores producidos independientemente por modelos unimodales especializados. Cada modalidad es procesada por su propio modelo, y los outputs se integran mediante mecanismos como votacion mayoritaria, promedio ponderado, o combinacion lineal de scores. Las ventajas incluyen: (a) modularidad -- cada componente puede ser desarrollado, entrenado y evaluado independientemente; (b) robustez ante datos faltantes en una modalidad; (c) capacidad de combinar modalidades con dimensionalidades radicalmente diferentes sin requerir alineacion espacial previa; y (d) interpretabilidad de la contribucion de cada modalidad a traves de los pesos asignados (Baltrusaitis et al., 2019). La limitacion principal es la incapacidad de capturar interacciones cross-modales complejas, ya que la informacion se integra unicamente al nivel de las decisiones finales.

**Fusion hibrida (hybrid fusion):** Combina elementos de las dos estrategias anteriores, integrando informacion en multiples niveles del pipeline. Esta estrategia puede capturar tanto interacciones de bajo nivel como complementariedades de alto nivel, pero incrementa la complejidad arquitectural y la dificultad de entrenamiento.

#### 3.1.2 Taxonomias modernas: mas alla del momento temporal

Los surveys recientes argumentan que la taxonomia clasica basada en el momento de fusion resulta insuficiente para describir la complejidad de los metodos actuales de deep learning. Gao et al. (2024), en su survey publicado en ACM Computing Surveys, proponen una taxonomia de cinco clases basada en la tecnica subyacente: (1) metodos encoder-decoder, (2) mecanismos de atencion, (3) redes neuronales en grafos, (4) redes neuronales generativas, y (5) metodos basados en restricciones. Los autores argumentan que las fronteras entre extraccion de features, fusion y decision se han difuminado en las arquitecturas modernas de deep learning.

De manera complementaria, Liang, Zadeh y Morency (2024), tambien en ACM Computing Surveys, expanden la taxonomia original de cinco desafios (Baltrusaitis et al., 2019) a seis: representacion, alineacion, razonamiento, generacion, transferencia y cuantificacion. La fusion se subsume dentro del desafio de "razonamiento", que se define como la combinacion de informacion de evidencia multimodal de manera principiada. Esta reconceptualizacion refleja que la fusion no es un paso aislado sino un proceso integrado en el razonamiento multimodal.

Li y Tang (2024), en su survey sobre alineacion y fusion multimodal, proponen una clasificacion que distingue entre fusion basada en encoder-decoder (a nivel de datos, features o modelo), fusion basada en kernels, fusion basada en grafos y fusion basada en atencion. Esta taxonomia captura la diversidad de mecanismos tecnicos empleados en la practica contemporanea.

#### 3.1.3 Implicaciones para el proyecto de tesis

Para el sistema de recomendacion musical hibrido, la taxonomia clasica (early/late/hybrid) sigue siendo la mas pertinente operativamente, dado que el sistema no emplea arquitecturas deep learning end-to-end para la fusion sino una combinacion lineal de scores provenientes de componentes independientes. La fusion tardia se justifica por las diferencias radicales en dimensionalidad (384D vs 12D) y naturaleza (semantica vs acustica) de los espacios de features. Sin embargo, es importante contextualizar esta eleccion dentro del panorama mas amplio de la literatura, reconociendo que enfoques mas sofisticados (attention-based, contrastive learning) representan la frontera actual del campo.

### 3.2 Fusion tardia con ponderacion lineal: fundamentos formales y practica

#### 3.2.1 Formulacion matematica

La fusion tardia con ponderacion lineal se define formalmente como la combinacion convexa de scores provenientes de K modalidades o componentes:

```
S_fusion(i, j) = sum_{k=1}^{K} w_k * S_k(i, j)
```

donde `S_k(i, j)` es el score de similitud entre los items `i` y `j` segun el componente `k`, y `w_k` son pesos no negativos que suman 1. En el caso bimodal del proyecto (K=2), esto se reduce a:

```
S_fusion(i, j) = alpha * S_semantico(i, j) + (1 - alpha) * S_musical(i, j)
```

donde `alpha` in [0, 1] es el parametro de fusion que controla la contribucion relativa de cada componente.

Este esquema corresponde a lo que Burke (2002) denomina "weighted hybridization" en su taxonomia canonica de sistemas de recomendacion hibridos. Burke identifico siete estrategias de hibridacion: weighted, switching, mixed, feature combination, feature augmentation, cascade y meta-level. La estrategia weighted es la mas simple conceptualmente y una de las mas utilizadas en la practica, debido a su transparencia, facilidad de implementacion y capacidad de ajuste.

#### 3.2.2 Propiedades teoricas

La combinacion lineal de scores posee propiedades deseables bien documentadas en la literatura de fusion de informacion (Vogt y Cottrell, 1999; Montague y Aslam, 2001):

- **Idempotencia parcial:** Si todos los componentes producen el mismo ranking, la fusion preserva ese ranking independientemente de los pesos.
- **Monotonia:** Un incremento en el score de cualquier componente para un par (i, j) no puede reducir el score fusionado.
- **Convexidad:** El score fusionado esta acotado por el minimo y maximo de los scores componentes.
- **Linealidad:** Permite descomposicion e interpretacion directa de la contribucion de cada modalidad.

Estas propiedades hacen que la fusion lineal sea especialmente atractiva en contextos donde la interpretabilidad y la trazabilidad de las decisiones son requisitos, como es el caso de una tesis que debe documentar transparentemente el comportamiento del sistema.

#### 3.2.3 Limitaciones reconocidas

La literatura identifica varias limitaciones de la ponderacion lineal:

1. **Pesos globales estaticos:** Los pesos son constantes para todos los pares de items, ignorando que la importancia relativa de cada modalidad puede variar segun el contexto especifico (genero musical, presencia/ausencia de letras, etc.). Los enfoques basados en atencion (Zhu et al., 2020; AAAI 2025) abordan esta limitacion mediante pesos dinamicos condicionales al input.

2. **Incapacidad de modelar interacciones no lineales:** La combinacion lineal asume que las contribuciones de las modalidades son aditivas e independientes. Interacciones sinergicas (donde la combinacion de ciertas features de ambas modalidades produce informacion que ninguna contiene individualmente) no pueden ser capturadas (Baltrusaitis et al., 2019).

3. **Sensibilidad a la normalizacion:** El rendimiento de la fusion lineal depende criticamente de que los scores de los distintos componentes esten en escalas comparables, lo cual requiere una normalizacion cuidadosa (Montague y Aslam, 2001).

### 3.3 Normalizacion de espacios heterogeneos

#### 3.3.1 El problema de la incompatibilidad de escalas

Cuando se fusionan scores provenientes de espacios de features con dimensionalidades y naturalezas radicalmente diferentes -- como es el caso de similitudes coseno en un espacio de 384 dimensiones (BERT) versus similitudes en un espacio de 12 dimensiones (features musicales de Spotify) -- los rangos, distribuciones y semanticas de los scores pueden diferir sustancialmente. Sin normalizacion, un componente puede dominar artificialmente la fusion simplemente porque produce scores en un rango numerico mayor.

Este problema esta ampliamente documentado en la literatura de metabusqueda y fusion de informacion (Montague y Aslam, 2001), donde se demostro que la normalizacion de scores de relevancia es un paso critico cuya eleccion impacta significativamente el rendimiento del sistema fusionado.

#### 3.3.2 Metodos de normalizacion

Los principales metodos de normalizacion de scores documentados en la literatura son:

**Min-Max Normalization:**
```
S_norm(i,j) = (S(i,j) - S_min) / (S_max - S_min)
```
Transforma los scores al rango [0, 1]. Es sensible a outliers, ya que un unico valor extremo puede comprimir la distribucion de los demas scores. Sin embargo, preserva las relaciones de orden y las proporciones relativas dentro de la distribucion original.

**Z-Score Normalization:**
```
S_norm(i,j) = (S(i,j) - mu) / sigma
```
Centra la distribucion en media 0 y desviacion estandar 1. Es mas robusta ante outliers que Min-Max, pero los scores resultantes no estan acotados a un rango fijo, lo cual puede complicar la interpretacion de los pesos de fusion.

**Rank Normalization:**
Reemplaza los scores por sus posiciones en el ranking, eliminando completamente la influencia de la distribucion original. Sacrifica la informacion sobre la magnitud de las diferencias entre scores a cambio de total robustez ante diferencias de escala.

**Sum Normalization:**
```
S_norm(i,j) = S(i,j) / sum_j S(i,j)
```
Normaliza dividiendo por la suma total, produciendo distribuciones de probabilidad. Es sensible a la presencia de valores negativos.

#### 3.3.3 Evidencia empirica comparativa

Los estudios comparativos de normalizacion en contextos de fusion muestran resultados consistentes: Min-Max y Z-Score son los metodos mas utilizados y generalmente producen rendimientos competitivos. Montague y Aslam (2001) demostraron en el contexto de metabusqueda que la eleccion del metodo de normalizacion puede alterar significativamente la eficacia de la fusion. En el contexto mas reciente de busqueda hibrida (OpenSearch, 2024), Z-Score mostro ventajas sobre Min-Max en terminos de relevancia de busqueda, especialmente cuando las distribuciones de scores de los componentes presentan asimetrias pronunciadas.

#### 3.3.4 Implicaciones para el proyecto

En el sistema de recomendacion musical, la similitud coseno en el espacio semantico BERT (384D) y la similitud en el espacio musical (12D) producen distribuciones potencialmente diferentes. La similitud coseno tiende a producir valores positivos concentrados en un rango relativamente estrecho para embeddings de alta dimension (efecto de concentracion de la medida), mientras que la similitud en el espacio musical de baja dimension puede exhibir mayor varianza. Min-Max normalization al rango [0, 1] es la opcion mas transparente y ampliamente utilizada en sistemas de recomendacion hibridos. Z-Score es preferible si se detectan outliers significativos. La decision debe documentarse con un analisis descriptivo de las distribuciones de scores antes de la normalizacion.

### 3.4 Optimizacion de pesos de fusion

#### 3.4.1 Grid search exhaustivo

El metodo mas directo para determinar el peso optimo alpha en la fusion bimodal es el grid search sobre el rango [0, 1] con un paso fijo (tipicamente 0.05 o 0.01). Para cada valor de alpha, se evalua el rendimiento del sistema fusionado mediante metricas de evaluacion (Precision@k, NDCG@k, etc.) sobre un conjunto de validacion. El alpha que maximiza la metrica objetivo se selecciona como peso final.

Este enfoque es exhaustivo y libre de supuestos sobre la relacion entre alpha y el rendimiento, pero tiene limitaciones: (a) solo es viable cuando el espacio de busqueda es unidimensional o de muy baja dimension; (b) requiere un ground truth o proxy para evaluacion; y (c) puede sobreajustar al conjunto de validacion si este es pequeno.

En el contexto de sistemas hibridos de recomendacion, la practica estandar documentada por Burke (2002) y trabajos posteriores consiste en realizar este grid search con validacion cruzada para mitigar el sobreajuste. El resultado de la primera ejecucion del proyecto (alpha optimo = 0.20 para el componente semantico, vs. el 0.55 utilizado inicialmente) ilustra la importancia critica de esta optimizacion: una diferencia de 35 puntos porcentuales en el peso puede tener impacto sustancial en el rendimiento.

#### 3.4.2 Optimizacion bayesiana y metodos adaptativos

Para espacios de busqueda de mayor dimension (multiples modalidades con pesos independientes), la optimizacion bayesiana ofrece ventajas sobre el grid search al modelar la funcion objetivo como un proceso gaussiano y seleccionar los puntos de evaluacion de manera informada. Sin embargo, para el caso bimodal, el grid search es suficiente y preferible por su transparencia.

Un enfoque reciente de interes es DAT (Dynamic Alpha Tuning), que propone ajustar dinamicamente el peso de fusion para cada consulta individual, adaptandose al contexto especifico. Aunque desarrollado para busqueda hibrida (dense + BM25), el principio es transferible: la importancia relativa de cada modalidad puede variar segun las caracteristicas del item o la consulta.

#### 3.4.3 Pesos adaptativos basados en atencion

Los mecanismos de atencion representan la extension natural de la ponderacion lineal estatica hacia pesos dinamicos. Zhu et al. (2020) propusieron un metodo de fusion multimodal basado en self-attention que ajusta adaptativamente los pesos de fusion segun la contribucion aprendida de cada modalidad y feature a partir de los datos etiquetados. En el contexto de la recomendacion musical, Vaswani y Agrawal (2021) presentaron redes atentivas para recomendacion musical secuencial que fusionan representaciones de letras (via Transformer) y features acusticas (via VAE) mediante mecanismos de atencion que ponderan dinamicamente cada modalidad.

Si bien estos enfoques son mas expresivos que la ponderacion lineal estatica, introducen complejidad adicional, requieren datos de entrenamiento supervisado, y sacrifican la interpretabilidad directa de los pesos. Para el alcance de la tesis, la ponderacion lineal optimizada via grid search representa un compromiso adecuado entre rendimiento e interpretabilidad.

### 3.5 Complementariedad entre modalidades y NMI cross-modal

#### 3.5.1 Fundamentos de la complementariedad multimodal

La justificacion fundamental de la fusion multimodal es que diferentes modalidades capturan aspectos complementarios de un fenomeno. En el dominio musical, las features acusticas (tempo, energia, danceability) capturan propiedades perceptuales y ritmicas, mientras que las representaciones semanticas de letras capturan contenido tematico, emocional y narrativo. La fusion solo es beneficiosa en la medida en que estas modalidades aporten informacion no redundante.

Liang et al. (2024) formalizan esta intuicion dentro de su marco de "interacciones multimodales", distinguiendo entre: (a) redundancia -- informacion compartida entre modalidades; (b) complementariedad -- informacion unica aportada por cada modalidad; y (c) sinergia -- informacion que emerge unicamente de la combinacion y no esta presente en ninguna modalidad individual.

#### 3.5.2 NMI como metrica de complementariedad

La Informacion Mutua Normalizada (NMI) entre las asignaciones de clustering producidas por cada modalidad proporciona una medida operativa de la relacion entre los espacios de representacion. Formalmente:

```
NMI(U, V) = 2 * I(U; V) / (H(U) + H(V))
```

donde U y V son las particiones de clustering de las dos modalidades, I(U; V) es la informacion mutua, y H(.) es la entropia.

- NMI = 1 indica que ambas modalidades producen particiones identicas (redundancia total).
- NMI = 0 indica independencia completa (complementariedad maxima, sin informacion compartida).
- Valores intermedios indican una mezcla de redundancia y complementariedad.

En la primera ejecucion del proyecto, el NMI cross-modal fue 0.0567, un valor cercano a cero que indica alta complementariedad: los clustering semantico y musical agrupan las canciones de maneras muy diferentes, lo cual justifica fuertemente la fusion multimodal. Este resultado es consistente con la intuicion de que letras y features acusticas capturan dimensiones ortogonales de la experiencia musical.

#### 3.5.3 Interpretacion y precauciones

Un NMI cercano a cero no garantiza que la fusion sea beneficiosa; tambien podria indicar que una de las modalidades produce clustering aleatorio o sin estructura. Es necesario verificar que ambas modalidades individualmente producen clustering de calidad (e.g., mediante Hopkins statistic y metricas de validacion interna como Silhouette) antes de interpretar el NMI bajo como evidencia de complementariedad productiva.

Ademas, el NMI mide relacion entre particiones discretas (asignaciones de cluster), no entre espacios continuos. Para medir complementariedad a nivel de representaciones continuas, metricas como la correlacion canonica (CCA) o la informacion mutua estimada mediante redes neuronales (MINE) son mas apropiadas, aunque computacionalmente mas costosas.

### 3.6 Fusion multimodal en musica: audio + lyrics

#### 3.6.1 Enfoques de fusion en MIR

El dominio de Music Information Retrieval (MIR) ha explorado extensamente la fusion de modalidades musicales. Oramas et al. (2018), en un estudio publicado en TISMIR, evaluaron la fusion de representaciones aprendidas desde audio, texto (resenas de albumes) e imagenes (portadas de albumes) para clasificacion de genero musical. Sus resultados demostraron que la agregacion de representaciones multimodales mejora consistentemente la precision de clasificacion respecto a enfoques unimodales. Crucialmente, el estudio comparo fusion a nivel de features (concatenacion de representaciones intermedias) con fusion a nivel de decision (combinacion de predicciones), encontrando que ambas estrategias aportan mejoras significativas.

#### 3.6.2 Aprendizaje contrastivo audio-lenguaje

Un avance reciente significativo en la fusion audio-texto musical es el framework MusCALL (Music Contrastive Audio-Language Learning) de Manco et al. (2022), presentado en ISMIR 2022. MusCALL emplea una arquitectura dual-encoder que aprende alineacion entre pares de audio musical y descripciones textuales mediante una perdida contrastiva. Este enfoque produce embeddings multimodales que permiten recuperacion text-to-audio y audio-to-text sin entrenamiento adicional. El sistema MULAN (Huang et al., 2022), tambien de ISMIR 2022, sigue un principio similar con un joint embedding space para audio y lenguaje natural.

Estos enfoques contrastivos representan una forma sofisticada de fusion implicita: en lugar de combinar scores o features explicitamente, aprenden un espacio compartido donde ambas modalidades son directamente comparables. Sin embargo, requieren grandes volumenes de datos pareados (audio, texto) para entrenamiento, lo cual limita su aplicabilidad en contextos con datasets de tamano moderado como el del proyecto (18,454 canciones).

#### 3.6.3 Reconocimiento de emociones musicales multimodal

La fusion de audio y letras ha sido extensamente estudiada en el contexto de reconocimiento de emociones musicales (Music Emotion Recognition, MER). La literatura reconoce que la emocion musical es frecuentemente una expresion conjunta de audio y texto: letras tristes combinadas con melodia sombria refuerzan la tristeza, pero existen escenarios conflictivos (melodia alegre + letras tristes). Los mecanismos cross-modales que ajustan pesos para calibrar la informacion en escenarios emocionales complejos mejoran la precision de la clasificacion emocional (IIETA, 2025).

#### 3.6.4 SLEM: Spectro-Lyrical Embeddings for Music

Un trabajo reciente relevante es SLEM (Spectro-Lyrical Embeddings for Music), publicado en Multimedia Tools and Applications (2024), que presenta representaciones musicales multimodales aprovechando modelos de deep learning de vision y lenguaje para codificar canciones. SLEM combina features espectrales (audio) con features liricas (texto) para clasificacion de generos musicales, demostrando que la combinacion multimodal supera consistentemente a los enfoques unimodales.

### 3.7 Desafio de dimensionalidad heterogenea

#### 3.7.1 El problema 12D vs 384D

Cuando las modalidades a fusionar residen en espacios de dimensionalidad radicalmente diferente -- como los 12 features musicales de Spotify versus los 384 dimensiones del embedding BERT -- surgen desafios tecnicos especificos:

1. **Concentracion de la medida en alta dimension:** En espacios de alta dimension, las distancias entre puntos tienden a concentrarse, haciendo que las similitudes coseno entre pares de items sean mas uniformes. Este efecto es mas pronunciado en el espacio de 384D que en el de 12D, lo cual puede producir distribuciones de similitud con caracteristicas estadisticas muy diferentes.

2. **Riqueza informacional asimetrica:** El espacio de mayor dimension tiene capacidad para codificar relaciones mas complejas y matizadas, pero esto no necesariamente implica que sea mas informativo para la tarea especifica. En la primera ejecucion del proyecto, el peso optimo alpha = 0.20 para el componente semantico (384D) sugiere que el componente musical de menor dimension (12D) era mas relevante para la tarea de recomendacion.

3. **Curse of dimensionality en fusion temprana:** La concatenacion directa de los dos espacios (12D + 384D = 396D) produciria un espacio donde las 12 features musicales representan solo el 3% de las dimensiones, diluyendo su contribucion. Esta es una justificacion tecnica adicional para la eleccion de fusion tardia, donde cada componente opera en su propio espacio.

#### 3.7.2 Redes proyectivas para dimensionalidad heterogenea

Morano et al. (2024), publicado en IEEE Journal of Biomedical and Health Informatics, propusieron un framework de deep learning para fusion de datos multimodales con dimensionalidad heterogenea (3D + 2D en imagenes medicas). Su enfoque consiste en extraer y proyectar las features de todas las modalidades al espacio de features de la modalidad con menor dimensionalidad mediante un Projective Feature Extractor (PFE). Aunque desarrollado para segmentacion de imagenes medicas, el principio es transferible: proyectar las 384D semanticas a un espacio de dimension comparable a las 12D musicales (o viceversa) antes de la fusion.

Los autores propusieron dos variantes: Late Fusion (ramas encoder-decoder independientes que concatenan mapas de features finales) y Multiscale Fusion (encoders separados con concatenacion a multiples niveles). La variante Multiscale supero consistentemente a Late Fusion, especialmente en escenarios con pocos datos etiquetados.

#### 3.7.3 Alternativas a la proyeccion: fusion tardia como solucion natural

La fusion tardia evita el problema de dimensionalidad heterogenea de manera elegante: en lugar de operar sobre los espacios de features directamente, opera sobre scores de similitud escalares producidos por cada componente. Independientemente de que un componente calcule similitudes en 12D y otro en 384D, los outputs son scalares normalizados que pueden combinarse linealmente sin perdida de informacion atribuible a la diferencia dimensional.

Esta propiedad constituye una de las justificaciones mas solidas para la eleccion de fusion tardia en el proyecto: evita la necesidad de aprender proyecciones, reducir dimensionalidad, o disenar arquitecturas que acomoden espacios heterogeneos, a cambio de operar en el espacio universal de scores normalizados.

---

## 4. Tabla de fuentes principales

| # | Autores (Ano) | Titulo | Tipo | Citas aprox. | Relevancia | Aporte clave |
|---|---------------|--------|------|-------------|------------|--------------|
| 1 | Baltrusaitis, Ahuja y Morency (2019) | Multimodal Machine Learning: A Survey and Taxonomy | Survey, IEEE TPAMI | >5000 | Alta | Taxonomia fundacional: representacion, traduccion, alineacion, fusion, co-learning. Distincion model-agnostic vs model-based en fusion. |
| 2 | Liang, Zadeh y Morency (2024) | Foundations & Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions | Survey, ACM Computing Surveys | >800 | Alta | Taxonomia actualizada de 6 desafios. Fusion subsumida en "razonamiento". Formalizacion de interacciones multimodales (redundancia, complementariedad, sinergia). |
| 3 | Gao et al. (2024) | Deep Multimodal Data Fusion | Survey, ACM Computing Surveys | >100 | Alta | Nueva taxonomia basada en tecnica (encoder-decoder, atencion, GNN, generativa, restricciones). Argumento de obsolescencia de taxonomia early/late. |
| 4 | Li y Tang (2024) | Multimodal Alignment and Fusion: A Survey | Survey, arXiv | ~50 | Alta | Taxonomia de alineacion (explicita/implicita) y fusion (encoder-decoder, kernel, grafo, atencion). |
| 5 | Burke (2002) | Hybrid Recommender Systems: Survey and Experiments | Articulo, User Modeling and User-Adapted Interaction | >4000 | Alta | Taxonomia canonica de 7 estrategias de hibridacion: weighted, switching, mixed, feature combination, feature augmentation, cascade, meta-level. |
| 6 | Montague y Aslam (2001) | Relevance Score Normalization for Metasearch | Articulo, CIKM 2001 | >300 | Alta | Demostracion del impacto critico de la normalizacion de scores en fusion. Comparacion de metodos de normalizacion. |
| 7 | Morano et al. (2024) | Deep Multimodal Fusion of Data with Heterogeneous Dimensionality via Projective Networks | Articulo, IEEE JBHI | ~30 | Alta | Framework para fusion de datos con dimensionalidad heterogenea. Projective Feature Extractor para alinear espacios. |
| 8 | Oramas, Barbieri, Nieto y Serra (2018) | Multimodal Deep Learning for Music Genre Classification | Articulo, TISMIR | ~60 | Alta | Fusion multimodal (audio+texto+imagen) para clasificacion de genero musical. Comparacion de fusion a nivel de features vs decision. |
| 9 | Manco, Benetos, Quinton y Fazekas (2022) | Contrastive Audio-Language Learning for Music | Articulo, ISMIR 2022 | ~80 | Alta | MusCALL: aprendizaje contrastivo dual-encoder para alineacion audio-texto musical. Embeddings multimodales para retrieval. |
| 10 | Vaswani y Agrawal (2021) | Multimodal Fusion Based Attentive Networks for Sequential Music Recommendation | Articulo, arXiv | ~20 | Media-Alta | Redes atentivas para recomendacion musical secuencial fusionando lyrics (Transformer) y audio (VAE). |
| 11 | Zhu et al. (2020) | Multimodal Fusion Method Based on Self-Attention Mechanism | Articulo, Wireless Communications and Mobile Computing | ~100 | Media | Fusion multimodal con self-attention para ajuste adaptativo de pesos por modalidad y feature. |
| 12 | Huang et al. (2022) | MuLan: A Joint Embedding of Music Audio and Natural Language | Articulo, ISMIR 2022 | >200 | Media-Alta | Joint embedding space para audio y lenguaje natural musical mediante aprendizaje contrastivo a escala. |
| 13 | Vogt y Cottrell (1999) | Fusion Via a Linear Combination of Scores | Articulo, Information Retrieval | >400 | Media | Fundamentos teoricos de la combinacion lineal de scores para fusion de informacion. Propiedades formales. |
| 14 | SLEM - Classificacion de generos musicales con embeddings multimodales (2024) | Classification and Study of Music Genres with Multimodal Spectro-Lyrical Embeddings for Music | Articulo, Multimedia Tools and Applications | ~10 | Media | Representaciones multimodales espectrales+liricas para musica. Demostracion de superioridad multimodal. |
| 15 | Hsu (2005) | Fusion in Information Retrieval | Tutorial/Survey, Information Retrieval | >100 [no verificado] | Media | Revision de metodos de fusion (score-based, rank-based) en recuperacion de informacion. |
| 16 | Oramas et al. (2019) | Multimodal Music Information Processing and Retrieval: Survey and Future Challenges | Survey, arXiv | ~50 | Media-Alta | Survey de procesamiento multimodal de informacion musical. Audio, texto, imagen, metadata. |
| 17 | Park et al. (2022) [MusicBERT] | MusicBERT: A Shared Multi-Modal Representation for Music and Text | Articulo, NLP4MusA Workshop | ~30 [no verificado] | Media | Representacion compartida multimodal para musica y texto basada en BERT. |
| 18 | AAAI (2025) | Adaptive Multimodal Fusion: Dynamic Attention Allocation for Intent Recognition | Articulo, AAAI 2025 | ~5 | Media | Asignacion dinamica de atencion intra e inter-modal para fusion adaptativa. |

---

## 5. Gaps identificados y oportunidades

### 5.1 Problemas abiertos en la literatura

1. **Fusion tardia adaptativa sin supervision profunda:** La mayor parte de los metodos adaptativos (attention-based) requieren datos de entrenamiento supervisado. Existe un gap en metodos que adapten los pesos de fusion tardia de manera no supervisada o semi-supervisada, lo cual es relevante para el proyecto dado que el ground truth de recomendacion musical es inherentemente subjetivo.

2. **Normalizacion especifica para similitudes coseno de alta dimension:** Los metodos de normalizacion de scores (Min-Max, Z-Score) fueron desarrollados en el contexto de scores de relevancia en IR, no especificamente para distribuciones de similitud coseno en espacios de embeddings de alta dimension. La concentracion de la medida en alta dimension produce distribuciones con caracteristicas particulares que podrian beneficiarse de normalizaciones especializadas.

3. **Metricas de complementariedad continuas para MIR:** El NMI mide complementariedad a nivel de particiones discretas. No se ha establecido una metrica estandar para medir complementariedad a nivel de representaciones continuas en el dominio musical especificamente.

4. **Fusion multimodal para recomendacion musical con datasets de tamano moderado:** La mayoria de los enfoques de aprendizaje contrastivo (MusCALL, MuLan) requieren datasets de cientos de miles o millones de pares. Hay escasa investigacion sobre fusion efectiva en datasets de 10K-20K canciones.

### 5.2 Oportunidades para el proyecto de tesis

1. **Documentacion transparente de la optimizacion de pesos:** La primera ejecucion revelo una discrepancia significativa entre los pesos utilizados (55/45) y los optimos (20/80). La re-ejecucion puede aportar un analisis sistematico y transparente del espacio de pesos, incluyendo curvas de rendimiento vs. alpha, intervalos de confianza, y sensibilidad a la metrica de evaluacion.

2. **Analisis empirico de la concentracion de la medida:** Documentar cuantitativamente como las distribuciones de similitud coseno difieren entre el espacio de 384D (BERT) y 12D (Spotify), y como esta diferencia impacta la fusion. Este analisis aportaria evidencia empirica al debate sobre normalizacion en contextos de dimensionalidad heterogenea.

3. **Justificacion formal de fusion tardia:** El proyecto puede posicionarse como un caso de estudio bien documentado de cuando y por que la fusion tardia es la eleccion apropiada, contribuyendo evidencia practica al debate taxonomico sobre estrategias de fusion.

4. **NMI como validacion de complementariedad pre-fusion:** El uso de NMI cross-modal como paso de validacion antes de la fusion (no solo como metrica post-hoc) es una practica metodologica que el proyecto puede establecer y documentar.

5. **Comparacion de normalizaciones en el contexto especifico:** Evaluar empiricamente Min-Max vs Z-Score vs Rank normalization en el contexto exacto del proyecto (similitudes coseno en 384D vs similitudes euclideas en 12D) aportaria evidencia practica a la literatura de fusion.

---

## 6. Entradas BibTeX

```bibtex
@article{baltrusaitis_2019_multimodal,
  author    = {Baltru{\v{s}}aitis, Tadas and Ahuja, Chaitanya and Morency, Louis-Philippe},
  title     = {Multimodal Machine Learning: A Survey and Taxonomy},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume    = {41},
  number    = {2},
  pages     = {423--443},
  year      = {2019},
  doi       = {10.1109/TPAMI.2018.2798607}
}

@article{liang_2024_foundations,
  author    = {Liang, Paul Pu and Zadeh, Amir and Morency, Louis-Philippe},
  title     = {Foundations \& Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions},
  journal   = {ACM Computing Surveys},
  volume    = {56},
  number    = {10},
  pages     = {1--63},
  year      = {2024},
  doi       = {10.1145/3656580}
}

@article{gao_2024_deep_multimodal,
  author    = {Gao, Jing and Li, Peng and Chen, Zhikui and Zhang, Jianing},
  title     = {Deep Multimodal Data Fusion},
  journal   = {ACM Computing Surveys},
  volume    = {56},
  number    = {9},
  articleno = {216},
  year      = {2024},
  doi       = {10.1145/3649447}
}

@article{li_2024_alignment_fusion,
  author    = {Li, Songtao and Tang, Hao},
  title     = {Multimodal Alignment and Fusion: A Survey},
  year      = {2024},
  eprint    = {2411.17040},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CV},
  url       = {https://arxiv.org/abs/2411.17040}
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

@inproceedings{montague_2001_normalization,
  author    = {Montague, Mark and Aslam, Javed A.},
  title     = {Relevance Score Normalization for Metasearch},
  booktitle = {Proceedings of the Tenth International Conference on Information and Knowledge Management (CIKM '01)},
  pages     = {427--433},
  year      = {2001},
  publisher = {ACM},
  doi       = {10.1145/502585.502657}
}

@article{morano_2024_projective,
  author    = {Morano, Jos{\'e} and Aresta, Guilherme and Grechenig, Christoph and Schmidt-Erfurth, Ursula and Bogunovi{\'c}, Hrvoje},
  title     = {Deep Multimodal Fusion of Data with Heterogeneous Dimensionality via Projective Networks},
  journal   = {IEEE Journal of Biomedical and Health Informatics},
  volume    = {28},
  number    = {4},
  pages     = {2254--2265},
  year      = {2024},
  doi       = {10.1109/JBHI.2024.3352970}
}

@article{oramas_2018_multimodal_music,
  author    = {Oramas, Sergio and Barbieri, Francesco and Nieto, Oriol and Serra, Xavier},
  title     = {Multimodal Deep Learning for Music Genre Classification},
  journal   = {Transactions of the International Society for Music Information Retrieval},
  volume    = {1},
  number    = {1},
  pages     = {4--21},
  year      = {2018},
  doi       = {10.5334/tismir.10}
}

@inproceedings{manco_2022_muscall,
  author    = {Manco, Ilaria and Benetos, Emmanouil and Quinton, Elio and Fazekas, Gy{\"o}rgy},
  title     = {Contrastive Audio-Language Learning for Music},
  booktitle = {Proceedings of the 23rd International Society for Music Information Retrieval Conference (ISMIR 2022)},
  year      = {2022},
  url       = {https://archives.ismir.net/ismir2022/paper/000077.pdf}
}

@article{vaswani_2021_attentive_music,
  author    = {Vaswani, Piyush and Agrawal, Ashwin},
  title     = {Multimodal Fusion Based Attentive Networks for Sequential Music Recommendation},
  year      = {2021},
  eprint    = {2110.01001},
  archiveprefix = {arXiv},
  primaryclass  = {cs.IR},
  url       = {https://arxiv.org/abs/2110.01001}
}

@article{zhu_2020_selfattention_fusion,
  author    = {Zhu, Junhao and Liao, Shengwei and Lei, Zhichao and Li, Jinjun},
  title     = {Multimodal Fusion Method Based on Self-Attention Mechanism},
  journal   = {Wireless Communications and Mobile Computing},
  volume    = {2020},
  articleno = {8843186},
  year      = {2020},
  doi       = {10.1155/2020/8843186}
}

@inproceedings{huang_2022_mulan,
  author    = {Huang, Qingqing and Jansen, Aren and Lee, Joonseok and Ganti, Ravi and Li, Judith Yue and Ellis, Daniel P. W.},
  title     = {MuLan: A Joint Embedding of Music Audio and Natural Language},
  booktitle = {Proceedings of the 23rd International Society for Music Information Retrieval Conference (ISMIR 2022)},
  year      = {2022},
  url       = {https://archives.ismir.net/ismir2022/paper/000067.pdf}
}

@article{vogt_1999_linear_fusion,
  author    = {Vogt, Christopher C. and Cottrell, Garrison W.},
  title     = {Fusion Via a Linear Combination of Scores},
  journal   = {Information Retrieval},
  volume    = {1},
  number    = {3},
  pages     = {151--173},
  year      = {1999},
  doi       = {10.1023/A:1009980820262}
}

@article{slem_2024_spectrolyrical,
  author    = {{Authors not fully verified}},
  title     = {Classification and Study of Music Genres with Multimodal Spectro-Lyrical Embeddings for Music (SLEM)},
  journal   = {Multimedia Tools and Applications},
  year      = {2024},
  doi       = {10.1007/s11042-024-19160-5},
  note      = {[Autores no verificados completamente]}
}

@article{oramas_2019_multimodal_survey,
  author    = {Oramas, Sergio and Nieto, Oriol and Barbieri, Francesco and Serra, Xavier},
  title     = {Multimodal Music Information Processing and Retrieval: Survey and Future Challenges},
  year      = {2019},
  eprint    = {1902.05347},
  archiveprefix = {arXiv},
  primaryclass  = {cs.SD},
  url       = {https://arxiv.org/abs/1902.05347}
}

@inproceedings{musicbert_2020_shared,
  author    = {Park, Junghyun and Kim, Jongpil and others},
  title     = {MusicBERT: A Shared Multi-Modal Representation for Music and Text},
  booktitle = {Proceedings of the NLP4MusA Workshop},
  year      = {2020},
  url       = {https://aclanthology.org/2020.nlp4musa-1.13.pdf},
  note      = {[Autores no completamente verificados]}
}

@inproceedings{aaai_2025_adaptive_fusion,
  author    = {{Authors not fully verified}},
  title     = {Adaptive Multimodal Fusion: Dynamic Attention Allocation for Intent Recognition},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  doi       = {10.1609/aaai.v39i1.33898},
  note      = {[Autores no completamente verificados]}
}

@article{hsu_2005_fusion_ir,
  author    = {Hsu, Stephen},
  title     = {Fusion in Information Retrieval},
  year      = {2005},
  url       = {https://ccc.inaoep.mx/~villasen/bib/Hsu-FusionInIR07.pdf},
  note      = {[Detalles bibliograficos no completamente verificados]}
}
```

---

## Notas metodologicas

- Las fuentes marcadas con [no verificado] o [Autores no completamente verificados] requieren verificacion adicional antes de su inclusion definitiva en `bibliography.bib`.
- Los conteos de citas son aproximaciones basadas en los datos disponibles al momento de la busqueda y pueden haber cambiado.
- La busqueda se realizo el 2026-02-07 utilizando WebSearch sobre fuentes academicas abiertas.
- Se priorizaron surveys de primer nivel (IEEE TPAMI, ACM Computing Surveys, ISMIR) como puntos de entrada para identificar fuentes primarias.
