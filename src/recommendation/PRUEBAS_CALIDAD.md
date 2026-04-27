# PRUEBAS_CALIDAD.md — Evaluación cualitativa del recomendador

Documento de trabajo sobre las pruebas cualitativas realizadas al sistema de recomendación y las pruebas pendientes. Complementa las métricas automáticas de `results/metrics/etapa5_recommendation.json`.

---

## 1. CONTEXTO Y MOTIVACIÓN

Las métricas automáticas del sistema (P@10 = 0.4447, ILD, cobertura) usan género musical como proxy de ground truth. Este proxy tiene limitaciones reconocidas: una recomendación puede ser excelente aunque cruce géneros, y una recomendación del mismo género puede ser mala. Por eso complementamos con evaluación cualitativa.

**Estado actual**: 5 pruebas cualitativas manuales realizadas sobre el sistema al α óptimo (0.80). Evaluación hecha combinando análisis humano + verificación de letras via web.

**Pendiente**: pruebas sistemáticas de ablación (aporte de cada componente), comparación contra baselines (TF-IDF, random), probes temáticos específicos, y análisis cross-language.

---

## 2. PRUEBAS REALIZADAS (2026-04-20)

### Metodología

- Ejecución vía `python -m src.recommendation.try_recommender`
- 5 queries representativas de distintos géneros e idiomas
- Top 10 recomendaciones al α = 0.80 (óptimo por grid search)
- Análisis manual + verificación de letras en fuentes web (letras.com, songfacts, songmeaningsandfacts)

### 2.1 Query: "Shape of You" — Ed Sheeran (idx 748, etiquetado `latin`)

**Observación del dataset**: Ed Sheeran etiquetado como "latin" es un error de clasificación (probablemente scrapeado de playlist latina que incluía el remix reggaetonero). Esto afecta la métrica OK/-- pero no la calidad real.

**Top 10 recomendaciones**:
| # | Canción | Artista | Género | Score |
|---|---------|---------|--------|-------|
| 1 | Shape of You - Galantis Remix | Ed Sheeran | edm | 0.9660 |
| 2 | Baby I'm Yours | Breakbot | pop | 0.9148 |
| 3 | On The Low | Burna Boy | r&b | 0.9090 |
| 4 | in my miNd | Maty Noyes | pop | 0.9076 |
| 5 | Because Of You | Ne-Yo | r&b | 0.9063 |
| 6-7 | I Feel It Coming (x2) | The Weeknd | pop | 0.9059 |
| 8 | Can't Stop Your Lovin' | Poolside | pop | 0.9043 |
| 9 | Rock Wit U | Ashanti | r&b | 0.9038 |
| 10 | Get Buck | Young Buck | rap | 0.9037 |

**Análisis**:
- Cluster detectado: pop/r&b mainstream romántico de los 2010s
- Match dominante: **vocabulario y registro de pop romántico** (no necesariamente temática profunda)
- Pregunta abierta: ¿el sistema encuentra estas canciones por similitud semántica real o por estilo superficial compartido?
- Outlier claro: "Get Buck" (Young Buck) — rap agresivo, no encaja
- Duplicado: "I Feel It Coming" aparece 2 veces (problema del dataset)

### 2.2 Query: "Lose Yourself" — Eminem (idx 998, `rap`)

**Top 10 recomendaciones**:
| # | Canción | Artista | Género | Score |
|---|---------|---------|--------|-------|
| 1-2 | Lose Yourself (duplicados) | Eminem | rap | 1.0000 |
| 3 | Future | DJ Khaled | rap | 0.9398 |
| 4 | ICONIC | Logic | rap | 0.9278 |
| 5 | APESHIT | The Carters | r&b | 0.9237 |
| 6 | MotorSport | Migos | r&b | 0.9235 |
| 7 | APESHIT (duplicado) | The Carters | rap | 0.9234 |
| 8-9 | 'Till I Collapse (x2) | Eminem | rap | 0.9224 |
| 10 | Get Back | Ludacris | rap | 0.9192 |

**Análisis**:
- Cluster detectado: rap motivacional/aspiracional
- Match semántico **fuerte**: temática coherente de perseverancia, éxito, superación
- Till I Collapse es especialmente relevante (otra canción motivacional icónica de Eminem)
- Problema crítico: **4 duplicados en top 10** (Lose Yourself x2, APESHIT x2, Till I Collapse x2) reducen la utilidad real a 6 canciones únicas

### 2.3 Query: "Vuelve" — Ricky Martin (idx 8323, `latin`) — **MEJOR CASO**

**Top 10 recomendaciones**:
| # | Canción | Artista | Género | Score |
|---|---------|---------|--------|-------|
| 1 | Vuelve - MTV Unplugged | Ricky Martin | latin | 0.9224 |
| 2 | No Te Olvidaré | Gloria Estefan | latin | 0.9069 |
| 3 | Cicatrices | Kalimba | latin | 0.9049 |
| 4 | Me Dediqué a Perderte | Alejandro Fernández | latin | 0.9031 |
| 5 | Just the Way You Are | Billy Joel | rock | 0.9007 |
| 6 | The Scientist | Coldplay | pop | 0.9004 |
| 7 | At The Same Time | Eric Roberson | r&b | 0.8999 |
| 8 | Si Te Vas | Lit Killah | rap | 0.8997 |
| 9 | Flotando | Francisca Valenzuela | latin | 0.8992 |
| 10 | No Te Cuesta Nada | Javiera Mena | pop | 0.8980 |

**Análisis detallado con verificación de letras**:

- **Matches temáticos impecables en español**: Gloria Estefan (no olvidar a alguien), Alejandro Fernández (arrepentimiento por perder), Kalimba ("Cicatrices" — heridas emocionales). Todas baladas sobre pérdida amorosa.
- **Si Te Vas** (Lit Killah): rap en español temáticamente alineado (pérdida/separación). Cruce de género válido.
- **Just the Way You Are** (Billy Joel): CORRECCIÓN — esta canción NO es sobre pérdida sino sobre **aceptación incondicional**. El match es principalmente **sonoro** (balada de piano en inglés), no temático-semántico. Match de menor calidad de lo que parece.
- **The Scientist** (Coldplay): sí es sobre pérdida/arrepentimiento amoroso. Match semántico válido cross-language.

**Veredicto**: este es el caso más fuerte de evidencia de recomendación semántica real. El sistema encontró **baladas de pérdida amorosa en español e inglés** — lo cual es justamente el producto multimodal funcionando.

### 2.4 Query: "Piano Man" — Billy Joel (idx 4919, `rock`)

**Top 10 recomendaciones**:
| # | Canción | Artista | Género | Score |
|---|---------|---------|--------|-------|
| 1-2 | Piano Man (duplicados, labels pop/rock) | Billy Joel | pop/rock | 0.97-0.96 |
| 3 | The Art of Suicide | Emilie Autumn | pop | 0.8860 |
| 4 | Tiny Dancer | Elton John | rock | 0.8855 |
| 5 | Dream On | Aerosmith | rock | 0.8800 |
| 6 | Ghetto Man | Marvin Sease | rap | 0.8796 |
| 7 | Mr. Blue Sky | ELO | rock | 0.8795 |
| 8, 10 | Suite: Judy Blue Eyes (x2) | CSN | rock | 0.8782 |
| 9 | Hallelujah | Leonard Cohen | rock | 0.8779 |

**Análisis**:
- Cluster detectado: **baladas narrativas clásicas** con piano/storytelling
- Matches fuertes: Tiny Dancer, Hallelujah, Suite Judy Blue Eyes — todos narrativos storytelling
- **Ghetto Man** (Marvin Sease) — CORRECCIÓN: originalmente lo descarté como error. Es una canción narrativa de primera persona sobre un hombre modesto. Match temático válido en formato narrativo (aunque sonoramente muy distinto). No es un error claro.
- Duplicados: Piano Man x2 (misma canción, etiquetas pop y rock distintas), Suite Judy Blue Eyes x2

### 2.5 Query: "Welcome To The Machine" — Pink Floyd (idx 2273, `rock`)

**Top 10 recomendaciones**:
| # | Canción | Artista | Género | Score |
|---|---------|---------|--------|-------|
| 1 | Mo Better | Raheem DeVaughn | r&b | 0.8702 |
| 2-3 | Suite: Judy Blue Eyes (x2) | CSN | rock | 0.8643 |
| 4 | Comfortably Numb - 2011 | Pink Floyd | rock | 0.8603 |
| 5 | Time | Pink Floyd | rock | 0.8589 |
| 6 | Tiny Dancer | Elton John | rock | 0.8586 |
| 7 | Holocene | Bon Iver | pop | 0.8563 |
| 8 | Comfortably Numb | Pink Floyd | rock | 0.8562 |
| 9 | Father To Son | Queen | rock | 0.8556 |
| 10 | Wind Of Change | Scorpions | rock | 0.8529 |

**Análisis con verificación de letras**:
- **Mo Better** (Raheem DeVaughn): confirmado como outlier. Canción R&B sobre amor/intimidad/crecimiento personal. No match temático con la crítica distópica de Pink Floyd.
- **Holocene** (Bon Iver): CORRECCIÓN — inicialmente lo llamé "match excelente". En realidad Holocene es sobre humildad existencial y redención personal; Welcome To The Machine es sobre deshumanización corporativa. Match **sonoro/atmosférico** (ambas etéreas, reflexivas), no temático.
- Matches fuertes: Comfortably Numb y Time (Pink Floyd) — mismo registro del álbum Wish You Were Here / Dark Side of the Moon.
- Duplicados: Comfortably Numb x2, Suite Judy Blue Eyes x2

---

## 3. ERRORES DETECTADOS EN EL ANÁLISIS PREVIO

Durante la verificación de letras se identificaron errores de aserción en el análisis cualitativo inicial:

| Afirmación original | Corrección |
|---------------------|------------|
| "Baby I'm Yours de Breakbot es electro-funk francés" | Artista francés pero **canción en inglés**. El match con Shape of You sigue siendo válido pero por otro motivo |
| "Just the Way You Are es match temático perfecto para Vuelve" | Match **sonoro** (balada piano en inglés), no temático. Just the Way You Are es sobre aceptación, Vuelve es sobre pérdida |
| "Holocene es match excelente para Welcome To The Machine" | Match **atmosférico/sonoro**, no temático. Temas existenciales distintos |
| "Ghetto Man para Piano Man es un error claro" | En realidad hay un match válido en **formato narrativo**. Revisar sin sonoramente es distinto |

**Lección metodológica**: al evaluar recomendaciones hay que distinguir entre tipos de similitud:
1. **Sonora**: instrumentación, tempo, producción similares
2. **Temática**: letras sobre el mismo tipo de situación
3. **De registro lírico**: estilo narrativo, vocabulario, tono
4. **De género**: mismo cluster industrial
5. **De función**: canciones para el mismo contexto

El sistema está captando combinaciones de estas dimensiones en cada recomendación. No todas son "match temático" — muchas son match sonoro o de registro. Esta distinción **no se hizo** en el análisis inicial y llevó a aserciones sobre confiadas.

---

## 4. PROBLEMAS DETECTADOS EN EL DATASET (no en el algoritmo)

### 4.1 Duplicados

Canciones idénticas aparecen con `track_id` distintos. Verificadas en las pruebas:
- "Lose Yourself" (Eminem): 2 entradas
- "'Till I Collapse" (Eminem): 2 entradas
- "APESHIT" (The Carters): 2 entradas (una rap, una r&b)
- "Piano Man" (Billy Joel): 2 entradas (una pop, una rock)
- "Suite: Judy Blue Eyes" (CSN): 2 entradas
- "I Feel It Coming" (The Weeknd): 2 entradas
- "Comfortably Numb" (Pink Floyd): 2 entradas

**Impacto**: reduce la diversidad real del top 10. Típicamente 2-4 de cada 10 recomendaciones son duplicados funcionales.

**Causa probable**: dataset fuente (Kaggle spotify_songs) scrapeó la misma canción desde playlists diferentes con `track_id` ligeramente distintos.

**Propuesta de mitigación** (no implementada):
- Deduplicar en preprocesamiento por (nombre_normalizado, artista_normalizado)
- O filtrar en post-procesamiento: si dos recomendaciones comparten título normalizado, conservar solo la de mayor score

### 4.2 Etiquetas de género inconsistentes

Casos documentados:
- Ed Sheeran "Shape of You" → etiquetado `latin` (debería ser pop)
- Billy Joel "Piano Man" → aparece con `pop` y `rock` en distintas filas
- The Carters "APESHIT" → aparece con `rap` y `r&b`

**Impacto**: afecta la métrica P@K (muchas recomendaciones buenas marcadas como `--`). La calidad real no es afectada.

**Causa**: playlists de Spotify cruzan géneros; el scrapper asigna género según playlist, no según contenido.

---

## 5. PRUEBAS PENDIENTES

Ordenadas por prioridad. Cada prueba responde una pregunta específica que las pruebas cualitativas no respondieron.

### 5.1 PRUEBA A — Ablación visual por α (prioridad alta, esfuerzo bajo)

**Pregunta**: ¿qué aporta específicamente cada componente (semántico vs musical)?

**Metodología**:
1. Seleccionar 5-6 queries representativas (cubriendo géneros e idiomas)
2. Para cada query, generar top 10 recomendaciones con α ∈ {0.0, 0.5, 0.8, 1.0}
3. Poner las 4 listas lado a lado
4. Identificar:
   - Canciones que aparecen solo en α=1.0 → aporte puro del semántico
   - Canciones que aparecen solo en α=0.0 → aporte puro del musical
   - Canciones que aparecen en todas las listas → matches robustos
5. Verificar letras de casos críticos

**Entregable**: reporte markdown con las 4 listas por query y análisis de qué cambia al mover α.

**Tiempo estimado**: 30-60 minutos (script + ejecución + análisis).

### 5.2 PRUEBA B — Baselines numéricos (prioridad crítica, esfuerzo medio)

**Pregunta**: ¿el BERT realmente aporta sobre métodos más simples? ¿El sistema es mejor que baselines triviales?

**Baselines a implementar**:
1. **Random total**: recomendar 10 canciones aleatorias
2. **Random-same-genre**: aleatorio del mismo género que la query
3. **Popularity-based**: siempre las 10 más populares (según `track_popularity`)
4. **TF-IDF sobre letras**: similitud coseno sobre TF-IDF, sin BERT
5. **Only musical** (α=0.0): ya lo tenemos (P@10 = 0.3648)
6. **Only semantic** (α=1.0): ya lo tenemos (P@10 = 0.3613)
7. **Nuestro sistema** (α=0.80): ya lo tenemos (P@10 = 0.4447)

**Interpretación esperada**:
- Si TF-IDF ≈ nuestro α=1.0: el BERT no aporta mucho sobre vocabulario superficial
- Si TF-IDF << nuestro α=1.0: el BERT capta algo más profundo
- Random-same-genre **probablemente** dará P@10 cercano a 1.0 por construcción — sirve para mostrar que el proxy de género tiene techo artificial
- Popularity: si es alto, indica sesgo de popularidad del catálogo; si es bajo, confirma que el sistema no solo recomienda populares

**Entregable**: nueva tabla LaTeX comparando todos los baselines, figura con barplot.

**Tiempo estimado**: 2-3 horas de implementación + ejecución.

### 5.3 PRUEBA C — Probes temáticos específicos (prioridad media, esfuerzo bajo)

**Pregunta**: ¿el sistema capta temas específicos o solo clusters amplios (género, idioma)?

**Metodología**:
1. Seleccionar 3-4 temas específicos y concretos:
   - Canciones sobre adicción (ej: Eminem "When I'm Gone", Amy Winehouse "Rehab")
   - Canciones sobre crecer / nostalgia (ej: Taylor Swift "Fifteen")
   - Canciones sobre protesta política (ej: Rage Against the Machine, Bob Dylan)
   - Canciones sobre muerte / duelo (ej: Eric Clapton "Tears in Heaven")
2. Para cada query, verificar que:
   - Las top 10 incluyen al menos 3-4 canciones temáticamente relacionadas
   - Las canciones relacionadas cruzan géneros/épocas
3. Documentar casos donde el sistema **no capta** el tema

**Entregable**: reporte con análisis temático por query.

**Tiempo estimado**: 1-2 horas.

### 5.4 PRUEBA D — Análisis cross-language (prioridad media, esfuerzo bajo)

**Pregunta**: ¿funciona realmente la capacidad multilingüe del E5, o el sistema está atado al idioma de la query?

**Metodología**:
1. Seleccionar 3-4 queries en inglés con temas universales:
   - Amor romántico
   - Fiesta / celebración
   - Tristeza / melancolía
   - Autoconfianza / empowerment
2. Para cada una, verificar cuántas de las top 10 son en español/portugués/alemán/otros
3. Si son pocas o ninguna, el sistema **no cruza idiomas** — la "multilingualidad" es más teórica que práctica
4. Repetir con queries en español: ¿recomienda canciones en inglés con el mismo tema?

**Entregable**: tabla de cruces de idiomas por query.

**Tiempo estimado**: 1 hora.

---

## 6. IMPLICACIONES PARA EL INFORME

Esta evaluación cualitativa y las pruebas pendientes afectan los siguientes puntos del informe:

### Para §6 Resultados

- Agregar sección de **análisis cualitativo** con casos específicos (el caso Vuelve como evidencia fuerte, casos problemáticos como outliers y duplicados)
- Si se implementa Prueba B (baselines), **rediseñar la tabla comparativa**: no solo v1 vs v2 sino todos los baselines
- Reportar honestamente los casos donde el sistema falla (outliers, duplicados)

### Para §7 Conclusiones y Limitaciones

- **§7.3 Limitaciones** debe incluir:
  - Evaluación exclusivamente con proxy de género (sin evaluadores humanos)
  - Duplicados no filtrados del dataset fuente
  - Etiquetas de género inconsistentes
  - Ambigüedad entre similitud sonora vs semántica en las recomendaciones
- **§7.4 Futuras Líneas**:
  - Evaluación con usuarios reales
  - Deduplicación automática del dataset
  - Separación explícita de componentes de similitud (sonora, temática, de registro)

### Para §5.7 Sistema de Recomendación

- No describir el comportamiento del sistema como más robusto de lo que la evidencia soporta
- Declarar que α=0.80 óptimo por P@K con género, pero el óptimo puede ser distinto con otras métricas
- Mencionar que solo-semántico y solo-musical dan P@10 casi idéntico (0.361 vs 0.365), sugiriendo que el valor está en la **complementariedad**, no en que el semántico sea fundamentalmente superior

---

## 7. HISTORIAL

| Fecha | Acción | Responsable |
|-------|--------|-------------|
| 2026-04-05 | Ejecución inicial del sistema (Etapa 5) | — |
| 2026-04-20 | 5 pruebas cualitativas iniciales (Shape of You, Lose Yourself, Vuelve, Piano Man, Welcome To The Machine) | — |
| 2026-04-20 | Verificación de letras via web, corrección de 4 errores de aserción | — |
| Pendiente | Prueba A — ablación por α | — |
| Pendiente | Prueba B — baselines numéricos | — |
| Pendiente | Prueba C — probes temáticos | — |
| Pendiente | Prueba D — cross-language | — |
