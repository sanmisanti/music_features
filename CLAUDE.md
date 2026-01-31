# CLAUDE.md

Archivo de configuracion para Claude Code en este repositorio.

## DIRECTIVA CRITICA: EJECUCION DE SCRIPTS

**NUNCA ejecutar scripts, comandos o tests directamente.**
- SIEMPRE avisar al usuario antes de ejecutar cualquier comando
- ESPERAR que el usuario ejecute el script y muestre la salida
- DESPUES analizar los resultados y continuar segun corresponda
- Aplica a: python scripts, bash commands, tests, jupyter notebooks

## DIRECTIVA CRITICA: COMUNICACION PROFESIONAL

**NUNCA utilizar emojis, frases cortas o comunicacion informal.** Toda interaccion debe mantener rigor tecnico y profesionalismo de proyecto de tesis de Ingenieria Informatica. Respuestas como comunicacion entre ingenieros informaticos profesionales con explicaciones tecnicas fundamentadas.

**Estructura obligatoria**: Introduccion contextual, desarrollo tecnico con fundamentacion cientifica, analisis comparativo cuando aplique, implicaciones practicas, sintesis y conclusiones.

---

## ESTADO DEL PROYECTO: RE-EJECUCION EN CURSO

**Fase actual: Etapa 1 completada, iniciando Etapa 2 (Enero 2026)**

El proyecto se encuentra en proceso de re-ejecucion desde cero. La primera iteracion (2025) produjo resultados funcionales pero con problemas metodologicos identificados durante evaluacion academica. La re-ejecucion preserva el mismo dataset y objetivos, pero reconstruye codigo y documentacion con rigor mejorado.

### Objetivo de la re-ejecucion

Desarrollar el proyecto paso a paso, produciendo simultaneamente codigo funcional y el informe de tesis en LaTeX. Cada etapa genera artefactos de codigo y contenido de capitulos del informe, de modo que al completar la ultima etapa el informe este esencialmente terminado.

---

## METODOLOGIA DE RE-EJECUCION

### Principio central: Desarrollo y documentacion simultaneos

Cada etapa del proyecto produce DOS entregables:
1. **Codigo**: Modulos funcionales, validados, reproducibles
2. **LaTeX**: Secciones del informe escritas con los resultados obtenidos

Las tablas y figuras del informe se generan desde codigo. No se editan manualmente. Si un resultado cambia, se regenera el artefacto automaticamente.

### Etapas del proyecto

| Etapa | Descripcion | Capitulos LaTeX asociados |
|-------|-------------|---------------------------|
| 1. Fundacion | Estructura repositorio, esqueleto LaTeX, configuracion | Introduccion (borrador) |
| 2. Datos | Carga, exploracion, analisis descriptivo del dataset | Marco Teorico (datos), Solucion Propuesta (fuentes) |
| 3. Features | Vectorizacion BERT, features musicales, unificacion | Marco Teorico (NLP/BERT), Solucion Propuesta (vectorizacion) |
| 4. Clustering | Hopkins, evaluacion multi-algoritmo, purificacion | Marco Teorico (clustering), Solucion Propuesta (clustering), Resultados |
| 5. Recomendacion | Sistema hibrido, optimizacion pesos, evaluacion | Solucion Propuesta (integracion), Resultados |
| 6. Sintesis | Conclusiones, revision coherencia, compilacion final | Conclusiones, Apendices |

### Progreso de etapas

- [x] Etapa 1: Fundacion (estructura repositorio, esqueleto LaTeX 7 capitulos + 3 anexos, config centralizada, APA 7 via biblatex+biber)
- [ ] Etapa 2: Datos
- [ ] Etapa 3: Features
- [ ] Etapa 4: Clustering
- [ ] Etapa 5: Recomendacion
- [ ] Etapa 6: Sintesis

---

## LECCIONES APRENDIDAS DE LA PRIMERA EJECUCION

Problemas identificados que la re-ejecucion debe evitar:

### Problemas de datos
- El dataset final de 7,811 canciones fue resultado residual de filtros, no disenado intencionalmente. **Correccion**: Documentar presupuesto de perdida explicito en cada etapa del pipeline.
- El analisis de sesgo (linguistico, por genero, por instrumentalidad) se realizo post-hoc. **Correccion**: Analizar sesgo ANTES de cada filtrado, no despues.
- El 57.7% de perdida de datos no tenia justificacion intencional. **Correccion**: Registrar por que se excluye cada cancion.

### Problemas de codigo
- El performance predictor usaba 20 observaciones sinteticas hardcoded. **Correccion**: Solo usar resultados reales; eliminar datos inventados.
- Dos funciones devolvian valores placeholder (gap analysis, natural structure score). **Correccion**: Toda funcion devuelve resultados reales o se elimina.
- Asignacion de genero incorrecta en script de exportacion. **Correccion**: Validar integridad en cada paso de transformacion.

### Problemas de evaluacion
- Metricas proxy (genero como ground truth) sin declaracion explicita. **Correccion**: Declarar upfront que el ground truth es proxy, no verdad absoluta.
- Pesos de fusion suboptimos (55/45 vs 20/80 optimo) presentados como "decision de diseno". **Correccion**: Si se eligen pesos suboptimos, documentar como limitacion, no como decision positiva.
- Hipotesis formuladas despues de ver resultados (post-hoc). **Correccion**: Pre-registrar hipotesis antes de ejecutar experimentos.

### Problemas de documentacion
- Documentacion escrita retroactivamente, no junto al desarrollo. **Correccion**: Escribir LaTeX en cada etapa como entregable obligatorio.
- Sin controles de reproducibilidad (seeds, versionado de intermedios). **Correccion**: Seeds fijos en configuracion centralizada, resultados versionados.

---

## DATASET

El dataset base es el mismo de la primera ejecucion:

| Dataset | Ubicacion | Registros | Descripcion |
|---------|-----------|-----------|-------------|
| Fuente con letras | `data/2_with_lyrics/spotify_songs_fixed.csv` | 18,454 | Spotify Kaggle + Genius lyrics, separador `@@` |

Los datasets intermedios (3_selected, 4_vectorized, 5_unified) seran regenerados durante la re-ejecucion con el nuevo pipeline.

---

## MATERIAL DE REFERENCIA (PRIMERA EJECUCION)

El codigo y documentacion de la primera ejecucion permanecen en el repositorio como referencia. No se reutilizan directamente; se reimplementan con las correcciones indicadas.

### Documentacion de referencia

| Documento | Contenido | Utilidad |
|-----------|-----------|----------|
| `docs/FULL_PROJECT.md` | Documento maestro v1: metodologia, resultados, 56 configuraciones | Referencia de resultados y metodologia |
| `docs/SOLUCION_PROPUESTA_SEMANTICO.md` | Componente semantico completo | Referencia de estructura y contenido |
| `docs/SOLUCION_PROPUESTA_SEMANTICO.tex` | Version LaTeX del componente semantico | Template de formato LaTeX |
| `docs/SEMANTIC_RECOMMENDATIONS_ARCHITECTURE.md` | Decision vectores BERT directos vs clustering | Referencia arquitectural |

### Codigo de referencia

| Modulo | Ubicacion | Lineas | Utilidad |
|--------|-----------|--------|----------|
| Seleccion clustering-aware | `data_selection/clustering_aware/select_optimal_10k_from_18k.py` | 878 | Logica MaxMin + Hopkins a reimplementar |
| Vectorizacion BERT | `clustering/algorithms/lyrics/vectorization/bert_vectorizer.py` | ~300 | Arquitectura BERT a reimplementar |
| Funcion objetivo multi-criterio | `clustering/evaluation_project/phase3_.../config/evaluation_metrics.py` | ~300 | Metricas y pesos a reimplementar |
| Purificacion hibrida | `scripts/cluster_purification.py` | 841 | Algoritmo central a reimplementar |
| Sistema recomendacion | `recommendation_system/scripts/` | ~4800 | Motor hibrido a reimplementar |

### Resultados de referencia (primera ejecucion)

Estos resultados sirven como benchmark para validar la re-ejecucion:

| Metrica | Valor v1 | Notas |
|---------|----------|-------|
| Hopkins Semantico (384D) | 0.7752 +/- 0.0015 | Debe reproducirse similar |
| Hopkins Musical (12D) | 0.7871 +/- 0.0022 | Debe reproducirse similar |
| Silhouette post-purificacion | 0.2893 | Benchmark de mejora |
| Precision@10 hibrido | 0.398 | Benchmark de recomendacion |
| NMI cross-modal | 0.0567 | Benchmark de complementariedad |

---

## ESTRUCTURA DEL INFORME (ETAPA 1)

Formato: APA 7ma Edicion, compilacion con biblatex + biber.

| Archivo | Capitulo |
|---------|----------|
| `thesis/main.tex` | Documento maestro (preambulo, portada, inputs) |
| `thesis/chapters/01_introduccion.tex` | Cap 1: Introduccion |
| `thesis/chapters/03_estado_cuestion.tex` | Cap 2: Estado de la Cuestion |
| `thesis/chapters/04_definicion_problema.tex` | Cap 3: Definicion del Problema |
| `thesis/chapters/02_marco_teorico.tex` | Cap 4: Marco Teorico |
| `thesis/chapters/05_solucion_propuesta.tex` | Cap 5: Solucion Propuesta |
| `thesis/chapters/06_resultados.tex` | Cap 6: Resultados Experimentales |
| `thesis/chapters/07_conclusiones.tex` | Cap 7: Conclusiones y Futuras Lineas |
| `thesis/appendices/A_configuraciones.tex` | Anexo A: Configuraciones |
| `thesis/appendices/B_metricas.tex` | Anexo B: Metricas Detalladas |
| `thesis/appendices/C_reproducibilidad.tex` | Anexo C: Reproducibilidad |
| `thesis/bibliography.bib` | Referencias bibliograficas |

**Nota**: Los nombres de archivo (01\_, 02\_, etc.) no corresponden al orden de capitulos. El orden real esta definido por la secuencia de `\input` en `main.tex`.

### Configuracion centralizada

`src/config.py` define todos los paths, seeds, y parametros globales. Ningun script debe definir estos valores localmente.

---

## DIRECTIVA: ACTUALIZACION DE ESTE ARCHIVO

Este archivo se actualiza al completar cada etapa:
1. Marcar etapa completada en la seccion "Progreso de etapas"
2. Agregar referencias a nuevos modulos/documentos creados
3. Registrar decisiones tecnicas importantes tomadas durante la etapa
4. Mantener actualizada la seccion de lecciones aprendidas si surgen nuevas
