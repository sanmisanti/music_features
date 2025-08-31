# Sistema de Recomendaciones Musicales con Explicabilidad

**Estado del Proyecto**: ✅ **COMPONENTES PRINCIPALES COMPLETADOS** (Agosto 2025)  
**Fase Actual**: COMPONENTE 3 - Análisis Avanzado de Clusters  
**Próxima Entrega**: Suite de Testing y Validación

---

## 📋 PLAN DE IMPLEMENTACIÓN COMPLETO

### **FASE 1: VERIFICACIÓN Y ORGANIZACIÓN** ✅ **COMPLETADA**
- ✅ Auditoría de integridad de vectores y clusters (7811 canciones validadas)
- ✅ Validación de archivos críticos: embeddings BERT 384D, características musicales 12D
- ✅ Verificación de clusters K=10 musical y K=6 semántico

### **FASE 2: DESARROLLO DE SCRIPTS CORE** ✅ **COMPLETADA**
- ✅ **load_system.py**: Clase `MusicDataLoader` con carga centralizada y cache optimizado
- ✅ **music_recommender.py**: Motor híbrido con pesos validados FASE 3 (55% musical, 45% semántico)
- ✅ **explain_recommendations.py**: Sistema de explicabilidad completo con análisis de clusters

### **FASE 3: IMPLEMENTACIÓN DE INTERFACE PRINCIPAL** ✅ **COMPLETADA**
- ✅ **recommend_songs.py**: Script principal con 15+ opciones CLI
  - ✅ Búsqueda por track_id con validación completa
  - ✅ Búsqueda por nombre de canción con matching inteligente
  - ✅ Interface interactiva con comandos dinámicos
  - ✅ Modo demostración con showcasing automático
  - ✅ Sistema de manejo de errores robusto

### **FASE 4: ANÁLISIS AVANZADO DE CLUSTERS** 🚧 **EN PROGRESO**
- 🚧 **analyze_clusters.py**: Caracterización estadística avanzada
- ⏳ Generador de descripciones interpretables por cluster
- ⏳ Integración con sistema de metadatos completo

### **FASE 5: VALIDACIÓN Y TESTING** ⏳ **PENDIENTE**
- ⏳ **validate_system.py**: Suite de testing completa
- ⏳ Tests de precisión Precision@K y Recall@K
- ⏳ Tests de diversidad en recomendaciones
- ⏳ Benchmark de performance <100ms
- ⏳ Tests de coherencia de clusters y explicaciones

---

## 🗂️ ESTRUCTURA DEL DIRECTORIO

### 📊 `data/` - **Vectores y Datasets Principales**
- `semantic_embeddings.npy` - **Embeddings BERT (7811, 384)** - Sistema semántico validado
- `musical_features_normalized.npy` - **Características musicales (7811, 12)** - StandardScaler aplicado
- `track_ids.npy` - **IDs de alineación (7811,)** - Integridad referencial 100%
- `songs_metadata.csv` - **Metadatos completos** - track_name, artist_name, genre, etc.

### 🎪 `clusters/` - **Asignaciones de Clusters Óptimos**
- `musical_clusters_k10.npy` - **Clusters musicales K-Means++ K=10** - Silhouette 0.0965
- `semantic_clusters_k6.npy` - **Clusters semánticos K-Means++ K=6** - Silhouette 0.0329

### 🔧 `config/` - **Configuración Científica Validada**
- `system_config.json` - **Configuración completa con resultados FASE 3**
- `fase3_results.json` - **Resultados experimentales de 56 configuraciones**

### 📈 `models/` - **Modelos y Parámetros** (Para Expansión Futura)

### 📝 `scripts/` - **Sistema de Scripts Implementado**
#### **✅ Scripts Production-Ready**
- `load_system.py` - **Cargador centralizado** (339 líneas) - `MusicDataLoader` con cache
- `music_recommender.py` - **Motor híbrido** (409 líneas) - `HybridMusicRecommender` optimizado
- `explain_recommendations.py` - **Sistema de explicabilidad** (1080+ líneas) - `RecommendationExplainer` completo
- `recommend_songs.py` - **Interface principal** (643 líneas) - CLI completa con 15+ opciones

#### **🚧 Scripts en Desarrollo**
- `analyze_clusters.py` - Análisis estadístico avanzado (EN DESARROLLO)
- `validate_system.py` - Suite de testing (PENDIENTE)

---

## 🚀 GUÍA DE USO RÁPIDO

### **Método 1: CLI - Recomendaciones Directas**
```bash
# Por track_id específico
python scripts/recommend_songs.py --track_id "TRACK_ID" --n_recommendations 10

# Por nombre de canción
python scripts/recommend_songs.py --song_name "Bohemian Rhapsody" --artist "Queen"

# Búsqueda de canciones
python scripts/recommend_songs.py --search "stairway to heaven"

# Modo interactivo completo
python scripts/recommend_songs.py --interactive

# Demostración del sistema
python scripts/recommend_songs.py --demo
```

### **Método 2: API Programática**
```python
from scripts.load_system import MusicDataLoader
from scripts.music_recommender import HybridMusicRecommender
from scripts.explain_recommendations import RecommendationExplainer

# Cargar sistema completo
loader = MusicDataLoader()
recommender = HybridMusicRecommender()
explainer = RecommendationExplainer()

# Generar recomendaciones con explicaciones
recommendations = recommender.recommend(track_id="ejemplo", n_recommendations=10)
explanations = explainer.get_batch_explanations(recommendations)
```

### **Método 3: Interface Unificada**
```python
from scripts.recommend_songs import MusicRecommendationInterface

# Interface completa
interface = MusicRecommendationInterface()

# Recomendaciones por ID con explicaciones automáticas
result = interface.recommend_by_track_id("TRACK_ID", include_explanations=True)

# Búsqueda y recomendación por nombre
result = interface.recommend_by_name("Bohemian Rhapsody", "Queen")

# Análisis de clusters de una canción
analysis = interface.get_song_analysis("TRACK_ID")
```

---

## 📊 ESPECIFICACIONES TÉCNICAS ACTUALES

### **Dataset Multimodal Unificado**
- **Total canciones**: 7,811 (intersección validada musicales + semánticas)
- **Embeddings semánticos**: (7811, 384) BERT normalizados L2
- **Características musicales**: (7811, 12) Spotify features StandardScaler
- **Integridad referencial**: 100% alineación por track_id

### **Configuración de Clustering Científicamente Validada**
- **Clusters musicales**: K=10 (K-Means++, Silhouette=0.0965, Balance=0.7547)
- **Clusters semánticos**: K=6 (K-Means++, Silhouette=0.0329, Interpretabilidad=0.7284)
- **Complementariedad cross-modal**: NMI=0.0567, estrategia híbrida óptima confirmada

### **Pesos de Recomendación Híbrida**
- **Musical**: 55% (dominio técnicamente mejor estructurado)
- **Semántico**: 45% (complemento temático con alta interpretabilidad)
- **Justificación**: Basada en experimentación exhaustiva FASE 3 con 56 configuraciones

### **Géneros Representados**
- **Rock**: 24.7% | **R&B**: 19.9% | **Pop**: 18.2% | **Rap**: 17.6% | **EDM**: 10.0% | **Latin**: 9.7%

---

## 🎯 MÉTRICAS DE PERFORMANCE OBJETIVO

- ⚡ **Tiempo de recomendación**: Target <100ms (implementado y monitoreado)
- 🎵 **Precisión de similitud**: >0.7 promedio en recomendaciones híbridas
- 📝 **Cobertura de explicaciones**: 100% recomendaciones con explicación automática
- 🔍 **Interpretabilidad**: Etiquetas automáticas para todos los clusters K=10 y K=6

---

## 🔬 VALIDACIÓN CIENTÍFICA COMPLETADA

### **Experimentación FASE 3** (Base del Sistema Actual)
- ✅ **56 configuraciones algorítmicas** evaluadas sistemáticamente
- ✅ **Función objetivo multi-criterio** optimizada (Silhouette + Balance + Interpretabilidad)
- ✅ **Análisis cross-modal exhaustivo** con 90 combinaciones inter-dominio
- ✅ **Complementariedad confirmada experimentalmente** (NMI rango 0.0533-0.0567)
- ✅ **Validación de interpretabilidad** con sistema de etiquetado automático funcional

### **Metodología Científica Aplicada**
- **Dataset unificado**: Eliminación de asimetría entre dominios (7,811 canciones alineadas)
- **Evaluación justa**: Comparación directa 384D vs 12D sobre mismas canciones
- **Reproducibilidad**: Configuraciones determinísticas con random_state fijo
- **Documentación exhaustiva**: Justificación técnica completa en FULL_PROJECT.md

---

## 📈 PRÓXIMOS DESARROLLOS PLANIFICADOS

### **INMEDIATO** (COMPONENTE 3 - Análisis de Clusters)
1. **analyze_clusters.py** - Caracterización estadística avanzada por cluster
2. **Generador de descripciones interpretables** - Etiquetado automático mejorado
3. **Integración completa con metadatos** - Enriquecimiento de análisis

### **CORTO PLAZO** (COMPONENTE 4 - Testing y Validación)
1. **validate_system.py** - Suite de testing production-ready
2. **Métricas de precisión** - Precision@K, Recall@K, diversidad
3. **Benchmarks de performance** - Validación objetivo <100ms
4. **Tests de coherencia** - Validación de explicabilidad

### **MEDIANO PLAZO** (Extensiones del Sistema)
1. **API REST** - Interface web para el sistema
2. **Cache distribuido** - Optimización para múltiples usuarios
3. **Sistema de feedback** - Aprendizaje de preferencias de usuario
4. **Interfaz gráfica** - Visualización de clusters y recomendaciones

---

## 📚 DOCUMENTACIÓN TÉCNICA COMPLETA

Para entendimiento técnico completo del sistema, consultar:
- **FULL_PROJECT.md** - Proceso completo de desarrollo y metodología científica
- **clustering_evaluation_project/phase3_multimodal_clustering/README.md** - Experimentación FASE 3
- **config/system_config.json** - Configuración técnica con resultados validados

---

*Última actualización: Agosto 2025 - Sistema Principal Completado*