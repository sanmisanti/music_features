# 🎵 Módulo de Clustering Semántico de Letras Musicales

## 📋 Descripción

Este módulo implementa un sistema avanzado de clustering semántico para letras musicales multilingües, basado en modelos BERT state-of-the-art. Forma parte del sistema multimodal de análisis musical desarrollado para mejorar las recomendaciones musicales mediante análisis del contenido lírico.

## 🎯 Características Principales

### ✨ **Clustering Semántico Multilingüe**
- **Modelo BERT**: `paraphrase-multilingual-MiniLM-L12-v2` optimizado
- **Soporte multilingüe**: Inglés, Español, Alemán, Portugués
- **Clustering unificado**: Sin separación por idiomas para cross-cultural discovery

### ⚡ **Optimización CPU**
- **Sin GPU requerida**: Optimizado completamente para CPU
- **Cache inteligente**: Sistema multi-nivel (RAM + Disco + Database)
- **Performance**: <3 horas primera ejecución, <10 min re-ejecuciones

### 🔧 **Sistema Robusto**
- **Error handling**: Recuperación automática de fallos
- **Monitoring**: Performance tracking en tiempo real
- **Escalabilidad**: Probado con 16,081 canciones

## 📊 Dataset

- **Tamaño**: 16,081 canciones con letras
- **Distribución idiomas**: 84.4% EN, 7.5% ES, 1.3% DE, 1.0% PT
- **Formato**: CSV con separador '^', encoding UTF-8
- **Fuente**: Dataset optimizado del proyecto principal

## 🏗️ Arquitectura

```
clustering/algorithms/lyrics/
├── config/                 # 📋 Configuración centralizada
├── preprocessing/          # 🔧 Limpieza texto multilingüe
├── vectorization/          # 🧠 BERT embeddings + cache
├── clustering/             # 🎯 Algoritmos clustering semántico
├── evaluation/             # 📊 Métricas evaluación
├── recommendation/         # 🎵 Sistema recomendación
├── utils/                  # 🛠️ Utilidades
├── tests/                  # ✅ Suite testing completa
├── scripts/                # 🚀 Scripts usuario final
└── docs/                   # 📚 Documentación técnica
```

## 🚀 Instalación y Uso

### 1. **Instalación Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Configuración Inicial**
```bash
python scripts/setup_environment.py
```

### 3. **Ejecución Completa**
```bash
python scripts/run_lyrics_clustering.py
```

### 4. **Análisis Resultados**
```bash
python scripts/analyze_clusters.py
```

## 📈 Performance

### **Especificaciones Técnicas**
- **CPU**: Intel i5/i7 estándar (4+ cores recomendado)
- **RAM**: 4GB mínimo, 8GB recomendado
- **Almacenamiento**: 5GB para cache completo

### **Métricas Target**
- **Throughput**: >4 canciones/segundo
- **Silhouette Score**: >0.3
- **Topic Coherence**: >0.6
- **Cross-lingual Consistency**: >75%

## 🔬 Metodología Científica

### **Algoritmos Implementados**
1. **Hopkins Statistic**: Validación clustering readiness
2. **K-Means Cosine**: Clustering base optimizado
3. **Hierarchical Ward**: Análisis jerárquico temas
4. **HDBSCAN**: Detección automática clusters densos

### **Métricas Evaluación**
- **Coherencia Temática**: Topic Coherence Score (LDA)
- **Diversidad Semántica**: Intra/Inter-cluster balance
- **Consistencia Multilingüe**: Cross-lingual validation
- **Benchmarking**: Comparación vs TF-IDF, Word2Vec

## 🧪 Testing

### **Suite Tests Completa**
```bash
# Tests unitarios
pytest tests/test_preprocessing.py
pytest tests/test_vectorization.py
pytest tests/test_clustering.py

# Tests integración
pytest tests/test_integration.py

# Coverage completo
pytest --cov=clustering.algorithms.lyrics tests/
```

### **Validación Calidad**
```bash
python scripts/validate_quality.py
python scripts/benchmark_performance.py
```

## 📚 Documentación

- **[Plan de Implementación](IMPLEMENTATION_PLAN.md)**: Roadmap completo desarrollo
- **[Arquitectura Técnica](docs/architecture.md)**: Detalles técnicos sistema
- **[API Reference](docs/api_reference.md)**: Documentación completa API
- **[Guía Usuario](docs/user_guide.md)**: Manual uso sistema

## 🤝 Integración

### **Sistema Musical Existente**
Este módulo está diseñado para integrarse seamlessly con el clustering musical existente:
- **Interface compatible**: APIs estandarizadas
- **Weighted fusion**: Balance música vs letras
- **A/B testing**: Framework evaluación comparativa

### **Recomendaciones Híbridas**
```python
from clustering.algorithms.lyrics.recommendation import LyricsRecommender
from clustering.algorithms.musical import MusicalRecommender

# Sistema híbrido
hybrid_recommender = HybridRecommender(
    lyrics_weight=0.4,
    music_weight=0.6
)
```

## 🎓 Contexto Académico

Este módulo forma parte de la tesis de Ingeniería Informática enfocada en sistemas multimodales de recomendación musical. La metodología implementada está basada en literatura científica current en Music Information Retrieval y Natural Language Processing.

### **Contribuciones Científicas**
1. **Clustering multilingüe unificado** para letras musicales
2. **Optimización CPU extrema** para modelos BERT
3. **Sistema cache inteligente** multi-nivel
4. **Métricas evaluación especializadas** para dominio musical

## 📄 Licencia

Este proyecto es parte del sistema de análisis musical desarrollado para investigación académica.

---

## results/ - Resultados Vectorizacion

Outputs de tests/test_vectorization_analysis.py:

| Archivo | Contenido |
|---------|-----------|
| semantic_clustering_tsne.png | Visualizacion t-SNE de embeddings semanticos |
| cluster_distribution.png | Distribucion de clusters |
| analysis_report.json | Metricas: 8,567 embeddings, Silhouette 0.673 |

Fecha: 2025-08-19.

---

**Estado**: PRODUCCION - Sistema validado
**Versión**: 1.0.0
**Última actualización**: Diciembre 2025