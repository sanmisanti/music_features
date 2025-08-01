# Modelos de Clustering de Letras

## 🎵 Propósito

Este directorio está preparado para albergar modelos de clustering específicos para el análisis de letras musicales, como parte del sistema multimodal de recomendación musical.

## 🔮 Implementación Futura

### Componentes Planificados

1. **Procesamiento de Texto**
   - Tokenización y limpieza de letras
   - Eliminación de stopwords específicas para música
   - Stemming/Lemmatización en múltiples idiomas

2. **Vectorización**
   - TF-IDF con parámetros optimizados para letras
   - Word2Vec pre-entrenado para música
   - Embeddings específicos del dominio musical

3. **Clustering de Letras**
   - K-Means adaptado para vectores de texto
   - Clustering jerárquico para temas musicales
   - Clustering por sentimientos/emociones

4. **Evaluación Especializada**
   - Coherencia temática intra-cluster
   - Diversidad semántica inter-cluster
   - Análisis de sentimientos por cluster

## 📁 Estructura Planificada

```
lyrics_models/
├── preprocessing/
│   ├── text_vectorizers.pkl
│   └── stopwords_music.txt
├── clustering/
│   ├── kmeans_lyrics_kX.pkl
│   └── hierarchical_lyrics.pkl
├── evaluation/
│   ├── topic_coherence.json
│   └── sentiment_analysis.json
└── results/
    └── lyrics_clustering_results.csv
```

---

**Estado**: 📋 PLANIFICADO PARA FASE 2