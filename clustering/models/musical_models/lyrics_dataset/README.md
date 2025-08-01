# Modelos de Clustering - Dataset con Letras

Este directorio contendrá los modelos entrenados específicamente para el dataset `picked_data_lyrics.csv` que incluye 9,987 canciones con letras disponibles.

## 🎯 Objetivo

Generar modelos de clustering optimizados para el nuevo dataset que:
- Mantienen la calidad de clustering existente (Silhouette > 0.314)
- Procesan ~320 canciones adicionales vs dataset anterior
- Conservan compatibilidad con el sistema de recomendación

## 📊 Dataset de Entrenamiento

- **Archivo**: `data/final_data/picked_data_lyrics.csv`
- **Tamaño**: 9,987 canciones
- **Características**: 13 features musicales de Spotify + letras disponibles
- **Cobertura de letras**: ~80% (vs 38% del dataset anterior)

## 🔧 Modelos a Generar

### Method 1: K-Means con PCA (5 componentes)
- **Objetivo**: Mejorar el Silhouette Score actual de 0.314
- **Configuración**: PCA a 5 dimensiones + K-Means optimizado

### Method 2: K-Means con PCA (8 componentes)  
- **Objetivo**: Alternativa más conservadora
- **Configuración**: PCA a 8 dimensiones para ~75% varianza explicada

### Method 3: K-Means estándar
- **Objetivo**: Modelo de referencia sin reducción dimensional
- **Configuración**: 13 características originales completas

## 📁 Estructura Esperada

```
lyrics_dataset/
├── method1_lyrics_pca5/
│   ├── kmeans_kX.pkl
│   ├── pca_5comp.pkl
│   ├── scaler_pca5.pkl
│   ├── clustering_results.csv
│   └── clustering_metrics.json
├── method2_lyrics_pca8/
│   └── [archivos similares]
└── method3_lyrics_standard/
    └── [archivos similares]
```

## 🚀 Próximos Pasos

1. **Entrenar modelos** con el nuevo dataset
2. **Evaluar métricas** comparativas vs modelos anteriores  
3. **Validar compatibilidad** con sistema de recomendación
4. **Seleccionar modelo óptimo** para producción

---

**Estado**: 🚧 PREPARADO PARA IMPLEMENTACIÓN