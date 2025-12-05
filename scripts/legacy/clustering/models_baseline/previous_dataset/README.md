# Top 3 Métodos de Clustering Musical - Resultados Finales

Este directorio contiene los **3 mejores métodos de clustering** desarrollados y probados para el sistema de recomendación musical con 9,677 canciones.

## 🏆 Ranking de Métodos

### 🥇 Method 1: PCA 5 Componentes (Silhouette: 0.314)
**Directorio**: `method1_pca5_silhouette0314/`

- **Silhouette Score**: 0.314 (+37.2% vs baseline)
- **Algoritmo**: K-Means con PCA (5 componentes)
- **K óptimo**: 3 clusters
- **Varianza explicada**: 66.6%
- **Distribución**: [866, 3648, 5163] canciones
- **Archivos**:
  - `scaler_pca5.pkl` - Normalizador StandardScaler
  - `pca_5comp.pkl` - Transformación PCA a 5 dimensiones
  - `kmeans_k3.pkl` - Modelo K-Means entrenado
  - `clustering_results.csv` - Dataset con clusters asignados

**Uso recomendado**: **SISTEMA PRINCIPAL DE PRODUCCIÓN**

### 🥈 Method 2: PCA 8 Componentes (Silhouette: 0.251)
**Directorio**: `method2_pca8_silhouette0251/`

- **Silhouette Score**: 0.251 (+8.7% vs baseline)
- **Algoritmo**: K-Means con PCA (8 componentes)
- **K óptimo**: 3 clusters
- **Varianza explicada**: ~75%
- **Distribución**: [870, 5171, 3636] canciones
- **Archivos**:
  - `scaler_pca8.pkl` - Normalizador StandardScaler
  - `pca_8comp.pkl` - Transformación PCA a 8 dimensiones
  - `kmeans_k3.pkl` - Modelo K-Means entrenado
  - `clustering_results.csv` - Dataset con clusters asignados

**Uso recomendado**: **SISTEMA ALTERNATIVO** (más conservador en reducción dimensional)

### 🥉 Method 3: Optimizado Base (Silhouette: 0.231)
**Directorio**: `method3_optimized_silhouette0231/`

- **Silhouette Score**: 0.231 (baseline optimizado)
- **Algoritmo**: K-Means estándar (13 dimensiones originales)
- **K óptimo**: 3 clusters
- **Features**: 13 características musicales completas
- **Distribución**: [342, 5572, 3763] canciones
- **Archivos**:
  - `scaler.pkl` - Normalizador StandardScaler
  - `kmeans_k3.pkl` - Modelo K-Means entrenado
  - `clustering_results.csv` - Dataset con clusters asignados
  - `clustering_metrics.json` - Métricas detalladas

**Uso recomendado**: **SISTEMA DE REFERENCIA** (sin reducción dimensional)

## 📊 Comparación de Performance

| Método | Silhouette | Tiempo | Memoria | Complejidad | Interpretabilidad |
|--------|------------|---------|---------|-------------|-------------------|
| PCA 5  | **0.314**  | ⚡ Rápido | 💾 Baja | 🔧 Media | 📊 Media |
| PCA 8  | 0.251      | ⚡ Rápido | 💾 Baja | 🔧 Media | 📊 Buena |
| Base   | 0.231      | ⚡ Rápido | 💾 Alta | 🔧 Baja  | 📊 Excelente |

## 🎯 Recomendaciones de Uso

### Para Producción
- **Usar Method 1 (PCA 5)** - Mejor balance calidad/eficiencia

### Para Investigación
- **Usar Method 3 (Base)** - Mejor interpretabilidad de features originales

### Para Validación
- **Comparar Method 1 vs Method 2** - Evaluar impacto de reducción dimensional

## 🔧 Comandos de Uso

```bash
# Recomendaciones con Method 1 (MEJOR)
python music_recommender.py --models-dir final_models/method1_pca5_silhouette0314 --song-id "ID" --top-n 5

# Recomendaciones con Method 2
python music_recommender.py --models-dir final_models/method2_pca8_silhouette0251 --song-id "ID" --top-n 5

# Recomendaciones con Method 3
python music_recommender.py --models-dir final_models/method3_optimized_silhouette0231 --song-id "ID" --top-n 5
```

## 📝 Notas Técnicas

- Todos los métodos procesaron **9,677 canciones** con **13 características musicales**
- Tiempo de entrenamiento: ~3-6 segundos por método
- **K=3 es consistentemente óptimo** en todos los métodos
- La reducción PCA mejora significativamente la separación de clusters
- Menor número de componentes PCA = mejor Silhouette Score (hasta cierto punto)

**Fecha de generación**: 2025-01-28
**Dataset**: picked_data_0.csv (9,677 canciones representativas)
**Estado**: ✅ VALIDADO Y LISTO PARA PRODUCCIÓN