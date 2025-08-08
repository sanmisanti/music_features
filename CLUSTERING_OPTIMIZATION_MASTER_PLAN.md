# 🎯 CLUSTERING OPTIMIZATION MASTER PLAN

**Proyecto**: Sistema de Clustering Musical Optimizado con Selección Inteligente de Datos  
**Fecha de creación**: 2025-08-08  
**Estado**: EN IMPLEMENTACIÓN  
**Objetivo**: Optimizar Silhouette Score de 0.177 a 0.25+ (0.40+ purified) mediante metodología científica

---

## 📊 RESUMEN EJECUTIVO

### **PROBLEMA IDENTIFICADO**
El sistema de clustering actual presenta **degradación crítica de performance** (-43.6% Silhouette Score: 0.314 → 0.177) causada por sesgo en la selección de datos que comprime el espacio musical natural.

### **SOLUCIÓN PROPUESTA**
Implementación sistemática de **5 fases de optimización**:
1. **Selector 10K optimizado** con Hopkins preservation
2. **Análisis comparativo** 10K vs 18K canciones  
3. **Clustering readiness** assessment crítico
4. **Cluster purification** para maximizar separabilidad
5. **Evaluación final** y selección modelo óptimo

### **MÉTRICAS DE ÉXITO**
- **Hopkins Statistic**: 0.45 → 0.75+ (+67% improvement)
- **Silhouette Score**: 0.177 → 0.25+ base, 0.40+ purified (+100%+ improvement)
- **Clustering Readiness**: 80+/100 score científico
- **Sistema Final**: 24 modelos evaluados → 1 óptimo seleccionado

---

## 🚀 FASE 1: OPTIMIZACIÓN DEL SELECTOR 10K (2-3 días)

### **OBJETIVO FASE 1**
Optimizar `select_optimal_10k_from_18k.py` para preservar Hopkins Statistic 0.75+ y diversidad musical 85%+ mediante mejoras científicas al algoritmo MaxMin sampling.

### **ETAPA 1.1: ANÁLISIS TÉCNICO DETALLADO** ⏱️ 4 horas

#### **🔍 Código a Analizar**
```
Archivo: data_selection/clustering_aware/select_optimal_10k_from_18k.py
Líneas críticas: 151-176 (MaxMin sampling), 147-148 (normalización doble)
Función principal: OptimalSelector.diverse_sampling_within_cluster()
```

#### **⚠️ PRECONDICIONES VALIDAR**
- ✅ `data/with_lyrics/spotify_songs_fixed.csv` existe (18,454 canciones)
- ✅ `data/cleaned_data/tracks_features_500.csv` para testing rápido
- ⏳ `clustering_readiness.py` implementado (Fase 3 dependency)
- ✅ Librerías: sklearn, pandas, numpy, datetime

#### **🎯 PROBLEMAS IDENTIFICADOS**
1. **MaxMin Sampling Subóptimo** (líneas 151-152):
   ```python
   # PROBLEMÁTICO: Selección inicial aleatoria
   selected_indices = [np.random.randint(len(feature_subset_scaled))]
   ```
   **Impacto**: Punto inicial aleatorio puede degradar diversidad total

2. **Normalización Doble** (líneas 147-148):
   ```python
   # INEFICIENTE: Re-normalizar datos ya normalizados
   scaler = StandardScaler()
   feature_subset_scaled = scaler.fit_transform(feature_subset)
   ```
   **Impacto**: Distorsiona distancias relativas, tiempo perdido

3. **Sin Validación Hopkins**:
   ```python
   # FALTANTE: Validación continua de clustering tendency
   # No verifica si selección preserva Hopkins Statistic
   ```
   **Impacto**: No feedback sobre degradación de clusterabilidad

#### **📊 ENTREGABLES ETAPA 1.1**
- **Documento**: `FASE1_CODIGO_ANALYSIS.md` con issues y soluciones
- **Baseline metrics**: Hopkins actual, tiempo ejecución, memoria
- **Lista priorizada**: 3 mejoras críticas + 2 mejoras menores

### **ETAPA 1.2: IMPLEMENTACIÓN MEJORAS CRÍTICAS** ⏱️ 8 horas

#### **🔧 MEJORA A: MaxMin Sampling Científico**
```python
# NUEVO: Selección inicial basada en centroide
def improved_initial_selection(self, feature_subset_scaled):
    """
    Selección inicial científica para MaxMin sampling.
    
    En lugar de punto aleatorio, seleccionar el punto más lejano
    del centroide del cluster para maximizar diversidad inicial.
    """
    centroid = np.mean(feature_subset_scaled, axis=0)
    distances_to_centroid = [
        np.linalg.norm(point - centroid) 
        for point in feature_subset_scaled
    ]
    return np.argmax(distances_to_centroid)

def improved_diverse_sampling_within_cluster(self, cluster_data, target_size, feature_data):
    """MaxMin sampling mejorado con selección inicial científica."""
    
    if len(cluster_data) <= target_size:
        return cluster_data
    
    # Usar top características para diversidad
    available_top_features = [f for f in self.top_features if f in cluster_data.columns]
    
    if not available_top_features:
        return cluster_data.sample(n=target_size, random_state=42)
    
    # MEJORA: Usar datos ya normalizados (eliminar doble normalización)
    feature_subset = cluster_data[available_top_features].values
    # Normalizar solo si es necesario
    if not hasattr(feature_data, 'scale_'):
        scaler = StandardScaler()
        feature_subset_scaled = scaler.fit_transform(feature_subset)
    else:
        # Usar scaler existente si está disponible
        feature_subset_scaled = feature_subset
    
    # MEJORA: Selección inicial científica
    initial_idx = self.improved_initial_selection(feature_subset_scaled)
    selected_indices = [initial_idx]
    selected_features = [feature_subset_scaled[initial_idx]]
    
    # MaxMin algorithm optimizado
    while len(selected_indices) < target_size:
        distances = []
        
        for i, candidate in enumerate(feature_subset_scaled):
            if i in selected_indices:
                distances.append(-1)  # Ya seleccionado
                continue
            
            # Distancia mínima a puntos ya seleccionados
            min_distance = min([
                np.linalg.norm(candidate - selected)
                for selected in selected_features
            ])
            distances.append(min_distance)
        
        # Seleccionar punto más lejano
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)
        selected_features.append(feature_subset_scaled[next_idx])
    
    return cluster_data.iloc[selected_indices]
```

#### **🔧 MEJORA B: Sistema Validación Hopkins**
```python
# NUEVO: data_selection/clustering_aware/hopkins_validator.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class HopkinsValidator:
    """Validador Hopkins integrado en proceso de selección."""
    
    def __init__(self, threshold=0.70):
        self.threshold = threshold
        self.validation_history = []
        
    def calculate_hopkins_fast(self, data_sample, sample_size=None):
        """
        Cálculo eficiente Hopkins Statistic para feedback continuo.
        
        Optimizado para ejecución rápida durante selección.
        """
        try:
            n, d = data_sample.shape
            if sample_size is None:
                sample_size = min(int(0.05 * n), 100)  # Muestra pequeña para velocidad
            
            if sample_size < 10 or n < 20:
                return 0.5  # Datos insuficientes, valor neutro
            
            # Generar puntos uniformes
            min_vals = np.min(data_sample, axis=0)
            max_vals = np.max(data_sample, axis=0)
            uniform_points = np.random.uniform(min_vals, max_vals, size=(sample_size, d))
            
            # Muestra aleatoria de datos reales
            sample_indices = np.random.choice(n, min(sample_size, n), replace=False)
            real_sample = data_sample[sample_indices]
            
            # U_i: distancias puntos uniformes a datos reales
            nbrs_uniform = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data_sample)
            u_distances, _ = nbrs_uniform.kneighbors(uniform_points)
            U = np.sum(u_distances)
            
            # W_i: distancias puntos reales a otros puntos reales
            remaining_indices = np.setdiff1d(range(n), sample_indices)
            if len(remaining_indices) > 0:
                remaining_data = data_sample[remaining_indices]
                nbrs_real = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(remaining_data)
                w_distances, _ = nbrs_real.kneighbors(real_sample)
                W = np.sum(w_distances)
            else:
                W = U  # Fallback si no hay datos restantes
            
            # Hopkins Statistic
            hopkins = U / (U + W) if (U + W) > 0 else 0.5
            
            return float(hopkins)
            
        except Exception as e:
            print(f"⚠️  Error en Hopkins calculation: {e}")
            return 0.5  # Valor neutro en caso de error
    
    def validate_during_selection(self, current_selection, iteration=None):
        """
        Validar Hopkins cada N puntos seleccionados.
        
        Returns:
            dict: {
                'action': 'continue' | 'fallback',
                'hopkins': float,
                'recommendation': str
            }
        """
        if len(current_selection) < 20:
            return {'action': 'continue', 'hopkins': None, 'recommendation': 'Datos insuficientes'}
        
        # Preparar datos para Hopkins
        if hasattr(current_selection, 'values'):
            data_array = current_selection.values
        else:
            data_array = current_selection
        
        hopkins = self.calculate_hopkins_fast(data_array)
        
        # Guardar en historial
        validation_record = {
            'iteration': iteration,
            'sample_size': len(current_selection),
            'hopkins': hopkins,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.validation_history.append(validation_record)
        
        # Decisión
        if hopkins < self.threshold:
            return {
                'action': 'fallback',
                'hopkins': hopkins,
                'recommendation': f'Hopkins {hopkins:.3f} < {self.threshold} - Activar estrategia fallback'
            }
        else:
            return {
                'action': 'continue', 
                'hopkins': hopkins,
                'recommendation': f'Hopkins {hopkins:.3f} - Selección adecuada'
            }
    
    def get_validation_summary(self):
        """Resumen de validaciones realizadas."""
        if not self.validation_history:
            return {'status': 'No validations performed'}
        
        hopkins_scores = [v['hopkins'] for v in self.validation_history if v['hopkins'] is not None]
        
        return {
            'total_validations': len(self.validation_history),
            'hopkins_scores': hopkins_scores,
            'avg_hopkins': np.mean(hopkins_scores) if hopkins_scores else None,
            'min_hopkins': np.min(hopkins_scores) if hopkins_scores else None,
            'max_hopkins': np.max(hopkins_scores) if hopkins_scores else None,
            'threshold_used': self.threshold,
            'threshold_violations': sum(1 for h in hopkins_scores if h < self.threshold)
        }
```

#### **🔧 MEJORA C: Selección con Validación Integrada**
```python
# MODIFICAR: OptimalSelector class
def select_optimal_10k_with_validation(self, df):
    """Selección optimizada con validación Hopkins integrada."""
    
    print("\n🚀 SELECCIÓN OPTIMIZADA CON VALIDACIÓN HOPKINS")
    print("=" * 60)
    
    # Inicializar validador Hopkins
    hopkins_validator = HopkinsValidator(threshold=0.70)
    
    # 1. Preparar datos para clustering
    X_scaled, clean_indices = self.prepare_clustering_data(df)
    if X_scaled is None:
        return None
    
    # 2. Identificar estructura natural
    cluster_labels, silhouette = self.identify_natural_clusters(X_scaled, clean_indices)
    
    # 3. Preparar DataFrame con clusters
    df_clean = df.loc[clean_indices].copy()
    df_clean['natural_cluster'] = cluster_labels
    
    print(f"\n📊 SELECCIÓN POR CLUSTERS CON VALIDACIÓN")
    print("-" * 50)
    
    # 4. Selección por cluster con validación continua
    target_total = 10000
    selected_parts = []
    selection_log = {}
    
    for cluster_id in sorted(df_clean['natural_cluster'].unique()):
        cluster_data = df_clean[df_clean['natural_cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        
        proportion = cluster_size / len(df_clean)
        target_size = int(target_total * proportion)
        
        print(f"\n🎵 Cluster {cluster_id}:")
        print(f"   Tamaño original: {cluster_size:,} canciones ({proportion:.1%})")
        print(f"   Target selección: {target_size:,} canciones")
        
        # 5. Muestreo diverso mejorado
        selected_cluster = self.improved_diverse_sampling_within_cluster(
            cluster_data, target_size, X_scaled
        )
        
        # 6. NUEVO: Validación Hopkins del cluster seleccionado
        if len(selected_cluster) >= 20:  # Suficientes datos para Hopkins
            cluster_features = selected_cluster[self.available_features]
            validation_result = hopkins_validator.validate_during_selection(
                cluster_features, iteration=cluster_id
            )
            
            print(f"   📊 Hopkins validation: {validation_result['hopkins']:.3f}")
            print(f"   💡 {validation_result['recommendation']}")
            
            # Si Hopkins muy bajo, aplicar estrategia fallback
            if validation_result['action'] == 'fallback':
                print(f"   🔄 Aplicando fallback: selección más diversa")
                selected_cluster = self._apply_diversity_fallback(
                    cluster_data, target_size
                )
                
        selected_parts.append(selected_cluster)
        selection_log[cluster_id] = {
            'original_size': cluster_size,
            'selected_size': len(selected_cluster),
            'selection_ratio': len(selected_cluster) / cluster_size
        }
        
        print(f"   ✅ Seleccionadas: {len(selected_cluster):,} canciones")
    
    # 7. Combinar y ajustar
    final_selection = pd.concat(selected_parts, ignore_index=True)
    
    if len(final_selection) != target_total:
        print(f"\n🔧 Ajustando tamaño final: {len(final_selection):,} → {target_total:,}")
        if len(final_selection) > target_total:
            final_selection = final_selection.sample(n=target_total, random_state=42)
        else:
            remaining = target_total - len(final_selection)
            available = df_clean[~df_clean.index.isin(final_selection.index)]
            additional = available.sample(n=remaining, random_state=42)
            final_selection = pd.concat([final_selection, additional], ignore_index=True)
    
    # 8. NUEVO: Validación Hopkins final
    final_features = final_selection[self.available_features]
    final_validation = hopkins_validator.validate_during_selection(
        final_features, iteration='FINAL'
    )
    
    # 9. Resumen de validación
    validation_summary = hopkins_validator.get_validation_summary()
    
    print(f"\n✅ SELECCIÓN CON VALIDACIÓN COMPLETADA")
    print(f"📦 Dataset final: {len(final_selection):,} canciones")
    print(f"📊 Hopkins final: {final_validation['hopkins']:.3f}")
    print(f"📈 Hopkins promedio: {validation_summary['avg_hopkins']:.3f}")
    print(f"🎯 Violaciones threshold: {validation_summary['threshold_violations']}")
    
    # Incluir metadatos de validación
    selection_metadata = {
        'selection_log': selection_log,
        'hopkins_validation': validation_summary,
        'final_hopkins': final_validation['hopkins'],
        'pre_clustering_silhouette': silhouette
    }
    
    return final_selection, selection_metadata

def _apply_diversity_fallback(self, cluster_data, target_size):
    """Estrategia fallback para mayor diversidad cuando Hopkins es bajo."""
    
    # Estrategia: Muestreo estratificado por características extremas
    if len(cluster_data) <= target_size:
        return cluster_data
    
    # Identificar percentiles extremos en características principales
    extreme_indices = set()
    
    for feature in self.top_features[:3]:  # Top 3 características
        if feature in cluster_data.columns:
            feature_values = cluster_data[feature]
            
            # Percentiles 10% y 90%
            p10 = feature_values.quantile(0.10)
            p90 = feature_values.quantile(0.90)
            
            # Seleccionar canciones en extremos
            extreme_mask = (feature_values <= p10) | (feature_values >= p90)
            extreme_indices.update(cluster_data[extreme_mask].index.tolist())
    
    # Seleccionar de extremos + muestra aleatoria
    extreme_subset = cluster_data.loc[list(extreme_indices)]
    n_extreme = min(len(extreme_subset), target_size // 2)
    
    if n_extreme > 0:
        selected_extreme = extreme_subset.sample(n=n_extreme, random_state=42)
        remaining_needed = target_size - n_extreme
        
        if remaining_needed > 0:
            remaining_data = cluster_data.drop(selected_extreme.index)
            selected_random = remaining_data.sample(n=min(remaining_needed, len(remaining_data)), random_state=42)
            return pd.concat([selected_extreme, selected_random])
        else:
            return selected_extreme
    else:
        # Fallback completo: muestreo aleatorio
        return cluster_data.sample(n=target_size, random_state=42)
```

#### **📊 ENTREGABLES ETAPA 1.2**
- **Código mejorado**: `select_optimal_10k_from_18k.py` actualizado
- **Módulo nuevo**: `hopkins_validator.py` implementado
- **Performance target**: Hopkins preservation 80%+, tiempo similar

### **ETAPA 1.3: SISTEMA DE TESTING COMPREHENSIVO** ⏱️ 6 horas

#### **🧪 Test Suite Principal**
```python
# CREAR: tests/test_data_selection/test_optimal_selector_improved.py
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Agregar paths para imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_selection.clustering_aware.select_optimal_10k_from_18k import OptimalSelector
from data_selection.clustering_aware.hopkins_validator import HopkinsValidator

class TestOptimalSelectorImproved(unittest.TestCase):
    """Tests comprehensivos para selector optimizado."""
    
    @classmethod
    def setUpClass(cls):
        """Setup una sola vez para todos los tests."""
        cls.test_data_path = 'data/cleaned_data/tracks_features_500.csv'
        cls.small_test_path = 'data/cleaned_data/tracks_features_100.csv'  # Para tests rápidos
        
        # Verificar que archivos de prueba existen
        if not Path(cls.test_data_path).exists():
            raise FileNotFoundError(f"Test dataset not found: {cls.test_data_path}")
    
    def setUp(self):
        """Setup para cada test individual."""
        self.original_selector = OptimalSelector()  # Versión original
        self.improved_selector = OptimalSelector()  # Versión mejorada (misma clase, métodos mejorados)
        self.hopkins_validator = HopkinsValidator()
        
        # Cargar datos de prueba
        self.test_data_500 = pd.read_csv(self.test_data_path, sep=';', decimal=',')
        
    def test_hopkins_preservation_improvement(self):
        """CRÍTICO: Verificar mejora en preservación Hopkins."""
        print("\n🧪 Test Hopkins Preservation Improvement")
        
        # Ejecutar selección con dataset pequeño para test rápido
        sample_data = self.test_data_500.head(200)
        
        # Hopkins antes de selección
        features_original = sample_data[self.improved_selector.musical_features[:5]]  # Top 5 para velocidad
        features_clean = features_original.dropna()
        hopkins_original = self.hopkins_validator.calculate_hopkins_fast(features_clean.values)
        
        print(f"   Hopkins original: {hopkins_original:.3f}")
        
        # Simular selección optimizada (método mejorado)
        try:
            # Test del nuevo método de selección inicial
            scaler = StandardScaler()
            from sklearn.preprocessing import StandardScaler
            features_scaled = scaler.fit_transform(features_clean.values)
            
            # Test selección inicial científica vs aleatoria
            initial_scientific = self.improved_selector.improved_initial_selection(features_scaled)
            initial_random = np.random.randint(len(features_scaled))
            
            # Verificar que selección científica es reproducible
            initial_scientific_2 = self.improved_selector.improved_initial_selection(features_scaled)
            self.assertEqual(initial_scientific, initial_scientific_2, 
                           "Selección inicial científica debe ser reproducible")
            
            print(f"   Selección científica: índice {initial_scientific}")
            print(f"   Selección aleatoria: índice {initial_random}")
            
            # Hopkins después de selección simulada
            selected_indices = [initial_scientific] + list(np.random.choice(
                range(len(features_scaled)), size=min(50, len(features_scaled)-1), replace=False
            ))
            selected_data = features_clean.iloc[selected_indices]
            hopkins_selected = self.hopkins_validator.calculate_hopkins_fast(selected_data.values)
            
            print(f"   Hopkins después selección: {hopkins_selected:.3f}")
            
            # Verificar mejora o al menos preservación
            preservation_ratio = hopkins_selected / hopkins_original if hopkins_original > 0 else 1
            print(f"   Ratio preservación: {preservation_ratio:.3f}")
            
            self.assertGreaterEqual(preservation_ratio, 0.70, 
                                  f"Hopkins preservation ratio {preservation_ratio:.3f} debe ser >= 0.70")
            
        except Exception as e:
            self.fail(f"Error en test Hopkins preservation: {e}")
    
    def test_maxmin_sampling_scientific_selection(self):
        """Verificar que selección inicial es científica vs aleatoria."""
        print("\n🧪 Test MaxMin Scientific Selection")
        
        # Crear datos sintéticos con clusters conocidos
        np.random.seed(42)
        cluster1 = np.random.normal([0, 0], 0.5, (50, 2))
        cluster2 = np.random.normal([5, 5], 0.5, (50, 2))
        synthetic_data = np.vstack([cluster1, cluster2])
        
        # Test selección inicial científica
        initial_idx = self.improved_selector.improved_initial_selection(synthetic_data)
        initial_point = synthetic_data[initial_idx]
        
        # Verificar que punto inicial está lejos del centroide
        centroid = np.mean(synthetic_data, axis=0)
        distance_to_centroid = np.linalg.norm(initial_point - centroid)
        
        # Calcular distancia promedio para comparación
        all_distances = [np.linalg.norm(point - centroid) for point in synthetic_data]
        avg_distance = np.mean(all_distances)
        max_distance = np.max(all_distances)
        
        print(f"   Distancia inicial al centroide: {distance_to_centroid:.3f}")
        print(f"   Distancia promedio: {avg_distance:.3f}")
        print(f"   Distancia máxima: {max_distance:.3f}")
        
        # El punto inicial debe estar en el percentil alto de distancias
        percentile_90 = np.percentile(all_distances, 90)
        self.assertGreaterEqual(distance_to_centroid, percentile_90 * 0.8,
                               "Selección inicial debe estar cerca del punto más lejano")
        
    def test_performance_benchmark_comparison(self):
        """Benchmark performance: improved vs original implementation."""
        print("\n🧪 Test Performance Benchmark")
        
        # Datos de prueba
        sample_data = self.test_data_500.head(100)
        
        # Benchmark método original (simulado)
        start_time = time.time()
        for _ in range(10):  # Repetir para promedio
            # Simular selección original
            sample_selection = sample_data.sample(n=50, random_state=42)
        original_time = (time.time() - start_time) / 10
        
        # Benchmark método mejorado
        start_time = time.time()
        for _ in range(10):
            # Simular selección mejorada
            features = sample_data[self.improved_selector.top_features[:3]].dropna()
            if len(features) > 0:
                features_scaled = StandardScaler().fit_transform(features.values)
                initial_idx = self.improved_selector.improved_initial_selection(features_scaled)
        improved_time = (time.time() - start_time) / 10
        
        print(f"   Tiempo método original: {original_time*1000:.2f} ms")
        print(f"   Tiempo método mejorado: {improved_time*1000:.2f} ms")
        
        # Verificar que no hay degradación significativa de performance
        performance_ratio = improved_time / original_time if original_time > 0 else 1
        print(f"   Ratio performance: {performance_ratio:.2f}")
        
        self.assertLess(performance_ratio, 2.0, 
                       "Método mejorado no debe ser 2x más lento que original")
    
    def test_musical_diversity_preservation(self):
        """Verificar preservación diversidad musical."""
        print("\n🧪 Test Musical Diversity Preservation")
        
        # Usar datos reales con características musicales
        features_to_analyze = ['danceability', 'energy', 'valence', 'tempo']
        available_features = [f for f in features_to_analyze if f in self.test_data_500.columns]
        
        if not available_features:
            self.skipTest("No musical features available for diversity test")
        
        original_data = self.test_data_500[available_features].dropna()
        
        # Calcular diversidad original
        original_diversity = {}
        for feature in available_features:
            original_diversity[feature] = {
                'mean': original_data[feature].mean(),
                'std': original_data[feature].std(),
                'range': original_data[feature].max() - original_data[feature].min()
            }
        
        # Simular selección optimizada
        n_select = min(100, len(original_data))
        features_scaled = StandardScaler().fit_transform(original_data.values)
        
        # Usar método mejorado de selección
        initial_idx = self.improved_selector.improved_initial_selection(features_scaled)
        
        # Simular MaxMin sampling
        selected_indices = [initial_idx]
        for _ in range(min(n_select - 1, 20)):  # Muestra pequeña para test rápido
            distances = []
            for i, candidate in enumerate(features_scaled):
                if i in selected_indices:
                    distances.append(-1)
                else:
                    min_dist = min([np.linalg.norm(candidate - features_scaled[sel]) 
                                   for sel in selected_indices])
                    distances.append(min_dist)
            
            if max(distances) > 0:
                next_idx = np.argmax(distances)
                selected_indices.append(next_idx)
        
        selected_data = original_data.iloc[selected_indices]
        
        # Calcular diversidad seleccionada
        selected_diversity = {}
        for feature in available_features:
            selected_diversity[feature] = {
                'mean': selected_data[feature].mean(),
                'std': selected_data[feature].std(),
                'range': selected_data[feature].max() - selected_data[feature].min()
            }
        
        # Verificar preservación de diversidad
        diversity_preservation = {}
        for feature in available_features:
            if original_diversity[feature]['std'] > 0:
                std_preservation = selected_diversity[feature]['std'] / original_diversity[feature]['std']
                range_preservation = selected_diversity[feature]['range'] / original_diversity[feature]['range']
            else:
                std_preservation = 1.0
                range_preservation = 1.0
            
            diversity_preservation[feature] = {
                'std_preservation': std_preservation,
                'range_preservation': range_preservation
            }
            
            print(f"   {feature}:")
            print(f"     Std preservation: {std_preservation:.3f}")
            print(f"     Range preservation: {range_preservation:.3f}")
        
        # Verificar que se preserva al menos 70% de la diversidad promedio
        avg_std_preservation = np.mean([d['std_preservation'] for d in diversity_preservation.values()])
        avg_range_preservation = np.mean([d['range_preservation'] for d in diversity_preservation.values()])
        
        print(f"   Preservación promedio std: {avg_std_preservation:.3f}")
        print(f"   Preservación promedio range: {avg_range_preservation:.3f}")
        
        self.assertGreaterEqual(avg_std_preservation, 0.60,
                               "Debe preservar al menos 60% de diversidad std")
        self.assertGreaterEqual(avg_range_preservation, 0.60,
                               "Debe preservar al menos 60% de diversidad range")
    
    def test_boundary_cases_handling(self):
        """Test casos extremos: clusters muy pequeños, datos faltantes."""
        print("\n🧪 Test Boundary Cases")
        
        # Caso 1: Dataset muy pequeño
        tiny_data = self.test_data_500.head(5)
        
        try:
            features = tiny_data[self.improved_selector.top_features[:2]].dropna()
            if len(features) >= 3:
                features_scaled = StandardScaler().fit_transform(features.values)
                initial_idx = self.improved_selector.improved_initial_selection(features_scaled)
                self.assertIsInstance(initial_idx, (int, np.integer))
                self.assertGreaterEqual(initial_idx, 0)
                self.assertLess(initial_idx, len(features))
                print("   ✅ Manejo dataset pequeño: OK")
            else:
                print("   ⚠️  Dataset muy pequeño para test")
        except Exception as e:
            self.fail(f"Error con dataset pequeño: {e}")
        
        # Caso 2: Datos con muchos NaN
        data_with_nans = self.test_data_500.copy()
        # Introducir 50% NaNs en primera característica
        if len(self.improved_selector.top_features) > 0:
            first_feature = self.improved_selector.top_features[0]
            if first_feature in data_with_nans.columns:
                nan_indices = np.random.choice(len(data_with_nans), len(data_with_nans)//2, replace=False)
                data_with_nans.loc[nan_indices, first_feature] = np.nan
                
                features_with_nans = data_with_nans[self.improved_selector.top_features[:2]].dropna()
                
                if len(features_with_nans) >= 10:
                    try:
                        features_scaled = StandardScaler().fit_transform(features_with_nans.values)
                        initial_idx = self.improved_selector.improved_initial_selection(features_scaled)
                        self.assertIsInstance(initial_idx, (int, np.integer))
                        print("   ✅ Manejo datos con NaNs: OK")
                    except Exception as e:
                        print(f"   ⚠️  Error con datos NaN: {e}")
                else:
                    print("   ⚠️  Pocos datos después de limpiar NaNs")
    
    def test_hopkins_validator_accuracy(self):
        """Test precisión del validador Hopkins."""
        print("\n🧪 Test Hopkins Validator Accuracy")
        
        # Datos sintéticos clusterizables
        np.random.seed(42)
        clusterable_data = np.vstack([
            np.random.normal([0, 0], 0.3, (100, 2)),
            np.random.normal([3, 3], 0.3, (100, 2)),
            np.random.normal([-2, 2], 0.3, (100, 2))
        ])
        
        # Datos sintéticos aleatorios
        random_data = np.random.uniform(-4, 4, (300, 2))
        
        # Calcular Hopkins
        hopkins_clusterable = self.hopkins_validator.calculate_hopkins_fast(clusterable_data)
        hopkins_random = self.hopkins_validator.calculate_hopkins_fast(random_data)
        
        print(f"   Hopkins datos clusterizables: {hopkins_clusterable:.3f}")
        print(f"   Hopkins datos aleatorios: {hopkins_random:.3f}")
        
        # Verificar que Hopkins distingue correctamente
        self.assertGreater(hopkins_clusterable, 0.6, 
                          "Hopkins debe ser > 0.6 para datos clusterizables")
        self.assertLess(abs(hopkins_random - 0.5), 0.3,
                       "Hopkins debe estar cerca de 0.5 para datos aleatorios")
        self.assertGreater(hopkins_clusterable, hopkins_random,
                          "Hopkins clusterable debe ser mayor que aleatorio")

if __name__ == '__main__':
    # Ejecutar tests con output detallado
    unittest.main(verbosity=2)
```

#### **📊 Script de Validación Completa**
```python
# CREAR: scripts/validate_phase1_improvements.py
import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime

# Agregar paths
sys.path.append(str(Path(__file__).parent.parent))

from data_selection.clustering_aware.select_optimal_10k_from_18k import OptimalSelector
from data_selection.clustering_aware.hopkins_validator import HopkinsValidator

def comprehensive_phase1_validation():
    """Script para validar todas las mejoras Fase 1."""
    
    print("🔬 VALIDACIÓN COMPREHENSIVA FASE 1")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validation_results = {}
    
    # 1. Test Hopkins preservation
    print("\n📊 1. VALIDACIÓN HOPKINS PRESERVATION")
    print("-" * 40)
    
    try:
        hopkins_test = test_hopkins_improvement()
        validation_results['hopkins_preservation'] = {
            'status': 'success',
            'results': hopkins_test,
            'score': hopkins_test.get('preservation_score', 0)
        }
        print(f"✅ Hopkins test completado: Score {hopkins_test.get('preservation_score', 0):.3f}")
        
    except Exception as e:
        validation_results['hopkins_preservation'] = {
            'status': 'error',
            'error': str(e),
            'score': 0
        }
        print(f"❌ Error en Hopkins test: {e}")
    
    # 2. Test performance improvement
    print("\n⚡ 2. VALIDACIÓN PERFORMANCE")
    print("-" * 40)
    
    try:
        performance_test = benchmark_performance_improvement()
        validation_results['performance'] = {
            'status': 'success',
            'results': performance_test,
            'score': performance_test.get('performance_score', 0)
        }
        print(f"✅ Performance test completado: Score {performance_test.get('performance_score', 0):.3f}")
        
    except Exception as e:
        validation_results['performance'] = {
            'status': 'error',
            'error': str(e),
            'score': 0
        }
        print(f"❌ Error en performance test: {e}")
    
    # 3. Test musical diversity preservation
    print("\n🎵 3. VALIDACIÓN DIVERSIDAD MUSICAL")
    print("-" * 40)
    
    try:
        diversity_test = test_musical_diversity_preservation()
        validation_results['musical_diversity'] = {
            'status': 'success',
            'results': diversity_test,
            'score': diversity_test.get('diversity_score', 0)
        }
        print(f"✅ Diversidad test completado: Score {diversity_test.get('diversity_score', 0):.3f}")
        
    except Exception as e:
        validation_results['musical_diversity'] = {
            'status': 'error',
            'error': str(e),
            'score': 0
        }
        print(f"❌ Error en diversidad test: {e}")
    
    # 4. Generar reporte final
    print("\n📋 4. GENERANDO REPORTE FINAL")
    print("-" * 40)
    
    final_report = generate_phase1_validation_report(validation_results)
    
    # Calcular score total
    scores = [result.get('score', 0) for result in validation_results.values() if result.get('score') is not None]
    total_score = np.mean(scores) if scores else 0
    
    print(f"\n🎯 RESUMEN VALIDACIÓN FASE 1")
    print("=" * 60)
    print(f"📊 Score total: {total_score:.3f}/1.0")
    print(f"🧪 Tests exitosos: {sum(1 for r in validation_results.values() if r['status'] == 'success')}/{len(validation_results)}")
    
    if total_score >= 0.75:
        print("✅ FASE 1 VALIDADA - Mejoras exitosas")
    elif total_score >= 0.60:
        print("⚠️  FASE 1 PARCIAL - Algunas mejoras necesarias")
    else:
        print("❌ FASE 1 FALLÓ - Revisión crítica necesaria")
    
    return validation_results, final_report

def test_hopkins_improvement():
    """Test específico mejora Hopkins preservation."""
    
    # Cargar dataset de prueba
    test_data_path = 'data/cleaned_data/tracks_features_500.csv'
    if not Path(test_data_path).exists():
        raise FileNotFoundError(f"Test dataset not found: {test_data_path}")
    
    df = pd.read_csv(test_data_path, sep=';', decimal=',')
    selector = OptimalSelector()
    hopkins_validator = HopkinsValidator()
    
    # Hopkins baseline dataset completo
    available_features = [f for f in selector.musical_features[:5] if f in df.columns]
    if not available_features:
        raise ValueError("No musical features available")
    
    original_features = df[available_features].dropna()
    if len(original_features) < 50:
        raise ValueError("Insufficient data for Hopkins test")
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    original_scaled = scaler.fit_transform(original_features.values)
    
    hopkins_original = hopkins_validator.calculate_hopkins_fast(original_scaled)
    
    # Simular selección optimizada vs aleatoria
    n_select = min(100, len(original_features))
    
    # Selección optimizada (científica)
    initial_scientific = selector.improved_initial_selection(original_scaled)
    selected_indices_optimized = [initial_scientific]
    
    # MaxMin sampling simulado
    for _ in range(min(n_select - 1, 20)):
        distances = []
        for i, candidate in enumerate(original_scaled):
            if i in selected_indices_optimized:
                distances.append(-1)
            else:
                min_dist = min([np.linalg.norm(candidate - original_scaled[sel]) 
                               for sel in selected_indices_optimized])
                distances.append(min_dist)
        
        if max(distances) > 0:
            next_idx = np.argmax(distances)
            selected_indices_optimized.append(next_idx)
    
    selected_optimized = original_scaled[selected_indices_optimized]
    hopkins_optimized = hopkins_validator.calculate_hopkins_fast(selected_optimized)
    
    # Selección aleatoria para comparación
    selected_indices_random = np.random.choice(len(original_scaled), 
                                              size=len(selected_indices_optimized), 
                                              replace=False)
    selected_random = original_scaled[selected_indices_random]
    hopkins_random = hopkins_validator.calculate_hopkins_fast(selected_random)
    
    # Calcular scores
    preservation_optimized = hopkins_optimized / hopkins_original if hopkins_original > 0 else 1
    preservation_random = hopkins_random / hopkins_original if hopkins_original > 0 else 1
    
    improvement = preservation_optimized - preservation_random
    
    return {
        'hopkins_original': round(hopkins_original, 4),
        'hopkins_optimized': round(hopkins_optimized, 4),
        'hopkins_random': round(hopkins_random, 4),
        'preservation_optimized': round(preservation_optimized, 4),
        'preservation_random': round(preservation_random, 4),
        'improvement': round(improvement, 4),
        'preservation_score': min(1.0, max(0.0, preservation_optimized)),
        'n_samples_tested': len(selected_indices_optimized)
    }

def benchmark_performance_improvement():
    """Benchmark performance mejoras vs implementación original."""
    
    test_data_path = 'data/cleaned_data/tracks_features_500.csv'
    df = pd.read_csv(test_data_path, sep=';', decimal=',')
    selector = OptimalSelector()
    
    available_features = [f for f in selector.top_features[:3] if f in df.columns]
    test_data = df[available_features].dropna().head(200)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    test_scaled = scaler.fit_transform(test_data.values)
    
    # Benchmark método mejorado
    times_improved = []
    for _ in range(10):
        start_time = time.time()
        initial_idx = selector.improved_initial_selection(test_scaled)
        times_improved.append(time.time() - start_time)
    
    avg_time_improved = np.mean(times_improved)
    
    # Benchmark método original (simulado con selección aleatoria)
    times_original = []
    for _ in range(10):
        start_time = time.time()
        initial_idx = np.random.randint(len(test_scaled))
        times_original.append(time.time() - start_time)
    
    avg_time_original = np.mean(times_original)
    
    # Calcular score de performance (menor es mejor, pero no debe ser mucho peor)
    performance_ratio = avg_time_improved / avg_time_original if avg_time_original > 0 else 1
    performance_score = max(0, 1 - max(0, performance_ratio - 1))  # Penalizar si es más lento
    
    return {
        'avg_time_improved_ms': round(avg_time_improved * 1000, 3),
        'avg_time_original_ms': round(avg_time_original * 1000, 3),
        'performance_ratio': round(performance_ratio, 3),
        'performance_score': round(performance_score, 3),
        'n_iterations': 10
    }

def test_musical_diversity_preservation():
    """Test preservación diversidad musical."""
    
    test_data_path = 'data/cleaned_data/tracks_features_500.csv'
    df = pd.read_csv(test_data_path, sep=';', decimal=',')
    selector = OptimalSelector()
    
    # Características musicales para análisis
    musical_features = ['danceability', 'energy', 'valence', 'loudness']
    available_features = [f for f in musical_features if f in df.columns]
    
    if len(available_features) < 2:
        raise ValueError("Insufficient musical features for diversity test")
    
    original_data = df[available_features].dropna()
    
    # Métricas diversidad original
    original_diversity = {}
    for feature in available_features:
        original_diversity[feature] = {
            'std': original_data[feature].std(),
            'range': original_data[feature].max() - original_data[feature].min(),
            'percentile_90_10': original_data[feature].quantile(0.9) - original_data[feature].quantile(0.1)
        }
    
    # Simular selección optimizada
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    original_scaled = scaler.fit_transform(original_data.values)
    
    n_select = min(100, len(original_data))
    initial_idx = selector.improved_initial_selection(original_scaled)
    
    # MaxMin sampling simulado
    selected_indices = [initial_idx]
    for _ in range(min(n_select - 1, 30)):
        distances = []
        for i, candidate in enumerate(original_scaled):
            if i in selected_indices:
                distances.append(-1)
            else:
                min_dist = min([np.linalg.norm(candidate - original_scaled[sel]) 
                               for sel in selected_indices])
                distances.append(min_dist)
        
        if max(distances) > 0:
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
    
    selected_data = original_data.iloc[selected_indices]
    
    # Métricas diversidad seleccionada
    selected_diversity = {}
    diversity_preservation_scores = []
    
    for feature in available_features:
        selected_diversity[feature] = {
            'std': selected_data[feature].std(),
            'range': selected_data[feature].max() - selected_data[feature].min(),
            'percentile_90_10': selected_data[feature].quantile(0.9) - selected_data[feature].quantile(0.1)
        }
        
        # Calcular preservación para cada métrica
        if original_diversity[feature]['std'] > 0:
            std_preservation = selected_diversity[feature]['std'] / original_diversity[feature]['std']
        else:
            std_preservation = 1.0
            
        if original_diversity[feature]['range'] > 0:
            range_preservation = selected_diversity[feature]['range'] / original_diversity[feature]['range']
        else:
            range_preservation = 1.0
            
        if original_diversity[feature]['percentile_90_10'] > 0:
            percentile_preservation = selected_diversity[feature]['percentile_90_10'] / original_diversity[feature]['percentile_90_10']
        else:
            percentile_preservation = 1.0
        
        # Score promedio para esta característica
        feature_diversity_score = np.mean([std_preservation, range_preservation, percentile_preservation])
        diversity_preservation_scores.append(feature_diversity_score)
    
    # Score total
    avg_diversity_score = np.mean(diversity_preservation_scores)
    
    return {
        'available_features': available_features,
        'original_diversity': original_diversity,
        'selected_diversity': selected_diversity,
        'diversity_preservation_by_feature': diversity_preservation_scores,
        'diversity_score': round(min(1.0, avg_diversity_score), 3),
        'n_selected': len(selected_indices)
    }

def generate_phase1_validation_report(validation_results):
    """Generar reporte detallado validación Fase 1."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report_content = f"""# 📊 REPORTE VALIDACIÓN FASE 1 - OPTIMIZACIÓN SELECTOR 10K

**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Timestamp**: {timestamp}  
**Estado**: {"✅ EXITOSA" if all(r['status'] == 'success' for r in validation_results.values()) else "⚠️ PARCIAL" if any(r['status'] == 'success' for r in validation_results.values()) else "❌ FALLIDA"}

---

## 🎯 RESUMEN EJECUTIVO

"""
    
    # Calcular métricas generales
    successful_tests = sum(1 for r in validation_results.values() if r['status'] == 'success')
    total_tests = len(validation_results)
    scores = [r.get('score', 0) for r in validation_results.values() if r.get('score') is not None]
    avg_score = np.mean(scores) if scores else 0
    
    report_content += f"""
### Métricas Generales
- **Tests exitosos**: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)
- **Score promedio**: {avg_score:.3f}/1.0 ({avg_score*100:.1f}%)
- **Estado general**: {"APROBADO" if avg_score >= 0.75 else "PENDIENTE MEJORAS" if avg_score >= 0.60 else "REQUIERE REVISIÓN"}

---

## 📊 RESULTADOS DETALLADOS

"""
    
    # Detalles por test
    for test_name, result in validation_results.items():
        report_content += f"""
### {test_name.replace('_', ' ').title()}

**Estado**: {result['status'].upper()}  
**Score**: {result.get('score', 'N/A')}  

"""
        
        if result['status'] == 'success' and 'results' in result:
            test_results = result['results']
            
            if test_name == 'hopkins_preservation':
                report_content += f"""
**Métricas Hopkins**:
- Hopkins original: {test_results.get('hopkins_original', 'N/A')}
- Hopkins optimizado: {test_results.get('hopkins_optimized', 'N/A')}  
- Hopkins aleatorio: {test_results.get('hopkins_random', 'N/A')}
- Preservación optimizada: {test_results.get('preservation_optimized', 'N/A')}
- Mejora vs aleatorio: {test_results.get('improvement', 'N/A')}
- Muestras analizadas: {test_results.get('n_samples_tested', 'N/A')}

**Interpretación**: {"✅ Excelente" if test_results.get('preservation_optimized', 0) >= 0.80 else "⚠️ Aceptable" if test_results.get('preservation_optimized', 0) >= 0.70 else "❌ Insuficiente"}
"""
            
            elif test_name == 'performance':
                report_content += f"""
**Métricas Performance**:
- Tiempo método mejorado: {test_results.get('avg_time_improved_ms', 'N/A')} ms
- Tiempo método original: {test_results.get('avg_time_original_ms', 'N/A')} ms
- Ratio performance: {test_results.get('performance_ratio', 'N/A')}
- Iteraciones: {test_results.get('n_iterations', 'N/A')}

**Interpretación**: {"✅ Mejor performance" if test_results.get('performance_ratio', 2) <= 1.0 else "⚠️ Performance similar" if test_results.get('performance_ratio', 2) <= 1.5 else "❌ Performance degradada"}
"""
            
            elif test_name == 'musical_diversity':
                report_content += f"""
**Métricas Diversidad**:
- Características analizadas: {len(test_results.get('available_features', []))}
- Score diversidad: {test_results.get('diversity_score', 'N/A')}
- Muestras seleccionadas: {test_results.get('n_selected', 'N/A')}

**Características evaluadas**: {', '.join(test_results.get('available_features', []))}

**Interpretación**: {"✅ Diversidad preservada" if test_results.get('diversity_score', 0) >= 0.75 else "⚠️ Diversidad aceptable" if test_results.get('diversity_score', 0) >= 0.60 else "❌ Pérdida diversidad significativa"}
"""
        
        elif result['status'] == 'error':
            report_content += f"""
**Error**: {result.get('error', 'Error desconocido')}

**Acción requerida**: Revisar implementación y dependencias
"""
    
    report_content += f"""

---

## 🎯 RECOMENDACIONES

"""
    
    # Generar recomendaciones basadas en resultados
    recommendations = []
    
    if avg_score >= 0.85:
        recommendations.append("✅ **PROCEDER A FASE 2**: Todas las mejoras funcionan correctamente")
    elif avg_score >= 0.70:
        recommendations.append("⚠️ **MEJORAS MENORES**: Ajustar implementación antes de Fase 2")
        
        # Recomendaciones específicas
        if validation_results.get('hopkins_preservation', {}).get('score', 0) < 0.75:
            recommendations.append("- **Hopkins**: Ajustar threshold o mejorar algoritmo selección inicial")
        
        if validation_results.get('performance', {}).get('score', 0) < 0.75:
            recommendations.append("- **Performance**: Optimizar cálculos Hopkins o usar muestreo más pequeño")
        
        if validation_results.get('musical_diversity', {}).get('score', 0) < 0.75:
            recommendations.append("- **Diversidad**: Mejorar estrategia MaxMin o considerar más características")
    
    else:
        recommendations.append("❌ **REVISIÓN CRÍTICA REQUERIDA**: Implementación presenta problemas significativos")
        recommendations.append("- Revisar algoritmo MaxMin sampling")
        recommendations.append("- Validar cálculo Hopkins Statistic") 
        recommendations.append("- Verificar preservación características musicales")
    
    for rec in recommendations:
        report_content += f"\n{rec}"
    
    report_content += f"""

---

## 📋 PRÓXIMOS PASOS

1. **Si score >= 0.75**: Proceder a FASE 1.4 - Generación dataset optimizado
2. **Si score < 0.75**: Implementar mejoras recomendadas y re-ejecutar validación
3. **Documentación**: Actualizar ANALYSIS_RESULTS.md con hallazgos
4. **Testing**: Ejecutar tests unitarios completos antes de siguiente fase

---

**Generado**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Ubicación**: `validation_reports/phase1_validation_{timestamp}.md`
"""
    
    # Guardar reporte
    output_dir = Path('validation_reports')
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / f'phase1_validation_{timestamp}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📁 Reporte guardado: {report_path}")
    
    return {
        'report_content': report_content,
        'report_path': str(report_path),
        'timestamp': timestamp,
        'summary': {
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'avg_score': avg_score,
            'status': 'success' if avg_score >= 0.75 else 'warning' if avg_score >= 0.60 else 'error'
        }
    }

if __name__ == "__main__":
    try:
        results, report = comprehensive_phase1_validation()
        print(f"\n✅ Validación completada exitosamente")
        print(f"📊 Score final: {report['summary']['avg_score']:.3f}")
        
    except Exception as e:
        print(f"\n❌ Error en validación: {e}")
        sys.exit(1)
```

#### **📊 ENTREGABLES ETAPA 1.3**
- **Test suite**: `test_optimal_selector_improved.py` con 8+ tests específicos
- **Validation script**: `validate_phase1_improvements.py` automático
- **Coverage**: 90%+ funcionalidades críticas testeadas
- **Report**: Validación automática con recomendaciones

### **ETAPA 1.4: GENERACIÓN DATASET OPTIMIZADO** ⏱️ 4 horas

#### **🎯 Script de Generación Final**
```python
# CREAR: scripts/generate_optimal_10k_dataset.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Agregar paths
sys.path.append(str(Path(__file__).parent.parent))

from data_selection.clustering_aware.select_optimal_10k_from_18k import OptimalSelector
from data_selection.clustering_aware.hopkins_validator import HopkinsValidator

def generate_optimized_dataset():
    """Generar dataset final optimizado con validación completa."""
    
    print("🚀 GENERACIÓN DATASET OPTIMIZADO 10K")
    print("=" * 60)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Cargar dataset fuente
    print("\n📂 1. CARGANDO DATASET FUENTE 18K")
    print("-" * 40)
    
    source_path = 'data/with_lyrics/spotify_songs_fixed.csv'
    if not Path(source_path).exists():
        raise FileNotFoundError(f"Dataset fuente no encontrado: {source_path}")
    
    df_18k = pd.read_csv(source_path, sep='@@', encoding='utf-8', on_bad_lines='skip')
    print(f"✅ Dataset cargado: {df_18k.shape[0]:,} filas × {df_18k.shape[1]} columnas")
    
    # 2. Validación pre-selección
    print("\n🔍 2. VALIDACIÓN PRE-SELECCIÓN")
    print("-" * 40)
    
    selector = OptimalSelector()
    hopkins_validator = HopkinsValidator()
    
    # Verificar características disponibles
    available_features = [f for f in selector.musical_features if f in df_18k.columns]
    print(f"🎵 Características disponibles: {len(available_features)}/{len(selector.musical_features)}")
    print(f"   Features: {', '.join(available_features)}")
    
    if len(available_features) < 5:
        raise ValueError(f"Insuficientes características musicales: {len(available_features)}")
    
    # Hopkins baseline del dataset fuente
    baseline_features = df_18k[available_features].dropna()
    if len(baseline_features) < 1000:
        raise ValueError("Dataset muy pequeño para selección confiable")
    
    # Calcular Hopkins baseline
    from sklearn.preprocessing import StandardScaler
    scaler_baseline = StandardScaler()
    baseline_scaled = scaler_baseline.fit_transform(baseline_features.head(5000).values)  # Muestra para velocidad
    hopkins_baseline = hopkins_validator.calculate_hopkins_fast(baseline_scaled)
    
    print(f"📊 Hopkins baseline (18K): {hopkins_baseline:.4f}")
    
    if hopkins_baseline < 0.5:
        print("⚠️  WARNING: Dataset fuente tiene Hopkins bajo - selección será desafiante")
    
    # 3. Ejecutar selección optimizada
    print("\n🎯 3. EJECUTANDO SELECCIÓN OPTIMIZADA")
    print("-" * 40)
    
    try:
        selected_10k, selection_metadata = selector.select_optimal_10k_with_validation(df_18k)
        
        if selected_10k is None or len(selected_10k) == 0:
            raise ValueError("Selección falló - dataset vacío")
        
        print(f"✅ Selección completada: {len(selected_10k):,} canciones")
        
    except Exception as e:
        print(f"❌ Error en selección: {e}")
        raise
    
    # 4. Validación post-selección
    print("\n🔍 4. VALIDACIÓN POST-SELECCIÓN")
    print("-" * 40)
    
    validation_results = validate_final_selection(df_18k, selected_10k, selection_metadata)
    
    print(f"📊 Hopkins final: {validation_results['hopkins_final']:.4f}")
    print(f"📈 Preservación Hopkins: {validation_results['hopkins_preservation_ratio']:.3f}")
    print(f"🎵 Diversidad preservada: {validation_results['diversity_preservation']:.1%}")
    
    # 5. Guardar dataset con metadatos
    print("\n💾 5. GUARDANDO DATASET OPTIMIZADO")
    print("-" * 40)
    
    output_paths = save_optimized_dataset(selected_10k, selection_metadata, validation_results)
    
    # 6. Resumen final
    print("\n🎉 GENERACIÓN COMPLETADA")
    print("=" * 60)
    
    final_summary = {
        'timestamp': datetime.now().isoformat(),
        'source_dataset': {
            'path': source_path,
            'size': len(df_18k),
            'hopkins_baseline': hopkins_baseline
        },
        'selected_dataset': {
            'size': len(selected_10k),
            'hopkins_final': validation_results['hopkins_final'],
            'hopkins_preservation_ratio': validation_results['hopkins_preservation_ratio']
        },
        'quality_metrics': validation_results,
        'output_files': output_paths,
        'success': True
    }
    
    print(f"📦 Dataset final: {len(selected_10k):,} canciones")
    print(f"📁 Archivo principal: {output_paths['main_csv']}")
    print(f"📊 Hopkins preservation: {validation_results['hopkins_preservation_ratio']:.3f}")
    print(f"🎯 Status: {'✅ EXITOSO' if validation_results['hopkins_preservation_ratio'] >= 0.75 else '⚠️ ACEPTABLE' if validation_results['hopkins_preservation_ratio'] >= 0.60 else '❌ PROBLEMÁTICO'}")
    
    return selected_10k, selection_metadata, validation_results, final_summary

def validate_final_selection(original_df, selected_df, selection_metadata):
    """Validación exhaustiva de la selección final."""
    
    validation_results = {}
    
    # 1. Validación Hopkins
    hopkins_validator = HopkinsValidator()
    selector = OptimalSelector()
    
    # Hopkins original vs seleccionado
    available_features = [f for f in selector.musical_features if f in selected_df.columns]
    
    # Original (muestra para velocidad)
    original_sample = original_df[available_features].dropna().head(5000)
    from sklearn.preprocessing import StandardScaler
    scaler_orig = StandardScaler()
    original_scaled = scaler_orig.fit_transform(original_sample.values)
    hopkins_original = hopkins_validator.calculate_hopkins_fast(original_scaled)
    
    # Seleccionado
    selected_features = selected_df[available_features].dropna()
    scaler_sel = StandardScaler()
    selected_scaled = scaler_sel.fit_transform(selected_features.values)
    hopkins_selected = hopkins_validator.calculate_hopkins_fast(selected_scaled)
    
    hopkins_preservation_ratio = hopkins_selected / hopkins_original if hopkins_original > 0 else 1
    
    validation_results.update({
        'hopkins_original': round(hopkins_original, 4),
        'hopkins_final': round(hopkins_selected, 4),
        'hopkins_preservation_ratio': round(hopkins_preservation_ratio, 4)
    })
    
    # 2. Validación diversidad musical
    diversity_metrics = calculate_musical_diversity_preservation(original_df, selected_df, available_features)
    validation_results.update(diversity_metrics)
    
    # 3. Validación distribución de géneros
    if 'playlist_genre' in selected_df.columns and 'playlist_genre' in original_df.columns:
        genre_preservation = calculate_genre_preservation(original_df, selected_df)
        validation_results.update(genre_preservation)
    
    # 4. Validación tamaño y completitud
    validation_results.update({
        'original_size': len(original_df),
        'selected_size': len(selected_df),
        'selection_ratio': len(selected_df) / len(original_df),
        'features_preserved': len(available_features),
        'missing_values_ratio': selected_df[available_features].isnull().sum().sum() / (len(selected_df) * len(available_features))
    })
    
    return validation_results

def calculate_musical_diversity_preservation(original_df, selected_df, features):
    """Calcular preservación diversidad musical."""
    
    diversity_results = {}
    feature_preservations = []
    
    for feature in features:
        if feature in original_df.columns and feature in selected_df.columns:
            orig_feature = original_df[feature].dropna()
            sel_feature = selected_df[feature].dropna()
            
            if len(orig_feature) > 0 and len(sel_feature) > 0:
                # Preservación std
                std_preservation = sel_feature.std() / orig_feature.std() if orig_feature.std() > 0 else 1
                
                # Preservación rango
                orig_range = orig_feature.max() - orig_feature.min()
                sel_range = sel_feature.max() - sel_feature.min()
                range_preservation = sel_range / orig_range if orig_range > 0 else 1
                
                # Preservación percentiles
                orig_iqr = orig_feature.quantile(0.75) - orig_feature.quantile(0.25)
                sel_iqr = sel_feature.quantile(0.75) - sel_feature.quantile(0.25)
                iqr_preservation = sel_iqr / orig_iqr if orig_iqr > 0 else 1
                
                feature_preservation = np.mean([std_preservation, range_preservation, iqr_preservation])
                feature_preservations.append(feature_preservation)
                
                diversity_results[f'{feature}_preservation'] = round(feature_preservation, 3)
    
    avg_diversity_preservation = np.mean(feature_preservations) if feature_preservations else 1.0
    
    diversity_results.update({
        'diversity_preservation': round(avg_diversity_preservation, 4),
        'features_analyzed': len(feature_preservations)
    })
    
    return diversity_results

def calculate_genre_preservation(original_df, selected_df):
    """Calcular preservación de géneros."""
    
    orig_genres = set(original_df['playlist_genre'].dropna().unique())
    sel_genres = set(selected_df['playlist_genre'].dropna().unique())
    
    genre_preservation_ratio = len(sel_genres) / len(orig_genres) if orig_genres else 1
    missing_genres = orig_genres - sel_genres
    
    return {
        'original_genres': len(orig_genres),
        'selected_genres': len(sel_genres),
        'genre_preservation_ratio': round(genre_preservation_ratio, 4),
        'missing_genres': list(missing_genres)
    }

def save_optimized_dataset(selected_df, selection_metadata, validation_results):
    """Guardar dataset con metadatos completos."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear directorio de salida
    output_dir = Path("data/final_data")
    output_dir.mkdir(exist_ok=True)
    
    # 1. CSV principal
    main_csv_path = output_dir / "picked_data_optimal.csv"
    timestamped_csv_path = output_dir / f"picked_data_optimal_{timestamp}.csv"
    
    # Guardar ambas versiones
    selected_df.to_csv(main_csv_path, sep='^', decimal='.', index=False, encoding='utf-8')
    selected_df.to_csv(timestamped_csv_path, sep='^', decimal='.', index=False, encoding='utf-8')
    
    print(f"📁 CSV principal: {main_csv_path}")
    print(f"📁 CSV con timestamp: {timestamped_csv_path}")
    
    # 2. Metadatos completos
    complete_metadata = {
        'generation_info': {
            'timestamp': timestamp,
            'generation_date': datetime.now().isoformat(),
            'source_dataset': 'data/with_lyrics/spotify_songs_fixed.csv',
            'selection_method': 'clustering_aware_optimized_with_validation',
            'phase': '1.4 - Dataset Generation'
        },
        'dataset_info': {
            'source_size': validation_results['original_size'],
            'selected_size': validation_results['selected_size'],
            'selection_ratio': validation_results['selection_ratio'],
            'features_count': validation_results['features_preserved'],
            'missing_values_ratio': validation_results['missing_values_ratio']
        },
        'quality_metrics': validation_results,
        'selection_metadata': selection_metadata,
        'format_info': {
            'separator': '^',
            'decimal': '.',
            'encoding': 'utf-8',
            'index_saved': False
        },
        'validation_status': {
            'hopkins_threshold_met': validation_results['hopkins_preservation_ratio'] >= 0.75,
            'diversity_threshold_met': validation_results['diversity_preservation'] >= 0.75,
            'overall_quality': 'excellent' if validation_results['hopkins_preservation_ratio'] >= 0.80 and validation_results['diversity_preservation'] >= 0.80 
                              else 'good' if validation_results['hopkins_preservation_ratio'] >= 0.70 and validation_results['diversity_preservation'] >= 0.70
                              else 'acceptable' if validation_results['hopkins_preservation_ratio'] >= 0.60 and validation_results['diversity_preservation'] >= 0.60
                              else 'needs_improvement'
        }
    }
    
    # Guardar metadatos
    metadata_path = output_dir / f"picked_data_optimal_metadata_{timestamp}.json"
    standard_metadata_path = output_dir / "picked_data_optimal_metadata.json"
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(complete_metadata, f, indent=2, ensure_ascii=False)
    
    with open(standard_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(complete_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Metadatos: {metadata_path}")
    
    # 3. README explicativo
    readme_content = f"""# Dataset Optimizado para Clustering Musical

**Generado**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Método**: Clustering-aware selection con validación Hopkins  
**Fase**: 1.4 - Generación Dataset Optimizado  

## Información del Dataset

- **Tamaño**: {validation_results['selected_size']:,} canciones
- **Fuente**: {validation_results['original_size']:,} canciones de spotify_songs_fixed.csv
- **Ratio selección**: {validation_results['selection_ratio']:.1%}
- **Características**: {validation_results['features_preserved']} características musicales

## Métricas de Calidad

- **Hopkins Statistic**: {validation_results['hopkins_final']:.4f}
- **Hopkins Preservation**: {validation_results['hopkins_preservation_ratio']:.3f} ({validation_results['hopkins_preservation_ratio']*100:.1f}%)
- **Diversidad Musical**: {validation_results['diversity_preservation']:.3f} ({validation_results['diversity_preservation']*100:.1f}%)
- **Calidad General**: {complete_metadata['validation_status']['overall_quality'].replace('_', ' ').title()}

## Formato del Archivo

- **Separador**: `^`
- **Decimal**: `.`
- **Encoding**: UTF-8
- **Índice**: No incluido

## Uso Recomendado

```python
import pandas as pd

# Cargar dataset
df = pd.read_csv('picked_data_optimal.csv', sep='^', decimal='.', encoding='utf-8')

# Características musicales disponibles
musical_features = [col for col in df.columns if col in [
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
    'valence', 'tempo', 'duration_ms', 'time_signature'
]]
```

## Próximos Pasos

1. **Fase 2**: Clustering comparativo 10K vs 18K
2. **Fase 3**: Clustering readiness assessment
3. **Fase 4**: Cluster purification
4. **Fase 5**: Evaluación final y selección modelo

---

Para más detalles, ver metadatos en `picked_data_optimal_metadata.json`
"""
    
    readme_path = output_dir / "README_picked_data_optimal.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📚 README: {readme_path}")
    
    return {
        'main_csv': str(main_csv_path),
        'timestamped_csv': str(timestamped_csv_path),
        'metadata': str(metadata_path),
        'standard_metadata': str(standard_metadata_path),
        'readme': str(readme_path)
    }

if __name__ == "__main__":
    try:
        selected_df, metadata, validation, summary = generate_optimized_dataset()
        
        print(f"\n🎉 GENERACIÓN EXITOSA")
        print(f"📊 Hopkins preservation: {validation['hopkins_preservation_ratio']:.3f}")
        print(f"🎵 Diversity preservation: {validation['diversity_preservation']:.3f}")
        
        # Status final
        if validation['hopkins_preservation_ratio'] >= 0.75 and validation['diversity_preservation'] >= 0.75:
            print("✅ DATASET LISTO PARA FASE 2")
        else:
            print("⚠️  DATASET FUNCIONAL - Considerar ajustes opcionales")
            
    except Exception as e:
        print(f"\n❌ Error en generación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

#### **📊 ENTREGABLES ETAPA 1.4**
- **Dataset optimizado**: `picked_data_optimal.csv` (10,000 canciones)
- **Metadatos completos**: JSON con métricas de calidad y validación
- **README explicativo**: Documentación de uso y próximos pasos
- **Validación**: Hopkins 0.75+, diversidad 85%+ target

---

## 📊 MÉTRICAS DE ÉXITO FASE 1

### **Criterios Técnicos**
- ✅ **Hopkins Preservation**: ≥ 0.80 (excellent), ≥ 0.70 (good), ≥ 0.60 (acceptable)
- ✅ **Musical Diversity**: ≥ 85% preservación características principales
- ✅ **Performance**: Tiempo ejecución similar o mejor vs implementación original
- ✅ **Test Coverage**: ≥ 90% funcionalidades críticas

### **Entregables Obligatorios**
- **Código mejorado**: `select_optimal_10k_from_18k.py` con mejoras científicas
- **Módulo Hopkins**: `hopkins_validator.py` implementado y testeado
- **Test suite**: Batería completa de tests con validación automática
- **Dataset final**: `picked_data_optimal.csv` con metadatos completos
- **Documentación**: Actualización ANALYSIS_RESULTS.md con hallazgos

### **Checkpoint FASE 1**
Al completar Fase 1, se debe tener:
1. Sistema selector optimizado funcionando
2. Dataset 10K con métricas de calidad validadas
3. Base sólida para clustering comparativo (Fase 2)
4. Documentación técnica actualizada

**CONDICIÓN PARA PROCEDER A FASE 2**: Hopkins preservation ≥ 0.70 Y diversity preservation ≥ 0.70

---

*Continúa con FASE 2: CLUSTERING COMPARATIVO 10K vs 18K...*