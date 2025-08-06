#!/usr/bin/env python3
"""
Script directo para analizar clustering readiness del dataset spotify_songs_fixed.csv

Este script importa directamente las dependencias necesarias para evitar problemas
de importaciones del módulo exploratory_analysis.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importaciones para clustering readiness
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from collections import Counter

class ClusteringReadinessDirect:
    """
    Versión directa del evaluador de clustering readiness.
    Evita problemas de importaciones del módulo exploratory_analysis.
    """
    
    def __init__(self):
        self.musical_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo', 'duration_ms', 'time_signature'
        ]
    
    def load_and_prepare_data(self, dataset_path):
        """Cargar y preparar el dataset."""
        if not os.path.exists(dataset_path):
            print(f"❌ ERROR: Dataset no encontrado en {dataset_path}")
            return None, None
        
        # Cargar con separador @@ usando motor python
        try:
            df = pd.read_csv(dataset_path, sep='@@', encoding='utf-8', 
                           on_bad_lines='skip', engine='python')
            print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            
            # Identificar características musicales disponibles
            available_features = [f for f in self.musical_features if f in df.columns]
            print(f"🎵 Características musicales disponibles: {len(available_features)}/13")
            print(f"📋 Features: {', '.join(available_features)}")
            
            if len(available_features) < 8:
                print("❌ ERROR: Insuficientes características musicales para análisis")
                return None, None
                
            return df, available_features
            
        except Exception as e:
            print(f"❌ ERROR cargando dataset: {e}")
            return None, None
    
    def prepare_features(self, df, features):
        """Preparar características para análisis."""
        # Extraer y limpiar datos
        X = df[features].copy()
        original_size = len(X)
        X = X.dropna()
        cleaned_size = len(X)
        
        print(f"🧹 Limpieza: {original_size:,} → {cleaned_size:,} filas (-{original_size-cleaned_size:,} nulos)")
        
        if len(X) < 100:
            print("❌ ERROR: Dataset muy pequeño después de limpieza")
            return None
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled
    
    def calculate_hopkins_statistic(self, X, sample_size=200):
        """Calcular Hopkins Statistic."""
        try:
            n, d = X.shape
            sample_size = min(sample_size, int(0.1 * n))
            
            # Puntos aleatorios uniformes
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            uniform_points = np.random.uniform(min_vals, max_vals, size=(sample_size, d))
            
            # Muestra de datos reales
            sample_indices = np.random.choice(n, sample_size, replace=False)
            real_sample = X[sample_indices]
            
            # Distancias mínimas para puntos uniformes
            nbrs_uniform = NearestNeighbors(n_neighbors=1).fit(X)
            u_distances, _ = nbrs_uniform.kneighbors(uniform_points)
            U = np.sum(u_distances)
            
            # Distancias mínimas para puntos reales
            remaining_data = np.delete(X, sample_indices, axis=0)
            nbrs_real = NearestNeighbors(n_neighbors=1).fit(remaining_data)
            w_distances, _ = nbrs_real.kneighbors(real_sample)
            W = np.sum(w_distances)
            
            hopkins = U / (U + W)
            return float(hopkins)
            
        except Exception as e:
            print(f"❌ Error calculando Hopkins: {e}")
            return 0.5
    
    def find_optimal_k(self, X, k_range=(2, 12)):
        """Encontrar K óptimo usando múltiples métodos."""
        # Limitar muestra para performance
        if len(X) > 3000:
            indices = np.random.choice(len(X), 3000, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            
        k_min, k_max = k_range
        k_values = range(k_min, min(k_max + 1, len(X_sample) // 2))
        
        metrics = {
            'k_values': list(k_values),
            'silhouette_scores': []
        }
        
        print(f"🔍 Evaluando K óptimo en rango {k_min}-{k_max}...")
        
        best_k = k_min
        best_silhouette = -1
        
        for k in k_values:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_sample)
                
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X_sample, labels)
                    metrics['silhouette_scores'].append(silhouette)
                    
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_k = k
                else:
                    metrics['silhouette_scores'].append(-1.0)
                    
                print(f"   K={k}: Silhouette = {silhouette:.3f}" if len(np.unique(labels)) > 1 else f"   K={k}: Cluster único")
                    
            except Exception as e:
                print(f"   K={k}: Error - {e}")
                metrics['silhouette_scores'].append(-1.0)
        
        return {
            'recommended_k': best_k,
            'best_silhouette': best_silhouette,
            'all_metrics': metrics
        }
    
    def analyze_separability(self, X):
        """Analizar separabilidad del dataset."""
        # Muestra para análisis si es muy grande
        if len(X) > 2000:
            indices = np.random.choice(len(X), 2000, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Calcular distancias
        distances = pdist(X_sample, metric='euclidean')
        
        distance_stats = {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
        
        # Análisis k-nearest neighbors
        k = min(10, len(X_sample) // 4)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_sample)
        distances_knn, _ = nbrs.kneighbors(X_sample)
        avg_knn_distance = np.mean(distances_knn[:, 1:])
        
        # Score de separabilidad
        mean_dist = distance_stats['mean_distance']
        std_dist = distance_stats['std_distance']
        
        if mean_dist > 0:
            cv = std_dist / mean_dist
            neighbor_ratio = avg_knn_distance / mean_dist
            separability_score = min(1.0, (cv + neighbor_ratio) / 2)
        else:
            separability_score = 0.0
        
        # Predecir Silhouette esperado
        base_silhouette = separability_score * 0.6
        expected_silhouette = (max(0, base_silhouette - 0.1), min(1, base_silhouette + 0.1))
        
        return {
            'separability_score': separability_score,
            'expected_silhouette_range': expected_silhouette,
            'distance_stats': distance_stats,
            'avg_neighbor_distance': avg_knn_distance
        }
    
    def analyze_features(self, df, features):
        """Analizar calidad de características."""
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Varianzas
        feature_variances = np.var(X_scaled, axis=0)
        
        # Correlaciones
        correlation_matrix = np.corrcoef(X_scaled.T)
        
        # Ranking de características
        feature_scores = []
        for i, feature in enumerate(features):
            variance_score = feature_variances[i]
            
            # Penalizar alta correlación
            correlations = np.abs(correlation_matrix[i])
            correlations[i] = 0
            max_correlation = np.max(correlations)
            
            final_score = variance_score * (1 - max_correlation ** 2)
            
            feature_scores.append({
                'feature': feature,
                'variance_score': variance_score,
                'max_correlation': max_correlation,
                'final_score': final_score
            })
        
        feature_scores.sort(key=lambda x: x['final_score'], reverse=True)
        
        redundant_features = [f['feature'] for f in feature_scores if f['max_correlation'] > 0.8]
        recommended_features = [f['feature'] for f in feature_scores[:8]]
        
        return {
            'feature_ranking': feature_scores,
            'redundant_features': redundant_features,
            'recommended_features': recommended_features,
            'n_features_analyzed': len(features)
        }
    
    def calculate_readiness_score(self, hopkins, separability, features, k_analysis):
        """Calcular score final de clustering readiness."""
        
        # Componentes del score
        hopkins_score = min(30, hopkins * 30)
        feature_score = min(25, (len(features['recommended_features']) / 8) * 25)
        separability_score = min(20, separability['separability_score'] * 20)
        
        # Score basado en varianza
        high_variance_features = sum(1 for f in features['feature_ranking'] if f['variance_score'] > 0.5)
        distribution_score = min(15, (high_variance_features / len(features['feature_ranking'])) * 15)
        
        # Score de preprocessing
        redundant_ratio = len(features['redundant_features']) / len(features['feature_ranking'])
        preprocessing_score = max(0, 10 * (1 - redundant_ratio))
        
        total_score = hopkins_score + feature_score + separability_score + distribution_score + preprocessing_score
        
        # Determinar nivel
        if total_score >= 80:
            level = 'excellent'
        elif total_score >= 60:
            level = 'good'
        elif total_score >= 40:
            level = 'fair'
        else:
            level = 'poor'
        
        return {
            'readiness_score': round(total_score, 1),
            'readiness_level': level,
            'score_breakdown': {
                'clustering_tendency': hopkins_score,
                'feature_quality': feature_score,
                'separability': separability_score,
                'distribution_compatibility': distribution_score,
                'preprocessing_simplicity': preprocessing_score
            }
        }

def main():
    print("🎵 ANÁLISIS DIRECTO DE CLUSTERING READINESS")
    print("="*60)
    
    # Inicializar analizador
    analyzer = ClusteringReadinessDirect()
    
    # Cargar dataset
    dataset_path = 'data/with_lyrics/spotify_songs_fixed.csv'
    df, available_features = analyzer.load_and_prepare_data(dataset_path)
    
    if df is None:
        return
    
    print(f"\n📊 DATASET CARGADO EXITOSAMENTE")
    print(f"📦 Dimensiones: {df.shape}")
    print(f"🎵 Características musicales: {len(available_features)}/13")
    
    # Preparar datos
    print(f"\n🔧 PREPARANDO DATOS...")
    X = analyzer.prepare_features(df, available_features)
    if X is None:
        return
    
    # Análisis de clustering readiness
    print(f"\n🧮 ANÁLISIS DE CLUSTERING READINESS")
    print("-" * 40)
    
    # 1. Hopkins Statistic
    print("📊 Calculando Hopkins Statistic...")
    hopkins = analyzer.calculate_hopkins_statistic(X)
    print(f"   Hopkins Statistic: {hopkins:.3f}")
    
    if hopkins > 0.75:
        hopkins_interp = "EXCELENTE - Altamente clusterable"
    elif hopkins > 0.6:
        hopkins_interp = "BUENO - Moderadamente clusterable"
    elif hopkins > 0.5:
        hopkins_interp = "ACEPTABLE - Clustering posible"
    else:
        hopkins_interp = "PROBLEMÁTICO - Datos aleatorios"
    
    print(f"   Interpretación: {hopkins_interp}")
    
    # 2. K Óptimo
    print(f"\n🎯 Encontrando K óptimo...")
    k_analysis = analyzer.find_optimal_k(X)
    print(f"   K recomendado: {k_analysis['recommended_k']}")
    print(f"   Mejor Silhouette: {k_analysis['best_silhouette']:.3f}")
    
    # 3. Separabilidad
    print(f"\n📐 Analizando separabilidad...")
    separability = analyzer.analyze_separability(X)
    print(f"   Score separabilidad: {separability['separability_score']:.3f}")
    print(f"   Silhouette esperado: {separability['expected_silhouette_range']}")
    
    # 4. Características
    print(f"\n🎵 Analizando características...")
    features = analyzer.analyze_features(df, available_features)
    print(f"   Top 3 características: {', '.join(features['recommended_features'][:3])}")
    if features['redundant_features']:
        print(f"   Características redundantes: {', '.join(features['redundant_features'])}")
    
    # 5. Score final
    print(f"\n🏆 CLUSTERING READINESS SCORE")
    print("-" * 40)
    
    final_results = analyzer.calculate_readiness_score(hopkins, separability, features, k_analysis)
    
    print(f"🎯 SCORE FINAL: {final_results['readiness_score']}/100")
    print(f"📊 NIVEL: {final_results['readiness_level'].upper()}")
    
    print(f"\n📋 DESGLOSE:")
    for component, score in final_results['score_breakdown'].items():
        print(f"   {component.replace('_', ' ').title()}: {score:.1f}/100")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    score = final_results['readiness_score']
    
    if score >= 80:
        print("✅ Dataset óptimo para clustering")
        print("🚀 Proceder con selección estándar de 10K canciones")
    elif score >= 60:
        print("⚠️  Dataset adecuado con optimización")
        print("🔧 Aplicar transformaciones antes de clustering:")
        if hopkins < 0.6:
            print("   • Aumentar diversidad en selección")
        if separability['separability_score'] < 0.5:
            print("   • Considerar PCA o feature engineering")
        if features['redundant_features']:
            print(f"   • Eliminar características redundantes: {', '.join(features['redundant_features'])}")
    else:
        print("❌ Dataset problemático para clustering")
        print("🚨 Recomendaciones críticas:")
        print("   • Revisar estrategia de selección de datos")
        print("   • Considerar algoritmos alternativos (DBSCAN, GMM)")
        print("   • Aplicar feature engineering extensivo")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/clustering_readiness"
    os.makedirs(results_dir, exist_ok=True)
    
    results_data = {
        'timestamp': timestamp,
        'dataset_info': {
            'path': dataset_path,
            'shape': df.shape,
            'features_analyzed': available_features
        },
        'hopkins_statistic': hopkins,
        'optimal_k': k_analysis,
        'separability_analysis': separability,
        'feature_analysis': features,
        'final_score': final_results
    }
    
    output_file = f"{results_dir}/clustering_readiness_direct_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Resultados guardados en: {output_file}")
    print("🎉 ¡Análisis completado exitosamente!")

if __name__ == "__main__":
    main()