#!/usr/bin/env python3
"""
FASE 4.1: CLUSTER PURIFICATION SYSTEM
====================================

Sistema avanzado de purificación de clusters para mejorar Silhouette Score
mediante eliminación estratégica de puntos problemáticos.

Objetivo: Mejorar Silhouette de 0.1554 → 0.20-0.25 (+28-61% mejora)
Configuración base: Hierarchical + Baseline + K=3

Técnicas implementadas:
1. Negative Silhouette Removal
2. Cluster Outliers Removal  
3. Discriminative Feature Selection
4. Cluster Size Balancing
5. Hybrid Purification Strategy

Autor: Clustering Optimization Team
Fecha: 2025-01-12
Estado: FASE 4.1 - System Implementation
"""

import numpy as np
import pandas as pd
import os
import time
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports para clustering y métricas
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Imports para análisis estadístico y visualización
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

class ClusterPurifier:
    """
    Sistema avanzado de purificación de clusters.
    
    Funcionalidades:
    - Análisis de calidad de clusters
    - Múltiples estrategias de purificación
    - Evaluación automática before/after
    - Optimización iterativa
    - Métricas de trade-off
    """
    
    def __init__(self, base_path=None, random_state=42):
        """Inicializar purificador de clusters."""
        
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent.parent
        else:
            base_path = Path(base_path)
            
        self.base_path = base_path
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Configuración de características musicales  
        self.musical_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms'
        ]
        
        # Configuración de estrategias de purificación
        self.purification_strategies = {
            'remove_negative_silhouette': self._remove_negative_silhouette,
            'remove_outliers': self._remove_cluster_outliers,
            'feature_selection': self._select_discriminative_features,
            'balance_clusters': self._balance_cluster_sizes,
            'hybrid': self._hybrid_purification
        }
        
        # Almacenamiento de resultados
        self.purification_results = {}
        self.analysis_history = []
        
        # Configuración de salida
        self.output_dir = base_path / 'outputs' / 'fase4_purification'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 ClusterPurifier inicializado")
        print(f"📁 Base path: {base_path}")
        print(f"📊 Output dir: {self.output_dir}")
        print(f"🎯 Estrategias disponibles: {list(self.purification_strategies.keys())}")
    
    def load_baseline_configuration(self):
        """Cargar configuración óptima de FASE 2."""
        
        print("\n📂 Cargando configuración baseline de FASE 2...")
        
        # Configuración óptima identificada en FASE 2
        baseline_config = {
            'dataset_path': self.base_path / 'data' / 'with_lyrics' / 'spotify_songs_fixed.csv',
            'separator': '@@',
            'algorithm': 'hierarchical',
            'n_clusters': 3,
            'baseline_silhouette': 0.1554,
            'target_improvement': 0.28,  # 28% mínimo
            'target_silhouette': 0.20,   # Mínimo objetivo
            'optimal_silhouette': 0.25   # Objetivo óptimo
        }
        
        # Cargar dataset
        try:
            df = pd.read_csv(
                baseline_config['dataset_path'], 
                sep=baseline_config['separator'], 
                decimal='.', 
                encoding='utf-8'
            )
            
            print(f"✅ Dataset cargado: {len(df):,} filas × {len(df.columns)} columnas")
            
            # Extraer características musicales disponibles
            available_features = [f for f in self.musical_features if f in df.columns]
            features_df = df[available_features].dropna()
            
            print(f"🎵 Características disponibles: {len(available_features)}/{len(self.musical_features)}")
            print(f"📊 Datos limpios: {len(features_df):,} canciones × {len(available_features)} características")
            
            baseline_config.update({
                'dataset': df,
                'features_df': features_df,
                'available_features': available_features,
                'n_samples': len(features_df),
                'n_features': len(available_features)
            })
            
            return baseline_config
            
        except Exception as e:
            print(f"❌ Error cargando configuración baseline: {e}")
            return None
    
    def analyze_cluster_quality(self, data, cluster_labels, detailed=True):
        """
        Analizar calidad detallada de clusters.
        
        Args:
            data: Datos normalizados (numpy array)
            cluster_labels: Etiquetas de clusters
            detailed: Si incluir análisis detallado por cluster
            
        Returns:
            dict: Análisis completo de calidad
        """
        print(f"\n🔍 Analizando calidad de clusters...")
        
        # Métricas globales
        global_silhouette = silhouette_score(data, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(data, cluster_labels)
        davies_bouldin = davies_bouldin_score(data, cluster_labels)
        
        # Silhouette scores por punto
        sample_silhouette_values = silhouette_samples(data, cluster_labels)
        
        print(f"📊 Métricas globales:")
        print(f"   Silhouette Score: {global_silhouette:.4f}")
        print(f"   Calinski-Harabasz: {calinski_harabasz:.2f}")
        print(f"   Davies-Bouldin: {davies_bouldin:.4f}")
        
        quality_analysis = {
            'global_metrics': {
                'silhouette_score': global_silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'n_samples': len(data),
                'n_clusters': len(np.unique(cluster_labels)),
                'n_features': data.shape[1]
            },
            'sample_analysis': {
                'silhouette_samples': sample_silhouette_values,
                'negative_silhouette_count': np.sum(sample_silhouette_values < 0),
                'negative_silhouette_ratio': np.mean(sample_silhouette_values < 0),
                'silhouette_mean': np.mean(sample_silhouette_values),
                'silhouette_std': np.std(sample_silhouette_values),
                'silhouette_min': np.min(sample_silhouette_values),
                'silhouette_max': np.max(sample_silhouette_values)
            }
        }
        
        print(f"📈 Análisis por puntos:")
        print(f"   Puntos con Silhouette negativo: {quality_analysis['sample_analysis']['negative_silhouette_count']:,} ({quality_analysis['sample_analysis']['negative_silhouette_ratio']:.1%})")
        print(f"   Silhouette promedio: {quality_analysis['sample_analysis']['silhouette_mean']:.4f} ± {quality_analysis['sample_analysis']['silhouette_std']:.4f}")
        
        if detailed:
            # Análisis detallado por cluster
            cluster_analysis = {}
            
            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = data[cluster_mask]
                cluster_silhouettes = sample_silhouette_values[cluster_mask]
                
                # Calcular centroide y distancias
                centroid = np.mean(cluster_data, axis=0)
                distances_to_centroid = np.linalg.norm(cluster_data - centroid, axis=1)
                
                # Identificar outliers (> 2σ del centroide)
                distance_threshold = np.mean(distances_to_centroid) + 2 * np.std(distances_to_centroid)
                outlier_mask = distances_to_centroid > distance_threshold
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': np.sum(cluster_mask),
                    'size_ratio': np.mean(cluster_mask),
                    'silhouette_mean': np.mean(cluster_silhouettes),
                    'silhouette_std': np.std(cluster_silhouettes),
                    'negative_count': np.sum(cluster_silhouettes < 0),
                    'negative_ratio': np.mean(cluster_silhouettes < 0),
                    'centroid': centroid,
                    'intra_cluster_distances': {
                        'mean': np.mean(distances_to_centroid),
                        'std': np.std(distances_to_centroid),
                        'max': np.max(distances_to_centroid)
                    },
                    'outlier_count': np.sum(outlier_mask),
                    'outlier_ratio': np.mean(outlier_mask),
                    'cohesion_score': 1 / (1 + np.mean(distances_to_centroid))  # Custom cohesion metric
                }
                
                print(f"   🎵 Cluster {cluster_id}: {cluster_analysis[f'cluster_{cluster_id}']['size']:,} samples, "
                      f"Silhouette {cluster_analysis[f'cluster_{cluster_id}']['silhouette_mean']:.4f}, "
                      f"Outliers {cluster_analysis[f'cluster_{cluster_id}']['outlier_ratio']:.1%}")
            
            quality_analysis['cluster_analysis'] = cluster_analysis
        
        return quality_analysis
    
    def _remove_negative_silhouette(self, data, cluster_labels):
        """Eliminar puntos con silhouette score negativo."""
        
        print(f"\n🔧 Aplicando estrategia: Remove Negative Silhouette")
        
        silhouette_scores = silhouette_samples(data, cluster_labels)
        positive_mask = silhouette_scores >= 0
        
        purified_data = data[positive_mask]
        purified_labels = cluster_labels[positive_mask]
        
        removed_count = np.sum(~positive_mask)
        retention_ratio = len(purified_data) / len(data)
        
        print(f"   📊 Puntos eliminados: {removed_count:,} ({1-retention_ratio:.1%})")
        print(f"   📊 Datos retenidos: {len(purified_data):,} ({retention_ratio:.1%})")
        
        return purified_data, purified_labels, {
            'strategy': 'remove_negative_silhouette',
            'removed_count': removed_count,
            'retention_ratio': retention_ratio,
            'original_size': len(data),
            'purified_size': len(purified_data)
        }
    
    def _remove_cluster_outliers(self, data, cluster_labels, threshold=2.0):
        """Eliminar outliers por cluster usando distancia al centroide."""
        
        print(f"\n🔧 Aplicando estrategia: Remove Cluster Outliers (threshold={threshold}σ)")
        
        purified_indices = []
        outlier_counts_by_cluster = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) < 3:  # Skip clusters muy pequeños
                purified_indices.extend(cluster_indices)
                continue
            
            # Calcular centroide y distancias
            centroid = np.mean(cluster_data, axis=0)
            distances = np.linalg.norm(cluster_data - centroid, axis=1)
            
            # Threshold adaptativo por cluster
            distance_threshold = np.mean(distances) + threshold * np.std(distances)
            inlier_mask = distances <= distance_threshold
            
            # Agregar índices de inliers
            purified_indices.extend(cluster_indices[inlier_mask])
            
            outlier_count = np.sum(~inlier_mask)
            outlier_counts_by_cluster[cluster_id] = {
                'outliers': outlier_count,
                'total': len(cluster_data),
                'outlier_ratio': outlier_count / len(cluster_data)
            }
            
            print(f"   🎵 Cluster {cluster_id}: {outlier_count}/{len(cluster_data)} outliers eliminados ({outlier_count/len(cluster_data):.1%})")
        
        purified_data = data[purified_indices]
        purified_labels = cluster_labels[purified_indices]
        
        removed_count = len(data) - len(purified_data)
        retention_ratio = len(purified_data) / len(data)
        
        print(f"   📊 Total eliminados: {removed_count:,} ({1-retention_ratio:.1%})")
        print(f"   📊 Datos retenidos: {len(purified_data):,} ({retention_ratio:.1%})")
        
        return purified_data, purified_labels, {
            'strategy': 'remove_cluster_outliers',
            'threshold': threshold,
            'removed_count': removed_count,
            'retention_ratio': retention_ratio,
            'outlier_counts_by_cluster': outlier_counts_by_cluster,
            'original_size': len(data),
            'purified_size': len(purified_data)
        }
    
    def _select_discriminative_features(self, data, cluster_labels, k_features=8):
        """Seleccionar características más discriminativas para clustering."""
        
        print(f"\n🔧 Aplicando estrategia: Feature Selection (top {k_features} de {data.shape[1]})")
        
        if k_features >= data.shape[1]:
            print(f"   ⚠️  K features ({k_features}) >= total features ({data.shape[1]}), devolviendo datos originales")
            return data, cluster_labels, {
                'strategy': 'feature_selection',
                'k_features': k_features,
                'original_features': data.shape[1],
                'selected_features': data.shape[1],
                'feature_reduction': False
            }
        
        # Feature selection usando F-statistic
        selector = SelectKBest(score_func=f_classif, k=k_features)
        data_selected = selector.fit_transform(data, cluster_labels)
        
        # Obtener scores de características
        feature_scores = selector.scores_
        selected_feature_indices = selector.get_support(indices=True)
        
        print(f"   📊 Características seleccionadas: {k_features}/{data.shape[1]}")
        print(f"   📊 Reducción dimensional: {data.shape[1]} → {data_selected.shape[1]}")
        
        # Mostrar top características
        feature_ranking = sorted(zip(selected_feature_indices, feature_scores[selected_feature_indices]), 
                                key=lambda x: x[1], reverse=True)
        
        print(f"   🎯 Top 3 características discriminativas:")
        for i, (feat_idx, score) in enumerate(feature_ranking[:3]):
            feat_name = self.musical_features[feat_idx] if feat_idx < len(self.musical_features) else f"feature_{feat_idx}"
            print(f"      {i+1}. {feat_name}: {score:.2f}")
        
        return data_selected, cluster_labels, {
            'strategy': 'feature_selection',
            'k_features': k_features,
            'original_features': data.shape[1],
            'selected_features': data_selected.shape[1],
            'feature_scores': feature_scores,
            'selected_feature_indices': selected_feature_indices,
            'feature_ranking': feature_ranking,
            'feature_reduction': True
        }
    
    def _balance_cluster_sizes(self, data, cluster_labels, target_balance=0.8):
        """Balancear tamaños de clusters eliminando excess de clusters grandes."""
        
        print(f"\n🔧 Aplicando estrategia: Balance Cluster Sizes (target balance={target_balance})")
        
        # Analizar distribución actual
        unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        
        print(f"   📊 Distribución actual:")
        for label, size in zip(unique_labels, cluster_sizes):
            print(f"      Cluster {label}: {size:,} samples ({size/len(data):.1%})")
        
        # Calcular target sizes
        total_samples = len(data)
        target_size = total_samples // len(unique_labels)  # Tamaño ideal por cluster
        
        # Identificar clusters que necesitan reducción
        purified_indices = []
        reduction_log = {}
        
        for label, current_size in zip(unique_labels, cluster_sizes):
            cluster_mask = cluster_labels == label
            cluster_indices = np.where(cluster_mask)[0]
            
            if current_size <= target_size * (1 + target_balance):
                # Cluster ya está balanceado
                purified_indices.extend(cluster_indices)
                reduction_log[label] = {'action': 'no_reduction', 'kept': current_size, 'removed': 0}
            else:
                # Cluster necesita reducción
                target_keep = int(target_size * (1 + target_balance))
                
                # Seleccionar mejores puntos por silhouette score
                cluster_data = data[cluster_mask]
                cluster_silhouettes = silhouette_samples(data, cluster_labels)[cluster_mask]
                
                # Ordenar por silhouette score (mantener los mejores)
                best_indices = np.argsort(cluster_silhouettes)[-target_keep:]
                selected_cluster_indices = cluster_indices[best_indices]
                
                purified_indices.extend(selected_cluster_indices)
                
                reduction_log[label] = {
                    'action': 'reduced',
                    'original': current_size,
                    'kept': target_keep,
                    'removed': current_size - target_keep,
                    'reduction_ratio': (current_size - target_keep) / current_size
                }
                
                print(f"      Cluster {label}: {current_size} → {target_keep} (-{current_size - target_keep}, -{reduction_log[label]['reduction_ratio']:.1%})")
        
        purified_data = data[purified_indices]
        purified_labels = cluster_labels[purified_indices]
        
        removed_count = len(data) - len(purified_data)
        retention_ratio = len(purified_data) / len(data)
        
        print(f"   📊 Total balanceado: {removed_count:,} eliminados ({1-retention_ratio:.1%})")
        
        return purified_data, purified_labels, {
            'strategy': 'balance_cluster_sizes',
            'target_balance': target_balance,
            'removed_count': removed_count,
            'retention_ratio': retention_ratio,
            'reduction_log': reduction_log,
            'original_size': len(data),
            'purified_size': len(purified_data)
        }
    
    def _hybrid_purification(self, data, cluster_labels):
        """Estrategia híbrida combinando múltiples técnicas."""
        
        print(f"\n🔧 Aplicando estrategia: Hybrid Purification")
        print(f"   Combinando: Negative Silhouette + Mild Outlier Removal + Feature Selection")
        
        current_data = data.copy()
        current_labels = cluster_labels.copy()
        hybrid_log = []
        
        # Paso 1: Remove negative silhouette (más agresivo)
        print(f"\n   🔄 Paso 1: Negative Silhouette Removal")
        step1_data, step1_labels, step1_info = self._remove_negative_silhouette(current_data, current_labels)
        hybrid_log.append(step1_info)
        
        # Paso 2: Mild outlier removal (menos agresivo threshold=2.5)
        print(f"\n   🔄 Paso 2: Mild Outlier Removal")
        step2_data, step2_labels, step2_info = self._remove_cluster_outliers(step1_data, step1_labels, threshold=2.5)
        hybrid_log.append(step2_info)
        
        # Paso 3: Feature selection (mantener 9 de 12 características)
        print(f"\n   🔄 Paso 3: Feature Selection")
        k_features = max(6, min(9, step2_data.shape[1] - 2))  # Conservador
        step3_data, step3_labels, step3_info = self._select_discriminative_features(step2_data, step2_labels, k_features)
        hybrid_log.append(step3_info)
        
        # Resumen de hybrid purification
        total_removed = len(data) - len(step3_data)
        total_retention = len(step3_data) / len(data)
        
        print(f"\n   📊 RESUMEN HYBRID PURIFICATION:")
        print(f"      Original: {len(data):,} samples × {data.shape[1]} features")
        print(f"      Final: {len(step3_data):,} samples × {step3_data.shape[1]} features")
        print(f"      Eliminados: {total_removed:,} samples ({1-total_retention:.1%})")
        print(f"      Retenidos: {len(step3_data):,} samples ({total_retention:.1%})")
        
        return step3_data, step3_labels, {
            'strategy': 'hybrid_purification',
            'steps': hybrid_log,
            'total_removed': total_removed,
            'total_retention': total_retention,
            'original_size': len(data),
            'purified_size': len(step3_data),
            'feature_reduction': data.shape[1] != step3_data.shape[1],
            'original_features': data.shape[1],
            'final_features': step3_data.shape[1]
        }
    
    def apply_purification_strategy(self, data, cluster_labels, strategy='hybrid'):
        """
        Aplicar estrategia de purification específica.
        
        Args:
            data: Datos normalizados
            cluster_labels: Etiquetas de clusters
            strategy: Estrategia a aplicar
            
        Returns:
            tuple: (purified_data, purified_labels, purification_info, quality_analysis)
        """
        if strategy not in self.purification_strategies:
            raise ValueError(f"Estrategia '{strategy}' no disponible. Opciones: {list(self.purification_strategies.keys())}")
        
        print(f"\n🚀 APLICANDO PURIFICATION STRATEGY: {strategy.upper()}")
        print("="*60)
        
        # Análisis de calidad inicial
        print(f"📊 Análisis de calidad ANTES de purification:")
        quality_before = self.analyze_cluster_quality(data, cluster_labels, detailed=False)
        
        # Aplicar estrategia de purification
        start_time = time.time()
        purified_data, purified_labels, purification_info = self.purification_strategies[strategy](data, cluster_labels)
        purification_time = time.time() - start_time
        
        # Análisis de calidad después de purification
        print(f"\n📊 Análisis de calidad DESPUÉS de purification:")
        quality_after = self.analyze_cluster_quality(purified_data, purified_labels, detailed=False)
        
        # Calcular mejoras
        silhouette_improvement = quality_after['global_metrics']['silhouette_score'] - quality_before['global_metrics']['silhouette_score']
        relative_improvement = silhouette_improvement / quality_before['global_metrics']['silhouette_score']
        
        # Resumen de resultados
        print(f"\n🎯 RESUMEN DE PURIFICATION:")
        print(f"   Estrategia: {strategy}")
        print(f"   Tiempo ejecución: {purification_time:.2f}s")
        print(f"   Silhouette antes: {quality_before['global_metrics']['silhouette_score']:.4f}")
        print(f"   Silhouette después: {quality_after['global_metrics']['silhouette_score']:.4f}")
        print(f"   Mejora absoluta: {silhouette_improvement:+.4f}")
        print(f"   Mejora relativa: {relative_improvement:+.1%}")
        print(f"   Datos retenidos: {purification_info.get('retention_ratio', 1.0):.1%}")
        
        # Compilar resultado completo
        complete_result = {
            'strategy': strategy,
            'execution_time': purification_time,
            'quality_before': quality_before,
            'quality_after': quality_after,
            'improvements': {
                'silhouette_absolute': silhouette_improvement,
                'silhouette_relative': relative_improvement,
                'calinski_harabasz_change': quality_after['global_metrics']['calinski_harabasz_score'] - quality_before['global_metrics']['calinski_harabasz_score'],
                'davies_bouldin_change': quality_after['global_metrics']['davies_bouldin_score'] - quality_before['global_metrics']['davies_bouldin_score']
            },
            'purification_info': purification_info,
            'timestamp': datetime.now().isoformat()
        }
        
        return purified_data, purified_labels, complete_result
    
    def compare_purification_strategies(self, data, cluster_labels, strategies=None):
        """
        Comparar múltiples estrategias de purification.
        
        Args:
            data: Datos normalizados
            cluster_labels: Etiquetas de clusters  
            strategies: Lista de estrategias (None = todas)
            
        Returns:
            dict: Resultados comparativos completos
        """
        if strategies is None:
            strategies = list(self.purification_strategies.keys())
        
        print(f"\n🔬 COMPARACIÓN DE ESTRATEGIAS DE PURIFICATION")
        print("="*70)
        print(f"Estrategias a comparar: {strategies}")
        print(f"Dataset: {len(data):,} samples × {data.shape[1]} features")
        
        comparison_results = {
            'comparison_config': {
                'strategies': strategies,
                'original_samples': len(data),
                'original_features': data.shape[1],
                'n_clusters': len(np.unique(cluster_labels)),
                'timestamp': datetime.now().isoformat()
            },
            'strategy_results': {},
            'comparative_analysis': {},
            'best_strategy': None
        }
        
        # Ejecutar cada estrategia
        for strategy in strategies:
            print(f"\n" + "🔄"*20 + f" TESTING: {strategy.upper()} " + "🔄"*20)
            
            try:
                purified_data, purified_labels, result = self.apply_purification_strategy(
                    data, cluster_labels, strategy
                )
                
                comparison_results['strategy_results'][strategy] = result
                
                # Agregar datos purificados para análisis posterior si es necesario
                # (Nota: No guardamos los datos para evitar uso excesivo de memoria)
                
            except Exception as e:
                print(f"❌ Error en estrategia {strategy}: {e}")
                comparison_results['strategy_results'][strategy] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Análisis comparativo
        comparison_results['comparative_analysis'] = self._generate_comparative_analysis(
            comparison_results['strategy_results']
        )
        
        # Seleccionar mejor estrategia
        comparison_results['best_strategy'] = self._select_best_strategy(
            comparison_results['strategy_results']
        )
        
        print(f"\n🏆 MEJOR ESTRATEGIA: {comparison_results['best_strategy']['strategy']}")
        print(f"   Silhouette: {comparison_results['best_strategy']['silhouette_after']:.4f}")
        print(f"   Mejora: {comparison_results['best_strategy']['improvement_relative']:+.1%}")
        print(f"   Retención: {comparison_results['best_strategy']['retention_ratio']:.1%}")
        
        return comparison_results
    
    def _generate_comparative_analysis(self, strategy_results):
        """Generar análisis comparativo entre estrategias."""
        
        successful_strategies = {k: v for k, v in strategy_results.items() if 'error' not in v}
        
        if not successful_strategies:
            return {'error': 'No successful strategies to compare'}
        
        # Extraer métricas clave
        comparison_table = []
        
        for strategy, result in successful_strategies.items():
            silhouette_before = result['quality_before']['global_metrics']['silhouette_score']
            silhouette_after = result['quality_after']['global_metrics']['silhouette_score']
            improvement = result['improvements']['silhouette_relative']
            retention = result['purification_info'].get('retention_ratio', 1.0)
            
            comparison_table.append({
                'strategy': strategy,
                'silhouette_before': silhouette_before,
                'silhouette_after': silhouette_after,
                'improvement_absolute': silhouette_after - silhouette_before,
                'improvement_relative': improvement,
                'retention_ratio': retention,
                'execution_time': result['execution_time'],
                'quality_score': self._calculate_quality_score(improvement, retention)
            })
        
        # Convertir a DataFrame para análisis
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_table)
        
        # Ranking por diferentes criterios
        rankings = {
            'by_silhouette': comparison_df.nlargest(3, 'silhouette_after')['strategy'].tolist(),
            'by_improvement': comparison_df.nlargest(3, 'improvement_relative')['strategy'].tolist(),
            'by_retention': comparison_df.nlargest(3, 'retention_ratio')['strategy'].tolist(),
            'by_quality_score': comparison_df.nlargest(3, 'quality_score')['strategy'].tolist()
        }
        
        return {
            'comparison_table': comparison_table,
            'rankings': rankings,
            'summary_stats': {
                'best_silhouette': comparison_df['silhouette_after'].max(),
                'best_improvement': comparison_df['improvement_relative'].max(),
                'best_retention': comparison_df['retention_ratio'].max(),
                'average_improvement': comparison_df['improvement_relative'].mean(),
                'strategies_with_positive_improvement': (comparison_df['improvement_relative'] > 0).sum()
            }
        }
    
    def _calculate_quality_score(self, improvement, retention, improvement_weight=0.7):
        """Calcular score de calidad balanceado."""
        
        # Normalizar improvement (asumiendo rango -0.5 a +1.0)
        improvement_normalized = max(0, min(1, (improvement + 0.5) / 1.5))
        
        # Combinar con pesos
        quality_score = (improvement_normalized * improvement_weight + 
                        retention * (1 - improvement_weight))
        
        return quality_score
    
    def _select_best_strategy(self, strategy_results):
        """Seleccionar mejor estrategia basada en criterios múltiples."""
        
        successful_strategies = {k: v for k, v in strategy_results.items() if 'error' not in v}
        
        if not successful_strategies:
            return None
        
        best_strategy = None
        best_score = -1
        
        # Criterios de selección
        for strategy, result in successful_strategies.items():
            silhouette_after = result['quality_after']['global_metrics']['silhouette_score']
            improvement = result['improvements']['silhouette_relative']
            retention = result['purification_info'].get('retention_ratio', 1.0)
            
            # Aplicar criterios mínimos
            if silhouette_after < 0.15:  # Mínimo absoluto
                continue
            if improvement < 0.05:  # Mínimo 5% mejora
                continue
            if retention < 0.50:  # Mínimo 50% retención
                continue
            
            # Calcular score compuesto
            quality_score = self._calculate_quality_score(improvement, retention)
            
            if quality_score > best_score:
                best_score = quality_score
                best_strategy = {
                    'strategy': strategy,
                    'silhouette_before': result['quality_before']['global_metrics']['silhouette_score'],
                    'silhouette_after': silhouette_after,
                    'improvement_absolute': result['improvements']['silhouette_absolute'],
                    'improvement_relative': improvement,
                    'retention_ratio': retention,
                    'quality_score': quality_score,
                    'execution_time': result['execution_time']
                }
        
        return best_strategy
    
    def save_purification_results(self, results, filename_suffix=""):
        """Guardar resultados de purification."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar JSON
        json_filename = f"purification_results_{timestamp}{filename_suffix}.json"
        json_path = self.output_dir / json_filename
        
        try:
            # Convertir datos a formato JSON-serializable
            json_safe_results = self._convert_to_json_safe(results)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Resultados guardados: {json_path}")
            
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
        
        return json_path
    
    def _convert_to_json_safe(self, obj):
        """Convertir objeto a formato JSON-serializable."""
        
        # Importar numpy para verificaciones más robustas
        import numpy as np
        import pandas as pd
        from datetime import datetime
        
        if isinstance(obj, dict):
            return {str(k): self._convert_to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_safe(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):  # Captura todos los tipos enteros numpy
            return int(obj)
        elif isinstance(obj, np.floating):  # Captura todos los tipos float numpy
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif hasattr(obj, 'item'):  # Para escalares numpy que tienen método .item()
            return obj.item()
        elif str(type(obj)).startswith("<class 'numpy."):  # Fallback para cualquier tipo numpy
            try:
                return obj.item() if hasattr(obj, 'item') else str(obj)
            except:
                return str(obj)
        else:
            return obj

# Función principal para testing
def main():
    """Función principal para testing del sistema."""
    
    print("🚀 FASE 4.2: CLUSTER PURIFICATION SYSTEM - DATASET COMPLETO")
    print("="*70)
    
    # Inicializar purificador
    purifier = ClusterPurifier()
    
    # Cargar configuración baseline
    config = purifier.load_baseline_configuration()
    
    if config is None:
        print("❌ No se pudo cargar configuración baseline")
        return False
    
    # Preparar datos para clustering
    print(f"\n🔧 Preparando datos para clustering...")
    features_df = config['features_df']
    
    # Normalizar datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(features_df)
    
    # Ejecutar clustering baseline
    print(f"🎯 Ejecutando clustering baseline (Hierarchical, K=3)...")
    clusterer = AgglomerativeClustering(n_clusters=3)
    cluster_labels = clusterer.fit_predict(data_scaled)
    
    # Verificar silhouette baseline
    baseline_silhouette = silhouette_score(data_scaled, cluster_labels)
    print(f"✅ Silhouette baseline: {baseline_silhouette:.4f}")
    
    # DATASET COMPLETO - Sin sampling
    test_data = data_scaled  # DATASET COMPLETO (18,454 canciones)
    test_labels = cluster_labels
    
    print(f"\n🎯 Ejecutando purification strategies en DATASET COMPLETO ({len(test_data):,} points)...")
    print(f"⚠️  Tiempo estimado: 5-10 minutos")
    
    # Comparar estrategias - Solo hybrid por ser la mejor
    comparison_results = purifier.compare_purification_strategies(
        test_data, test_labels, 
        strategies=['hybrid']  # Solo la mejor estrategia
    )
    
    # Guardar resultados
    json_path = purifier.save_purification_results(comparison_results, "_full_dataset")
    
    print(f"\n🎉 SISTEMA PURIFICATION DATASET COMPLETO COMPLETADO")
    
    return True

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    exit(exit_code)