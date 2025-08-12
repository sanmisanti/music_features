#!/usr/bin/env python3
"""
FASE 2.1: CLUSTERING COMPARATIVO - Script Principal
==================================================

Este módulo implementa análisis comparativo robusto entre:
1. Dataset Optimizado (10K, Hopkins 0.933) 
2. Dataset Original (18K, Hopkins 0.787)
3. Dataset Control (10K, Hopkins ~0.45)

Objetivo: Validar mejora Silhouette Score 0.177 → 0.25+ (+41% mínimo)

Autor: Clustering Optimization Team
Fecha: 2025-01-12
Estado: FASE 2.1 - Setup Comparativo
"""

import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports para clustering y métricas
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Imports para análisis estadístico
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")

class ClusteringComparator:
    """
    Comparador avanzado para análisis clustering multi-dataset.
    
    Funcionalidades:
    - Clustering con múltiples algoritmos y valores K
    - Métricas robustas con análisis estadístico
    - Visualizaciones comparativas automáticas  
    - Reportes técnicos automatizados
    """
    
    def __init__(self, base_path=None):
        """Inicializar comparador con rutas de datasets."""
        
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent.parent
        else:
            base_path = Path(base_path)
            
        self.base_path = base_path
        
        # Configuración de datasets para comparación
        self.datasets_config = {
            'optimal': {
                'path': base_path / 'data' / 'final_data' / 'picked_data_optimal.csv',
                'separator': '^',
                'name': 'Dataset Optimizado (10K)',
                'expected_hopkins': 0.933,
                'description': 'MaxMin optimizado con KD-Tree y validación Hopkins'
            },
            'baseline': {
                'path': base_path / 'data' / 'with_lyrics' / 'spotify_songs_fixed.csv', 
                'separator': '@@',
                'name': 'Dataset Original (18K)',
                'expected_hopkins': 0.787,
                'description': 'Dataset completo sin optimización'
            },
            'control': {
                'path': base_path / 'data' / 'final_data' / 'picked_data_lyrics.csv',
                'separator': '^', 
                'name': 'Dataset Control (10K)',
                'expected_hopkins': 0.45,
                'description': 'Selección previa sin optimización Hopkins'
            }
        }
        
        # Configuración de características musicales  
        self.musical_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms'
        ]
        
        # Configuración de análisis
        self.analysis_config = {
            'k_range': [3, 4, 5, 6, 7, 8, 9, 10],
            'algorithms': ['kmeans', 'hierarchical'],  # DBSCAN puede ser problemático con estos datos
            'n_runs': 10,  # Múltiples runs para robustez estadística
            'random_states': list(range(42, 52)),  # Seeds fijos para reproducibilidad
            'test_size_limit': 5000  # Límite para tests rápidos
        }
        
        # Almacenamiento de resultados
        self.results = {
            'datasets': {},
            'comparisons': {},
            'statistical_tests': {},
            'visualizations': {}
        }
        
        # Configuración de salida
        self.output_dir = base_path / 'outputs' / 'fase2_clustering'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 ClusteringComparator inicializado")
        print(f"📁 Base path: {base_path}")
        print(f"📊 Output dir: {self.output_dir}")
        
    def load_dataset(self, dataset_key, sample_size=None, test_mode=False):
        """
        Cargar y preparar dataset para análisis clustering.
        
        Args:
            dataset_key: 'optimal', 'baseline', 'control'
            sample_size: Límite de filas (None = todos)
            test_mode: Si True, usa sample pequeño para tests rápidos
            
        Returns:
            tuple: (features_df, metadata_dict)
        """
        if dataset_key not in self.datasets_config:
            raise ValueError(f"Dataset key '{dataset_key}' no válido. Opciones: {list(self.datasets_config.keys())}")
        
        config = self.datasets_config[dataset_key]
        
        print(f"\n📂 Cargando {config['name']}...")
        print(f"   Archivo: {config['path']}")
        
        # Verificar archivo existe
        if not config['path'].exists():
            raise FileNotFoundError(f"Dataset no encontrado: {config['path']}")
        
        try:
            # Cargar dataset con configuración específica
            df = pd.read_csv(
                config['path'], 
                sep=config['separator'], 
                decimal='.', 
                encoding='utf-8'
            )
            
            print(f"✅ Dataset cargado: {len(df):,} filas × {len(df.columns)} columnas")
            
            # Filtrar solo características musicales disponibles
            available_features = [f for f in self.musical_features if f in df.columns]
            missing_features = [f for f in self.musical_features if f not in df.columns]
            
            print(f"🎵 Características disponibles: {len(available_features)}/{len(self.musical_features)}")
            if missing_features:
                print(f"⚠️  Características faltantes: {missing_features}")
            
            # Extraer características musicales
            features_df = df[available_features].copy()
            
            # Limpiar datos nulos
            original_size = len(features_df)
            features_df = features_df.dropna()
            cleaned_size = len(features_df)
            
            if original_size != cleaned_size:
                print(f"🧹 Limpieza: {original_size:,} → {cleaned_size:,} filas (-{original_size-cleaned_size} nulos)")
            
            # Aplicar sampling si se especifica
            if test_mode:
                sample_size = min(self.analysis_config['test_size_limit'], len(features_df))
                print(f"🧪 Modo test: limitando a {sample_size:,} filas")
            
            if sample_size and sample_size < len(features_df):
                features_df = features_df.sample(n=sample_size, random_state=42)
                print(f"📏 Sampling aplicado: {len(features_df):,} filas seleccionadas")
            
            # Preparar metadata
            metadata = {
                'dataset_key': dataset_key,
                'dataset_name': config['name'],
                'original_size': original_size,
                'cleaned_size': cleaned_size,
                'final_size': len(features_df),
                'n_features': len(available_features),
                'available_features': available_features,
                'missing_features': missing_features,
                'expected_hopkins': config['expected_hopkins'],
                'description': config['description'],
                'load_timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Dataset preparado: {len(features_df):,} canciones × {len(available_features)} características")
            
            return features_df, metadata
            
        except Exception as e:
            print(f"❌ ERROR cargando dataset {dataset_key}: {e}")
            raise
    
    def run_clustering_analysis(self, features_df, dataset_metadata, algorithm='kmeans', k_range=None, n_runs=None):
        """
        Ejecutar análisis clustering robusto con múltiples runs.
        
        Args:
            features_df: DataFrame con características musicales
            dataset_metadata: Metadata del dataset
            algorithm: 'kmeans' o 'hierarchical'
            k_range: Lista de valores K (None = usar config)
            n_runs: Número de runs (None = usar config)
            
        Returns:
            dict: Resultados completos del análisis
        """
        if k_range is None:
            k_range = self.analysis_config['k_range']
        if n_runs is None:
            n_runs = self.analysis_config['n_runs']
            
        dataset_name = dataset_metadata['dataset_name']
        print(f"\n🎯 Ejecutando análisis clustering: {dataset_name}")
        print(f"   Algoritmo: {algorithm}")
        print(f"   K range: {k_range}")
        print(f"   Runs: {n_runs}")
        
        # Normalizar características  
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        print(f"📊 Datos normalizados: {features_scaled.shape}")
        
        # Almacenar resultados por K
        results_by_k = {}
        
        for k in k_range:
            print(f"\n   🔍 Analizando K={k}...")
            
            k_results = {
                'k': k,
                'runs': [],
                'metrics_summary': {},
                'best_run': None,
                'worst_run': None
            }
            
            # Múltiples runs para robustez estadística
            silhouette_scores = []
            calinski_scores = []
            davies_bouldin_scores = []
            inertias = []
            
            for run_idx in range(n_runs):
                random_state = self.analysis_config['random_states'][run_idx]
                
                try:
                    # Clustering según algoritmo
                    if algorithm == 'kmeans':
                        clusterer = KMeans(
                            n_clusters=k, 
                            random_state=random_state,
                            n_init=10,
                            max_iter=300
                        )
                    elif algorithm == 'hierarchical':
                        clusterer = AgglomerativeClustering(n_clusters=k)
                    else:
                        raise ValueError(f"Algoritmo '{algorithm}' no soportado")
                    
                    # Fit clustering
                    cluster_labels = clusterer.fit_predict(features_scaled)
                    
                    # Calcular métricas
                    silhouette = silhouette_score(features_scaled, cluster_labels)
                    calinski = calinski_harabasz_score(features_scaled, cluster_labels)
                    davies_bouldin = davies_bouldin_score(features_scaled, cluster_labels)
                    
                    # Inertia solo para K-Means
                    inertia = clusterer.inertia_ if hasattr(clusterer, 'inertia_') else None
                    
                    # Almacenar métricas del run
                    run_metrics = {
                        'run_idx': run_idx,
                        'random_state': random_state,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski,
                        'davies_bouldin_score': davies_bouldin,
                        'inertia': inertia,
                        'n_clusters_found': len(np.unique(cluster_labels)),
                        'cluster_sizes': np.bincount(cluster_labels).tolist(),
                        'algorithm': algorithm
                    }
                    
                    k_results['runs'].append(run_metrics)
                    
                    # Agregar a listas para estadísticas
                    silhouette_scores.append(silhouette)
                    calinski_scores.append(calinski)
                    davies_bouldin_scores.append(davies_bouldin)
                    if inertia is not None:
                        inertias.append(inertia)
                        
                except Exception as e:
                    print(f"      ⚠️  Error en run {run_idx}: {e}")
                    continue
            
            # Calcular estadísticas resumidas
            if silhouette_scores:
                k_results['metrics_summary'] = {
                    'silhouette': {
                        'mean': np.mean(silhouette_scores),
                        'std': np.std(silhouette_scores),
                        'min': np.min(silhouette_scores),
                        'max': np.max(silhouette_scores),
                        'median': np.median(silhouette_scores)
                    },
                    'calinski_harabasz': {
                        'mean': np.mean(calinski_scores),
                        'std': np.std(calinski_scores),
                        'min': np.min(calinski_scores), 
                        'max': np.max(calinski_scores),
                        'median': np.median(calinski_scores)
                    },
                    'davies_bouldin': {
                        'mean': np.mean(davies_bouldin_scores),
                        'std': np.std(davies_bouldin_scores),
                        'min': np.min(davies_bouldin_scores),
                        'max': np.max(davies_bouldin_scores),
                        'median': np.median(davies_bouldin_scores)
                    }
                }
                
                if inertias:
                    k_results['metrics_summary']['inertia'] = {
                        'mean': np.mean(inertias),
                        'std': np.std(inertias),
                        'min': np.min(inertias),
                        'max': np.max(inertias), 
                        'median': np.median(inertias)
                    }
                
                # Identificar mejor y peor run por Silhouette Score
                best_idx = np.argmax(silhouette_scores)
                worst_idx = np.argmin(silhouette_scores)
                
                k_results['best_run'] = k_results['runs'][best_idx]
                k_results['worst_run'] = k_results['runs'][worst_idx]
                
                silhouette_mean = k_results['metrics_summary']['silhouette']['mean']
                silhouette_std = k_results['metrics_summary']['silhouette']['std']
                
                print(f"      ✅ K={k}: Silhouette {silhouette_mean:.4f} ± {silhouette_std:.4f}")
            else:
                print(f"      ❌ K={k}: Falló todos los runs")
                
            results_by_k[k] = k_results
        
        # Preparar resultado final
        analysis_result = {
            'dataset_metadata': dataset_metadata,
            'algorithm': algorithm,
            'analysis_config': {
                'k_range': k_range,
                'n_runs': n_runs,
                'n_features': features_scaled.shape[1],
                'n_samples': features_scaled.shape[0]
            },
            'results_by_k': results_by_k,
            'analysis_timestamp': datetime.now().isoformat(),
            'best_k': self._find_best_k(results_by_k),
            'summary_statistics': self._calculate_summary_statistics(results_by_k)
        }
        
        print(f"✅ Análisis clustering completado: {len(results_by_k)} valores K analizados")
        
        return analysis_result
    
    def _find_best_k(self, results_by_k):
        """Encontrar mejor K basado en Silhouette Score promedio."""
        
        best_k = None
        best_silhouette = -1
        
        for k, k_results in results_by_k.items():
            if 'metrics_summary' in k_results and 'silhouette' in k_results['metrics_summary']:
                silhouette_mean = k_results['metrics_summary']['silhouette']['mean']
                
                if silhouette_mean > best_silhouette:
                    best_silhouette = silhouette_mean
                    best_k = k
        
        return {
            'k': best_k,
            'silhouette_score': best_silhouette
        } if best_k is not None else None
    
    def _calculate_summary_statistics(self, results_by_k):
        """Calcular estadísticas generales del análisis."""
        
        all_silhouette = []
        all_calinski = []
        all_davies_bouldin = []
        
        for k_results in results_by_k.values():
            if 'runs' in k_results:
                for run in k_results['runs']:
                    all_silhouette.append(run['silhouette_score'])
                    all_calinski.append(run['calinski_harabasz_score'])
                    all_davies_bouldin.append(run['davies_bouldin_score'])
        
        if all_silhouette:
            return {
                'total_runs': len(all_silhouette),
                'overall_silhouette': {
                    'mean': np.mean(all_silhouette),
                    'std': np.std(all_silhouette),
                    'min': np.min(all_silhouette),
                    'max': np.max(all_silhouette)
                },
                'overall_calinski': {
                    'mean': np.mean(all_calinski),
                    'std': np.std(all_calinski), 
                    'min': np.min(all_calinski),
                    'max': np.max(all_calinski)
                },
                'overall_davies_bouldin': {
                    'mean': np.mean(all_davies_bouldin),
                    'std': np.std(all_davies_bouldin),
                    'min': np.min(all_davies_bouldin),
                    'max': np.max(all_davies_bouldin)
                }
            }
        else:
            return None
    
    def compare_datasets(self, dataset_keys=None, algorithm='kmeans', test_mode=False):
        """
        Ejecutar comparación completa entre datasets.
        
        Args:
            dataset_keys: Lista de keys ('optimal', 'baseline', 'control')
            algorithm: Algoritmo clustering
            test_mode: Si True, usa samples pequeños para tests rápidos
            
        Returns:
            dict: Resultados comparativos completos
        """
        if dataset_keys is None:
            dataset_keys = ['optimal', 'control', 'baseline']  # Orden de prioridad
        
        print(f"\n🚀 INICIANDO COMPARACIÓN CLUSTERING")
        print(f"📊 Datasets: {dataset_keys}")
        print(f"🎯 Algoritmo: {algorithm}")
        print(f"🧪 Modo test: {test_mode}")
        
        comparison_results = {
            'comparison_config': {
                'dataset_keys': dataset_keys,
                'algorithm': algorithm,
                'test_mode': test_mode,
                'timestamp': datetime.now().isoformat()
            },
            'dataset_analyses': {},
            'comparative_metrics': {},
            'statistical_tests': {},
            'recommendations': {}
        }
        
        # Analizar cada dataset
        for dataset_key in dataset_keys:
            print(f"\n" + "="*60)
            print(f"📊 ANALIZANDO DATASET: {dataset_key.upper()}")
            print(f"="*60)
            
            try:
                # Cargar dataset
                features_df, metadata = self.load_dataset(dataset_key, test_mode=test_mode)
                
                # Ejecutar análisis clustering
                analysis_result = self.run_clustering_analysis(
                    features_df, metadata, algorithm=algorithm
                )
                
                comparison_results['dataset_analyses'][dataset_key] = analysis_result
                
                # Mostrar resumen
                best_k = analysis_result.get('best_k')
                if best_k:
                    print(f"✅ {metadata['dataset_name']}: Mejor K={best_k['k']}, Silhouette={best_k['silhouette_score']:.4f}")
                else:
                    print(f"⚠️  {metadata['dataset_name']}: No se pudo determinar mejor K")
                    
            except Exception as e:
                print(f"❌ ERROR analizando {dataset_key}: {e}")
                comparison_results['dataset_analyses'][dataset_key] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Generar métricas comparativas
        comparison_results['comparative_metrics'] = self._generate_comparative_metrics(
            comparison_results['dataset_analyses']
        )
        
        # Ejecutar tests estadísticos
        comparison_results['statistical_tests'] = self._run_statistical_tests(
            comparison_results['dataset_analyses']
        )
        
        # Generar recomendaciones
        comparison_results['recommendations'] = self._generate_recommendations(
            comparison_results
        )
        
        print(f"\n🎉 COMPARACIÓN COMPLETADA")
        return comparison_results
    
    def _generate_comparative_metrics(self, dataset_analyses):
        """Generar métricas comparativas entre datasets."""
        
        print(f"\n📊 Generando métricas comparativas...")
        
        comparative_metrics = {
            'silhouette_comparison': {},
            'best_k_comparison': {},
            'improvement_analysis': {}
        }
        
        # Extraer mejores métricas por dataset
        dataset_metrics = {}
        
        for dataset_key, analysis in dataset_analyses.items():
            if 'error' not in analysis and analysis.get('best_k'):
                best_k_info = analysis['best_k']
                dataset_metrics[dataset_key] = {
                    'name': analysis['dataset_metadata']['dataset_name'],
                    'best_k': best_k_info['k'],
                    'best_silhouette': best_k_info['silhouette_score'],
                    'expected_hopkins': analysis['dataset_metadata']['expected_hopkins'],
                    'sample_size': analysis['dataset_metadata']['final_size']
                }
        
        comparative_metrics['dataset_metrics'] = dataset_metrics
        
        # Calcular mejoras relativas
        if 'optimal' in dataset_metrics and 'control' in dataset_metrics:
            optimal_silhouette = dataset_metrics['optimal']['best_silhouette']
            control_silhouette = dataset_metrics['control']['best_silhouette']
            
            improvement = (optimal_silhouette - control_silhouette) / control_silhouette
            
            comparative_metrics['improvement_analysis']['optimal_vs_control'] = {
                'optimal_silhouette': optimal_silhouette,
                'control_silhouette': control_silhouette,
                'absolute_improvement': optimal_silhouette - control_silhouette,
                'relative_improvement': improvement,
                'improvement_percentage': improvement * 100
            }
            
            print(f"📈 Mejora Optimal vs Control: {improvement*100:+.1f}% ({optimal_silhouette:.4f} vs {control_silhouette:.4f})")
        
        return comparative_metrics
    
    def _run_statistical_tests(self, dataset_analyses):
        """Ejecutar tests estadísticos de significancia."""
        
        print(f"\n🔬 Ejecutando tests estadísticos...")
        
        statistical_tests = {}
        
        # Extraer todas las métricas Silhouette por dataset
        silhouette_data = {}
        
        for dataset_key, analysis in dataset_analyses.items():
            if 'error' not in analysis and 'results_by_k' in analysis:
                all_silhouette = []
                
                for k_results in analysis['results_by_k'].values():
                    if 'runs' in k_results:
                        for run in k_results['runs']:
                            all_silhouette.append(run['silhouette_score'])
                
                if all_silhouette:
                    silhouette_data[dataset_key] = all_silhouette
        
        # Tests de comparación por pares
        for dataset1 in silhouette_data:
            for dataset2 in silhouette_data:
                if dataset1 < dataset2:  # Evitar duplicados
                    
                    data1 = silhouette_data[dataset1]
                    data2 = silhouette_data[dataset2]
                    
                    # T-test para diferencias de medias
                    t_stat, t_pvalue = ttest_ind(data1, data2)
                    
                    # Wilcoxon para diferencias no-paramétricas
                    try:
                        w_stat, w_pvalue = wilcoxon(data1[:min(len(data1), len(data2))], 
                                                  data2[:min(len(data1), len(data2))])
                    except:
                        w_stat, w_pvalue = None, None
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(data1)-1)*np.var(data1) + (len(data2)-1)*np.var(data2)) / 
                                       (len(data1) + len(data2) - 2))
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                    
                    test_key = f"{dataset1}_vs_{dataset2}"
                    statistical_tests[test_key] = {
                        'datasets': [dataset1, dataset2],
                        'sample_sizes': [len(data1), len(data2)],
                        'means': [np.mean(data1), np.mean(data2)],
                        'stds': [np.std(data1), np.std(data2)],
                        't_test': {
                            'statistic': t_stat,
                            'p_value': t_pvalue,
                            'significant': t_pvalue < 0.05 if t_pvalue is not None else False
                        },
                        'wilcoxon_test': {
                            'statistic': w_stat,
                            'p_value': w_pvalue,
                            'significant': w_pvalue < 0.05 if w_pvalue is not None else False
                        },
                        'effect_size': {
                            'cohens_d': cohens_d,
                            'interpretation': self._interpret_effect_size(cohens_d)
                        }
                    }
                    
                    if t_pvalue is not None:
                        significance = "***" if t_pvalue < 0.001 else "**" if t_pvalue < 0.01 else "*" if t_pvalue < 0.05 else "ns"
                        print(f"📊 {dataset1} vs {dataset2}: p={t_pvalue:.4f} {significance}, d={cohens_d:.3f}")
        
        return statistical_tests
    
    def _interpret_effect_size(self, cohens_d):
        """Interpretar magnitud del effect size."""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "small"
        elif abs_d < 0.5:
            return "medium" 
        elif abs_d < 0.8:
            return "large"
        else:
            return "very_large"
    
    def _generate_recommendations(self, comparison_results):
        """Generar recomendaciones basadas en los resultados."""
        
        recommendations = {
            'best_dataset': None,
            'best_k': None,
            'silhouette_target_met': False,
            'statistical_significance': False,
            'next_steps': []
        }
        
        # Determinar mejor dataset
        best_silhouette = -1
        best_dataset = None
        
        if 'comparative_metrics' in comparison_results:
            dataset_metrics = comparison_results['comparative_metrics'].get('dataset_metrics', {})
            
            for dataset_key, metrics in dataset_metrics.items():
                if metrics['best_silhouette'] > best_silhouette:
                    best_silhouette = metrics['best_silhouette']
                    best_dataset = dataset_key
            
            if best_dataset:
                recommendations['best_dataset'] = {
                    'dataset': best_dataset,
                    'silhouette_score': best_silhouette,
                    'k': dataset_metrics[best_dataset]['best_k']
                }
        
        # Verificar si se cumplió objetivo Silhouette > 0.25
        recommendations['silhouette_target_met'] = best_silhouette > 0.25
        
        # Verificar significancia estadística
        if 'statistical_tests' in comparison_results:
            significant_tests = [test for test in comparison_results['statistical_tests'].values() 
                               if test['t_test']['significant']]
            recommendations['statistical_significance'] = len(significant_tests) > 0
        
        # Generar próximos pasos
        if recommendations['silhouette_target_met']:
            recommendations['next_steps'].append("✅ Objetivo Silhouette >0.25 ALCANZADO - Proceder con FASE 3")
            recommendations['next_steps'].append(f"🎯 Usar dataset '{best_dataset}' para clustering readiness assessment")
        else:
            recommendations['next_steps'].append("⚠️ Objetivo Silhouette >0.25 NO alcanzado - Revisar optimización")
            recommendations['next_steps'].append("🔄 Considerar FASE 4 (Cluster Purification) para mejora adicional")
        
        if recommendations['statistical_significance']:
            recommendations['next_steps'].append("📊 Diferencias estadísticamente significativas confirmadas")
        else:
            recommendations['next_steps'].append("📊 Considerar aumentar sample size para mayor poder estadístico")
        
        return recommendations
    
    def save_results(self, comparison_results, filename_suffix=""):
        """Guardar resultados en archivos JSON y generar reporte."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar resultados JSON
        json_filename = f"clustering_comparison_{timestamp}{filename_suffix}.json"
        json_path = self.output_dir / json_filename
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Resultados guardados: {json_path}")
            
        except Exception as e:
            print(f"❌ Error guardando resultados JSON: {e}")
        
        # Generar reporte markdown
        report_filename = f"clustering_comparison_report_{timestamp}{filename_suffix}.md"
        report_path = self.output_dir / report_filename
        
        try:
            self._generate_markdown_report(comparison_results, report_path)
            print(f"📋 Reporte generado: {report_path}")
            
        except Exception as e:
            print(f"❌ Error generando reporte: {e}")
        
        return json_path, report_path
    
    def _generate_markdown_report(self, comparison_results, report_path):
        """Generar reporte técnico en formato Markdown."""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 CLUSTERING COMPARISON REPORT - FASE 2\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Phase**: FASE 2.1 - Clustering Comparativo\n")
            f.write(f"**Objective**: Validate Silhouette Score improvement 0.177 → 0.25+\n\n")
            
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## 🎯 EXECUTIVE SUMMARY\n\n")
            
            recommendations = comparison_results.get('recommendations', {})
            if recommendations.get('best_dataset'):
                best_info = recommendations['best_dataset']
                f.write(f"**Best Dataset**: {best_info['dataset']}\n")
                f.write(f"**Best Silhouette Score**: {best_info['silhouette_score']:.4f}\n")
                f.write(f"**Optimal K**: {best_info['k']}\n")
                
                target_met = "✅ YES" if recommendations.get('silhouette_target_met') else "❌ NO"
                f.write(f"**Target >0.25 Met**: {target_met}\n")
                
                significance = "✅ YES" if recommendations.get('statistical_significance') else "❌ NO"
                f.write(f"**Statistical Significance**: {significance}\n\n")
            
            # Dataset Metrics
            f.write("## 📊 DATASET METRICS\n\n")
            
            if 'comparative_metrics' in comparison_results:
                dataset_metrics = comparison_results['comparative_metrics'].get('dataset_metrics', {})
                
                f.write("| Dataset | Sample Size | Best K | Silhouette Score | Expected Hopkins |\n")
                f.write("|---------|-------------|--------|------------------|------------------|\n")
                
                for dataset_key, metrics in dataset_metrics.items():
                    f.write(f"| {metrics['name']} | {metrics['sample_size']:,} | {metrics['best_k']} | "
                           f"{metrics['best_silhouette']:.4f} | {metrics['expected_hopkins']:.3f} |\n")
                
                f.write("\n")
            
            # Statistical Tests
            f.write("## 🔬 STATISTICAL TESTS\n\n")
            
            if 'statistical_tests' in comparison_results:
                for test_key, test_result in comparison_results['statistical_tests'].items():
                    datasets = test_result['datasets']
                    f.write(f"### {datasets[0].upper()} vs {datasets[1].upper()}\n\n")
                    
                    f.write(f"- **Sample Sizes**: {test_result['sample_sizes'][0]} vs {test_result['sample_sizes'][1]}\n")
                    f.write(f"- **Means**: {test_result['means'][0]:.4f} vs {test_result['means'][1]:.4f}\n")
                    f.write(f"- **T-test p-value**: {test_result['t_test']['p_value']:.6f}\n")
                    f.write(f"- **Significant**: {'✅ YES' if test_result['t_test']['significant'] else '❌ NO'}\n")
                    f.write(f"- **Effect Size (Cohen's d)**: {test_result['effect_size']['cohens_d']:.3f} ({test_result['effect_size']['interpretation']})\n\n")
            
            # Recommendations
            f.write("## 🎯 RECOMMENDATIONS\n\n")
            
            if 'next_steps' in recommendations:
                for step in recommendations['next_steps']:
                    f.write(f"- {step}\n")
                f.write("\n")
            
            # Technical Details
            f.write("## 🔧 TECHNICAL DETAILS\n\n")
            
            config = comparison_results.get('comparison_config', {})
            f.write(f"- **Algorithm**: {config.get('algorithm', 'N/A')}\n")
            f.write(f"- **Test Mode**: {config.get('test_mode', False)}\n")
            f.write(f"- **Datasets Analyzed**: {len(comparison_results.get('dataset_analyses', {}))}\n")
            f.write(f"- **Timestamp**: {config.get('timestamp', 'N/A')}\n")

# Función principal para ejecución standalone
def main():
    """Función principal para ejecutar comparación clustering."""
    
    print("🚀 FASE 2.1: CLUSTERING COMPARATIVO - EXECUTION")
    print("="*60)
    
    # Inicializar comparador
    comparator = ClusteringComparator()
    
    # Ejecutar comparación (modo test para desarrollo rápido)
    results = comparator.compare_datasets(
        dataset_keys=['optimal', 'control'],  # Comparar principales primero
        algorithm='kmeans',
        test_mode=True  # Cambiar a False para análisis completo
    )
    
    # Guardar resultados
    json_path, report_path = comparator.save_results(results, "_test")
    
    print(f"\n🎉 FASE 2.1 COMPLETADA")
    print(f"📁 Resultados en: {comparator.output_dir}")
    
    return results

if __name__ == "__main__":
    results = main()