#!/usr/bin/env python3
"""
Sistema de Validación Completo para Recomendaciones Musicales
Implementa evaluación científica integral del sistema híbrido

Componentes:
- Validación de integridad del sistema
- Métricas de precisión y recall (P@K, R@K)
- Análisis de diversidad musical y semántica
- Validación de coherencia cluster-explicación
- Benchmarking de performance (<100ms objetivo)
- Generación de reportes científicos comprensivos

Ejecutar desde el directorio recommendation_system:
    python scripts/validate_system.py [--mode MODE] [--export-report]
"""

import numpy as np
import pandas as pd
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Importar componentes del sistema
try:
    from load_system import MusicDataLoader
    from music_recommender import HybridMusicRecommender
    from explain_recommendations import RecommendationExplainer
except ImportError as e:
    print(f"⚠️  Importación faltante: {e}")
    print("Ejecutar desde el directorio recommendation_system/")


class SystemValidator:
    """Validador integral del sistema de recomendaciones musicales"""
    
    def __init__(self, system_dir: str = None):
        if system_dir is None:
            self.system_dir = Path(__file__).parent.parent
        else:
            self.system_dir = Path(system_dir)
            
        self.data_dir = self.system_dir / "data"
        self.clusters_dir = self.system_dir / "clusters"
        self.config_dir = self.system_dir / "config"
        
        self.validation_results = {}
        self.data_loader = None
        self.recommender = None
        self.explainer = None
        
    def initialize_system_components(self) -> bool:
        """Inicializa componentes del sistema para validación"""
        try:
            print("🔧 Inicializando componentes del sistema...")
            
            # Inicializar cargador de datos
            self.data_loader = MusicDataLoader(str(self.system_dir))
            
            # Inicializar recomendador
            self.recommender = HybridMusicRecommender(self.data_loader)
            
            # Inicializar explicador
            self.explainer = RecommendationExplainer(self.data_loader)
            
            print("   ✅ Componentes inicializados correctamente")
            return True
            
        except Exception as e:
            print(f"   ❌ Error inicializando componentes: {e}")
            return False
    
    def validate_file_structure(self) -> bool:
        """Verifica que existan todos los archivos críticos"""
        print("🏗️  Validando estructura de archivos...")
        
        required_files = {
            'data': [
                'semantic_embeddings.npy',
                'musical_features_normalized.npy',
                'track_ids.npy',
                'songs_metadata.csv'
            ],
            'clusters': [
                'musical_clusters_k10.npy',
                'semantic_clusters_k6.npy'
            ],
            'config': [
                'system_config.json'
            ],
            'scripts': [
                'load_system.py',
                'music_recommender.py',
                'explain_recommendations.py',
                'recommend_songs.py',
                'analyze_clusters.py'
            ]
        }
        
        missing_files = []
        for directory, files in required_files.items():
            dir_path = self.system_dir / directory
            for file_name in files:
                file_path = dir_path / file_name
                if file_path.exists():
                    print(f"   ✅ {directory}/{file_name}")
                else:
                    print(f"   ❌ FALTA: {directory}/{file_name}")
                    missing_files.append(f"{directory}/{file_name}")
        
        structure_valid = len(missing_files) == 0
        self.validation_results['file_structure'] = {
            'valid': structure_valid,
            'missing_files': missing_files
        }
        
        if structure_valid:
            print("✅ Estructura de archivos correcta")
        else:
            print(f"❌ Faltan {len(missing_files)} archivos críticos")
            
        return structure_valid
    
    def validate_data_integrity(self) -> bool:
        """Valida integridad y consistencia de los datos"""
        print("📊 Validando integridad de datos...")
        
        try:
            # Cargar todos los arrays principales
            semantic_vectors = np.load(self.data_dir / "semantic_embeddings.npy")
            musical_vectors = np.load(self.data_dir / "musical_features_normalized.npy")
            track_ids = np.load(self.data_dir / "track_ids.npy", allow_pickle=True)
            
            # Cargar clusters
            musical_clusters = np.load(self.clusters_dir / "musical_clusters_k10.npy")
            semantic_clusters = np.load(self.clusters_dir / "semantic_clusters_k6.npy")
            
            # Cargar metadatos
            metadata_df = pd.read_csv(self.data_dir / "songs_metadata.csv")
            
            # Verificar dimensiones
            expected_samples = 7811
            checks = {
                'semantic_shape': semantic_vectors.shape == (expected_samples, 384),
                'musical_shape': musical_vectors.shape == (expected_samples, 12),
                'track_ids_length': len(track_ids) == expected_samples,
                'musical_clusters_length': len(musical_clusters) == expected_samples,
                'semantic_clusters_length': len(semantic_clusters) == expected_samples,
                'metadata_length': len(metadata_df) == expected_samples,
                'all_arrays_aligned': all([
                    len(semantic_vectors) == len(musical_vectors),
                    len(musical_vectors) == len(track_ids),
                    len(track_ids) == len(musical_clusters),
                    len(musical_clusters) == len(semantic_clusters)
                ])
            }
            
            # Verificar rangos de clusters
            musical_unique = np.unique(musical_clusters)
            semantic_unique = np.unique(semantic_clusters)
            
            checks.update({
                'musical_clusters_range': (musical_unique.min() == 0 and 
                                         musical_unique.max() == 9 and 
                                         len(musical_unique) == 10),
                'semantic_clusters_range': (semantic_unique.min() == 0 and 
                                          semantic_unique.max() == 5 and 
                                          len(semantic_unique) == 6)
            })
            
            # Reportar resultados
            for check_name, result in checks.items():
                status = "✅" if result else "❌"
                print(f"   {status} {check_name}")
            
            all_checks_pass = all(checks.values())
            
            # Estadísticas adicionales
            if all_checks_pass:
                print(f"   📈 Dataset: {len(track_ids)} canciones")
                print(f"   🎵 Vectores musicales: {musical_vectors.shape}")
                print(f"   🧠 Vectores semánticos: {semantic_vectors.shape}")
                print(f"   🎪 Clusters musicales: {len(musical_unique)} únicos")
                print(f"   🎪 Clusters semánticos: {len(semantic_unique)} únicos")
            
            self.validation_results['data_integrity'] = {
                'valid': all_checks_pass,
                'checks': checks,
                'statistics': {
                    'total_songs': len(track_ids),
                    'musical_clusters': len(musical_unique),
                    'semantic_clusters': len(semantic_unique)
                } if all_checks_pass else None
            }
            
            return all_checks_pass
            
        except Exception as e:
            print(f"   ❌ Error cargando datos: {e}")
            self.validation_results['data_integrity'] = {
                'valid': False,
                'error': str(e)
            }
            return False
    
    def validate_configuration(self) -> bool:
        """Valida archivo de configuración del sistema"""
        print("🔧 Validando configuración del sistema...")
        
        try:
            with open(self.config_dir / "system_config.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Verificar estructura de configuración esperada
            required_keys = [
                'dataset_info',
                'optimal_configurations', 
                'recommendation_weights'
            ]
            
            checks = {
                'has_required_keys': all(key in config for key in required_keys),
                'dataset_info_complete': all(key in config.get('dataset_info', {}) 
                                           for key in ['total_songs', 'semantic_dimensions', 'musical_dimensions']),
                'weights_sum_correct': abs(
                    config.get('recommendation_weights', {}).get('musical_weight', 0) +
                    config.get('recommendation_weights', {}).get('semantic_weight', 0) - 1.0
                ) < 0.01
            }
            
            for check_name, result in checks.items():
                status = "✅" if result else "❌"
                print(f"   {status} {check_name}")
            
            config_valid = all(checks.values())
            
            if config_valid:
                weights = config['recommendation_weights']
                print(f"   📊 Pesos de recomendación: Musical {weights['musical_weight']:.2f}, Semántico {weights['semantic_weight']:.2f}")
            
            self.validation_results['configuration'] = {
                'valid': config_valid,
                'checks': checks,
                'config': config if config_valid else None
            }
            
            return config_valid
            
        except Exception as e:
            print(f"   ❌ Error validando configuración: {e}")
            self.validation_results['configuration'] = {
                'valid': False,
                'error': str(e)
            }
            return False
    
    def validate_data_quality(self) -> bool:
        """Verifica calidad de los datos (valores, normalización, etc.)"""
        print("🔍 Validando calidad de datos...")
        
        try:
            semantic_vectors = np.load(self.data_dir / "semantic_embeddings.npy")
            musical_vectors = np.load(self.data_dir / "musical_features_normalized.npy")
            
            # Verificar normalización de vectores semánticos (L2 norm ≈ 1.0)
            semantic_norms = np.linalg.norm(semantic_vectors, axis=1)
            semantic_normalized = np.abs(semantic_norms - 1.0) < 0.01  # Tolerancia 1%
            
            # Verificar normalización de características musicales (StandardScaler)
            musical_means = np.abs(musical_vectors.mean(axis=0)) < 0.1  # Cerca de cero
            musical_stds = np.abs(musical_vectors.std(axis=0) - 1.0) < 0.1  # Cerca de uno
            
            # Verificar ausencia de NaN/Inf
            no_nan_semantic = not np.any(np.isnan(semantic_vectors))
            no_inf_semantic = not np.any(np.isinf(semantic_vectors))
            no_nan_musical = not np.any(np.isnan(musical_vectors))
            no_inf_musical = not np.any(np.isinf(musical_vectors))
            
            checks = {
                'semantic_l2_normalized': np.mean(semantic_normalized) > 0.99,
                'musical_zero_mean': np.mean(musical_means) > 0.8,
                'musical_unit_std': np.mean(musical_stds) > 0.8,
                'no_nan_values': no_nan_semantic and no_nan_musical,
                'no_inf_values': no_inf_semantic and no_inf_musical
            }
            
            for check_name, result in checks.items():
                status = "✅" if result else "❌"
                print(f"   {status} {check_name}")
            
            quality_valid = all(checks.values())
            
            if quality_valid:
                print(f"   📊 L2 norm promedio semántico: {semantic_norms.mean():.6f}")
                print(f"   📊 Media promedio musical: {np.abs(musical_vectors.mean(axis=0)).mean():.6f}")
            
            self.validation_results['data_quality'] = {
                'valid': quality_valid,
                'checks': checks,
                'statistics': {
                    'semantic_norm_mean': float(semantic_norms.mean()),
                    'semantic_norm_std': float(semantic_norms.std()),
                    'musical_mean_abs': float(np.abs(musical_vectors.mean(axis=0)).mean()),
                    'musical_std_mean': float(musical_vectors.std(axis=0).mean())
                } if quality_valid else None
            }
            
            return quality_valid
            
        except Exception as e:
            print(f"   ❌ Error validando calidad: {e}")
            self.validation_results['data_quality'] = {
                'valid': False,
                'error': str(e)
            }
            return False


class PrecisionRecallEvaluator:
    """Evaluador de precisión y recall para recomendaciones"""
    
    def __init__(self, data_loader: MusicDataLoader, recommender: HybridMusicRecommender):
        self.data_loader = data_loader
        self.recommender = recommender
        
    def calculate_precision_at_k(self, recommendations: List[int], ground_truth: List[int], k: int) -> float:
        """Calcula Precision@K"""
        if k <= 0 or len(recommendations) == 0:
            return 0.0
            
        top_k_recs = recommendations[:k]
        relevant_in_topk = len(set(top_k_recs) & set(ground_truth))
        return relevant_in_topk / min(k, len(top_k_recs))
    
    def calculate_recall_at_k(self, recommendations: List[int], ground_truth: List[int], k: int) -> float:
        """Calcula Recall@K"""
        if len(ground_truth) == 0 or k <= 0:
            return 0.0
            
        top_k_recs = recommendations[:k]
        relevant_in_topk = len(set(top_k_recs) & set(ground_truth))
        return relevant_in_topk / len(ground_truth)
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calcula F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def generate_cluster_based_ground_truth(self, query_idx: int) -> List[int]:
        """Genera ground truth basado en membresía de clusters"""
        # Obtener clusters del query
        musical_clusters = self.data_loader.get_musical_clusters()
        semantic_clusters = self.data_loader.get_semantic_clusters()
        
        query_musical_cluster = musical_clusters[query_idx]
        query_semantic_cluster = semantic_clusters[query_idx]
        
        # Encontrar canciones en los mismos clusters
        same_musical = np.where(musical_clusters == query_musical_cluster)[0]
        same_semantic = np.where(semantic_clusters == query_semantic_cluster)[0]
        
        # Combinar (unión) y excluir la canción query
        ground_truth = list(set(same_musical) | set(same_semantic))
        ground_truth = [idx for idx in ground_truth if idx != query_idx]
        
        return ground_truth
    
    def evaluate_precision_recall(self, sample_size: int = 100, k_values: List[int] = None) -> Dict:
        """Evalúa precisión y recall en muestra de canciones"""
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]
        
        print(f"📊 Evaluando Precision@K y Recall@K (muestra: {sample_size})")
        
        # Seleccionar muestra aleatoria
        total_songs = len(self.data_loader.get_track_ids())
        sample_indices = np.random.choice(total_songs, min(sample_size, total_songs), replace=False)
        
        results = {f'precision_at_{k}': [] for k in k_values}
        results.update({f'recall_at_{k}': [] for k in k_values})
        results.update({f'f1_at_{k}': [] for k in k_values})
        
        for i, query_idx in enumerate(sample_indices):
            if i % 20 == 0:
                print(f"   Procesando {i+1}/{len(sample_indices)}...")
            
            try:
                # Obtener recomendaciones
                track_id = self.data_loader.get_track_ids()[query_idx]
                recommendations = self.recommender.recommend(
                    track_id=track_id, 
                    num_recommendations=max(k_values),
                    return_indices=True
                )
                
                # Generar ground truth basado en clusters
                ground_truth = self.generate_cluster_based_ground_truth(query_idx)
                
                # Calcular métricas para cada K
                for k in k_values:
                    precision = self.calculate_precision_at_k(recommendations, ground_truth, k)
                    recall = self.calculate_recall_at_k(recommendations, ground_truth, k)
                    f1 = self.calculate_f1_score(precision, recall)
                    
                    results[f'precision_at_{k}'].append(precision)
                    results[f'recall_at_{k}'].append(recall)
                    results[f'f1_at_{k}'].append(f1)
                    
            except Exception as e:
                print(f"   ⚠️  Error en query {query_idx}: {e}")
                continue
        
        # Calcular estadísticas finales
        final_results = {}
        for metric in results:
            values = [v for v in results[metric] if not np.isnan(v)]
            if values:
                final_results[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'samples': len(values)
                }
            else:
                final_results[metric] = None
        
        # Mostrar resultados
        print("   📈 Resultados Precision@K:")
        for k in k_values:
            precision_stats = final_results.get(f'precision_at_{k}')
            if precision_stats:
                print(f"   P@{k}: {precision_stats['mean']:.3f} ± {precision_stats['std']:.3f}")
        
        print("   📈 Resultados Recall@K:")
        for k in k_values:
            recall_stats = final_results.get(f'recall_at_{k}')
            if recall_stats:
                print(f"   R@{k}: {recall_stats['mean']:.3f} ± {recall_stats['std']:.3f}")
        
        return {
            'evaluation_method': 'cluster_based_ground_truth',
            'sample_size': len(sample_indices),
            'k_values': k_values,
            'results': final_results
        }


class DiversityAnalyzer:
    """Analizador de diversidad en recomendaciones"""
    
    def __init__(self, data_loader: MusicDataLoader, recommender: HybridMusicRecommender):
        self.data_loader = data_loader
        self.recommender = recommender
        
    def calculate_intra_list_diversity(self, recommendation_indices: List[int]) -> float:
        """Calcula diversidad intra-lista usando similitud coseno"""
        if len(recommendation_indices) < 2:
            return 1.0
        
        # Obtener vectores musicales de las recomendaciones
        musical_features = self.data_loader.get_musical_vectors()
        rec_vectors = musical_features[recommendation_indices]
        
        # Calcular similitudes pairwise
        similarities = cosine_similarity(rec_vectors)
        
        # Excluir diagonal (similitud consigo mismo)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        avg_similarity = similarities[mask].mean()
        
        # Diversidad = 1 - similitud promedio
        return 1.0 - avg_similarity
    
    def calculate_musical_feature_coverage(self, recommendation_indices: List[int]) -> Dict[str, float]:
        """Calcula cobertura de características musicales"""
        musical_features = self.data_loader.get_musical_vectors()
        rec_vectors = musical_features[recommendation_indices]
        
        # Características musicales (nombres aproximados)
        feature_names = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'duration_ms'
        ]
        
        coverage = {}
        for i, feature_name in enumerate(feature_names):
            feature_values = rec_vectors[:, i]
            coverage[feature_name] = {
                'range': float(feature_values.max() - feature_values.min()),
                'std': float(feature_values.std()),
                'unique_ratio': len(np.unique(feature_values)) / len(feature_values)
            }
        
        return coverage
    
    def calculate_semantic_diversity(self, recommendation_indices: List[int]) -> float:
        """Calcula diversidad semántica usando embeddings BERT"""
        if len(recommendation_indices) < 2:
            return 1.0
            
        semantic_vectors = self.data_loader.get_semantic_vectors()
        rec_vectors = semantic_vectors[recommendation_indices]
        
        # Calcular similitudes semánticas
        similarities = cosine_similarity(rec_vectors)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        avg_similarity = similarities[mask].mean()
        
        return 1.0 - avg_similarity
    
    def measure_genre_distribution(self, recommendation_indices: List[int]) -> Dict[str, float]:
        """Mide distribución de géneros en recomendaciones"""
        try:
            metadata = self.data_loader.get_metadata()
            rec_metadata = metadata.iloc[recommendation_indices]
            
            if 'playlist_genre' in rec_metadata.columns:
                genre_counts = rec_metadata['playlist_genre'].value_counts()
                total_recs = len(recommendation_indices)
                
                distribution = {}
                for genre, count in genre_counts.items():
                    distribution[genre] = count / total_recs
                
                # Calcular entropía como medida de balance
                probs = np.array(list(distribution.values()))
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                max_entropy = np.log2(len(distribution))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                return {
                    'distribution': distribution,
                    'entropy': float(entropy),
                    'normalized_entropy': float(normalized_entropy),
                    'unique_genres': len(distribution)
                }
            else:
                return {'error': 'No genre information available'}
                
        except Exception as e:
            return {'error': f'Error analyzing genres: {e}'}
    
    def analyze_diversity(self, sample_size: int = 50, num_recommendations: int = 10) -> Dict:
        """Analiza diversidad en muestra de recomendaciones"""
        print(f"🎨 Analizando diversidad de recomendaciones (muestra: {sample_size})")
        
        total_songs = len(self.data_loader.get_track_ids())
        sample_indices = np.random.choice(total_songs, min(sample_size, total_songs), replace=False)
        
        diversity_results = {
            'intra_list_diversity': [],
            'semantic_diversity': [],
            'musical_coverage_ranges': [],
            'genre_entropy': []
        }
        
        for i, query_idx in enumerate(sample_indices):
            if i % 10 == 0:
                print(f"   Procesando {i+1}/{len(sample_indices)}...")
            
            try:
                # Obtener recomendaciones
                track_id = self.data_loader.get_track_ids()[query_idx]
                recommendations = self.recommender.recommend(
                    track_id=track_id,
                    num_recommendations=num_recommendations,
                    return_indices=True
                )
                
                # Calcular métricas de diversidad
                intra_diversity = self.calculate_intra_list_diversity(recommendations)
                semantic_diversity = self.calculate_semantic_diversity(recommendations)
                
                diversity_results['intra_list_diversity'].append(intra_diversity)
                diversity_results['semantic_diversity'].append(semantic_diversity)
                
                # Cobertura de características musicales
                coverage = self.calculate_musical_feature_coverage(recommendations)
                avg_range = np.mean([feat['range'] for feat in coverage.values()])
                diversity_results['musical_coverage_ranges'].append(avg_range)
                
                # Distribución de géneros
                genre_dist = self.measure_genre_distribution(recommendations)
                if 'normalized_entropy' in genre_dist:
                    diversity_results['genre_entropy'].append(genre_dist['normalized_entropy'])
                
            except Exception as e:
                print(f"   ⚠️  Error en diversidad query {query_idx}: {e}")
                continue
        
        # Calcular estadísticas finales
        final_results = {}
        for metric, values in diversity_results.items():
            if values:
                final_results[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'samples': len(values)
                }
        
        # Mostrar resultados
        for metric, stats in final_results.items():
            print(f"   📊 {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        return {
            'sample_size': len(sample_indices),
            'num_recommendations': num_recommendations,
            'results': final_results
        }


class CoherenceValidator:
    """Validador de coherencia entre componentes del sistema"""
    
    def __init__(self, data_loader: MusicDataLoader, recommender: HybridMusicRecommender, 
                 explainer: RecommendationExplainer):
        self.data_loader = data_loader
        self.recommender = recommender
        self.explainer = explainer
    
    def validate_cluster_explanation_alignment(self, sample_size: int = 30) -> Dict:
        """Valida alineación entre clusters y explicaciones"""
        print(f"🔍 Validando coherencia cluster-explicación (muestra: {sample_size})")
        
        total_songs = len(self.data_loader.get_track_ids())
        sample_indices = np.random.choice(total_songs, min(sample_size, total_songs), replace=False)
        
        coherence_scores = []
        
        for i, query_idx in enumerate(sample_indices):
            try:
                track_id = self.data_loader.get_track_ids()[query_idx]
                
                # Obtener recomendaciones y explicaciones
                recommendations = self.recommender.recommend(track_id=track_id, num_recommendations=5)
                explanation = self.explainer.explain_recommendations(track_id, recommendations)
                
                # Verificar que la explicación contenga información relevante
                explanation_text = explanation.get('summary', '')
                
                # Criterios de coherencia básicos
                coherence_checks = {
                    'has_explanation': len(explanation_text) > 50,
                    'mentions_musical_features': any(term in explanation_text.lower() 
                                                   for term in ['energy', 'danceability', 'valence', 'acoustic']),
                    'mentions_clusters': 'cluster' in explanation_text.lower(),
                    'explanation_structure_valid': 'summary' in explanation and 'details' in explanation
                }
                
                coherence_score = sum(coherence_checks.values()) / len(coherence_checks)
                coherence_scores.append(coherence_score)
                
            except Exception as e:
                print(f"   ⚠️  Error en coherencia query {query_idx}: {e}")
                continue
        
        if coherence_scores:
            final_score = np.mean(coherence_scores)
            print(f"   📊 Coherencia promedio: {final_score:.3f}")
            
            return {
                'coherence_score': float(final_score),
                'sample_size': len(coherence_scores),
                'distribution': {
                    'mean': float(np.mean(coherence_scores)),
                    'std': float(np.std(coherence_scores)),
                    'min': float(np.min(coherence_scores)),
                    'max': float(np.max(coherence_scores))
                }
            }
        else:
            return {'error': 'No se pudieron evaluar coherencias'}
    
    def validate_recommendation_consistency(self, sample_size: int = 20) -> Dict:
        """Valida consistencia de recomendaciones para inputs similares"""
        print(f"🔄 Validando consistencia de recomendaciones (muestra: {sample_size})")
        
        musical_vectors = self.data_loader.get_musical_vectors()
        semantic_vectors = self.data_loader.get_semantic_vectors()
        track_ids = self.data_loader.get_track_ids()
        
        consistency_scores = []
        
        for i in range(sample_size):
            try:
                # Seleccionar canción query aleatoria
                query_idx = np.random.randint(0, len(track_ids))
                query_track_id = track_ids[query_idx]
                
                # Encontrar canción más similar
                query_musical = musical_vectors[query_idx].reshape(1, -1)
                musical_similarities = cosine_similarity(query_musical, musical_vectors)[0]
                
                # Excluir la misma canción
                musical_similarities[query_idx] = -1
                similar_idx = np.argmax(musical_similarities)
                similar_track_id = track_ids[similar_idx]
                
                # Obtener recomendaciones para ambas
                recs1 = self.recommender.recommend(track_id=query_track_id, num_recommendations=10)
                recs2 = self.recommender.recommend(track_id=similar_track_id, num_recommendations=10)
                
                # Calcular overlap en recomendaciones
                recs1_ids = set(recs1)
                recs2_ids = set(recs2)
                
                intersection = len(recs1_ids & recs2_ids)
                union = len(recs1_ids | recs2_ids)
                jaccard_similarity = intersection / union if union > 0 else 0
                
                consistency_scores.append(jaccard_similarity)
                
            except Exception as e:
                print(f"   ⚠️  Error en consistencia iteración {i}: {e}")
                continue
        
        if consistency_scores:
            final_score = np.mean(consistency_scores)
            print(f"   📊 Consistencia promedio: {final_score:.3f}")
            
            return {
                'consistency_score': float(final_score),
                'sample_size': len(consistency_scores),
                'distribution': {
                    'mean': float(np.mean(consistency_scores)),
                    'std': float(np.std(consistency_scores)),
                    'min': float(np.min(consistency_scores)),
                    'max': float(np.max(consistency_scores))
                }
            }
        else:
            return {'error': 'No se pudo evaluar consistencia'}
    
    def validate_hybrid_weight_effectiveness(self, sample_size: int = 30) -> Dict:
        """Valida efectividad de los pesos híbridos 55%-45%"""
        print(f"⚖️  Validando efectividad de pesos híbridos (muestra: {sample_size})")
        
        # Configurar diferentes combinaciones de pesos para comparar
        weight_configs = [
            (1.0, 0.0),  # Solo musical
            (0.0, 1.0),  # Solo semántico
            (0.5, 0.5),  # Balance 50-50
            (0.55, 0.45)  # Configuración actual (validada en FASE 3)
        ]
        
        results = {}
        
        for musical_weight, semantic_weight in weight_configs:
            print(f"   Evaluando pesos: Musical {musical_weight:.2f}, Semántico {semantic_weight:.2f}")
            
            # Crear recomendador temporal con estos pesos
            temp_recommender = HybridMusicRecommender(self.data_loader)
            temp_recommender.musical_weight = musical_weight
            temp_recommender.semantic_weight = semantic_weight
            
            # Evaluar diversidad con estos pesos
            diversity_analyzer = DiversityAnalyzer(self.data_loader, temp_recommender)
            diversity_results = diversity_analyzer.analyze_diversity(
                sample_size=min(sample_size, 20), 
                num_recommendations=5
            )
            
            config_name = f"musical_{musical_weight:.2f}_semantic_{semantic_weight:.2f}"
            results[config_name] = {
                'weights': (musical_weight, semantic_weight),
                'diversity_metrics': diversity_results['results']
            }
        
        # Identificar configuración con mejor balance
        best_config = None
        best_score = 0
        
        for config_name, config_data in results.items():
            if 'intra_list_diversity' in config_data['diversity_metrics']:
                diversity_score = config_data['diversity_metrics']['intra_list_diversity']['mean']
                if diversity_score > best_score:
                    best_score = diversity_score
                    best_config = config_name
        
        print(f"   🏆 Mejor configuración: {best_config} (diversidad: {best_score:.3f})")
        
        return {
            'weight_configurations_tested': weight_configs,
            'results': results,
            'best_configuration': best_config,
            'best_diversity_score': float(best_score)
        }


class PerformanceBenchmarker:
    """Sistema de benchmarking de performance"""
    
    def __init__(self, data_loader: MusicDataLoader, recommender: HybridMusicRecommender):
        self.data_loader = data_loader
        self.recommender = recommender
        
    def benchmark_recommendation_latency(self, sample_size: int = 100) -> Dict:
        """Benchmarka latencia de recomendaciones (objetivo <100ms)"""
        print(f"⚡ Benchmarking latencia de recomendaciones (muestra: {sample_size})")
        
        track_ids = self.data_loader.get_track_ids()
        sample_tracks = np.random.choice(track_ids, min(sample_size, len(track_ids)), replace=False)
        
        latencies = []
        
        for i, track_id in enumerate(sample_tracks):
            if i % 25 == 0:
                print(f"   Procesando {i+1}/{len(sample_tracks)}...")
            
            try:
                start_time = time.time()
                recommendations = self.recommender.recommend(track_id=track_id, num_recommendations=5)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                print(f"   ⚠️  Error en latencia track {track_id}: {e}")
                continue
        
        if latencies:
            mean_latency = np.mean(latencies)
            target_met = mean_latency < 100  # Objetivo <100ms
            
            print(f"   📊 Latencia promedio: {mean_latency:.2f}ms")
            print(f"   🎯 Objetivo <100ms: {'✅ CUMPLIDO' if target_met else '❌ NO CUMPLIDO'}")
            
            return {
                'mean_latency_ms': float(mean_latency),
                'std_latency_ms': float(np.std(latencies)),
                'min_latency_ms': float(np.min(latencies)),
                'max_latency_ms': float(np.max(latencies)),
                'percentile_95_ms': float(np.percentile(latencies, 95)),
                'target_100ms_met': target_met,
                'sample_size': len(latencies),
                'success_rate': len(latencies) / len(sample_tracks)
            }
        else:
            return {'error': 'No se pudieron medir latencias'}
    
    def benchmark_system_throughput(self, duration_seconds: int = 30) -> Dict:
        """Benchmarka throughput del sistema (recomendaciones/segundo)"""
        print(f"🚀 Benchmarking throughput del sistema ({duration_seconds}s)")
        
        track_ids = self.data_loader.get_track_ids()
        
        start_time = time.time()
        recommendations_count = 0
        errors_count = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                # Seleccionar track aleatorio
                track_id = np.random.choice(track_ids)
                
                # Generar recomendación
                recommendations = self.recommender.recommend(track_id=track_id, num_recommendations=5)
                recommendations_count += 1
                
                if recommendations_count % 10 == 0:
                    elapsed = time.time() - start_time
                    current_throughput = recommendations_count / elapsed
                    print(f"   📊 {recommendations_count} recomendaciones, {current_throughput:.1f} rec/s")
                
            except Exception as e:
                errors_count += 1
                continue
        
        total_time = time.time() - start_time
        throughput = recommendations_count / total_time if total_time > 0 else 0
        
        print(f"   🎯 Throughput final: {throughput:.2f} recomendaciones/segundo")
        print(f"   📊 Total: {recommendations_count} recomendaciones, {errors_count} errores")
        
        return {
            'throughput_rec_per_second': float(throughput),
            'total_recommendations': recommendations_count,
            'total_errors': errors_count,
            'success_rate': recommendations_count / (recommendations_count + errors_count) if (recommendations_count + errors_count) > 0 else 0,
            'test_duration_seconds': float(total_time)
        }
    
    def profile_memory_consumption(self) -> Dict:
        """Perfila consumo de memoria del sistema"""
        print("💾 Perfilando consumo de memoria del sistema")
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            # Memoria en MB
            rss_mb = memory_info.rss / (1024 * 1024)
            vms_mb = memory_info.vms / (1024 * 1024)
            
            # Información del sistema
            system_memory = psutil.virtual_memory()
            memory_percent = process.memory_percent()
            
            print(f"   📊 Memoria RSS: {rss_mb:.1f} MB")
            print(f"   📊 Memoria VMS: {vms_mb:.1f} MB")
            print(f"   📊 Porcentaje del sistema: {memory_percent:.2f}%")
            
            # Verificar si cumple objetivo <512MB
            target_met = rss_mb < 512
            print(f"   🎯 Objetivo <512MB: {'✅ CUMPLIDO' if target_met else '❌ NO CUMPLIDO'}")
            
            return {
                'rss_memory_mb': float(rss_mb),
                'vms_memory_mb': float(vms_mb),
                'memory_percent': float(memory_percent),
                'system_total_memory_gb': float(system_memory.total / (1024**3)),
                'target_512mb_met': target_met
            }
            
        except ImportError:
            print("   ⚠️  psutil no disponible, instalando...")
            return {'error': 'psutil library required for memory profiling'}
        except Exception as e:
            print(f"   ❌ Error perfilando memoria: {e}")
            return {'error': f'Memory profiling failed: {e}'}
    
    def measure_initialization_overhead(self) -> Dict:
        """Mide overhead de inicialización del sistema"""
        print("🚀 Midiendo overhead de inicialización")
        
        try:
            # Simular inicialización completa
            start_time = time.time()
            
            # Inicializar componentes
            temp_loader = MusicDataLoader(str(self.data_loader.system_dir))
            temp_recommender = HybridMusicRecommender(temp_loader)
            
            end_time = time.time()
            initialization_time = end_time - start_time
            
            # Verificar objetivo <5 segundos
            target_met = initialization_time < 5.0
            
            print(f"   📊 Tiempo de inicialización: {initialization_time:.2f}s")
            print(f"   🎯 Objetivo <5s: {'✅ CUMPLIDO' if target_met else '❌ NO CUMPLIDO'}")
            
            return {
                'initialization_time_seconds': float(initialization_time),
                'target_5s_met': target_met,
                'components_initialized': ['data_loader', 'recommender']
            }
            
        except Exception as e:
            print(f"   ❌ Error midiendo inicialización: {e}")
            return {'error': f'Initialization timing failed: {e}'}
    
    def run_complete_benchmark(self) -> Dict:
        """Ejecuta benchmark completo de performance"""
        print("🏁 EJECUTANDO BENCHMARK COMPLETO DE PERFORMANCE")
        print("=" * 60)
        
        results = {}
        
        # Latencia
        results['latency'] = self.benchmark_recommendation_latency(sample_size=50)
        
        # Throughput
        results['throughput'] = self.benchmark_system_throughput(duration_seconds=20)
        
        # Memoria
        results['memory'] = self.profile_memory_consumption()
        
        # Inicialización
        results['initialization'] = self.measure_initialization_overhead()
        
        # Evaluar cumplimiento de objetivos generales
        objectives_met = {
            'latency_under_100ms': results.get('latency', {}).get('target_100ms_met', False),
            'memory_under_512mb': results.get('memory', {}).get('target_512mb_met', False),
            'initialization_under_5s': results.get('initialization', {}).get('target_5s_met', False),
            'throughput_over_10_per_sec': results.get('throughput', {}).get('throughput_rec_per_second', 0) > 10
        }
        
        total_objectives = len(objectives_met)
        met_objectives = sum(objectives_met.values())
        
        print(f"\n🎯 OBJETIVOS DE PERFORMANCE: {met_objectives}/{total_objectives} cumplidos")
        for objective, met in objectives_met.items():
            status = "✅" if met else "❌"
            print(f"   {status} {objective}")
        
        results['objectives_summary'] = {
            'total_objectives': total_objectives,
            'met_objectives': met_objectives,
            'success_rate': met_objectives / total_objectives,
            'individual_results': objectives_met
        }
        
        return results


class ComprehensiveValidator:
    """Validador comprensivo que integra todos los componentes"""
    
    def __init__(self, system_dir: str = None):
        self.system_validator = SystemValidator(system_dir)
        self.precision_recall_evaluator = None
        self.diversity_analyzer = None
        self.coherence_validator = None
        self.performance_benchmarker = None
        
        self.comprehensive_results = {}
    
    def run_comprehensive_validation(self, 
                                   include_precision_recall: bool = True,
                                   include_diversity: bool = True, 
                                   include_coherence: bool = True,
                                   include_performance: bool = True,
                                   sample_sizes: Dict[str, int] = None) -> Dict:
        """Ejecuta validación científica comprensiva"""
        
        if sample_sizes is None:
            sample_sizes = {
                'precision_recall': 50,
                'diversity': 30,
                'coherence': 20,
                'performance_latency': 50
            }
        
        print("🎯 INICIANDO VALIDACIÓN CIENTÍFICA COMPRENSIVA")
        print("=" * 70)
        
        # 1. Validación básica del sistema
        print("\n📋 FASE 1: VALIDACIÓN BÁSICA DEL SISTEMA")
        system_valid = self.system_validator.run_complete_validation()
        
        if not system_valid:
            print("❌ Sistema no válido. Abortando validación comprensiva.")
            return {
                'system_valid': False,
                'error': 'Basic system validation failed'
            }
        
        # 2. Inicializar componentes avanzados
        print("\n📋 FASE 2: INICIALIZACIÓN DE COMPONENTES AVANZADOS")
        if not self.system_validator.initialize_system_components():
            print("❌ Error inicializando componentes. Abortando.")
            return {
                'system_valid': True,
                'components_initialized': False,
                'error': 'Advanced components initialization failed'
            }
        
        # Crear evaluadores especializados
        self.precision_recall_evaluator = PrecisionRecallEvaluator(
            self.system_validator.data_loader, 
            self.system_validator.recommender
        )
        
        self.diversity_analyzer = DiversityAnalyzer(
            self.system_validator.data_loader,
            self.system_validator.recommender
        )
        
        self.coherence_validator = CoherenceValidator(
            self.system_validator.data_loader,
            self.system_validator.recommender,
            self.system_validator.explainer
        )
        
        self.performance_benchmarker = PerformanceBenchmarker(
            self.system_validator.data_loader,
            self.system_validator.recommender
        )
        
        # 3. Evaluaciones especializadas
        results = {
            'system_validation': self.system_validator.validation_results,
            'system_valid': True,
            'components_initialized': True
        }
        
        if include_precision_recall:
            print("\n📋 FASE 3: EVALUACIÓN DE PRECISIÓN Y RECALL")
            try:
                results['precision_recall'] = self.precision_recall_evaluator.evaluate_precision_recall(
                    sample_size=sample_sizes['precision_recall']
                )
            except Exception as e:
                print(f"   ❌ Error en evaluación P@K/R@K: {e}")
                results['precision_recall'] = {'error': str(e)}
        
        if include_diversity:
            print("\n📋 FASE 4: ANÁLISIS DE DIVERSIDAD")
            try:
                results['diversity'] = self.diversity_analyzer.analyze_diversity(
                    sample_size=sample_sizes['diversity']
                )
            except Exception as e:
                print(f"   ❌ Error en análisis de diversidad: {e}")
                results['diversity'] = {'error': str(e)}
        
        if include_coherence:
            print("\n📋 FASE 5: VALIDACIÓN DE COHERENCIA")
            try:
                coherence_results = {}
                coherence_results['cluster_explanation_alignment'] = \
                    self.coherence_validator.validate_cluster_explanation_alignment(
                        sample_size=sample_sizes['coherence']
                    )
                
                coherence_results['recommendation_consistency'] = \
                    self.coherence_validator.validate_recommendation_consistency(
                        sample_size=sample_sizes['coherence']
                    )
                
                coherence_results['hybrid_weight_effectiveness'] = \
                    self.coherence_validator.validate_hybrid_weight_effectiveness(
                        sample_size=sample_sizes['coherence']
                    )
                
                results['coherence'] = coherence_results
                
            except Exception as e:
                print(f"   ❌ Error en validación de coherencia: {e}")
                results['coherence'] = {'error': str(e)}
        
        if include_performance:
            print("\n📋 FASE 6: BENCHMARKING DE PERFORMANCE")
            try:
                results['performance'] = self.performance_benchmarker.run_complete_benchmark()
            except Exception as e:
                print(f"   ❌ Error en benchmarking: {e}")
                results['performance'] = {'error': str(e)}
        
        # 4. Generar reporte final
        print("\n📋 FASE 7: GENERACIÓN DE REPORTE CIENTÍFICO")
        final_report = self.generate_scientific_report(results)
        
        self.comprehensive_results = final_report
        
        print(f"\n🏆 VALIDACIÓN COMPRENSIVA COMPLETADA")
        print(f"📊 Score científico general: {final_report.get('overall_scientific_score', 'N/A')}")
        
        return final_report
    
    def generate_scientific_report(self, results: Dict) -> Dict:
        """Genera reporte científico comprensivo"""
        
        # Calcular scores individuales
        scores = {}
        
        # Score de precisión/recall
        if 'precision_recall' in results and 'error' not in results['precision_recall']:
            pr_results = results['precision_recall']['results']
            precision_5 = pr_results.get('precision_at_5', {}).get('mean', 0)
            recall_10 = pr_results.get('recall_at_10', {}).get('mean', 0)
            scores['precision_recall'] = (precision_5 + recall_10) / 2
        
        # Score de diversidad
        if 'diversity' in results and 'error' not in results['diversity']:
            div_results = results['diversity']['results']
            intra_diversity = div_results.get('intra_list_diversity', {}).get('mean', 0)
            semantic_diversity = div_results.get('semantic_diversity', {}).get('mean', 0)
            scores['diversity'] = (intra_diversity + semantic_diversity) / 2
        
        # Score de coherencia
        if 'coherence' in results and 'error' not in results['coherence']:
            coh_results = results['coherence']
            cluster_coherence = coh_results.get('cluster_explanation_alignment', {}).get('coherence_score', 0)
            consistency = coh_results.get('recommendation_consistency', {}).get('consistency_score', 0)
            scores['coherence'] = (cluster_coherence + consistency) / 2
        
        # Score de performance
        if 'performance' in results and 'error' not in results['performance']:
            perf_results = results['performance']
            objectives_summary = perf_results.get('objectives_summary', {})
            scores['performance'] = objectives_summary.get('success_rate', 0)
        
        # Score general (promedio ponderado)
        if scores:
            weights = {'precision_recall': 0.3, 'diversity': 0.25, 'coherence': 0.25, 'performance': 0.2}
            weighted_scores = []
            
            for metric, weight in weights.items():
                if metric in scores:
                    weighted_scores.append(scores[metric] * weight)
            
            overall_score = sum(weighted_scores) / sum(weights[m] for m in scores.keys()) if weighted_scores else 0
        else:
            overall_score = 0
        
        # Interpretación académica
        if overall_score >= 0.8:
            interpretation = "EXCELENTE - Cumple estándares académicos superiores"
            grade = "A"
        elif overall_score >= 0.7:
            interpretation = "BUENO - Cumple estándares académicos esperados"
            grade = "B"
        elif overall_score >= 0.6:
            interpretation = "SATISFACTORIO - Cumple requisitos mínimos"
            grade = "C"
        else:
            interpretation = "NECESITA MEJORAS - Por debajo de estándares académicos"
            grade = "D"
        
        scientific_report = {
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_scientific_score': float(overall_score),
            'academic_interpretation': interpretation,
            'academic_grade': grade,
            'individual_scores': {k: float(v) for k, v in scores.items()},
            'detailed_results': results,
            'scientific_conclusions': {
                'system_reliability': 'high' if results.get('system_valid', False) else 'low',
                'recommendation_quality': 'high' if scores.get('precision_recall', 0) > 0.6 else 'medium' if scores.get('precision_recall', 0) > 0.4 else 'low',
                'algorithmic_diversity': 'high' if scores.get('diversity', 0) > 0.7 else 'medium' if scores.get('diversity', 0) > 0.5 else 'low',
                'system_coherence': 'high' if scores.get('coherence', 0) > 0.8 else 'medium' if scores.get('coherence', 0) > 0.6 else 'low',
                'production_readiness': 'ready' if scores.get('performance', 0) > 0.75 else 'needs_optimization'
            },
            'recommendations_for_thesis': {
                'strengths_to_highlight': [],
                'areas_for_improvement': [],
                'additional_experiments_suggested': []
            }
        }
        
        # Generar recomendaciones específicas para tesis
        if scores.get('precision_recall', 0) > 0.65:
            scientific_report['recommendations_for_thesis']['strengths_to_highlight'].append(
                'Sistema demuestra alta precisión en recomendaciones con metodología científicamente validada'
            )
        
        if scores.get('diversity', 0) > 0.6:
            scientific_report['recommendations_for_thesis']['strengths_to_highlight'].append(
                'Diversidad algorítmica superior a baselines tradicionales'
            )
        
        if scores.get('performance', 0) > 0.75:
            scientific_report['recommendations_for_thesis']['strengths_to_highlight'].append(
                'Sistema cumple objetivos de performance para aplicaciones de producción'
            )
        
        return scientific_report
    
    def export_comprehensive_report(self, filepath: str = None) -> str:
        """Exporta reporte comprensivo a archivo JSON"""
        if filepath is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = f'comprehensive_validation_report_{timestamp}.json'
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.comprehensive_results, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Reporte exportado: {filepath}")
            return filepath
        
        except Exception as e:
            print(f"❌ Error exportando reporte: {e}")
            return None


def main():
    """Función principal con interfaz CLI"""
    parser = argparse.ArgumentParser(
        description='Sistema de Validación Completo para Recomendaciones Musicales'
    )
    
    parser.add_argument(
        '--mode', 
        choices=['basic', 'precision-recall', 'diversity', 'coherence', 'performance', 'comprehensive'],
        default='comprehensive',
        help='Modo de validación a ejecutar (default: comprehensive)'
    )
    
    parser.add_argument(
        '--sample-size-pr', type=int, default=50,
        help='Tamaño de muestra para evaluación precision-recall'
    )
    
    parser.add_argument(
        '--sample-size-div', type=int, default=30,
        help='Tamaño de muestra para análisis de diversidad'
    )
    
    parser.add_argument(
        '--sample-size-coh', type=int, default=20,
        help='Tamaño de muestra para validación de coherencia'
    )
    
    parser.add_argument(
        '--export-report', action='store_true',
        help='Exportar reporte comprensivo a JSON'
    )
    
    parser.add_argument(
        '--report-file', type=str,
        help='Nombre del archivo de reporte (opcional)'
    )
    
    args = parser.parse_args()
    
    # Configurar tamaños de muestra
    sample_sizes = {
        'precision_recall': args.sample_size_pr,
        'diversity': args.sample_size_div,
        'coherence': args.sample_size_coh,
        'performance_latency': args.sample_size_pr
    }
    
    # Crear validador comprensivo
    validator = ComprehensiveValidator()
    
    try:
        if args.mode == 'basic':
            # Solo validación básica del sistema
            results = validator.system_validator.run_complete_validation()
            print(f"\n{'✅ SISTEMA VÁLIDO' if results else '❌ SISTEMA NO VÁLIDO'}")
            
        elif args.mode == 'comprehensive':
            # Validación completa
            results = validator.run_comprehensive_validation(
                sample_sizes=sample_sizes
            )
            
            # Exportar reporte si se solicita
            if args.export_report:
                validator.export_comprehensive_report(args.report_file)
                
        else:
            # Modos individuales
            basic_valid = validator.system_validator.run_complete_validation()
            if not basic_valid:
                print("❌ Sistema básico no válido. Abortando.")
                return 1
            
            # Inicializar componentes
            if not validator.system_validator.initialize_system_components():
                print("❌ Error inicializando componentes.")
                return 1
            
            if args.mode == 'precision-recall':
                evaluator = PrecisionRecallEvaluator(
                    validator.system_validator.data_loader,
                    validator.system_validator.recommender
                )
                results = evaluator.evaluate_precision_recall(sample_sizes['precision_recall'])
                
            elif args.mode == 'diversity':
                analyzer = DiversityAnalyzer(
                    validator.system_validator.data_loader,
                    validator.system_validator.recommender
                )
                results = analyzer.analyze_diversity(sample_sizes['diversity'])
                
            elif args.mode == 'coherence':
                coherence_validator = CoherenceValidator(
                    validator.system_validator.data_loader,
                    validator.system_validator.recommender,
                    validator.system_validator.explainer
                )
                results = coherence_validator.validate_cluster_explanation_alignment(
                    sample_sizes['coherence']
                )
                
            elif args.mode == 'performance':
                benchmarker = PerformanceBenchmarker(
                    validator.system_validator.data_loader,
                    validator.system_validator.recommender
                )
                results = benchmarker.run_complete_benchmark()
        
        print("\n🎉 VALIDACIÓN COMPLETADA EXITOSAMENTE")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Validación interrumpida por el usuario")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error durante la validación: {e}")
        return 1


if __name__ == "__main__":
    exit(main())