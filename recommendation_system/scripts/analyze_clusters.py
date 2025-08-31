#!/usr/bin/env python3
"""
Analizador Avanzado de Clusters Musicales y Semánticos
====================================================

Herramienta dedicada para análisis estadístico exhaustivo de clusters
Proporciona caracterización completa, reportes y visualizaciones

Uso:
    python analyze_clusters.py --all                    # Análisis completo
    python analyze_clusters.py --musical --cluster_id 5 # Cluster musical específico
    python analyze_clusters.py --semantic --overview    # Overview semántico
    python analyze_clusters.py --compare               # Comparación entre dominios
    python analyze_clusters.py --export_report        # Exportar reporte completo

Autor: Sistema de Clustering Multimodal
Fecha: Agosto 2025
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Agregar directorio de scripts al path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from load_system import MusicDataLoader
    from explain_recommendations import RecommendationExplainer
except ImportError as e:
    print(f"❌ Error importando componentes: {e}")
    sys.exit(1)

class AdvancedClusterAnalyzer:
    """
    Analizador avanzado de clusters musicales y semánticos
    Proporciona análisis estadístico exhaustivo y reportes interpretativos
    """
    
    def __init__(self, system_dir: str = None, verbose: bool = True):
        """
        Inicializa el analizador avanzado de clusters
        
        Args:
            system_dir: Directorio del sistema (None para auto-detectar)
            verbose: Mostrar información detallada
        """
        self.verbose = verbose
        
        try:
            if verbose:
                print("🔬 Inicializando Analizador Avanzado de Clusters...")
            
            self.loader = MusicDataLoader(system_dir)
            self.explainer = RecommendationExplainer(system_dir)
            
            # Cargar configuración
            self.config = self.loader.get_config()
            self.system_stats = self.loader.get_system_stats()
            
            if verbose:
                print(f"✅ Sistema inicializado")
                print(f"   Dataset: {self.system_stats['dataset_size']} canciones")
                print(f"   Clusters: {self.system_stats['cluster_counts']['musical']} musicales, {self.system_stats['cluster_counts']['semantic']} semánticos")
                
        except Exception as e:
            print(f"❌ Error inicializando analizador: {e}")
            raise
    
    def analyze_musical_cluster_advanced(self, cluster_id: int) -> Dict:
        """
        Análisis musical avanzado con estadísticas extendidas
        
        Args:
            cluster_id: ID del cluster musical (0-9)
            
        Returns:
            Dict con análisis estadístico avanzado
        """
        # Obtener análisis base
        base_analysis = self.explainer.analyze_musical_cluster(cluster_id)
        
        if 'error' in base_analysis:
            return base_analysis
        
        # Extender con análisis avanzado
        cluster_mask = self.loader.get_musical_clusters() == cluster_id
        cluster_vectors = self.loader.get_musical_vectors()[cluster_mask]
        cluster_track_ids = self.loader.get_track_ids()[cluster_mask]
        
        # Análisis de variabilidad intra-cluster
        variability_analysis = self._analyze_cluster_variability(cluster_vectors)
        
        # Análisis de separabilidad inter-cluster
        separability_analysis = self._analyze_cluster_separability(cluster_id, 'musical')
        
        # Análisis de representatividad de canciones
        representativeness_analysis = self._analyze_song_representativeness(
            cluster_vectors, cluster_track_ids, 'musical'
        )
        
        # Análisis temporal y de popularidad
        temporal_analysis = self._analyze_cluster_temporal_characteristics(cluster_track_ids)
        
        # Combinar análisis
        extended_analysis = base_analysis.copy()
        extended_analysis.update({
            'advanced_statistics': {
                'variability_metrics': variability_analysis,
                'separability_metrics': separability_analysis,
                'representativeness_analysis': representativeness_analysis,
                'temporal_characteristics': temporal_analysis
            },
            'cluster_quality_score': self._calculate_cluster_quality_score(
                variability_analysis, separability_analysis, base_analysis
            )
        })
        
        return extended_analysis
    
    def analyze_semantic_cluster_advanced(self, cluster_id: int) -> Dict:
        """
        Análisis semántico avanzado con métricas de coherencia extendidas
        
        Args:
            cluster_id: ID del cluster semántico (0-5)
            
        Returns:
            Dict con análisis semántico avanzado
        """
        # Obtener análisis base
        base_analysis = self.explainer.analyze_semantic_cluster(cluster_id)
        
        if 'error' in base_analysis:
            return base_analysis
        
        # Extender con análisis avanzado
        cluster_mask = self.loader.get_semantic_clusters() == cluster_id
        cluster_vectors = self.loader.get_semantic_vectors()[cluster_mask]
        cluster_track_ids = self.loader.get_track_ids()[cluster_mask]
        
        # Análisis de coherencia semántica extendida
        coherence_extended = self._analyze_semantic_coherence_extended(cluster_vectors)
        
        # Análisis de separabilidad semántica
        separability_analysis = self._analyze_cluster_separability(cluster_id, 'semantic')
        
        # Análisis de diversidad temática
        thematic_diversity = self._analyze_thematic_diversity(cluster_track_ids)
        
        # Análisis de representatividad semántica
        semantic_representativeness = self._analyze_song_representativeness(
            cluster_vectors, cluster_track_ids, 'semantic'
        )
        
        # Combinar análisis
        extended_analysis = base_analysis.copy()
        extended_analysis.update({
            'advanced_statistics': {
                'coherence_extended': coherence_extended,
                'separability_metrics': separability_analysis,
                'thematic_diversity': thematic_diversity,
                'semantic_representativeness': semantic_representativeness
            },
            'semantic_quality_score': self._calculate_semantic_quality_score(
                coherence_extended, separability_analysis, base_analysis
            )
        })
        
        return extended_analysis
    
    def _analyze_cluster_variability(self, cluster_vectors: np.ndarray) -> Dict:
        """Analiza la variabilidad interna del cluster"""
        if len(cluster_vectors) < 2:
            return {'error': 'Cluster con muy pocas canciones para análisis de variabilidad'}
        
        # Calcular centroide
        centroid = np.mean(cluster_vectors, axis=0)
        
        # Distancias al centroide
        distances_to_centroid = np.linalg.norm(cluster_vectors - centroid, axis=1)
        
        # Análisis de dispersión por característica
        feature_variance = np.var(cluster_vectors, axis=0)
        feature_std = np.std(cluster_vectors, axis=0)
        
        return {
            'mean_distance_to_centroid': float(np.mean(distances_to_centroid)),
            'std_distance_to_centroid': float(np.std(distances_to_centroid)),
            'max_distance_to_centroid': float(np.max(distances_to_centroid)),
            'min_distance_to_centroid': float(np.min(distances_to_centroid)),
            'feature_variances': feature_variance.tolist(),
            'feature_std_devs': feature_std.tolist(),
            'compactness_score': float(1.0 / (1.0 + np.mean(distances_to_centroid))),
            'uniformity_score': float(1.0 / (1.0 + np.std(distances_to_centroid)))
        }
    
    def _analyze_cluster_separability(self, cluster_id: int, cluster_type: str) -> Dict:
        """Analiza la separabilidad del cluster respecto a otros clusters"""
        if cluster_type == 'musical':
            all_clusters = self.loader.get_musical_clusters()
            all_vectors = self.loader.get_musical_vectors()
            max_clusters = self.config['optimal_configurations']['musical_clustering']['n_clusters']
        else:
            all_clusters = self.loader.get_semantic_clusters()
            all_vectors = self.loader.get_semantic_vectors()
            max_clusters = self.config['optimal_configurations']['semantic_clustering']['n_clusters']
        
        # Vectores del cluster actual
        current_cluster_mask = all_clusters == cluster_id
        current_vectors = all_vectors[current_cluster_mask]
        
        if len(current_vectors) == 0:
            return {'error': f'Cluster {cluster_id} vacío'}
        
        current_centroid = np.mean(current_vectors, axis=0)
        
        # Calcular distancias a otros clusters
        inter_cluster_distances = []
        cluster_sizes = []
        
        for other_cluster_id in range(max_clusters):
            if other_cluster_id == cluster_id:
                continue
                
            other_cluster_mask = all_clusters == other_cluster_id
            other_vectors = all_vectors[other_cluster_mask]
            
            if len(other_vectors) > 0:
                other_centroid = np.mean(other_vectors, axis=0)
                distance = np.linalg.norm(current_centroid - other_centroid)
                inter_cluster_distances.append(distance)
                cluster_sizes.append(len(other_vectors))
        
        if not inter_cluster_distances:
            return {'error': 'No se encontraron otros clusters para comparar'}
        
        return {
            'mean_inter_cluster_distance': float(np.mean(inter_cluster_distances)),
            'min_inter_cluster_distance': float(np.min(inter_cluster_distances)),
            'max_inter_cluster_distance': float(np.max(inter_cluster_distances)),
            'nearest_cluster_distance': float(np.min(inter_cluster_distances)),
            'separability_score': float(np.min(inter_cluster_distances)),
            'relative_separation': float(np.mean(inter_cluster_distances) / np.std(inter_cluster_distances)) if np.std(inter_cluster_distances) > 0 else 0.0
        }
    
    def _analyze_song_representativeness(self, cluster_vectors: np.ndarray, 
                                       track_ids: np.ndarray, cluster_type: str) -> Dict:
        """Analiza qué canciones son más representativas del cluster"""
        if len(cluster_vectors) == 0:
            return {'error': 'Cluster vacío'}
        
        centroid = np.mean(cluster_vectors, axis=0)
        distances_to_centroid = np.linalg.norm(cluster_vectors - centroid, axis=1)
        
        # Canciones más representativas (cercanas al centroide)
        most_representative_indices = np.argsort(distances_to_centroid)[:5]
        least_representative_indices = np.argsort(distances_to_centroid)[-3:]
        
        # Obtener información de las canciones
        most_representative_songs = []
        for idx in most_representative_indices:
            track_id = track_ids[idx]
            song_info = self.loader.get_song_by_track_id(track_id)
            if song_info:
                most_representative_songs.append({
                    'track_name': song_info['track_name'],
                    'artist_name': song_info['artist_name'],
                    'distance_to_centroid': float(distances_to_centroid[idx]),
                    'representativeness_score': float(1.0 / (1.0 + distances_to_centroid[idx]))
                })
        
        least_representative_songs = []
        for idx in least_representative_indices:
            track_id = track_ids[idx]
            song_info = self.loader.get_song_by_track_id(track_id)
            if song_info:
                least_representative_songs.append({
                    'track_name': song_info['track_name'],
                    'artist_name': song_info['artist_name'],
                    'distance_to_centroid': float(distances_to_centroid[idx]),
                    'representativeness_score': float(1.0 / (1.0 + distances_to_centroid[idx]))
                })
        
        return {
            'most_representative_songs': most_representative_songs,
            'least_representative_songs': least_representative_songs,
            'mean_representativeness': float(np.mean(1.0 / (1.0 + distances_to_centroid))),
            'representativeness_std': float(np.std(1.0 / (1.0 + distances_to_centroid)))
        }
    
    def _analyze_cluster_temporal_characteristics(self, track_ids: np.ndarray) -> Dict:
        """Analiza características temporales del cluster si están disponibles"""
        metadata_df = self.loader.get_songs_metadata()
        cluster_metadata = metadata_df[metadata_df['track_id'].isin(track_ids)]
        
        temporal_analysis = {'available': False}
        
        # Análisis de popularidad si está disponible
        if 'track_popularity' in cluster_metadata.columns:
            popularity_scores = cluster_metadata['track_popularity'].dropna()
            if len(popularity_scores) > 0:
                temporal_analysis.update({
                    'available': True,
                    'popularity_stats': {
                        'mean': float(popularity_scores.mean()),
                        'std': float(popularity_scores.std()),
                        'median': float(popularity_scores.median()),
                        'min': float(popularity_scores.min()),
                        'max': float(popularity_scores.max()),
                        'quartile_25': float(popularity_scores.quantile(0.25)),
                        'quartile_75': float(popularity_scores.quantile(0.75))
                    }
                })
        
        return temporal_analysis
    
    def _analyze_semantic_coherence_extended(self, cluster_vectors: np.ndarray) -> Dict:
        """Análisis extendido de coherencia semántica"""
        if len(cluster_vectors) < 2:
            return {'error': 'Cluster insuficiente para análisis de coherencia'}
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Matriz de similitudes completa
        similarity_matrix = cosine_similarity(cluster_vectors)
        
        # Excluir diagonal
        np.fill_diagonal(similarity_matrix, np.nan)
        similarities = similarity_matrix[~np.isnan(similarity_matrix)]
        
        # Análisis de coherencia
        return {
            'mean_pairwise_similarity': float(np.mean(similarities)),
            'std_pairwise_similarity': float(np.std(similarities)),
            'min_pairwise_similarity': float(np.min(similarities)),
            'max_pairwise_similarity': float(np.max(similarities)),
            'coherence_score': float(np.mean(similarities)),
            'coherence_consistency': float(1.0 / (1.0 + np.std(similarities))),
            'high_coherence_pairs_ratio': float(np.sum(similarities > 0.7) / len(similarities)) if len(similarities) > 0 else 0.0
        }
    
    def _analyze_thematic_diversity(self, track_ids: np.ndarray) -> Dict:
        """Analiza la diversidad temática del cluster"""
        metadata_df = self.loader.get_songs_metadata()
        cluster_metadata = metadata_df[metadata_df['track_id'].isin(track_ids)]
        
        diversity_metrics = {}
        
        # Diversidad de géneros
        if 'playlist_genre' in cluster_metadata.columns:
            genre_counts = cluster_metadata['playlist_genre'].value_counts()
            total_songs = len(cluster_metadata)
            
            # Calcular entropía de Shannon para diversidad
            probabilities = genre_counts.values / total_songs
            shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            diversity_metrics['genre_diversity'] = {
                'shannon_entropy': float(shannon_entropy),
                'unique_genres': len(genre_counts),
                'dominant_genre_ratio': float(genre_counts.iloc[0] / total_songs),
                'simpson_index': float(1 - np.sum(probabilities ** 2))
            }
        
        # Diversidad de artistas
        if 'artist_name' in cluster_metadata.columns:
            artist_counts = cluster_metadata['artist_name'].value_counts()
            total_songs = len(cluster_metadata)
            
            probabilities = artist_counts.values / total_songs
            shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            diversity_metrics['artist_diversity'] = {
                'shannon_entropy': float(shannon_entropy),
                'unique_artists': len(artist_counts),
                'max_songs_per_artist': int(artist_counts.iloc[0]),
                'artist_concentration': float(artist_counts.iloc[0] / total_songs)
            }
        
        return diversity_metrics
    
    def _calculate_cluster_quality_score(self, variability: Dict, separability: Dict, base_analysis: Dict) -> Dict:
        """Calcula score de calidad del cluster musical"""
        if 'error' in variability or 'error' in separability:
            return {'error': 'No se puede calcular calidad por errores en análisis previos'}
        
        # Componentes del score de calidad
        compactness = variability.get('compactness_score', 0.5)
        uniformity = variability.get('uniformity_score', 0.5)
        separability_score = separability.get('separability_score', 0.5)
        
        # Normalizar separabilidad (ajustar según rango esperado)
        separability_normalized = min(separability_score / 2.0, 1.0)
        
        # Score compuesto
        quality_score = (0.4 * compactness + 0.3 * uniformity + 0.3 * separability_normalized)
        
        return {
            'overall_quality_score': float(quality_score),
            'components': {
                'compactness': float(compactness),
                'uniformity': float(uniformity), 
                'separability': float(separability_normalized)
            },
            'quality_level': 'high' if quality_score > 0.7 else 'medium' if quality_score > 0.5 else 'low'
        }
    
    def _calculate_semantic_quality_score(self, coherence: Dict, separability: Dict, base_analysis: Dict) -> Dict:
        """Calcula score de calidad del cluster semántico"""
        if 'error' in coherence or 'error' in separability:
            return {'error': 'No se puede calcular calidad por errores en análisis previos'}
        
        # Componentes del score de calidad semántica
        coherence_score = coherence.get('coherence_score', 0.5)
        consistency = coherence.get('coherence_consistency', 0.5)
        separability_score = separability.get('separability_score', 0.5)
        
        # Normalizar valores
        coherence_normalized = max(0, min(coherence_score, 1.0))
        separability_normalized = min(separability_score / 1.0, 1.0)  # Ajustar según rango BERT
        
        # Score compuesto semántico
        semantic_quality = (0.5 * coherence_normalized + 0.3 * consistency + 0.2 * separability_normalized)
        
        return {
            'overall_semantic_quality': float(semantic_quality),
            'components': {
                'coherence': float(coherence_normalized),
                'consistency': float(consistency),
                'separability': float(separability_normalized)
            },
            'semantic_quality_level': 'high' if semantic_quality > 0.6 else 'medium' if semantic_quality > 0.4 else 'low'
        }
    
    def compare_domains_analysis(self) -> Dict:
        """Compara análisis entre dominios musical y semántico"""
        print("🔄 Ejecutando análisis comparativo entre dominios...")
        
        start_time = time.time()
        
        # Análisis de todos los clusters
        musical_analyses = {}
        semantic_analyses = {}
        
        # Clusters musicales
        n_musical = self.config['optimal_configurations']['musical_clustering']['n_clusters']
        for i in range(n_musical):
            print(f"   Analizando cluster musical {i}...")
            musical_analyses[f'cluster_{i}'] = self.analyze_musical_cluster_advanced(i)
        
        # Clusters semánticos
        n_semantic = self.config['optimal_configurations']['semantic_clustering']['n_clusters']
        for i in range(n_semantic):
            print(f"   Analizando cluster semántico {i}...")
            semantic_analyses[f'cluster_{i}'] = self.analyze_semantic_cluster_advanced(i)
        
        # Comparación estadística
        comparison_stats = self._calculate_domain_comparison_stats(musical_analyses, semantic_analyses)
        
        elapsed_time = time.time() - start_time
        
        return {
            'analysis_metadata': {
                'execution_time_seconds': elapsed_time,
                'musical_clusters_analyzed': n_musical,
                'semantic_clusters_analyzed': n_semantic,
                'total_clusters': n_musical + n_semantic
            },
            'musical_domain': {
                'cluster_analyses': musical_analyses,
                'domain_statistics': self._calculate_domain_statistics(musical_analyses, 'musical')
            },
            'semantic_domain': {
                'cluster_analyses': semantic_analyses,
                'domain_statistics': self._calculate_domain_statistics(semantic_analyses, 'semantic')
            },
            'cross_domain_comparison': comparison_stats
        }
    
    def _calculate_domain_comparison_stats(self, musical_analyses: Dict, semantic_analyses: Dict) -> Dict:
        """Calcula estadísticas comparativas entre dominios"""
        # Extraer scores de calidad
        musical_quality_scores = []
        semantic_quality_scores = []
        
        for analysis in musical_analyses.values():
            if 'cluster_quality_score' in analysis and 'error' not in analysis['cluster_quality_score']:
                musical_quality_scores.append(analysis['cluster_quality_score']['overall_quality_score'])
        
        for analysis in semantic_analyses.values():
            if 'semantic_quality_score' in analysis and 'error' not in analysis['semantic_quality_score']:
                semantic_quality_scores.append(analysis['semantic_quality_score']['overall_semantic_quality'])
        
        comparison = {}
        
        if musical_quality_scores and semantic_quality_scores:
            comparison = {
                'quality_comparison': {
                    'musical_mean_quality': float(np.mean(musical_quality_scores)),
                    'semantic_mean_quality': float(np.mean(semantic_quality_scores)),
                    'musical_std_quality': float(np.std(musical_quality_scores)),
                    'semantic_std_quality': float(np.std(semantic_quality_scores)),
                    'quality_difference': float(np.mean(musical_quality_scores) - np.mean(semantic_quality_scores)),
                    'domain_with_higher_quality': 'musical' if np.mean(musical_quality_scores) > np.mean(semantic_quality_scores) else 'semantic'
                }
            }
        
        # Comparación de tamaños de clusters
        musical_sizes = []
        semantic_sizes = []
        
        for analysis in musical_analyses.values():
            if 'cluster_size' in analysis:
                musical_sizes.append(analysis['cluster_size'])
        
        for analysis in semantic_analyses.values():
            if 'cluster_size' in analysis:
                semantic_sizes.append(analysis['cluster_size'])
        
        if musical_sizes and semantic_sizes:
            comparison['size_comparison'] = {
                'musical_mean_size': float(np.mean(musical_sizes)),
                'semantic_mean_size': float(np.mean(semantic_sizes)),
                'musical_size_std': float(np.std(musical_sizes)),
                'semantic_size_std': float(np.std(semantic_sizes)),
                'size_balance_musical': float(np.std(musical_sizes) / np.mean(musical_sizes)) if np.mean(musical_sizes) > 0 else 0,
                'size_balance_semantic': float(np.std(semantic_sizes) / np.mean(semantic_sizes)) if np.mean(semantic_sizes) > 0 else 0
            }
        
        return comparison
    
    def _calculate_domain_statistics(self, domain_analyses: Dict, domain_type: str) -> Dict:
        """Calcula estadísticas agregadas por dominio"""
        cluster_sizes = []
        quality_scores = []
        
        for analysis in domain_analyses.values():
            if 'error' not in analysis:
                cluster_sizes.append(analysis.get('cluster_size', 0))
                
                if domain_type == 'musical' and 'cluster_quality_score' in analysis:
                    if 'error' not in analysis['cluster_quality_score']:
                        quality_scores.append(analysis['cluster_quality_score']['overall_quality_score'])
                elif domain_type == 'semantic' and 'semantic_quality_score' in analysis:
                    if 'error' not in analysis['semantic_quality_score']:
                        quality_scores.append(analysis['semantic_quality_score']['overall_semantic_quality'])
        
        stats = {
            'total_clusters': len(domain_analyses),
            'successful_analyses': sum(1 for a in domain_analyses.values() if 'error' not in a),
            'total_songs': sum(cluster_sizes),
            'mean_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0,
            'std_cluster_size': float(np.std(cluster_sizes)) if cluster_sizes else 0,
            'cluster_balance_coefficient': float(np.std(cluster_sizes) / np.mean(cluster_sizes)) if cluster_sizes and np.mean(cluster_sizes) > 0 else 0
        }
        
        if quality_scores:
            stats['quality_statistics'] = {
                'mean_quality': float(np.mean(quality_scores)),
                'std_quality': float(np.std(quality_scores)),
                'min_quality': float(np.min(quality_scores)),
                'max_quality': float(np.max(quality_scores)),
                'high_quality_clusters': sum(1 for q in quality_scores if q > 0.7),
                'low_quality_clusters': sum(1 for q in quality_scores if q < 0.5)
            }
        
        return stats
    
    def export_comprehensive_report(self, output_file: str = None) -> str:
        """Exporta reporte completo del análisis"""
        if not output_file:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"cluster_analysis_report_{timestamp}.json"
        
        print(f"📊 Generando reporte completo...")
        
        # Ejecutar análisis completo
        comprehensive_analysis = self.compare_domains_analysis()
        
        # Agregar metadatos del sistema
        comprehensive_analysis['system_metadata'] = {
            'dataset_size': self.system_stats['dataset_size'],
            'system_configuration': self.config,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'analysis_version': '1.0'
        }
        
        # Guardar reporte
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Reporte guardado en: {output_file}")
        return output_file


def format_cluster_analysis_output(analysis: Dict, cluster_type: str, cluster_id: int) -> str:
    """Formatea salida del análisis de cluster individual"""
    if 'error' in analysis:
        return f"❌ Error en análisis del cluster {cluster_type} {cluster_id}: {analysis['error']}"
    
    lines = []
    
    # Información básica
    lines.append(f"🎯 ANÁLISIS DEL CLUSTER {cluster_type.upper()} {cluster_id}")
    lines.append(f"   Etiqueta: {analysis['cluster_label']}")
    lines.append(f"   Tamaño: {analysis['cluster_size']} canciones ({analysis['cluster_percentage']:.1f}% del total)")
    lines.append("")
    
    # Interpretación
    lines.append(f"📝 INTERPRETACIÓN:")
    lines.append(f"   {analysis['interpretation']}")
    lines.append("")
    
    # Estadísticas avanzadas si están disponibles
    if 'advanced_statistics' in analysis:
        advanced = analysis['advanced_statistics']
        lines.append(f"📊 ESTADÍSTICAS AVANZADAS:")
        
        # Variabilidad (musical)
        if 'variability_metrics' in advanced:
            var_metrics = advanced['variability_metrics']
            if 'error' not in var_metrics:
                lines.append(f"   🎵 Compacidad: {var_metrics['compactness_score']:.3f}")
                lines.append(f"   🎵 Uniformidad: {var_metrics['uniformity_score']:.3f}")
                lines.append(f"   🎵 Distancia media al centroide: {var_metrics['mean_distance_to_centroid']:.3f}")
        
        # Coherencia (semántico)
        if 'coherence_extended' in advanced:
            coh_metrics = advanced['coherence_extended']
            if 'error' not in coh_metrics:
                lines.append(f"   🧠 Coherencia semántica: {coh_metrics['coherence_score']:.3f}")
                lines.append(f"   🧠 Consistencia: {coh_metrics['coherence_consistency']:.3f}")
                lines.append(f"   🧠 Similitud pairwise media: {coh_metrics['mean_pairwise_similarity']:.3f}")
        
        # Separabilidad
        if 'separability_metrics' in advanced:
            sep_metrics = advanced['separability_metrics']
            if 'error' not in sep_metrics:
                lines.append(f"   📏 Separabilidad: {sep_metrics['separability_score']:.3f}")
                lines.append(f"   📏 Distancia al cluster más cercano: {sep_metrics['nearest_cluster_distance']:.3f}")
        
        lines.append("")
    
    # Score de calidad
    quality_key = 'cluster_quality_score' if cluster_type == 'musical' else 'semantic_quality_score'
    if quality_key in analysis:
        quality = analysis[quality_key]
        if 'error' not in quality:
            score_key = 'overall_quality_score' if cluster_type == 'musical' else 'overall_semantic_quality'
            level_key = 'quality_level' if cluster_type == 'musical' else 'semantic_quality_level'
            
            lines.append(f"⭐ SCORE DE CALIDAD: {quality[score_key]:.3f} ({quality[level_key]})")
            lines.append("")
    
    # Canciones representativas
    if 'advanced_statistics' in analysis and 'representativeness_analysis' in analysis['advanced_statistics']:
        repr_analysis = analysis['advanced_statistics']['representativeness_analysis']
        if 'most_representative_songs' in repr_analysis:
            lines.append("🎵 CANCIONES MÁS REPRESENTATIVAS:")
            for i, song in enumerate(repr_analysis['most_representative_songs'][:3], 1):
                lines.append(f"   {i}. {song['track_name']} - {song['artist_name']} (score: {song['representativeness_score']:.3f})")
            lines.append("")
    
    # Canciones de muestra del análisis base
    if 'sample_songs' in analysis and analysis['sample_songs']:
        lines.append("🎼 CANCIONES DE MUESTRA:")
        for i, song in enumerate(analysis['sample_songs'][:3], 1):
            lines.append(f"   {i}. {song['track_name']} - {song['artist_name']}")
        lines.append("")
    
    return "\n".join(lines)


def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(
        description="Analizador Avanzado de Clusters Musicales y Semánticos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python analyze_clusters.py --all                     # Análisis completo de todos los clusters
  python analyze_clusters.py --musical --cluster_id 3  # Análisis musical específico
  python analyze_clusters.py --semantic --overview     # Overview de clusters semánticos
  python analyze_clusters.py --compare                 # Comparación entre dominios
  python analyze_clusters.py --export_report          # Exportar reporte JSON completo
        """
    )
    
    # Opciones de análisis
    parser.add_argument('--all', action='store_true', help='Análisis completo de todos los clusters')
    parser.add_argument('--musical', action='store_true', help='Análisis de clusters musicales')
    parser.add_argument('--semantic', action='store_true', help='Análisis de clusters semánticos')
    parser.add_argument('--cluster_id', type=int, help='ID específico del cluster a analizar')
    parser.add_argument('--overview', action='store_true', help='Overview general sin detalles')
    parser.add_argument('--compare', action='store_true', help='Comparación entre dominios musical y semántico')
    
    # Opciones de exportación
    parser.add_argument('--export_report', action='store_true', help='Exportar reporte completo en JSON')
    parser.add_argument('--output_file', type=str, help='Archivo de salida para reporte')
    
    # Opciones técnicas
    parser.add_argument('--system_dir', type=str, help='Directorio del sistema (para desarrollo)')
    parser.add_argument('--quiet', action='store_true', help='Modo silencioso')
    
    args = parser.parse_args()
    
    try:
        # Inicializar analizador
        verbose = not args.quiet
        analyzer = AdvancedClusterAnalyzer(args.system_dir, verbose=verbose)
        
        # Exportar reporte completo
        if args.export_report:
            report_file = analyzer.export_comprehensive_report(args.output_file)
            print(f"📄 Reporte completo exportado a: {report_file}")
            return
        
        # Comparación entre dominios
        if args.compare:
            comparison = analyzer.compare_domains_analysis()
            
            print("🔄 COMPARACIÓN ENTRE DOMINIOS MUSICAL Y SEMÁNTICO")
            print("=" * 60)
            
            # Estadísticas de dominio musical
            musical_stats = comparison['musical_domain']['domain_statistics']
            print(f"\n🎵 DOMINIO MUSICAL:")
            print(f"   Clusters analizados: {musical_stats['successful_analyses']}/{musical_stats['total_clusters']}")
            print(f"   Total canciones: {musical_stats['total_songs']}")
            print(f"   Tamaño promedio cluster: {musical_stats['mean_cluster_size']:.1f} ± {musical_stats['std_cluster_size']:.1f}")
            
            if 'quality_statistics' in musical_stats:
                qual_stats = musical_stats['quality_statistics']
                print(f"   Calidad promedio: {qual_stats['mean_quality']:.3f}")
                print(f"   Clusters alta calidad: {qual_stats['high_quality_clusters']}")
            
            # Estadísticas de dominio semántico
            semantic_stats = comparison['semantic_domain']['domain_statistics']
            print(f"\n🧠 DOMINIO SEMÁNTICO:")
            print(f"   Clusters analizados: {semantic_stats['successful_analyses']}/{semantic_stats['total_clusters']}")
            print(f"   Total canciones: {semantic_stats['total_songs']}")
            print(f"   Tamaño promedio cluster: {semantic_stats['mean_cluster_size']:.1f} ± {semantic_stats['std_cluster_size']:.1f}")
            
            if 'quality_statistics' in semantic_stats:
                qual_stats = semantic_stats['quality_statistics']
                print(f"   Calidad promedio: {qual_stats['mean_quality']:.3f}")
                print(f"   Clusters alta calidad: {qual_stats['high_quality_clusters']}")
            
            # Comparación directa
            if 'cross_domain_comparison' in comparison:
                cross_comp = comparison['cross_domain_comparison']
                print(f"\n⚖️  COMPARACIÓN DIRECTA:")
                
                if 'quality_comparison' in cross_comp:
                    qual_comp = cross_comp['quality_comparison']
                    print(f"   Dominio con mayor calidad: {qual_comp['domain_with_higher_quality']}")
                    print(f"   Diferencia de calidad: {qual_comp['quality_difference']:.3f}")
                
                if 'size_comparison' in cross_comp:
                    size_comp = cross_comp['size_comparison']
                    print(f"   Balance tamaños musical: {size_comp['size_balance_musical']:.3f}")
                    print(f"   Balance tamaños semántico: {size_comp['size_balance_semantic']:.3f}")
            
            print(f"\n⏱️  Tiempo de análisis: {comparison['analysis_metadata']['execution_time_seconds']:.1f}s")
            return
        
        # Análisis de cluster específico
        if args.cluster_id is not None:
            if args.musical:
                analysis = analyzer.analyze_musical_cluster_advanced(args.cluster_id)
                print(format_cluster_analysis_output(analysis, 'musical', args.cluster_id))
            elif args.semantic:
                analysis = analyzer.analyze_semantic_cluster_advanced(args.cluster_id)
                print(format_cluster_analysis_output(analysis, 'semantic', args.cluster_id))
            else:
                print("❌ Especifica --musical o --semantic para análisis de cluster específico")
            return
        
        # Overview de clusters
        if args.overview:
            if args.musical:
                print("🎵 OVERVIEW DE CLUSTERS MUSICALES")
                print("=" * 40)
                n_clusters = analyzer.config['optimal_configurations']['musical_clustering']['n_clusters']
                for i in range(n_clusters):
                    analysis = analyzer.explainer.analyze_musical_cluster(i)
                    if 'error' not in analysis:
                        print(f"Cluster {i}: {analysis['cluster_label']} ({analysis['cluster_size']} canciones)")
                        
            elif args.semantic:
                print("🧠 OVERVIEW DE CLUSTERS SEMÁNTICOS") 
                print("=" * 40)
                n_clusters = analyzer.config['optimal_configurations']['semantic_clustering']['n_clusters']
                for i in range(n_clusters):
                    analysis = analyzer.explainer.analyze_semantic_cluster(i)
                    if 'error' not in analysis:
                        print(f"Cluster {i}: {analysis['cluster_label']} ({analysis['cluster_size']} canciones)")
            else:
                print("❌ Especifica --musical o --semantic para overview")
            return
        
        # Análisis completo
        if args.all:
            comparison = analyzer.compare_domains_analysis()
            
            print("🎯 ANÁLISIS COMPLETO DE CLUSTERS")
            print("=" * 50)
            
            # Mostrar análisis de cada cluster musical
            print("\n🎵 CLUSTERS MUSICALES:")
            musical_analyses = comparison['musical_domain']['cluster_analyses']
            for cluster_key, analysis in musical_analyses.items():
                cluster_id = int(cluster_key.split('_')[1])
                print(f"\nCluster Musical {cluster_id}: {analysis.get('cluster_label', 'N/A')}")
                print(f"   Canciones: {analysis.get('cluster_size', 0)}")
                if 'cluster_quality_score' in analysis and 'error' not in analysis['cluster_quality_score']:
                    quality = analysis['cluster_quality_score']['overall_quality_score']
                    level = analysis['cluster_quality_score']['quality_level']
                    print(f"   Calidad: {quality:.3f} ({level})")
            
            # Mostrar análisis de cada cluster semántico
            print("\n🧠 CLUSTERS SEMÁNTICOS:")
            semantic_analyses = comparison['semantic_domain']['cluster_analyses']
            for cluster_key, analysis in semantic_analyses.items():
                cluster_id = int(cluster_key.split('_')[1])
                print(f"\nCluster Semántico {cluster_id}: {analysis.get('cluster_label', 'N/A')}")
                print(f"   Canciones: {analysis.get('cluster_size', 0)}")
                if 'semantic_quality_score' in analysis and 'error' not in analysis['semantic_quality_score']:
                    quality = analysis['semantic_quality_score']['overall_semantic_quality']
                    level = analysis['semantic_quality_score']['semantic_quality_level']
                    print(f"   Calidad: {quality:.3f} ({level})")
            
            # Resumen estadístico
            print(f"\n📊 RESUMEN ESTADÍSTICO:")
            musical_stats = comparison['musical_domain']['domain_statistics']
            semantic_stats = comparison['semantic_domain']['domain_statistics']
            
            print(f"   Total clusters analizados: {musical_stats['total_clusters'] + semantic_stats['total_clusters']}")
            print(f"   Análisis exitosos: {musical_stats['successful_analyses'] + semantic_stats['successful_analyses']}")
            print(f"   Tiempo total: {comparison['analysis_metadata']['execution_time_seconds']:.1f}s")
            
            return
        
        # Si no se especifica nada, mostrar ayuda
        parser.print_help()
        
    except KeyboardInterrupt:
        print("\n👋 Análisis interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()