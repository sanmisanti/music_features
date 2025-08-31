#!/usr/bin/env python3
"""
Sistema de Explicabilidad para Recomendaciones Musicales
======================================================

Genera explicaciones automáticas basadas en análisis de clusters K=10 musical y K=6 semántico
Proporciona descripciones interpretables y justificaciones para recomendaciones híbridas

Uso:
    from explain_recommendations import RecommendationExplainer
    explainer = RecommendationExplainer()
    explanation = explainer.explain_recommendation(input_track_id, recommended_track_id)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import json
from load_system import MusicDataLoader

class RecommendationExplainer:
    """
    Sistema de explicabilidad para recomendaciones musicales
    Genera descripciones automáticas basadas en clustering científicamente validado
    """
    
    def __init__(self, system_dir: str = None):
        """
        Inicializa el sistema de explicabilidad
        
        Args:
            system_dir: Directorio del sistema (None para auto-detectar)
        """
        # Cargar datos del sistema
        self.loader = MusicDataLoader(system_dir)
        
        # Cargar configuración científica validada
        config = self.loader.get_config()
        self.musical_features = config['dataset_info']['musical_features']
        self.musical_config = config['optimal_configurations']['musical_clustering']
        self.semantic_config = config['optimal_configurations']['semantic_clustering']
        
        # Cache para análisis de clusters pre-calculados
        self._cluster_analysis_cache = {}
        
        print("🔍 RecommendationExplainer inicializado")
        print(f"   Clusters musicales: K={self.musical_config['n_clusters']}")
        print(f"   Clusters semánticos: K={self.semantic_config['n_clusters']}")
    
    def analyze_musical_cluster(self, cluster_id: int) -> Dict:
        """
        Analiza las características estadísticas de un cluster musical específico
        
        Args:
            cluster_id: ID del cluster musical (0-9)
            
        Returns:
            Dict con análisis estadístico completo del cluster
        """
        if cluster_id in self._cluster_analysis_cache:
            return self._cluster_analysis_cache[cluster_id]
        
        # Validar cluster_id
        max_cluster = self.musical_config['n_clusters'] - 1
        if not (0 <= cluster_id <= max_cluster):
            raise ValueError(f"cluster_id debe estar en rango 0-{max_cluster}")
        
        # Obtener datos del cluster
        musical_clusters = self.loader.get_musical_clusters()
        musical_vectors = self.loader.get_musical_vectors()
        metadata_df = self.loader.get_songs_metadata()
        track_ids = self.loader.get_track_ids()
        
        # Índices de canciones en este cluster
        cluster_mask = musical_clusters == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            return {'error': f'No hay canciones en cluster musical {cluster_id}'}
        
        # Vectores musicales del cluster
        cluster_vectors = musical_vectors[cluster_mask]
        
        # Análisis estadístico por característica
        feature_stats = {}
        for i, feature_name in enumerate(self.musical_features):
            feature_values = cluster_vectors[:, i]
            
            feature_stats[feature_name] = {
                'mean': float(np.mean(feature_values)),
                'std': float(np.std(feature_values)),
                'min': float(np.min(feature_values)),
                'max': float(np.max(feature_values)),
                'median': float(np.median(feature_values)),
                'q25': float(np.percentile(feature_values, 25)),
                'q75': float(np.percentile(feature_values, 75))
            }
        
        # Identificar características dominantes (valores extremos)
        dominant_features = self._identify_dominant_features(feature_stats)
        
        # Análisis de géneros en el cluster
        cluster_track_ids = track_ids[cluster_mask]
        cluster_metadata = metadata_df[metadata_df['track_id'].isin(cluster_track_ids)]
        
        genre_distribution = {}
        if 'playlist_genre' in cluster_metadata.columns:
            genre_counts = cluster_metadata['playlist_genre'].value_counts()
            genre_distribution = genre_counts.to_dict()
        
        # Generar etiqueta descriptiva automática
        cluster_label = self._generate_musical_cluster_label(feature_stats, dominant_features)
        
        # Análisis completo del cluster
        analysis = {
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_indices),
            'cluster_percentage': (len(cluster_indices) / len(musical_clusters)) * 100,
            'cluster_label': cluster_label,
            'feature_statistics': feature_stats,
            'dominant_features': dominant_features,
            'genre_distribution': genre_distribution,
            'interpretation': self._interpret_musical_cluster(dominant_features, genre_distribution),
            'sample_songs': self._get_cluster_sample_songs(cluster_track_ids, n_samples=5)
        }
        
        # Cache del análisis
        self._cluster_analysis_cache[cluster_id] = analysis
        
        return analysis
    
    def _identify_dominant_features(self, feature_stats: Dict) -> Dict:
        """
        Identifica características dominantes basadas en valores estadísticos
        
        Args:
            feature_stats: Estadísticas por característica
            
        Returns:
            Dict con características dominantes y sus interpretaciones
        """
        dominant = {
            'high_values': [],    # Características con valores altos (>0.7)
            'low_values': [],     # Características con valores bajos (<0.3)
            'variable': [],       # Características con alta variabilidad (std >0.3)
            'stable': []         # Características con baja variabilidad (std <0.1)
        }
        
        for feature, stats in feature_stats.items():
            mean_val = stats['mean']
            std_val = stats['std']
            
            # Características con valores altos (normalizados 0-1)
            if mean_val > 0.7:
                dominant['high_values'].append({
                    'feature': feature,
                    'mean': mean_val,
                    'interpretation': f'Alto {feature}'
                })
            
            # Características con valores bajos
            elif mean_val < 0.3:
                dominant['low_values'].append({
                    'feature': feature,
                    'mean': mean_val,
                    'interpretation': f'Bajo {feature}'
                })
            
            # Alta variabilidad
            if std_val > 0.3:
                dominant['variable'].append({
                    'feature': feature,
                    'std': std_val,
                    'interpretation': f'{feature} variable'
                })
            
            # Baja variabilidad (características estables)
            elif std_val < 0.1:
                dominant['stable'].append({
                    'feature': feature,
                    'std': std_val,
                    'interpretation': f'{feature} consistente'
                })
        
        return dominant
    
    def _generate_musical_cluster_label(self, feature_stats: Dict, dominant_features: Dict) -> str:
        """
        Genera etiqueta descriptiva automática para cluster musical
        
        Args:
            feature_stats: Estadísticas del cluster
            dominant_features: Características dominantes identificadas
            
        Returns:
            String con etiqueta interpretable del cluster
        """
        # Mapeo de características a descriptores interpretables
        feature_descriptors = {
            'energy': {'high': 'Energético', 'low': 'Calmado'},
            'danceability': {'high': 'Bailable', 'low': 'Contemplativo'},
            'valence': {'high': 'Positivo', 'low': 'Melancólico'},
            'acousticness': {'high': 'Acústico', 'low': 'Electrónico'},
            'instrumentalness': {'high': 'Instrumental', 'low': 'Vocal'},
            'loudness': {'high': 'Potente', 'low': 'Suave'},
            'tempo': {'high': 'Rápido', 'low': 'Lento'},
            'speechiness': {'high': 'Hablado', 'low': 'Melódico'}
        }
        
        # Construir etiqueta basada en características dominantes
        label_parts = []
        
        # Procesar características altas
        for feature_info in dominant_features['high_values'][:3]:  # Top 3
            feature = feature_info['feature']
            if feature in feature_descriptors:
                label_parts.append(feature_descriptors[feature]['high'])
        
        # Procesar características bajas importantes
        for feature_info in dominant_features['low_values'][:2]:  # Top 2
            feature = feature_info['feature']
            if feature in feature_descriptors:
                label_parts.append(feature_descriptors[feature]['low'])
        
        # Si no hay características dominantes claras, usar análisis básico
        if not label_parts:
            energy_mean = feature_stats['energy']['mean']
            valence_mean = feature_stats['valence']['mean']
            
            if energy_mean > 0.6:
                label_parts.append('Energético')
            elif energy_mean < 0.4:
                label_parts.append('Relajado')
            
            if valence_mean > 0.6:
                label_parts.append('Positivo')
            elif valence_mean < 0.4:
                label_parts.append('Introspectivo')
        
        # Construir etiqueta final
        if label_parts:
            label = ' y '.join(label_parts[:3])  # Máximo 3 descriptores
        else:
            label = f"Cluster Musical {feature_stats.get('cluster_id', 'N/A')}"
        
        return label
    
    def _interpret_musical_cluster(self, dominant_features: Dict, genre_distribution: Dict) -> str:
        """
        Genera interpretación textual del cluster musical
        
        Args:
            dominant_features: Características dominantes
            genre_distribution: Distribución de géneros
            
        Returns:
            String con interpretación del cluster
        """
        interpretation_parts = []
        
        # Interpretación por características dominantes
        if dominant_features['high_values']:
            high_features = [f['interpretation'] for f in dominant_features['high_values'][:2]]
            interpretation_parts.append(f"Caracterizado por {', '.join(high_features)}")
        
        if dominant_features['low_values']:
            low_features = [f['interpretation'] for f in dominant_features['low_values'][:2]]
            interpretation_parts.append(f"Con tendencia hacia {', '.join(low_features)}")
        
        # Interpretación por géneros predominantes
        if genre_distribution:
            top_genres = sorted(genre_distribution.items(), key=lambda x: x[1], reverse=True)[:2]
            if top_genres:
                genre_names = [genre for genre, _ in top_genres]
                interpretation_parts.append(f"Predominantemente {' y '.join(genre_names)}")
        
        # Interpretación por variabilidad
        if len(dominant_features['variable']) > 3:
            interpretation_parts.append("Con alta diversidad interna")
        elif len(dominant_features['stable']) > 5:
            interpretation_parts.append("Con características muy consistentes")
        
        return '. '.join(interpretation_parts) if interpretation_parts else "Cluster con características balanceadas"
    
    def _get_cluster_sample_songs(self, cluster_track_ids: np.ndarray, n_samples: int = 5) -> List[Dict]:
        """
        Obtiene canciones de muestra representativas del cluster
        
        Args:
            cluster_track_ids: IDs de canciones en el cluster
            n_samples: Número de muestras a retornar
            
        Returns:
            Lista de canciones representativas
        """
        if len(cluster_track_ids) == 0:
            return []
        
        # Seleccionar muestra aleatoria o todas si son pocas
        sample_ids = np.random.choice(
            cluster_track_ids, 
            size=min(n_samples, len(cluster_track_ids)), 
            replace=False
        )
        
        sample_songs = []
        for track_id in sample_ids:
            song_info = self.loader.get_song_by_track_id(track_id)
            if song_info:
                sample_songs.append({
                    'track_name': song_info['track_name'],
                    'artist_name': song_info['artist_name'],
                    'genre': song_info.get('genre', 'unknown')
                })
        
        return sample_songs
    
    def get_all_musical_clusters_analysis(self) -> Dict:
        """
        Genera análisis completo de todos los clusters musicales
        
        Returns:
            Dict con análisis de todos los clusters K=10
        """
        print("📊 Generando análisis completo de clusters musicales...")
        
        all_clusters_analysis = {}
        n_clusters = self.musical_config['n_clusters']
        
        for cluster_id in range(n_clusters):
            print(f"   Analizando cluster musical {cluster_id}...")
            analysis = self.analyze_musical_cluster(cluster_id)
            all_clusters_analysis[f'cluster_{cluster_id}'] = analysis
        
        # Estadísticas globales
        total_songs = sum(analysis['cluster_size'] for analysis in all_clusters_analysis.values())
        
        global_stats = {
            'total_clusters': n_clusters,
            'total_songs': total_songs,
            'average_cluster_size': total_songs / n_clusters,
            'cluster_size_distribution': {
                f'cluster_{i}': all_clusters_analysis[f'cluster_{i}']['cluster_size'] 
                for i in range(n_clusters)
            },
            'silhouette_score': self.musical_config['silhouette_score'],
            'interpretability_score': self.musical_config['interpretability_score']
        }
        
        return {
            'global_statistics': global_stats,
            'cluster_analyses': all_clusters_analysis,
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def analyze_semantic_cluster(self, cluster_id: int) -> Dict:
        """
        Analiza un cluster semántico específico basado en embeddings BERT
        
        Args:
            cluster_id: ID del cluster semántico (0-5)
            
        Returns:
            Dict con análisis temático del cluster semántico
        """
        # Validar cluster_id
        max_cluster = self.semantic_config['n_clusters'] - 1
        if not (0 <= cluster_id <= max_cluster):
            raise ValueError(f"cluster_id debe estar en rango 0-{max_cluster}")
        
        # Obtener datos del cluster
        semantic_clusters = self.loader.get_semantic_clusters()
        semantic_vectors = self.loader.get_semantic_vectors()
        track_ids = self.loader.get_track_ids()
        metadata_df = self.loader.get_songs_metadata()
        
        # Índices de canciones en este cluster
        cluster_mask = semantic_clusters == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            return {'error': f'No hay canciones en cluster semántico {cluster_id}'}
        
        # Vectores semánticos del cluster
        cluster_vectors = semantic_vectors[cluster_mask]
        
        # Calcular centroide del cluster
        cluster_centroid = np.mean(cluster_vectors, axis=0)
        
        # Análisis de coherencia semántica
        coherence_analysis = self._analyze_semantic_coherence(cluster_vectors, cluster_centroid)
        
        # Obtener metadatos de canciones del cluster
        cluster_track_ids = track_ids[cluster_mask]
        cluster_metadata = metadata_df[metadata_df['track_id'].isin(cluster_track_ids)]
        
        # Análisis de géneros y artistas
        genre_distribution = {}
        artist_distribution = {}
        
        if 'playlist_genre' in cluster_metadata.columns:
            genre_counts = cluster_metadata['playlist_genre'].value_counts()
            genre_distribution = genre_counts.to_dict()
        
        if 'artist_name' in cluster_metadata.columns:
            artist_counts = cluster_metadata['artist_name'].value_counts()
            # Solo top artistas (más de 1 canción)
            artist_distribution = {k: v for k, v in artist_counts.items() if v > 1}
        
        # Generar etiqueta temática automática
        semantic_label = self._generate_semantic_cluster_label(
            cluster_id, genre_distribution, artist_distribution, coherence_analysis
        )
        
        # Análisis de características temáticas
        thematic_analysis = self._analyze_thematic_characteristics(
            cluster_metadata, genre_distribution, artist_distribution
        )
        
        return {
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_indices),
            'cluster_percentage': (len(cluster_indices) / len(semantic_clusters)) * 100,
            'cluster_label': semantic_label,
            'coherence_metrics': coherence_analysis,
            'genre_distribution': genre_distribution,
            'artist_distribution': artist_distribution,
            'thematic_analysis': thematic_analysis,
            'interpretation': self._interpret_semantic_cluster(
                semantic_label, thematic_analysis, coherence_analysis
            ),
            'sample_songs': self._get_cluster_sample_songs(cluster_track_ids, n_samples=5)
        }
    
    def _analyze_semantic_coherence(self, cluster_vectors: np.ndarray, centroid: np.ndarray) -> Dict:
        """
        Analiza la coherencia semántica del cluster
        
        Args:
            cluster_vectors: Vectores BERT del cluster
            centroid: Centroide del cluster
            
        Returns:
            Dict con métricas de coherencia semántica
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Similitud promedio al centroide
        centroid_similarities = cosine_similarity(cluster_vectors, centroid.reshape(1, -1)).flatten()
        
        # Similitud intra-cluster (promedio de todas las similitudes par a par)
        if len(cluster_vectors) > 1:
            intra_similarities = cosine_similarity(cluster_vectors)
            # Excluir diagonal (similitud consigo mismo = 1.0)
            np.fill_diagonal(intra_similarities, np.nan)
            intra_cluster_mean = np.nanmean(intra_similarities)
        else:
            intra_cluster_mean = 1.0
        
        # Análisis de dispersión semántica
        semantic_dispersion = np.std(centroid_similarities)
        
        return {
            'centroid_similarity_mean': float(np.mean(centroid_similarities)),
            'centroid_similarity_std': float(np.std(centroid_similarities)),
            'intra_cluster_similarity': float(intra_cluster_mean),
            'semantic_dispersion': float(semantic_dispersion),
            'coherence_score': float(np.mean(centroid_similarities) * (1 - semantic_dispersion)),
            'cluster_tightness': 'high' if semantic_dispersion < 0.1 else 'medium' if semantic_dispersion < 0.2 else 'low'
        }
    
    def _generate_semantic_cluster_label(self, cluster_id: int, genre_distribution: Dict, 
                                       artist_distribution: Dict, coherence_analysis: Dict) -> str:
        """
        Genera etiqueta temática automática para cluster semántico
        
        Args:
            cluster_id: ID del cluster
            genre_distribution: Distribución de géneros
            artist_distribution: Distribución de artistas
            coherence_analysis: Análisis de coherencia
            
        Returns:
            String con etiqueta temática del cluster
        """
        # Estrategia de etiquetado basada en géneros predominantes
        if genre_distribution:
            # Obtener géneros más frecuentes
            top_genres = sorted(genre_distribution.items(), key=lambda x: x[1], reverse=True)[:2]
            
            if len(top_genres) == 1:
                dominant_genre = top_genres[0][0]
                return f"Temática {dominant_genre.title()}"
            
            elif len(top_genres) >= 2:
                genre1, count1 = top_genres[0]
                genre2, count2 = top_genres[1]
                
                # Si un género es muy dominante (>60%)
                total_songs = sum(genre_distribution.values())
                if count1 / total_songs > 0.6:
                    return f"Temática {genre1.title()}"
                else:
                    return f"Fusión {genre1.title()}-{genre2.title()}"
        
        # Si no hay géneros claros, usar artistas dominantes
        if artist_distribution:
            top_artists = sorted(artist_distribution.items(), key=lambda x: x[1], reverse=True)[:2]
            if len(top_artists) >= 1:
                artist_name = top_artists[0][0]
                return f"Estilo {artist_name}"
        
        # Etiquetado por coherencia semántica
        coherence_score = coherence_analysis.get('coherence_score', 0)
        if coherence_score > 0.8:
            return f"Temática Cohesiva {cluster_id}"
        elif coherence_score > 0.6:
            return f"Temática Moderada {cluster_id}"
        else:
            return f"Temática Diversa {cluster_id}"
    
    def _analyze_thematic_characteristics(self, cluster_metadata: pd.DataFrame, 
                                        genre_distribution: Dict, artist_distribution: Dict) -> Dict:
        """
        Analiza características temáticas del cluster
        
        Args:
            cluster_metadata: Metadatos de canciones del cluster
            genre_distribution: Distribución de géneros
            artist_distribution: Distribución de artistas
            
        Returns:
            Dict con análisis temático
        """
        analysis = {}
        
        # Análisis de diversidad de géneros
        if genre_distribution:
            genre_entropy = self._calculate_entropy(list(genre_distribution.values()))
            analysis['genre_diversity'] = {
                'entropy': genre_entropy,
                'diversity_level': 'high' if genre_entropy > 1.5 else 'medium' if genre_entropy > 0.8 else 'low',
                'dominant_genre_percentage': max(genre_distribution.values()) / sum(genre_distribution.values()) * 100,
                'total_genres': len(genre_distribution)
            }
        
        # Análisis de diversidad de artistas
        if artist_distribution:
            artist_entropy = self._calculate_entropy(list(artist_distribution.values()))
            analysis['artist_diversity'] = {
                'entropy': artist_entropy,
                'total_featured_artists': len(artist_distribution),
                'max_songs_per_artist': max(artist_distribution.values()),
                'artist_concentration': 'high' if max(artist_distribution.values()) > 5 else 'medium'
            }
        
        # Análisis temporal (si hay información de fecha)
        if 'track_popularity' in cluster_metadata.columns:
            popularity_stats = {
                'mean_popularity': float(cluster_metadata['track_popularity'].mean()),
                'popularity_std': float(cluster_metadata['track_popularity'].std()),
                'popularity_range': 'high' if cluster_metadata['track_popularity'].mean() > 70 else 'medium' if cluster_metadata['track_popularity'].mean() > 40 else 'low'
            }
            analysis['popularity_profile'] = popularity_stats
        
        return analysis
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """
        Calcula entropía de Shannon para diversidad
        
        Args:
            values: Lista de conteos
            
        Returns:
            Entropía de Shannon
        """
        if not values:
            return 0.0
        
        total = sum(values)
        probabilities = [v / total for v in values]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return entropy
    
    def _interpret_semantic_cluster(self, cluster_label: str, thematic_analysis: Dict, 
                                  coherence_analysis: Dict) -> str:
        """
        Genera interpretación textual del cluster semántico
        
        Args:
            cluster_label: Etiqueta del cluster
            thematic_analysis: Análisis temático
            coherence_analysis: Análisis de coherencia
            
        Returns:
            String con interpretación del cluster
        """
        interpretation_parts = []
        
        # Interpretación base por etiqueta
        interpretation_parts.append(f"Cluster {cluster_label.lower()}")
        
        # Interpretación por coherencia
        coherence_score = coherence_analysis.get('coherence_score', 0)
        if coherence_score > 0.7:
            interpretation_parts.append("con alta coherencia temática")
        elif coherence_score > 0.5:
            interpretation_parts.append("con coherencia temática moderada")
        else:
            interpretation_parts.append("con temática diversa")
        
        # Interpretación por diversidad de géneros
        if 'genre_diversity' in thematic_analysis:
            genre_div = thematic_analysis['genre_diversity']['diversity_level']
            dominant_pct = thematic_analysis['genre_diversity']['dominant_genre_percentage']
            
            if genre_div == 'low':
                interpretation_parts.append(f"dominado por un género principal ({dominant_pct:.1f}%)")
            elif genre_div == 'high':
                interpretation_parts.append("con amplia diversidad de géneros")
        
        # Interpretación por concentración de artistas
        if 'artist_diversity' in thematic_analysis:
            concentration = thematic_analysis['artist_diversity']['artist_concentration']
            if concentration == 'high':
                interpretation_parts.append("con algunos artistas muy representados")
        
        # Interpretación por popularidad
        if 'popularity_profile' in thematic_analysis:
            pop_range = thematic_analysis['popularity_profile']['popularity_range']
            if pop_range == 'high':
                interpretation_parts.append("caracterizado por canciones muy populares")
            elif pop_range == 'low':
                interpretation_parts.append("con tendencia hacia canciones menos mainstream")
        
        return '. '.join(interpretation_parts).capitalize()
    
    def get_all_semantic_clusters_analysis(self) -> Dict:
        """
        Genera análisis completo de todos los clusters semánticos
        
        Returns:
            Dict con análisis de todos los clusters K=6
        """
        print("🧠 Generando análisis completo de clusters semánticos...")
        
        all_clusters_analysis = {}
        n_clusters = self.semantic_config['n_clusters']
        
        for cluster_id in range(n_clusters):
            print(f"   Analizando cluster semántico {cluster_id}...")
            analysis = self.analyze_semantic_cluster(cluster_id)
            all_clusters_analysis[f'cluster_{cluster_id}'] = analysis
        
        # Estadísticas globales semánticas
        total_songs = sum(analysis['cluster_size'] for analysis in all_clusters_analysis.values() if 'error' not in analysis)
        
        global_stats = {
            'total_clusters': n_clusters,
            'total_songs': total_songs,
            'average_cluster_size': total_songs / n_clusters if n_clusters > 0 else 0,
            'cluster_size_distribution': {
                f'cluster_{i}': all_clusters_analysis[f'cluster_{i}'].get('cluster_size', 0)
                for i in range(n_clusters)
            },
            'silhouette_score': self.semantic_config['silhouette_score'],
            'interpretability_score': self.semantic_config['interpretability_score']
        }
        
        return {
            'global_statistics': global_stats,
            'cluster_analyses': all_clusters_analysis,
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def explain_recommendation(self, input_track_id: str, recommended_track_id: str) -> Dict:
        """
        Genera explicación completa en texto natural para una recomendación específica
        
        Args:
            input_track_id: ID de la canción de entrada
            recommended_track_id: ID de la canción recomendada
            
        Returns:
            Dict con explicación completa y estructurada
        """
        # Obtener información de ambas canciones
        input_song = self.loader.get_song_by_track_id(input_track_id)
        recommended_song = self.loader.get_song_by_track_id(recommended_track_id)
        
        if not input_song:
            return {'error': f'Canción de entrada {input_track_id} no encontrada'}
        if not recommended_song:
            return {'error': f'Canción recomendada {recommended_track_id} no encontrada'}
        
        # Obtener análisis de clusters
        input_musical_analysis = self.analyze_musical_cluster(input_song['musical_cluster'])
        input_semantic_analysis = self.analyze_semantic_cluster(input_song['semantic_cluster'])
        
        rec_musical_analysis = self.analyze_musical_cluster(recommended_song['musical_cluster'])
        rec_semantic_analysis = self.analyze_semantic_cluster(recommended_song['semantic_cluster'])
        
        # Calcular similitudes específicas
        similarity_analysis = self._calculate_recommendation_similarity(input_track_id, recommended_track_id)
        
        # Generar explicación estructurada
        explanation = self._generate_natural_explanation(
            input_song, recommended_song,
            input_musical_analysis, input_semantic_analysis,
            rec_musical_analysis, rec_semantic_analysis,
            similarity_analysis
        )
        
        return {
            'input_song': {
                'track_name': input_song['track_name'],
                'artist_name': input_song['artist_name'],
                'musical_cluster': input_song['musical_cluster'],
                'semantic_cluster': input_song['semantic_cluster'],
                'musical_cluster_label': input_musical_analysis['cluster_label'],
                'semantic_cluster_label': input_semantic_analysis['cluster_label']
            },
            'recommended_song': {
                'track_name': recommended_song['track_name'],
                'artist_name': recommended_song['artist_name'],
                'musical_cluster': recommended_song['musical_cluster'],
                'semantic_cluster': recommended_song['semantic_cluster'],
                'musical_cluster_label': rec_musical_analysis['cluster_label'],
                'semantic_cluster_label': rec_semantic_analysis['cluster_label']
            },
            'explanation': explanation,
            'similarity_metrics': similarity_analysis,
            'cluster_relationship': self._analyze_cluster_relationship(
                input_song, recommended_song,
                input_musical_analysis, input_semantic_analysis,
                rec_musical_analysis, rec_semantic_analysis
            )
        }
    
    def _calculate_recommendation_similarity(self, input_track_id: str, recommended_track_id: str) -> Dict:
        """
        Calcula similitudes específicas entre dos canciones
        
        Args:
            input_track_id: ID de la canción de entrada
            recommended_track_id: ID de la canción recomendada
            
        Returns:
            Dict con métricas de similitud detalladas
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Obtener índices
        input_idx = self.loader.get_track_ids().tolist().index(input_track_id)
        rec_idx = self.loader.get_track_ids().tolist().index(recommended_track_id)
        
        # Obtener vectores
        musical_vectors = self.loader.get_musical_vectors()
        semantic_vectors = self.loader.get_semantic_vectors()
        
        # Calcular similitudes
        musical_sim = cosine_similarity(
            musical_vectors[input_idx].reshape(1, -1),
            musical_vectors[rec_idx].reshape(1, -1)
        )[0][0]
        
        semantic_sim = cosine_similarity(
            semantic_vectors[input_idx].reshape(1, -1),
            semantic_vectors[rec_idx].reshape(1, -1)
        )[0][0]
        
        # Similitud híbrida usando pesos del sistema
        config = self.loader.get_config()
        musical_weight = config['recommendation_weights']['musical_weight']
        semantic_weight = config['recommendation_weights']['semantic_weight']
        
        hybrid_sim = (musical_weight * musical_sim + semantic_weight * semantic_sim)
        
        return {
            'musical_similarity': float(musical_sim),
            'semantic_similarity': float(semantic_sim),
            'hybrid_similarity': float(hybrid_sim),
            'similarity_interpretation': self._interpret_similarity_scores(musical_sim, semantic_sim, hybrid_sim)
        }
    
    def _interpret_similarity_scores(self, musical_sim: float, semantic_sim: float, hybrid_sim: float) -> Dict:
        """
        Interpreta los scores de similitud en términos comprensibles
        """
        def similarity_level(score):
            if score >= 0.8:
                return "muy alta"
            elif score >= 0.6:
                return "alta"
            elif score >= 0.4:
                return "moderada"
            elif score >= 0.2:
                return "baja"
            else:
                return "muy baja"
        
        return {
            'musical_level': similarity_level(musical_sim),
            'semantic_level': similarity_level(semantic_sim),
            'hybrid_level': similarity_level(hybrid_sim),
            'dominant_factor': 'musical' if musical_sim > semantic_sim else 'semántica' if semantic_sim > musical_sim else 'equilibrada'
        }
    
    def _analyze_cluster_relationship(self, input_song: Dict, recommended_song: Dict,
                                    input_musical: Dict, input_semantic: Dict,
                                    rec_musical: Dict, rec_semantic: Dict) -> Dict:
        """
        Analiza la relación entre clusters de ambas canciones
        """
        musical_match = input_song['musical_cluster'] == recommended_song['musical_cluster']
        semantic_match = input_song['semantic_cluster'] == recommended_song['semantic_cluster']
        
        relationship_type = "unknown"
        if musical_match and semantic_match:
            relationship_type = "identical_clusters"
        elif musical_match and not semantic_match:
            relationship_type = "musical_match_only"
        elif not musical_match and semantic_match:
            relationship_type = "semantic_match_only"
        else:
            relationship_type = "different_clusters"
        
        return {
            'musical_cluster_match': musical_match,
            'semantic_cluster_match': semantic_match,
            'relationship_type': relationship_type,
            'cluster_distance': {
                'musical': abs(input_song['musical_cluster'] - recommended_song['musical_cluster']),
                'semantic': abs(input_song['semantic_cluster'] - recommended_song['semantic_cluster'])
            }
        }
    
    def _generate_natural_explanation(self, input_song: Dict, recommended_song: Dict,
                                    input_musical: Dict, input_semantic: Dict,
                                    rec_musical: Dict, rec_semantic: Dict,
                                    similarity_analysis: Dict) -> Dict:
        """
        Genera explicación en texto natural comprensible
        """
        # Análisis de relación entre clusters
        cluster_rel = self._analyze_cluster_relationship(
            input_song, recommended_song,
            input_musical, input_semantic,
            rec_musical, rec_semantic
        )
        
        # Componentes de la explicación
        explanation_parts = {
            'main_reason': self._generate_main_reason(cluster_rel, similarity_analysis),
            'musical_explanation': self._generate_musical_explanation(
                input_musical, rec_musical, cluster_rel['musical_cluster_match']
            ),
            'semantic_explanation': self._generate_semantic_explanation(
                input_semantic, rec_semantic, cluster_rel['semantic_cluster_match']
            ),
            'similarity_explanation': self._generate_similarity_explanation(similarity_analysis),
            'full_explanation': ""
        }
        
        # Construir explicación completa
        full_parts = []
        
        if explanation_parts['main_reason']:
            full_parts.append(explanation_parts['main_reason'])
        
        if explanation_parts['musical_explanation']:
            full_parts.append(explanation_parts['musical_explanation'])
        
        if explanation_parts['semantic_explanation']:
            full_parts.append(explanation_parts['semantic_explanation'])
        
        if explanation_parts['similarity_explanation']:
            full_parts.append(explanation_parts['similarity_explanation'])
        
        explanation_parts['full_explanation'] = ' '.join(full_parts)
        
        return explanation_parts
    
    def _generate_main_reason(self, cluster_rel: Dict, similarity_analysis: Dict) -> str:
        """Genera la razón principal de la recomendación"""
        relationship = cluster_rel['relationship_type']
        hybrid_level = similarity_analysis['similarity_interpretation']['hybrid_level']
        dominant = similarity_analysis['similarity_interpretation']['dominant_factor']
        
        if relationship == "identical_clusters":
            return f"Esta recomendación tiene similitud {hybrid_level} porque ambas canciones pertenecen a los mismos clusters musicales y semánticos."
        
        elif relationship == "musical_match_only":
            return f"Recomendación con similitud {hybrid_level} basada principalmente en compatibilidad musical ({dominant})."
        
        elif relationship == "semantic_match_only":
            return f"Recomendación con similitud {hybrid_level} basada principalmente en coherencia temática ({dominant})."
        
        else:
            return f"Recomendación con similitud {hybrid_level} basada en complementariedad entre características musicales y temáticas."
    
    def _generate_musical_explanation(self, input_analysis: Dict, rec_analysis: Dict, same_cluster: bool) -> str:
        """Genera explicación musical específica"""
        if same_cluster:
            cluster_label = input_analysis['cluster_label']
            return f"Musicalmente, ambas canciones pertenecen al cluster '{cluster_label}', compartiendo características sonoras similares."
        else:
            input_label = input_analysis['cluster_label']
            rec_label = rec_analysis['cluster_label']
            return f"Musicalmente, combinan estilos complementarios: '{input_label}' y '{rec_label}'."
    
    def _generate_semantic_explanation(self, input_analysis: Dict, rec_analysis: Dict, same_cluster: bool) -> str:
        """Genera explicación semántica específica"""
        if same_cluster:
            cluster_label = input_analysis['cluster_label']
            return f"Temáticamente, ambas pertenecen al cluster '{cluster_label}', compartiendo contenido lírico similar."
        else:
            input_label = input_analysis['cluster_label']
            rec_label = rec_analysis['cluster_label']
            return f"Temáticamente, conectan diferentes estilos: '{input_label}' con '{rec_label}'."
    
    def _generate_similarity_explanation(self, similarity_analysis: Dict) -> str:
        """Genera explicación basada en métricas de similitud"""
        musical_level = similarity_analysis['similarity_interpretation']['musical_level']
        semantic_level = similarity_analysis['similarity_interpretation']['semantic_level']
        
        return f"La similitud musical es {musical_level} y la similitud semántica es {semantic_level}."
    
    def get_batch_explanations(self, recommendations_result: Dict) -> Dict:
        """
        Genera explicaciones para un conjunto completo de recomendaciones
        
        Args:
            recommendations_result: Resultado del HybridMusicRecommender
            
        Returns:
            Dict con explicaciones para todas las recomendaciones
        """
        if 'input_song' not in recommendations_result or 'recommendations' not in recommendations_result:
            return {'error': 'Formato de recomendaciones inválido'}
        
        input_track_id = recommendations_result['input_song']['track_id']
        recommendations = recommendations_result['recommendations']
        
        explanations = []
        
        print(f"📝 Generando explicaciones para {len(recommendations)} recomendaciones...")
        
        for i, rec in enumerate(recommendations):
            try:
                explanation = self.explain_recommendation(input_track_id, rec['track_id'])
                
                # Agregar información de ranking
                explanation['recommendation_rank'] = rec['rank']
                explanation['hybrid_score'] = rec['scores']['hybrid']
                
                explanations.append(explanation)
                
                if (i + 1) % 3 == 0:
                    print(f"   Procesadas {i + 1}/{len(recommendations)} explicaciones...")
                    
            except Exception as e:
                explanations.append({
                    'recommendation_rank': rec['rank'],
                    'error': f'Error generando explicación: {str(e)}'
                })
        
        return {
            'input_song_info': recommendations_result['input_song'],
            'total_explanations': len(explanations),
            'explanations': explanations,
            'generation_metadata': recommendations_result['metadata']
        }


if __name__ == "__main__":
    # Test básico del sistema de explicabilidad
    print("🧪 Ejecutando test del sistema de explicabilidad...")
    
    try:
        explainer = RecommendationExplainer()
        
        # Test de análisis de cluster musical específico
        print("\n🔍 Analizando cluster musical 0...")
        cluster_0_analysis = explainer.analyze_musical_cluster(0)
        
        if 'error' not in cluster_0_analysis:
            print(f"✅ Análisis exitoso del cluster musical 0:")
            print(f"   Etiqueta: {cluster_0_analysis['cluster_label']}")
            print(f"   Tamaño: {cluster_0_analysis['cluster_size']} canciones")
            print(f"   Interpretación: {cluster_0_analysis['interpretation']}")
            
            # Mostrar características dominantes
            dominant = cluster_0_analysis['dominant_features']
            if dominant['high_values']:
                print(f"   Características altas: {[f['feature'] for f in dominant['high_values']]}")
            
            # Mostrar canciones de muestra
            if cluster_0_analysis['sample_songs']:
                print("   Canciones ejemplo:")
                for song in cluster_0_analysis['sample_songs'][:3]:
                    print(f"     - {song['track_name']} - {song['artist_name']}")
        else:
            print(f"❌ Error en análisis musical: {cluster_0_analysis['error']}")
        
        # Test de análisis de cluster semántico
        print("\n🧠 Analizando cluster semántico 0...")
        semantic_cluster_0_analysis = explainer.analyze_semantic_cluster(0)
        
        if 'error' not in semantic_cluster_0_analysis:
            print(f"✅ Análisis exitoso del cluster semántico 0:")
            print(f"   Etiqueta: {semantic_cluster_0_analysis['cluster_label']}")
            print(f"   Tamaño: {semantic_cluster_0_analysis['cluster_size']} canciones")
            print(f"   Interpretación: {semantic_cluster_0_analysis['interpretation']}")
            print(f"   Coherencia: {semantic_cluster_0_analysis['coherence_metrics']['coherence_score']:.3f}")
        else:
            print(f"❌ Error en análisis semántico: {semantic_cluster_0_analysis['error']}")
        
        # Test de explicación de recomendación específica
        loader = explainer.loader
        track_ids = loader.get_track_ids()
        
        if len(track_ids) >= 2:
            print("\n📝 Generando explicación de recomendación...")
            input_track = track_ids[0]
            rec_track = track_ids[1]
            
            explanation = explainer.explain_recommendation(input_track, rec_track)
            
            if 'error' not in explanation:
                print("✅ Explicación generada exitosamente:")
                print(f"   Entrada: {explanation['input_song']['track_name']} - {explanation['input_song']['artist_name']}")
                print(f"   Recomendada: {explanation['recommended_song']['track_name']} - {explanation['recommended_song']['artist_name']}")
                print(f"   Explicación principal: {explanation['explanation']['main_reason']}")
                print(f"   Similitud híbrida: {explanation['similarity_metrics']['hybrid_similarity']:.3f}")
            else:
                print(f"❌ Error en explicación: {explanation['error']}")
        
        # Test de integración con HybridMusicRecommender
        print("\n🎯 Test de integración con motor de recomendaciones...")
        try:
            from music_recommender import HybridMusicRecommender
            
            recommender = HybridMusicRecommender()
            
            if len(track_ids) > 0:
                # Generar recomendaciones
                recommendations = recommender.recommend(track_ids[0], n_recommendations=3)
                
                if isinstance(recommendations, dict) and 'recommendations' in recommendations:
                    # Generar explicaciones para las recomendaciones
                    batch_explanations = explainer.get_batch_explanations(recommendations)
                    
                    if 'error' not in batch_explanations:
                        print(f"✅ Integración exitosa - {batch_explanations['total_explanations']} explicaciones generadas")
                        
                        # Mostrar primera explicación
                        if batch_explanations['explanations'] and 'error' not in batch_explanations['explanations'][0]:
                            first_exp = batch_explanations['explanations'][0]
                            print(f"   Primera recomendación explicada:")
                            print(f"   {first_exp['recommended_song']['track_name']} - {first_exp['recommended_song']['artist_name']}")
                            print(f"   Explicación: {first_exp['explanation']['full_explanation'][:100]}...")
                    else:
                        print(f"❌ Error en explicaciones batch: {batch_explanations['error']}")
                else:
                    print("❌ Error en recomendaciones base")
            
        except ImportError:
            print("⚠️ HybridMusicRecommender no disponible para test de integración")
        except Exception as e:
            print(f"❌ Error en integración: {e}")
        
        print("\n🏆 Test completo del sistema de explicabilidad finalizado")
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        raise