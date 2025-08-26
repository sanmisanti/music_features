"""
Configuración de Interpretabilidad para Sistema Clustering Multimodal
====================================================================

Define parámetros y umbrales para evaluación de interpretabilidad,
etiquetado automático, y generación de explicaciones multimodales
con prioridad en granularidad explicativa.

Autor: Proyecto FASE 3 - Sistema Clustering Multimodal
Fecha: Agosto 2025
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class InterpretabilitySettings:
    """
    Configuración centralizada para interpretabilidad y explicabilidad.
    
    Define umbrales, parámetros, y métodos para evaluación automática
    de interpretabilidad en clustering multimodal.
    """
    
    def __init__(self):
        """Inicializar configuración de interpretabilidad."""
        
        # Umbrales de calidad
        self.quality_thresholds = {
            'min_silhouette_score': 0.15,  # Mínimo silhouette aceptable
            'min_balance_score': 0.6,      # Mínimo balance distribución
            'min_cluster_size': 3,          # Tamaño mínimo cluster interpretable
            'max_fragmentation_percent': 3.0,  # Máximo % cluster fragmentado
            'max_dominance_percent': 50.0,     # Máximo % cluster dominante
            'min_cross_modal_nmi': 0.2      # Mínimo NMI cross-modal
        }
        
        # Parámetros granularidad
        self.granularity_config = {
            'minimum_k': 5,                # K mínimo para explicabilidad
            'optimal_k_range': (5, 8),    # Rango K óptimo
            'bonus_threshold': 5,          # K mínimo para bonus
            'penalty_factor': 0.8          # Factor penalización K<5
        }
        
        # Configuración etiquetado musical
        self.musical_labeling = {
            'feature_names': [
                'acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                'mode', 'time_signature', 'key'
            ],
            'feature_ranges': {
                'low': (0.0, 0.33),
                'medium': (0.33, 0.67), 
                'high': (0.67, 1.0)
            },
            'tempo_ranges': {
                'slow': (0, 100),
                'moderate': (100, 140),
                'fast': (140, 200),
                'very_fast': (200, 300)
            },
            'descriptive_terms': {
                'energy': {
                    'low': 'Relajado',
                    'medium': 'Moderado',
                    'high': 'Energético'
                },
                'valence': {
                    'low': 'Melancólico',
                    'medium': 'Neutral',
                    'high': 'Alegre'
                },
                'acousticness': {
                    'low': 'Electrónico',
                    'medium': 'Híbrido',
                    'high': 'Acústico'
                },
                'danceability': {
                    'low': 'Contemplativo',
                    'medium': 'Movible',
                    'high': 'Bailable'
                }
            }
        }
        
        # Configuración etiquetado semántico
        self.semantic_labeling = {
            'coherence_threshold': 0.7,    # Umbral coherencia interna
            'min_representative_songs': 3,  # Mínimo canciones representativas
            'max_label_words': 4,          # Máximo palabras por etiqueta
            'common_themes': [
                'Amor y Romance', 'Introspección Personal', 'Celebración y Fiesta',
                'Protesta Social', 'Narrativa Descriptiva', 'Espiritualidad',
                'Nostalgia y Memoria', 'Libertad y Escape', 'Dolor y Pérdida',
                'Amistad y Comunidad', 'Naturaleza y Paisajes', 'Tiempo y Cambio'
            ]
        }
        
        # Configuración correspondencia cross-modal
        self.cross_modal_config = {
            'min_overlap_percent': 30.0,   # Mínimo % overlap para correspondencia
            'strong_correspondence_nmi': 0.4,  # NMI para correspondencia fuerte
            'visualization_min_size': 10,  # Tamaño mínimo para visualización
            'max_correspondences_display': 6  # Máximo correspondencias mostrar
        }
    
    def validate_cluster_interpretability(self, cluster_size: int, 
                                        silhouette_avg: float,
                                        coherence_score: float) -> bool:
        """
        Validar si un cluster cumple criterios de interpretabilidad.
        
        Args:
            cluster_size: Tamaño del cluster
            silhouette_avg: Silhouette promedio del cluster
            coherence_score: Score coherencia interna
            
        Returns:
            True si el cluster es interpretable
        """
        return (
            cluster_size >= self.quality_thresholds['min_cluster_size'] and
            silhouette_avg >= self.quality_thresholds['min_silhouette_score'] and
            coherence_score >= 0.5  # Umbral coherencia interna
        )
    
    def generate_musical_cluster_label(self, cluster_features_mean: np.ndarray,
                                     cluster_features_std: np.ndarray) -> str:
        """
        Generar etiqueta descriptiva para cluster musical.
        
        Args:
            cluster_features_mean: Medias de características del cluster
            cluster_features_std: Desviaciones estándar del cluster
            
        Returns:
            Etiqueta descriptiva del cluster
        """
        if len(cluster_features_mean) != len(self.musical_labeling['feature_names']):
            return "Cluster Musical Indefinido"
        
        # Identificar características dominantes
        dominant_features = []
        feature_names = self.musical_labeling['feature_names']
        
        for i, (feature_name, mean_val) in enumerate(zip(feature_names, cluster_features_mean)):
            if feature_name in self.musical_labeling['descriptive_terms']:
                # Determinar rango
                if mean_val < 0.33:
                    level = 'low'
                elif mean_val < 0.67:
                    level = 'medium'
                else:
                    level = 'high'
                
                # Agregar término descriptivo si es característico
                if cluster_features_std[i] < 0.2:  # Característica consistente
                    descriptor = self.musical_labeling['descriptive_terms'][feature_name][level]
                    dominant_features.append(descriptor)
        
        # Manejar tempo especialmente
        tempo_idx = feature_names.index('tempo') if 'tempo' in feature_names else -1
        if tempo_idx >= 0:
            tempo_mean = cluster_features_mean[tempo_idx] * 200  # Desnormalizar tempo aproximado
            if tempo_mean < 100:
                dominant_features.append("Tempo Lento")
            elif tempo_mean > 150:
                dominant_features.append("Tempo Rápido")
        
        # Construir etiqueta
        if len(dominant_features) == 0:
            return "Cluster Musical Genérico"
        elif len(dominant_features) <= 3:
            return " - ".join(dominant_features[:3])
        else:
            return f"{dominant_features[0]} - {dominant_features[1]} - Complejo"
    
    def generate_semantic_cluster_label(self, representative_embeddings: np.ndarray,
                                      cluster_coherence: float) -> str:
        """
        Generar etiqueta descriptiva para cluster semántico.
        
        Args:
            representative_embeddings: Embeddings representativos del cluster
            cluster_coherence: Score coherencia del cluster
            
        Returns:
            Etiqueta descriptiva del cluster
        """
        # Para esta implementación inicial, usar coherencia para determinar tipo
        if cluster_coherence > 0.8:
            base_label = "Temática Muy Coherente"
        elif cluster_coherence > 0.6:
            base_label = "Temática Coherente" 
        else:
            base_label = "Temática Diversa"
        
        # En implementación futura: análisis de embeddings para identificar temas
        # Por ahora, usar placeholder basado en coherencia
        cluster_id = hash(representative_embeddings.tobytes()) % len(self.semantic_labeling['common_themes'])
        theme = self.semantic_labeling['common_themes'][cluster_id]
        
        return f"{theme} ({base_label})"
    
    def evaluate_cross_modal_correspondence(self, musical_labels: np.ndarray,
                                          semantic_labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluar correspondencia entre clusters musicales y semánticos.
        
        Args:
            musical_labels: Etiquetas clustering musical
            semantic_labels: Etiquetas clustering semántico
            
        Returns:
            Dict con análisis de correspondencia
        """
        if len(musical_labels) != len(semantic_labels):
            return {'error': 'Arrays de etiquetas tienen longitudes diferentes'}
        
        # Matriz de contingencia
        unique_musical = np.unique(musical_labels[musical_labels != -1])
        unique_semantic = np.unique(semantic_labels[semantic_labels != -1])
        
        correspondences = []
        
        for m_label in unique_musical:
            for s_label in unique_semantic:
                # Contar co-ocurrencias
                overlap = np.sum((musical_labels == m_label) & (semantic_labels == s_label))
                total_musical = np.sum(musical_labels == m_label)
                total_semantic = np.sum(semantic_labels == s_label)
                
                if total_musical > 0 and total_semantic > 0:
                    overlap_percent = (overlap / min(total_musical, total_semantic)) * 100
                    
                    if overlap_percent >= self.cross_modal_config['min_overlap_percent']:
                        correspondences.append({
                            'musical_cluster': int(m_label),
                            'semantic_cluster': int(s_label),
                            'overlap_count': int(overlap),
                            'overlap_percent': overlap_percent,
                            'strength': 'strong' if overlap_percent > 50 else 'moderate'
                        })
        
        # Ordenar por strength
        correspondences.sort(key=lambda x: x['overlap_percent'], reverse=True)
        
        return {
            'correspondences': correspondences[:self.cross_modal_config['max_correspondences_display']],
            'total_correspondences': len(correspondences),
            'strong_correspondences': len([c for c in correspondences if c['strength'] == 'strong'])
        }
    
    def get_interpretability_summary(self, clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar resumen de interpretabilidad para resultados de clustering.
        
        Args:
            clustering_results: Resultados completos de clustering
            
        Returns:
            Resumen de interpretabilidad
        """
        summary = {
            'overall_interpretability': 'unknown',
            'interpretable_clusters_count': 0,
            'total_clusters': 0,
            'granularity_assessment': 'insufficient',
            'recommendations': []
        }
        
        # Evaluar granularidad
        n_clusters = clustering_results.get('n_clusters', 0)
        if n_clusters >= self.granularity_config['minimum_k']:
            summary['granularity_assessment'] = 'adequate'
            if n_clusters in range(*self.granularity_config['optimal_k_range']):
                summary['granularity_assessment'] = 'optimal'
        
        # Evaluar balance
        balance_score = clustering_results.get('balance_distribution_score', 0)
        if balance_score < self.quality_thresholds['min_balance_score']:
            summary['recommendations'].append('Mejorar balance de distribución de clusters')
        
        # Evaluar calidad técnica
        silhouette_score = clustering_results.get('silhouette_score', -1)
        if silhouette_score < self.quality_thresholds['min_silhouette_score']:
            summary['recommendations'].append('Mejorar cohesión intra-cluster')
        
        # Determinar interpretabilidad general
        if (balance_score >= 0.6 and 
            silhouette_score >= 0.15 and 
            n_clusters >= 5):
            summary['overall_interpretability'] = 'good'
        elif (balance_score >= 0.4 and 
              silhouette_score >= 0.1 and 
              n_clusters >= 3):
            summary['overall_interpretability'] = 'moderate'
        else:
            summary['overall_interpretability'] = 'poor'
        
        summary['total_clusters'] = n_clusters
        summary['interpretable_clusters_count'] = max(0, n_clusters - 1)  # Estimación conservadora
        
        return summary


# Instancia global de configuración de interpretabilidad
interpretability_settings = InterpretabilitySettings()