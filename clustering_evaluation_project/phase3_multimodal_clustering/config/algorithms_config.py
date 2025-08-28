#!/usr/bin/env python3
"""
Configuración algorítmica especializada para clustering multimodal FASE 3.
Optimizada para espacios vectoriales de diferentes dimensionalidades.
"""

from typing import Dict, List, Any
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

class AlgorithmConfig:
    """Configuración de algoritmos especializados por dominio."""
    
    def __init__(self):
        """Inicializar configuraciones optimizadas por dimensionalidad."""
        self.musical_algorithms = self._get_musical_config()
        self.semantic_algorithms = self._get_semantic_config()
        
    def _get_musical_config(self) -> Dict[str, Dict]:
        """Configuración optimizada para espacio musical 12D."""
        return {
            'hierarchical_ward': {
                'algorithm_class': AgglomerativeClustering,
                'base_params': {
                    'linkage': 'ward',
                    'metric': 'euclidean'
                },
                'k_range': [5, 6, 7, 8, 9, 10],
                'description': 'Hierarchical Ward - Óptimo para características normalizadas'
            },
            'hierarchical_complete': {
                'algorithm_class': AgglomerativeClustering,
                'base_params': {
                    'linkage': 'complete',
                    'metric': 'euclidean'
                },
                'k_range': [5, 6, 7, 8, 9, 10],
                'description': 'Hierarchical Complete - Clusters compactos'
            },
            'hierarchical_average': {
                'algorithm_class': AgglomerativeClustering,
                'base_params': {
                    'linkage': 'average',
                    'metric': 'euclidean'
                },
                'k_range': [5, 6, 7, 8, 9, 10],
                'description': 'Hierarchical Average - Balance compactness/separación'
            },
            'kmeans_plus': {
                'algorithm_class': KMeans,
                'base_params': {
                    'init': 'k-means++',
                    'n_init': 20,
                    'max_iter': 300,
                    'random_state': 42
                },
                'k_range': [5, 6, 7, 8, 9, 10],
                'description': 'K-Means++ - Inicialización inteligente'
            },
            'gmm_full': {
                'algorithm_class': GaussianMixture,
                'base_params': {
                    'covariance_type': 'full',
                    'n_init': 5,
                    'max_iter': 200,
                    'random_state': 42
                },
                'k_range': [5, 6, 7, 8, 9, 10],
                'description': 'GMM Full Covariance - Clusters elípticos'
            },
            'dbscan_euclidean': {
                'algorithm_class': DBSCAN,
                'base_params': {
                    'metric': 'euclidean',
                    'min_samples': 50  # ~0.6% de 7811 canciones
                },
                'eps_range': [0.5, 0.7, 1.0, 1.2, 1.5],
                'description': 'DBSCAN Euclidiano - Clusters basados en densidad'
            }
        }
    
    def _get_semantic_config(self) -> Dict[str, Dict]:
        """Configuración especializada para espacio semántico 384D."""
        return {
            'hierarchical_ward': {
                'algorithm_class': AgglomerativeClustering,
                'base_params': {
                    'linkage': 'ward',
                    'metric': 'euclidean'  # Ward requiere euclidean
                },
                'k_range': [5, 6, 7, 8],
                'description': 'Hierarchical Ward - Adaptado para alta dimensionalidad'
            },
            'hierarchical_average': {
                'algorithm_class': AgglomerativeClustering,
                'base_params': {
                    'linkage': 'average',
                    'metric': 'cosine'
                },
                'k_range': [5, 6, 7, 8],
                'description': 'Hierarchical Average Cosine - Óptimo para embeddings'
            },
            'kmeans_plus': {
                'algorithm_class': KMeans,
                'base_params': {
                    'init': 'k-means++',
                    'n_init': 10,  # Reducido para 384D
                    'max_iter': 200,  # Reducido por estabilidad
                    'random_state': 42
                },
                'k_range': [5, 6, 7, 8],
                'description': 'K-Means++ Optimized - Estabilidad numérica 384D'
            },
            'gmm_tied': {
                'algorithm_class': GaussianMixture,
                'base_params': {
                    'covariance_type': 'tied',  # Compartida para alta dim
                    'n_init': 3,  # Reducido por complejidad
                    'max_iter': 100,
                    'random_state': 42
                },
                'k_range': [5, 6, 7, 8],
                'description': 'GMM Tied Covariance - Eficiente para 384D'
            },
            'dbscan_cosine': {
                'algorithm_class': DBSCAN,
                'base_params': {
                    'metric': 'cosine',
                    'min_samples': 30  # Ajustado para alta dimensionalidad
                },
                'eps_range': [0.1, 0.15, 0.2, 0.25, 0.3],
                'description': 'DBSCAN Cosine - Especializado para embeddings BERT'
            }
        }
    
    def get_algorithm_instance(self, domain: str, algorithm_name: str, 
                             k: int = None, eps: float = None) -> Any:
        """
        Crear instancia de algoritmo configurada.
        
        Args:
            domain: 'musical' o 'semantic'
            algorithm_name: Nombre del algoritmo
            k: Número de clusters (si aplica)
            eps: Parámetro epsilon para DBSCAN (si aplica)
            
        Returns:
            Instancia configurada del algoritmo
        """
        config_dict = self.musical_algorithms if domain == 'musical' else self.semantic_algorithms
        
        if algorithm_name not in config_dict:
            raise ValueError(f"Algoritmo {algorithm_name} no disponible para dominio {domain}")
        
        algorithm_config = config_dict[algorithm_name]
        algorithm_class = algorithm_config['algorithm_class']
        base_params = algorithm_config['base_params'].copy()
        
        # Configurar parámetros específicos
        if algorithm_name.startswith('dbscan'):
            if eps is not None:
                base_params['eps'] = eps
        else:
            if k is not None:
                if algorithm_name.startswith('gmm'):
                    base_params['n_components'] = k
                else:
                    base_params['n_clusters'] = k
        
        return algorithm_class(**base_params)
    
    def get_algorithm_configs(self, domain: str) -> Dict[str, Dict]:
        """Obtener todas las configuraciones para un dominio."""
        return self.musical_algorithms if domain == 'musical' else self.semantic_algorithms
    
    def get_total_configurations(self) -> int:
        """Calcular número total de configuraciones a evaluar."""
        total = 0
        
        for domain_config in [self.musical_algorithms, self.semantic_algorithms]:
            for algorithm_name, config in domain_config.items():
                if 'k_range' in config:
                    total += len(config['k_range'])
                elif 'eps_range' in config:
                    total += len(config['eps_range'])
        
        return total

# Instancia global de configuración
algorithm_config = AlgorithmConfig()