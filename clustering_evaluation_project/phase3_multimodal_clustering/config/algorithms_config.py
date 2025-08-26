"""
Configuración Especializada de Algoritmos para Clustering Multimodal
====================================================================

Implementa configuraciones específicas por dimensionalidad para optimizar
performance computacional y calidad de clustering en espacios vectoriales
de diferentes características estructurales.

Autor: Proyecto FASE 3 - Sistema Clustering Multimodal
Fecha: Agosto 2025
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Any, Tuple


class AlgorithmsConfig:
    """
    Configuración especializada de algoritmos de clustering por dominio vectorial.
    
    Implementa parámetros optimizados para:
    - Espacio Musical (12D): Configuraciones standard con énfasis en interpretabilidad
    - Espacio Semántico (384D): Optimizaciones para alta dimensionalidad
    """
    
    def __init__(self):
        """Inicializar configuraciones por dominio vectorial."""
        self.musical_k_range = list(range(5, 11))  # [5, 6, 7, 8, 9, 10]
        self.semantic_k_range = list(range(5, 9))  # [5, 6, 7, 8]
        
        # Semillas determinísticas para reproducibilidad
        self.random_state = 42
        
        # Configuraciones especializadas por dominio
        self._setup_musical_configs()
        self._setup_semantic_configs()
    
    def _setup_musical_configs(self) -> None:
        """Configurar algoritmos para espacio musical (12D)."""
        self.musical_algorithms = {
            'hierarchical_ward': {
                'class': AgglomerativeClustering,
                'params': {
                    'linkage': 'ward',
                    'n_clusters': None  # Se establecerá dinámicamente
                },
                'description': 'Hierarchical Ward - Minimización varianza intra-cluster'
            },
            'hierarchical_complete': {
                'class': AgglomerativeClustering,
                'params': {
                    'linkage': 'complete',
                    'n_clusters': None,
                    'metric': 'euclidean'
                },
                'description': 'Hierarchical Complete - Máxima distancia intra-cluster'
            },
            'hierarchical_average': {
                'class': AgglomerativeClustering,
                'params': {
                    'linkage': 'average',
                    'n_clusters': None,
                    'metric': 'euclidean'
                },
                'description': 'Hierarchical Average - Distancia promedio intra-cluster'
            },
            'kmeans_plus': {
                'class': KMeans,
                'params': {
                    'n_clusters': None,
                    'init': 'k-means++',
                    'random_state': self.random_state,
                    'n_init': 20,
                    'max_iter': 500
                },
                'description': 'K-Means++ - Inicialización inteligente centroides'
            },
            'gmm_full': {
                'class': GaussianMixture,
                'params': {
                    'n_components': None,
                    'covariance_type': 'full',
                    'random_state': self.random_state,
                    'n_init': 10,
                    'max_iter': 200
                },
                'description': 'GMM Full Covariance - Modelado gaussiano completo'
            },
            'dbscan': {
                'class': DBSCAN,
                'params': {
                    'eps': 0.5,  # Se optimizará dinámicamente
                    'min_samples': 20,
                    'metric': 'euclidean'
                },
                'description': 'DBSCAN - Clustering basado en densidad',
                'requires_eps_optimization': True
            }
        }
    
    def _setup_semantic_configs(self) -> None:
        """Configurar algoritmos para espacio semántico (384D)."""
        self.semantic_algorithms = {
            'hierarchical_ward': {
                'class': AgglomerativeClustering,
                'params': {
                    'linkage': 'ward',
                    'n_clusters': None
                },
                'description': 'Hierarchical Ward - Optimizado alta dimensionalidad'
            },
            'hierarchical_average': {
                'class': AgglomerativeClustering,
                'params': {
                    'linkage': 'average',
                    'n_clusters': None,
                    'metric': 'cosine'  # Métrica coseno para embeddings
                },
                'description': 'Hierarchical Average - Métrica coseno semántica'
            },
            'kmeans_plus': {
                'class': KMeans,
                'params': {
                    'n_clusters': None,
                    'init': 'k-means++',
                    'random_state': self.random_state,
                    'n_init': 15,  # Reducido para alta dimensionalidad
                    'max_iter': 300
                },
                'description': 'K-Means++ - Optimizado espacios semánticos'
            },
            'gmm_tied': {
                'class': GaussianMixture,
                'params': {
                    'n_components': None,
                    'covariance_type': 'tied',  # Tied para estabilidad 384D
                    'random_state': self.random_state,
                    'n_init': 5,  # Reducido por complejidad computacional
                    'max_iter': 150
                },
                'description': 'GMM Tied Covariance - Estabilidad alta dimensionalidad'
            },
            'dbscan_cosine': {
                'class': DBSCAN,
                'params': {
                    'eps': 0.3,  # Eps menor para coseno
                    'min_samples': 15,
                    'metric': 'cosine'
                },
                'description': 'DBSCAN Cosine - Optimizado embeddings BERT',
                'requires_eps_optimization': True
            }
        }
    
    def get_algorithm_configs(self, domain: str) -> Dict[str, Dict[str, Any]]:
        """
        Obtener configuraciones de algoritmos para dominio específico.
        
        Args:
            domain: 'musical' o 'semantic'
            
        Returns:
            Dict con configuraciones de algoritmos
            
        Raises:
            ValueError: Si el dominio no es válido
        """
        if domain == 'musical':
            return self.musical_algorithms.copy()
        elif domain == 'semantic':
            return self.semantic_algorithms.copy()
        else:
            raise ValueError(f"Dominio inválido: {domain}. Usar 'musical' o 'semantic'.")
    
    def get_k_range(self, domain: str) -> List[int]:
        """
        Obtener rango de valores K para dominio específico.
        
        Args:
            domain: 'musical' o 'semantic'
            
        Returns:
            Lista de valores K a evaluar
        """
        if domain == 'musical':
            return self.musical_k_range.copy()
        elif domain == 'semantic':
            return self.semantic_k_range.copy()
        else:
            raise ValueError(f"Dominio inválido: {domain}. Usar 'musical' o 'semantic'.")
    
    def create_algorithm_instance(self, domain: str, algorithm_name: str, k: int) -> Any:
        """
        Crear instancia configurada de algoritmo específico.
        
        Args:
            domain: 'musical' o 'semantic'
            algorithm_name: Nombre del algoritmo
            k: Número de clusters
            
        Returns:
            Instancia configurada del algoritmo
        """
        configs = self.get_algorithm_configs(domain)
        
        if algorithm_name not in configs:
            raise ValueError(f"Algoritmo {algorithm_name} no disponible para dominio {domain}")
        
        config = configs[algorithm_name]
        params = config['params'].copy()
        
        # Establecer parámetro K según tipo de algoritmo
        if 'n_clusters' in params:
            params['n_clusters'] = k
        elif 'n_components' in params:
            params['n_components'] = k
        
        # DBSCAN no usa K, requiere optimización eps
        if algorithm_name.startswith('dbscan'):
            # DBSCAN se manejará con optimización eps especializada
            pass
        
        return config['class'](**params)
    
    def get_eps_optimization_range(self, domain: str) -> np.ndarray:
        """
        Obtener rango de valores eps para optimización DBSCAN.
        
        Args:
            domain: 'musical' o 'semantic'
            
        Returns:
            Array numpy con valores eps a evaluar
        """
        if domain == 'musical':
            # Rango eps para espacio euclidiano 12D
            return np.arange(0.3, 1.5, 0.1)
        elif domain == 'semantic':
            # Rango eps para espacio coseno 384D
            return np.arange(0.1, 0.6, 0.05)
        else:
            raise ValueError(f"Dominio inválido: {domain}")
    
    def get_experiment_matrix_size(self) -> Tuple[int, int, int]:
        """
        Calcular tamaño de matriz experimental total.
        
        Returns:
            Tuple (experimentos_musical, experimentos_semantic, total)
        """
        # Algoritmos que usan K (excluyendo DBSCAN)
        musical_k_algorithms = len([alg for alg in self.musical_algorithms.keys() 
                                   if not alg.startswith('dbscan')])
        semantic_k_algorithms = len([alg for alg in self.semantic_algorithms.keys() 
                                    if not alg.startswith('dbscan')])
        
        # Experimentos con K
        musical_k_experiments = musical_k_algorithms * len(self.musical_k_range)
        semantic_k_experiments = semantic_k_algorithms * len(self.semantic_k_range)
        
        # Experimentos DBSCAN (1 por dominio, eps se optimiza internamente)
        dbscan_experiments = 2  # 1 musical + 1 semántico
        
        total_experiments = musical_k_experiments + semantic_k_experiments + dbscan_experiments
        
        return (musical_k_experiments, semantic_k_experiments, total_experiments)
    
    def get_algorithm_description(self, domain: str, algorithm_name: str) -> str:
        """
        Obtener descripción técnica de algoritmo específico.
        
        Args:
            domain: 'musical' o 'semantic'
            algorithm_name: Nombre del algoritmo
            
        Returns:
            Descripción técnica del algoritmo
        """
        configs = self.get_algorithm_configs(domain)
        return configs.get(algorithm_name, {}).get('description', 'Descripción no disponible')


# Instancia global de configuración
algorithms_config = AlgorithmsConfig()