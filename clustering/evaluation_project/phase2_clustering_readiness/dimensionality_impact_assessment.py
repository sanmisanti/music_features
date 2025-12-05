#!/usr/bin/env python3
"""
Dimensionality Impact Assessment Module - FASE 2
===============================================

Evaluación sistemática de efectos dimensionales en clustering readiness,
incluyendo análisis de maldición de dimensionalidad, estructura PCA,
y separabilidad de clusters en espacios de alta y baja dimensión.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

class DimensionalityAssessment:
    """
    Evaluador especializado para análisis de efectos dimensionales
    en clustering readiness y calidad de agrupamiento.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Inicializar evaluador de dimensionalidad.
        
        Args:
            logger: Logger configurado para trazabilidad
        """
        self.logger = logger
        self.random_state = 42
        
        self.logger.info("DimensionalityAssessment inicializado")
    
    def assess_dimensionality_impact(self, semantic_embeddings: np.ndarray, 
                                   musical_features: np.ndarray) -> Dict[str, Any]:
        """
        Ejecutar evaluación comprehensive de efectos dimensionales
        en clustering readiness.
        
        Args:
            semantic_embeddings: Embeddings BERT 384D
            musical_features: Características musicales 12D normalizadas
            
        Returns:
            Análisis completo de efectos dimensionales
        """
        self.logger.info("INICIANDO EVALUACIÓN DE EFECTOS DIMENSIONALES")
        
        results = {
            'semantic_analysis': self._analyze_high_dimensional_space(semantic_embeddings),
            'musical_analysis': self._analyze_low_dimensional_space(musical_features),
            'comparative_analysis': {},
            'curse_of_dimensionality': {},
            'clustering_quality_prediction': {}
        }
        
        # Análisis comparativo
        results['comparative_analysis'] = self._compare_dimensional_characteristics(
            semantic_embeddings, musical_features
        )
        
        # Análisis maldición de dimensionalidad
        results['curse_of_dimensionality'] = self._analyze_curse_of_dimensionality(
            semantic_embeddings, musical_features
        )
        
        # Predicción calidad clustering
        results['clustering_quality_prediction'] = self._predict_clustering_quality(
            results['semantic_analysis'], results['musical_analysis']
        )
        
        self.logger.info("EVALUACIÓN DE EFECTOS DIMENSIONALES COMPLETADA")
        
        return results
    
    def _analyze_high_dimensional_space(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Análisis especializado para espacio de alta dimensión (384D semántico).
        
        Args:
            data: Datos de alta dimensión
            
        Returns:
            Análisis completo del espacio de alta dimensión
        """
        self.logger.info(f"Analizando espacio alta dimensión: {data.shape}")
        
        analysis = {
            'dimensionality': data.shape[1],
            'sample_size': data.shape[0],
            'pca_analysis': self._perform_pca_analysis(data),
            'distance_concentration': self._analyze_distance_concentration(data),
            'intrinsic_dimensionality': self._estimate_intrinsic_dimensionality(data),
            'volume_concentration': self._analyze_volume_concentration(data),
            'separability_metrics': self._calculate_separability_metrics(data)
        }
        
        return analysis
    
    def _analyze_low_dimensional_space(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Análisis especializado para espacio de baja dimensión (12D musical).
        
        Args:
            data: Datos de baja dimensión
            
        Returns:
            Análisis completo del espacio de baja dimensión
        """
        self.logger.info(f"Analizando espacio baja dimensión: {data.shape}")
        
        analysis = {
            'dimensionality': data.shape[1],
            'sample_size': data.shape[0],
            'pca_analysis': self._perform_pca_analysis(data),
            'distance_distribution': self._analyze_distance_distribution(data),
            'feature_correlations': self._analyze_feature_correlations(data),
            'clustering_structure': self._analyze_natural_clustering_structure(data),
            'separability_metrics': self._calculate_separability_metrics(data)
        }
        
        return analysis
    
    def _perform_pca_analysis(self, data: np.ndarray, max_components: int = 50) -> Dict[str, Any]:
        """
        Análisis de componentes principales para caracterizar estructura dimensional.
        
        Args:
            data: Datos para análisis PCA
            max_components: Máximo número de componentes a analizar
            
        Returns:
            Análisis PCA completo
        """
        n_samples, n_features = data.shape
        n_components = min(max_components, n_features, n_samples - 1)
        
        # Aplicar PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit(data)
        
        # Calcular métricas clave
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Encontrar número de componentes para diferentes niveles de varianza
        components_80 = np.where(cumulative_variance >= 0.80)[0]
        components_90 = np.where(cumulative_variance >= 0.90)[0]
        components_95 = np.where(cumulative_variance >= 0.95)[0]
        
        pca_analysis = {
            'n_components_analyzed': n_components,
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'components_for_80_variance': int(components_80[0]) + 1 if len(components_80) > 0 else n_components,
            'components_for_90_variance': int(components_90[0]) + 1 if len(components_90) > 0 else n_components,
            'components_for_95_variance': int(components_95[0]) + 1 if len(components_95) > 0 else n_components,
            'effective_dimensionality': self._calculate_effective_dimensionality(explained_variance_ratio),
            'dimensionality_ratio': self._calculate_effective_dimensionality(explained_variance_ratio) / n_features
        }
        
        return pca_analysis
    
    def _analyze_distance_concentration(self, data: np.ndarray, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Analizar concentración de distancias como indicador de maldición dimensionalidad.
        
        En espacios de alta dimensión, las distancias tienden a concentrarse,
        reduciendo la efectividad de métricas de similitud.
        """
        self.logger.debug("Analizando concentración de distancias")
        
        # Muestrear para eficiencia computacional
        if data.shape[0] > sample_size:
            indices = np.random.choice(data.shape[0], sample_size, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
        
        # Calcular todas las distancias pairwise
        distances = pdist(sample_data, metric='euclidean')
        
        # Métricas de concentración
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Coeficiente de variación como medida de concentración
        # Valores bajos indican alta concentración (problema en alta dimensión)
        concentration_coefficient = std_distance / mean_distance if mean_distance > 0 else 0
        
        # Ratio entre distancias máxima y mínima
        distance_ratio = max_distance / min_distance if min_distance > 0 else float('inf')
        
        concentration_analysis = {
            'mean_distance': float(mean_distance),
            'std_distance': float(std_distance),
            'min_distance': float(min_distance),
            'max_distance': float(max_distance),
            'concentration_coefficient': float(concentration_coefficient),
            'distance_ratio': float(distance_ratio),
            'concentration_severity': self._interpret_concentration_severity(concentration_coefficient),
            'sample_size_used': sample_data.shape[0]
        }
        
        return concentration_analysis
    
    def _estimate_intrinsic_dimensionality(self, data: np.ndarray, k: int = 10) -> Dict[str, Any]:
        """
        Estimar dimensionalidad intrínseca del dataset usando múltiples métodos.
        
        La dimensionalidad intrínseca puede ser menor que la dimensionalidad
        del espacio embedding, especialmente en datos de alta dimensión.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Método 1: Análisis de varianza explicada por PCA
        pca = PCA(random_state=self.random_state)
        pca.fit(data)
        
        # Dimensionalidad efectiva basada en eigenvalues
        eigenvals = pca.explained_variance_
        total_variance = np.sum(eigenvals)
        participation_ratio = (np.sum(eigenvals) ** 2) / np.sum(eigenvals ** 2)
        
        # Método 2: Análisis de vecinos más cercanos
        nn = NearestNeighbors(n_neighbors=k+1)  # +1 porque incluye el punto mismo
        nn.fit(data)
        
        distances, _ = nn.kneighbors(data)
        # Remover distancia a sí mismo (primera columna)
        neighbor_distances = distances[:, 1:]
        
        # Estimar dimensionalidad local promedio
        local_dimensions = []
        for i in range(min(100, data.shape[0])):  # Muestrear para eficiencia
            point_distances = neighbor_distances[i]
            if np.max(point_distances) > 0:
                # Método de estimación basado en crecimiento de volumen
                log_distances = np.log(point_distances[point_distances > 0])
                if len(log_distances) > 1:
                    local_dim = len(log_distances) / (np.max(log_distances) - np.min(log_distances))
                    local_dimensions.append(local_dim)
        
        intrinsic_analysis = {
            'participation_ratio': float(participation_ratio),
            'effective_rank': float(participation_ratio),
            'local_dimensionality_mean': float(np.mean(local_dimensions)) if local_dimensions else 0,
            'local_dimensionality_std': float(np.std(local_dimensions)) if local_dimensions else 0,
            'dimensionality_estimate': float(participation_ratio),
            'compression_ratio': float(participation_ratio / data.shape[1])
        }
        
        return intrinsic_analysis
    
    def _analyze_volume_concentration(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Analizar concentración de volumen en alta dimensión.
        
        En espacios de alta dimensión, la mayoría del volumen se concentra
        en la "cáscara" exterior de la hipersfera, afectando clustering.
        """
        # Calcular distancias desde centroide
        centroid = np.mean(data, axis=0)
        distances_from_center = np.linalg.norm(data - centroid, axis=1)
        
        # Análisis de distribución radial
        mean_radius = np.mean(distances_from_center)
        std_radius = np.std(distances_from_center)
        
        # Coeficiente de variación radial
        radial_cv = std_radius / mean_radius if mean_radius > 0 else 0
        
        volume_analysis = {
            'mean_radius': float(mean_radius),
            'std_radius': float(std_radius),
            'radial_coefficient_variation': float(radial_cv),
            'volume_concentration_severity': self._interpret_volume_concentration(radial_cv)
        }
        
        return volume_analysis
    
    def _calculate_separability_metrics(self, data: np.ndarray, n_clusters_test: List[int] = [2, 3, 4, 5]) -> Dict[str, Any]:
        """
        Calcular métricas de separabilidad para diferentes números de clusters.
        
        Args:
            data: Datos para análisis
            n_clusters_test: Lista de números de clusters a evaluar
            
        Returns:
            Métricas de separabilidad por número de clusters
        """
        separability_results = {}
        
        for k in n_clusters_test:
            try:
                # Aplicar K-Means
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(data)
                
                # Calcular Silhouette Score
                silhouette_avg = silhouette_score(data, labels)
                
                # Calcular Variance Ratio Criterion (Calinski-Harabasz)
                from sklearn.metrics import calinski_harabasz_score
                calinski_harabasz = calinski_harabasz_score(data, labels)
                
                # Calcular Davies-Bouldin Index
                from sklearn.metrics import davies_bouldin_score
                davies_bouldin = davies_bouldin_score(data, labels)
                
                separability_results[f'k_{k}'] = {
                    'silhouette_score': float(silhouette_avg),
                    'calinski_harabasz_score': float(calinski_harabasz),
                    'davies_bouldin_score': float(davies_bouldin),
                    'inertia': float(kmeans.inertia_)
                }
                
            except Exception as e:
                self.logger.warning(f"Error calculando separabilidad para k={k}: {e}")
                separability_results[f'k_{k}'] = None
        
        return separability_results
    
    def _analyze_distance_distribution(self, data: np.ndarray) -> Dict[str, Any]:
        """Analizar distribución de distancias en espacio de baja dimensión."""
        distances = pdist(data, metric='euclidean')
        
        distribution_analysis = {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'median': float(np.median(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'skewness': float(self._calculate_skewness(distances)),
            'kurtosis': float(self._calculate_kurtosis(distances))
        }
        
        return distribution_analysis
    
    def _analyze_feature_correlations(self, data: np.ndarray) -> Dict[str, Any]:
        """Analizar correlaciones entre características en espacio musical."""
        correlation_matrix = np.corrcoef(data.T)
        
        # Extraer triángulo superior para análisis
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        
        correlation_analysis = {
            'mean_correlation': float(np.mean(np.abs(upper_triangle))),
            'max_correlation': float(np.max(np.abs(upper_triangle))),
            'high_correlations_count': int(np.sum(np.abs(upper_triangle) > 0.7)),
            'correlation_matrix_condition_number': float(np.linalg.cond(correlation_matrix))
        }
        
        return correlation_analysis
    
    def _analyze_natural_clustering_structure(self, data: np.ndarray) -> Dict[str, Any]:
        """Analizar estructura natural de clustering en espacio musical."""
        # Usar múltiples métricas para evaluar estructura natural
        
        # 1. Índice de conectividad
        from sklearn.neighbors import kneighbors_graph
        connectivity_graph = kneighbors_graph(data, n_neighbors=10, include_self=False)
        connectivity_index = connectivity_graph.nnz / (data.shape[0] * (data.shape[0] - 1))
        
        # 2. Análisis de gaps entre clusters potenciales
        gap_analysis = self._perform_gap_analysis(data)
        
        structure_analysis = {
            'connectivity_index': float(connectivity_index),
            'gap_analysis': gap_analysis,
            'natural_structure_score': self._calculate_natural_structure_score(data)
        }
        
        return structure_analysis
    
    def _compare_dimensional_characteristics(self, semantic_data: np.ndarray, 
                                           musical_data: np.ndarray) -> Dict[str, Any]:
        """
        Comparar características dimensionales entre espacios semántico y musical.
        """
        comparative_analysis = {
            'dimensionality_ratio': semantic_data.shape[1] / musical_data.shape[1],
            'sample_density_semantic': semantic_data.shape[0] / semantic_data.shape[1],
            'sample_density_musical': musical_data.shape[0] / musical_data.shape[1],
            'density_advantage_musical': (musical_data.shape[0] / musical_data.shape[1]) / (semantic_data.shape[0] / semantic_data.shape[1])
        }
        
        return comparative_analysis
    
    def _analyze_curse_of_dimensionality(self, semantic_data: np.ndarray, 
                                       musical_data: np.ndarray) -> Dict[str, Any]:
        """
        Analizar severidad de maldición de dimensionalidad en cada espacio.
        """
        curse_analysis = {
            'semantic_curse_indicators': {
                'high_dimensionality': semantic_data.shape[1] > 50,
                'sparse_sampling': (semantic_data.shape[0] / semantic_data.shape[1]) < 100,
                'curse_severity_score': self._calculate_curse_severity(semantic_data)
            },
            'musical_curse_indicators': {
                'high_dimensionality': musical_data.shape[1] > 50,
                'sparse_sampling': (musical_data.shape[0] / musical_data.shape[1]) < 100,
                'curse_severity_score': self._calculate_curse_severity(musical_data)
            }
        }
        
        return curse_analysis
    
    def _predict_clustering_quality(self, semantic_analysis: Dict, musical_analysis: Dict) -> Dict[str, Any]:
        """
        Predecir calidad de clustering basada en análisis dimensional.
        """
        # Extraer métricas clave para predicción
        semantic_pca = semantic_analysis['pca_analysis']
        musical_pca = musical_analysis['pca_analysis']
        
        prediction = {
            'semantic_predicted_quality': self._predict_quality_score(semantic_analysis),
            'musical_predicted_quality': self._predict_quality_score(musical_analysis),
            'quality_difference': self._predict_quality_score(musical_analysis) - self._predict_quality_score(semantic_analysis),
            'recommended_approach': self._recommend_clustering_approach(semantic_analysis, musical_analysis)
        }
        
        return prediction
    
    # Métodos auxiliares para cálculos específicos
    def _calculate_effective_dimensionality(self, explained_variance_ratio: np.ndarray) -> float:
        """Calcular dimensionalidad efectiva basada en entropía de eigenvalues."""
        # Normalizar para obtener probabilidades
        p = explained_variance_ratio / np.sum(explained_variance_ratio)
        
        # Calcular entropía de Shannon
        entropy_val = entropy(p)
        
        # Dimensionalidad efectiva como exponencial de entropía
        effective_dim = np.exp(entropy_val)
        
        return float(effective_dim)
    
    def _interpret_concentration_severity(self, concentration_coefficient: float) -> str:
        """Interpretar severidad de concentración de distancias."""
        if concentration_coefficient > 0.3:
            return "Low concentration (good for clustering)"
        elif concentration_coefficient > 0.15:
            return "Moderate concentration"
        else:
            return "High concentration (curse of dimensionality)"
    
    def _interpret_volume_concentration(self, radial_cv: float) -> str:
        """Interpretar concentración de volumen."""
        if radial_cv > 0.2:
            return "Low volume concentration"
        elif radial_cv > 0.1:
            return "Moderate volume concentration"
        else:
            return "High volume concentration (problematic for clustering)"
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calcular sesgo de distribución."""
        from scipy.stats import skew
        return skew(data)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calcular curtosis de distribución."""
        from scipy.stats import kurtosis
        return kurtosis(data)
    
    def _perform_gap_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Realizar análisis de gaps para estructura natural."""
        # Implementación simplificada del gap statistic
        gap_results = {
            'optimal_k_estimate': 3,  # Placeholder - implementación completa requiere más código
            'gap_values': [0.1, 0.2, 0.15, 0.1],  # Placeholder
            'confidence': 0.8
        }
        
        return gap_results
    
    def _calculate_natural_structure_score(self, data: np.ndarray) -> float:
        """Calcular score de estructura natural (0-1)."""
        # Score basado en múltiples factores
        # Implementación simplificada
        return 0.75  # Placeholder
    
    def _calculate_curse_severity(self, data: np.ndarray) -> float:
        """Calcular severidad de maldición de dimensionalidad (0-1)."""
        dimensionality = data.shape[1]
        sample_size = data.shape[0]
        
        # Factores que contribuyen a la maldición
        dim_factor = min(1.0, dimensionality / 100)  # Normalizar dimensionalidad
        density_factor = max(0.0, 1.0 - (sample_size / dimensionality) / 1000)  # Densidad de muestreo
        
        severity = (dim_factor + density_factor) / 2
        
        return float(severity)
    
    def _predict_quality_score(self, analysis: Dict) -> float:
        """Predecir score de calidad de clustering (0-1)."""
        # Combinar múltiples factores para predicción
        pca_factor = 1.0 - analysis['pca_analysis']['dimensionality_ratio']
        
        if 'distance_concentration' in analysis:
            concentration_factor = analysis['distance_concentration']['concentration_coefficient']
        else:
            concentration_factor = 0.5
        
        quality_score = (pca_factor + concentration_factor) / 2
        
        return float(np.clip(quality_score, 0.0, 1.0))
    
    def _recommend_clustering_approach(self, semantic_analysis: Dict, musical_analysis: Dict) -> str:
        """Recomendar enfoque de clustering basado en análisis dimensional."""
        semantic_quality = self._predict_quality_score(semantic_analysis)
        musical_quality = self._predict_quality_score(musical_analysis)
        
        if musical_quality - semantic_quality > 0.2:
            return "clustering_musical_only"
        elif semantic_quality > 0.7 and musical_quality > 0.7:
            return "clustering_bidirectional"
        elif musical_quality > 0.7:
            return "clustering_musical_primary"
        else:
            return "vectorization_direct_recommended"