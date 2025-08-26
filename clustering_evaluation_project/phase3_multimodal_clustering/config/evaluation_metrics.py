"""
Métricas de Evaluación Multi-Criterio para Clustering Multimodal
================================================================

Implementa sistema especializado de métricas que balancean calidad técnica,
interpretabilidad, y granularidad explicativa para clustering multimodal
con prioridad en explicabilidad (K≥5).

Autor: Proyecto FASE 3 - Sistema Clustering Multimodal
Fecha: Agosto 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Any, Optional
import warnings


class EvaluationMetrics:
    """
    Sistema de evaluación multi-criterio para clustering multimodal.
    
    Implementa métricas especializadas que priorizan interpretabilidad
    y granularidad explicativa sobre optimización métrica pura.
    """
    
    def __init__(self, granularity_minimum: int = 5):
        """
        Inicializar sistema de métricas de evaluación.
        
        Args:
            granularity_minimum: Valor mínimo de K para bonus granularidad
        """
        self.granularity_minimum = granularity_minimum
        
        # Pesos para función objetivo multi-criterio
        self.weights = {
            'silhouette_normalized': 0.3,
            'balance_distribution': 0.3,
            'interpretability_score': 0.2,
            'cross_modal_correspondence': 0.1,
            'granularity_bonus': 0.1
        }
    
    def calculate_traditional_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calcular métricas tradicionales de clustering.
        
        Args:
            X: Matriz de características
            labels: Etiquetas de clustering
            
        Returns:
            Dict con métricas tradicionales
        """
        metrics = {}
        
        # Validar entrada
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            # Clustering inválido
            return {
                'silhouette_score': -1.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'n_clusters': n_clusters,
                'n_noise_points': 0
            }
        
        # Manejar ruido en DBSCAN (-1 labels)
        n_noise_points = np.sum(labels == -1)
        if n_noise_points > 0:
            # Filtrar puntos de ruido para métricas
            mask = labels != -1
            X_clean = X[mask]
            labels_clean = labels[mask]
            
            if len(np.unique(labels_clean)) < 2:
                # Todos los puntos son ruido o un solo cluster
                return {
                    'silhouette_score': -1.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': float('inf'),
                    'n_clusters': 1,
                    'n_noise_points': n_noise_points
                }
        else:
            X_clean = X
            labels_clean = labels
        
        try:
            # Silhouette Score
            if len(np.unique(labels_clean)) > 1 and len(X_clean) > len(np.unique(labels_clean)):
                metrics['silhouette_score'] = silhouette_score(X_clean, labels_clean)
            else:
                metrics['silhouette_score'] = -1.0
            
            # Calinski-Harabasz Index
            if len(np.unique(labels_clean)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_clean, labels_clean)
            else:
                metrics['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin Index
            if len(np.unique(labels_clean)) > 1:
                metrics['davies_bouldin_score'] = davies_bouldin_score(X_clean, labels_clean)
            else:
                metrics['davies_bouldin_score'] = float('inf')
                
        except Exception as e:
            warnings.warn(f"Error calculando métricas tradicionales: {e}")
            metrics['silhouette_score'] = -1.0
            metrics['calinski_harabasz_score'] = 0.0
            metrics['davies_bouldin_score'] = float('inf')
        
        metrics['n_clusters'] = n_clusters
        metrics['n_noise_points'] = n_noise_points
        
        return metrics
    
    def calculate_balance_distribution_score(self, labels: np.ndarray) -> float:
        """
        Calcular score de balance de distribución de clusters.
        
        Penaliza clusters dominantes (>50%) y excesiva fragmentación (<3%).
        
        Args:
            labels: Etiquetas de clustering
            
        Returns:
            Score de balance [0, 1] donde 1 es óptimo
        """
        if len(labels) == 0:
            return 0.0
        
        # Contar distribución excluyendo ruido
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        
        if len(unique_labels) < 2:
            return 0.0
        
        # Calcular porcentajes
        total_points = np.sum(counts)
        percentages = counts / total_points
        
        # Penalizaciones
        dominance_penalty = 0.0
        fragmentation_penalty = 0.0
        
        # Penalizar dominancia (clusters >50%)
        dominant_clusters = percentages > 0.5
        if np.any(dominant_clusters):
            dominance_penalty = np.max(percentages[dominant_clusters]) - 0.5
        
        # Penalizar fragmentación excesiva (clusters <3%)
        fragmented_clusters = percentages < 0.03
        if np.any(fragmented_clusters):
            fragmentation_penalty = 0.03 - np.min(percentages[fragmented_clusters])
        
        # Calcular score de balance
        balance_score = 1.0 - dominance_penalty - fragmentation_penalty
        
        return max(0.0, balance_score)
    
    def calculate_interpretability_score(self, X: np.ndarray, labels: np.ndarray, 
                                       domain: str) -> float:
        """
        Calcular score de interpretabilidad basado en coherencia intra-cluster.
        
        Args:
            X: Matriz de características
            labels: Etiquetas de clustering
            domain: 'musical' o 'semantic'
            
        Returns:
            Score de interpretabilidad [0, 1]
        """
        unique_labels = np.unique(labels[labels != -1])
        
        if len(unique_labels) < 2:
            return 0.0
        
        interpretability_scores = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) < 3:
                # Cluster muy pequeño
                interpretability_scores.append(0.0)
                continue
            
            if domain == 'musical':
                # Para dominio musical: coherencia en características dominantes
                score = self._calculate_musical_coherence(cluster_data)
            elif domain == 'semantic':
                # Para dominio semántico: coherencia coseno interna
                score = self._calculate_semantic_coherence(cluster_data)
            else:
                score = 0.0
            
            interpretability_scores.append(score)
        
        return np.mean(interpretability_scores) if interpretability_scores else 0.0
    
    def _calculate_musical_coherence(self, cluster_data: np.ndarray) -> float:
        """
        Calcular coherencia musical basada en características dominantes.
        
        Args:
            cluster_data: Datos del cluster musical
            
        Returns:
            Score de coherencia musical
        """
        if len(cluster_data) < 3:
            return 0.0
        
        # Calcular coeficiente de variación promedio
        # Menor variación = mayor coherencia
        cv_scores = []
        for feature_idx in range(cluster_data.shape[1]):
            feature_data = cluster_data[:, feature_idx]
            if np.std(feature_data) > 0:
                cv = np.std(feature_data) / (np.abs(np.mean(feature_data)) + 1e-6)
                cv_scores.append(1.0 / (1.0 + cv))  # Invertir para que menor CV = mayor score
            else:
                cv_scores.append(1.0)  # Variación cero = coherencia perfecta
        
        return np.mean(cv_scores)
    
    def _calculate_semantic_coherence(self, cluster_data: np.ndarray) -> float:
        """
        Calcular coherencia semántica basada en similitud coseno interna.
        
        Args:
            cluster_data: Datos del cluster semántico (embeddings BERT)
            
        Returns:
            Score de coherencia semántica
        """
        if len(cluster_data) < 3:
            return 0.0
        
        # Normalizar embeddings para similitud coseno
        norms = np.linalg.norm(cluster_data, axis=1)
        normalized_data = cluster_data / (norms.reshape(-1, 1) + 1e-8)
        
        # Calcular matriz de similitud coseno
        similarity_matrix = np.dot(normalized_data, normalized_data.T)
        
        # Extraer triángulo superior (sin diagonal)
        n = len(similarity_matrix)
        upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]
        
        # Coherencia = similitud promedio interna
        return np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
    
    def calculate_granularity_bonus(self, n_clusters: int) -> float:
        """
        Calcular bonus por granularidad explicativa.
        
        Args:
            n_clusters: Número de clusters
            
        Returns:
            Bonus granularidad [0, 1]
        """
        if n_clusters >= self.granularity_minimum:
            # Bonus logarítmico para K≥5
            bonus = min(1.0, np.log(n_clusters / self.granularity_minimum + 1) / np.log(3))
            return bonus
        else:
            # Penalización lineal para K<5
            penalty = (self.granularity_minimum - n_clusters) / self.granularity_minimum
            return max(0.0, 1.0 - penalty)
    
    def calculate_cross_modal_correspondence(self, labels_musical: np.ndarray, 
                                           labels_semantic: np.ndarray) -> float:
        """
        Calcular correspondencia cross-modal entre clustering musical y semántico.
        
        Args:
            labels_musical: Etiquetas clustering musical
            labels_semantic: Etiquetas clustering semántico
            
        Returns:
            Score correspondencia cross-modal [0, 1]
        """
        if len(labels_musical) != len(labels_semantic):
            warnings.warn("Arrays de etiquetas tienen longitudes diferentes")
            return 0.0
        
        # Filtrar puntos de ruido de ambos dominios
        valid_mask = (labels_musical != -1) & (labels_semantic != -1)
        
        if np.sum(valid_mask) < 10:
            return 0.0
        
        labels_m_clean = labels_musical[valid_mask]
        labels_s_clean = labels_semantic[valid_mask]
        
        try:
            # Normalized Mutual Information como proxy de correspondencia
            nmi = normalized_mutual_info_score(labels_m_clean, labels_s_clean)
            return nmi
        except Exception as e:
            warnings.warn(f"Error calculando correspondencia cross-modal: {e}")
            return 0.0
    
    def calculate_composite_score(self, evaluation_results: Dict[str, Any]) -> float:
        """
        Calcular score compuesto multi-criterio.
        
        Args:
            evaluation_results: Resultados de evaluación completa
            
        Returns:
            Score compuesto ponderado [0, 1]
        """
        # Normalizar silhouette score [-1, 1] -> [0, 1]
        silhouette_normalized = (evaluation_results.get('silhouette_score', -1) + 1) / 2
        
        # Componentes del score
        components = {
            'silhouette_normalized': silhouette_normalized,
            'balance_distribution': evaluation_results.get('balance_distribution_score', 0),
            'interpretability_score': evaluation_results.get('interpretability_score', 0),
            'cross_modal_correspondence': evaluation_results.get('cross_modal_correspondence', 0),
            'granularity_bonus': evaluation_results.get('granularity_bonus', 0)
        }
        
        # Calcular score ponderado
        composite_score = sum(
            self.weights[component] * value 
            for component, value in components.items()
        )
        
        return max(0.0, min(1.0, composite_score))
    
    def evaluate_clustering_complete(self, X: np.ndarray, labels: np.ndarray, 
                                   domain: str, k: int) -> Dict[str, Any]:
        """
        Evaluación completa de clustering para un experimento.
        
        Args:
            X: Matriz de características
            labels: Etiquetas de clustering
            domain: 'musical' o 'semantic'
            k: Número de clusters objetivo
            
        Returns:
            Dict con evaluación completa
        """
        results = {}
        
        # Métricas tradicionales
        traditional_metrics = self.calculate_traditional_metrics(X, labels)
        results.update(traditional_metrics)
        
        # Métricas especializadas
        results['balance_distribution_score'] = self.calculate_balance_distribution_score(labels)
        results['interpretability_score'] = self.calculate_interpretability_score(X, labels, domain)
        results['granularity_bonus'] = self.calculate_granularity_bonus(results['n_clusters'])
        
        # Score compuesto (sin cross-modal aún)
        results['composite_score_partial'] = self.calculate_composite_score(results)
        
        # Metadatos
        results['domain'] = domain
        results['k_target'] = k
        results['evaluation_timestamp'] = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        return results


# Instancia global de métricas
evaluation_metrics = EvaluationMetrics()