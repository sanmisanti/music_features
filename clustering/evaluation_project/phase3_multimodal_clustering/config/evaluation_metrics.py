#!/usr/bin/env python3
"""
Función objetivo multi-criterio y métricas de evaluación para FASE 3.
Implementa el balance óptimo entre calidad técnica e interpretabilidad.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
import warnings

class MultiCriteriaEvaluator:
    """Evaluador con función objetivo multi-criterio balanceada."""
    
    def __init__(self):
        """Inicializar pesos de función objetivo."""
        self.weights = {
            'silhouette': 0.30,      # Calidad técnica
            'balance': 0.30,         # Balance de distribución  
            'interpretability': 0.20, # Interpretabilidad automática
            'cross_modal': 0.10,     # Correspondencia cross-modal
            'granularity': 0.10      # Bonus granularidad K≥5
        }
        
    def evaluate_clustering_quality(self, X: np.ndarray, labels: np.ndarray, 
                                   domain: str) -> Dict[str, float]:
        """
        Evaluar calidad técnica del clustering.
        
        Args:
            X: Datos (N, features)
            labels: Etiquetas de clustering (N,)
            domain: 'musical' o 'semantic'
            
        Returns:
            Dict con métricas de calidad técnica
        """
        # Validar entrada
        if len(np.unique(labels)) < 2:
            return {'silhouette_score': -1.0, 'calinski_harabasz_score': 0.0, 
                   'davies_bouldin_score': 10.0, 'valid': False}
        
        # Filtrar outliers para DBSCAN
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2:
            return {'silhouette_score': -1.0, 'calinski_harabasz_score': 0.0,
                   'davies_bouldin_score': 10.0, 'valid': False}
        
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        
        try:
            # Calcular métricas técnicas
            silhouette = silhouette_score(X_valid, labels_valid)
            calinski_harabasz = calinski_harabasz_score(X_valid, labels_valid)
            davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
            
            return {
                'silhouette_score': float(silhouette),
                'calinski_harabasz_score': float(calinski_harabasz),
                'davies_bouldin_score': float(davies_bouldin),
                'n_clusters': len(np.unique(labels_valid)),
                'n_outliers': np.sum(labels == -1),
                'outlier_ratio': float(np.sum(labels == -1) / len(labels)),
                'valid': True
            }
            
        except Exception as e:
            warnings.warn(f"Error calculating clustering metrics: {e}")
            return {'silhouette_score': -1.0, 'calinski_harabasz_score': 0.0,
                   'davies_bouldin_score': 10.0, 'valid': False}
    
    def evaluate_balance_distribution(self, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluar balance en distribución de clusters.
        
        Args:
            labels: Etiquetas de clustering
            
        Returns:
            Dict con métricas de balance
        """
        # Filtrar outliers
        valid_labels = labels[labels != -1]
        if len(valid_labels) == 0:
            return {'balance_score': 0.0, 'entropy_score': 0.0}
        
        # Contar elementos por cluster
        cluster_counts = Counter(valid_labels)
        cluster_sizes = np.array(list(cluster_counts.values()))
        
        if len(cluster_sizes) < 2:
            return {'balance_score': 0.0, 'entropy_score': 0.0}
        
        # Balance score: 1 - coeficiente de variación
        mean_size = np.mean(cluster_sizes)
        std_size = np.std(cluster_sizes)
        cv = std_size / mean_size if mean_size > 0 else float('inf')
        balance_score = max(0.0, 1.0 - cv)
        
        # Entropy score: entropía normalizada
        cluster_probs = cluster_sizes / np.sum(cluster_sizes)
        entropy = -np.sum(cluster_probs * np.log2(cluster_probs + 1e-10))
        max_entropy = np.log2(len(cluster_sizes))
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return {
            'balance_score': float(balance_score),
            'entropy_score': float(entropy_score),
            'n_clusters': len(cluster_sizes),
            'min_cluster_size': int(np.min(cluster_sizes)),
            'max_cluster_size': int(np.max(cluster_sizes)),
            'mean_cluster_size': float(mean_size),
            'std_cluster_size': float(std_size)
        }
    
    def evaluate_interpretability(self, X: np.ndarray, labels: np.ndarray, 
                                domain: str) -> Dict[str, float]:
        """
        Evaluar interpretabilidad automática de clusters.
        
        Args:
            X: Datos originales
            labels: Etiquetas de clustering
            domain: 'musical' o 'semantic'
            
        Returns:
            Dict con métricas de interpretabilidad
        """
        # Filtrar outliers
        valid_mask = labels != -1
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        
        if len(np.unique(labels_valid)) < 2:
            return {'interpretability_score': 0.0, 'coherence_score': 0.0}
        
        if domain == 'musical':
            return self._evaluate_musical_interpretability(X_valid, labels_valid)
        else:
            return self._evaluate_semantic_interpretability(X_valid, labels_valid)
    
    def _evaluate_musical_interpretability(self, X: np.ndarray, 
                                         labels: np.ndarray) -> Dict[str, float]:
        """Evaluar interpretabilidad musical basada en características dominantes."""
        unique_labels = np.unique(labels)
        coherence_scores = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) < 2:
                continue
                
            # Calcular coherencia interna basada en características dominantes
            # Para cada característica, calcular qué tan consistente es en el cluster
            feature_consistency = []
            for feature_idx in range(X.shape[1]):
                feature_values = cluster_data[:, feature_idx]
                # Coherencia = 1 - coeficiente de variación normalizado
                if np.std(feature_values) == 0:
                    consistency = 1.0
                else:
                    cv = np.std(feature_values) / (np.abs(np.mean(feature_values)) + 1e-10)
                    consistency = 1.0 / (1.0 + cv)  # Normalizar entre 0 y 1
                feature_consistency.append(consistency)
            
            # Coherencia del cluster = media de consistencias de características
            cluster_coherence = np.mean(feature_consistency)
            coherence_scores.append(cluster_coherence)
        
        interpretability_score = np.mean(coherence_scores) if coherence_scores else 0.0
        
        return {
            'interpretability_score': float(interpretability_score),
            'coherence_score': float(interpretability_score),
            'n_interpretable_clusters': len(coherence_scores)
        }
    
    def _evaluate_semantic_interpretability(self, X: np.ndarray, 
                                          labels: np.ndarray) -> Dict[str, float]:
        """Evaluar interpretabilidad semántica basada en similitud coseno interna."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        unique_labels = np.unique(labels)
        coherence_scores = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) < 2:
                continue
            
            # Calcular similitud coseno interna promedio
            similarity_matrix = cosine_similarity(cluster_data)
            # Extraer triángulo superior (sin diagonal)
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            if len(upper_triangle) > 0:
                avg_similarity = np.mean(upper_triangle)
                coherence_scores.append(avg_similarity)
        
        interpretability_score = np.mean(coherence_scores) if coherence_scores else 0.0
        
        return {
            'interpretability_score': float(interpretability_score),
            'coherence_score': float(interpretability_score),
            'n_interpretable_clusters': len(coherence_scores)
        }
    
    def evaluate_cross_modal_correspondence(self, labels_musical: np.ndarray, 
                                          labels_semantic: np.ndarray) -> Dict[str, float]:
        """
        Evaluar correspondencia cross-modal usando NMI.
        
        Args:
            labels_musical: Etiquetas clustering musical
            labels_semantic: Etiquetas clustering semántico
            
        Returns:
            Dict con métricas de correspondencia
        """
        # Filtrar outliers de ambos dominios
        valid_mask = (labels_musical != -1) & (labels_semantic != -1)
        
        if np.sum(valid_mask) < 2:
            return {'nmi_score': 0.0, 'correspondence_score': 0.0}
        
        labels_musical_valid = labels_musical[valid_mask]
        labels_semantic_valid = labels_semantic[valid_mask]
        
        # Calcular Normalized Mutual Information
        nmi_score = normalized_mutual_info_score(labels_musical_valid, 
                                               labels_semantic_valid)
        
        return {
            'nmi_score': float(nmi_score),
            'correspondence_score': float(nmi_score),
            'n_valid_samples': int(np.sum(valid_mask))
        }
    
    def evaluate_granularity_bonus(self, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluar bonus por granularidad (K≥5).
        
        Args:
            labels: Etiquetas de clustering
            
        Returns:
            Dict con métricas de granularidad
        """
        valid_labels = labels[labels != -1]
        n_clusters = len(np.unique(valid_labels)) if len(valid_labels) > 0 else 0
        
        # Bonus escalado: máximo cuando K≥5
        if n_clusters >= 5:
            granularity_bonus = 1.0
        elif n_clusters >= 3:
            granularity_bonus = 0.5
        else:
            granularity_bonus = 0.0
        
        return {
            'granularity_bonus': float(granularity_bonus),
            'n_clusters': int(n_clusters),
            'meets_granularity_criteria': n_clusters >= 5
        }
    
    def compute_composite_score(self, silhouette_score: float, balance_score: float,
                              interpretability_score: float, cross_modal_score: float,
                              granularity_bonus: float) -> float:
        """
        Computar score compuesto con función objetivo multi-criterio.
        
        Args:
            silhouette_score: Score de silhouette normalizado [0,1]
            balance_score: Score de balance [0,1] 
            interpretability_score: Score de interpretabilidad [0,1]
            cross_modal_score: Score cross-modal [0,1]
            granularity_bonus: Bonus granularidad [0,1]
            
        Returns:
            Score compuesto [0,1]
        """
        # Normalizar silhouette score de [-1,1] a [0,1]
        silhouette_normalized = (silhouette_score + 1.0) / 2.0
        
        composite_score = (
            self.weights['silhouette'] * silhouette_normalized +
            self.weights['balance'] * balance_score +
            self.weights['interpretability'] * interpretability_score +
            self.weights['cross_modal'] * cross_modal_score +
            self.weights['granularity'] * granularity_bonus
        )
        
        return float(np.clip(composite_score, 0.0, 1.0))

# Instancia global del evaluador
multi_criteria_evaluator = MultiCriteriaEvaluator()