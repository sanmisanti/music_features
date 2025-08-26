"""
Validador de Interpretabilidad para Sistema Clustering Multimodal
================================================================

Implementa validación avanzada de interpretabilidad, coherencia temática,
y generación de explicaciones automatizadas para clustering multimodal
con enfoque en granularidad explicativa.

Autor: Proyecto FASE 3 - Sistema Clustering Multimodal
Fecha: Agosto 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import pearsonr, spearmanr
import warnings
import logging

from .config.interpretability_settings import interpretability_settings
from .config.evaluation_metrics import evaluation_metrics


class InterpretabilityValidator:
    """
    Validador especializado de interpretabilidad para clustering multimodal.
    
    Evalúa coherencia temática, calidad explicativa, y capacidades
    de etiquetado automático con criterios científicos rigurosos.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Inicializar validador de interpretabilidad.
        
        Args:
            verbose: Activar logging detallado
        """
        self.verbose = verbose
        self.logger = self._setup_logging()
        
        # Configuraciones de interpretabilidad
        self.settings = interpretability_settings
        
        # Cache para optimizar cálculos repetidos
        self._coherence_cache = {}
        self._label_cache = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Configurar logging para validador."""
        logger = logging.getLogger(f'InterpretabilityValidator_{id(self)}')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_cluster_interpretability_complete(self, X: np.ndarray, labels: np.ndarray, 
                                                 domain: str) -> Dict[str, Any]:
        """
        Validación completa de interpretabilidad para clustering.
        
        Args:
            X: Matriz de características
            labels: Etiquetas de clustering
            domain: 'musical' o 'semantic'
            
        Returns:
            Dict con validación completa de interpretabilidad
        """
        self.logger.info(f"Validando interpretabilidad completa para dominio {domain}")
        
        validation_results = {
            'domain': domain,
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else 0,
            'validation_timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # Análisis por cluster individual
        unique_labels = np.unique(labels[labels != -1])
        validation_results['n_clusters'] = len(unique_labels)
        
        if len(unique_labels) == 0:
            return self._create_empty_validation_result(domain, "Sin clusters válidos")
        
        cluster_validations = []
        interpretable_clusters_count = 0
        
        for cluster_label in unique_labels:
            cluster_validation = self.validate_single_cluster_interpretability(
                X, labels, cluster_label, domain
            )
            cluster_validations.append(cluster_validation)
            
            if cluster_validation['is_interpretable']:
                interpretable_clusters_count += 1
        
        validation_results['cluster_validations'] = cluster_validations
        validation_results['interpretable_clusters_count'] = interpretable_clusters_count
        validation_results['interpretability_ratio'] = interpretable_clusters_count / len(unique_labels)
        
        # Evaluación de granularidad
        granularity_analysis = self.evaluate_granularity_adequacy(
            len(unique_labels), validation_results['interpretability_ratio']
        )
        validation_results.update(granularity_analysis)
        
        # Evaluación de balance
        balance_analysis = self.evaluate_cluster_balance_interpretability(labels)
        validation_results.update(balance_analysis)
        
        # Score global de interpretabilidad
        global_interpretability_score = self.calculate_global_interpretability_score(validation_results)
        validation_results['global_interpretability_score'] = global_interpretability_score
        
        # Recomendaciones de mejora
        recommendations = self.generate_interpretability_recommendations(validation_results)
        validation_results['improvement_recommendations'] = recommendations
        
        # Clasificación final
        validation_results['interpretability_classification'] = self.classify_interpretability_level(
            global_interpretability_score
        )
        
        self.logger.info(f"Validación completada: {interpretable_clusters_count}/{len(unique_labels)} "
                        f"clusters interpretables, score global: {global_interpretability_score:.3f}")
        
        return validation_results
    
    def validate_single_cluster_interpretability(self, X: np.ndarray, labels: np.ndarray, 
                                               cluster_label: int, domain: str) -> Dict[str, Any]:
        """
        Validar interpretabilidad de cluster individual.
        
        Args:
            X: Matriz de características
            labels: Etiquetas de clustering
            cluster_label: Etiqueta del cluster a validar
            domain: 'musical' o 'semantic'
            
        Returns:
            Dict con validación del cluster individual
        """
        # Extraer datos del cluster
        cluster_mask = labels == cluster_label
        cluster_data = X[cluster_mask]
        cluster_size = len(cluster_data)
        
        validation_result = {
            'cluster_label': cluster_label,
            'cluster_size': cluster_size,
            'domain': domain
        }
        
        if cluster_size < self.settings.quality_thresholds['min_cluster_size']:
            validation_result.update({
                'is_interpretable': False,
                'reason': f'Cluster muy pequeño ({cluster_size} < {self.settings.quality_thresholds["min_cluster_size"]})',
                'coherence_score': 0.0,
                'silhouette_avg': -1.0,
                'auto_label': f'Cluster Pequeño {domain.title()}',
                'descriptive_features': []
            })
            return validation_result
        
        # Calcular coherencia interna
        coherence_score = self.calculate_cluster_coherence(cluster_data, domain)
        
        # Calcular silhouette promedio del cluster
        if len(np.unique(labels)) > 1:
            cluster_silhouette = self.calculate_cluster_silhouette_avg(X, labels, cluster_label)
        else:
            cluster_silhouette = -1.0
        
        # Evaluar interpretabilidad según criterios
        is_interpretable = self.settings.validate_cluster_interpretability(
            cluster_size, cluster_silhouette, coherence_score
        )
        
        # Generar etiqueta automática
        auto_label = self.generate_cluster_automatic_label(cluster_data, domain, coherence_score)
        
        # Identificar características descriptivas
        descriptive_features = self.identify_descriptive_features(cluster_data, domain)
        
        validation_result.update({
            'is_interpretable': is_interpretable,
            'coherence_score': coherence_score,
            'silhouette_avg': cluster_silhouette,
            'auto_label': auto_label,
            'descriptive_features': descriptive_features,
            'reason': 'Interpretable' if is_interpretable else 'Baja coherencia o calidad técnica'
        })
        
        return validation_result
    
    def calculate_cluster_coherence(self, cluster_data: np.ndarray, domain: str) -> float:
        """
        Calcular coherencia interna de cluster específico por dominio.
        
        Args:
            cluster_data: Datos del cluster
            domain: 'musical' o 'semantic'
            
        Returns:
            Score de coherencia [0, 1]
        """
        if len(cluster_data) < 2:
            return 0.0
        
        # Usar cache si está disponible
        cache_key = (cluster_data.tobytes(), domain)
        if cache_key in self._coherence_cache:
            return self._coherence_cache[cache_key]
        
        if domain == 'musical':
            coherence = self._calculate_musical_cluster_coherence(cluster_data)
        elif domain == 'semantic':
            coherence = self._calculate_semantic_cluster_coherence(cluster_data)
        else:
            coherence = 0.0
        
        # Cache resultado
        self._coherence_cache[cache_key] = coherence
        return coherence
    
    def _calculate_musical_cluster_coherence(self, cluster_data: np.ndarray) -> float:
        """
        Calcular coherencia específica para cluster musical.
        
        Args:
            cluster_data: Datos del cluster musical
            
        Returns:
            Score coherencia musical
        """
        if len(cluster_data) < 3:
            return 0.0
        
        # Múltiples medidas de coherencia musical
        coherence_measures = []
        
        # 1. Coherencia por coeficiente de variación
        cv_coherence = self._calculate_cv_coherence(cluster_data)
        coherence_measures.append(cv_coherence)
        
        # 2. Coherencia por correlación interna
        correlation_coherence = self._calculate_internal_correlation_coherence(cluster_data)
        coherence_measures.append(correlation_coherence)
        
        # 3. Coherencia por estabilidad de características dominantes
        dominance_coherence = self._calculate_dominance_coherence(cluster_data)
        coherence_measures.append(dominance_coherence)
        
        # Promedio ponderado de medidas
        weights = [0.4, 0.3, 0.3]  # CV más importante para música
        return np.average(coherence_measures, weights=weights)
    
    def _calculate_semantic_cluster_coherence(self, cluster_data: np.ndarray) -> float:
        """
        Calcular coherencia específica para cluster semántico.
        
        Args:
            cluster_data: Datos del cluster semántico (embeddings BERT)
            
        Returns:
            Score coherencia semántica
        """
        if len(cluster_data) < 3:
            return 0.0
        
        # Normalizar embeddings para similitud coseno
        norms = np.linalg.norm(cluster_data, axis=1, keepdims=True)
        normalized_embeddings = cluster_data / (norms + 1e-8)
        
        # Múltiples medidas de coherencia semántica
        coherence_measures = []
        
        # 1. Coherencia por similitud coseno promedio
        cosine_coherence = self._calculate_cosine_coherence(normalized_embeddings)
        coherence_measures.append(cosine_coherence)
        
        # 2. Coherencia por dispersión en espacio normalizado
        dispersion_coherence = self._calculate_dispersion_coherence(normalized_embeddings)
        coherence_measures.append(dispersion_coherence)
        
        # 3. Coherencia por centroide
        centroid_coherence = self._calculate_centroid_coherence(normalized_embeddings)
        coherence_measures.append(centroid_coherence)
        
        # Promedio ponderado
        weights = [0.5, 0.25, 0.25]  # Coseno más importante para semántica
        return np.average(coherence_measures, weights=weights)
    
    def _calculate_cv_coherence(self, data: np.ndarray) -> float:
        """Calcular coherencia basada en coeficiente de variación."""
        cv_scores = []
        for feature_idx in range(data.shape[1]):
            feature_data = data[:, feature_idx]
            if np.std(feature_data) > 0:
                cv = np.std(feature_data) / (np.abs(np.mean(feature_data)) + 1e-6)
                cv_scores.append(1.0 / (1.0 + cv))
            else:
                cv_scores.append(1.0)
        return np.mean(cv_scores)
    
    def _calculate_internal_correlation_coherence(self, data: np.ndarray) -> float:
        """Calcular coherencia basada en correlaciones internas."""
        if data.shape[1] < 2:
            return 1.0
        
        try:
            correlation_matrix = np.corrcoef(data.T)
            # Extraer correlaciones únicas (triángulo superior sin diagonal)
            n_features = correlation_matrix.shape[0]
            upper_triangle = correlation_matrix[np.triu_indices(n_features, k=1)]
            
            # Coherencia = promedio de correlaciones absolutas
            return np.mean(np.abs(upper_triangle))
        except:
            return 0.5
    
    def _calculate_dominance_coherence(self, data: np.ndarray) -> float:
        """Calcular coherencia basada en estabilidad de características dominantes."""
        feature_means = np.mean(data, axis=0)
        feature_stds = np.std(data, axis=0)
        
        # Identificar características dominantes (extremos)
        dominant_features = (feature_means > 0.7) | (feature_means < 0.3)
        
        if not np.any(dominant_features):
            return 0.5  # Sin características dominantes claras
        
        # Coherencia = estabilidad de características dominantes
        dominant_stds = feature_stds[dominant_features]
        return np.mean(1.0 / (1.0 + dominant_stds))
    
    def _calculate_cosine_coherence(self, normalized_embeddings: np.ndarray) -> float:
        """Calcular coherencia basada en similitud coseno."""
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        n = similarity_matrix.shape[0]
        upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]
        return np.mean(upper_triangle)
    
    def _calculate_dispersion_coherence(self, normalized_embeddings: np.ndarray) -> float:
        """Calcular coherencia basada en dispersión espacial."""
        centroid = np.mean(normalized_embeddings, axis=0)
        distances = np.linalg.norm(normalized_embeddings - centroid, axis=1)
        # Coherencia inversa a dispersión
        return 1.0 / (1.0 + np.mean(distances))
    
    def _calculate_centroid_coherence(self, normalized_embeddings: np.ndarray) -> float:
        """Calcular coherencia basada en similitud al centroide."""
        centroid = np.mean(normalized_embeddings, axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        similarities = np.dot(normalized_embeddings, centroid_norm)
        return np.mean(similarities)
    
    def calculate_cluster_silhouette_avg(self, X: np.ndarray, labels: np.ndarray, 
                                       cluster_label: int) -> float:
        """
        Calcular silhouette promedio para cluster específico.
        
        Args:
            X: Matriz completa de características
            labels: Etiquetas completas de clustering
            cluster_label: Etiqueta del cluster objetivo
            
        Returns:
            Silhouette promedio del cluster
        """
        try:
            from sklearn.metrics import silhouette_samples
            
            if len(np.unique(labels)) < 2:
                return -1.0
            
            silhouette_samples_scores = silhouette_samples(X, labels)
            cluster_mask = labels == cluster_label
            
            return np.mean(silhouette_samples_scores[cluster_mask])
            
        except Exception:
            return -1.0
    
    def generate_cluster_automatic_label(self, cluster_data: np.ndarray, domain: str, 
                                       coherence_score: float) -> str:
        """
        Generar etiqueta automática para cluster.
        
        Args:
            cluster_data: Datos del cluster
            domain: 'musical' o 'semantic'
            coherence_score: Score de coherencia del cluster
            
        Returns:
            Etiqueta descriptiva automática
        """
        # Usar cache si está disponible
        cache_key = (cluster_data.tobytes(), domain, round(coherence_score, 3))
        if cache_key in self._label_cache:
            return self._label_cache[cache_key]
        
        if domain == 'musical':
            cluster_mean = np.mean(cluster_data, axis=0)
            cluster_std = np.std(cluster_data, axis=0)
            label = self.settings.generate_musical_cluster_label(cluster_mean, cluster_std)
        
        elif domain == 'semantic':
            # Seleccionar embeddings representativos
            representative_embeddings = cluster_data[:min(5, len(cluster_data))]
            label = self.settings.generate_semantic_cluster_label(representative_embeddings, coherence_score)
        
        else:
            label = f"Cluster {domain.title()}"
        
        # Cache resultado
        self._label_cache[cache_key] = label
        return label
    
    def identify_descriptive_features(self, cluster_data: np.ndarray, domain: str) -> List[Dict[str, Any]]:
        """
        Identificar características más descriptivas del cluster.
        
        Args:
            cluster_data: Datos del cluster
            domain: 'musical' o 'semantic'
            
        Returns:
            Lista de características descriptivas ordenadas por importancia
        """
        if domain == 'musical':
            return self._identify_musical_descriptive_features(cluster_data)
        elif domain == 'semantic':
            return self._identify_semantic_descriptive_features(cluster_data)
        else:
            return []
    
    def _identify_musical_descriptive_features(self, cluster_data: np.ndarray) -> List[Dict[str, Any]]:
        """Identificar características musicales descriptivas."""
        feature_names = self.settings.musical_labeling['feature_names']
        
        if len(feature_names) != cluster_data.shape[1]:
            return []
        
        feature_descriptions = []
        
        feature_means = np.mean(cluster_data, axis=0)
        feature_stds = np.std(cluster_data, axis=0)
        
        for i, (feature_name, mean_val, std_val) in enumerate(zip(feature_names, feature_means, feature_stds)):
            # Determinar descriptividad basada en extremos y consistencia
            is_extreme = mean_val > 0.7 or mean_val < 0.3
            is_consistent = std_val < 0.2
            
            if is_extreme and is_consistent:
                # Determinar nivel descriptivo
                if mean_val > 0.7:
                    level = 'Alto'
                elif mean_val < 0.3:
                    level = 'Bajo'
                else:
                    level = 'Medio'
                
                feature_descriptions.append({
                    'feature_name': feature_name,
                    'mean_value': float(mean_val),
                    'std_value': float(std_val),
                    'level': level,
                    'descriptiveness_score': float((0.5 - abs(mean_val - 0.5)) * (1.0 / (1.0 + std_val)))
                })
        
        # Ordenar por descriptiveness_score
        feature_descriptions.sort(key=lambda x: x['descriptiveness_score'], reverse=True)
        
        return feature_descriptions[:5]  # Top 5 características
    
    def _identify_semantic_descriptive_features(self, cluster_data: np.ndarray) -> List[Dict[str, Any]]:
        """Identificar características semánticas descriptivas (placeholder)."""
        # Para embeddings BERT, las características individuales no son interpretables directamente
        # Retornar información estadística general
        return [
            {
                'feature_name': 'Dimensionalidad Semántica',
                'mean_value': float(np.mean(cluster_data)),
                'std_value': float(np.std(cluster_data)),
                'level': 'Embedding BERT 384D',
                'descriptiveness_score': 1.0
            }
        ]
    
    def evaluate_granularity_adequacy(self, n_clusters: int, interpretability_ratio: float) -> Dict[str, Any]:
        """
        Evaluar adecuación de granularidad para explicabilidad.
        
        Args:
            n_clusters: Número de clusters
            interpretability_ratio: Ratio de clusters interpretables
            
        Returns:
            Dict con evaluación de granularidad
        """
        granularity_config = self.settings.granularity_config
        
        analysis = {
            'n_clusters': n_clusters,
            'minimum_k_threshold': granularity_config['minimum_k'],
            'meets_minimum_granularity': n_clusters >= granularity_config['minimum_k']
        }
        
        # Evaluar granularidad
        if n_clusters < granularity_config['minimum_k']:
            analysis['granularity_assessment'] = 'Insuficiente'
            analysis['granularity_score'] = max(0.0, n_clusters / granularity_config['minimum_k'] * 0.5)
        elif n_clusters in range(*granularity_config['optimal_k_range']):
            analysis['granularity_assessment'] = 'Óptima'
            analysis['granularity_score'] = 1.0
        elif n_clusters > granularity_config['optimal_k_range'][1]:
            analysis['granularity_assessment'] = 'Excesiva'
            # Penalizar granularidad excesiva
            excess = n_clusters - granularity_config['optimal_k_range'][1]
            penalty = min(0.5, excess * 0.1)
            analysis['granularity_score'] = max(0.5, 1.0 - penalty)
        else:
            analysis['granularity_assessment'] = 'Adecuada'
            analysis['granularity_score'] = 0.8
        
        # Ajustar por interpretabilidad
        final_score = analysis['granularity_score'] * interpretability_ratio
        analysis['final_granularity_score'] = final_score
        
        return analysis
    
    def evaluate_cluster_balance_interpretability(self, labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluar balance de clusters desde perspectiva de interpretabilidad.
        
        Args:
            labels: Etiquetas de clustering
            
        Returns:
            Dict con evaluación de balance
        """
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        
        if len(unique_labels) == 0:
            return {
                'balance_assessment': 'Sin clusters válidos',
                'balance_score': 0.0,
                'dominant_clusters_count': 0,
                'fragmented_clusters_count': 0
            }
        
        total_points = np.sum(counts)
        percentages = counts / total_points
        
        # Identificar problemas de balance
        dominant_threshold = self.settings.quality_thresholds['max_dominance_percent'] / 100
        fragmentation_threshold = self.settings.quality_thresholds['max_fragmentation_percent'] / 100
        
        dominant_clusters = percentages > dominant_threshold
        fragmented_clusters = percentages < fragmentation_threshold
        
        balance_analysis = {
            'dominant_clusters_count': int(np.sum(dominant_clusters)),
            'fragmented_clusters_count': int(np.sum(fragmented_clusters)),
            'max_cluster_percentage': float(np.max(percentages) * 100),
            'min_cluster_percentage': float(np.min(percentages) * 100),
        }
        
        # Calcular score de balance
        dominance_penalty = np.sum(percentages[dominant_clusters] - dominant_threshold) if np.any(dominant_clusters) else 0
        fragmentation_penalty = np.sum(fragmentation_threshold - percentages[fragmented_clusters]) if np.any(fragmented_clusters) else 0
        
        balance_score = max(0.0, 1.0 - dominance_penalty - fragmentation_penalty)
        balance_analysis['balance_score'] = balance_score
        
        # Evaluación cualitativa
        if balance_score > 0.8:
            balance_analysis['balance_assessment'] = 'Excelente'
        elif balance_score > 0.6:
            balance_analysis['balance_assessment'] = 'Bueno'
        elif balance_score > 0.4:
            balance_analysis['balance_assessment'] = 'Moderado'
        else:
            balance_analysis['balance_assessment'] = 'Problemático'
        
        return balance_analysis
    
    def calculate_global_interpretability_score(self, validation_results: Dict[str, Any]) -> float:
        """
        Calcular score global de interpretabilidad.
        
        Args:
            validation_results: Resultados completos de validación
            
        Returns:
            Score global de interpretabilidad [0, 1]
        """
        # Componentes del score
        interpretability_ratio = validation_results.get('interpretability_ratio', 0)
        granularity_score = validation_results.get('final_granularity_score', 0)
        balance_score = validation_results.get('balance_score', 0)
        
        # Promedio coherencia de clusters interpretables
        interpretable_clusters = [
            cv for cv in validation_results.get('cluster_validations', [])
            if cv.get('is_interpretable', False)
        ]
        
        avg_coherence = np.mean([cv.get('coherence_score', 0) for cv in interpretable_clusters]) if interpretable_clusters else 0
        
        # Score compuesto ponderado
        weights = {
            'interpretability_ratio': 0.4,
            'granularity_score': 0.25,
            'balance_score': 0.2,
            'avg_coherence': 0.15
        }
        
        global_score = (
            weights['interpretability_ratio'] * interpretability_ratio +
            weights['granularity_score'] * granularity_score +
            weights['balance_score'] * balance_score +
            weights['avg_coherence'] * avg_coherence
        )
        
        return max(0.0, min(1.0, global_score))
    
    def generate_interpretability_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """
        Generar recomendaciones para mejorar interpretabilidad.
        
        Args:
            validation_results: Resultados de validación
            
        Returns:
            Lista de recomendaciones específicas
        """
        recommendations = []
        
        # Evaluar granularidad
        n_clusters = validation_results.get('n_clusters', 0)
        if n_clusters < 5:
            recommendations.append(f"Incrementar número de clusters (actual: {n_clusters}, mínimo recomendado: 5)")
        elif n_clusters > 10:
            recommendations.append(f"Considerar reducir número de clusters (actual: {n_clusters}, óptimo: 5-8)")
        
        # Evaluar interpretabilidad
        interpretability_ratio = validation_results.get('interpretability_ratio', 0)
        if interpretability_ratio < 0.6:
            recommendations.append(f"Mejorar coherencia de clusters (solo {interpretability_ratio:.1%} son interpretables)")
        
        # Evaluar balance
        balance_score = validation_results.get('balance_score', 0)
        if balance_score < 0.6:
            recommendations.append("Mejorar balance de distribución de clusters")
        
        # Evaluar clusters dominantes o fragmentados
        if validation_results.get('dominant_clusters_count', 0) > 0:
            recommendations.append("Reducir clusters dominantes que concentran >50% de datos")
        
        if validation_results.get('fragmented_clusters_count', 0) > 0:
            recommendations.append("Eliminar clusters fragmentados con <3% de datos")
        
        # Recomendaciones por coherencia baja
        low_coherence_clusters = [
            cv for cv in validation_results.get('cluster_validations', [])
            if cv.get('coherence_score', 1) < 0.5
        ]
        
        if len(low_coherence_clusters) > 0:
            recommendations.append(f"Optimizar coherencia interna de {len(low_coherence_clusters)} clusters con baja coherencia")
        
        if not recommendations:
            recommendations.append("Sistema presenta interpretabilidad adecuada")
        
        return recommendations
    
    def classify_interpretability_level(self, global_score: float) -> str:
        """
        Clasificar nivel de interpretabilidad basado en score global.
        
        Args:
            global_score: Score global de interpretabilidad
            
        Returns:
            Clasificación cualitativa
        """
        if global_score >= 0.8:
            return "Excelente"
        elif global_score >= 0.6:
            return "Bueno" 
        elif global_score >= 0.4:
            return "Moderado"
        elif global_score >= 0.2:
            return "Bajo"
        else:
            return "Muy Bajo"
    
    def _create_empty_validation_result(self, domain: str, reason: str) -> Dict[str, Any]:
        """
        Crear resultado vacío para casos sin clusters válidos.
        
        Args:
            domain: Dominio de clustering
            reason: Razón del resultado vacío
            
        Returns:
            Dict con resultado vacío estandarizado
        """
        return {
            'domain': domain,
            'n_clusters': 0,
            'interpretable_clusters_count': 0,
            'interpretability_ratio': 0.0,
            'global_interpretability_score': 0.0,
            'interpretability_classification': 'Sin clusters',
            'cluster_validations': [],
            'improvement_recommendations': [reason],
            'validation_timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        }


# Instancia global del validador
interpretability_validator = InterpretabilityValidator()