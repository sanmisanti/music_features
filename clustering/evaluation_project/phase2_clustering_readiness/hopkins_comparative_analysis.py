#!/usr/bin/env python3
"""
Hopkins Comparative Analysis Module - FASE 2
============================================

Implementación especializada de Hopkins Statistic para análisis comparativo
entre espacios vectoriales de diferentes dimensionalidades.

El Hopkins Statistic mide la tendencia de un dataset hacia clustering
versus distribución aleatoria, proporcionando métrica fundamental para
evaluación de clustering readiness.
"""

import numpy as np
import logging
from typing import Dict, Tuple, List, Any
from scipy.spatial.distance import pdist, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class HopkinsComparativeAnalyzer:
    """
    Analizador especializado para cálculo y comparación de Hopkins Statistic
    entre múltiples espacios vectoriales.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Inicializar analizador Hopkins comparativo.
        
        Args:
            logger: Logger configurado para trazabilidad
        """
        self.logger = logger
        self.random_state = 42
        np.random.seed(self.random_state)
        
        self.logger.info("HopkinsComparativeAnalyzer inicializado")
    
    def calculate_hopkins_statistic(self, data: np.ndarray, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Calcular Hopkins Statistic para un espacio vectorial específico.
        
        El Hopkins Statistic se define como:
        H = U / (U + W)
        
        Donde:
        - U: suma de distancias desde puntos aleatorios uniformes a sus vecinos más cercanos
        - W: suma de distancias desde puntos de muestra a sus vecinos más cercanos
        
        Interpretación:
        - H ≈ 0.5: distribución aleatoria (no clustering)
        - H > 0.7: fuerte tendencia hacia clustering
        - H < 0.3: distribución regular/anti-clustering
        
        Args:
            data: Array de datos (n_samples, n_features)
            n_samples: Número de puntos para muestreo Hopkins
            
        Returns:
            Diccionario con Hopkins statistic y métricas auxiliares
        """
        self.logger.info(f"Calculando Hopkins Statistic para espacio {data.shape}")
        
        n_points, n_features = data.shape
        
        # Ajustar n_samples si es mayor que el dataset
        n_samples = min(n_samples, n_points // 2)
        
        self.logger.info(f"Usando {n_samples} puntos para cálculo Hopkins")
        
        try:
            # Paso 1: Generar puntos aleatorios uniformes en el mismo espacio
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            
            random_points = np.random.uniform(
                low=min_vals,
                high=max_vals,
                size=(n_samples, n_features)
            )
            
            # Paso 2: Seleccionar muestra aleatoria de puntos de datos
            sample_indices = np.random.choice(n_points, size=n_samples, replace=False)
            sample_points = data[sample_indices]
            
            # Paso 3: Calcular distancias U (puntos aleatorios a dataset)
            nn_uniform = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_uniform.fit(data)
            
            distances_uniform, _ = nn_uniform.kneighbors(random_points)
            U = np.sum(distances_uniform[:, 0])
            
            # Paso 4: Calcular distancias W (puntos de muestra a dataset)
            # Crear dataset sin los puntos de muestra para evitar distancia cero
            remaining_data = np.delete(data, sample_indices, axis=0)
            
            nn_sample = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_sample.fit(remaining_data)
            
            distances_sample, _ = nn_sample.kneighbors(sample_points)
            W = np.sum(distances_sample[:, 0])
            
            # Paso 5: Calcular Hopkins Statistic
            hopkins_statistic = U / (U + W)
            
            # Calcular métricas auxiliares para análisis
            mean_dist_uniform = np.mean(distances_uniform[:, 0])
            mean_dist_sample = np.mean(distances_sample[:, 0])
            std_dist_uniform = np.std(distances_uniform[:, 0])
            std_dist_sample = np.std(distances_sample[:, 0])
            
            results = {
                'hopkins_statistic': float(hopkins_statistic),
                'U_sum': float(U),
                'W_sum': float(W),
                'n_samples_used': n_samples,
                'mean_distance_uniform': float(mean_dist_uniform),
                'mean_distance_sample': float(mean_dist_sample),
                'std_distance_uniform': float(std_dist_uniform),
                'std_distance_sample': float(std_dist_sample),
                'clustering_tendency': self._interpret_hopkins(hopkins_statistic)
            }
            
            self.logger.info(f"Hopkins Statistic calculado: {hopkins_statistic:.4f}")
            self.logger.info(f"Interpretación: {results['clustering_tendency']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculando Hopkins Statistic: {e}")
            raise
    
    def calculate_hopkins_with_iterations(self, data: np.ndarray, n_samples: int = 1000, 
                                        n_iterations: int = 10) -> Dict[str, Any]:
        """
        Calcular Hopkins Statistic con múltiples iteraciones para estabilidad estadística.
        
        Args:
            data: Array de datos
            n_samples: Puntos por iteración
            n_iterations: Número de iteraciones
            
        Returns:
            Diccionario con estadísticas Hopkins robustas
        """
        self.logger.info(f"Calculando Hopkins con {n_iterations} iteraciones")
        
        hopkins_values = []
        all_results = []
        
        for i in range(n_iterations):
            self.logger.debug(f"Iteración Hopkins {i+1}/{n_iterations}")
            
            # Usar semilla diferente para cada iteración
            np.random.seed(self.random_state + i)
            
            result = self.calculate_hopkins_statistic(data, n_samples)
            hopkins_values.append(result['hopkins_statistic'])
            all_results.append(result)
        
        # Calcular estadísticas robustas
        hopkins_array = np.array(hopkins_values)
        
        robust_results = {
            'hopkins_mean': float(np.mean(hopkins_array)),
            'hopkins_std': float(np.std(hopkins_array)),
            'hopkins_median': float(np.median(hopkins_array)),
            'hopkins_min': float(np.min(hopkins_array)),
            'hopkins_max': float(np.max(hopkins_array)),
            'confidence_interval_95': self._calculate_confidence_interval(hopkins_array),
            'n_iterations': n_iterations,
            'stability_score': self._calculate_stability_score(hopkins_array),
            'all_iterations': all_results,
            'clustering_tendency_robust': self._interpret_hopkins(np.mean(hopkins_array))
        }
        
        self.logger.info(f"Hopkins robusto: {robust_results['hopkins_mean']:.4f} ± {robust_results['hopkins_std']:.4f}")
        self.logger.info(f"Intervalo confianza 95%: {robust_results['confidence_interval_95']}")
        
        return robust_results
    
    def analyze_comparative_hopkins(self, semantic_embeddings: np.ndarray, 
                                  musical_features: np.ndarray,
                                  n_samples: int = 1000, 
                                  n_iterations: int = 10) -> Dict[str, Any]:
        """
        Ejecutar análisis Hopkins comparativo entre espacios semántico y musical.
        
        Args:
            semantic_embeddings: Embeddings BERT 384D
            musical_features: Características musicales normalizadas 12D
            n_samples: Puntos por iteración Hopkins
            n_iterations: Iteraciones para estabilidad
            
        Returns:
            Diccionario con análisis comparativo completo
        """
        self.logger.info("INICIANDO ANÁLISIS HOPKINS COMPARATIVO")
        self.logger.info(f"Espacio semántico: {semantic_embeddings.shape}")
        self.logger.info(f"Espacio musical: {musical_features.shape}")
        
        # Análisis Hopkins para espacio semántico
        self.logger.info("Analizando clustering readiness - Espacio Semántico (384D)")
        semantic_results = self.calculate_hopkins_with_iterations(
            semantic_embeddings, n_samples, n_iterations
        )
        
        # Análisis Hopkins para espacio musical  
        self.logger.info("Analizando clustering readiness - Espacio Musical (12D)")
        musical_results = self.calculate_hopkins_with_iterations(
            musical_features, n_samples, n_iterations
        )
        
        # Análisis comparativo
        semantic_hopkins = semantic_results['hopkins_mean']
        musical_hopkins = musical_results['hopkins_mean']
        difference = musical_hopkins - semantic_hopkins
        
        # Análisis de significancia
        significance_analysis = self._analyze_statistical_significance(
            semantic_results, musical_results
        )
        
        comparative_results = {
            'semantic_analysis': semantic_results,
            'musical_analysis': musical_results,
            'comparative_metrics': {
                'semantic_hopkins': semantic_hopkins,
                'musical_hopkins': musical_hopkins,
                'difference': difference,
                'relative_improvement': (difference / semantic_hopkins) * 100 if semantic_hopkins > 0 else 0,
                'superiority_factor': musical_hopkins / semantic_hopkins if semantic_hopkins > 0 else float('inf')
            },
            'statistical_significance': significance_analysis,
            'architectural_implications': self._analyze_architectural_implications(
                semantic_hopkins, musical_hopkins, difference
            )
        }
        
        self.logger.info("ANÁLISIS HOPKINS COMPARATIVO COMPLETADO")
        self.logger.info(f"Resultado: Espacial Musical SUPERIOR por {difference:.4f} puntos")
        self.logger.info(f"Factor superioridad: {comparative_results['comparative_metrics']['superiority_factor']:.2f}x")
        
        return comparative_results
    
    def _interpret_hopkins(self, hopkins_value: float) -> str:
        """
        Interpretar valor Hopkins según rangos estándar.
        
        Args:
            hopkins_value: Valor Hopkins calculado
            
        Returns:
            Interpretación textual del clustering readiness
        """
        if hopkins_value > 0.75:
            return "Excellent clustering tendency"
        elif hopkins_value > 0.60:
            return "Good clustering tendency"  
        elif hopkins_value > 0.40:
            return "Moderate clustering tendency"
        elif hopkins_value < 0.25:
            return "Regular/Anti-clustering tendency"
        else:
            return "Random distribution (poor clustering readiness)"
    
    def _calculate_confidence_interval(self, values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calcular intervalo de confianza para valores Hopkins."""
        from scipy import stats
        
        n = len(values)
        mean = np.mean(values)
        std_error = stats.sem(values)
        
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * std_error
        
        return (float(mean - margin_error), float(mean + margin_error))
    
    def _calculate_stability_score(self, values: np.ndarray) -> float:
        """
        Calcular score de estabilidad basado en variabilidad de Hopkins.
        
        Score alto indica resultados consistentes entre iteraciones.
        """
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0
        
        # Convertir coeficiente de variación a score de estabilidad (0-1)
        stability_score = max(0, 1 - cv * 2)  # Factor 2 para escalar apropiadamente
        
        return float(stability_score)
    
    def _analyze_statistical_significance(self, semantic_results: Dict, musical_results: Dict) -> Dict[str, Any]:
        """
        Analizar significancia estadística de diferencias Hopkins.
        
        Args:
            semantic_results: Resultados Hopkins espacio semántico
            musical_results: Resultados Hopkins espacio musical
            
        Returns:
            Análisis de significancia estadística
        """
        from scipy import stats
        
        semantic_values = [r['hopkins_statistic'] for r in semantic_results['all_iterations']]
        musical_values = [r['hopkins_statistic'] for r in musical_results['all_iterations']]
        
        # Test t pareado para diferencias
        t_stat, p_value = stats.ttest_rel(musical_values, semantic_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(semantic_values) + np.var(musical_values)) / 2)
        cohens_d = (np.mean(musical_values) - np.mean(semantic_values)) / pooled_std
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant_at_01': p_value < 0.01,
            'significant_at_05': p_value < 0.05,
            'effect_size_interpretation': self._interpret_effect_size(cohens_d)
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpretar tamaño del efecto según Cohen's d."""
        abs_d = abs(cohens_d)
        
        if abs_d >= 0.8:
            return "Large effect size"
        elif abs_d >= 0.5:
            return "Medium effect size"  
        elif abs_d >= 0.2:
            return "Small effect size"
        else:
            return "Negligible effect size"
    
    def _analyze_architectural_implications(self, semantic_hopkins: float, 
                                         musical_hopkins: float, 
                                         difference: float) -> Dict[str, Any]:
        """
        Analizar implicaciones arquitecturales basadas en resultados Hopkins.
        
        Args:
            semantic_hopkins: Hopkins del espacio semántico
            musical_hopkins: Hopkins del espacio musical
            difference: Diferencia entre Hopkins
            
        Returns:
            Análisis de implicaciones para arquitectura del sistema
        """
        implications = {
            'clustering_semantic_recommended': semantic_hopkins > 0.6,
            'clustering_musical_recommended': musical_hopkins > 0.6,
            'hybrid_architecture_justified': difference > 0.2,
            'vectorization_primary_validated': semantic_hopkins < 0.6,
            'clustering_auxiliary_validated': musical_hopkins > 0.6,
            'architectural_decision': self._recommend_architecture(
                semantic_hopkins, musical_hopkins, difference
            )
        }
        
        return implications
    
    def _recommend_architecture(self, semantic_hopkins: float, musical_hopkins: float, 
                              difference: float) -> str:
        """
        Recomendar arquitectura basada en análisis Hopkins.
        
        Returns:
            Recomendación arquitectural fundamentada
        """
        if difference > 0.2 and musical_hopkins > 0.6 and semantic_hopkins < 0.6:
            return "hybrid_vectorization_primary_clustering_auxiliary"
        elif semantic_hopkins > 0.6 and musical_hopkins > 0.6:
            return "clustering_bidirectional"
        elif musical_hopkins > 0.6:
            return "clustering_musical_only"
        elif semantic_hopkins > 0.6:
            return "clustering_semantic_only"
        else:
            return "vectorization_complete"