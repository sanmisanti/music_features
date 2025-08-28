#!/usr/bin/env python3
"""
Evaluador de algoritmos especializado por dominio para FASE 3.
Maneja la evaluación sistemática de configuraciones algorítmicas.
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

from config.algorithms_config import algorithm_config
from config.evaluation_metrics import multi_criteria_evaluator

@dataclass
class ClusteringResult:
    """Estructura para resultados de clustering."""
    domain: str
    algorithm_name: str
    algorithm_params: Dict[str, Any]
    labels: np.ndarray
    execution_time: float
    n_clusters: int
    n_outliers: int
    
    # Métricas de calidad técnica
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    
    # Métricas de balance
    balance_score: float
    entropy_score: float
    
    # Métricas de interpretabilidad
    interpretability_score: float
    coherence_score: float
    
    # Métricas de granularidad
    granularity_bonus: float
    meets_granularity_criteria: bool
    
    # Score compuesto
    composite_score: float
    
    # Validez del resultado
    valid_result: bool
    error_message: Optional[str] = None

class DomainAlgorithmEvaluator:
    """Evaluador especializado para un dominio específico."""
    
    def __init__(self, domain: str, verbose: bool = True):
        """
        Inicializar evaluador de dominio.
        
        Args:
            domain: 'musical' o 'semantic'
            verbose: Si mostrar progreso detallado
        """
        if domain not in ['musical', 'semantic']:
            raise ValueError(f"Dominio debe ser 'musical' o 'semantic', recibido: {domain}")
        
        self.domain = domain
        self.verbose = verbose
        self.logger = self._setup_logging()
        
        # Configuraciones algorítmicas para este dominio
        self.algorithm_configs = algorithm_config.get_algorithm_configs(domain)
        
    def _setup_logging(self) -> logging.Logger:
        """Configurar logging para el evaluador."""
        logger = logging.getLogger(f'DomainEvaluator_{self.domain}')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[{self.domain.upper()}] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def evaluate_single_configuration(self, X: np.ndarray, algorithm_name: str,
                                    k: int = None, eps: float = None) -> ClusteringResult:
        """
        Evaluar una configuración algorítmica específica.
        
        Args:
            X: Datos de entrada (N, features)
            algorithm_name: Nombre del algoritmo
            k: Número de clusters (si aplica)
            eps: Parámetro epsilon para DBSCAN (si aplica)
            
        Returns:
            Resultado completo de la evaluación
        """
        self.logger.info(f"Evaluando {algorithm_name} con K={k}, eps={eps}")
        
        try:
            # Crear instancia del algoritmo
            algorithm = algorithm_config.get_algorithm_instance(
                self.domain, algorithm_name, k=k, eps=eps
            )
            
            # Ejecutar clustering con medición de tiempo
            start_time = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = algorithm.fit_predict(X)
            execution_time = time.time() - start_time
            
            # Obtener parámetros utilizados
            algorithm_params = {
                'algorithm': algorithm_name,
                'k': k,
                'eps': eps,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
            # Calcular métricas básicas
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels != -1])
            n_outliers = np.sum(labels == -1)
            
            # Evaluación de calidad técnica
            quality_metrics = multi_criteria_evaluator.evaluate_clustering_quality(
                X, labels, self.domain
            )
            
            if not quality_metrics['valid']:
                return ClusteringResult(
                    domain=self.domain,
                    algorithm_name=algorithm_name,
                    algorithm_params=algorithm_params,
                    labels=labels,
                    execution_time=execution_time,
                    n_clusters=n_clusters,
                    n_outliers=n_outliers,
                    silhouette_score=-1.0,
                    calinski_harabasz_score=0.0,
                    davies_bouldin_score=10.0,
                    balance_score=0.0,
                    entropy_score=0.0,
                    interpretability_score=0.0,
                    coherence_score=0.0,
                    granularity_bonus=0.0,
                    meets_granularity_criteria=False,
                    composite_score=0.0,
                    valid_result=False,
                    error_message="Invalid clustering result"
                )
            
            # Evaluación de balance
            balance_metrics = multi_criteria_evaluator.evaluate_balance_distribution(labels)
            
            # Evaluación de interpretabilidad
            interpretability_metrics = multi_criteria_evaluator.evaluate_interpretability(
                X, labels, self.domain
            )
            
            # Evaluación de granularidad
            granularity_metrics = multi_criteria_evaluator.evaluate_granularity_bonus(labels)
            
            # Calcular score compuesto
            composite_score = multi_criteria_evaluator.compute_composite_score(
                quality_metrics['silhouette_score'],
                balance_metrics['balance_score'],
                interpretability_metrics['interpretability_score'],
                0.0,  # cross_modal_score se calcula después
                granularity_metrics['granularity_bonus']
            )
            
            result = ClusteringResult(
                domain=self.domain,
                algorithm_name=algorithm_name,
                algorithm_params=algorithm_params,
                labels=labels,
                execution_time=execution_time,
                n_clusters=n_clusters,
                n_outliers=n_outliers,
                silhouette_score=quality_metrics['silhouette_score'],
                calinski_harabasz_score=quality_metrics['calinski_harabasz_score'],
                davies_bouldin_score=quality_metrics['davies_bouldin_score'],
                balance_score=balance_metrics['balance_score'],
                entropy_score=balance_metrics['entropy_score'],
                interpretability_score=interpretability_metrics['interpretability_score'],
                coherence_score=interpretability_metrics['coherence_score'],
                granularity_bonus=granularity_metrics['granularity_bonus'],
                meets_granularity_criteria=granularity_metrics['meets_granularity_criteria'],
                composite_score=composite_score,
                valid_result=True
            )
            
            self.logger.info(
                f"Completado: Score={composite_score:.3f}, "
                f"Silhouette={quality_metrics['silhouette_score']:.3f}, "
                f"K={n_clusters}, Tiempo={execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating {algorithm_name}: {e}")
            
            return ClusteringResult(
                domain=self.domain,
                algorithm_name=algorithm_name,
                algorithm_params={'algorithm': algorithm_name, 'k': k, 'eps': eps},
                labels=np.array([-1] * len(X)),
                execution_time=0.0,
                n_clusters=0,
                n_outliers=len(X),
                silhouette_score=-1.0,
                calinski_harabasz_score=0.0,
                davies_bouldin_score=10.0,
                balance_score=0.0,
                entropy_score=0.0,
                interpretability_score=0.0,
                coherence_score=0.0,
                granularity_bonus=0.0,
                meets_granularity_criteria=False,
                composite_score=0.0,
                valid_result=False,
                error_message=str(e)
            )
    
    def evaluate_all_configurations(self, X: np.ndarray) -> List[ClusteringResult]:
        """
        Evaluar todas las configuraciones algorítmicas para este dominio.
        
        Args:
            X: Datos de entrada
            
        Returns:
            Lista de resultados ordenados por score compuesto
        """
        self.logger.info(f"Iniciando evaluación exhaustiva para dominio {self.domain}")
        self.logger.info(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
        
        results = []
        total_configs = 0
        
        # Contar configuraciones totales
        for algorithm_name, config in self.algorithm_configs.items():
            if 'k_range' in config:
                total_configs += len(config['k_range'])
            elif 'eps_range' in config:
                total_configs += len(config['eps_range'])
        
        self.logger.info(f"Total configuraciones a evaluar: {total_configs}")
        
        config_count = 0
        for algorithm_name, config in self.algorithm_configs.items():
            self.logger.info(f"Evaluando algoritmo: {algorithm_name}")
            
            if 'k_range' in config:
                # Algoritmos con parámetro K
                for k in config['k_range']:
                    config_count += 1
                    self.logger.info(f"Progreso: {config_count}/{total_configs}")
                    
                    result = self.evaluate_single_configuration(X, algorithm_name, k=k)
                    results.append(result)
                    
            elif 'eps_range' in config:
                # Algoritmos DBSCAN con parámetro eps
                for eps in config['eps_range']:
                    config_count += 1
                    self.logger.info(f"Progreso: {config_count}/{total_configs}")
                    
                    result = self.evaluate_single_configuration(X, algorithm_name, eps=eps)
                    results.append(result)
        
        # Ordenar por score compuesto (descendente)
        results.sort(key=lambda r: r.composite_score, reverse=True)
        
        self.logger.info(f"Evaluación completada. Mejor score: {results[0].composite_score:.3f}")
        
        return results
    
    def get_top_configurations(self, results: List[ClusteringResult], 
                              top_n: int = 5) -> List[ClusteringResult]:
        """Obtener las mejores N configuraciones."""
        valid_results = [r for r in results if r.valid_result]
        return valid_results[:top_n]
    
    def export_results_to_dataframe(self, results: List[ClusteringResult]) -> pd.DataFrame:
        """
        Exportar resultados a DataFrame para análisis.
        
        Args:
            results: Lista de resultados
            
        Returns:
            DataFrame con todos los resultados
        """
        rows = []
        for result in results:
            row = asdict(result)
            # Expandir algorithm_params
            row.update(result.algorithm_params)
            # Remover campos complejos
            row.pop('labels', None)
            row.pop('algorithm_params', None)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results(self, results: List[ClusteringResult], 
                    output_dir: Path, timestamp: str) -> Dict[str, str]:
        """
        Guardar resultados en múltiples formatos.
        
        Args:
            results: Lista de resultados
            output_dir: Directorio de salida
            timestamp: Timestamp para nombres de archivo
            
        Returns:
            Dict con rutas de archivos guardados
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # DataFrame con todos los resultados
        df = self.export_results_to_dataframe(results)
        csv_path = output_dir / f"{self.domain}_clustering_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        saved_files['results_csv'] = str(csv_path)
        
        # Top 5 configuraciones
        top_5 = self.get_top_configurations(results, top_n=5)
        top_5_df = self.export_results_to_dataframe(top_5)
        top_5_path = output_dir / f"{self.domain}_top5_configurations_{timestamp}.csv"
        top_5_df.to_csv(top_5_path, index=False)
        saved_files['top5_csv'] = str(top_5_path)
        
        # Labels de la mejor configuración
        if results and results[0].valid_result:
            best_labels_path = output_dir / f"{self.domain}_best_labels_{timestamp}.npy"
            np.save(best_labels_path, results[0].labels)
            saved_files['best_labels'] = str(best_labels_path)
        
        self.logger.info(f"Resultados guardados en: {output_dir}")
        
        return saved_files