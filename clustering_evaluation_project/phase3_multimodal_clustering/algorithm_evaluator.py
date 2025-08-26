"""
Evaluador Especializado de Algoritmos para Clustering Multimodal
===============================================================

Implementa evaluación sistemática de algoritmos de clustering con
optimizaciones específicas por dimensionalidad y prioridad en
interpretabilidad sobre métricas puras.

Autor: Proyecto FASE 3 - Sistema Clustering Multimodal
Fecha: Agosto 2025
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import time
import logging

from .config.algorithms_config import algorithms_config
from .config.evaluation_metrics import evaluation_metrics
from .config.interpretability_settings import interpretability_settings


class AlgorithmEvaluator:
    """
    Evaluador especializado de algoritmos de clustering multimodal.
    
    Ejecuta evaluación sistemática de configuraciones algorítmicas
    con optimizaciones por dimensionalidad y enfoque en interpretabilidad.
    """
    
    def __init__(self, dataset_path: str, verbose: bool = True):
        """
        Inicializar evaluador con dataset unificado multimodal.
        
        Args:
            dataset_path: Ruta al dataset unificado (.pkl)
            verbose: Activar logging detallado
        """
        self.dataset_path = dataset_path
        self.verbose = verbose
        
        # Configurar logging
        self.logger = self._setup_logging()
        
        # Cargar dataset
        self.dataset = self._load_unified_dataset()
        
        # Extraer componentes
        self.musical_features = self.dataset['musical_features_normalized']
        self.semantic_embeddings = self.dataset['semantic_embeddings']
        self.track_ids = self.dataset['track_ids']
        
        # Validar integridad
        self._validate_dataset_integrity()
        
        # Inicializar resultados
        self.evaluation_results = {
            'musical': [],
            'semantic': [],
            'metadata': {
                'dataset_path': dataset_path,
                'n_samples': len(self.track_ids),
                'evaluation_timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            }
        }
        
        self.logger.info(f"AlgorithmEvaluator inicializado con {len(self.track_ids)} muestras")
    
    def _setup_logging(self) -> logging.Logger:
        """Configurar logging para evaluador."""
        logger = logging.getLogger(f'AlgorithmEvaluator_{id(self)}')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_unified_dataset(self) -> Dict[str, Any]:
        """
        Cargar dataset unificado multimodal.
        
        Returns:
            Dict con componentes del dataset
        """
        try:
            with open(self.dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            
            self.logger.info(f"Dataset cargado exitosamente desde {self.dataset_path}")
            return dataset
        
        except Exception as e:
            self.logger.error(f"Error cargando dataset: {e}")
            raise RuntimeError(f"No se pudo cargar dataset desde {self.dataset_path}")
    
    def _validate_dataset_integrity(self) -> None:
        """Validar integridad del dataset cargado."""
        required_keys = ['musical_features_normalized', 'semantic_embeddings', 'track_ids']
        
        for key in required_keys:
            if key not in self.dataset:
                raise ValueError(f"Dataset no contiene clave requerida: {key}")
        
        # Verificar dimensiones
        n_samples = len(self.track_ids)
        
        if self.musical_features.shape[0] != n_samples:
            raise ValueError("Inconsistencia en número de muestras musicales")
        
        if self.semantic_embeddings.shape[0] != n_samples:
            raise ValueError("Inconsistencia en número de muestras semánticas")
        
        # Verificar dimensionalidad esperada
        if self.musical_features.shape[1] != 12:
            self.logger.warning(f"Dimensionalidad musical inesperada: {self.musical_features.shape[1]} (esperado: 12)")
        
        if self.semantic_embeddings.shape[1] != 384:
            self.logger.warning(f"Dimensionalidad semántica inesperada: {self.semantic_embeddings.shape[1]} (esperado: 384)")
        
        self.logger.info(f"Dataset validado: {n_samples} muestras, {self.musical_features.shape[1]}D musical, {self.semantic_embeddings.shape[1]}D semántico")
    
    def evaluate_single_algorithm(self, domain: str, algorithm_name: str, k: int) -> Dict[str, Any]:
        """
        Evaluar algoritmo específico en dominio determinado.
        
        Args:
            domain: 'musical' o 'semantic'
            algorithm_name: Nombre del algoritmo según configuración
            k: Número de clusters objetivo
            
        Returns:
            Dict con resultados de evaluación completa
        """
        self.logger.info(f"Evaluando {algorithm_name} en dominio {domain} con K={k}")
        
        start_time = time.time()
        
        try:
            # Obtener datos y algoritmo
            X = self.musical_features if domain == 'musical' else self.semantic_embeddings
            algorithm = algorithms_config.create_algorithm_instance(domain, algorithm_name, k)
            
            # Manejo especial para DBSCAN
            if algorithm_name.startswith('dbscan'):
                labels = self._evaluate_dbscan(domain, algorithm, X)
                k_effective = len(np.unique(labels[labels != -1]))
            else:
                # Clustering estándar
                labels = algorithm.fit_predict(X)
                k_effective = len(np.unique(labels))
            
            # Evaluación completa
            results = evaluation_metrics.evaluate_clustering_complete(X, labels, domain, k)
            
            # Agregar información algoritmo
            results.update({
                'algorithm_name': algorithm_name,
                'algorithm_description': algorithms_config.get_algorithm_description(domain, algorithm_name),
                'k_effective': k_effective,
                'execution_time_seconds': time.time() - start_time,
                'n_samples': len(X)
            })
            
            # Generar etiqueta automática si es interpretable
            if results['interpretability_score'] > 0.5:
                results['auto_label'] = self._generate_cluster_label(domain, X, labels)
            else:
                results['auto_label'] = f"Cluster {domain.title()} K={k_effective}"
            
            self.logger.info(f"Evaluación completada: Silhouette={results['silhouette_score']:.3f}, "
                           f"Balance={results['balance_distribution_score']:.3f}, "
                           f"Interpretabilidad={results['interpretability_score']:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluando {algorithm_name}: {e}")
            return self._create_error_result(domain, algorithm_name, k, str(e), time.time() - start_time)
    
    def _evaluate_dbscan(self, domain: str, dbscan_algorithm: DBSCAN, X: np.ndarray) -> np.ndarray:
        """
        Evaluar DBSCAN con optimización de eps.
        
        Args:
            domain: 'musical' o 'semantic'
            dbscan_algorithm: Instancia configurada de DBSCAN
            X: Matriz de características
            
        Returns:
            Etiquetas de clustering optimizadas
        """
        eps_range = algorithms_config.get_eps_optimization_range(domain)
        best_labels = None
        best_score = -1.0
        
        for eps in eps_range:
            # Actualizar eps
            dbscan_algorithm.eps = eps
            
            try:
                labels = dbscan_algorithm.fit_predict(X)
                unique_labels = np.unique(labels[labels != -1])
                
                # Validar clustering válido
                if len(unique_labels) >= 2:
                    # Calcular silhouette para comparar
                    silhouette = evaluation_metrics.calculate_traditional_metrics(X, labels)['silhouette_score']
                    
                    if silhouette > best_score:
                        best_score = silhouette
                        best_labels = labels.copy()
                        
            except Exception:
                continue
        
        # Si no se encontró configuración válida, usar configuración original
        if best_labels is None:
            self.logger.warning(f"DBSCAN {domain}: No se encontró eps óptimo, usando configuración original")
            best_labels = dbscan_algorithm.fit_predict(X)
        
        return best_labels
    
    def _generate_cluster_label(self, domain: str, X: np.ndarray, labels: np.ndarray) -> str:
        """
        Generar etiqueta automática para clustering interpretable.
        
        Args:
            domain: 'musical' o 'semantic'
            X: Matriz de características
            labels: Etiquetas de clustering
            
        Returns:
            Etiqueta descriptiva automática
        """
        try:
            unique_labels = np.unique(labels[labels != -1])
            
            if len(unique_labels) == 0:
                return f"Sin Clusters {domain.title()}"
            
            # Seleccionar cluster más grande para etiqueta representativa
            cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
            largest_cluster_label = max(cluster_sizes, key=lambda x: x[1])[0]
            
            cluster_mask = labels == largest_cluster_label
            cluster_data = X[cluster_mask]
            
            if domain == 'musical':
                cluster_mean = np.mean(cluster_data, axis=0)
                cluster_std = np.std(cluster_data, axis=0)
                return interpretability_settings.generate_musical_cluster_label(cluster_mean, cluster_std)
            
            elif domain == 'semantic':
                coherence_score = evaluation_metrics._calculate_semantic_coherence(cluster_data)
                return interpretability_settings.generate_semantic_cluster_label(cluster_data[:5], coherence_score)
            
            else:
                return f"Cluster {domain.title()}"
                
        except Exception as e:
            self.logger.warning(f"Error generando etiqueta automática: {e}")
            return f"Cluster {domain.title()} K={len(np.unique(labels[labels != -1]))}"
    
    def _create_error_result(self, domain: str, algorithm_name: str, k: int, 
                           error_message: str, execution_time: float) -> Dict[str, Any]:
        """
        Crear resultado de error estandarizado.
        
        Args:
            domain: 'musical' o 'semantic'
            algorithm_name: Nombre del algoritmo
            k: Valor K objetivo
            error_message: Mensaje de error
            execution_time: Tiempo de ejecución
            
        Returns:
            Dict con resultado de error
        """
        return {
            'domain': domain,
            'algorithm_name': algorithm_name,
            'k_target': k,
            'k_effective': 0,
            'silhouette_score': -1.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': float('inf'),
            'balance_distribution_score': 0.0,
            'interpretability_score': 0.0,
            'granularity_bonus': 0.0,
            'composite_score_partial': 0.0,
            'n_clusters': 0,
            'n_noise_points': 0,
            'auto_label': f"Error: {domain.title()}",
            'execution_time_seconds': execution_time,
            'error': True,
            'error_message': error_message,
            'evaluation_timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        }
    
    def evaluate_domain_complete(self, domain: str, max_parallel: int = 1) -> List[Dict[str, Any]]:
        """
        Evaluar todos los algoritmos y valores K para un dominio.
        
        Args:
            domain: 'musical' o 'semantic'
            max_parallel: Número máximo de evaluaciones paralelas (futuro)
            
        Returns:
            Lista con resultados de todas las evaluaciones
        """
        self.logger.info(f"Iniciando evaluación completa del dominio {domain}")
        
        algorithms = algorithms_config.get_algorithm_configs(domain)
        k_range = algorithms_config.get_k_range(domain)
        
        domain_results = []
        total_experiments = 0
        
        # Contar experimentos
        for algorithm_name in algorithms.keys():
            if algorithm_name.startswith('dbscan'):
                total_experiments += 1  # DBSCAN no usa K
            else:
                total_experiments += len(k_range)
        
        self.logger.info(f"Ejecutando {total_experiments} experimentos para dominio {domain}")
        
        experiment_count = 0
        
        for algorithm_name in algorithms.keys():
            if algorithm_name.startswith('dbscan'):
                # DBSCAN: un experimento sin K específico
                experiment_count += 1
                self.logger.info(f"Progreso: {experiment_count}/{total_experiments} - {algorithm_name}")
                
                result = self.evaluate_single_algorithm(domain, algorithm_name, k=0)  # K=0 para DBSCAN
                domain_results.append(result)
                
            else:
                # Algoritmos con K
                for k in k_range:
                    experiment_count += 1
                    self.logger.info(f"Progreso: {experiment_count}/{total_experiments} - {algorithm_name} K={k}")
                    
                    result = self.evaluate_single_algorithm(domain, algorithm_name, k)
                    domain_results.append(result)
        
        # Guardar resultados del dominio
        self.evaluation_results[domain] = domain_results
        
        self.logger.info(f"Evaluación del dominio {domain} completada: {len(domain_results)} resultados")
        
        return domain_results
    
    def evaluate_multimodal_complete(self) -> Dict[str, Any]:
        """
        Evaluar clustering completo multimodal (ambos dominios).
        
        Returns:
            Dict con resultados completos de evaluación
        """
        self.logger.info("Iniciando evaluación multimodal completa")
        start_time = time.time()
        
        # Evaluar cada dominio
        musical_results = self.evaluate_domain_complete('musical')
        semantic_results = self.evaluate_domain_complete('semantic')
        
        # Agregar metadatos finales
        self.evaluation_results['metadata'].update({
            'total_execution_time_seconds': time.time() - start_time,
            'musical_experiments': len(musical_results),
            'semantic_experiments': len(semantic_results),
            'total_experiments': len(musical_results) + len(semantic_results),
            'evaluation_completed': True
        })
        
        self.logger.info(f"Evaluación multimodal completada en {self.evaluation_results['metadata']['total_execution_time_seconds']:.1f} segundos")
        self.logger.info(f"Total experimentos: {self.evaluation_results['metadata']['total_experiments']}")
        
        return self.evaluation_results
    
    def get_top_configurations(self, domain: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Obtener top-N configuraciones por score compuesto para dominio.
        
        Args:
            domain: 'musical' o 'semantic'
            top_n: Número de configuraciones top a retornar
            
        Returns:
            Lista de top configuraciones ordenadas por score
        """
        if domain not in self.evaluation_results:
            self.logger.warning(f"No hay resultados para dominio {domain}")
            return []
        
        results = self.evaluation_results[domain]
        
        # Filtrar errores y ordenar por score compuesto
        valid_results = [r for r in results if not r.get('error', False)]
        valid_results.sort(key=lambda x: x.get('composite_score_partial', 0), reverse=True)
        
        return valid_results[:top_n]
    
    def save_results(self, output_path: str) -> None:
        """
        Guardar resultados de evaluación en archivo.
        
        Args:
            output_path: Ruta para guardar resultados (.pkl)
        """
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.evaluation_results, f)
            
            self.logger.info(f"Resultados guardados en {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error guardando resultados: {e}")
            raise


# Función de conveniencia para evaluación rápida
def evaluate_multimodal_clustering(dataset_path: str, 
                                 output_path: Optional[str] = None,
                                 verbose: bool = True) -> Dict[str, Any]:
    """
    Función de conveniencia para evaluación multimodal completa.
    
    Args:
        dataset_path: Ruta al dataset unificado
        output_path: Ruta opcional para guardar resultados
        verbose: Activar logging detallado
        
    Returns:
        Resultados completos de evaluación
    """
    evaluator = AlgorithmEvaluator(dataset_path, verbose=verbose)
    results = evaluator.evaluate_multimodal_complete()
    
    if output_path:
        evaluator.save_results(output_path)
    
    return results