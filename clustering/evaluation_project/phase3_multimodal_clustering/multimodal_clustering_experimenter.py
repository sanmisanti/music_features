#!/usr/bin/env python3
"""
Orquestador principal para experimentación de clustering multimodal FASE 3.
Coordina evaluación exhaustiva de ambos dominios y análisis cross-modal.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
import time

# Importar módulos propios
from algorithm_evaluator import DomainAlgorithmEvaluator, ClusteringResult
from interpretability_validator import InterpretabilityValidator
from cross_modal_analyzer import CrossModalAnalyzer, CrossModalAnalysisResult
from config.evaluation_metrics import multi_criteria_evaluator

class MultimodalClusteringExperimenter:
    """Orquestador de experimentación clustering multimodal."""
    
    def __init__(self, output_dir: str = "./results", verbose: bool = True):
        """
        Inicializar experimentador.
        
        Args:
            output_dir: Directorio para guardar resultados
            verbose: Si mostrar progreso detallado
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.logger = self._setup_logging()
        
        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        self.musical_evaluator = DomainAlgorithmEvaluator('musical', verbose=verbose)
        self.semantic_evaluator = DomainAlgorithmEvaluator('semantic', verbose=verbose)
        self.interpretability_validator = InterpretabilityValidator(verbose=verbose)
        self.cross_modal_analyzer = CrossModalAnalyzer(verbose=verbose)
        
        # Almacenar resultados
        self.musical_results: List[ClusteringResult] = []
        self.semantic_results: List[ClusteringResult] = []
        self.cross_modal_results: Dict[str, CrossModalAnalysisResult] = {}
        
        # Timestamp para esta ejecución
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _setup_logging(self) -> logging.Logger:
        """Configurar logging principal."""
        logger = logging.getLogger('MultimodalExperimenter')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[EXPERIMENTER] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cargar dataset multimodal unificado.
        
        Args:
            dataset_path: Ruta al archivo .pkl del dataset
            
        Returns:
            Tupla (semantic_embeddings, musical_features, track_ids)
        """
        self.logger.info(f"Cargando dataset desde: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
        
        # Cargar dataset completo
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Extraer arrays principales
        data = dataset['data']
        semantic_embeddings = data['semantic_embeddings']
        musical_features = data['musical_features_normalized']
        track_ids = data['track_ids']
        
        self.logger.info(f"Dataset cargado: {len(track_ids):,} canciones")
        self.logger.info(f"Semántico: {semantic_embeddings.shape}, Musical: {musical_features.shape}")
        
        return semantic_embeddings, musical_features, track_ids
    
    def run_musical_evaluation(self, X_musical: np.ndarray) -> List[ClusteringResult]:
        """Ejecutar evaluación exhaustiva del dominio musical."""
        self.logger.info("=== INICIANDO EVALUACIÓN MUSICAL ===")
        start_time = time.time()
        
        results = self.musical_evaluator.evaluate_all_configurations(X_musical)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Evaluación musical completada en {elapsed:.1f}s")
        self.logger.info(f"Mejores 3 configuraciones musicales:")
        
        for i, result in enumerate(results[:3]):
            self.logger.info(
                f"  {i+1}. {result.algorithm_name} (K={result.algorithm_params.get('k', 'N/A')}): "
                f"Score={result.composite_score:.3f}, Silhouette={result.silhouette_score:.3f}"
            )
        
        self.musical_results = results
        return results
    
    def run_semantic_evaluation(self, X_semantic: np.ndarray) -> List[ClusteringResult]:
        """Ejecutar evaluación exhaustiva del dominio semántico."""
        self.logger.info("=== INICIANDO EVALUACIÓN SEMÁNTICA ===")
        start_time = time.time()
        
        results = self.semantic_evaluator.evaluate_all_configurations(X_semantic)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Evaluación semántica completada en {elapsed:.1f}s")
        self.logger.info(f"Mejores 3 configuraciones semánticas:")
        
        for i, result in enumerate(results[:3]):
            self.logger.info(
                f"  {i+1}. {result.algorithm_name} (K={result.algorithm_params.get('k', 'N/A')}): "
                f"Score={result.composite_score:.3f}, Silhouette={result.silhouette_score:.3f}"
            )
        
        self.semantic_results = results
        return results
    
    def run_cross_modal_analysis(self, X_musical: np.ndarray, X_semantic: np.ndarray,
                                track_ids: np.ndarray, top_n: int = 3) -> Dict[str, CrossModalAnalysisResult]:
        """
        Ejecutar análisis cross-modal para las mejores configuraciones.
        
        Args:
            X_musical: Características musicales
            X_semantic: Embeddings semánticos
            track_ids: IDs de canciones
            top_n: Número de mejores configuraciones a analizar por dominio
            
        Returns:
            Dict con resultados cross-modal
        """
        self.logger.info("=== INICIANDO ANÁLISIS CROSS-MODAL ===")
        
        # Obtener mejores configuraciones
        top_musical = self.musical_results[:top_n]
        top_semantic = self.semantic_results[:top_n]
        
        cross_modal_results = {}
        
        # Analizar todas las combinaciones top musical x top semántico
        for i, musical_result in enumerate(top_musical):
            for j, semantic_result in enumerate(top_semantic):
                combination_id = f"M{i+1}_S{j+1}"
                
                self.logger.info(f"Analizando combinación {combination_id}: "
                               f"{musical_result.algorithm_name} x {semantic_result.algorithm_name}")
                
                # Ejecutar análisis cross-modal
                cross_result = self.cross_modal_analyzer.analyze_cross_modal_correspondence(
                    musical_result.labels, semantic_result.labels, track_ids
                )
                
                # Actualizar scores compuestos con información cross-modal
                updated_musical_score = multi_criteria_evaluator.compute_composite_score(
                    musical_result.silhouette_score,
                    musical_result.balance_score,
                    musical_result.interpretability_score,
                    cross_result.nmi_score,  # Incluir cross-modal
                    musical_result.granularity_bonus
                )
                
                updated_semantic_score = multi_criteria_evaluator.compute_composite_score(
                    semantic_result.silhouette_score,
                    semantic_result.balance_score,
                    semantic_result.interpretability_score,
                    cross_result.nmi_score,  # Incluir cross-modal
                    semantic_result.granularity_bonus
                )
                
                # Agregar información adicional al resultado cross-modal
                cross_result.musical_config = {
                    'algorithm': musical_result.algorithm_name,
                    'params': musical_result.algorithm_params,
                    'original_score': musical_result.composite_score,
                    'updated_score': updated_musical_score
                }
                
                cross_result.semantic_config = {
                    'algorithm': semantic_result.algorithm_name,
                    'params': semantic_result.algorithm_params,
                    'original_score': semantic_result.composite_score,
                    'updated_score': updated_semantic_score
                }
                
                cross_modal_results[combination_id] = cross_result
                
                self.logger.info(f"  NMI: {cross_result.nmi_score:.3f}, "
                               f"Correspondencias fuertes: {len(cross_result.strong_correspondences)}")
        
        self.cross_modal_results = cross_modal_results
        return cross_modal_results
    
    def run_interpretability_validation(self, X_musical: np.ndarray, X_semantic: np.ndarray,
                                      top_n: int = 3) -> Dict[str, Any]:
        """Validar interpretabilidad para las mejores configuraciones."""
        self.logger.info("=== VALIDANDO INTERPRETABILIDAD ===")
        
        interpretability_results = {}
        
        # Validar mejores configuraciones
        for i in range(min(top_n, len(self.musical_results), len(self.semantic_results))):
            config_id = f"config_{i+1}"
            
            musical_result = self.musical_results[i]
            semantic_result = self.semantic_results[i]
            
            self.logger.info(f"Validando interpretabilidad {config_id}")
            
            interpretations = self.interpretability_validator.validate_all_clusters(
                X_musical, musical_result.labels,
                X_semantic, semantic_result.labels
            )
            
            # Calcular métricas de interpretabilidad
            musical_interpretable = len([i for i in interpretations['musical'] if i.confidence_score > 0.5])
            semantic_interpretable = len([i for i in interpretations['semantic'] if i.confidence_score > 0.5])
            
            interpretability_results[config_id] = {
                'musical_config': {
                    'algorithm': musical_result.algorithm_name,
                    'params': musical_result.algorithm_params
                },
                'semantic_config': {
                    'algorithm': semantic_result.algorithm_name,
                    'params': semantic_result.algorithm_params
                },
                'interpretations': interpretations,
                'musical_interpretable_clusters': musical_interpretable,
                'semantic_interpretable_clusters': semantic_interpretable,
                'musical_interpretability_rate': musical_interpretable / len(interpretations['musical']) if interpretations['musical'] else 0.0,
                'semantic_interpretability_rate': semantic_interpretable / len(interpretations['semantic']) if interpretations['semantic'] else 0.0
            }
            
            self.logger.info(f"  Musical: {musical_interpretable}/{len(interpretations['musical'])} interpretables")
            self.logger.info(f"  Semántico: {semantic_interpretable}/{len(interpretations['semantic'])} interpretables")
        
        return interpretability_results
    
    def generate_comprehensive_report(self, interpretability_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generar reporte científico completo."""
        self.logger.info("Generando reporte científico completo")
        
        # Identificar mejor configuración cross-modal
        best_cross_modal = max(self.cross_modal_results.items(),
                              key=lambda x: x[1].nmi_score) if self.cross_modal_results else None
        
        # Crear reporte estructurado
        report = {
            'metadata': {
                'timestamp': self.timestamp,
                'total_configurations_evaluated': len(self.musical_results) + len(self.semantic_results),
                'dataset_size': len(self.musical_results[0].labels) if self.musical_results else 0
            },
            'musical_domain': {
                'total_configurations': len(self.musical_results),
                'best_configuration': self._extract_result_summary(self.musical_results[0]) if self.musical_results else None,
                'top_5_configurations': [self._extract_result_summary(r) for r in self.musical_results[:5]]
            },
            'semantic_domain': {
                'total_configurations': len(self.semantic_results),
                'best_configuration': self._extract_result_summary(self.semantic_results[0]) if self.semantic_results else None,
                'top_5_configurations': [self._extract_result_summary(r) for r in self.semantic_results[:5]]
            },
            'cross_modal_analysis': {
                'total_combinations_analyzed': len(self.cross_modal_results),
                'best_combination': {
                    'combination_id': best_cross_modal[0],
                    'nmi_score': best_cross_modal[1].nmi_score,
                    'strong_correspondences': len(best_cross_modal[1].strong_correspondences),
                    'correspondence_coverage': best_cross_modal[1].correspondence_coverage
                } if best_cross_modal else None,
                'all_combinations': {
                    combo_id: {
                        'nmi_score': result.nmi_score,
                        'adjusted_rand_score': result.adjusted_rand_score,
                        'strong_correspondences': len(result.strong_correspondences),
                        'correspondence_coverage': result.correspondence_coverage
                    } for combo_id, result in self.cross_modal_results.items()
                }
            },
            'interpretability_analysis': interpretability_results,
            'scientific_conclusions': self._generate_scientific_conclusions()
        }
        
        return report
    
    def _extract_result_summary(self, result: ClusteringResult) -> Dict[str, Any]:
        """Extraer resumen de un resultado de clustering."""
        return {
            'algorithm': result.algorithm_name,
            'parameters': result.algorithm_params,
            'composite_score': result.composite_score,
            'silhouette_score': result.silhouette_score,
            'balance_score': result.balance_score,
            'interpretability_score': result.interpretability_score,
            'n_clusters': result.n_clusters,
            'meets_granularity_criteria': result.meets_granularity_criteria,
            'execution_time': result.execution_time
        }
    
    def _generate_scientific_conclusions(self) -> Dict[str, Any]:
        """Generar conclusiones científicas basadas en resultados."""
        conclusions = {}
        
        if self.musical_results and self.semantic_results:
            # Comparar dominios
            best_musical_score = self.musical_results[0].composite_score
            best_semantic_score = self.semantic_results[0].composite_score
            
            conclusions['domain_comparison'] = {
                'musical_best_score': best_musical_score,
                'semantic_best_score': best_semantic_score,
                'superior_domain': 'musical' if best_musical_score > best_semantic_score else 'semantic',
                'score_difference': abs(best_musical_score - best_semantic_score)
            }
        
        if self.cross_modal_results:
            # Análisis cross-modal
            nmi_scores = [result.nmi_score for result in self.cross_modal_results.values()]
            avg_nmi = np.mean(nmi_scores)
            max_nmi = np.max(nmi_scores)
            
            conclusions['cross_modal_coherence'] = {
                'average_nmi': avg_nmi,
                'maximum_nmi': max_nmi,
                'coherence_assessment': 'High' if max_nmi > 0.6 else 'Moderate' if max_nmi > 0.3 else 'Low'
            }
        
        return conclusions
    
    def save_all_results(self) -> Dict[str, str]:
        """Guardar todos los resultados en múltiples formatos."""
        self.logger.info("Guardando todos los resultados")
        
        saved_files = {}
        
        # Guardar resultados de dominios individuales
        if self.musical_results:
            musical_files = self.musical_evaluator.save_results(
                self.musical_results, self.output_dir, self.timestamp
            )
            saved_files.update({f'musical_{k}': v for k, v in musical_files.items()})
        
        if self.semantic_results:
            semantic_files = self.semantic_evaluator.save_results(
                self.semantic_results, self.output_dir, self.timestamp
            )
            saved_files.update({f'semantic_{k}': v for k, v in semantic_files.items()})
        
        # Guardar análisis cross-modal
        if self.cross_modal_results:
            cross_modal_path = self.output_dir / f"cross_modal_analysis_{self.timestamp}.json"
            cross_modal_data = {}
            
            for combo_id, result in self.cross_modal_results.items():
                cross_modal_data[combo_id] = {
                    'nmi_score': result.nmi_score,
                    'adjusted_rand_score': result.adjusted_rand_score,
                    'n_musical_clusters': result.n_musical_clusters,
                    'n_semantic_clusters': result.n_semantic_clusters,
                    'strong_correspondences': len(result.strong_correspondences),
                    'correspondence_coverage': result.correspondence_coverage,
                    'avg_correspondence_strength': result.avg_correspondence_strength
                }
            
            with open(cross_modal_path, 'w') as f:
                json.dump(cross_modal_data, f, indent=2)
            
            saved_files['cross_modal_analysis'] = str(cross_modal_path)
        
        return saved_files
    
    def run_complete_experiment(self, dataset_path: str, run_cross_modal: bool = True,
                              top_n_cross_modal: int = 3) -> Dict[str, Any]:
        """
        Ejecutar experimento completo de clustering multimodal.
        
        Args:
            dataset_path: Ruta al dataset unificado
            run_cross_modal: Si ejecutar análisis cross-modal
            top_n_cross_modal: Número de configuraciones top para cross-modal
            
        Returns:
            Reporte científico completo
        """
        self.logger.info("=== INICIANDO EXPERIMENTO MULTIMODAL COMPLETO ===")
        experiment_start = time.time()
        
        # 1. Cargar dataset
        X_semantic, X_musical, track_ids = self.load_dataset(dataset_path)
        
        # 2. Evaluación exhaustiva por dominios
        self.run_musical_evaluation(X_musical)
        self.run_semantic_evaluation(X_semantic)
        
        # 3. Análisis cross-modal (opcional)
        if run_cross_modal:
            self.run_cross_modal_analysis(X_musical, X_semantic, track_ids, top_n_cross_modal)
        
        # 4. Validación de interpretabilidad
        interpretability_results = self.run_interpretability_validation(X_musical, X_semantic)
        
        # 5. Generar reporte científico
        comprehensive_report = self.generate_comprehensive_report(interpretability_results)
        
        # 6. Guardar todos los resultados
        saved_files = self.save_all_results()
        
        # 7. Guardar reporte principal
        report_path = self.output_dir / f"comprehensive_report_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        saved_files['comprehensive_report'] = str(report_path)
        
        experiment_elapsed = time.time() - experiment_start
        self.logger.info(f"=== EXPERIMENTO COMPLETADO EN {experiment_elapsed:.1f}s ===")
        self.logger.info(f"Resultados guardados en: {self.output_dir}")
        
        return comprehensive_report