#!/usr/bin/env python3
"""
Clustering Readiness Assessment Comparativo - FASE 2
====================================================

Sistema principal de evaluación comparativa entre espacios vectoriales
semántico (384D) y musical (12D) para validación empírica de la 
arquitectura híbrida propuesta.

Objetivo: Demostrar que el espacio musical posee clustering readiness
significativamente superior al espacio semántico, justificando la
implementación de clustering auxiliar únicamente en el dominio musical.
"""

import sys
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional

# Importar módulos auxiliares del sistema
from hopkins_comparative_analysis import HopkinsComparativeAnalyzer
from dimensionality_impact_assessment import DimensionalityAssessment
from clustering_readiness_visualizer import ClusteringReadinessVisualizer
from statistical_validation import StatisticalValidator
from performance_predictor import PerformancePredictor

class ClusteringReadinessComparativeEvaluator:
    """
    Sistema integral de evaluación comparativa de clustering readiness
    entre espacios vectoriales de diferentes dimensionalidades.
    """
    
    def __init__(self, unified_dataset_path: str, output_dir: str = None):
        """
        Inicializar evaluador comparativo con dataset unificado.
        
        Args:
            unified_dataset_path: Ruta al dataset multimodal unificado
            output_dir: Directorio para outputs (por defecto, mismo directorio)
        """
        self.unified_dataset_path = Path(unified_dataset_path)
        self.output_dir = Path(output_dir) if output_dir else self.unified_dataset_path.parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configurar logging detallado
        self.setup_logging()
        
        # Inicializar componentes especializados
        self.hopkins_analyzer = HopkinsComparativeAnalyzer(self.logger)
        self.dimensionality_assessor = DimensionalityAssessment(self.logger)
        self.visualizer = ClusteringReadinessVisualizer(self.output_dir, self.logger)
        self.statistical_validator = StatisticalValidator(self.logger)
        self.performance_predictor = PerformancePredictor(self.logger)
        
        # Storage para resultados
        self.unified_data = None
        self.semantic_embeddings = None
        self.musical_features = None
        self.evaluation_results = {}
        
        self.logger.info("ClusteringReadinessComparativeEvaluator inicializado")
        self.logger.info(f"Dataset path: {self.unified_dataset_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Configurar sistema de logging detallado para trazabilidad."""
        log_file = self.output_dir / f"clustering_readiness_evaluation_{self.timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Sistema de logging configurado exitosamente")
    
    def load_unified_dataset(self):
        """
        Cargar dataset multimodal unificado y extraer componentes
        para evaluación comparativa.
        """
        self.logger.info("Iniciando carga de dataset multimodal unificado")
        
        try:
            with open(self.unified_dataset_path, 'rb') as f:
                self.unified_data = pickle.load(f)
            
            # Extraer componentes principales
            self.semantic_embeddings = self.unified_data['data']['semantic_embeddings']
            self.musical_features = self.unified_data['data']['musical_features_normalized']
            
            # Validar dimensiones
            semantic_shape = self.semantic_embeddings.shape
            musical_shape = self.musical_features.shape
            
            self.logger.info(f"Dataset cargado exitosamente")
            self.logger.info(f"Embeddings semánticos: {semantic_shape}")
            self.logger.info(f"Características musicales: {musical_shape}")
            
            # Verificar alineación
            if semantic_shape[0] != musical_shape[0]:
                raise ValueError(f"Desalineación en número de muestras: {semantic_shape[0]} vs {musical_shape[0]}")
            
            # Verificar dimensionalidades esperadas
            if semantic_shape[1] != 384:
                self.logger.warning(f"Dimensionalidad semántica inesperada: {semantic_shape[1]} (esperado: 384)")
            
            if musical_shape[1] != 12:
                self.logger.warning(f"Dimensionalidad musical inesperada: {musical_shape[1]} (esperado: 12)")
            
            self.logger.info("Validación de dataset completada exitosamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando dataset unificado: {e}")
            return False
    
    def execute_comparative_evaluation(self):
        """
        Ejecutar evaluación comparativa completa entre espacios
        semántico y musical.
        """
        self.logger.info("INICIANDO EVALUACIÓN COMPARATIVA DE CLUSTERING READINESS")
        
        # Paso 1: Análisis Hopkins Comparativo
        self.logger.info("[PASO 1] Ejecutando análisis Hopkins comparativo")
        hopkins_results = self.hopkins_analyzer.analyze_comparative_hopkins(
            self.semantic_embeddings, 
            self.musical_features
        )
        
        self.evaluation_results['hopkins_analysis'] = hopkins_results
        self.logger.info(f"Hopkins semántico: {hopkins_results['comparative_metrics']['semantic_hopkins']:.4f}")
        self.logger.info(f"Hopkins musical: {hopkins_results['comparative_metrics']['musical_hopkins']:.4f}")
        self.logger.info(f"Diferencia: {hopkins_results['comparative_metrics']['difference']:.4f}")
        
        # Paso 2: Evaluación de Efectos Dimensionales
        self.logger.info("[PASO 2] Evaluando efectos dimensionales")
        dimensionality_results = self.dimensionality_assessor.assess_dimensionality_impact(
            self.semantic_embeddings,
            self.musical_features
        )
        
        self.evaluation_results['dimensionality_analysis'] = dimensionality_results
        self.logger.info("Análisis dimensional completado")
        
        # Paso 3: Validación Estadística
        self.logger.info("[PASO 3] Ejecutando validación estadística")
        statistical_results = self.statistical_validator.validate_comparative_significance(
            hopkins_results,
            dimensionality_results
        )
        
        self.evaluation_results['statistical_validation'] = statistical_results
        self.logger.info(f"Significancia estadística: p = {statistical_results.get('p_value', 'N/A')}")
        
        # Paso 4: Predicción de Performance
        self.logger.info("[PASO 4] Prediciendo performance de clustering")
        performance_results = self.performance_predictor.predict_clustering_performance(
            hopkins_results,
            dimensionality_results
        )
        
        self.evaluation_results['performance_prediction'] = performance_results
        self.logger.info("Predicción de performance completada")
        
        # Paso 5: Generación de Visualizaciones
        self.logger.info("[PASO 5] Generando visualizaciones científicas")
        visualization_results = self.visualizer.generate_comparative_visualizations(
            self.semantic_embeddings,
            self.musical_features,
            self.evaluation_results
        )
        
        self.evaluation_results['visualizations'] = visualization_results
        self.logger.info("Visualizaciones generadas exitosamente")
        
        self.logger.info("EVALUACIÓN COMPARATIVA COMPLETADA EXITOSAMENTE")
    
    def generate_technical_report(self):
        """
        Generar reporte técnico comprehensive con todos los resultados
        de la evaluación comparativa.
        """
        self.logger.info("Generando reporte técnico comprehensive")
        
        report = {
            'metadata': {
                'evaluation_timestamp': self.timestamp,
                'dataset_source': str(self.unified_dataset_path),
                'sample_size': self.semantic_embeddings.shape[0],
                'semantic_dimensions': self.semantic_embeddings.shape[1],
                'musical_dimensions': self.musical_features.shape[1]
            },
            'evaluation_results': self.evaluation_results,
            'technical_conclusions': self._generate_technical_conclusions(),
            'recommendations': self._generate_architectural_recommendations()
        }
        
        # Guardar reporte en JSON
        report_path = self.output_dir / f"clustering_readiness_comparative_report_{self.timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        # Generar reporte markdown para documentación
        markdown_report = self._generate_markdown_report(report)
        markdown_path = self.output_dir / f"clustering_readiness_comparative_report_{self.timestamp}.md"
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        self.logger.info(f"Reporte técnico generado: {report_path.name}")
        self.logger.info(f"Reporte markdown generado: {markdown_path.name}")
        
        return report_path, markdown_path
    
    def _generate_technical_conclusions(self):
        """Generar conclusiones técnicas basadas en resultados de evaluación."""
        conclusions = {
            'clustering_readiness_comparison': {},
            'architectural_validation': {},
            'statistical_significance': {},
            'dimensionality_impact': {}
        }
        
        # Análisis Hopkins
        if 'hopkins_analysis' in self.evaluation_results:
            hopkins = self.evaluation_results['hopkins_analysis']
            comparative_metrics = hopkins.get('comparative_metrics', {})
            conclusions['clustering_readiness_comparison'] = {
                'semantic_readiness': 'Poor' if comparative_metrics.get('semantic_hopkins', 0) < 0.6 else 'Good',
                'musical_readiness': 'Excellent' if comparative_metrics.get('musical_hopkins', 0) > 0.7 else 'Moderate',
                'significant_difference': comparative_metrics.get('difference', 0) > 0.2
            }
        
        # Validación arquitectural
        conclusions['architectural_validation'] = {
            'hybrid_architecture_justified': self._is_hybrid_architecture_justified(),
            'clustering_auxiliary_recommended': self._is_clustering_auxiliary_recommended(),
            'vectorization_primary_validated': self._is_vectorization_primary_validated()
        }
        
        return conclusions
    
    def _generate_architectural_recommendations(self):
        """Generar recomendaciones arquitecturales basadas en evaluación."""
        recommendations = {
            'primary_system': 'vectorization_direct',
            'auxiliary_system': 'clustering_musical_only',
            'fusion_strategy': 'linear_combination_cosine_similarity',
            'clustering_parameters': {
                'domain': 'musical_only',
                'recommended_k': self._recommend_optimal_k(),
                'algorithm': 'hierarchical_clustering'
            }
        }
        
        return recommendations
    
    def _is_hybrid_architecture_justified(self):
        """Determinar si la arquitectura híbrida está empíricamente justificada."""
        if 'hopkins_analysis' not in self.evaluation_results:
            return False
        
        hopkins = self.evaluation_results['hopkins_analysis']
        comparative_metrics = hopkins.get('comparative_metrics', {})
        difference = comparative_metrics.get('difference', 0)
        
        # Arquitectura híbrida justificada si diferencia > 0.2
        return difference > 0.2
    
    def _is_clustering_auxiliary_recommended(self):
        """Determinar si clustering auxiliar está recomendado para dominio musical."""
        if 'hopkins_analysis' not in self.evaluation_results:
            return False
        
        hopkins = self.evaluation_results['hopkins_analysis']
        comparative_metrics = hopkins.get('comparative_metrics', {})
        musical_hopkins = comparative_metrics.get('musical_hopkins', 0)
        
        # Clustering auxiliar recomendado si Hopkins musical > 0.7
        return musical_hopkins > 0.7
    
    def _is_vectorization_primary_validated(self):
        """Validar vectorización como sistema primario."""
        if 'hopkins_analysis' not in self.evaluation_results:
            return False
        
        hopkins = self.evaluation_results['hopkins_analysis']
        comparative_metrics = hopkins.get('comparative_metrics', {})
        semantic_hopkins = comparative_metrics.get('semantic_hopkins', 1)
        
        # Vectorización validada si Hopkins semántico < 0.6 (inadecuado para clustering)
        return semantic_hopkins < 0.6
    
    def _recommend_optimal_k(self):
        """Recomendar K óptimo para clustering auxiliar musical."""
        # Implementación simple - puede ser refinada con análisis adicional
        return 3  # Basado en evidencia previa del proyecto
    
    def _generate_markdown_report(self, report_data):
        """Generar reporte en formato markdown para documentación académica."""
        markdown = f"""# Clustering Readiness Comparative Assessment - FASE 2

**Fecha de Evaluación**: {report_data['metadata']['evaluation_timestamp']}  
**Dataset Evaluado**: {report_data['metadata']['sample_size']} canciones alineadas  
**Dimensionalidades**: {report_data['metadata']['semantic_dimensions']}D semántico vs {report_data['metadata']['musical_dimensions']}D musical

## Resumen Ejecutivo

Esta evaluación comparativa proporciona evidencia empírica para validar la arquitectura híbrida propuesta mediante análisis estadístico riguroso de clustering readiness entre espacios vectoriales de diferentes dimensionalidades.

## Resultados Hopkins Statistic

"""
        
        if 'hopkins_analysis' in report_data['evaluation_results']:
            hopkins = report_data['evaluation_results']['hopkins_analysis']
            comparative_metrics = hopkins.get('comparative_metrics', {})
            markdown += f"""
- **Hopkins Semántico (384D)**: {comparative_metrics.get('semantic_hopkins', 'N/A'):.4f}
- **Hopkins Musical (12D)**: {comparative_metrics.get('musical_hopkins', 'N/A'):.4f}
- **Diferencia**: {comparative_metrics.get('difference', 'N/A'):.4f}

"""
        
        markdown += f"""## Conclusiones Técnicas

{json.dumps(report_data['technical_conclusions'], indent=2)}

## Recomendaciones Arquitecturales

{json.dumps(report_data['recommendations'], indent=2)}

## Validación de Hipótesis

La evaluación comparativa {"valida" if self._is_hybrid_architecture_justified() else "no valida"} empíricamente la arquitectura híbrida propuesta.
"""
        
        return markdown
    
    def run_complete_evaluation(self):
        """
        Ejecutar evaluación comparativa completa desde carga de datos
        hasta generación de reportes finales.
        """
        try:
            self.logger.info("INICIANDO EVALUACIÓN COMPARATIVA COMPLETA - FASE 2")
            
            # Cargar dataset
            if not self.load_unified_dataset():
                raise RuntimeError("Fallo en carga de dataset unificado")
            
            # Ejecutar evaluación
            self.execute_comparative_evaluation()
            
            # Generar reportes
            report_json, report_md = self.generate_technical_report()
            
            self.logger.info("EVALUACIÓN COMPARATIVA COMPLETADA EXITOSAMENTE")
            self.logger.info(f"Reportes generados: {report_json.name}, {report_md.name}")
            
            return {
                'success': True,
                'report_json': report_json,
                'report_markdown': report_md,
                'evaluation_results': self.evaluation_results
            }
            
        except Exception as e:
            self.logger.error(f"Error en evaluación comparativa: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Función principal para ejecución standalone."""
    # Detectar dataset unificado más reciente
    project_root = Path(__file__).parent.parent
    unification_dir = project_root / "phase1_dataset_unification"
    
    unified_files = list(unification_dir.glob("unified_multimodal_dataset_*.pkl"))
    
    if not unified_files:
        print("ERROR: No se encontró dataset unificado")
        sys.exit(1)
    
    latest_unified = max(unified_files, key=lambda x: x.stat().st_mtime)
    print(f"Usando dataset: {latest_unified.name}")
    
    # Crear evaluador y ejecutar
    output_dir = Path(__file__).parent
    evaluator = ClusteringReadinessComparativeEvaluator(latest_unified, output_dir)
    
    results = evaluator.run_complete_evaluation()
    
    if results['success']:
        print(f"\nEVALUACIÓN EXITOSA")
        print(f"Reporte JSON: {results['report_json'].name}")
        print(f"Reporte Markdown: {results['report_markdown'].name}")
    else:
        print(f"\nERROR EN EVALUACIÓN: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()