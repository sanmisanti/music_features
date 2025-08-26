"""
FASE 3: Sistema de Clustering Multimodal con Prioridad en Interpretabilidad
===========================================================================

Módulo completo para experimentación algorítmica exhaustiva de clustering
multimodal en espacios vectoriales musical (12D) y semántico (384D),
con enfoque en granularidad explicativa y correspondencias cross-modales.

Componentes principales:
- MultimodalClusteringExperimenter: Orquestador principal
- AlgorithmEvaluator: Evaluador especializado por dominio
- InterpretabilityValidator: Validador de interpretabilidad
- CrossModalAnalyzer: Análisis de correspondencias
- Configuraciones especializadas por dimensionalidad

Autor: Proyecto FASE 3 - Sistema Clustering Multimodal
Fecha: Agosto 2025
"""

__version__ = "1.0.0"
__author__ = "Proyecto FASE 3 - Sistema Clustering Multimodal"

# Importaciones principales para API pública
from .multimodal_clustering_experimenter import (
    MultimodalClusteringExperimenter,
    run_complete_multimodal_experimentation
)

from .algorithm_evaluator import AlgorithmEvaluator, evaluate_multimodal_clustering
from .interpretability_validator import InterpretabilityValidator
from .cross_modal_analyzer import CrossModalAnalyzer

# Configuraciones
from .config.algorithms_config import algorithms_config
from .config.evaluation_metrics import evaluation_metrics
from .config.interpretability_settings import interpretability_settings

__all__ = [
    # Clases principales
    'MultimodalClusteringExperimenter',
    'AlgorithmEvaluator', 
    'InterpretabilityValidator',
    'CrossModalAnalyzer',
    
    # Funciones de conveniencia
    'run_complete_multimodal_experimentation',
    'evaluate_multimodal_clustering',
    
    # Configuraciones
    'algorithms_config',
    'evaluation_metrics',
    'interpretability_settings'
]