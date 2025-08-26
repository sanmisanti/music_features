#!/usr/bin/env python3
"""
Script Principal para Ejecución de FASE 3: Sistema Clustering Multimodal
=========================================================================

Script ejecutable para evaluación completa de clustering multimodal
con prioridad en interpretabilidad. Orquesta experimentación algorítmica
exhaustiva, validación de interpretabilidad, y análisis cross-modal.

Autor: Proyecto FASE 3 - Sistema Clustering Multimodal
Fecha: Agosto 2025
"""

import argparse
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Importar componentes del sistema
from multimodal_clustering_experimenter import MultimodalClusteringExperimenter, run_complete_multimodal_experimentation
from config.algorithms_config import algorithms_config


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configurar parser de argumentos de línea de comandos.
    
    Returns:
        Parser configurado
    """
    parser = argparse.ArgumentParser(
        description="FASE 3: Sistema de Clustering Multimodal con Prioridad en Interpretabilidad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Ejecución completa con configuración estándar
  python run_multimodal_clustering_evaluation.py \\
    --dataset ../phase1_dataset_unification/unified_multimodal_dataset_20250822_004929.pkl \\
    --output ./results

  # Ejecución sin análisis cross-modal (más rápido)
  python run_multimodal_clustering_evaluation.py \\
    --dataset ../phase1_dataset_unification/unified_multimodal_dataset_20250822_004929.pkl \\
    --output ./results \\
    --no-cross-modal

  # Ejecución silenciosa con solo reportes finales
  python run_multimodal_clustering_evaluation.py \\
    --dataset ../phase1_dataset_unification/unified_multimodal_dataset_20250822_004929.pkl \\
    --output ./results \\
    --quiet

  # Mostrar información de configuración experimental
  python run_multimodal_clustering_evaluation.py --show-config
        """
    )
    
    # Argumentos principales
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Ruta al dataset unificado multimodal (.pkl)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Directorio para guardar resultados de experimentación'
    )
    
    # Opciones de experimentación
    parser.add_argument(
        '--no-cross-modal',
        action='store_true',
        help='Omitir análisis cross-modal (reduce tiempo de ejecución)'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Omitir generación de visualizaciones'
    )
    
    # Opciones de logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Activar logging detallado'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimizar output (solo errores y resultados finales)'
    )
    
    # Opciones de información
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Mostrar configuración experimental y salir'
    )
    
    parser.add_argument(
        '--validate-dataset',
        type=str,
        help='Validar dataset especificado y salir'
    )
    
    return parser


def show_experimental_configuration() -> None:
    """
    Mostrar información detallada de configuración experimental.
    """
    print("="*80)
    print("FASE 3: CONFIGURACIÓN EXPERIMENTAL CLUSTERING MULTIMODAL")
    print("="*80)
    
    # Información de algoritmos
    print("\nALGORITMOS DE CLUSTERING:")
    
    print("\n  Musical (12D):")
    musical_algorithms = algorithms_config.get_algorithm_configs('musical')
    for alg_name, config in musical_algorithms.items():
        print(f"    - {alg_name}: {config['description']}")
    
    print("\n  Semántico (384D):")
    semantic_algorithms = algorithms_config.get_algorithm_configs('semantic')
    for alg_name, config in semantic_algorithms.items():
        print(f"    - {alg_name}: {config['description']}")
    
    # Información de rangos K
    print("\nRANGOS DE CLUSTERING:")
    musical_k_range = algorithms_config.get_k_range('musical')
    semantic_k_range = algorithms_config.get_k_range('semantic')
    
    print(f"    - Musical K: {musical_k_range}")
    print(f"    - Semántico K: {semantic_k_range}")
    
    # Cálculo de experimentos
    musical_exp, semantic_exp, total_exp = algorithms_config.get_experiment_matrix_size()
    
    print("\nMATRIZ EXPERIMENTAL:")
    print(f"    - Experimentos musicales: {musical_exp}")
    print(f"    - Experimentos semánticos: {semantic_exp}")
    print(f"    - Total experimentos: {total_exp}")
    
    # Criterios de éxito
    print("\nCRITERIOS DE ÉXITO:")
    print("    - Granularidad mínima: K >= 5 (prioridad interpretabilidad)")
    print("    - Silhouette Score mínimo: 0.15")
    print("    - Balance distribución mínimo: 0.6")
    print("    - Interpretabilidad objetivo: 100% clusters etiquetables")
    print("    - Correspondencia cross-modal: >=60% interpretables")
    
    # Función objetivo
    print("\nFUNCIÓN OBJETIVO MULTI-CRITERIO:")
    print("    score = 0.3×silhouette + 0.3×balance + 0.2×interpretabilidad")
    print("           + 0.1×correspondencia + 0.1×granularidad")
    
    print("\n" + "="*80)


def validate_dataset(dataset_path: str) -> bool:
    """
    Validar integridad y estructura del dataset.
    
    Args:
        dataset_path: Ruta al dataset
        
    Returns:
        True si el dataset es válido
    """
    import pickle
    import numpy as np
    
    print(f"VALIDANDO DATASET: {dataset_path}")
    print("-" * 60)
    
    try:
        # Cargar dataset
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Verificar claves requeridas
        required_keys = ['musical_features_normalized', 'semantic_embeddings', 'track_ids']
        missing_keys = [key for key in required_keys if key not in dataset]
        
        if missing_keys:
            print(f"ERROR: Claves faltantes: {missing_keys}")
            return False
        
        # Verificar dimensiones
        musical_features = dataset['musical_features_normalized']
        semantic_embeddings = dataset['semantic_embeddings']
        track_ids = dataset['track_ids']
        
        print(f"Características musicales: {musical_features.shape}")
        print(f"Embeddings semánticos: {semantic_embeddings.shape}")
        print(f"Track IDs: {len(track_ids)}")
        
        # Verificar consistencia dimensional
        n_samples = len(track_ids)
        
        if musical_features.shape[0] != n_samples:
            print(f"ERROR: Inconsistencia dimensional musical")
            return False
        
        if semantic_embeddings.shape[0] != n_samples:
            print(f"ERROR: Inconsistencia dimensional semántica")
            return False
        
        # Verificar dimensionalidad esperada
        if musical_features.shape[1] != 12:
            print(f"ADVERTENCIA: Dimensionalidad musical inesperada: {musical_features.shape[1]} (esperado: 12)")
        
        if semantic_embeddings.shape[1] != 384:
            print(f"ADVERTENCIA: Dimensionalidad semántica inesperada: {semantic_embeddings.shape[1]} (esperado: 384)")
        
        # Verificar normalización
        musical_mean = np.mean(musical_features)
        musical_std = np.std(musical_features)
        
        print(f"Estadísticas musicales - Mean: {musical_mean:.4f}, Std: {musical_std:.4f}")
        
        semantic_norms = np.linalg.norm(semantic_embeddings, axis=1)
        semantic_norm_mean = np.mean(semantic_norms)
        semantic_norm_std = np.std(semantic_norms)
        
        print(f"Normas semánticas - Mean: {semantic_norm_mean:.4f}, Std: {semantic_norm_std:.4f}")
        
        # Verificar datos faltantes
        musical_nan_count = np.sum(np.isnan(musical_features))
        semantic_nan_count = np.sum(np.isnan(semantic_embeddings))
        
        if musical_nan_count > 0:
            print(f"ADVERTENCIA: {musical_nan_count} valores NaN en características musicales")
        
        if semantic_nan_count > 0:
            print(f"ADVERTENCIA: {semantic_nan_count} valores NaN en embeddings semánticos")
        
        print("\nDATASET VÁLIDO PARA EXPERIMENTACIÓN")
        print(f"Listo para {algorithms_config.get_experiment_matrix_size()[2]} experimentos")
        
        return True
        
    except FileNotFoundError:
        print(f"ERROR: Dataset no encontrado: {dataset_path}")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def main() -> int:
    """
    Función principal del script.
    
    Returns:
        Código de salida (0 = éxito, 1 = error)
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Configurar logging global
    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
    elif args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
    # Procesar opciones de información
    if args.show_config:
        show_experimental_configuration()
        return 0
    
    if args.validate_dataset:
        is_valid = validate_dataset(args.validate_dataset)
        return 0 if is_valid else 1
    
    # Validar argumentos requeridos
    if not args.dataset or not args.output:
        print("ERROR: Dataset y directorio de salida son requeridos")
        print("Usar --help para ver opciones disponibles")
        return 1
    
    # Validar rutas
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset no encontrado: {args.dataset}")
        return 1
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar parámetros de experimentación
    include_cross_modal = not args.no_cross_modal
    generate_visualizations = not args.no_visualizations
    verbose = args.verbose and not args.quiet
    
    # Mostrar configuración de ejecución
    if not args.quiet:
        print("="*80)
        print("INICIANDO FASE 3: CLUSTERING MULTIMODAL EXHAUSTIVO")
        print("="*80)
        print(f"Dataset: {args.dataset}")
        print(f"Output: {args.output}")
        print(f"Cross-modal: {'Habilitado' if include_cross_modal else 'Deshabilitado'}")
        print(f"Visualizaciones: {'Habilitado' if generate_visualizations else 'Deshabilitado'}")
        print(f"Logging: {'Detallado' if verbose else 'Silencioso' if args.quiet else 'Estándar'}")
        print("="*80)
    
    try:
        # Ejecutar experimentación completa
        start_time = datetime.now()
        
        results = run_complete_multimodal_experimentation(
            dataset_path=str(dataset_path),
            output_directory=str(output_dir),
            include_cross_modal=include_cross_modal,
            generate_visualizations=generate_visualizations,
            verbose=verbose
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Mostrar resumen de resultados
        if not args.quiet:
            print("\n" + "="*80)
            print("EXPERIMENTACIÓN COMPLETADA EXITOSAMENTE")
            print("="*80)
            
            musical_ranking = results['ranking_results']['musical_ranking']
            semantic_ranking = results['ranking_results']['semantic_ranking']
            
            print(f"Tiempo total: {execution_time:.1f} segundos")
            print(f"Total experimentos: {results['metadata'].get('musical_experiments', 0) + results['metadata'].get('semantic_experiments', 0)}")
            print(f"Algoritmos evaluados: {len(set([exp.get('algorithm_name') for exp in results.get('musical_experiments', []) + results.get('semantic_experiments', [])]))}")
            
            # Mejores configuraciones
            print("\nMEJORES CONFIGURACIONES:")
            
            if musical_ranking.get('best_configuration'):
                best_musical = musical_ranking['best_configuration']
                print(f"   Musical: {best_musical['algorithm_name']} K={best_musical['k_effective']} "
                     f"(Silhouette: {best_musical['silhouette_score']:.3f})")
            
            if semantic_ranking.get('best_configuration'):
                best_semantic = semantic_ranking['best_configuration']
                print(f"   Semántico: {best_semantic['algorithm_name']} K={best_semantic['k_effective']} "
                     f"(Silhouette: {best_semantic['silhouette_score']:.3f})")
            
            # Interpretabilidad
            interpretability_summary = results['interpretability_analysis']['summary']
            print(f"\nINTERPRETABILIDAD: {interpretability_summary.get('overall_assessment', 'No disponible')}")
            
            # Archivos generados
            print(f"\nRESULTADOS GUARDADOS EN: {output_dir}")
            print("   - Reporte JSON completo")
            print("   - Reporte Markdown académico")
            print("   - Logs de experimentación")
            if include_cross_modal:
                print("   - Análisis cross-modal")
            if generate_visualizations:
                print("   - Configuración visualizaciones")
            
            print("\n" + "="*80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nEXPERIMENTACIÓN INTERRUMPIDA POR USUARIO")
        return 1
    except Exception as e:
        print(f"\nERROR DURANTE EXPERIMENTACIÓN: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)