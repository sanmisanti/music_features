#!/usr/bin/env python3
"""
CLI principal para ejecutar evaluación de clustering multimodal FASE 3.
Interfaz de línea de comandos con configuración completa.
"""

import argparse
import sys
import json
from pathlib import Path
import logging

# Importar el experimentador principal
from multimodal_clustering_experimenter import MultimodalClusteringExperimenter
from config.algorithms_config import algorithm_config

def setup_logging(verbose: bool = False) -> None:
    """Configurar logging global."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def show_algorithm_configuration() -> None:
    """Mostrar configuración completa de algoritmos."""
    print("=== CONFIGURACIÓN ALGORÍTMICA FASE 3 ===\n")
    
    print("🎵 DOMINIO MUSICAL (12D):")
    musical_configs = algorithm_config.get_algorithm_configs('musical')
    total_musical = 0
    
    for alg_name, config in musical_configs.items():
        print(f"  • {alg_name}: {config['description']}")
        if 'k_range' in config:
            print(f"    K range: {config['k_range']} ({len(config['k_range'])} configuraciones)")
            total_musical += len(config['k_range'])
        elif 'eps_range' in config:
            print(f"    Eps range: {config['eps_range']} ({len(config['eps_range'])} configuraciones)")
            total_musical += len(config['eps_range'])
        print()
    
    print(f"📊 Total configuraciones musicales: {total_musical}\n")
    
    print("🧠 DOMINIO SEMÁNTICO (384D):")
    semantic_configs = algorithm_config.get_algorithm_configs('semantic')
    total_semantic = 0
    
    for alg_name, config in semantic_configs.items():
        print(f"  • {alg_name}: {config['description']}")
        if 'k_range' in config:
            print(f"    K range: {config['k_range']} ({len(config['k_range'])} configuraciones)")
            total_semantic += len(config['k_range'])
        elif 'eps_range' in config:
            print(f"    Eps range: {config['eps_range']} ({len(config['eps_range'])} configuraciones)")
            total_semantic += len(config['eps_range'])
        print()
    
    print(f"📊 Total configuraciones semánticas: {total_semantic}")
    print(f"🎯 TOTAL CONFIGURACIONES: {total_musical + total_semantic}")
    
    print("\n⚙️ FUNCIÓN OBJETIVO MULTI-CRITERIO:")
    print("  • 30% Silhouette Score (calidad técnica)")
    print("  • 30% Balance Distribution (evitar fragmentación)")
    print("  • 20% Interpretability Score (coherencia automática)")
    print("  • 10% Cross-Modal Correspondence (NMI)")
    print("  • 10% Granularity Bonus (K≥5)")

def validate_dataset(dataset_path: str) -> bool:
    """Validar que el dataset existe y tiene la estructura correcta."""
    print(f"Validando dataset: {dataset_path}")
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset no encontrado: {dataset_path}")
        return False
    
    try:
        import pickle
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Verificar estructura esperada
        required_keys = ['data', 'metadata']
        if not all(key in dataset for key in required_keys):
            print(f"❌ Error: Dataset no tiene estructura esperada. Claves requeridas: {required_keys}")
            return False
        
        data = dataset['data']
        required_data_keys = ['semantic_embeddings', 'musical_features_normalized', 'track_ids']
        if not all(key in data for key in required_data_keys):
            print(f"❌ Error: Data no tiene estructura esperada. Claves requeridas: {required_data_keys}")
            return False
        
        # Verificar dimensiones
        n_samples = len(data['track_ids'])
        semantic_shape = data['semantic_embeddings'].shape
        musical_shape = data['musical_features_normalized'].shape
        
        print(f"✅ Dataset válido:")
        print(f"   📊 Muestras: {n_samples:,}")
        print(f"   🧠 Semántico: {semantic_shape}")
        print(f"   🎵 Musical: {musical_shape}")
        
        if semantic_shape[0] != n_samples or musical_shape[0] != n_samples:
            print("❌ Error: Inconsistencia en número de muestras entre arrays")
            return False
        
        if semantic_shape[1] != 384:
            print(f"⚠️  Advertencia: Dimensiones semánticas esperadas 384, encontradas {semantic_shape[1]}")
        
        if musical_shape[1] != 12:
            print(f"⚠️  Advertencia: Dimensiones musicales esperadas 12, encontradas {musical_shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validando dataset: {e}")
        return False

def main():
    """Función principal CLI."""
    parser = argparse.ArgumentParser(
        description="Sistema de Evaluación de Clustering Multimodal FASE 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluación completa con análisis cross-modal
  python run_multimodal_clustering_evaluation.py \\
    --dataset ../phase1_dataset_unification/unified_multimodal_dataset_20250822_004929.pkl \\
    --output ./results

  # Evaluación rápida sin cross-modal
  python run_multimodal_clustering_evaluation.py \\
    --dataset dataset.pkl --output ./results --no-cross-modal

  # Mostrar configuración algorítmica
  python run_multimodal_clustering_evaluation.py --show-config

  # Validar dataset
  python run_multimodal_clustering_evaluation.py --validate-dataset dataset.pkl
        """
    )
    
    # Argumentos principales
    parser.add_argument(
        '--dataset', '-d', 
        type=str,
        help='Ruta al dataset multimodal unificado (.pkl)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./results',
        help='Directorio de salida para resultados (default: ./results)'
    )
    
    # Opciones de configuración
    parser.add_argument(
        '--no-cross-modal',
        action='store_true',
        help='Omitir análisis cross-modal (más rápido)'
    )
    
    parser.add_argument(
        '--top-n-cross-modal',
        type=int,
        default=3,
        help='Número de mejores configuraciones para análisis cross-modal (default: 3)'
    )
    
    # Opciones de información
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Mostrar configuración algorítmica completa y salir'
    )
    
    parser.add_argument(
        '--validate-dataset',
        type=str,
        help='Validar dataset y salir'
    )
    
    # Opciones de ejecución
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar progreso detallado'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Mostrar solo resultados finales'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging(verbose=args.verbose and not args.quiet)
    
    # Manejar opciones de información
    if args.show_config:
        show_algorithm_configuration()
        return 0
    
    if args.validate_dataset:
        if validate_dataset(args.validate_dataset):
            print("✅ Dataset válido para experimentación")
            return 0
        else:
            print("❌ Dataset no válido")
            return 1
    
    # Validar argumentos requeridos
    if not args.dataset:
        print("❌ Error: --dataset es requerido")
        print("Usa --help para ver opciones disponibles")
        return 1
    
    # Validar dataset antes de experimentar
    if not validate_dataset(args.dataset):
        return 1
    
    # Crear experimentador
    print("\n🔬 Inicializando Experimentador Multimodal FASE 3")
    experimenter = MultimodalClusteringExperimenter(
        output_dir=args.output,
        verbose=args.verbose and not args.quiet
    )
    
    try:
        # Ejecutar experimento completo
        print("🚀 Iniciando experimentación...")
        
        comprehensive_report = experimenter.run_complete_experiment(
            dataset_path=args.dataset,
            run_cross_modal=not args.no_cross_modal,
            top_n_cross_modal=args.top_n_cross_modal
        )
        
        # Mostrar resumen de resultados
        print("\n" + "="*60)
        print("🎯 RESULTADOS EXPERIMENTACIÓN MULTIMODAL")
        print("="*60)
        
        metadata = comprehensive_report['metadata']
        print(f"📊 Dataset: {metadata['dataset_size']:,} canciones")
        print(f"⚙️  Configuraciones evaluadas: {metadata['total_configurations_evaluated']}")
        
        # Mejores configuraciones por dominio
        musical_best = comprehensive_report['musical_domain']['best_configuration']
        semantic_best = comprehensive_report['semantic_domain']['best_configuration']
        
        if musical_best:
            print(f"\n🎵 MEJOR CONFIGURACIÓN MUSICAL:")
            print(f"   Algoritmo: {musical_best['algorithm']}")
            print(f"   Score: {musical_best['composite_score']:.3f}")
            print(f"   Silhouette: {musical_best['silhouette_score']:.3f}")
            print(f"   Clusters: {musical_best['n_clusters']}")
        
        if semantic_best:
            print(f"\n🧠 MEJOR CONFIGURACIÓN SEMÁNTICA:")
            print(f"   Algoritmo: {semantic_best['algorithm']}")
            print(f"   Score: {semantic_best['composite_score']:.3f}")
            print(f"   Silhouette: {semantic_best['silhouette_score']:.3f}")
            print(f"   Clusters: {semantic_best['n_clusters']}")
        
        # Análisis cross-modal
        if not args.no_cross_modal:
            cross_modal = comprehensive_report['cross_modal_analysis']
            best_combo = cross_modal['best_combination']
            
            if best_combo:
                print(f"\n🔗 MEJOR COMBINACIÓN CROSS-MODAL:")
                print(f"   Combinación: {best_combo['combination_id']}")
                print(f"   NMI Score: {best_combo['nmi_score']:.3f}")
                print(f"   Correspondencias fuertes: {best_combo['strong_correspondences']}")
                print(f"   Cobertura: {best_combo['correspondence_coverage']:.1%}")
        
        # Conclusiones científicas
        conclusions = comprehensive_report.get('scientific_conclusions', {})
        if conclusions:
            print(f"\n📈 CONCLUSIONES CIENTÍFICAS:")
            
            domain_comp = conclusions.get('domain_comparison')
            if domain_comp:
                superior = domain_comp['superior_domain']
                print(f"   Dominio superior: {superior.upper()}")
                print(f"   Diferencia de score: {domain_comp['score_difference']:.3f}")
            
            cross_coherence = conclusions.get('cross_modal_coherence')
            if cross_coherence:
                print(f"   Coherencia cross-modal: {cross_coherence['coherence_assessment']}")
                print(f"   NMI máximo: {cross_coherence['maximum_nmi']:.3f}")
        
        print(f"\n📁 Resultados guardados en: {args.output}")
        print("✅ Experimentación completada exitosamente")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Experimentación interrumpida por el usuario")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error durante experimentación: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())