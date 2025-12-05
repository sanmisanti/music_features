#!/usr/bin/env python3
"""
FASE 2.2: ANÁLISIS CLUSTERING COMPLETO
======================================

Script para ejecutar análisis clustering comparativo completo según
FASE2_CLUSTERING_COMPARATIVO_PLAN.md

Configuración:
- Datasets completos (no test mode)
- Múltiples algoritmos (K-Means + Hierarchical)
- K range amplio [3-10]
- 10 runs por configuración
- Análisis estadístico robusto

Objetivo: Alcanzar Silhouette Score > 0.25 o confirmar necesidad FASE 4
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Añadir path para imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import del comparador
from clustering_comparative import ClusteringComparator

def run_complete_analysis():
    """Ejecutar análisis clustering completo FASE 2.2."""
    
    print("🚀 FASE 2.2: ANÁLISIS CLUSTERING COMPLETO")
    print("="*60)
    print("Objetivo: Silhouette Score > 0.25 o confirmar necesidad FASE 4")
    print("Configuración: Datasets completos, múltiples algoritmos, análisis robusto")
    print()
    
    # Inicializar comparador
    comparator = ClusteringComparator()
    
    # ETAPA 2.2.1: Análisis K-Means completo
    print("📊 ETAPA 2.2.1: ANÁLISIS K-MEANS COMPLETO")
    print("-"*50)
    
    start_time = time.time()
    
    kmeans_results = comparator.compare_datasets(
        dataset_keys=['optimal', 'control', 'baseline'],
        algorithm='kmeans',
        test_mode=False  # ¡DATASETS COMPLETOS!
    )
    
    kmeans_time = time.time() - start_time
    
    # Guardar resultados K-Means
    kmeans_json_path, kmeans_report_path = comparator.save_results(
        kmeans_results, "_kmeans_complete"
    )
    
    print(f"✅ K-Means analysis completado en {kmeans_time:.1f}s")
    
    # Analizar resultados K-Means
    kmeans_best = analyze_results(kmeans_results, "K-Means")
    
    # ETAPA 2.2.2: Análisis Hierarchical completo
    print(f"\n📊 ETAPA 2.2.2: ANÁLISIS HIERARCHICAL COMPLETO")
    print("-"*50)
    
    start_time = time.time()
    
    hierarchical_results = comparator.compare_datasets(
        dataset_keys=['optimal', 'control', 'baseline'],
        algorithm='hierarchical',
        test_mode=False  # ¡DATASETS COMPLETOS!
    )
    
    hierarchical_time = time.time() - start_time
    
    # Guardar resultados Hierarchical
    hierarchical_json_path, hierarchical_report_path = comparator.save_results(
        hierarchical_results, "_hierarchical_complete"
    )
    
    print(f"✅ Hierarchical analysis completado en {hierarchical_time:.1f}s")
    
    # Analizar resultados Hierarchical
    hierarchical_best = analyze_results(hierarchical_results, "Hierarchical")
    
    # ETAPA 2.2.3: Comparación algoritmos
    print(f"\n📊 ETAPA 2.2.3: COMPARACIÓN ALGORITMOS")
    print("-"*50)
    
    algorithm_comparison = compare_algorithms(kmeans_best, hierarchical_best)
    
    # ETAPA 2.2.4: Recomendaciones finales
    print(f"\n🎯 ETAPA 2.2.4: RECOMENDACIONES FINALES")
    print("-"*50)
    
    final_recommendations = generate_final_recommendations(
        kmeans_results, hierarchical_results, algorithm_comparison
    )
    
    # Resumen ejecutivo
    print(f"\n🎉 FASE 2.2 COMPLETADA")
    print("="*40)
    print(f"⏱️  Tiempo total: {(kmeans_time + hierarchical_time)/60:.1f} minutos")
    print(f"📁 Reportes generados:")
    print(f"   - K-Means: {kmeans_report_path.name}")
    print(f"   - Hierarchical: {hierarchical_report_path.name}")
    
    return {
        'kmeans_results': kmeans_results,
        'hierarchical_results': hierarchical_results,
        'algorithm_comparison': algorithm_comparison,
        'final_recommendations': final_recommendations,
        'execution_summary': {
            'kmeans_time': kmeans_time,
            'hierarchical_time': hierarchical_time,
            'total_time': kmeans_time + hierarchical_time,
            'timestamp': datetime.now().isoformat()
        }
    }

def analyze_results(results, algorithm_name):
    """Analizar resultados de un algoritmo específico."""
    
    print(f"\n🔍 ANÁLISIS {algorithm_name.upper()}:")
    
    best_overall = None
    best_silhouette = -1
    target_met = False
    
    if 'dataset_analyses' in results:
        dataset_results = {}
        
        for dataset_key, analysis in results['dataset_analyses'].items():
            if 'error' not in analysis and analysis.get('best_k'):
                best_k_info = analysis['best_k']
                dataset_name = analysis['dataset_metadata']['dataset_name']
                
                silhouette = best_k_info['silhouette_score']
                k = best_k_info['k']
                
                dataset_results[dataset_key] = {
                    'name': dataset_name,
                    'silhouette': silhouette,
                    'k': k
                }
                
                print(f"  📊 {dataset_name}: K={k}, Silhouette={silhouette:.4f}")
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_overall = dataset_key
                
                if silhouette >= 0.25:
                    target_met = True
        
        # Mostrar mejor resultado
        if best_overall:
            best_info = dataset_results[best_overall]
            status = "✅ TARGET MET" if target_met else "⚠️  TARGET NOT MET"
            print(f"  🎯 MEJOR: {best_info['name']} - Silhouette {best_info['silhouette']:.4f} - {status}")
        
        return {
            'algorithm': algorithm_name,
            'best_dataset': best_overall,
            'best_silhouette': best_silhouette,
            'target_met': target_met,
            'dataset_results': dataset_results
        }
    
    return None

def compare_algorithms(kmeans_best, hierarchical_best):
    """Comparar resultados entre algoritmos."""
    
    print(f"\n📈 COMPARACIÓN ALGORITMOS:")
    
    comparison = {
        'kmeans': kmeans_best,
        'hierarchical': hierarchical_best
    }
    
    # Determinar mejor algoritmo
    best_algorithm = None
    best_silhouette_overall = -1
    
    for algo_name, results in comparison.items():
        if results and results['best_silhouette'] > best_silhouette_overall:
            best_silhouette_overall = results['best_silhouette']
            best_algorithm = algo_name
    
    print(f"  🥇 MEJOR ALGORITMO: {best_algorithm.upper()}")
    print(f"  📊 Mejor Silhouette: {best_silhouette_overall:.4f}")
    
    # Target analysis
    target_met_any = any(results['target_met'] for results in comparison.values() if results)
    target_status = "✅ ALCANZADO" if target_met_any else "❌ NO ALCANZADO"
    print(f"  🎯 Objetivo >0.25: {target_status}")
    
    return {
        'best_algorithm': best_algorithm,
        'best_silhouette_overall': best_silhouette_overall,
        'target_met_any': target_met_any,
        'algorithm_comparison': comparison
    }

def generate_final_recommendations(kmeans_results, hierarchical_results, algorithm_comparison):
    """Generar recomendaciones finales basadas en todos los resultados."""
    
    print(f"\n💡 RECOMENDACIONES FINALES:")
    
    recommendations = {
        'next_phase': None,
        'best_configuration': None,
        'actions': []
    }
    
    target_met = algorithm_comparison['target_met_any']
    best_silhouette = algorithm_comparison['best_silhouette_overall']
    
    if target_met:
        # Objetivo alcanzado - proceder a FASE 3
        recommendations['next_phase'] = 'FASE_3_CLUSTERING_READINESS'
        recommendations['actions'] = [
            "✅ Objetivo Silhouette >0.25 ALCANZADO",
            "🎯 Proceder con FASE 3: Clustering Readiness Assessment",
            f"🔧 Usar configuración óptima: {algorithm_comparison['best_algorithm']} con mejor dataset",
            "📊 Realizar análisis Hopkins detallado en FASE 3"
        ]
        print("  ✅ ÉXITO: Proceder a FASE 3 - Clustering Readiness Assessment")
        
    elif best_silhouette > 0.20:
        # Cerca del objetivo - considerar FASE 4
        recommendations['next_phase'] = 'FASE_4_CLUSTER_PURIFICATION'
        recommendations['actions'] = [
            f"⚠️  Objetivo NO alcanzado: {best_silhouette:.4f} vs 0.25 target",
            "🔄 RECOMENDACIÓN: Proceder a FASE 4 - Cluster Purification",
            "🎯 Cluster purification puede lograr mejora adicional +0.05-0.10",
            "📊 Potential final: 0.25-0.30 tras purification"
        ]
        print("  🔄 FASE 4 RECOMENDADA: Cluster Purification para alcanzar objetivo")
        
    else:
        # Lejos del objetivo - revisar estrategia
        recommendations['next_phase'] = 'STRATEGY_REVIEW'
        recommendations['actions'] = [
            f"❌ Objetivo NO alcanzado: {best_silhouette:.4f} vs 0.25 target (gap -{0.25-best_silhouette:.3f})",
            "🔄 REVISAR: Estrategia de optimización requiere ajustes",
            "🧪 CONSIDERAR: Algoritmos adicionales (DBSCAN, Gaussian Mixture)",
            "📊 EVALUAR: Feature engineering o dimensionality reduction adicional"
        ]
        print("  ⚠️  STRATEGY REVIEW: Revisar enfoque de optimización")
    
    # Configuración recomendada
    if algorithm_comparison['best_algorithm']:
        best_algo_results = algorithm_comparison['algorithm_comparison'][algorithm_comparison['best_algorithm']]
        
        recommendations['best_configuration'] = {
            'algorithm': algorithm_comparison['best_algorithm'],
            'dataset': best_algo_results['best_dataset'],
            'silhouette': best_algo_results['best_silhouette'],
            'k': best_algo_results['dataset_results'][best_algo_results['best_dataset']]['k']
        }
    
    for action in recommendations['actions']:
        print(f"  {action}")
    
    return recommendations

def main():
    """Función principal."""
    
    print("Iniciando FASE 2.2: Análisis Clustering Completo...")
    print("Estimado: 15-20 minutos para análisis completo")
    print()
    
    try:
        results = run_complete_analysis()
        
        print(f"\n🎊 FASE 2.2 EJECUTADA EXITOSAMENTE")
        
        # Determinar próximo paso
        next_phase = results['final_recommendations']['next_phase']
        
        if next_phase == 'FASE_3_CLUSTERING_READINESS':
            print("🎯 PRÓXIMO: Implementar FASE 3 - Clustering Readiness Assessment")
        elif next_phase == 'FASE_4_CLUSTER_PURIFICATION':
            print("🎯 PRÓXIMO: Implementar FASE 4 - Cluster Purification")
        else:
            print("🎯 PRÓXIMO: Revisar estrategia de optimización")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR en FASE 2.2: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)