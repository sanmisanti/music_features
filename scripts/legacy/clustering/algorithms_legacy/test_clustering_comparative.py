#!/usr/bin/env python3
"""
FASE 2.1.2: TEST INICIAL CLUSTERING COMPARATIVO
===============================================

Script de validación para verificar configuración de datasets y 
funcionamiento básico del sistema comparativo.

Objetivo: Validar que todos los datasets cargan correctamente antes
de ejecutar análisis completo.
"""

import os
import sys
from pathlib import Path

# Añadir path para imports - ir al directorio raíz del proyecto
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import directo del comparador en el mismo directorio
from clustering_comparative import ClusteringComparator

def test_dataset_loading():
    """Test carga de todos los datasets configurados."""
    
    print("🧪 FASE 2.1.2: TEST DATASET LOADING")
    print("="*50)
    
    # Inicializar comparador
    comparator = ClusteringComparator()
    
    # Test cada dataset
    datasets_to_test = ['optimal', 'control', 'baseline']
    loading_results = {}
    
    for dataset_key in datasets_to_test:
        print(f"\n📊 Testing dataset: {dataset_key}")
        print("-"*30)
        
        try:
            # Cargar con sample pequeño para test rápido
            features_df, metadata = comparator.load_dataset(
                dataset_key, 
                sample_size=100, 
                test_mode=True
            )
            
            loading_results[dataset_key] = {
                'status': 'success',
                'shape': features_df.shape,
                'features': metadata['n_features'],
                'expected_hopkins': metadata['expected_hopkins'],
                'separator': comparator.datasets_config[dataset_key]['separator']
            }
            
            print(f"✅ {dataset_key}: {features_df.shape[0]} samples × {features_df.shape[1]} features")
            print(f"   Expected Hopkins: {metadata['expected_hopkins']}")
            
        except Exception as e:
            loading_results[dataset_key] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"❌ {dataset_key}: ERROR - {e}")
    
    # Resumen
    print(f"\n📊 LOADING TEST SUMMARY")
    print("="*30)
    
    success_count = sum(1 for result in loading_results.values() if result['status'] == 'success')
    total_count = len(loading_results)
    
    print(f"✅ Successful loads: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 ALL DATASETS LOADED SUCCESSFULLY - Ready for clustering comparison")
        return True
    else:
        print("⚠️  Some datasets failed to load - Check configuration")
        return False

def test_clustering_analysis_quick():
    """Test rápido de análisis clustering en dataset optimal."""
    
    print(f"\n🧪 FASE 2.1.3: TEST CLUSTERING ANALYSIS")
    print("="*50)
    
    try:
        # Inicializar comparador
        comparator = ClusteringComparator()
        
        # Cargar dataset optimal (más pequeño y confiable)
        print("📂 Cargando dataset optimal para test...")
        features_df, metadata = comparator.load_dataset('optimal', sample_size=200, test_mode=True)
        
        # Ejecutar análisis clustering reducido
        print("🎯 Ejecutando análisis clustering test...")
        
        # Configuración reducida para test rápido
        original_config = comparator.analysis_config.copy()
        comparator.analysis_config.update({
            'k_range': [3, 5, 7],  # Solo 3 valores K
            'n_runs': 3,           # Solo 3 runs por K
            'random_states': [42, 43, 44]
        })
        
        analysis_result = comparator.run_clustering_analysis(
            features_df, metadata, algorithm='kmeans'
        )
        
        # Restaurar configuración original
        comparator.analysis_config = original_config
        
        # Verificar resultados
        if analysis_result and 'best_k' in analysis_result and analysis_result['best_k']:
            best_k = analysis_result['best_k']
            print(f"✅ Clustering test successful:")
            print(f"   Best K: {best_k['k']}")
            print(f"   Best Silhouette: {best_k['silhouette_score']:.4f}")
            
            # Verificar si cumple objetivo
            target_met = best_k['silhouette_score'] > 0.25
            target_status = "✅ MET" if target_met else "⚠️  NOT MET"
            print(f"   Target >0.25: {target_status}")
            
            return True
        else:
            print("❌ Clustering test failed - No best K found")
            return False
            
    except Exception as e:
        print(f"❌ Clustering test failed: {e}")
        return False

def test_comparison_quick():
    """Test rápido de comparación entre datasets."""
    
    print(f"\n🧪 FASE 2.1.4: TEST COMPARISON SYSTEM")
    print("="*50)
    
    try:
        # Inicializar comparador
        comparator = ClusteringComparator()
        
        # Ejecutar comparación reducida (solo optimal vs control)
        print("📊 Ejecutando comparación test...")
        
        comparison_results = comparator.compare_datasets(
            dataset_keys=['optimal', 'control'],  # Solo 2 datasets
            algorithm='kmeans',
            test_mode=True
        )
        
        # Verificar resultados
        if comparison_results and 'recommendations' in comparison_results:
            recommendations = comparison_results['recommendations']
            
            print(f"✅ Comparison test successful:")
            
            if recommendations.get('best_dataset'):
                best_info = recommendations['best_dataset']
                print(f"   Best dataset: {best_info['dataset']}")
                print(f"   Best silhouette: {best_info['silhouette_score']:.4f}")
                
                target_met = recommendations.get('silhouette_target_met', False)
                target_status = "✅ YES" if target_met else "❌ NO"
                print(f"   Target >0.25 met: {target_status}")
                
                significance = recommendations.get('statistical_significance', False)
                sig_status = "✅ YES" if significance else "❌ NO" 
                print(f"   Statistical significance: {sig_status}")
            
            # Guardar resultados test
            json_path, report_path = comparator.save_results(comparison_results, "_validation_test")
            print(f"📁 Test results saved: {json_path.name}")
            
            return True
        else:
            print("❌ Comparison test failed - No recommendations generated")
            return False
            
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def main():
    """Ejecutar todos los tests de validación."""
    
    print("🚀 CLUSTERING COMPARATIVE VALIDATION TESTS")
    print("="*60)
    print("Objective: Validate system readiness for FASE 2 execution")
    print()
    
    # Ejecutar tests secuencialmente
    tests_results = []
    
    # Test 1: Dataset loading
    test1_success = test_dataset_loading()
    tests_results.append(('Dataset Loading', test1_success))
    
    if test1_success:
        # Test 2: Clustering analysis
        test2_success = test_clustering_analysis_quick()
        tests_results.append(('Clustering Analysis', test2_success))
        
        if test2_success:
            # Test 3: Comparison system
            test3_success = test_comparison_quick()
            tests_results.append(('Comparison System', test3_success))
    
    # Resumen final
    print(f"\n🎯 VALIDATION TESTS SUMMARY")
    print("="*40)
    
    for test_name, success in tests_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    total_tests = len(tests_results)
    passed_tests = sum(1 for _, success in tests_results if success)
    
    print(f"\nPassed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR FASE 2 EXECUTION")
        print("\nNext steps:")
        print("1. Run full comparison: python clustering_comparative.py")
        print("2. Analyze results and proceed to FASE 2.2")
    else:
        print("⚠️  SOME TESTS FAILED - FIX ISSUES BEFORE PROCEEDING")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)