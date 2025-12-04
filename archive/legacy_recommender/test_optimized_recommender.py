#!/usr/bin/env python3
"""
🧪 TEST SUITE - RECOMENDADOR MUSICAL OPTIMIZADO
===============================================

Suite de tests para validar la integración completa del sistema optimizado
con ClusterPurifier y verificar performance objetivo (<100ms).

TESTS INCLUIDOS:
1. Inicialización del sistema
2. Integración con ClusterPurifier
3. Performance de las 6 estrategias
4. Calidad de recomendaciones
5. Validación de datos optimizados

Autor: Test Suite Optimized Recommender
Fecha: Enero 2025
Estado: ✨ VALIDATION READY
"""

import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

def test_system_initialization():
    """Test 1: Validar inicialización del sistema optimizado."""
    
    print("🧪 TEST 1: Inicialización del Sistema")
    print("-" * 50)
    
    try:
        from optimized_music_recommender import OptimizedMusicRecommender
        
        # Medir tiempo de inicialización
        start_time = time.time()
        recommender = OptimizedMusicRecommender()
        init_time = time.time() - start_time
        
        print(f"✅ Instancia creada en {init_time:.2f}s")
        
        # Verificar configuración
        assert hasattr(recommender, 'discriminative_features'), "Features discriminativas no configuradas"
        assert len(recommender.discriminative_features) == 9, f"Expected 9 features, got {len(recommender.discriminative_features)}"
        assert hasattr(recommender, 'recommendation_strategies'), "Estrategias no configuradas"
        assert len(recommender.recommendation_strategies) == 6, f"Expected 6 strategies, got {len(recommender.recommendation_strategies)}"
        
        print(f"✅ Configuración validada:")
        print(f"   📊 Features: {len(recommender.discriminative_features)}/9")
        print(f"   🎯 Estrategias: {len(recommender.recommendation_strategies)}/6")
        
        return recommender, True
        
    except Exception as e:
        print(f"❌ Error en inicialización: {e}")
        return None, False


def test_system_setup(recommender):
    """Test 2: Validar setup completo del sistema con ClusterPurifier."""
    
    print("\n🧪 TEST 2: Setup Completo del Sistema")
    print("-" * 50)
    
    try:
        # Medir tiempo de setup completo
        start_time = time.time()
        success = recommender.initialize_system()
        setup_time = time.time() - start_time
        
        if not success:
            print("❌ Fallo en initialize_system()")
            return False
        
        print(f"✅ Sistema inicializado en {setup_time:.2f}s")
        
        # Validar componentes
        assert hasattr(recommender, 'dataset'), "Dataset no cargado"
        assert len(recommender.dataset) > 0, "Dataset vacío"
        assert hasattr(recommender, 'cluster_assignments'), "Clusters no asignados"
        assert hasattr(recommender, 'cluster_centroids'), "Centroides no calculados"
        
        print(f"✅ Componentes validados:")
        print(f"   📊 Dataset: {len(recommender.dataset):,} canciones")
        print(f"   🎯 Clusters: {len(recommender.cluster_centroids)} centroides")
        print(f"   💾 Asignaciones: {len(recommender.cluster_assignments):,}")
        
        # Validar integración ClusterPurifier
        if hasattr(recommender, 'similarity_matrix'):
            matrix_size = recommender.similarity_matrix.shape
            print(f"   🔧 Matriz similitud: {matrix_size}")
        elif hasattr(recommender, 'cluster_similarities'):
            cluster_count = len(recommender.cluster_similarities)
            print(f"   🔧 Índices por cluster: {cluster_count} clusters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en setup: {e}")
        return False


def test_performance_benchmarks(recommender):
    """Test 3: Validar performance objetivo (<100ms por recomendación)."""
    
    print("\n🧪 TEST 3: Performance Benchmarks")
    print("-" * 50)
    
    # Test queries
    test_queries = [0, 1, 2, 3, 4]  # Primeras 5 canciones como test
    strategies = list(recommender.recommendation_strategies.keys())
    
    performance_results = {}
    target_time_ms = 100
    
    for strategy in strategies:
        print(f"\n   🔄 Testing estrategia: {strategy}")
        
        times = []
        successes = 0
        
        for query in test_queries:
            try:
                start_time = time.time()
                result = recommender.recommend(
                    query=query,
                    strategy=strategy,
                    n_recommendations=5,
                    explain=False
                )
                exec_time = (time.time() - start_time) * 1000
                
                if 'error' not in result:
                    times.append(exec_time)
                    successes += 1
                
            except Exception as e:
                print(f"      ⚠️ Error en query {query}: {e}")
        
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            performance_results[strategy] = {
                "avg_time_ms": round(avg_time, 2),
                "min_time_ms": round(min_time, 2),
                "max_time_ms": round(max_time, 2),
                "success_rate": round((successes / len(test_queries)) * 100, 1),
                "target_met": avg_time < target_time_ms
            }
            
            status = "✅" if avg_time < target_time_ms else "⚠️"
            print(f"      {status} {avg_time:.1f}ms avg (target: <{target_time_ms}ms)")
        else:
            performance_results[strategy] = {"error": "No successful executions"}
            print(f"      ❌ Todas las ejecuciones fallaron")
    
    # Resumen de performance
    successful_strategies = [s for s, r in performance_results.items() if r.get('target_met', False)]
    
    print(f"\n📊 RESUMEN PERFORMANCE:")
    print(f"   🎯 Objetivo: <{target_time_ms}ms")
    print(f"   ✅ Estrategias exitosas: {len(successful_strategies)}/{len(strategies)}")
    
    if successful_strategies:
        best_strategy = min(successful_strategies, 
                          key=lambda s: performance_results[s]['avg_time_ms'])
        best_time = performance_results[best_strategy]['avg_time_ms']
        print(f"   🏆 Mejor performance: {best_strategy} ({best_time}ms)")
    
    return performance_results


def test_recommendation_quality(recommender):
    """Test 4: Validar calidad de recomendaciones."""
    
    print("\n🧪 TEST 4: Calidad de Recomendaciones")
    print("-" * 50)
    
    try:
        # Test con canción conocida
        test_song_idx = 0
        test_song = recommender.dataset.iloc[test_song_idx]
        
        print(f"🎵 Canción test: \"{test_song.get('track_name', 'N/A')}\" - {test_song.get('track_artist', 'N/A')}")
        
        # Generar recomendaciones con estrategia híbrida
        result = recommender.recommend(
            query=test_song_idx,
            strategy="hybrid_balanced",
            n_recommendations=10,
            explain=True
        )
        
        if 'error' in result:
            print(f"❌ Error generando recomendaciones: {result['error']}")
            return False
        
        recommendations = result.get('recommendations', [])
        
        print(f"✅ Recomendaciones generadas: {len(recommendations)}")
        
        # Validar estructura de recomendaciones
        required_fields = ['track_id', 'track_name', 'track_artist', 'strategy_info']
        
        for i, rec in enumerate(recommendations[:3]):
            missing_fields = [f for f in required_fields if f not in rec]
            if missing_fields:
                print(f"⚠️  Recomendación {i+1} falta campos: {missing_fields}")
            else:
                print(f"   {i+1}. {rec['track_name']} - {rec['track_artist']}")
                if 'explanation' in rec:
                    print(f"      💡 {rec['explanation']}")
        
        # Validar diversidad
        clusters = [r.get('cluster', -1) for r in recommendations if 'cluster' in r]
        unique_clusters = len(set(clusters))
        
        print(f"✅ Diversidad: {unique_clusters} clusters únicos de {recommender.n_clusters} posibles")
        
        # Validar scores
        similarities = [r.get('similarity', 0) for r in recommendations if 'similarity' in r]
        if similarities:
            avg_sim = np.mean(similarities)
            print(f"✅ Similitud promedio: {avg_sim:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en test de calidad: {e}")
        return False


def test_data_validation(recommender):
    """Test 5: Validar datos optimizados y clustering."""
    
    print("\n🧪 TEST 5: Validación de Datos Optimizados")
    print("-" * 50)
    
    try:
        # Validar dataset
        dataset = recommender.dataset
        print(f"📊 Dataset: {len(dataset):,} canciones")
        
        # Validar características discriminativas
        available_features = [f for f in recommender.discriminative_features if f in dataset.columns]
        print(f"🎵 Features disponibles: {len(available_features)}/{len(recommender.discriminative_features)}")
        
        if len(available_features) < 8:
            print("⚠️  Pocas características disponibles para clustering óptimo")
        
        # Validar clusters
        cluster_assignments = recommender.cluster_assignments
        unique_clusters = len(set(cluster_assignments))
        print(f"🎯 Clusters únicos: {unique_clusters}")
        
        # Distribución de clusters
        import pandas as pd
        cluster_counts = pd.Series(cluster_assignments).value_counts().sort_index()
        print(f"📈 Distribución clusters:")
        
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(cluster_assignments)) * 100
            print(f"   Cluster {cluster_id}: {count:,} canciones ({percentage:.1f}%)")
        
        # Validar que no hay clusters vacíos
        empty_clusters = [i for i in range(recommender.n_clusters) if i not in cluster_counts.index]
        if empty_clusters:
            print(f"⚠️  Clusters vacíos: {empty_clusters}")
        else:
            print("✅ Todos los clusters tienen canciones asignadas")
        
        # Validar centroides
        if hasattr(recommender, 'cluster_centroids'):
            centroids_count = len(recommender.cluster_centroids)
            print(f"🎯 Centroides calculados: {centroids_count}/{recommender.n_clusters}")
            
            # Validar dimensionalidad de centroides
            if centroids_count > 0:
                first_centroid = list(recommender.cluster_centroids.values())[0]
                centroid_dims = len(first_centroid)
                expected_dims = len(recommender.discriminative_features)
                print(f"📐 Dimensionalidad centroides: {centroid_dims}/{expected_dims}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en validación de datos: {e}")
        return False


def run_complete_test_suite():
    """Ejecutar suite completa de tests."""
    
    print("🚀 INICIANDO TEST SUITE COMPLETO")
    print("=" * 70)
    print("🎯 Objetivo: Validar sistema optimizado y performance <100ms")
    print("=" * 70)
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "target_performance_ms": 100,
        "tests": {}
    }
    
    # Test 1: Inicialización
    recommender, init_success = test_system_initialization()
    test_results["tests"]["initialization"] = {"success": init_success}
    
    if not init_success:
        print("\n❌ FALLO CRÍTICO: No se pudo inicializar el sistema")
        return test_results
    
    # Test 2: Setup completo
    setup_success = test_system_setup(recommender)
    test_results["tests"]["system_setup"] = {"success": setup_success}
    
    if not setup_success:
        print("\n❌ FALLO CRÍTICO: No se pudo completar setup del sistema")
        return test_results
    
    # Test 3: Performance
    performance_results = test_performance_benchmarks(recommender)
    test_results["tests"]["performance"] = performance_results
    
    # Test 4: Calidad
    quality_success = test_recommendation_quality(recommender)
    test_results["tests"]["quality"] = {"success": quality_success}
    
    # Test 5: Validación datos
    validation_success = test_data_validation(recommender)
    test_results["tests"]["data_validation"] = {"success": validation_success}
    
    # Resumen final
    print("\n" + "=" * 70)
    print("🏆 RESUMEN FINAL DEL TEST SUITE")
    print("=" * 70)
    
    all_tests = [
        ("Inicialización", init_success),
        ("Setup Sistema", setup_success),
        ("Calidad", quality_success),
        ("Validación Datos", validation_success)
    ]
    
    passed_tests = sum(1 for _, success in all_tests if success)
    total_tests = len(all_tests)
    
    for test_name, success in all_tests:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    # Resumen performance
    if performance_results:
        successful_strategies = sum(1 for r in performance_results.values() 
                                  if isinstance(r, dict) and r.get('target_met', False))
        total_strategies = len(performance_results)
        print(f"⚡ Performance: {successful_strategies}/{total_strategies} estrategias <100ms")
    
    # Veredicto final
    success_rate = (passed_tests / total_tests) * 100
    
    if success_rate == 100 and successful_strategies >= 3:
        print(f"\n🎉 ¡TEST SUITE EXITOSO! ({success_rate:.0f}% tests passed)")
        print("✅ Sistema optimizado listo para producción")
    elif success_rate >= 80:
        print(f"\n🎯 Test suite mayormente exitoso ({success_rate:.0f}% tests passed)")
        print("⚠️  Revisar fallos menores antes de producción")
    else:
        print(f"\n❌ Test suite falló ({success_rate:.0f}% tests passed)")
        print("🔧 Correcciones requeridas antes de usar sistema")
    
    test_results["summary"] = {
        "success_rate": success_rate,
        "tests_passed": passed_tests,
        "total_tests": total_tests,
        "performance_strategies_passed": successful_strategies if performance_results else 0,
        "production_ready": success_rate == 100 and successful_strategies >= 3
    }
    
    return test_results


def main():
    """Función principal del test suite."""
    
    try:
        # Ejecutar tests
        results = run_complete_test_suite()
        
        # Guardar resultados
        output_file = Path("outputs") / "test_results_optimized_recommender.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Resultados guardados en: {output_file}")
        
        # Código de salida
        if results["summary"]["production_ready"]:
            return 0
        else:
            return 1
        
    except Exception as e:
        print(f"\n❌ Error crítico en test suite: {e}")
        return 1


if __name__ == "__main__":
    print(__doc__)
    exit_code = main()
    exit(exit_code)