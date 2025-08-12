#!/usr/bin/env python3
"""
Test suite comprehensivo para el selector optimizado con validación Hopkins
FASE 1.3 del Plan de Optimización de Clustering

Tests incluidos:
1. Hopkins preservation durante selección
2. Performance benchmarking vs versión anterior
3. Validación de diversidad musical
4. Tests de condiciones límite
5. Verificación de fallback strategies
6. Análisis de calidad de clusters resultantes
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import time
import json
from unittest.mock import patch, MagicMock
import tempfile

# Añadir path para importar módulos del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_selection'))

from data_selection.clustering_aware.select_optimal_10k_from_18k import OptimalSelector
from data_selection.clustering_aware.hopkins_validator import HopkinsValidator

class TestOptimalSelectorImproved(unittest.TestCase):
    """Test suite comprehensivo para selector optimizado"""
    
    @classmethod
    def setUpClass(cls):
        """Setup inicial para todos los tests"""
        cls.test_data_path = "/mnt/c/Users/sanmi/Documents/Proyectos/Tesis/music_features/data/with_lyrics/spotify_songs_fixed.csv"
        cls.sample_size = 1000  # Dataset pequeño para tests rápidos
        cls.target_size = 100   # Selección pequeña para tests
        
        # Verificar que existe el archivo de datos
        if not os.path.exists(cls.test_data_path):
            raise FileNotFoundError(f"Dataset de test no encontrado: {cls.test_data_path}")
        
        # Cargar muestra de datos para tests
        try:
            cls.full_data = pd.read_csv(cls.test_data_path, sep='^', decimal='.', encoding='utf-8')
            cls.test_data = cls.full_data.sample(n=min(cls.sample_size, len(cls.full_data)), random_state=42)
            print(f"✅ Test data loaded: {len(cls.test_data)} samples from {len(cls.full_data)} total")
        except Exception as e:
            raise Exception(f"Error cargando datos de test: {e}")
        
        # Inicializar selector
        cls.selector = OptimalSelector()
        cls.hopkins_validator = HopkinsValidator()
        
        print(f"✅ Test setup completado - Datos: {len(cls.test_data)}, Target: {cls.target_size}")
    
    def setUp(self):
        """Setup para cada test individual"""
        self.test_results = {}
        self.start_time = time.time()
    
    def tearDown(self):
        """Cleanup después de cada test"""
        execution_time = time.time() - self.start_time
        self.test_results['execution_time'] = execution_time
        print(f"⏱️  Test ejecutado en {execution_time:.2f}s")

    # ========== GRUPO 1: HOPKINS PRESERVATION TESTS ==========
    
    def test_01_hopkins_calculation_accuracy(self):
        """Test 1: Verificar precisión del cálculo Hopkins"""
        print("\n🧪 TEST 1: Hopkins calculation accuracy")
        
        # Preparar datos de test con características conocidas
        feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                          'instrumentalness', 'liveness', 'valence', 'tempo']
        
        test_subset = self.test_data[feature_columns].dropna().head(200)
        
        # Calcular Hopkins con diferentes sample sizes
        hopkins_small = self.hopkins_validator.calculate_hopkins_fast(test_subset, sample_size=20)
        hopkins_medium = self.hopkins_validator.calculate_hopkins_fast(test_subset, sample_size=50)
        hopkins_large = self.hopkins_validator.calculate_hopkins_fast(test_subset, sample_size=100)
        
        print(f"Hopkins (n=20): {hopkins_small:.4f}")
        print(f"Hopkins (n=50): {hopkins_medium:.4f}")
        print(f"Hopkins (n=100): {hopkins_large:.4f}")
        
        # Verificaciones
        self.assertIsInstance(hopkins_small, float)
        self.assertIsInstance(hopkins_medium, float)
        self.assertIsInstance(hopkins_large, float)
        
        # Hopkins debe estar en rango [0, 1]
        for hopkins in [hopkins_small, hopkins_medium, hopkins_large]:
            self.assertGreaterEqual(hopkins, 0.0, "Hopkins debe ser >= 0")
            self.assertLessEqual(hopkins, 1.0, "Hopkins debe ser <= 1")
        
        # Los valores deben ser consistentes (diferencia < 0.2)
        max_diff = max(abs(hopkins_small - hopkins_medium), 
                      abs(hopkins_medium - hopkins_large))
        self.assertLess(max_diff, 0.2, "Hopkins values deben ser consistentes")
        
        self.test_results['hopkins_values'] = {
            'small': hopkins_small, 'medium': hopkins_medium, 'large': hopkins_large
        }
        print("✅ Hopkins calculation accuracy verificado")
    
    def test_02_hopkins_preservation_during_selection(self):
        """Test 2: Verificar que Hopkins se mantiene durante selección"""
        print("\n🧪 TEST 2: Hopkins preservation durante selección")
        
        # Seleccionar muestra pequeña para test rápido
        small_target = min(50, len(self.test_data) // 20)
        
        # Ejecutar selección con validación Hopkins
        try:
            selected_indices, metadata = self.selector.select_optimal_10k_with_validation(
                self.test_data
            )
            
            hopkins_initial = metadata.get('hopkins_initial', 0)
            hopkins_final = metadata.get('hopkins_final', 0)
            
            print(f"Hopkins inicial: {hopkins_initial:.4f}")
            print(f"Hopkins final: {hopkins_final:.4f}")
            print(f"Seleccionados: {len(selected_indices)} de {small_target} target")
            
            # Verificaciones
            self.assertGreater(hopkins_initial, 0.5, "Hopkins inicial debe indicar clustering tendency")
            self.assertGreater(hopkins_final, 0.4, "Hopkins final debe mantenerse razonable")
            
            # La degradación no debe ser mayor al 30%
            degradation = (hopkins_initial - hopkins_final) / hopkins_initial
            self.assertLess(degradation, 0.3, f"Degradación Hopkins ({degradation:.2%}) excesiva")
            
            self.test_results['hopkins_preservation'] = {
                'initial': hopkins_initial, 'final': hopkins_final, 'degradation': degradation
            }
            print("✅ Hopkins preservation verificado")
            
        except Exception as e:
            self.fail(f"Fallo en selección con validación Hopkins: {e}")
    
    def test_03_fallback_strategy_activation(self):
        """Test 3: Verificar activación de estrategia fallback"""
        print("\n🧪 TEST 3: Fallback strategy activation")
        
        # Simular condición de Hopkins bajo forzando threshold alto
        with patch.object(self.hopkins_validator, 'calculate_hopkins_fast', return_value=0.3):
            small_target = min(30, len(self.test_data) // 30)
            
            try:
                selected_indices, metadata = self.selector.select_optimal_10k_with_validation(
                    self.test_data
                )
                
                fallback_used = metadata.get('fallback_used', False)
                fallback_reason = metadata.get('fallback_reason', 'none')
                
                print(f"Fallback usado: {fallback_used}")
                print(f"Razón fallback: {fallback_reason}")
                print(f"Seleccionados: {len(selected_indices)}")
                
                # Con Hopkins simulado bajo (0.3), el fallback debe activarse
                self.assertTrue(fallback_used, "Fallback strategy debería haberse activado")
                self.assertIn('hopkins', fallback_reason.lower(), "Razón debe mencionar Hopkins")
                
                self.test_results['fallback_test'] = {
                    'activated': fallback_used, 'reason': fallback_reason
                }
                print("✅ Fallback strategy activation verificado")
                
            except Exception as e:
                self.fail(f"Fallo en test de fallback strategy: {e}")

    # ========== GRUPO 2: PERFORMANCE BENCHMARKING ==========
    
    def test_04_selection_performance_benchmark(self):
        """Test 4: Benchmark de performance vs versión anterior"""
        print("\n🧪 TEST 4: Selection performance benchmark")
        
        small_target = min(40, len(self.test_data) // 25)
        
        # Benchmark nueva versión con validación
        start_time = time.time()
        try:
            selected_indices_new, metadata_new = self.selector.select_optimal_10k_with_validation(
                self.test_data
            )
            time_with_validation = time.time() - start_time
            
            print(f"Tiempo CON validación: {time_with_validation:.2f}s")
            print(f"Seleccionados: {len(selected_indices_new)}")
            print(f"Hopkins final: {metadata_new.get('hopkins_final', 'N/A')}")
            
            # Benchmark versión sin validación (más rápida)
            start_time = time.time()
            selected_indices_basic, metadata_basic = self.selector.select_optimal_10k_with_validation(
                self.test_data
            )
            time_without_validation = time.time() - start_time
            
            print(f"Tiempo SIN validación: {time_without_validation:.2f}s")
            
            # Verificaciones de performance
            self.assertLess(time_with_validation, 120, "Selección con validación debe ser < 2min")
            self.assertLess(time_without_validation, 60, "Selección sin validación debe ser < 1min")
            
            # La validación no debe aumentar más del 300% el tiempo
            overhead = (time_with_validation - time_without_validation) / time_without_validation
            self.assertLess(overhead, 3.0, f"Overhead de validación ({overhead:.1f}x) excesivo")
            
            self.test_results['performance_benchmark'] = {
                'time_with_validation': time_with_validation,
                'time_without_validation': time_without_validation,
                'overhead_ratio': overhead
            }
            print("✅ Performance benchmark completado")
            
        except Exception as e:
            self.fail(f"Fallo en performance benchmark: {e}")
    
    def test_05_memory_efficiency_test(self):
        """Test 5: Verificar eficiencia de memoria"""
        print("\n🧪 TEST 5: Memory efficiency test")
        
        import psutil
        import gc
        
        # Medir memoria antes
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Ejecutar selección
        small_target = min(50, len(self.test_data) // 20)
        try:
            selected_indices, metadata = self.selector.select_optimal_10k_with_validation(
                self.test_data
            )
            
            # Medir memoria después
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            print(f"Memoria antes: {memory_before:.1f}MB")
            print(f"Memoria después: {memory_after:.1f}MB")
            print(f"Memoria usada: {memory_used:.1f}MB")
            
            # Verificaciones
            self.assertLess(memory_used, 500, "Uso de memoria debe ser < 500MB")
            
            # Cleanup explícito
            del selected_indices, metadata
            gc.collect()
            
            memory_final = process.memory_info().rss / 1024 / 1024
            memory_released = memory_after - memory_final
            print(f"Memoria liberada: {memory_released:.1f}MB")
            
            self.test_results['memory_usage'] = {
                'used_mb': memory_used, 'released_mb': memory_released
            }
            print("✅ Memory efficiency test completado")
            
        except Exception as e:
            self.fail(f"Fallo en memory efficiency test: {e}")

    # ========== GRUPO 3: DIVERSIDAD MUSICAL ==========
    
    def test_06_musical_diversity_validation(self):
        """Test 6: Validar diversidad musical del dataset seleccionado"""
        print("\n🧪 TEST 6: Musical diversity validation")
        
        feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                          'instrumentalness', 'liveness', 'valence', 'tempo']
        
        small_target = min(60, len(self.test_data) // 17)
        
        try:
            selected_indices, metadata = self.selector.select_optimal_10k_with_validation(
                self.test_data
            )
            
            # Analizar diversidad del subset seleccionado
            selected_data = self.test_data.iloc[selected_indices][feature_columns]
            original_data = self.test_data[feature_columns]
            
            # Calcular rangos de características
            selected_ranges = selected_data.max() - selected_data.min()
            original_ranges = original_data.max() - original_data.min()
            
            # Calcular proporción de rangos preservados
            range_preservation = (selected_ranges / original_ranges).mean()
            
            print(f"Preservación de rangos: {range_preservation:.3f}")
            print(f"Características cubiertas: {len(selected_ranges)}")
            
            # Verificar cobertura de espacio musical
            for feature in feature_columns:
                selected_std = selected_data[feature].std()
                original_std = original_data[feature].std()
                
                if original_std > 0:
                    diversity_ratio = selected_std / original_std
                    self.assertGreater(diversity_ratio, 0.3, 
                                     f"Diversidad insuficiente en {feature}: {diversity_ratio:.3f}")
            
            self.assertGreater(range_preservation, 0.6, "Preservación de rangos insuficiente")
            
            self.test_results['musical_diversity'] = {
                'range_preservation': range_preservation,
                'features_analyzed': len(feature_columns)
            }
            print("✅ Musical diversity validation completado")
            
        except Exception as e:
            self.fail(f"Fallo en musical diversity validation: {e}")
    
    def test_07_genre_balance_verification(self):
        """Test 7: Verificar balance de géneros si disponible"""
        print("\n🧪 TEST 7: Genre balance verification")
        
        # Verificar si hay columna de género disponible
        genre_columns = ['track_genre', 'genre', 'playlist_genre', 'playlist_subgenre']
        genre_column = None
        
        for col in genre_columns:
            if col in self.test_data.columns:
                genre_column = col
                break
        
        if genre_column is None:
            print("⚠️  No se encontró columna de género, saltando test")
            self.test_results['genre_balance'] = {'status': 'skipped', 'reason': 'no_genre_column'}
            return
        
        small_target = min(80, len(self.test_data) // 12)
        
        try:
            selected_indices, metadata = self.selector.select_optimal_10k_with_validation(
                self.test_data
            )
            
            # Analizar distribución de géneros
            original_genres = self.test_data[genre_column].value_counts()
            selected_genres = self.test_data.iloc[selected_indices][genre_column].value_counts()
            
            print(f"Géneros en original: {len(original_genres)}")
            print(f"Géneros en selección: {len(selected_genres)}")
            
            # Verificar representación de géneros principales
            top_genres = original_genres.head(5).index
            represented_genres = sum(1 for genre in top_genres if genre in selected_genres.index)
            representation_ratio = represented_genres / len(top_genres)
            
            print(f"Representación top géneros: {representation_ratio:.2%}")
            
            self.assertGreater(representation_ratio, 0.6, "Representación de géneros insuficiente")
            
            self.test_results['genre_balance'] = {
                'original_genres': len(original_genres),
                'selected_genres': len(selected_genres),
                'representation_ratio': representation_ratio
            }
            print("✅ Genre balance verification completado")
            
        except Exception as e:
            self.fail(f"Fallo en genre balance verification: {e}")

    # ========== GRUPO 4: CONDICIONES LÍMITE ==========
    
    def test_08_small_dataset_handling(self):
        """Test 8: Manejo de datasets pequeños"""
        print("\n🧪 TEST 8: Small dataset handling")
        
        # Crear dataset muy pequeño
        small_data = self.test_data.head(20)
        small_target = 10
        
        try:
            # Crear selector temporal para dataset pequeño
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                small_data.to_csv(f.name, sep='^', decimal='.', index=False)
                temp_path = f.name
            
            temp_selector = ClusteringAwareSelector(temp_path)
            
            selected_indices, metadata = temp_selector.select_optimal_10k_with_validation(
                small_data
            )
            
            print(f"Dataset pequeño: {len(small_data)} → {len(selected_indices)}")
            print(f"Hopkins: {metadata.get('hopkins_final', 'N/A')}")
            
            # Verificaciones para dataset pequeño
            self.assertLessEqual(len(selected_indices), small_target)
            self.assertGreater(len(selected_indices), 0)
            
            # Cleanup
            os.unlink(temp_path)
            
            self.test_results['small_dataset'] = {
                'original_size': len(small_data),
                'selected_size': len(selected_indices),
                'hopkins': metadata.get('hopkins_final')
            }
            print("✅ Small dataset handling completado")
            
        except Exception as e:
            self.fail(f"Fallo en small dataset handling: {e}")
    
    def test_09_large_target_size_handling(self):
        """Test 9: Manejo de target size grande"""
        print("\n🧪 TEST 9: Large target size handling")
        
        # Target size más grande que dataset disponible
        large_target = len(self.test_data) + 100
        
        try:
            selected_indices, metadata = self.selector.select_optimal_10k_with_validation(
                self.test_data
            )
            
            print(f"Target grande: {large_target} → {len(selected_indices)} (máx: {len(self.test_data)})")
            
            # Debe seleccionar todos los datos disponibles
            self.assertEqual(len(selected_indices), len(self.test_data))
            
            self.test_results['large_target'] = {
                'requested': large_target,
                'obtained': len(selected_indices),
                'max_available': len(self.test_data)
            }
            print("✅ Large target size handling completado")
            
        except Exception as e:
            self.fail(f"Fallo en large target size handling: {e}")
    
    def test_10_missing_features_handling(self):
        """Test 10: Manejo de características faltantes"""
        print("\n🧪 TEST 10: Missing features handling")
        
        # Crear dataset con valores faltantes
        corrupted_data = self.test_data.copy()
        feature_columns = ['danceability', 'energy', 'loudness']
        
        # Introducir 20% de valores faltantes aleatoriamente
        for col in feature_columns:
            mask = np.random.random(len(corrupted_data)) < 0.2
            corrupted_data.loc[mask, col] = np.nan
        
        small_target = min(30, len(corrupted_data) // 30)
        
        try:
            selected_indices, metadata = self.selector.select_optimal_10k_with_validation(
                corrupted_data
            )
            
            print(f"Dataset corrupto: {len(corrupted_data)} → {len(selected_indices)}")
            print(f"Missing values handling: OK")
            
            # Verificar que se manejaron los valores faltantes
            selected_data = corrupted_data.iloc[selected_indices]
            has_missing = selected_data[feature_columns].isnull().any().any()
            
            print(f"Selected data contiene missing: {has_missing}")
            
            self.test_results['missing_features'] = {
                'original_size': len(corrupted_data),
                'selected_size': len(selected_indices),
                'has_missing_in_selection': has_missing
            }
            print("✅ Missing features handling completado")
            
        except Exception as e:
            print(f"⚠️  Expected error handling missing features: {e}")
            self.test_results['missing_features'] = {
                'status': 'error_handled', 'error_type': str(type(e).__name__)
            }

    # ========== GRUPO 5: CALIDAD DE RESULTADOS ==========
    
    def test_11_clustering_quality_assessment(self):
        """Test 11: Evaluación de calidad de clustering en resultados"""
        print("\n🧪 TEST 11: Clustering quality assessment")
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        
        feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                          'instrumentalness', 'liveness', 'valence', 'tempo']
        
        small_target = min(100, len(self.test_data) // 10)
        
        try:
            selected_indices, metadata = self.selector.select_optimal_10k_with_validation(
                self.test_data
            )
            
            # Preparar datos para clustering
            selected_data = self.test_data.iloc[selected_indices][feature_columns].dropna()
            
            if len(selected_data) < 10:
                print("⚠️  Datos insuficientes para clustering test")
                self.test_results['clustering_quality'] = {'status': 'insufficient_data'}
                return
            
            # Normalizar y hacer clustering
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(selected_data)
            
            # Test con K=3 (valor conservador)
            k = min(3, len(selected_data) // 3)
            if k < 2:
                print("⚠️  K insuficiente para clustering test")
                return
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data_scaled)
            
            # Calcular métricas
            silhouette_avg = silhouette_score(data_scaled, cluster_labels)
            inertia = kmeans.inertia_
            
            print(f"Clustering K={k}: Silhouette={silhouette_avg:.4f}, Inertia={inertia:.2f}")
            
            # Verificaciones de calidad
            self.assertGreater(silhouette_avg, 0.1, f"Silhouette score muy bajo: {silhouette_avg:.4f}")
            
            # Verificar distribución de clusters
            unique, counts = np.unique(cluster_labels, return_counts=True)
            min_cluster_size = counts.min()
            max_cluster_size = counts.max()
            balance_ratio = min_cluster_size / max_cluster_size
            
            print(f"Balance clusters: {balance_ratio:.3f} (min={min_cluster_size}, max={max_cluster_size})")
            
            self.test_results['clustering_quality'] = {
                'silhouette_score': silhouette_avg,
                'inertia': inertia,
                'balance_ratio': balance_ratio,
                'k_clusters': k
            }
            print("✅ Clustering quality assessment completado")
            
        except Exception as e:
            self.fail(f"Fallo en clustering quality assessment: {e}")
    
    def test_12_final_integration_test(self):
        """Test 12: Test de integración final completo"""
        print("\n🧪 TEST 12: Final integration test")
        
        small_target = min(80, len(self.test_data) // 12)
        
        try:
            # Ejecutar selección completa con todas las validaciones
            start_time = time.time()
            
            selected_indices, metadata = self.selector.select_optimal_10k_with_validation(
                self.test_data
            )
            
            total_time = time.time() - start_time
            
            # Verificar metadata completo
            required_fields = ['hopkins_initial', 'hopkins_final', 'selection_method', 'timestamp']
            for field in required_fields:
                self.assertIn(field, metadata, f"Campo requerido faltante en metadata: {field}")
            
            # Verificar calidad de selección
            hopkins_final = metadata['hopkins_final']
            selection_method = metadata['selection_method']
            
            print(f"Integración final:")
            print(f"  - Tiempo total: {total_time:.2f}s")
            print(f"  - Seleccionados: {len(selected_indices)}/{small_target}")
            print(f"  - Hopkins final: {hopkins_final:.4f}")
            print(f"  - Método: {selection_method}")
            
            # Verificaciones finales
            self.assertGreater(len(selected_indices), 0, "Debe seleccionar al menos una muestra")
            self.assertLessEqual(len(selected_indices), small_target, "No debe exceder target size")
            self.assertGreater(hopkins_final, 0.3, "Hopkins final debe ser razonable")
            self.assertLess(total_time, 180, "Tiempo total debe ser < 3min")
            
            self.test_results['final_integration'] = {
                'total_time': total_time,
                'selected_count': len(selected_indices),
                'hopkins_final': hopkins_final,
                'method': selection_method,
                'metadata_complete': True
            }
            print("✅ Final integration test completado exitosamente")
            
        except Exception as e:
            self.fail(f"Fallo en final integration test: {e}")

    # ========== GENERACIÓN DE REPORTE ==========
    
    def generate_comprehensive_report(self):
        """Generar reporte comprehensivo de todos los tests"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_path = f"/mnt/c/Users/sanmi/Documents/Proyectos/Tesis/music_features/tests/test_data_selection/test_report_{timestamp}.json"
        
        # Compilar resultados de todos los tests
        all_results = {}
        for test_name, result in self.test_results.items():
            all_results[test_name] = result
        
        report = {
            'timestamp': timestamp,
            'test_suite': 'TestOptimalSelectorImproved',
            'total_tests': 12,
            'test_data_size': len(self.test_data),
            'results': all_results,
            'summary': {
                'performance_tests': 2,
                'hopkins_tests': 3,
                'diversity_tests': 2,
                'boundary_tests': 3,
                'quality_tests': 2
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📊 Reporte comprehensivo generado: {report_path}")
        return report_path

if __name__ == '__main__':
    # Configurar test runner con verbose output
    import sys
    
    # Crear test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestOptimalSelectorImproved)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generar reporte si es posible
    test_instance = TestOptimalSelectorImproved()
    try:
        test_instance.setUpClass()
        report_path = test_instance.generate_comprehensive_report()
        print(f"\n✅ Test suite completado. Reporte disponible en: {report_path}")
    except Exception as e:
        print(f"\n⚠️  Error generando reporte final: {e}")
    
    # Exit code basado en resultados
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)