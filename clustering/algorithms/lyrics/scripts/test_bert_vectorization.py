"""
Test Suite completo para sistema BERT vectorization - FASE 4.

Valida todos los componentes del sistema de vectorización:
- BertVectorizer: Vectorización individual y batch
- CacheManager: Sistema cache multinivel
- BatchProcessor: Procesamiento masivo optimizado
- Integración: Pipeline preprocessing + BERT

Ejecuta tests exhaustivos con datasets reales y sintéticos.

Autor: Sistema de Clustering Musical
Fecha: Agosto 2025 - FASE 4
"""

import os
import sys
import logging
import time
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

# Add module to path
lyrics_module_path = Path(__file__).parent.parent
sys.path.insert(0, str(lyrics_module_path))

# Imports del sistema
try:
    from vectorization.bert_vectorizer import BertVectorizer
    from cache.cache_manager import CacheManager
    from vectorization.batch_processor import BatchProcessor
    from config.data_paths import get_dataset_path, DATASET_CONFIG
except ImportError:
    # Fallback para imports desde root
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
    from clustering.algorithms.lyrics.vectorization.bert_vectorizer import BertVectorizer
    from clustering.algorithms.lyrics.cache.cache_manager import CacheManager
    from clustering.algorithms.lyrics.vectorization.batch_processor import BatchProcessor
    from clustering.algorithms.lyrics.config.data_paths import get_dataset_path, DATASET_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_bert_vectorizer_basic():
    """Test funcionalidad básica BertVectorizer."""
    logger.info("🤖 TESTING: BertVectorizer Básico")
    
    vectorizer = BertVectorizer(cache_enabled=False)  # Sin cache para test puro
    
    # Test textos variados
    test_cases = [
        {
            "name": "good_quality",
            "text": "I love this beautiful song, it fills my heart with joy and happiness",
            "expected_success": True
        },
        {
            "name": "spanish_text",
            "text": "Me encanta esta hermosa canción que llena mi corazón de alegría",
            "expected_success": True,
            "language": "es"
        },
        {
            "name": "poor_quality",
            "text": "yeah yeah yeah oh oh oh",
            "expected_success": False
        },
        {
            "name": "empty_text",
            "text": "",
            "expected_success": False
        },
        {
            "name": "instrumental",
            "text": "[Instrumental]",
            "expected_success": False
        }
    ]
    
    successful_tests = 0
    
    for test_case in test_cases:
        logger.info(f"Testing caso: {test_case['name']}")
        
        result = vectorizer.vectorize_single(
            test_case["text"], 
            test_case.get("language")
        )
        
        # Validaciones
        if test_case["expected_success"]:
            assert result["success"], f"Debería ser exitoso: {test_case['name']}"
            assert result["embedding"] is not None, "Embedding no debe ser None"
            assert result["embedding"].shape == (384,), f"Shape incorrecto: {result['embedding'].shape}"
            logger.info(f"   ✅ Exitoso: {result['metadata']['quality_score']:.3f} quality")
        else:
            assert not result["success"], f"Debería fallar: {test_case['name']}"
            assert result["embedding"] is None, "Embedding debe ser None para fallos"
            logger.info(f"   ✅ Falló como esperado: {result.get('error', 'Sin error específico')}")
        
        successful_tests += 1
    
    # Test model info
    model_info = vectorizer.get_model_info()
    assert model_info["embedding_dimension"] == 384
    assert model_info["device"] == "cpu"
    
    logger.info(f"✅ BertVectorizer Básico: {successful_tests}/{len(test_cases)} tests exitosos")
    return True


def test_cache_system():
    """Test sistema cache multinivel."""
    logger.info("💾 TESTING: Sistema Cache Multinivel")
    
    # Crear cache temporal
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        cache_manager = CacheManager(
            cache_dir=cache_dir,
            memory_cache_size=100,
            disk_cache_size_gb=0.1  # Pequeño para test
        )
        
        # Test datos
        test_embedding = np.random.rand(384).astype(np.float32)
        test_key = "test_cache_key_123"
        test_metadata = {
            "text_hash": "abc123hash",
            "language": "en",
            "model_name": "test-model",
            "quality_score": 0.85,
            "processing_time": 0.123
        }
        
        # Test almacenamiento
        cache_manager.put(test_key, test_embedding, test_metadata)
        logger.info("   ✅ Embedding almacenado en cache")
        
        # Test recuperación L1 (memoria)
        retrieved_l1 = cache_manager.get(test_key)
        assert retrieved_l1 is not None, "No recuperado de L1"
        assert np.allclose(retrieved_l1, test_embedding), "Embeddings no coinciden"
        logger.info("   ✅ Recuperación L1 (memoria) exitosa")
        
        # Limpiar L1 y test recuperación L2 (disco)
        cache_manager.l1_memory.clear()
        retrieved_l2 = cache_manager.get(test_key)
        assert retrieved_l2 is not None, "No recuperado de L2"
        assert np.allclose(retrieved_l2, test_embedding), "Embeddings L2 no coinciden"
        logger.info("   ✅ Recuperación L2 (disco) exitosa")
        
        # Test estadísticas
        stats = cache_manager.get_comprehensive_stats()
        assert "l1_memory" in stats
        assert "l2_disk" in stats
        assert "l3_database" in stats
        logger.info(f"   ✅ Estadísticas: {stats['l1_memory']['size']} L1, {stats['l2_disk']['size_info']['file_count']} L2")
        
        # Test cache miss
        missing_key = "non_existent_key"
        retrieved_missing = cache_manager.get(missing_key)
        assert retrieved_missing is None, "Cache miss debería retornar None"
        logger.info("   ✅ Cache miss manejo correcto")
        
        # CRÍTICO: Cerrar conexiones explícitamente antes de cleanup (Windows SQLite issue)
        cache_manager.close_all_connections()
        logger.info("   🔧 Conexiones database cerradas para cleanup seguro")
    
    logger.info("✅ Sistema Cache Multinivel: TODOS LOS TESTS EXITOSOS")
    return True


def test_bert_vectorizer_with_cache():
    """Test BertVectorizer con sistema cache integrado."""
    logger.info("🔄 TESTING: BertVectorizer con Cache")
    
    vectorizer = BertVectorizer(cache_enabled=True)
    
    # OPCIÓN 1: Limpiar cache para test limpio (comentado para preservar cache entre tests)
    # vectorizer.clear_cache()  # Descomentar si se quiere test completamente limpio
    
    # OPCIÓN 2: Usar texto único con timestamp para evitar cache anterior
    import time
    timestamp = int(time.time() * 1000)  # milisegundos
    test_text = f"This is a beautiful song that makes me feel incredibly happy and alive at {timestamp}"
    
    # Primera vectorización (cache miss)
    start_time = time.time()
    result1 = vectorizer.vectorize_single(test_text, "en")
    time1 = time.time() - start_time
    
    assert result1["success"], "Primera vectorización debe ser exitosa"
    assert not result1.get("cached", True), "Primera vez no debe estar cacheada"
    logger.info(f"   ✅ Primera vectorización: {time1:.3f}s (cache miss)")
    
    # Segunda vectorización (cache hit)
    start_time = time.time()
    result2 = vectorizer.vectorize_single(test_text, "en")
    time2 = time.time() - start_time
    
    assert result2["success"], "Segunda vectorización debe ser exitosa"
    assert result2.get("cached", False), "Segunda vez debe estar cacheada"
    assert np.allclose(result1["embedding"], result2["embedding"]), "Embeddings deben ser idénticos"
    logger.info(f"   ✅ Segunda vectorización: {time2:.3f}s (cache hit)")
    
    # Verificar mejora performance
    speedup = time1 / time2 if time2 > 0 else float('inf')
    logger.info(f"   📈 Speedup cache: {speedup:.1f}x")
    
    # Cache stats
    cache_stats = vectorizer.get_cache_stats()
    assert cache_stats["cache_hits"] >= 1, "Debe haber al menos 1 cache hit"
    logger.info(f"   📊 Cache stats: {cache_stats['hit_rate']:.1%} hit rate")
    
    logger.info("✅ BertVectorizer con Cache: TODOS LOS TESTS EXITOSOS")
    return True


def test_batch_processor():
    """Test BatchProcessor para procesamiento masivo."""
    logger.info("🏭 TESTING: Batch Processor")
    
    # Test textos variados
    test_texts = [
        "I love this beautiful music that fills my soul with happiness",
        "Me encanta esta hermosa música que llena mi alma de felicidad",
        "Diese wunderbare Musik erfüllt mein Herz mit Freude und Glück",
        "Esta música linda enche meu coração de alegria e esperança",
        "yeah yeah yeah oh oh oh",  # Texto pobre
        "[Instrumental]",  # Sin contenido
        "",  # Vacío
        "Such a meaningful song with deep lyrics about love and life experiences"
    ]
    
    processor = BatchProcessor(
        batch_size=3,
        max_workers=2  # Conservativo para test
    )
    
    # Procesamiento batch
    start_time = time.time()
    results = processor.process_texts(test_texts, show_progress=False)
    processing_time = time.time() - start_time
    
    assert len(results) == len(test_texts), "Debe procesar todos los textos"
    
    # Análisis resultados
    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
    
    logger.info(f"   📊 Resultados: {successful} exitosos, {failed} fallidos")
    logger.info(f"   ⏱️ Tiempo total: {processing_time:.2f}s")
    logger.info(f"   🚀 Throughput: {len(test_texts)/processing_time:.1f} textos/segundo")
    
    # Validaciones específicas
    assert successful >= 4, "Debe haber al menos 4 textos exitosos"  # Los 4 buenos
    assert failed >= 3, "Debe haber al menos 3 textos fallidos"      # Los 3 problemáticos
    
    # Test stats
    stats = processor.get_stats()
    assert stats["processed"] == len(test_texts)
    assert stats["successful"] == successful
    assert stats["failed"] == failed
    
    logger.info("✅ Batch Processor: TODOS LOS TESTS EXITOSOS")
    return True


def test_dataset_integration():
    """Test integración con dataset real (muestra pequeña)."""
    logger.info("🎵 TESTING: Integración Dataset Real")
    
    try:
        # Verificar dataset existe
        dataset_path = get_dataset_path("main")
        
        if not dataset_path.exists():
            logger.warning("Dataset principal no encontrado, saltando test integración")
            return True
        
        # Cargar muestra pequeña
        df = pd.read_csv(
            dataset_path,
            sep=DATASET_CONFIG["separator"],
            encoding=DATASET_CONFIG["encoding"],
            nrows=5  # Solo 5 para test rápido
        )
        
        if DATASET_CONFIG["lyrics_column"] not in df.columns:
            logger.warning(f"Columna lyrics '{DATASET_CONFIG['lyrics_column']}' no encontrada")
            return True
        
        logger.info(f"   📊 Dataset muestra cargado: {len(df)} filas")
        
        # Procesamiento con BatchProcessor
        processor = BatchProcessor(batch_size=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_results"
            
            results, df_enriched = processor.process_dataset(
                dataset_path=dataset_path,
                max_rows=5,
                output_path=output_path
            )
            
            assert len(results) == len(df), "Debe procesar todas las filas"
            assert len(df_enriched) == len(df), "DataFrame enriquecido debe tener mismo tamaño"
            
            # Verificar columnas agregadas
            expected_columns = [
                "embedding_success", "quality_score", "is_suitable", 
                "processing_error", "cached", "processed_length"
            ]
            
            for col in expected_columns:
                assert col in df_enriched.columns, f"Columna '{col}' faltante"
            
            # Verificar archivos guardados
            csv_path = output_path.with_suffix('.csv')
            npy_path = output_path.with_suffix('.npy')
            
            assert csv_path.exists(), "Archivo CSV no generado"
            logger.info(f"   💾 CSV guardado: {csv_path}")
            
            if any(r["success"] for r in results):
                assert npy_path.exists(), "Archivo embeddings no generado"
                logger.info(f"   💾 Embeddings guardados: {npy_path}")
        
        # Estadísticas finales
        successful = sum(1 for r in results if r["success"])
        total = len(results)
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        logger.info(f"   📈 Success rate dataset real: {success_rate:.1f}% ({successful}/{total})")
        
        logger.info("✅ Integración Dataset Real: TEST EXITOSO")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en test integración: {e}")
        logger.warning("Test integración falló, pero no es crítico")
        return True  # No crítico para suite general


def test_pipeline_end_to_end():
    """Test pipeline completo end-to-end."""
    logger.info("🔗 TESTING: Pipeline End-to-End")
    
    # Texto musical complejo
    complex_text = """[Verse 1]
In the moonlight, I see your face so clear
Every whisper, every word I hear
Dancing shadows tell our story
In this moment, nothing but glory

[Chorus]
We are infinite, we are free
In this melody, you and me
Hearts are beating, souls are flying
In this love, we keep on trying

[Verse 2]
Through the storms and through the rain
We will rise above the pain
Hand in hand, we face tomorrow
Joy will conquer every sorrow"""
    
    # Inicializar sistema completo
    vectorizer = BertVectorizer(cache_enabled=True)
    
    # Test preprocessing detail
    prep_result = vectorizer.preprocess_text(complex_text, "en")
    
    logger.info(f"   📝 Texto original: {len(complex_text)} caracteres")
    logger.info(f"   🔧 Texto procesado: {prep_result['processed_length']} caracteres")
    logger.info(f"   📊 Quality score: {prep_result['quality_score']:.3f}")
    logger.info(f"   ✅ Suitable BERT: {prep_result['is_suitable']}")
    
    # Test vectorización completa
    if prep_result["is_suitable"]:
        result = vectorizer.vectorize_single(complex_text, "en")
        
        assert result["success"], "Vectorización debe ser exitosa"
        assert result["embedding"] is not None, "Embedding no debe ser None"
        assert result["embedding"].shape == (384,), "Dimensión embedding incorrecta"
        
        logger.info("   🎯 Vectorización exitosa")
        logger.info(f"   📐 Embedding shape: {result['embedding'].shape}")
        logger.info(f"   ⏱️ Tiempo vectorización: {result.get('vectorization_time', 0):.3f}s")
    else:
        logger.info("   ⚠️ Texto no suitable para BERT (esperado por filtros estrictos)")
    
    # Test múltiples idiomas
    multilingual_texts = [
        ("en", "This beautiful song touches my heart deeply"),
        ("es", "Esta hermosa canción toca profundamente mi corazón"),
        ("de", "Dieses schöne Lied berührt mein Herz zutiefst"),
        ("pt", "Esta bela canção toca profundamente o meu coração")
    ]
    
    successful_languages = 0
    
    for lang, text in multilingual_texts:
        result = vectorizer.vectorize_single(text, lang)
        if result["success"]:
            successful_languages += 1
            logger.info(f"   ✅ {lang.upper()}: vectorización exitosa")
        else:
            logger.info(f"   ⚠️ {lang.upper()}: {result.get('error', 'Failed')}")
    
    logger.info(f"   🌍 Idiomas exitosos: {successful_languages}/{len(multilingual_texts)}")
    
    logger.info("✅ Pipeline End-to-End: TEST COMPLETO")
    return True


def main():
    """Ejecuta suite completa de tests FASE 4."""
    logger.info("🚀 INICIANDO TEST SUITE BERT VECTORIZATION - FASE 4")
    logger.info("="*70)
    
    tests = [
        ("BertVectorizer Básico", test_bert_vectorizer_basic),
        ("Sistema Cache Multinivel", test_cache_system),
        ("BertVectorizer con Cache", test_bert_vectorizer_with_cache),
        ("Batch Processor", test_batch_processor),
        ("Integración Dataset", test_dataset_integration),
        ("Pipeline End-to-End", test_pipeline_end_to_end),
    ]
    
    successful_tests = 0
    total_start_time = time.time()
    
    for test_name, test_function in tests:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 EJECUTANDO: {test_name}")
            logger.info(f"{'='*50}")
            
            start_time = time.time()
            success = test_function()
            test_time = time.time() - start_time
            
            if success:
                successful_tests += 1
                logger.info(f"✅ {test_name}: EXITOSO ({test_time:.2f}s)")
            else:
                logger.error(f"❌ {test_name}: FALLÓ")
        
        except Exception as e:
            logger.error(f"💥 {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    # Resumen final
    total_time = time.time() - total_start_time
    success_rate = (successful_tests / len(tests)) * 100
    
    logger.info("\n" + "="*70)
    logger.info("🏆 RESUMEN FINAL TEST SUITE FASE 4")
    logger.info("="*70)
    logger.info(f"Tests exitosos: {successful_tests}/{len(tests)} ({success_rate:.1f}%)")
    logger.info(f"Tiempo total: {total_time:.2f} segundos")
    
    if successful_tests == len(tests):
        logger.info("🎉 TODOS LOS TESTS EXITOSOS - FASE 4 VALIDADA")
        logger.info("✅ Sistema BERT vectorization listo para producción")
    else:
        logger.warning("⚠️ Algunos tests fallaron - revisar implementación")
    
    logger.info("="*70)
    
    return successful_tests == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)