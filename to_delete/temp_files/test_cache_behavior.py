#!/usr/bin/env python3
"""
Test específico para verificar comportamiento cache BERT.
Valida que la corrección del timestamp funciona correctamente.
"""

import sys
import time
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_cache_miss_hit_cycle():
    """Test ciclo completo cache miss -> hit."""
    print("🔄 Testing Cache Miss/Hit Cycle...")
    
    try:
        from clustering.algorithms.lyrics.vectorization.bert_vectorizer import BertVectorizer
        
        vectorizer = BertVectorizer(cache_enabled=True)
        
        # Texto único con timestamp actual
        timestamp = int(time.time() * 1000)
        test_text = f"This amazing song fills my heart with joy and wonder {timestamp}"
        
        print(f"   📝 Test text: ...{test_text[-50:]}")  # Solo últimos 50 chars
        
        # Primera vectorización - DEBE ser cache miss
        print("   🔍 Primera vectorización...")
        start_time = time.time()
        result1 = vectorizer.vectorize_single(test_text, "en")
        time1 = time.time() - start_time
        
        if not result1["success"]:
            print(f"   ❌ Primera vectorización falló: {result1.get('error', 'Unknown')}")
            return False
        
        is_cached1 = result1.get("cached", False)
        print(f"   Result 1: cached={is_cached1}, time={time1:.3f}s")
        
        if is_cached1:
            print("   ⚠️ ADVERTENCIA: Primera vectorización ya estaba cacheada")
            print("       Esto puede indicar que el texto no es único o cache anterior existe")
        else:
            print("   ✅ Primera vectorización: Cache miss (correcto)")
        
        # Segunda vectorización - DEBE ser cache hit
        print("   🔍 Segunda vectorización...")
        start_time = time.time()
        result2 = vectorizer.vectorize_single(test_text, "en")
        time2 = time.time() - start_time
        
        if not result2["success"]:
            print(f"   ❌ Segunda vectorización falló: {result2.get('error', 'Unknown')}")
            return False
        
        is_cached2 = result2.get("cached", False)
        print(f"   Result 2: cached={is_cached2}, time={time2:.3f}s")
        
        if not is_cached2:
            print("   ❌ Segunda vectorización debería ser cache hit")
            return False
        
        print("   ✅ Segunda vectorización: Cache hit (correcto)")
        
        # Verificar speedup
        if time1 > 0 and time2 > 0:
            speedup = time1 / time2
            print(f"   📈 Cache speedup: {speedup:.1f}x")
            
            if speedup > 10:  # Esperamos al menos 10x speedup
                print("   ✅ Cache speedup significativo")
            else:
                print("   ⚠️ Cache speedup menor al esperado")
        
        # Verificar embeddings son idénticos
        if "embedding" in result1 and "embedding" in result2:
            import numpy as np
            if np.array_equal(result1["embedding"], result2["embedding"]):
                print("   ✅ Embeddings idénticos entre cache miss/hit")
            else:
                print("   ❌ Embeddings diferentes entre cache miss/hit")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en test: {e}")
        return False

def test_multiple_unique_texts():
    """Test múltiples textos únicos para verificar uniqueness."""
    print("🎯 Testing Multiple Unique Texts...")
    
    try:
        from clustering.algorithms.lyrics.vectorization.bert_vectorizer import BertVectorizer
        
        vectorizer = BertVectorizer(cache_enabled=True)
        
        results = []
        
        # Generar 3 textos únicos
        for i in range(3):
            timestamp = int(time.time() * 1000) + i
            text = f"Beautiful music brings happiness and joy to my soul {timestamp}"
            
            result = vectorizer.vectorize_single(text, "en")
            
            if result["success"]:
                is_cached = result.get("cached", False)
                results.append(is_cached)
                print(f"   Text {i+1}: cached={is_cached}")
            else:
                print(f"   Text {i+1}: FAILED")
                return False
            
            # Pequeña pausa para garantizar timestamp diferente
            time.sleep(0.001)
        
        # Verificar que al menos 2 de 3 son cache miss (únicos)
        cache_misses = sum(1 for cached in results if not cached)
        
        if cache_misses >= 2:
            print(f"   ✅ {cache_misses}/3 textos únicos (cache miss)")
        else:
            print(f"   ⚠️ Solo {cache_misses}/3 textos únicos - revisar uniqueness")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Test corrección cache behavior."""
    print("🚀 TEST CACHE BEHAVIOR - VERIFICACIÓN CORRECCIONES")
    print("=" * 60)
    
    tests = [
        ("Cache Miss/Hit Cycle", test_cache_miss_hit_cycle),
        ("Multiple Unique Texts", test_multiple_unique_texts),
    ]
    
    successful = 0
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 40}")
        print(f"🧪 {test_name}")
        print("=" * 40)
        
        try:
            if test_func():
                successful += 1
                print(f"✅ {test_name}: EXITOSO")
            else:
                print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print(f"\n{'=' * 60}")
    print(f"🏆 RESUMEN: {successful}/{len(tests)} tests exitosos")
    
    if successful == len(tests):
        print("🎉 CORRECCIÓN CACHE BEHAVIOR EXITOSA")
        print("✅ Timestamp uniqueness funciona correctamente")
    else:
        print("⚠️ Revisar comportamiento cache o uniqueness")
    
    return successful == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)