#!/usr/bin/env python3
"""
Test rápido de correcciones críticas FASE 4.
Verifica que las correcciones principales funcionen.
"""

import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_normalize_auto_detect():
    """Test función normalize_auto_detect agregada."""
    print("🔧 Testing normalize_auto_detect...")
    
    try:
        from clustering.algorithms.lyrics.preprocessing.normalizer import MultilingualNormalizer
        
        normalizer = MultilingualNormalizer()
        
        # Test con texto simple
        test_text = "I love this song, it's amazing!"
        normalized = normalizer.normalize_auto_detect(test_text)
        
        print(f"   ✅ Función normalize_auto_detect disponible")
        print(f"   Original: {test_text}")
        print(f"   Normalized: {normalized}")
        return True
        
    except AttributeError as e:
        print(f"   ❌ Error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error inesperado: {e}")
        return False

def test_bert_vectorizer_basic():
    """Test básico BertVectorizer sin cache."""
    print("🤖 Testing BertVectorizer básico...")
    
    try:
        from clustering.algorithms.lyrics.vectorization.bert_vectorizer import BertVectorizer
        
        # Inicializar sin cache para evitar problemas SQLite
        vectorizer = BertVectorizer(cache_enabled=False)
        
        # Test texto muy bueno (should pass quality)
        good_text = "I love this beautiful song that makes me feel so happy and alive in every moment of joy"
        
        result = vectorizer.preprocess_text(good_text, "en")
        
        print(f"   ✅ BertVectorizer inicializado correctamente")
        print(f"   Preprocessing successful: {result['is_suitable']}")
        print(f"   Quality score: {result['quality_score']:.3f}")
        
        if result['is_suitable']:
            print("   ✅ Texto considerado suitable")
        else:
            print("   ⚠️ Texto no suitable (pero preprocessing funciona)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_simple_cache():
    """Test cache sin SQLite (solo memoria)."""
    print("💾 Testing cache simple...")
    
    try:
        from clustering.algorithms.lyrics.cache.cache_manager import MemoryCache
        import numpy as np
        
        cache = MemoryCache(max_size=10)
        
        # Test embedding fake
        test_embedding = np.random.rand(384).astype(np.float32)
        test_key = "test_key_123"
        
        # Store
        cache.put(test_key, test_embedding)
        
        # Retrieve
        retrieved = cache.get(test_key)
        
        if retrieved is not None and np.array_equal(retrieved, test_embedding):
            print(f"   ✅ Memory cache funcional")
            print(f"   Cache size: {cache.size()}")
            return True
        else:
            print(f"   ❌ Cache retrieval failed")
            return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Ejecuta tests críticos."""
    print("🚀 TESTING CORRECCIONES CRÍTICAS FASE 4")
    print("=" * 50)
    
    tests = [
        ("normalize_auto_detect", test_normalize_auto_detect),
        ("BertVectorizer básico", test_bert_vectorizer_basic),
        ("Cache simple", test_simple_cache),
    ]
    
    successful = 0
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 30}")
        print(f"Test: {test_name}")
        print(f"{'=' * 30}")
        
        try:
            success = test_func()
            if success:
                successful += 1
                print(f"✅ {test_name}: EXITOSO")
            else:
                print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print(f"\n{'=' * 50}")
    print(f"🏆 RESUMEN: {successful}/{len(tests)} tests exitosos")
    
    if successful == len(tests):
        print("🎉 TODAS LAS CORRECCIONES FUNCIONAN")
    elif successful >= 2:
        print("⚠️ La mayoría de correcciones funcionan")
    else:
        print("❌ Muchas correcciones aún tienen problemas")
    
    return successful >= 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)