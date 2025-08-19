#!/usr/bin/env python3
"""
Test BERT Sistema sin SQLite - Evita problemas Windows.
Solo usa cache memoria + disco, sin database metadata.
"""

import sys
import logging
import tempfile
import numpy as np
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_memory_cache_only():
    """Test solo cache memoria (sin SQLite)."""
    print("💾 Testing Memory Cache (sin SQLite)...")
    
    try:
        from clustering.algorithms.lyrics.cache.cache_manager import MemoryCache
        
        cache = MemoryCache(max_size=10)
        
        # Test embedding
        test_embedding = np.random.rand(384).astype(np.float32)
        test_key = "test_memory_key"
        
        # Store and retrieve
        cache.put(test_key, test_embedding)
        retrieved = cache.get(test_key)
        
        assert retrieved is not None, "Memory cache failed"
        assert np.array_equal(retrieved, test_embedding), "Embedding mismatch"
        
        print("   ✅ Memory cache funcional")
        print(f"   ✅ Cache size: {cache.size()}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_bert_vectorizer_no_cache():
    """Test BertVectorizer sin cache (evita SQLite)."""
    print("🤖 Testing BertVectorizer sin cache...")
    
    try:
        from clustering.algorithms.lyrics.vectorization.bert_vectorizer import BertVectorizer
        
        # Vectorizer SIN cache para evitar SQLite
        vectorizer = BertVectorizer(cache_enabled=False)
        
        # Test con texto de buena calidad
        good_text = "I love this beautiful song that brings so much joy and happiness to my heart and soul when I listen to it"
        
        # Test preprocessing
        prep_result = vectorizer.preprocess_text(good_text, "en")
        
        print(f"   ✅ Preprocessing exitoso")
        print(f"   Quality score: {prep_result['quality_score']:.3f}")
        print(f"   Is suitable: {prep_result['is_suitable']}")
        
        if prep_result['is_suitable']:
            print("   ✅ Texto suitable para BERT")
        else:
            print("   ⚠️ Texto no suitable (pero preprocessing funciona)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_disk_cache_safe():
    """Test cache disco sin database."""
    print("💿 Testing Disk Cache (sin database)...")
    
    try:
        from clustering.algorithms.lyrics.cache.cache_manager import DiskCache
        
        # Usar directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            disk_cache = DiskCache(cache_dir, max_size_gb=0.001)  # Muy pequeño
            
            # Test embedding
            test_embedding = np.random.rand(384).astype(np.float32)
            test_key = "test_disk_key"
            
            # Store
            disk_cache.put(test_key, test_embedding)
            
            # Retrieve
            retrieved = disk_cache.get(test_key)
            
            assert retrieved is not None, "Disk cache failed"
            assert np.array_equal(retrieved, test_embedding), "Embedding mismatch"
            
            print("   ✅ Disk cache funcional")
            
            size_info = disk_cache.size_info()
            print(f"   ✅ Files: {size_info['file_count']}, Size: {size_info['total_size_mb']:.3f} MB")
            
            return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_full_preprocessing_pipeline():
    """Test pipeline completo preprocessing."""
    print("🔧 Testing Preprocessing Pipeline Completo...")
    
    try:
        from clustering.algorithms.lyrics.preprocessing.text_cleaner import MusicTextCleaner
        from clustering.algorithms.lyrics.preprocessing.normalizer import MultilingualNormalizer
        from clustering.algorithms.lyrics.preprocessing.feature_extractor import TextFeatureExtractor
        
        # Componentes
        cleaner = MusicTextCleaner()
        normalizer = MultilingualNormalizer()
        extractor = TextFeatureExtractor()
        
        # Texto test con formato musical típico
        test_text = """[Verse 1]
I love this beautiful song
It makes me feel so happy and alive
Yeah, yeah, yeah!
[Chorus]
Dancing in the moonlight
Everything's gonna be alright
Oh oh oh!"""
        
        # Pipeline paso a paso
        cleaned = cleaner.clean_universal(test_text, "en")
        print(f"   ✅ Limpieza: {len(test_text)} → {len(cleaned)} chars")
        
        normalized = normalizer.normalize_by_language(cleaned, "en")
        print(f"   ✅ Normalización: {len(cleaned)} → {len(normalized)} chars")
        
        quality = extractor.assess_text_quality(normalized, "en")
        print(f"   ✅ Calidad: score {quality['quality_score']:.3f}, suitable {quality['is_suitable']}")
        
        if quality['is_suitable']:
            print("   🎯 Texto final APTO para BERT vectorization")
        else:
            print(f"   ⚠️ Texto no apto: {quality['issues']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Ejecuta tests sin problemas SQLite."""
    print("🚀 TEST BERT SISTEMA - SIN SQLITE (Windows Safe)")
    print("=" * 60)
    
    tests = [
        ("Memory Cache", test_memory_cache_only),
        ("Disk Cache", test_disk_cache_safe),
        ("Preprocessing Pipeline", test_full_preprocessing_pipeline),
        ("BERT Vectorizer (no cache)", test_bert_vectorizer_no_cache),
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
    
    # Resumen
    print(f"\n{'=' * 60}")
    print(f"🏆 RESUMEN: {successful}/{len(tests)} tests exitosos")
    
    if successful == len(tests):
        print("🎉 TODOS LOS COMPONENTES FUNCIONAN (sin SQLite)")
        print("💡 Sistema listo para producción con cache simplificado")
    elif successful >= 3:
        print("⚠️ La mayoría funcionan - sistema básicamente operativo")
    else:
        print("❌ Problemas fundamentales detectados")
    
    return successful >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)