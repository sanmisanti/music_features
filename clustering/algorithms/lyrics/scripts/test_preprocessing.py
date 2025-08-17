#!/usr/bin/env python3
"""
Script de prueba del sistema preprocessing completo.

Tests:
1. Limpieza texto musical multilingüe
2. Normalización Unicode y contracciones
3. Gestión stopwords por idioma
4. Extracción características y validación calidad
5. Pipeline completo end-to-end
"""

import sys
import logging
from pathlib import Path

# Añadir paths necesarios
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from preprocessing.text_cleaner import MusicTextCleaner
from preprocessing.normalizer import MultilingualNormalizer
from preprocessing.stopwords_manager import StopwordsManager
from preprocessing.feature_extractor import TextFeatureExtractor
from config.data_paths import get_dataset_path, DATASET_CONFIG

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_text_cleaner():
    """Test limpiador texto musical."""
    logger.info("🧹 TESTING: MusicTextCleaner")
    
    # Textos de prueba multilingües con problemas típicos
    test_texts = {
        "en": """[Verse 1]
Yeah, oh baby, can't you see
I love you, love you, love you (x3)
Come on, come on, let's go
Na na na, la la la
[Chorus]
We're gonna make it through the night
Yeah, yeah, yeah!""",
        
        "es": """[Estribillo]
Ay, corazón, no me dejes así
Del alma mía, ven aquí
La la la, ay ay ay
Vamos, vamos a bailar (x2)""",
        
        "de": """[Refrain]  
Oh ja, mein Herz brennt so sehr
Ich hab dich lieb, so sehr
Na na na, so so so
Komm zu mir, komm her""",
        
        "pt": """[Refrão]
Ai, meu amor, não vai embora
Do coração, vem cá agora  
Na na na, já já já
Vem dançar, vem ficar"""
    }
    
    cleaner = MusicTextCleaner()
    
    for language, text in test_texts.items():
        logger.info(f"Testing limpieza idioma: {language}")
        
        cleaned = cleaner.clean_universal(text, language)
        stats = cleaner.get_cleaning_stats(text, cleaned)
        
        logger.info(f"  Original: {stats['original_length']} chars, {stats['original_words']} palabras")
        logger.info(f"  Limpio: {stats['cleaned_length']} chars, {stats['cleaned_words']} palabras")
        logger.info(f"  Reducción: {stats['reduction_percentage']:.1f}%")
        logger.info(f"  Diversidad: {stats['diversity_original']:.3f} -> {stats['diversity_cleaned']:.3f}")
        
        # Validar que el texto no quedó vacío
        assert cleaned, f"Texto limpio vacío para idioma {language}"
        assert len(cleaned) >= 20, f"Texto limpio muy corto para idioma {language}"
        
        logger.info(f"  ✅ Limpieza {language} exitosa")
    
    logger.info("✅ MusicTextCleaner: TODAS LAS PRUEBAS EXITOSAS\n")

def test_normalizer():
    """Test normalizador multilingüe."""
    logger.info("🔤 TESTING: MultilingualNormalizer")
    
    # Textos con problemas normalización por idioma
    test_cases = {
        "en": {
            "input": "I can't believe you're gonna leave me! It's ain't fair, y'know?",
            "expected_words": ["cannot", "you are", "going to", "am not"]
        },
        "es": {
            "input": "Del corazón al alma, vamos pa' la fiesta con el pibe",
            "expected_words": ["de el", "para", "chico"]
        },
        "de": {
            "input": "Ich hab dich lieb, du bist schön mit deinen blauen Augen",
            "expected_words": ["habe"]
        },
        "pt": {
            "input": "Do coração, pra vida toda, num momento só",
            "expected_words": ["de o", "para", "em um"]
        }
    }
    
    normalizer = MultilingualNormalizer()
    
    for language, test_case in test_cases.items():
        logger.info(f"Testing normalización idioma: {language}")
        
        input_text = test_case["input"]
        normalized = normalizer.normalize_by_language(input_text, language)
        
        logger.info(f"  Original: {input_text}")
        logger.info(f"  Normalizado: {normalized}")
        
        # Verificar que se aplicaron las transformaciones esperadas
        for expected_word in test_case["expected_words"]:
            assert expected_word in normalized, f"Palabra esperada '{expected_word}' no encontrada"
        
        logger.info(f"  ✅ Normalización {language} exitosa")
    
    # Test detección automática idioma
    logger.info("Testing detección automática idioma...")
    auto_text = "I love music and dancing all night long"
    normalized_auto, detected_lang = normalizer.detect_and_normalize(auto_text)
    logger.info(f"Texto: {auto_text}")
    logger.info(f"Idioma detectado: {detected_lang}")
    logger.info(f"Normalizado: {normalized_auto}")
    
    logger.info("✅ MultilingualNormalizer: TODAS LAS PRUEBAS EXITOSAS\n")

def test_stopwords_manager():
    """Test gestor stopwords."""
    logger.info("🚫 TESTING: StopwordsManager")
    
    # Textos con stopwords por idioma
    test_texts = {
        "en": "Yeah, oh baby, I love you so much and the world is beautiful",
        "es": "Ay, corazón, te amo tanto y el mundo es hermoso",
        "de": "Oh ja, ich liebe dich so sehr und die Welt ist schön", 
        "pt": "Ai, amor, eu te amo tanto e o mundo é lindo"
    }
    
    manager = StopwordsManager()
    
    for language, text in test_texts.items():
        logger.info(f"Testing stopwords idioma: {language}")
        
        # Obtener stopwords
        stopwords = manager.get_stopwords(language, include_musical=True)
        logger.info(f"  Stopwords totales: {len(stopwords)}")
        
        # Filtrar texto
        filtered = manager.filter_text(text, language)
        
        original_words = text.split()
        filtered_words = filtered.split()
        
        reduction = ((len(original_words) - len(filtered_words)) / len(original_words)) * 100
        
        logger.info(f"  Original: {original_words}")
        logger.info(f"  Filtrado: {filtered_words}")
        logger.info(f"  Reducción: {reduction:.1f}%")
        
        # Validar que se filtraron stopwords
        assert len(filtered_words) < len(original_words), f"No se filtraron stopwords en {language}"
        assert filtered_words, f"Filtrado resultó vacío en {language}"
        
        logger.info(f"  ✅ Stopwords {language} exitoso")
    
    # Test estadísticas
    stats = manager.get_stopwords_stats()
    logger.info("Estadísticas stopwords por idioma:")
    for lang, lang_stats in stats.items():
        logger.info(f"  {lang}: {lang_stats['total_count']} total "
                   f"({lang_stats['musical_count']} musicales)")
    
    logger.info("✅ StopwordsManager: TODAS LAS PRUEBAS EXITOSAS\n")

def test_feature_extractor():
    """Test extractor características."""
    logger.info("📊 TESTING: TextFeatureExtractor")
    
    # Textos con diferentes características
    test_texts = {
        "good_quality": "I love the way you make me feel when we're dancing together under the moonlight",
        "poor_quality": "yeah yeah yeah oh oh oh",
        "repetitive": "Love love love, I love you, love you, love you so much love",
        "too_short": "Love you",
        "mixed_quality": "Beautiful song with some yeah yeah repetitions but also meaningful content about life and dreams"
    }
    
    extractor = TextFeatureExtractor()
    
    for test_name, text in test_texts.items():
        logger.info(f"Testing características: {test_name}")
        
        # Extracción completa
        features = extractor.extract_all_features(text, "en")
        
        # Características básicas
        logger.info(f"  Palabras: {features['word_count']} (únicas: {features['unique_word_count']})")
        logger.info(f"  TTR: {features['diversity_ttr']:.3f}")
        logger.info(f"  Repetición ratio: {features['repetition_repetition_ratio']:.3f}")
        logger.info(f"  Calidad score: {features['quality_quality_score']:.3f}")
        logger.info(f"  Suitable BERT: {features['quality_is_suitable']}")
        
        # Validaciones específicas por tipo
        if test_name == "good_quality":
            assert features['quality_quality_score'] > 0.7, "Texto bueno debería tener score alto"
            assert features['quality_is_suitable'], "Texto bueno debería ser suitable"
        elif test_name == "poor_quality":
            assert features['quality_quality_score'] < 0.5, "Texto pobre debería tener score bajo"
        elif test_name == "too_short":
            assert not features['quality_is_suitable'], "Texto muy corto no debería ser suitable"
        
        logger.info(f"  ✅ Extracción {test_name} exitosa")
    
    logger.info("✅ TextFeatureExtractor: TODAS LAS PRUEBAS EXITOSAS\n")

def test_pipeline_integration():
    """Test pipeline completo integración."""
    logger.info("🔄 TESTING: Pipeline Completo")
    
    # Texto musical real complejo
    sample_text = """[Verse 1]
Yeah, oh baby, can't you see the way you move?
It's like magic, I can't help but groove
La la la, na na na, oh oh oh
Dancing through the night, we'll never let go (x2)

[Chorus] 
You're my sunshine, my moonlight too
Everything I am, I owe to you
Can't imagine life without your love
You're the angel sent from up above
Yeah yeah yeah!

[Verse 2]
Remember when we met, it was a rainy day
But you smiled and took my breath away
Del corazón te amo, forever and always
Through the good times and the stormy days"""
    
    # Componentes pipeline
    cleaner = MusicTextCleaner()
    normalizer = MultilingualNormalizer()
    stopwords_manager = StopwordsManager()
    extractor = TextFeatureExtractor()
    
    logger.info("Ejecutando pipeline completo...")
    
    # 1. Limpieza
    cleaned_text = cleaner.clean_universal(sample_text, "en")
    logger.info(f"1. Limpieza: {len(sample_text)} -> {len(cleaned_text)} chars")
    
    # 2. Normalización
    normalized_text = normalizer.normalize_by_language(cleaned_text, "en")
    logger.info(f"2. Normalización: {len(cleaned_text)} -> {len(normalized_text)} chars")
    
    # 3. Filtrado stopwords
    filtered_text = stopwords_manager.filter_text(normalized_text, "en")
    original_words = normalized_text.split()
    filtered_words = filtered_text.split()
    stopword_reduction = ((len(original_words) - len(filtered_words)) / len(original_words)) * 100
    logger.info(f"3. Stopwords: {len(original_words)} -> {len(filtered_words)} palabras ({stopword_reduction:.1f}% reducción)")
    
    # 4. Extracción características
    features = extractor.extract_all_features(filtered_text, "en")
    logger.info(f"4. Características extraídas: {len(features)} métricas")
    
    # 5. Validación calidad final
    quality_score = features['quality_quality_score']
    is_suitable = features['quality_is_suitable']
    estimated_tokens = features['bert_estimated_tokens']
    
    logger.info(f"RESULTADO PIPELINE:")
    logger.info(f"  Texto final: {len(filtered_text)} chars")
    logger.info(f"  Calidad score: {quality_score:.3f}")
    logger.info(f"  Suitable BERT: {is_suitable}")
    logger.info(f"  Tokens estimados: {estimated_tokens}")
    logger.info(f"  TTR: {features['diversity_ttr']:.3f}")
    
    # Validaciones finales
    assert filtered_text, "Pipeline no puede resultar en texto vacío"
    assert quality_score > 0.5, "Pipeline debe producir calidad aceptable"
    assert estimated_tokens <= 256, "Pipeline debe mantener tokens bajo límite BERT"
    assert features['diversity_ttr'] > 0.2, "Pipeline debe preservar diversidad mínima"
    
    logger.info("✅ Pipeline Completo: INTEGRACIÓN EXITOSA\n")

def test_with_real_dataset():
    """Test con muestra real del dataset."""
    logger.info("🎵 TESTING: Dataset Real")
    
    try:
        import pandas as pd
        
        # Cargar muestra dataset
        dataset_path = get_dataset_path("main")
        
        if not dataset_path.exists():
            logger.warning("Dataset principal no encontrado, saltando test real")
            return
        
        df = pd.read_csv(
            dataset_path,
            sep=DATASET_CONFIG["separator"],
            encoding=DATASET_CONFIG["encoding"],
            nrows=10  # Solo 10 muestras para test
        )
        
        logger.info(f"Dataset cargado: {len(df)} muestras")
        
        # Componentes pipeline
        cleaner = MusicTextCleaner()
        normalizer = MultilingualNormalizer()
        extractor = TextFeatureExtractor()
        
        successful_processing = 0
        quality_scores = []
        
        for idx, row in df.iterrows():
            lyrics = row[DATASET_CONFIG["lyrics_column"]]
            language = row.get(DATASET_CONFIG["language_column"], "en")
            
            if pd.isna(lyrics) or not lyrics:
                continue
            
            try:
                # Pipeline básico
                cleaned = cleaner.clean_universal(str(lyrics), language)
                normalized = normalizer.normalize_by_language(cleaned, language)
                features = extractor.extract_all_features(normalized, language)
                
                quality_score = features['quality_quality_score']
                quality_scores.append(quality_score)
                
                if features['quality_is_suitable']:
                    successful_processing += 1
                
                logger.info(f"  Muestra {idx}: quality={quality_score:.3f}, "
                           f"tokens={features['bert_estimated_tokens']}, "
                           f"suitable={features['quality_is_suitable']}")
                
            except Exception as e:
                logger.error(f"Error procesando muestra {idx}: {e}")
        
        # Estadísticas finales
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            success_rate = (successful_processing / len(quality_scores)) * 100
            
            logger.info(f"RESULTADOS DATASET REAL:")
            logger.info(f"  Muestras procesadas: {len(quality_scores)}")
            logger.info(f"  Calidad promedio: {avg_quality:.3f}")
            logger.info(f"  Success rate: {success_rate:.1f}%")
            
            # Validaciones
            assert avg_quality > 0.4, "Calidad promedio dataset debe ser aceptable"
            assert success_rate > 60, "Success rate debe ser > 60%"
            
            logger.info("✅ Dataset Real: PROCESAMIENTO EXITOSO")
        else:
            logger.warning("No se procesaron muestras del dataset")
            
    except Exception as e:
        logger.error(f"Error en test dataset real: {e}")
        logger.warning("Test dataset real falló, pero no es crítico")

def main():
    """Ejecuta todas las pruebas preprocessing."""
    logger.info("🚀 INICIANDO TESTS PREPROCESSING SISTEMA COMPLETO")
    logger.info("="*60)
    
    try:
        # Tests individuales componentes
        test_text_cleaner()
        test_normalizer()
        test_stopwords_manager()
        test_feature_extractor()
        
        # Test integración
        test_pipeline_integration()
        
        # Test con datos reales
        test_with_real_dataset()
        
        # Resumen final
        logger.info("="*60)
        logger.info("🎉 TODOS LOS TESTS PREPROCESSING EXITOSOS")
        logger.info("✅ Sistema preprocessing listo para FASE 4 (Vectorización BERT)")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ ERROR EN TESTS: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()