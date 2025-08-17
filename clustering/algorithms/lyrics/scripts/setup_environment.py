#!/usr/bin/env python3
"""
Script de setup del environment para clustering semántico de letras.

Este script:
1. Verifica dependencies instaladas
2. Crea directorios necesarios
3. Descarga stopwords y recursos NLTK
4. Valida configuración BERT
5. Inicializa cache system
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any

# Añadir path del módulo
sys.path.append(str(Path(__file__).parent.parent))

from config.data_paths import ensure_directories_exist, LYRICS_MODULE_ROOT
from config.bert_models import get_optimal_config, PRIMARY_MODEL

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Verifica versión Python compatible."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        raise RuntimeError(f"Python {min_version[0]}.{min_version[1]}+ requerido. "
                          f"Versión actual: {current_version[0]}.{current_version[1]}")
    
    logger.info(f"✅ Python {current_version[0]}.{current_version[1]} compatible")

def check_dependencies():
    """Verifica dependencies críticas instaladas."""
    package_mapping = {
        "pandas": "pandas",
        "numpy": "numpy", 
        "scikit-learn": "sklearn",  # Importante: import name diferente
        "torch": "torch",
        "sentence_transformers": "sentence_transformers",
        "transformers": "transformers",
        "nltk": "nltk",
        "gensim": "gensim"
    }
    
    missing_packages = []
    
    for package_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
            logger.info(f"✅ {package_name} disponible")
        except ImportError:
            missing_packages.append(package_name)
            logger.warning(f"❌ {package_name} NO encontrado")
    
    return missing_packages

def install_missing_packages(missing_packages: List[str]):
    """Instala packages faltantes."""
    if not missing_packages:
        logger.info("✅ Todas las dependencies críticas están instaladas")
        return
    
    logger.info(f"📦 Instalando packages faltantes: {missing_packages}")
    
    requirements_file = LYRICS_MODULE_ROOT / "requirements.txt"
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True, text=True)
        
        logger.info("✅ Dependencies instaladas exitosamente")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error instalando dependencies: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise

def setup_nltk_resources():
    """Descarga recursos NLTK necesarios."""
    try:
        import nltk
        
        # Recursos necesarios para el sistema
        nltk_resources = [
            'punkt',           # Tokenización
            'stopwords',       # Stopwords multilingües
            'wordnet',         # WordNet para lemmatización
            'averaged_perceptron_tagger',  # POS tagging
            'vader_lexicon'    # Análisis sentimientos
        ]
        
        logger.info("📚 Descargando recursos NLTK...")
        
        for resource in nltk_resources:
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"✅ NLTK {resource} descargado")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo descargar {resource}: {e}")
        
        logger.info("✅ Recursos NLTK configurados")
        
    except ImportError:
        logger.warning("⚠️ NLTK no disponible, saltando configuración")

def create_stopwords_files():
    """Crea archivos stopwords personalizados por idioma."""
    stopwords_dir = LYRICS_MODULE_ROOT / "data" / "stopwords"
    
    # Stopwords personalizadas para letras musicales
    stopwords_data = {
        "english.txt": [
            # Stopwords estándar inglés (básicas)
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "a", "an", "is", "are", "was", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            # Stopwords musicales específicas
            "yeah", "oh", "ah", "eh", "uh", "mm", "hmm", "hey", "yo", "whoa", "ooh", "aah",
            "la", "da", "na", "ba", "sha", "doo", "woo", "tra", "ho", "ha",
            "chorus", "verse", "bridge", "outro", "intro", "repeat", "x2", "x3", "x4"
        ],
        
        "spanish.txt": [
            # Stopwords básicas español
            "el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo",
            "le", "da", "su", "por", "son", "con", "para", "al", "del", "los", "las",
            "yo", "tu", "él", "ella", "nosotros", "vosotros", "ellos", "ellas",
            "mi", "mis", "ti", "si", "sí", "más", "pero", "muy", "todo", "bien",
            # Stopwords musicales español
            "ay", "eh", "oh", "uy", "ah", "mm", "hey", "ya", "na", "la", "oh",
            "estribillo", "verso", "coro", "final", "intro"
        ],
        
        "german.txt": [
            # Stopwords básicas alemán
            "der", "die", "das", "und", "in", "zu", "den", "von", "mit", "ist", "im", "für",
            "auf", "des", "eine", "ein", "einer", "ich", "du", "er", "sie", "es", "wir", "ihr",
            "mein", "dein", "sein", "ihr", "unser", "euer", "aber", "oder", "wenn", "dann",
            # Stopwords musicales alemán
            "oh", "ja", "eh", "na", "so", "hey", "ah", "mm",
            "refrain", "strophe", "bridge", "outro", "intro"
        ],
        
        "portuguese.txt": [
            # Stopwords básicas portugués
            "o", "a", "de", "que", "e", "do", "da", "em", "um", "uma", "para", "com", "não",
            "eu", "tu", "você", "ele", "ela", "nós", "vós", "eles", "elas",
            "meu", "minha", "seu", "sua", "nosso", "nossa", "mas", "por", "se", "é", "são",
            # Stopwords musicales portugués
            "ai", "eh", "né", "oh", "já", "ah", "mm", "hey", "na", "la",
            "refrão", "verso", "ponte", "final", "intro"
        ]
    }
    
    for filename, words in stopwords_data.items():
        filepath = stopwords_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(f"{word}\n")
        
        logger.info(f"✅ Stopwords {filename} creadas ({len(words)} palabras)")

def validate_bert_model():
    """Valida que el modelo BERT se puede cargar correctamente."""
    try:
        from sentence_transformers import SentenceTransformer
        
        config = get_optimal_config()
        model_name = PRIMARY_MODEL
        
        logger.info(f"🤖 Validando modelo BERT: {model_name}")
        
        # Intentar cargar modelo (solo para validación)
        model = SentenceTransformer(model_name)
        
        # Test básico
        test_text = "This is a test sentence for validation."
        embedding = model.encode([test_text])
        
        expected_dims = config["dimensions"]
        actual_dims = embedding.shape[1]
        
        if actual_dims != expected_dims:
            raise ValueError(f"Dimensiones incorrectas: esperado {expected_dims}, "
                           f"obtenido {actual_dims}")
        
        logger.info(f"✅ Modelo BERT validado correctamente")
        logger.info(f"   - Dimensiones: {actual_dims}")
        logger.info(f"   - Tamaño modelo: ~{config['model_size_mb']}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validando modelo BERT: {e}")
        return False

def initialize_cache_system():
    """Inicializa sistema cache con estructura básica."""
    cache_dir = LYRICS_MODULE_ROOT / "models" / "bert_cache"
    
    # Crear archivo metadata cache
    cache_metadata = {
        "version": "1.0.0",
        "model": PRIMARY_MODEL,
        "initialized": True,
        "max_size_gb": 3.0,
        "compression": True
    }
    
    metadata_file = cache_dir / "cache_metadata.json"
    
    import json
    with open(metadata_file, 'w') as f:
        json.dump(cache_metadata, f, indent=2)
    
    logger.info("✅ Sistema cache inicializado")

def validate_dataset_access():
    """Valida acceso al dataset principal."""
    from config.data_paths import get_dataset_path, DATASET_CONFIG
    
    try:
        dataset_path = get_dataset_path("main")
        
        if not dataset_path.exists():
            logger.warning(f"⚠️ Dataset principal no encontrado: {dataset_path}")
            return False
        
        # Test lectura básica
        import pandas as pd
        
        df = pd.read_csv(
            dataset_path,
            sep=DATASET_CONFIG["separator"],
            encoding=DATASET_CONFIG["encoding"],
            nrows=5  # Solo primeras 5 filas para test
        )
        
        required_columns = [
            DATASET_CONFIG["lyrics_column"],
            DATASET_CONFIG["language_column"]
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"❌ Columnas faltantes en dataset: {missing_columns}")
            return False
        
        logger.info(f"✅ Dataset principal accesible")
        logger.info(f"   - Ruta: {dataset_path}")
        logger.info(f"   - Columnas: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error accediendo dataset: {e}")
        return False

def main():
    """Función principal setup environment."""
    logger.info("🚀 Iniciando setup environment clustering semántico letras")
    
    try:
        # 1. Verificar Python
        check_python_version()
        
        # 2. Crear directorios
        logger.info("📁 Creando estructura directorios...")
        ensure_directories_exist()
        
        # 3. Verificar dependencies
        logger.info("🔍 Verificando dependencies...")
        missing_packages = check_dependencies()
        
        if missing_packages:
            install_missing_packages(missing_packages)
        
        # 4. Setup NLTK
        setup_nltk_resources()
        
        # 5. Crear stopwords
        logger.info("📝 Creando stopwords personalizadas...")
        create_stopwords_files()
        
        # 6. Validar BERT
        logger.info("🤖 Validando modelo BERT...")
        bert_ok = validate_bert_model()
        
        # 7. Inicializar cache
        logger.info("💾 Inicializando sistema cache...")
        initialize_cache_system()
        
        # 8. Validar dataset
        logger.info("📊 Validando acceso dataset...")
        dataset_ok = validate_dataset_access()
        
        # Resumen final
        logger.info("\n" + "="*50)
        logger.info("📋 RESUMEN SETUP ENVIRONMENT")
        logger.info("="*50)
        logger.info(f"✅ Python compatible: SÍ")
        logger.info(f"✅ Dependencies: SÍ")
        logger.info(f"✅ Directorios: SÍ")
        logger.info(f"✅ NLTK resources: SÍ")
        logger.info(f"✅ Stopwords: SÍ")
        logger.info(f"✅ BERT model: {'SÍ' if bert_ok else 'NO'}")
        logger.info(f"✅ Cache system: SÍ")
        logger.info(f"✅ Dataset access: {'SÍ' if dataset_ok else 'NO'}")
        
        if bert_ok and dataset_ok:
            logger.info("\n🎉 SETUP COMPLETADO EXITOSAMENTE")
            logger.info("El sistema está listo para clustering semántico de letras")
        else:
            logger.warning("\n⚠️ SETUP COMPLETADO CON ADVERTENCIAS")
            if not bert_ok:
                logger.warning("- Modelo BERT necesita validación manual")
            if not dataset_ok:
                logger.warning("- Dataset principal necesita verificación")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR EN SETUP: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()