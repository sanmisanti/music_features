"""
BERT Vectorizer para clustering semántico de letras musicales.

Este módulo implementa la vectorización de textos musicales usando modelos BERT
multilingües optimizados para CPU, con integración del sistema de preprocessing
desarrollado en FASE 3.

Características principales:
- Modelo: paraphrase-multilingual-MiniLM-L12-v2 (384 dims)
- Soporte: 4 idiomas (EN, ES, DE, PT)
- Optimización: CPU-only, sin dependencia GPU
- Integración: Pipeline preprocessing FASE 3
- Cache: Sistema multinivel para performance

Autor: Sistema de Clustering Semántico Musical
Fecha: Agosto 2025 - FASE 4
"""

import os
import logging
import numpy as np
import hashlib
import pickle
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import time

# Sentence-BERT imports
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Configuración y preprocessing imports
try:
    from ..config.bert_models import get_optimal_config, PRIMARY_MODEL
    from ..config.data_paths import get_models_path
    from ..preprocessing.text_cleaner import MusicTextCleaner
    from ..preprocessing.normalizer import MultilingualNormalizer
    from ..preprocessing.stopwords_manager import StopwordsManager
    from ..preprocessing.feature_extractor import TextFeatureExtractor
except ImportError:
    # Imports absolutos para ejecución desde diferentes directorios
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    
    from config.bert_models import get_optimal_config, PRIMARY_MODEL
    from config.data_paths import get_models_path
    from preprocessing.text_cleaner import MusicTextCleaner
    from preprocessing.normalizer import MultilingualNormalizer
    from preprocessing.stopwords_manager import StopwordsManager
    from preprocessing.feature_extractor import TextFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertVectorizer:
    """
    Vectorizador BERT optimizado para clustering semántico de letras musicales.
    
    Integra completamente el pipeline de preprocessing de FASE 3 con
    vectorización BERT multinivel cache inteligente.
    """
    
    def __init__(self, 
                 model_name: str = None, 
                 cache_enabled: bool = True,
                 max_seq_length: int = 256,
                 device: str = "cpu"):
        """
        Inicializa el vectorizador BERT.
        
        Args:
            model_name: Nombre del modelo BERT (usa PRIMARY_MODEL si None)
            cache_enabled: Habilitar sistema cache
            max_seq_length: Longitud máxima secuencia
            device: Dispositivo ('cpu' por defecto, optimizado)
        """
        self.model_name = model_name or PRIMARY_MODEL
        self.cache_enabled = cache_enabled
        self.max_seq_length = max_seq_length
        self.device = device
        
        # Componentes preprocessing (FASE 3)
        self.text_cleaner = MusicTextCleaner()
        self.normalizer = MultilingualNormalizer()
        self.stopwords_manager = StopwordsManager()
        self.feature_extractor = TextFeatureExtractor()
        
        # Estado interno
        self.model = None
        self.cache_dir = None
        self.cache_stats = {"hits": 0, "misses": 0, "saves": 0}
        
        logger.info(f"🤖 Inicializando BertVectorizer")
        logger.info(f"   Modelo: {self.model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Cache: {'habilitado' if cache_enabled else 'deshabilitado'}")
        
        # Inicialización diferida
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Inicializa sistema de cache."""
        if not self.cache_enabled:
            return
            
        self.cache_dir = get_models_path() / "bert_cache" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Cache inicializado: {self.cache_dir}")
    
    def _load_model(self):
        """Carga modelo BERT con configuración optimizada."""
        if self.model is not None:
            return
            
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers no instalado. "
                "Instalar con: pip install sentence-transformers"
            )
        
        logger.info(f"🔄 Cargando modelo BERT: {self.model_name}")
        start_time = time.time()
        
        try:
            # Configuración optimizada CPU
            config = get_optimal_config()
            
            # Cargar modelo
            self.model = SentenceTransformer(
                self.model_name, 
                device=self.device
            )
            
            # Aplicar configuración optimizada
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_seq_length
            
            load_time = time.time() - start_time
            logger.info(f"✅ Modelo cargado exitosamente en {load_time:.2f}s")
            logger.info(f"   Dimensiones: {self.get_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo BERT: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Obtiene dimensión de los embeddings."""
        if self.model is None:
            # Dimensión conocida del modelo MiniLM-L12-v2
            return 384
        return self.model.get_sentence_embedding_dimension()
    
    def _get_cache_key(self, text: str, language: str = None) -> str:
        """Genera clave única para cache."""
        content = f"{text}|{language or 'auto'}|{self.model_name}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Carga embedding desde cache."""
        if not self.cache_enabled:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                self.cache_stats["hits"] += 1
                return embedding
            except Exception as e:
                logger.warning(f"Error leyendo cache {cache_key}: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Guarda embedding en cache."""
        if not self.cache_enabled:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            self.cache_stats["saves"] += 1
        except Exception as e:
            logger.warning(f"Error guardando cache {cache_key}: {e}")
    
    def preprocess_text(self, text: str, language: str = None) -> Dict[str, Any]:
        """
        Aplica pipeline completo de preprocessing FASE 3.
        
        Args:
            text: Texto a procesar
            language: Idioma (auto-detectado si None)
            
        Returns:
            Dict con texto procesado y metadata
        """
        if not text or not isinstance(text, str):
            return {
                "processed_text": "",
                "is_suitable": False,
                "quality_score": 0.0,
                "issues": ["empty_or_invalid_text"],
                "language": language
            }
        
        # Pipeline FASE 3
        try:
            # 1. Limpieza musical especializada
            cleaned_text = self.text_cleaner.clean_universal(text, language)
            
            # 2. Normalización multilingüe
            if language:
                normalized_text = self.normalizer.normalize_by_language(cleaned_text, language)
            else:
                # Auto-detect y normalizar
                normalized_text = self.normalizer.normalize_auto_detect(cleaned_text)
            
            # 3. Evaluación calidad (sin filtrado stopwords para mantener contexto BERT)
            quality_assessment = self.feature_extractor.assess_text_quality(
                normalized_text, language
            )
            
            return {
                "processed_text": normalized_text,
                "is_suitable": quality_assessment["is_suitable"],
                "quality_score": quality_assessment["quality_score"],
                "issues": quality_assessment.get("issues", []),
                "language": language or "auto",
                "original_length": len(text),
                "processed_length": len(normalized_text)
            }
            
        except Exception as e:
            logger.error(f"Error en preprocessing: {e}")
            return {
                "processed_text": "",
                "is_suitable": False,
                "quality_score": 0.0,
                "issues": [f"preprocessing_error: {e}"],
                "language": language
            }
    
    def vectorize_single(self, 
                        text: str, 
                        language: str = None,
                        force_process: bool = False) -> Dict[str, Any]:
        """
        Vectoriza un texto individual.
        
        Args:
            text: Texto a vectorizar
            language: Idioma del texto
            force_process: Procesar aunque quality score sea bajo
            
        Returns:
            Dict con embedding y metadata
        """
        # Preprocessing
        prep_result = self.preprocess_text(text, language)
        
        if not prep_result["processed_text"]:
            return {
                "embedding": None,
                "success": False,
                "error": "Texto vacío después de preprocessing",
                "metadata": prep_result
            }
        
        if not prep_result["is_suitable"] and not force_process:
            return {
                "embedding": None,
                "success": False,
                "error": f"Texto no suitable (score: {prep_result['quality_score']:.3f})",
                "metadata": prep_result
            }
        
        processed_text = prep_result["processed_text"]
        
        # Verificar cache
        cache_key = self._get_cache_key(processed_text, language)
        cached_embedding = self._load_from_cache(cache_key)
        
        if cached_embedding is not None:
            return {
                "embedding": cached_embedding,
                "success": True,
                "cached": True,
                "metadata": prep_result
            }
        
        # Generar embedding
        try:
            self._load_model()
            
            start_time = time.time()
            embedding = self.model.encode(
                processed_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            vectorization_time = time.time() - start_time
            
            # Guardar en cache
            self._save_to_cache(cache_key, embedding)
            
            return {
                "embedding": embedding,
                "success": True,
                "cached": False,
                "vectorization_time": vectorization_time,
                "metadata": prep_result
            }
            
        except Exception as e:
            logger.error(f"Error en vectorización: {e}")
            return {
                "embedding": None,
                "success": False,
                "error": str(e),
                "metadata": prep_result
            }
    
    def vectorize_batch(self, 
                       texts: List[str], 
                       languages: List[str] = None,
                       batch_size: int = 32,
                       force_process: bool = False) -> List[Dict[str, Any]]:
        """
        Vectoriza múltiples textos en lotes.
        
        Args:
            texts: Lista de textos
            languages: Lista de idiomas (None para auto-detect)
            batch_size: Tamaño de lote para procesamiento
            force_process: Procesar textos aunque quality score sea bajo
            
        Returns:
            Lista de resultados de vectorización
        """
        if not texts:
            return []
        
        languages = languages or [None] * len(texts)
        results = []
        
        logger.info(f"🔄 Vectorizando {len(texts)} textos en lotes de {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_langs = languages[i:i + batch_size]
            
            batch_results = []
            for text, lang in zip(batch_texts, batch_langs):
                result = self.vectorize_single(text, lang, force_process)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Progress logging
            if i % (batch_size * 5) == 0:
                logger.info(f"   Procesados: {min(i + batch_size, len(texts))}/{len(texts)}")
        
        # Estadísticas finales
        successful = sum(1 for r in results if r["success"])
        cached = sum(1 for r in results if r.get("cached", False))
        
        logger.info(f"✅ Vectorización batch completada:")
        logger.info(f"   Exitosos: {successful}/{len(texts)} ({successful/len(texts)*100:.1f}%)")
        logger.info(f"   Cache hits: {cached}/{len(texts)} ({cached/len(texts)*100:.1f}%)")
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Obtiene estadísticas del cache."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "total_requests": total_requests,
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "saves": self.cache_stats["saves"]
        }
    
    def clear_cache(self):
        """Limpia completamente el cache."""
        if not self.cache_enabled or not self.cache_dir:
            return
        
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True)
        
        self.cache_stats = {"hits": 0, "misses": 0, "saves": 0}
        logger.info("🗑️ Cache limpiado completamente")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Información del modelo cargado."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": self.max_seq_length,
            "embedding_dimension": self.get_embedding_dimension(),
            "cache_enabled": self.cache_enabled,
            "model_loaded": self.model is not None
        }


def vectorize_texts(texts: List[str], 
                   languages: List[str] = None,
                   **kwargs) -> List[Dict[str, Any]]:
    """
    Función helper para vectorización rápida de textos.
    
    Args:
        texts: Textos a vectorizar
        languages: Idiomas (opcional)
        **kwargs: Argumentos para BertVectorizer
        
    Returns:
        Lista de resultados vectorización
    """
    vectorizer = BertVectorizer(**kwargs)
    return vectorizer.vectorize_batch(texts, languages)


if __name__ == "__main__":
    # Test básico
    vectorizer = BertVectorizer()
    
    test_texts = [
        "I love this song, it makes me feel so happy and alive",
        "Me encanta esta canción, me hace sentir muy feliz",
        "Ich liebe dieses Lied, es macht mich sehr glücklich"
    ]
    
    print("🧪 Test BertVectorizer:")
    for i, text in enumerate(test_texts):
        result = vectorizer.vectorize_single(text)
        print(f"Texto {i+1}: {'✅' if result['success'] else '❌'}")
        if result["success"]:
            print(f"  Embedding shape: {result['embedding'].shape}")
            print(f"  Quality score: {result['metadata']['quality_score']:.3f}")
        else:
            print(f"  Error: {result['error']}")