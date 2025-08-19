"""
Módulo de vectorización BERT para clustering semántico de letras musicales.

Este módulo implementa la vectorización de textos musicales pre-procesados
usando modelos BERT multilingües optimizados para CPU.

Componentes principales:
- BertVectorizer: Generación de embeddings semánticos
- CacheManager: Sistema de cache multinivel
- BatchProcessor: Procesamiento optimizado en lotes

Autor: Sistema de Clustering Semántico Musical
Fase: 4 - Vectorización BERT y Cache Inteligente
"""

__version__ = "1.0.0"
__author__ = "Sistema de Clustering Musical"

# Imports principales
from .bert_vectorizer import BertVectorizer, vectorize_texts
from .batch_processor import BatchProcessor, process_dataset_batch

__all__ = [
    "BertVectorizer",
    "BatchProcessor",
    "vectorize_texts",
    "process_dataset_batch"
]