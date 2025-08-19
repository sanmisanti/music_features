"""
Sistema de cache multinivel para clustering semántico de letras.

Este módulo implementa un sistema de cache sofisticado con múltiples niveles
de almacenamiento para optimizar el rendimiento de la vectorización BERT.

Niveles de cache:
- L1 (RAM): Cache en memoria para acceso ultra-rápido
- L2 (Disco): Cache persistente en SSD para embeddings
- L3 (Database): Metadata e índices en SQLite

Autor: Sistema de Clustering Semántico Musical
Fase: 4 - Cache Inteligente Multinivel
"""

__version__ = "1.0.0"
__author__ = "Sistema de Clustering Musical"

from .cache_manager import CacheManager, MemoryCache, DiskCache, DatabaseCache

__all__ = [
    "CacheManager",
    "MemoryCache",
    "DiskCache", 
    "DatabaseCache"
]