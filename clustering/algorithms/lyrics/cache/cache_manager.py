"""
Cache Manager multinivel para optimización BERT vectorization.

Sistema de cache de 3 niveles diseñado para maximizar performance:
- L1: Memoria RAM (LRU cache, acceso inmediato)
- L2: Disco SSD (embeddings persistentes, pickle)
- L3: SQLite DB (metadata, índices, estadísticas)

Características:
- Gestión automática de memoria
- Persistencia entre sesiones
- Estadísticas detalladas
- Limpieza automática por edad/tamaño

Autor: Sistema de Clustering Musical
Fecha: Agosto 2025 - FASE 4
"""

import os
import logging
import sqlite3
import pickle
import hashlib
import time
import json
from typing import Dict, Optional, Any, List, Tuple, Union
from pathlib import Path
from collections import OrderedDict
import numpy as np
from datetime import datetime, timedelta

try:
    from ..config.data_paths import get_models_path
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    from config.data_paths import get_models_path

# Setup logging
logger = logging.getLogger(__name__)


class MemoryCache:
    """Cache L1: Memoria RAM con LRU eviction."""
    
    def __init__(self, max_size: int = 1000):
        """
        Inicializa cache memoria.
        
        Args:
            max_size: Máximo número de embeddings en memoria
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Obtiene embedding desde cache memoria."""
        if key in self.cache:
            # Mover al final (LRU)
            self.cache.move_to_end(key)
            self.stats["hits"] += 1
            return self.cache[key]
        
        self.stats["misses"] += 1
        return None
    
    def put(self, key: str, value: np.ndarray):
        """Almacena embedding en cache memoria."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            
        # LRU eviction si necesario
        while len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1
    
    def size(self) -> int:
        """Tamaño actual del cache."""
        return len(self.cache)
    
    def clear(self):
        """Limpia cache memoria."""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}


class DiskCache:
    """Cache L2: Disco persistente con gestión automática."""
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 2.0):
        """
        Inicializa cache disco.
        
        Args:
            cache_dir: Directorio para cache
            max_size_gb: Tamaño máximo en GB
        """
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"hits": 0, "misses": 0, "saves": 0, "cleanups": 0}
    
    def _get_file_path(self, key: str) -> Path:
        """Obtiene ruta archivo para clave."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Obtiene embedding desde disco."""
        file_path = self._get_file_path(key)
        
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    embedding = pickle.load(f)
                # Actualizar timestamp acceso
                os.utime(file_path, None)
                self.stats["hits"] += 1
                return embedding
            except Exception as e:
                logger.warning(f"Error leyendo cache disco {key}: {e}")
                # Eliminar archivo corrupto
                try:
                    file_path.unlink()
                except:
                    pass
        
        self.stats["misses"] += 1
        return None
    
    def put(self, key: str, value: np.ndarray):
        """Almacena embedding en disco."""
        file_path = self._get_file_path(key)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            self.stats["saves"] += 1
            
            # Verificar tamaño cache y limpiar si necesario
            self._cleanup_if_needed()
            
        except Exception as e:
            logger.warning(f"Error guardando cache disco {key}: {e}")
    
    def _cleanup_if_needed(self):
        """Limpia cache disco si excede tamaño máximo."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        
        if total_size > self.max_size_bytes:
            self._cleanup_old_files()
            self.stats["cleanups"] += 1
    
    def _cleanup_old_files(self):
        """Elimina archivos más antiguos para liberar espacio."""
        files = list(self.cache_dir.glob("*.pkl"))
        
        # Ordenar por fecha acceso (más antiguos primero)
        files.sort(key=lambda f: f.stat().st_atime)
        
        # Eliminar 25% de archivos más antiguos
        files_to_remove = files[:len(files) // 4]
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Error eliminando archivo cache: {e}")
        
        logger.info(f"🗑️ Cache cleanup: eliminados {len(files_to_remove)} archivos")
    
    def size_info(self) -> Dict[str, Union[int, float]]:
        """Información tamaño cache disco."""
        files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "file_count": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_gb": self.max_size_bytes / (1024 * 1024 * 1024)
        }
    
    def clear(self):
        """Limpia completamente cache disco."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True)
        self.stats = {"hits": 0, "misses": 0, "saves": 0, "cleanups": 0}


class DatabaseCache:
    """Cache L3: SQLite para metadata e índices."""
    
    def __init__(self, cache_dir: Path):
        """
        Inicializa cache database.
        
        Args:
            cache_dir: Directorio para database
        """
        self.db_path = cache_dir / "cache_metadata.db"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Inicializa schema database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    text_hash TEXT NOT NULL,
                    language TEXT,
                    model_name TEXT,
                    embedding_dim INTEGER,
                    quality_score REAL,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_hash ON embeddings(text_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed_at ON embeddings(accessed_at)
            """)
    
    def record_embedding(self, 
                        cache_key: str,
                        text_hash: str,
                        language: str,
                        model_name: str,
                        embedding_dim: int,
                        quality_score: float,
                        processing_time: float):
        """Registra metadata de embedding."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (cache_key, text_hash, language, model_name, embedding_dim, 
                     quality_score, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (cache_key, text_hash, language, model_name, 
                      embedding_dim, quality_score, processing_time))
        except Exception as e:
            logger.warning(f"Error registrando metadata: {e}")
    
    def record_access(self, cache_key: str):
        """Registra acceso a embedding."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE embeddings 
                    SET accessed_at = CURRENT_TIMESTAMP,
                        access_count = access_count + 1
                    WHERE cache_key = ?
                """, (cache_key,))
        except Exception as e:
            logger.warning(f"Error registrando acceso: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Estadísticas generales
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_embeddings,
                        AVG(quality_score) as avg_quality,
                        AVG(processing_time) as avg_processing_time,
                        SUM(access_count) as total_accesses
                    FROM embeddings
                """)
                
                general_stats = dict(zip([d[0] for d in cursor.description], cursor.fetchone()))
                
                # Estadísticas por idioma
                cursor = conn.execute("""
                    SELECT language, COUNT(*) as count
                    FROM embeddings 
                    GROUP BY language
                    ORDER BY count DESC
                """)
                
                language_stats = dict(cursor.fetchall())
                
                # Top embeddings más accedidos
                cursor = conn.execute("""
                    SELECT cache_key, access_count, quality_score
                    FROM embeddings 
                    ORDER BY access_count DESC 
                    LIMIT 10
                """)
                
                top_accessed = cursor.fetchall()
                
                return {
                    "general": general_stats,
                    "by_language": language_stats,
                    "top_accessed": top_accessed
                }
                
        except Exception as e:
            logger.warning(f"Error obteniendo stats database: {e}")
            return {}
    
    def cleanup_old_entries(self, days: int = 30):
        """Limpia entradas antiguas de database."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM embeddings 
                    WHERE accessed_at < ?
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                
            logger.info(f"🗑️ Database cleanup: eliminadas {deleted_count} entradas antiguas")
            return deleted_count
            
        except Exception as e:
            logger.warning(f"Error limpiando database: {e}")
            return 0

    def close_database(self):
        """Cierra explícitamente todas las conexiones database."""
        try:
            # Forzar cierre de conexiones SQLite
            import gc
            gc.collect()  # Force garbage collection
            
            # Intentar conexión explícita para cerrar
            if self.db_path.exists():
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("PRAGMA optimize")
                    conn.close()
            
            logger.debug("Database connections cerradas")
            
        except Exception as e:
            logger.warning(f"Error cerrando database: {e}")

    def safe_delete_database(self):
        """Elimina database de forma segura (Windows-friendly)."""
        try:
            if not self.db_path.exists():
                return True
            
            # Cerrar conexiones primero
            self.close_database()
            
            # Intentar eliminación múltiples veces (Windows quirk)
            import time
            for attempt in range(3):
                try:
                    self.db_path.unlink()
                    logger.debug(f"Database eliminada en intento {attempt + 1}")
                    return True
                except PermissionError:
                    if attempt < 2:  # No es el último intento
                        time.sleep(0.1)  # Esperar 100ms
                        continue
                    else:
                        # Último intento fallido - renombrar archivo
                        backup_path = self.db_path.with_suffix('.db.bak')
                        try:
                            self.db_path.rename(backup_path)
                            logger.warning(f"Database renombrada a {backup_path} (no se pudo eliminar)")
                            return True
                        except Exception:
                            logger.error("No se pudo eliminar ni renombrar database")
                            return False
            
            return False
            
        except Exception as e:
            logger.warning(f"Error eliminando database: {e}")
            return False


class CacheManager:
    """
    Gestor principal del sistema cache multinivel.
    
    Coordina los 3 niveles de cache para máxima eficiencia:
    L1 (RAM) -> L2 (Disco) -> L3 (Database metadata)
    """
    
    def __init__(self, 
                 cache_dir: Path = None,
                 memory_cache_size: int = 1000,
                 disk_cache_size_gb: float = 2.0):
        """
        Inicializa cache manager.
        
        Args:
            cache_dir: Directorio base cache
            memory_cache_size: Tamaño cache memoria
            disk_cache_size_gb: Tamaño cache disco en GB
        """
        self.cache_dir = cache_dir or (get_models_path() / "bert_cache")
        
        # Inicializar niveles cache
        self.l1_memory = MemoryCache(memory_cache_size)
        self.l2_disk = DiskCache(self.cache_dir / "embeddings", disk_cache_size_gb)
        self.l3_database = DatabaseCache(self.cache_dir)
        
        logger.info("🏗️ CacheManager inicializado:")
        logger.info(f"   Directorio: {self.cache_dir}")
        logger.info(f"   Memoria L1: {memory_cache_size} embeddings")
        logger.info(f"   Disco L2: {disk_cache_size_gb:.1f} GB")
        logger.info(f"   Database L3: SQLite metadata")
    
    def get(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Obtiene embedding del cache multinivel.
        
        Busca secuencialmente: L1 -> L2 -> None
        """
        # L1: Memoria RAM
        embedding = self.l1_memory.get(cache_key)
        if embedding is not None:
            self.l3_database.record_access(cache_key)
            return embedding
        
        # L2: Disco SSD
        embedding = self.l2_disk.get(cache_key)
        if embedding is not None:
            # Promocionar a L1
            self.l1_memory.put(cache_key, embedding)
            self.l3_database.record_access(cache_key)
            return embedding
        
        return None
    
    def put(self, 
            cache_key: str, 
            embedding: np.ndarray,
            metadata: Dict[str, Any] = None):
        """
        Almacena embedding en cache multinivel.
        
        Almacena simultáneamente en L1 y L2, registra metadata en L3.
        """
        # L1: Memoria
        self.l1_memory.put(cache_key, embedding)
        
        # L2: Disco
        self.l2_disk.put(cache_key, embedding)
        
        # L3: Database metadata
        if metadata:
            self.l3_database.record_embedding(
                cache_key=cache_key,
                text_hash=metadata.get("text_hash", ""),
                language=metadata.get("language", "unknown"),
                model_name=metadata.get("model_name", "unknown"),
                embedding_dim=embedding.shape[0] if embedding is not None else 0,
                quality_score=metadata.get("quality_score", 0.0),
                processing_time=metadata.get("processing_time", 0.0)
            )
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Estadísticas completas del sistema cache."""
        return {
            "l1_memory": {
                "size": self.l1_memory.size(),
                "max_size": self.l1_memory.max_size,
                "stats": self.l1_memory.stats
            },
            "l2_disk": {
                "size_info": self.l2_disk.size_info(),
                "stats": self.l2_disk.stats
            },
            "l3_database": self.l3_database.get_stats(),
            "cache_dir": str(self.cache_dir)
        }
    
    def cleanup(self, days: int = 30):
        """Limpieza general del sistema cache."""
        logger.info(f"🧹 Iniciando limpieza cache (>{days} días)")
        
        # L3: Limpiar metadata antigua
        deleted_entries = self.l3_database.cleanup_old_entries(days)
        
        # L2: Trigger cleanup si necesario
        self.l2_disk._cleanup_if_needed()
        
        logger.info(f"✅ Limpieza completada: {deleted_entries} entradas eliminadas")
    
    def clear_all(self):
        """Limpia completamente todos los niveles cache."""
        logger.info("🗑️ Limpiando todos los niveles cache...")
        
        self.l1_memory.clear()
        self.l2_disk.clear()
        
        # Eliminar database de forma segura
        if self.l3_database.safe_delete_database():
            self.l3_database._initialize_db()
            logger.info("✅ Todos los caches limpiados (database recreada)")
        else:
            logger.warning("⚠️ Cache L1/L2 limpiados, L3 database parcialmente limpiada")
            # Intentar recrear database aún si no se pudo eliminar
            try:
                self.l3_database._initialize_db()
            except Exception as e:
                logger.warning(f"Error recreando database: {e}")
        
    def close_all_connections(self):
        """Cierra todas las conexiones para cleanup limpio."""
        try:
            self.l3_database.close_database()
            logger.debug("Todas las conexiones cerradas")
        except Exception as e:
            logger.warning(f"Error cerrando conexiones: {e}")


if __name__ == "__main__":
    # Test básico
    cache_manager = CacheManager()
    
    # Test embedding fake
    test_embedding = np.random.rand(384).astype(np.float32)
    test_key = "test_key_123"
    test_metadata = {
        "text_hash": "abc123",
        "language": "en", 
        "model_name": "test-model",
        "quality_score": 0.85,
        "processing_time": 0.123
    }
    
    print("🧪 Test CacheManager:")
    
    # Almacenar
    cache_manager.put(test_key, test_embedding, test_metadata)
    print("✅ Embedding almacenado")
    
    # Recuperar
    retrieved = cache_manager.get(test_key)
    print(f"✅ Embedding recuperado: {retrieved is not None}")
    
    # Estadísticas
    stats = cache_manager.get_comprehensive_stats()
    print(f"✅ Stats L1: {stats['l1_memory']['size']} items")
    print(f"✅ Stats L2: {stats['l2_disk']['size_info']['file_count']} files")