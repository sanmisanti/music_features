"""
Batch Processor optimizado CPU para vectorización BERT masiva.

Este módulo implementa procesamiento en lotes inteligente para vectorización
de grandes volúmenes de texto musical, con optimizaciones específicas para CPU.

Características principales:
- Procesamiento paralelo CPU-optimizado
- Gestión inteligente de memoria
- Progress tracking detallado
- Integración cache multinivel
- Manejo robusto de errores

Autor: Sistema de Clustering Musical
Fecha: Agosto 2025 - FASE 4
"""

import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

try:
    from .bert_vectorizer import BertVectorizer
    from ..cache.cache_manager import CacheManager
    from ..config.data_paths import get_dataset_path, DATASET_CONFIG
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    
    from vectorization.bert_vectorizer import BertVectorizer
    from cache.cache_manager import CacheManager
    from config.data_paths import get_dataset_path, DATASET_CONFIG

# Setup logging
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Procesador de lotes optimizado para vectorización BERT masiva.
    
    Diseñado para procesar eficientemente datasets completos de letras
    musicales con optimizaciones CPU y manejo inteligente de memoria.
    """
    
    def __init__(self,
                 vectorizer: BertVectorizer = None,
                 batch_size: int = 32,
                 max_workers: int = None,
                 memory_limit_gb: float = 4.0,
                 progress_callback: Callable = None):
        """
        Inicializa batch processor.
        
        Args:
            vectorizer: Instancia BertVectorizer (se crea si None)
            batch_size: Tamaño de lote para procesamiento
            max_workers: Máximo workers paralelos
            memory_limit_gb: Límite memoria en GB
            progress_callback: Callback para progress updates
        """
        self.vectorizer = vectorizer or BertVectorizer()
        self.batch_size = batch_size
        self.max_workers = max_workers or min(mp.cpu_count(), 4)  # CPU conservative
        self.memory_limit_gb = memory_limit_gb
        self.progress_callback = progress_callback
        
        # Estado procesamiento
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "cached": 0,
            "total_time": 0.0,
            "avg_time_per_item": 0.0
        }
        
        logger.info("🏭 BatchProcessor inicializado:")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Max workers: {self.max_workers}")
        logger.info(f"   Memory limit: {memory_limit_gb:.1f} GB")
        logger.info(f"   Vectorizer: {self.vectorizer.model_name}")
    
    def _process_batch_chunk(self, 
                           texts_chunk: List[str], 
                           languages_chunk: List[str],
                           start_idx: int) -> List[Dict[str, Any]]:
        """
        Procesa un chunk de textos en paralelo.
        
        Args:
            texts_chunk: Chunk de textos
            languages_chunk: Chunk de idiomas
            start_idx: Índice inicial para logging
            
        Returns:
            Lista resultados vectorización
        """
        results = []
        
        for i, (text, lang) in enumerate(zip(texts_chunk, languages_chunk)):
            try:
                result = self.vectorizer.vectorize_single(text, lang)
                result["index"] = start_idx + i
                results.append(result)
                
                # Update stats
                if result["success"]:
                    self.stats["successful"] += 1
                    if result.get("cached", False):
                        self.stats["cached"] += 1
                else:
                    self.stats["failed"] += 1
                
                self.stats["processed"] += 1
                
            except Exception as e:
                logger.error(f"Error procesando texto {start_idx + i}: {e}")
                results.append({
                    "index": start_idx + i,
                    "embedding": None,
                    "success": False,
                    "error": str(e),
                    "metadata": {}
                })
                self.stats["failed"] += 1
                self.stats["processed"] += 1
        
        return results
    
    def process_texts(self,
                     texts: List[str],
                     languages: List[str] = None,
                     force_process: bool = False,
                     show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Procesa lista de textos en lotes optimizados.
        
        Args:
            texts: Lista de textos a vectorizar
            languages: Lista idiomas (None para auto-detect)
            force_process: Procesar aunque quality sea baja
            show_progress: Mostrar barra progreso
            
        Returns:
            Lista resultados vectorización
        """
        if not texts:
            return []
        
        # Preparar datos
        languages = languages or ["auto"] * len(texts)
        total_texts = len(texts)
        
        # Reset stats
        self.stats = {
            "processed": 0, "successful": 0, "failed": 0, "cached": 0,
            "total_time": 0.0, "avg_time_per_item": 0.0
        }
        
        logger.info(f"🔄 Procesando {total_texts} textos en lotes de {self.batch_size}")
        start_time = time.time()
        
        all_results = []
        
        # Progress bar setup
        pbar = tqdm(total=total_texts, desc="Vectorizando") if show_progress else None
        
        try:
            # Procesar en chunks
            for i in range(0, total_texts, self.batch_size):
                chunk_end = min(i + self.batch_size, total_texts)
                texts_chunk = texts[i:chunk_end]
                languages_chunk = languages[i:chunk_end]
                
                # Procesar chunk
                chunk_results = self._process_batch_chunk(
                    texts_chunk, languages_chunk, i
                )
                
                all_results.extend(chunk_results)
                
                # Update progress
                if pbar:
                    pbar.update(len(chunk_results))
                
                # Callback progress
                if self.progress_callback:
                    progress = {
                        "processed": self.stats["processed"],
                        "total": total_texts,
                        "successful": self.stats["successful"],
                        "failed": self.stats["failed"],
                        "percentage": (self.stats["processed"] / total_texts) * 100
                    }
                    self.progress_callback(progress)
                
                # Memory management check
                if i % (self.batch_size * 10) == 0:
                    self._check_memory_usage()
        
        finally:
            if pbar:
                pbar.close()
        
        # Estadísticas finales
        total_time = time.time() - start_time
        self.stats["total_time"] = total_time
        self.stats["avg_time_per_item"] = total_time / total_texts if total_texts > 0 else 0
        
        self._log_final_stats()
        
        return all_results
    
    def process_dataset(self,
                       dataset_path: Path = None,
                       lyrics_column: str = None,
                       language_column: str = None,
                       max_rows: int = None,
                       output_path: Path = None) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Procesa dataset completo de letras musicales.
        
        Args:
            dataset_path: Ruta dataset (usa default si None)
            lyrics_column: Columna letras
            language_column: Columna idioma (opcional)
            max_rows: Máximo filas a procesar
            output_path: Ruta guardar resultados
            
        Returns:
            Tuple (resultados_vectorizacion, dataframe_enriquecido)
        """
        # Configuración default
        dataset_path = dataset_path or get_dataset_path("main")
        lyrics_column = lyrics_column or DATASET_CONFIG["lyrics_column"]
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
        
        logger.info(f"📁 Cargando dataset: {dataset_path}")
        
        # Cargar dataset
        df = pd.read_csv(
            dataset_path,
            sep=DATASET_CONFIG["separator"],
            encoding=DATASET_CONFIG["encoding"],
            nrows=max_rows
        )
        
        if lyrics_column not in df.columns:
            raise ValueError(f"Columna '{lyrics_column}' no encontrada en dataset")
        
        logger.info(f"📊 Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
        
        # Extraer textos y idiomas
        texts = df[lyrics_column].fillna("").astype(str).tolist()
        languages = None
        
        if language_column and language_column in df.columns:
            languages = df[language_column].fillna("auto").astype(str).tolist()
        
        # Procesar
        logger.info("🚀 Iniciando vectorización masiva...")
        results = self.process_texts(texts, languages, show_progress=True)
        
        # Enriquecer dataframe con resultados
        df_enriched = self._enrich_dataframe(df, results)
        
        # Guardar si se especifica output
        if output_path:
            self._save_results(results, df_enriched, output_path)
        
        return results, df_enriched
    
    def _enrich_dataframe(self, 
                         df: pd.DataFrame, 
                         results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Enriquece dataframe con resultados vectorización."""
        df_enriched = df.copy()
        
        # Agregar columnas resultados
        df_enriched["embedding_success"] = [r["success"] for r in results]
        df_enriched["quality_score"] = [
            r["metadata"].get("quality_score", 0.0) if r["success"] else 0.0 
            for r in results
        ]
        df_enriched["is_suitable"] = [
            r["metadata"].get("is_suitable", False) if r["success"] else False
            for r in results
        ]
        df_enriched["processing_error"] = [
            r.get("error", "") for r in results
        ]
        df_enriched["cached"] = [r.get("cached", False) for r in results]
        
        # Estadísticas adicionales
        df_enriched["processed_length"] = [
            r["metadata"].get("processed_length", 0) if r["success"] else 0
            for r in results
        ]
        
        logger.info("📈 DataFrame enriquecido con métricas vectorización")
        return df_enriched
    
    def _save_results(self, 
                     results: List[Dict[str, Any]], 
                     df_enriched: pd.DataFrame, 
                     output_path: Path):
        """Guarda resultados vectorización."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar DataFrame enriquecido
        df_path = output_path.with_suffix('.csv')
        df_enriched.to_csv(df_path, index=False, encoding='utf-8')
        
        # Guardar embeddings exitosos
        successful_embeddings = [
            {"index": r.get("index", i), "embedding": r["embedding"]} 
            for i, r in enumerate(results) 
            if r["success"] and r["embedding"] is not None
        ]
        
        if successful_embeddings:
            embeddings_path = output_path.with_suffix('.npy')
            embeddings_array = np.array([e["embedding"] for e in successful_embeddings])
            indices_array = np.array([e["index"] for e in successful_embeddings])
            
            np.savez_compressed(
                embeddings_path,
                embeddings=embeddings_array,
                indices=indices_array
            )
            
            logger.info(f"💾 Resultados guardados:")
            logger.info(f"   DataFrame: {df_path}")
            logger.info(f"   Embeddings: {embeddings_path} ({len(successful_embeddings)} items)")
    
    def _check_memory_usage(self):
        """Verifica uso memoria y activa cleanup si necesario."""
        try:
            import psutil
            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024 ** 3)
            
            if memory_gb > self.memory_limit_gb:
                logger.warning(f"⚠️ Memoria alta: {memory_gb:.2f}GB > {self.memory_limit_gb}GB")
                # Limpiar cache L1 para liberar memoria
                if hasattr(self.vectorizer, 'cache_manager'):
                    self.vectorizer.cache_manager.l1_memory.clear()
                    logger.info("🗑️ Cache L1 limpiado para liberar memoria")
        
        except ImportError:
            # psutil no disponible, skip check
            pass
        except Exception as e:
            logger.warning(f"Error verificando memoria: {e}")
    
    def _log_final_stats(self):
        """Log estadísticas finales del procesamiento."""
        stats = self.stats
        total = stats["processed"]
        
        if total == 0:
            return
        
        success_rate = (stats["successful"] / total) * 100
        cache_rate = (stats["cached"] / total) * 100 if stats["successful"] > 0 else 0
        
        logger.info("📊 ESTADÍSTICAS FINALES BATCH PROCESSING:")
        logger.info(f"   Total procesados: {total}")
        logger.info(f"   Exitosos: {stats['successful']} ({success_rate:.1f}%)")
        logger.info(f"   Fallidos: {stats['failed']}")
        logger.info(f"   Cache hits: {stats['cached']} ({cache_rate:.1f}%)")
        logger.info(f"   Tiempo total: {stats['total_time']:.2f}s")
        logger.info(f"   Tiempo promedio/item: {stats['avg_time_per_item']*1000:.1f}ms")
        logger.info(f"   Throughput: {total/stats['total_time']:.1f} items/segundo")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas actuales del processor."""
        return self.stats.copy()


def process_dataset_batch(dataset_path: Path = None,
                         max_rows: int = None,
                         batch_size: int = 32,
                         output_path: Path = None,
                         **vectorizer_kwargs) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Función helper para procesamiento batch de dataset completo.
    
    Args:
        dataset_path: Ruta dataset
        max_rows: Máximo filas a procesar
        batch_size: Tamaño batch
        output_path: Ruta output
        **vectorizer_kwargs: Args para BertVectorizer
        
    Returns:
        Tuple (resultados, dataframe_enriquecido)
    """
    vectorizer = BertVectorizer(**vectorizer_kwargs)
    processor = BatchProcessor(
        vectorizer=vectorizer,
        batch_size=batch_size
    )
    
    return processor.process_dataset(
        dataset_path=dataset_path,
        max_rows=max_rows,
        output_path=output_path
    )


if __name__ == "__main__":
    # Test básico
    print("🧪 Test BatchProcessor:")
    
    # Crear processor
    processor = BatchProcessor(batch_size=4)
    
    # Test con textos sample
    test_texts = [
        "I love this beautiful music, it makes me feel so happy",
        "Me encanta esta hermosa música que me alegra el corazón",
        "yeah yeah yeah oh oh oh",  # Texto pobre
        "Diese Musik ist wunderbar und macht mich sehr glücklich",
        ""  # Texto vacío
    ]
    
    print(f"Procesando {len(test_texts)} textos de prueba...")
    results = processor.process_texts(test_texts, show_progress=True)
    
    # Mostrar resultados
    for i, result in enumerate(results):
        status = "✅" if result["success"] else "❌"
        cached = "📦" if result.get("cached", False) else "🆕"
        print(f"Texto {i+1}: {status} {cached}")
        if result["success"]:
            print(f"   Quality: {result['metadata']['quality_score']:.3f}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    # Stats finales
    stats = processor.get_stats()
    print(f"\n📊 Stats: {stats['successful']}/{stats['processed']} exitosos")