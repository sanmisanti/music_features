#!/usr/bin/env python3
"""
Vectorización Completa Dataset - Sistema Clustering Semántico

Script para vectorizar completamente el dataset optimizado de 10,000 canciones
usando embeddings BERT 384D y crear índice de similitud para recomendaciones.

Genera vectores persistentes para integración con sistema musical híbrido.

Autor: Sistema de Clustering Musical  
Fecha: Agosto 2025 - Vectorización Completa
"""

import sys
import logging
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging compatible Windows (sin emojis)
import sys

class WindowsCompatibleStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            # Remover emojis del mensaje
            if hasattr(record, 'msg'):
                record.msg = str(record.msg).encode('ascii', 'ignore').decode('ascii')
            super().emit(record)
        except UnicodeEncodeError:
            # Fallback: logging básico sin caracteres especiales
            print(f"{record.levelname}: {str(record.msg).encode('ascii', 'ignore').decode('ascii')}")

logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s:%(name)s: %(message)s',
    handlers=[
        WindowsCompatibleStreamHandler(),
        logging.FileHandler(f'complete_vectorization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Importar componentes FASE 5
try:
    from vectorization.bert_vectorizer import BertVectorizer
    from vectorization.batch_processor import BatchProcessor
    from clustering.semantic_kmeans import SemanticKMeans
    from evaluation.cluster_evaluator import ClusterEvaluator
    
    IMPORTS_SUCCESS = True
    logger.info("Componentes FASE 5 importados exitosamente")
except ImportError as e:
    logger.error(f"Error importando componentes FASE 5: {e}")
    IMPORTS_SUCCESS = False


class CompleteVectorizer:
    """
    Vectorizador completo para dataset musical.
    
    Procesa el dataset completo de canciones y genera embeddings persistentes
    con índice de similitud optimizado para recomendaciones híbridas.
    """
    
    def __init__(self, 
                 dataset_path: str = "data/final_data/picked_data_optimal.csv",
                 output_dir: str = "vectorization_complete_output"):
        """
        Inicializa vectorizador completo.
        
        Args:
            dataset_path: Ruta al dataset optimizado
            output_dir: Directorio para outputs
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp para versionado
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Componentes sistema
        self.vectorizer = None
        self.batch_processor = None
        self.evaluator = ClusterEvaluator()  # Restaurar inicialización
        
        # Estado procesamiento
        self.processing_stats = {
            "total_songs": 0,
            "valid_lyrics": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "cache_hits": 0,
            "processing_time": 0.0,
            "embeddings_size_mb": 0.0
        }
        
        # Resultados
        self.embeddings = None
        self.track_ids = None
        self.similarity_index = None
        
        logger.info("CompleteVectorizer inicializado:")
        logger.info(f"   Dataset: {self.dataset_path}")
        logger.info(f"   Output: {self.output_dir}")
        logger.info(f"   Timestamp: {self.timestamp}")
    
    def run_complete_vectorization(self, test_mode: bool = False) -> Dict[str, Any]:
        """
        Ejecuta vectorización completa del dataset.
        
        Args:
            test_mode: Si True, procesa solo 100 canciones para testing
            
        Returns:
            Dict con resultados completos
        """
        if not IMPORTS_SUCCESS:
            logger.error("Componentes no disponibles - vectorización cancelada")
            return {"success": False, "error": "Import failure"}
        
        logger.info("INICIANDO VECTORIZACIÓN COMPLETA")
        logger.info("=" * 70)
        
        total_start = time.time()
        
        try:
            # FASE 1: Preparación y validación
            logger.info("FASE 1: Preparación y validación")
            df = self._load_and_validate_dataset()
            
            if test_mode:
                df = df.head(100)
                logger.info(f"MODO TEST: Procesando solo {len(df)} canciones")
            
            # FASE 2: Vectorización masiva
            logger.info("FASE 2: Vectorización masiva")
            self._extract_all_embeddings(df)
            
            # FASE 3: Indexación y validación
            logger.info("FASE 3: Indexación y validación")
            self._create_similarity_index()
            self._validate_embeddings()
            
            # FASE 4: Persistencia y optimización
            logger.info("FASE 4: Persistencia y optimización")
            self._save_complete_results()
            
            # Estadísticas finales
            total_time = time.time() - total_start
            self.processing_stats["total_time"] = total_time
            
            final_results = self._generate_final_report()
            
            logger.info("VECTORIZACIÓN COMPLETA EXITOSA")
            logger.info(f"Tiempo total: {total_time:.2f}s")
            logger.info(f"Embeddings generados: {self.processing_stats['successful']}")
            logger.info(f"Archivo principal: embeddings_complete_{self.timestamp}.npy")
            
            return {"success": True, "results": final_results}
            
        except Exception as e:
            logger.error(f"Error en vectorización completa: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_and_validate_dataset(self) -> pd.DataFrame:
        """Carga y valida dataset completo."""
        logger.info("Cargando dataset optimizado...")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {self.dataset_path}")
        
        # Cargar dataset
        df = pd.read_csv(self.dataset_path, sep='^', encoding='utf-8')
        self.processing_stats["total_songs"] = len(df)
        
        logger.info(f"Dataset cargado: {len(df)} canciones, {len(df.columns)} columnas")
        
        # Validar estructura
        required_columns = ['track_id', 'lyrics']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columnas requeridas faltantes: {missing_columns}")
        
        # Filtrar letras válidas
        valid_lyrics = df['lyrics'].notna() & (df['lyrics'] != '') & (df['lyrics'] != 'nan')
        df_valid = df[valid_lyrics].reset_index(drop=True)
        
        self.processing_stats["valid_lyrics"] = len(df_valid)
        
        logger.info(f"Canciones con letras válidas: {len(df_valid)}/{len(df)} "
                   f"({len(df_valid)/len(df)*100:.1f}%)")
        
        if len(df_valid) < len(df) * 0.5:
            logger.warning(f"Solo {len(df_valid)/len(df)*100:.1f}% tienen letras válidas")
        
        return df_valid
    
    def _extract_all_embeddings(self, df: pd.DataFrame):
        """Extrae embeddings para todo el dataset."""
        logger.info(f"Extrayendo embeddings BERT para {len(df)} canciones...")
        
        # Inicializar componentes
        self.vectorizer = BertVectorizer(cache_enabled=True)
        self.batch_processor = BatchProcessor(self.vectorizer)
        
        # Preparar datos
        lyrics_list = df['lyrics'].tolist()
        track_ids = df['track_id'].tolist()
        
        # Procesamiento con barra de progreso
        logger.info("Iniciando procesamiento batch...")
        processing_start = time.time()
        
        # Usar batch processor optimizado
        results = self.batch_processor.process_texts(lyrics_list)
        
        processing_time = time.time() - processing_start
        self.processing_stats["processing_time"] = processing_time
        
        # Procesar resultados
        embeddings = []
        valid_track_ids = []
        success_count = 0
        cache_hits = 0
        
        for i, result in enumerate(results):
            if result['success'] and 'embedding' in result:
                embeddings.append(result['embedding'])
                valid_track_ids.append(track_ids[i])
                success_count += 1
                
                if result.get('cached', False):
                    cache_hits += 1
            else:
                # Para fallos, usar embedding cero pero mantener track
                embeddings.append(np.zeros(384))
                valid_track_ids.append(track_ids[i])
                logger.warning(f"Fallo procesando track {track_ids[i]}: {result.get('error', 'Unknown')}")
        
        # Convertir a arrays numpy
        self.embeddings = np.array(embeddings)
        self.track_ids = np.array(valid_track_ids)
        
        # Estadísticas
        self.processing_stats.update({
            "processed": len(results),
            "successful": success_count,
            "failed": len(results) - success_count,
            "cache_hits": cache_hits,
            "embeddings_size_mb": self.embeddings.nbytes / (1024 * 1024)
        })
        
        logger.info(f"Embeddings extraídos: {self.embeddings.shape}")
        logger.info(f"Éxito: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        logger.info(f"Cache hits: {cache_hits}/{len(results)} ({cache_hits/len(results)*100:.1f}%)")
        logger.info(f"Velocidad: {len(results)/processing_time:.2f} letras/segundo")
        logger.info(f"Tamaño embeddings: {self.processing_stats['embeddings_size_mb']:.2f} MB")
    
    def _create_similarity_index(self):
        """Crea índice de similitud optimizado."""
        logger.info("Creando índice de similitud...")
        
        # Normalizar embeddings para similitud coseno
        from sklearn.preprocessing import normalize
        normalized_embeddings = normalize(self.embeddings, norm='l2')
        
        # Crear índice simple usando sklearn NearestNeighbors
        from sklearn.neighbors import NearestNeighbors
        
        # Configurar índice con algoritmo optimizado para cosine
        nbrs = NearestNeighbors(
            n_neighbors=min(50, len(normalized_embeddings)),  # Top 50 o todos si hay menos
            algorithm='brute',  # Brute force soporta cosine, ball_tree no
            metric='cosine'
        )
        
        nbrs.fit(normalized_embeddings)
        
        self.similarity_index = {
            'model': nbrs,
            'embeddings': normalized_embeddings,
            'track_ids': self.track_ids,
            'index_info': {
                'n_samples': len(normalized_embeddings),
                'n_features': normalized_embeddings.shape[1],
                'algorithm': 'brute',
                'metric': 'cosine',
                'max_neighbors': min(50, len(normalized_embeddings))
            }
        }
        
        logger.info(f"Índice de similitud creado: {len(normalized_embeddings)} canciones")
        logger.info(f"Dimensionalidad: {normalized_embeddings.shape[1]}D")
        logger.info(f"Algoritmo: brute force con métrica coseno")
    
    def _validate_embeddings(self):
        """Valida integridad de embeddings generados."""
        logger.info("Validando integridad embeddings...")
        
        # Validaciones básicas
        assert self.embeddings is not None, "Embeddings no generados"
        assert len(self.embeddings) == len(self.track_ids), "Mismatch embeddings-tracks"
        assert self.embeddings.shape[1] == 384, f"Dimensión incorrecta: {self.embeddings.shape[1]}"
        
        # Estadísticas embeddings
        non_zero_embeddings = np.sum(np.any(self.embeddings != 0, axis=1))
        zero_embeddings = len(self.embeddings) - non_zero_embeddings
        
        # Variabilidad
        embedding_means = np.mean(self.embeddings, axis=1)
        embedding_stds = np.std(self.embeddings, axis=1)
        
        logger.info(f"Embeddings válidos (no-cero): {non_zero_embeddings}/{len(self.embeddings)}")
        logger.info(f"Embeddings fallidos (cero): {zero_embeddings}")
        logger.info(f"Variabilidad promedio: {np.mean(embedding_stds):.4f}")
        logger.info(f"Rango valores: [{np.min(self.embeddings):.4f}, {np.max(self.embeddings):.4f}]")
        
        # Test índice similitud
        if self.similarity_index:
            # Test búsqueda rápida
            test_embedding = self.embeddings[0:1]  # Primera canción
            distances, indices = self.similarity_index['model'].kneighbors(test_embedding, n_neighbors=5)
            
            logger.info(f"Test similitud: encontradas {len(indices[0])} canciones similares")
            logger.info(f"Distancias test: {distances[0][:3]}")
            
        logger.info("Validación completada exitosamente")
    
    def _save_complete_results(self):
        """Guarda todos los resultados persistentes."""
        logger.info("Guardando resultados completos...")
        
        # 1. Embeddings principales
        embeddings_file = self.output_dir / f"embeddings_complete_{self.timestamp}.npy"
        np.save(embeddings_file, self.embeddings)
        logger.info(f"Embeddings guardados: {embeddings_file}")
        
        # 2. Track IDs
        track_ids_file = self.output_dir / f"track_ids_complete_{self.timestamp}.npy"
        np.save(track_ids_file, self.track_ids)
        logger.info(f"Track IDs guardados: {track_ids_file}")
        
        # 3. Índice de similitud
        if self.similarity_index:
            similarity_file = self.output_dir / f"similarity_index_{self.timestamp}.pkl"
            with open(similarity_file, 'wb') as f:
                pickle.dump(self.similarity_index, f)
            logger.info(f"Índice similitud guardado: {similarity_file}")
        
        # 4. Metadatos y estadísticas
        metadata = {
            "timestamp": self.timestamp,
            "dataset_info": {
                "dataset_path": str(self.dataset_path),
                "total_songs": self.processing_stats["total_songs"],
                "valid_lyrics": self.processing_stats["valid_lyrics"]
            },
            "processing_stats": self.processing_stats,
            "embeddings_info": {
                "shape": list(self.embeddings.shape),
                "dtype": str(self.embeddings.dtype),
                "size_mb": self.processing_stats["embeddings_size_mb"],
                "non_zero_count": int(np.sum(np.any(self.embeddings != 0, axis=1)))
            },
            "similarity_index_info": self.similarity_index['index_info'] if self.similarity_index else None,
            "files_generated": {
                "embeddings": f"embeddings_complete_{self.timestamp}.npy",
                "track_ids": f"track_ids_complete_{self.timestamp}.npy", 
                "similarity_index": f"similarity_index_{self.timestamp}.pkl",
                "metadata": f"vectorization_metadata_{self.timestamp}.json"
            }
        }
        
        metadata_file = self.output_dir / f"vectorization_metadata_{self.timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadatos guardados: {metadata_file}")
        
        # 5. Script de carga rápida
        loader_script = f'''#!/usr/bin/env python3
"""
Script de carga rápida - Vectorización Completa {self.timestamp}
Generado automáticamente para cargar embeddings y índice de similitud.
"""

import numpy as np
import pickle
from pathlib import Path

def load_complete_vectorization():
    """Carga vectorización completa."""
    base_dir = Path(__file__).parent
    
    # Cargar componentes
    embeddings = np.load(base_dir / "embeddings_complete_{self.timestamp}.npy")
    track_ids = np.load(base_dir / "track_ids_complete_{self.timestamp}.npy")
    
    with open(base_dir / "similarity_index_{self.timestamp}.pkl", 'rb') as f:
        similarity_index = pickle.load(f)
    
    return {{
        "embeddings": embeddings,
        "track_ids": track_ids,
        "similarity_index": similarity_index,
        "timestamp": "{self.timestamp}"
    }}

def find_similar_songs(track_id, n_neighbors=10):
    """Encuentra canciones similares a una dada."""
    data = load_complete_vectorization()
    
    # Encontrar índice del track
    track_idx = np.where(data["track_ids"] == track_id)[0]
    if len(track_idx) == 0:
        return None
    
    # Buscar similares
    query_embedding = data["embeddings"][track_idx[0]:track_idx[0]+1]
    distances, indices = data["similarity_index"]["model"].kneighbors(
        query_embedding, n_neighbors=n_neighbors+1
    )
    
    # Retornar resultados (skip el primero que es la misma canción)
    similar_tracks = data["track_ids"][indices[0][1:]]
    similar_distances = distances[0][1:]
    
    return list(zip(similar_tracks, similar_distances))

if __name__ == "__main__":
    # Demo de uso
    data = load_complete_vectorization()
    print(f"Vectorización cargada: {{len(data['track_ids'])}} canciones")
    print(f"Dimensiones embeddings: {{data['embeddings'].shape}}")
    
    # Test similitud
    sample_track = data["track_ids"][0]
    similar = find_similar_songs(sample_track, 5)
    print(f"\\nCanciones similares a {{sample_track}}:")
    for track, distance in similar:
        print(f"  {{track}}: {{distance:.3f}}")
'''
        
        loader_file = self.output_dir / f"load_vectorization_{self.timestamp}.py"
        with open(loader_file, 'w', encoding='utf-8') as f:
            f.write(loader_script)
        logger.info(f"Script carga guardado: {loader_file}")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Genera reporte final completo."""
        
        # Calcular métricas adicionales
        success_rate = self.processing_stats["successful"] / self.processing_stats["processed"]
        cache_hit_rate = self.processing_stats["cache_hits"] / self.processing_stats["processed"]
        throughput = self.processing_stats["processed"] / self.processing_stats["processing_time"]
        
        final_report = {
            "vectorization_summary": {
                "timestamp": self.timestamp,
                "success": True,
                "total_time": self.processing_stats["total_time"],
                "dataset_songs": self.processing_stats["total_songs"],
                "valid_lyrics": self.processing_stats["valid_lyrics"],
                "embeddings_generated": self.processing_stats["successful"],
                "success_rate": success_rate,
                "cache_hit_rate": cache_hit_rate,
                "throughput_songs_per_second": throughput
            },
            "embeddings_analysis": {
                "shape": list(self.embeddings.shape),
                "size_mb": self.processing_stats["embeddings_size_mb"],
                "non_zero_embeddings": int(np.sum(np.any(self.embeddings != 0, axis=1))),
                "embedding_stats": {
                    "mean_value": float(np.mean(self.embeddings)),
                    "std_value": float(np.std(self.embeddings)),
                    "min_value": float(np.min(self.embeddings)),
                    "max_value": float(np.max(self.embeddings))
                }
            },
            "similarity_index": {
                "algorithm": "brute",
                "metric": "cosine", 
                "n_samples": len(self.track_ids),
                "max_neighbors": min(50, len(self.track_ids)),
                "ready_for_recommendations": True
            },
            "quality_metrics": {
                "embedding_coverage": success_rate,
                "cache_efficiency": cache_hit_rate,
                "processing_speed": throughput,
                "data_integrity": "validated",
                "similarity_index_status": "functional"
            },
            "next_steps": [
                "Integrar con sistema musical existente",
                "Implementar recomendaciones híbridas música + letras",
                "Validar calidad recomendaciones en test set",
                "Optimizar sistema para producción"
            ]
        }
        
        # Guardar reporte final
        report_file = self.output_dir / f"vectorization_final_report_{self.timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        return final_report


def main():
    """Función principal para ejecutar vectorización completa."""
    
    print("VECTORIZACIÓN COMPLETA - Sistema Clustering Semántico")
    print("=" * 70)
    
    # Opción de test
    import sys
    test_mode = "--test" in sys.argv
    
    if test_mode:
        print("MODO TEST: Procesando solo 100 canciones")
    else:
        print("MODO COMPLETO: Procesando dataset completo (~10,000 canciones)")
        confirm = input("¿Continuar? (y/N): ")
        if confirm.lower() != 'y':
            print("Operación cancelada")
            return
    
    # Ejecutar vectorización
    vectorizer = CompleteVectorizer()
    results = vectorizer.run_complete_vectorization(test_mode=test_mode)
    
    if results["success"]:
        print("\nVECTORIZACIÓN COMPLETA EXITOSA")
        summary = results["results"]["vectorization_summary"]
        print(f"Canciones procesadas: {summary['embeddings_generated']}")
        print(f"Tasa éxito: {summary['success_rate']*100:.1f}%")
        print(f"Tiempo total: {summary['total_time']:.2f}s")
        print(f"Velocidad: {summary['throughput_songs_per_second']:.2f} canciones/segundo")
        print(f"\nArchivos generados en: {vectorizer.output_dir}")
    else:
        print(f"\nERROR: {results['error']}")


if __name__ == "__main__":
    main()