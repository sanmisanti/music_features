#!/usr/bin/env python3
"""
Validación Experimental Simple - Sistema Clustering Semántico FASE 5
Versión simplificada sin emojis para compatibilidad Windows.
"""

import sys
import logging
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging sin emojis para Windows
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s:%(name)s: %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Importar componentes FASE 5
try:
    from clustering.semantic_kmeans import SemanticKMeans
    from clustering.hierarchical_clusterer import HierarchicalClusterer
    from evaluation.cluster_evaluator import ClusterEvaluator
    from evaluation.cluster_visualizer import ClusterVisualizer
    from vectorization.bert_vectorizer import BertVectorizer
    from vectorization.batch_processor import BatchProcessor
    
    IMPORTS_SUCCESS = True
    logger.info("Componentes FASE 5 importados exitosamente")
except ImportError as e:
    logger.error(f"Error importando componentes FASE 5: {e}")
    IMPORTS_SUCCESS = False


def run_simple_validation(sample_size: int = 500) -> Dict[str, Any]:
    """
    Ejecuta validación experimental simple.
    
    Args:
        sample_size: Número de canciones a analizar
        
    Returns:
        Dict con resultados
    """
    if not IMPORTS_SUCCESS:
        logger.error("Componentes no disponibles - experimentos cancelados")
        return {"success": False, "error": "Import failure"}
    
    logger.info("INICIANDO VALIDACION EXPERIMENTAL SIMPLE")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. Encontrar dataset disponible
        possible_datasets = [
            r"C:\Users\sanmi\Documents\Proyectos\Tesis\music_features\data\final_data\picked_data_optimal.csv",
            r"C:\Users\sanmi\Documents\Proyectos\Tesis\music_features\data\final_data\picked_data_lyrics.csv",
            r"C:\Users\sanmi\Documents\Proyectos\Tesis\music_features\data\with_lyrics\spotify_songs_fixed.csv"
        ]
        
        dataset_path = None
        separator = '^'
        
        for path in possible_datasets:
            if Path(path).exists():
                dataset_path = path
                logger.info(f"Dataset encontrado: {Path(path).name}")
                break
        
        if not dataset_path:
            return {"success": False, "error": "No se encontró dataset válido"}
        
        # 2. Cargar dataset
        logger.info(f"Cargando dataset...")
        
        if 'spotify_songs_fixed.csv' in dataset_path:
            separator = '@@'
        
        df = pd.read_csv(dataset_path, sep=separator, encoding='utf-8')
        logger.info(f"Dataset cargado: {len(df)} canciones, {len(df.columns)} columnas")
        
        # 3. Validar columna lyrics
        if 'lyrics' not in df.columns:
            return {"success": False, "error": "Columna 'lyrics' no encontrada"}
        
        # 4. Filtrar canciones válidas
        valid_lyrics = df['lyrics'].notna() & (df['lyrics'] != '') & (df['lyrics'] != 'nan')
        df_valid = df[valid_lyrics].reset_index(drop=True)
        logger.info(f"Canciones con letras válidas: {len(df_valid)}")
        
        # 5. Crear muestra
        if sample_size >= len(df_valid):
            df_sample = df_valid.copy()
            logger.info(f"Usando dataset completo: {len(df_sample)} canciones")
        else:
            df_sample = df_valid.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Muestra creada: {len(df_sample)} canciones")
        
        # 6. Extraer embeddings
        logger.info("Extrayendo embeddings BERT...")
        
        vectorizer = BertVectorizer(cache_enabled=True)
        batch_processor = BatchProcessor(vectorizer)
        
        lyrics_list = df_sample['lyrics'].tolist()[:sample_size]  # Limitar por seguridad
        
        processing_start = time.time()
        results = batch_processor.process_texts(lyrics_list)
        processing_time = time.time() - processing_start
        
        # 7. Preparar embeddings
        embeddings = []
        success_count = 0
        
        for result in results:
            if result['success'] and 'embedding' in result:
                embeddings.append(result['embedding'])
                success_count += 1
            else:
                embeddings.append(np.zeros(384))
        
        embeddings_array = np.array(embeddings)
        
        logger.info(f"Embeddings extraidos: {embeddings_array.shape}")
        logger.info(f"Exito: {success_count}/{len(lyrics_list)} ({success_count/len(lyrics_list)*100:.1f}%)")
        logger.info(f"Velocidad: {len(lyrics_list)/processing_time:.2f} letras/segundo")
        
        # 8. Clustering semántico
        logger.info("Ejecutando clustering semantico...")
        
        clusterer = SemanticKMeans(
            n_clusters=None,
            auto_optimize_k=True,
            metric='cosine',
            max_k=min(10, len(embeddings_array) // 20)
        )
        
        clustering_start = time.time()
        clusterer.fit(embeddings_array, lyrics_list)
        clustering_time = time.time() - clustering_start
        
        labels = clusterer.get_cluster_assignments()
        cluster_info = clusterer.get_cluster_info()
        
        logger.info(f"Clustering completado: {cluster_info.get('n_clusters_actual', 'N/A')} clusters")
        logger.info(f"Tiempo clustering: {clustering_time:.2f}s")
        
        # 9. Evaluación
        logger.info("Evaluando clustering...")
        
        evaluator = ClusterEvaluator()
        evaluation = evaluator.evaluate_clustering(embeddings_array, labels, lyrics_list)
        
        silhouette = evaluation['standard_metrics'].get('silhouette_score', 0)
        davies_bouldin = evaluation['standard_metrics'].get('davies_bouldin_index', 0)
        
        logger.info(f"Silhouette Score: {silhouette:.3f}")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
        
        # 10. Resultados finales
        total_time = time.time() - start_time
        
        final_results = {
            "success": True,
            "experiment_info": {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "sample_size": len(df_sample),
                "total_time": total_time,
                "dataset_used": Path(dataset_path).name
            },
            "processing_metrics": {
                "embeddings_success_rate": success_count / len(lyrics_list),
                "processing_speed": len(lyrics_list) / processing_time,
                "processing_time": processing_time
            },
            "clustering_results": {
                "n_clusters": cluster_info.get('n_clusters_actual', 0),
                "clustering_time": clustering_time,
                "cluster_info": cluster_info
            },
            "evaluation_metrics": {
                "silhouette_score": silhouette,
                "davies_bouldin_index": davies_bouldin,
                "n_samples_evaluated": len(labels)
            }
        }
        
        # 11. Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"validation_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"Resultados guardados: {results_file}")
        
        # 12. Resumen final
        logger.info("=" * 60)
        logger.info("VALIDACION EXPERIMENTAL COMPLETADA")
        logger.info(f"Tiempo total: {total_time:.2f}s")
        logger.info(f"Canciones procesadas: {len(df_sample)}")
        logger.info(f"Clusters generados: {cluster_info.get('n_clusters_actual', 0)}")
        logger.info(f"Silhouette Score: {silhouette:.3f}")
        logger.info(f"Archivo resultados: {results_file}")
        logger.info("=" * 60)
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error en validacion experimental: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Función principal."""
    print("VALIDACION EXPERIMENTAL SIMPLE - Sistema Clustering Semantico FASE 5")
    print("=" * 70)
    
    # Ejecutar validación con diferentes tamaños
    sample_sizes = [100, 500, 1000]
    
    for sample_size in sample_sizes:
        print(f"\nEXPERIMENTO: {sample_size} canciones")
        print("-" * 40)
        
        results = run_simple_validation(sample_size)
        
        if results["success"]:
            print(f"EXITO: Experimento {sample_size} completado")
            metrics = results["evaluation_metrics"]
            print(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")
            print(f"  Clusters: {results['clustering_results']['n_clusters']}")
            print(f"  Tiempo: {results['experiment_info']['total_time']:.2f}s")
        else:
            print(f"ERROR: {results['error']}")
            break  # Parar si hay error
    
    print("\nVALIDACION EXPERIMENTAL COMPLETA")


if __name__ == "__main__":
    main()