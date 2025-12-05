#!/usr/bin/env python3
"""
Sistema de Recomendación Musical - Dataset Completo (1.2M canciones)
================================================================

Sistema híbrido que usa:
- Modelos entrenados con 9,677 canciones representativas
- Recomendaciones desde dataset completo de 1.2M canciones

Uso:
    python music_recommender_full.py --models-dir final_models/method1_pca5_silhouette0314 --song-id "ID" --top-n 10
"""

import pandas as pd
import numpy as np
import argparse
import logging
import json
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import time

# Características musicales esperadas
MUSICAL_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
]

def setup_logging(log_level='INFO'):
    """Configurar sistema de logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parsear argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Sistema de Recomendación Musical - Dataset Completo (1.2M)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Argumentos principales
    parser.add_argument('--models-dir', type=str, required=True,
                       help='Directorio con modelos entrenados (scaler, pca, kmeans)')
    
    parser.add_argument('--full-dataset', type=str, 
                       default='../data/cleaned_data/tracks_features_clean.csv',
                       help='Dataset completo 1.2M canciones (default: tracks_features_clean.csv)')
    
    # Modos de entrada
    parser.add_argument('--song-id', type=str,
                       help='ID de canción para encontrar similares')
    
    parser.add_argument('--search', type=str,
                       help='Buscar canción por nombre/artista')
    
    parser.add_argument('--random', action='store_true',
                       help='Seleccionar canción aleatoria del dataset completo')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Modo interactivo')
    
    # Configuración de recomendación
    parser.add_argument('--top-n', type=int, default=10,
                       help='Número de canciones similares (default: 10)')
    
    parser.add_argument('--similarity-metric', choices=['cosine', 'euclidean', 'manhattan'],
                       default='manhattan', help='Métrica de similitud (default: manhattan)')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Nivel de logging (default: INFO)')
    
    # Optimización
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Tamaño de chunk para procesar dataset grande (default: 10000)')
    
    return parser.parse_args()

def load_models_pca(models_dir, logger=None):
    """Cargar modelos PCA entrenados"""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        error_msg = f"❌ Directorio de modelos no encontrado: {models_dir}"
        if logger:
            logger.error(error_msg)
        return None, None, None
    
    # Buscar archivos de modelos
    scaler_files = list(models_path.glob('scaler*.pkl'))
    kmeans_files = list(models_path.glob('kmeans*.pkl'))
    pca_files = list(models_path.glob('pca*.pkl'))
    
    if not scaler_files or not kmeans_files or not pca_files:
        error_msg = f"❌ Modelos PCA no encontrados en: {models_dir}"
        if logger:
            logger.error(error_msg)
        return None, None, None
    
    try:
        scaler = joblib.load(sorted(scaler_files)[-1])
        kmeans = joblib.load(sorted(kmeans_files)[-1])
        pca = joblib.load(sorted(pca_files)[-1])
        
        if logger:
            logger.info(f"✅ Modelos PCA cargados:")
            logger.info(f"   🔧 Scaler: StandardScaler")
            logger.info(f"   🎯 PCA: {pca.n_components} componentes ({pca.explained_variance_ratio_.sum():.1%} varianza)")
            logger.info(f"   🎵 K-Means: {kmeans.n_clusters} clusters")
        
        return scaler, pca, kmeans
        
    except Exception as e:
        error_msg = f"❌ Error al cargar modelos: {e}"
        if logger:
            logger.error(error_msg)
        return None, None, None

def load_full_dataset(dataset_path, logger=None):
    """Cargar dataset completo de 1.2M canciones de forma eficiente"""
    if not Path(dataset_path).exists():
        error_msg = f"❌ Dataset no encontrado: {dataset_path}"
        if logger:
            logger.error(error_msg)
        return None
    
    if logger:
        logger.info(f"🔍 Cargando dataset completo: {dataset_path}")
    
    start_time = time.time()
    
    try:
        # Cargar con optimizaciones para dataset grande
        df = pd.read_csv(
            dataset_path, 
            sep=';', 
            decimal=',', 
            encoding='utf-8', 
            on_bad_lines='skip',
            low_memory=False,
            dtype={
                'year': 'Int64',  # Permite NaN
                'explicit': 'bool'
            }
        )
        
        load_time = time.time() - start_time
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        if logger:
            logger.info(f"✅ Dataset completo cargado:")
            logger.info(f"   🎵 Canciones: {len(df):,}")
            logger.info(f"   ⏱️  Tiempo: {load_time:.2f}s")
            logger.info(f"   💾 Memoria: {memory_usage:.1f}MB")
        
        return df
        
    except Exception as e:
        error_msg = f"❌ Error al cargar dataset: {e}"
        if logger:
            logger.error(error_msg)
        return None

def vectorize_song(song_features, scaler, pca):
    """Vectorizar una canción usando el pipeline PCA"""
    # Convertir a array si es diccionario
    if isinstance(song_features, dict):
        features_array = np.array([[song_features[col] for col in MUSICAL_FEATURES]])
    else:
        features_array = song_features.reshape(1, -1)
    
    # Pipeline: Normalizar → PCA
    normalized = scaler.transform(features_array)
    vectorized = pca.transform(normalized)
    
    return vectorized

def predict_cluster(song_vector, kmeans):
    """Predecir cluster de una canción vectorizada"""
    cluster = kmeans.predict(song_vector)[0]
    distances = kmeans.transform(song_vector)[0]
    return cluster, distances

def find_similar_songs_optimized(target_song, full_dataset, scaler, pca, kmeans, 
                                similarity_metric='manhattan', top_n=10, 
                                chunk_size=10000, logger=None):
    """
    Encontrar canciones similares de forma optimizada para dataset grande
    """
    if logger:
        logger.info(f"🎯 Buscando canciones similares en {len(full_dataset):,} canciones...")
    
    start_time = time.time()
    
    # 1. Vectorizar canción objetivo
    target_features = {col: target_song[col] for col in MUSICAL_FEATURES}
    target_vector = vectorize_song(target_features, scaler, pca)
    target_cluster, cluster_distances = predict_cluster(target_vector, kmeans)
    
    if logger:
        logger.info(f"   🎯 Cluster predicho: {target_cluster}")
        logger.debug(f"   📊 Distancias: {cluster_distances}")
    
    # 2. Procesar dataset por chunks para eficiencia
    all_similarities = []
    total_processed = 0
    
    for chunk_start in range(0, len(full_dataset), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(full_dataset))
        chunk = full_dataset.iloc[chunk_start:chunk_end].copy()
        
        # Extraer features del chunk
        chunk_features = chunk[MUSICAL_FEATURES].values
        
        # Vectorizar chunk completo
        chunk_normalized = scaler.transform(chunk_features)
        chunk_vectorized = pca.transform(chunk_normalized)
        
        # Predecir clusters del chunk
        chunk_clusters = kmeans.predict(chunk_vectorized)
        
        # Filtrar solo canciones del mismo cluster
        same_cluster_mask = (chunk_clusters == target_cluster)
        same_cluster_indices = np.where(same_cluster_mask)[0]
        
        if len(same_cluster_indices) > 0:
            # Calcular similitudes solo con canciones del mismo cluster
            cluster_vectors = chunk_vectorized[same_cluster_indices]
            
            if similarity_metric == 'cosine':
                similarities = cosine_similarity(target_vector, cluster_vectors)[0]
            elif similarity_metric == 'euclidean':
                distances = euclidean_distances(target_vector, cluster_vectors)[0]
                similarities = 1 / (1 + distances)
            else:  # manhattan
                distances = manhattan_distances(target_vector, cluster_vectors)[0]
                similarities = 1 / (1 + distances)
            
            # Crear DataFrame con resultados del chunk
            chunk_results = chunk.iloc[same_cluster_indices].copy()
            chunk_results['similarity'] = similarities
            chunk_results['chunk_index'] = chunk_start + same_cluster_indices
            
            all_similarities.append(chunk_results)
        
        total_processed += len(chunk)
        
        if logger and total_processed % (chunk_size * 10) == 0:
            logger.debug(f"   📊 Procesadas: {total_processed:,}/{len(full_dataset):,} canciones")
    
    # 3. Combinar resultados y encontrar top-N
    if not all_similarities:
        if logger:
            logger.warning("⚠️  No se encontraron canciones en el mismo cluster")
        return pd.DataFrame(), target_cluster, cluster_distances
    
    # Concatenar todos los resultados
    combined_results = pd.concat(all_similarities, ignore_index=True)
    
    # Excluir la canción original si está presente
    if 'id' in target_song:
        combined_results = combined_results[combined_results['id'] != target_song['id']]
    
    # Obtener top-N más similares
    top_similar = combined_results.nlargest(top_n, 'similarity')
    
    processing_time = time.time() - start_time
    
    if logger:
        logger.info(f"✅ Búsqueda completada:")
        logger.info(f"   🎵 Candidatos en cluster {target_cluster}: {len(combined_results):,}")
        logger.info(f"   🏆 Top similares: {len(top_similar)}")
        logger.info(f"   ⏱️  Tiempo: {processing_time:.2f}s")
    
    return top_similar, target_cluster, cluster_distances

def print_recommendations_full(target_song, similar_songs, cluster, cluster_distances, 
                              similarity_metric, dataset_size):
    """Mostrar recomendaciones de forma formateada"""
    
    print("\n" + "="*80)
    print("🎵 RECOMENDACIONES MUSICALES - DATASET COMPLETO")
    print("="*80)
    
    print(f"📀 Canción de referencia:")
    print(f"   🎤 \"{target_song['name']}\" - {target_song['artists']}")
    if 'year' in target_song and pd.notna(target_song['year']):
        print(f"   📅 Año: {int(target_song['year'])}")
    if 'album' in target_song:
        print(f"   💿 Álbum: {target_song['album']}")
    
    print(f"\n🎯 Información del clustering:")
    print(f"   📊 Dataset: {dataset_size:,} canciones")
    print(f"   🎯 Cluster asignado: {cluster}")
    print(f"   📈 Distancias a centroides: {[f'{d:.3f}' for d in cluster_distances]}")
    print(f"   📊 Métrica de similitud: {similarity_metric}")
    
    if similar_songs.empty:
        print("\n❌ No se encontraron canciones similares")
        return
    
    print(f"\n🎵 Top {len(similar_songs)} canciones más similares:")
    print("-" * 80)
    
    for i, (_, song) in enumerate(similar_songs.iterrows(), 1):
        similarity_pct = song['similarity'] * 100
        year_str = f"{int(song['year'])}" if 'year' in song and pd.notna(song['year']) else 'N/A'
        
        print(f"{i:2d}. 🎤 \"{song['name']}\" - {song['artists']}")
        print(f"    📊 Similitud: {similarity_pct:.1f}% | 📅 Año: {year_str}")
        if 'album' in song and pd.notna(song['album']):
            print(f"    💿 Álbum: {song['album']}")
        if i < len(similar_songs):
            print()

def search_song_in_dataset(search_term, dataset, logger=None):
    """Buscar canción en el dataset por nombre o artista"""
    if logger:
        logger.info(f"🔍 Buscando: '{search_term}'")
    
    search_lower = search_term.lower()
    
    # Buscar en nombre y artistas
    name_matches = dataset[dataset['name'].str.lower().str.contains(search_lower, na=False)]
    artist_matches = dataset[dataset['artists'].str.lower().str.contains(search_lower, na=False)]
    
    # Combinar y eliminar duplicados
    matches = pd.concat([name_matches, artist_matches]).drop_duplicates()
    
    if logger:
        logger.info(f"   📊 Encontradas: {len(matches)} canciones")
    
    return matches

def main():
    """Función principal"""
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    logger.info("🎵 SISTEMA DE RECOMENDACIÓN - DATASET COMPLETO (1.2M)")
    logger.info("=" * 80)
    
    try:
        # 1. Cargar modelos PCA
        scaler, pca, kmeans = load_models_pca(args.models_dir, logger)
        if scaler is None:
            return 1
        
        # 2. Cargar dataset completo
        full_dataset = load_full_dataset(args.full_dataset, logger)
        if full_dataset is None:
            return 1
        
        # 3. Procesar según modo de entrada
        target_song = None
        
        if args.song_id:
            # Buscar por ID específico
            song_matches = full_dataset[full_dataset['id'] == args.song_id]
            if song_matches.empty:
                logger.error(f"❌ Canción no encontrada: {args.song_id}")
                return 1
            target_song = song_matches.iloc[0]
            
        elif args.search:
            # Buscar por término
            matches = search_song_in_dataset(args.search, full_dataset, logger)
            if matches.empty:
                logger.error(f"❌ No se encontraron canciones con: '{args.search}'")
                return 1
            
            # Mostrar opciones si hay múltiples matches
            if len(matches) > 1:
                print(f"\n🔍 Encontradas {len(matches)} canciones:")
                for i, (_, song) in enumerate(matches.head(10).iterrows(), 1):
                    year_str = f"({int(song['year'])})" if 'year' in song and pd.notna(song['year']) else ""
                    print(f"  {i}. {song['id']} - \"{song['name']}\" - {song['artists']} {year_str}")
                
                if len(matches) > 10:
                    print(f"    ... y {len(matches) - 10} más")
                
                print("\n💡 Usa --song-id con el ID específico para obtener recomendaciones")
                return 0
            else:
                target_song = matches.iloc[0]
                
        elif args.random:
            # Seleccionar canción aleatoria
            target_song = full_dataset.sample(n=1, random_state=42).iloc[0]
            logger.info(f"🎲 Canción aleatoria: {target_song['id']}")
            
        else:
            logger.error("❌ Debe especificar --song-id, --search, o --random")
            return 1
        
        # 4. Encontrar canciones similares
        if target_song is not None:
            logger.info(f"🎵 Procesando: \"{target_song['name']}\" - {target_song['artists']}")
            
            similar_songs, cluster, cluster_distances = find_similar_songs_optimized(
                target_song, full_dataset, scaler, pca, kmeans,
                args.similarity_metric, args.top_n, args.chunk_size, logger
            )
            
            # 5. Mostrar resultados
            print_recommendations_full(
                target_song, similar_songs, cluster, cluster_distances,
                args.similarity_metric, len(full_dataset)
            )
        
        logger.info("🎵 ¡RECOMENDACIÓN COMPLETADA EXITOSAMENTE! 🎉")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error durante la ejecución: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)