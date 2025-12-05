#!/usr/bin/env python3
"""
Sistema de Recomendación Musical basado en Clustering
===================================================

Sistema optimizado que usa modelos pre-entrenados para predecir clusters
y encontrar canciones similares basadas en características musicales.

Uso:
    python music_recommender.py --models-dir results/models --song-id "5zlcxSrYyFmCmSRbere3c5"
    python music_recommender.py --models-dir results/models --song-features song.json --top-n 10
"""

import pandas as pd
import numpy as np
import argparse
import logging
import json
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

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
        description='Sistema de Recomendación Musical basado en Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --models-dir results/models --song-id "5zlcxSrYyFmCmSRbere3c5"
  %(prog)s --models-dir results/models --song-features song.json --top-n 10
  %(prog)s --models-dir results/models --interactive
        """
    )
    
    # Argumentos principales
    parser.add_argument('--models-dir', type=str, required=True,
                       help='Directorio con modelos entrenados (scaler, kmeans)')
    
    parser.add_argument('--dataset', type=str, 
                       default='../../data/3_selected/picked_data_optimal.csv',
                       help='Dataset con canciones clusterizadas (default: ../../data/3_selected/picked_data_optimal.csv)')
    
    parser.add_argument('--clustered-data', type=str,
                       help='Archivo CSV con resultados de clustering (opcional)')
    
    # Modos de entrada
    parser.add_argument('--song-id', type=str,
                       help='ID de canción existente para encontrar similares')
    
    parser.add_argument('--song-features', type=str,
                       help='Archivo JSON con características de nueva canción')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Modo interactivo para probar múltiples canciones')
    
    # Configuración de recomendación
    parser.add_argument('--top-n', type=int, default=5,
                       help='Número de canciones similares a retornar (default: 5)')
    
    parser.add_argument('--similarity-metric', choices=['cosine', 'euclidean', 'manhattan'],
                       default='manhattan', help='Métrica de similitud (default: manhattan)')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Nivel de logging (default: INFO)')
    
    return parser.parse_args()

def load_models(models_dir, logger=None):
    """Cargar modelos entrenados (incluyendo PCA si existe)"""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        error_msg = f"❌ Directorio de modelos no encontrado: {models_dir}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None, None, None
    
    # Buscar archivos de modelos
    scaler_files = list(models_path.glob('scaler*.pkl'))
    kmeans_files = list(models_path.glob('kmeans*.pkl'))
    pca_files = list(models_path.glob('pca*.pkl'))
    
    if not scaler_files or not kmeans_files:
        error_msg = f"❌ Modelos no encontrados en: {models_dir}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None, None, None
    
    # Usar los archivos más recientes
    scaler_file = sorted(scaler_files)[-1]
    kmeans_file = sorted(kmeans_files)[-1]
    
    try:
        scaler = joblib.load(scaler_file)
        kmeans = joblib.load(kmeans_file)
        
        # Cargar PCA si existe
        pca = None
        if pca_files:
            pca_file = sorted(pca_files)[-1]
            pca = joblib.load(pca_file)
            
            if logger:
                logger.info(f"✅ Modelos cargados:")
                logger.info(f"   Scaler: {scaler_file.name}")
                logger.info(f"   PCA: {pca_file.name} ({pca.n_components} componentes)")
                logger.info(f"   K-Means: {kmeans_file.name} (K={kmeans.n_clusters})")
            else:
                print(f"✅ Modelos cargados:")
                print(f"   Scaler: {scaler_file.name}")
                print(f"   PCA: {pca_file.name} ({pca.n_components} componentes)")
                print(f"   K-Means: {kmeans_file.name} (K={kmeans.n_clusters})")
        else:
            if logger:
                logger.info(f"✅ Modelos cargados:")
                logger.info(f"   Scaler: {scaler_file.name}")
                logger.info(f"   K-Means: {kmeans_file.name} (K={kmeans.n_clusters})")
            else:
                print(f"✅ Modelos cargados:")
                print(f"   Scaler: {scaler_file.name}")
                print(f"   K-Means: {kmeans_file.name} (K={kmeans.n_clusters})")
        
        return scaler, kmeans, pca
        
    except Exception as e:
        error_msg = f"❌ Error al cargar modelos: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None, None, None

def load_clustered_dataset(dataset_path, clustered_data_path=None, logger=None):
    """Cargar dataset con clusters asignados"""
    
    # Priorizar dataset clusterizado si se proporciona
    if clustered_data_path and Path(clustered_data_path).exists():
        data_path = clustered_data_path
        if logger:
            logger.info(f"🔍 Cargando dataset clusterizado: {data_path}")
    else:
        # Buscar resultado más reciente en results/results/
        results_dir = Path('results/results')
        if results_dir.exists():
            cluster_files = list(results_dir.glob('clustering_results_*.csv'))
            if cluster_files:
                data_path = sorted(cluster_files)[-1]
                if logger:
                    logger.info(f"🔍 Usando dataset clusterizado más reciente: {data_path}")
            else:
                data_path = dataset_path
                if logger:
                    logger.warning(f"⚠️  Dataset clusterizado no encontrado, usando original: {data_path}")
        else:
            data_path = dataset_path
            if logger:
                logger.warning(f"⚠️  Directorio de resultados no encontrado, usando dataset original: {data_path}")
    
    try:
        df = pd.read_csv(data_path, sep='^', decimal='.', encoding='utf-8', on_bad_lines='skip')
        
        # Verificar si tiene columna cluster
        has_clusters = 'cluster' in df.columns
        
        if logger:
            logger.info(f"✅ Dataset cargado: {len(df):,} canciones")
            logger.info(f"   Clusters asignados: {'Sí' if has_clusters else 'No'}")
        else:
            print(f"✅ Dataset cargado: {len(df):,} canciones")
            print(f"   Clusters asignados: {'Sí' if has_clusters else 'No'}")
        
        return df, has_clusters
        
    except Exception as e:
        error_msg = f"❌ Error al cargar dataset: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None, False

def predict_cluster_and_similarities(song_features, scaler, kmeans, df, 
                                   similarity_metric='manhattan', top_n=5, 
                                   has_clusters=False, pca=None, logger=None):
    """
    Predecir cluster y encontrar canciones similares
    """
    
    # Preparar características de la canción
    if isinstance(song_features, dict):
        features_array = np.array([[song_features[col] for col in MUSICAL_FEATURES]])
    else:
        features_array = song_features.reshape(1, -1)
    
    # Normalizar características
    normalized_features = scaler.transform(features_array)
    
    # Aplicar PCA si existe
    if pca is not None:
        normalized_features = pca.transform(normalized_features)
    
    # Predecir cluster
    predicted_cluster = kmeans.predict(normalized_features)[0]
    cluster_probabilities = kmeans.transform(normalized_features)[0]  # Distancias a centroides
    
    if logger:
        logger.info(f"🎯 Cluster predicho: {predicted_cluster}")
        logger.debug(f"   Distancias a centroides: {cluster_probabilities}")
    
    # Si el dataset tiene clusters asignados, filtrar por cluster
    if has_clusters:
        cluster_songs = df[df['cluster'] == predicted_cluster].copy()
        if logger:
            logger.info(f"🎵 Canciones en cluster {predicted_cluster}: {len(cluster_songs)}")
    else:
        # Si no tiene clusters, usar todo el dataset y asignar clusters
        if logger:
            logger.info("⚠️  Dataset sin clusters, asignando clusters a todas las canciones...")
        
        # Asignar clusters a todo el dataset
        dataset_features = df[MUSICAL_FEATURES].values
        normalized_dataset = scaler.transform(dataset_features)
        
        # Aplicar PCA si existe
        if pca is not None:
            normalized_dataset = pca.transform(normalized_dataset)
            
        all_clusters = kmeans.predict(normalized_dataset)
        df_temp = df.copy()
        df_temp['cluster'] = all_clusters
        
        cluster_songs = df_temp[df_temp['cluster'] == predicted_cluster].copy()
        if logger:
            logger.info(f"🎵 Canciones asignadas al cluster {predicted_cluster}: {len(cluster_songs)}")
    
    if len(cluster_songs) == 0:
        if logger:
            logger.warning("⚠️  No se encontraron canciones en el cluster predicho")
        return predicted_cluster, pd.DataFrame(), cluster_probabilities
    
    # Calcular similitudes con canciones del cluster
    cluster_features = cluster_songs[MUSICAL_FEATURES].values
    cluster_features_normalized = scaler.transform(cluster_features)
    
    # Aplicar PCA si existe
    if pca is not None:
        cluster_features_normalized = pca.transform(cluster_features_normalized)
    
    # Seleccionar métrica de similitud
    if similarity_metric == 'cosine':
        similarities = cosine_similarity(normalized_features, cluster_features_normalized)[0]
    elif similarity_metric == 'euclidean':
        distances = euclidean_distances(normalized_features, cluster_features_normalized)[0]
        similarities = 1 / (1 + distances)  # Convertir distancia a similitud
    else:  # manhattan
        distances = manhattan_distances(normalized_features, cluster_features_normalized)[0]
        similarities = 1 / (1 + distances)
    
    # Añadir similitudes al dataframe
    cluster_songs_copy = cluster_songs.copy()
    cluster_songs_copy['similarity'] = similarities
    
    # Ordenar por similitud y retornar top-N
    most_similar = cluster_songs_copy.nlargest(top_n, 'similarity')
    
    return predicted_cluster, most_similar, cluster_probabilities

def find_similar_by_id(song_id, df, scaler, kmeans, similarity_metric='manhattan', 
                      top_n=5, has_clusters=False, pca=None, logger=None):
    """Encontrar canciones similares por ID de canción existente"""
    
    # Buscar la canción en el dataset
    song_row = df[df['id'] == song_id]
    
    if song_row.empty:
        error_msg = f"❌ Canción no encontrada: {song_id}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None, None, None, None
    
    song_data = song_row.iloc[0]
    song_features = {col: song_data[col] for col in MUSICAL_FEATURES}
    
    if logger:
        logger.info(f"🎵 Canción encontrada: \"{song_data['name']}\" - {song_data['artists']}")
    else:
        print(f"🎵 Canción encontrada: \"{song_data['name']}\" - {song_data['artists']}")
    
    # Usar la función de predicción
    cluster, similar_songs, cluster_probs = predict_cluster_and_similarities(
        song_features, scaler, kmeans, df, similarity_metric, top_n + 1, has_clusters, pca, logger
    )
    
    # Excluir la canción original de los resultados
    if not similar_songs.empty:
        similar_songs = similar_songs[similar_songs['id'] != song_id].head(top_n)
    
    return song_data, cluster, similar_songs, cluster_probs

def load_song_features_from_json(json_path, logger=None):
    """Cargar características de canción desde archivo JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        
        # Verificar que tenga todas las características necesarias
        missing_features = [f for f in MUSICAL_FEATURES if f not in features]
        if missing_features:
            error_msg = f"❌ Características faltantes en JSON: {missing_features}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            return None
        
        return features
        
    except Exception as e:
        error_msg = f"❌ Error al cargar JSON: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None

def print_recommendations(song_data, cluster, similar_songs, cluster_probs, similarity_metric):
    """Mostrar recomendaciones de forma formateada"""
    
    print("\n" + "="*70)
    print("🎵 RECOMENDACIONES MUSICALES")
    print("="*70)
    
    if song_data is not None:
        print(f"📀 Canción de referencia:")
        print(f"   🎤 \"{song_data['name']}\" - {song_data['artists']}")
        if 'year' in song_data:
            print(f"   📅 Año: {song_data['year']}")
    
    print(f"\n🎯 Cluster asignado: {cluster}")
    print(f"📊 Métrica de similitud: {similarity_metric}")
    print(f"📈 Distancias a centroides: {[f'{d:.3f}' for d in cluster_probs]}")
    
    if similar_songs.empty:
        print("\n❌ No se encontraron canciones similares")
        return
    
    print(f"\n🎵 Top {len(similar_songs)} canciones más similares:")
    print("-" * 70)
    
    for i, (_, song) in enumerate(similar_songs.iterrows(), 1):
        similarity_pct = song['similarity'] * 100
        print(f"{i:2d}. 🎤 \"{song['name']}\" - {song['artists']}")
        print(f"    📊 Similitud: {similarity_pct:.1f}% | 📅 Año: {song.get('year', 'N/A')}")
        if i < len(similar_songs):
            print()

def interactive_mode(scaler, kmeans, df, has_clusters, similarity_metric, pca, logger):
    """Modo interactivo para probar múltiples canciones"""
    
    print("\n🎵 MODO INTERACTIVO - RECOMENDADOR MUSICAL")
    print("="*60)
    print("Comandos disponibles:")
    print("  id <song_id>     - Buscar por ID de canción")  
    print("  search <término> - Buscar canciones por nombre/artista")
    print("  random          - Seleccionar canción aleatoria")
    print("  quit            - Salir")
    print("-"*60)
    
    while True:
        try:
            command = input("\n🎯 Ingresa comando: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 ¡Hasta luego!")
                break
            
            elif command.lower() == 'random':
                # Seleccionar canción aleatoria
                random_song = df.sample(n=1).iloc[0]
                song_id = random_song['id']
                print(f"🎲 Canción aleatoria seleccionada: {song_id}")
                
                # Procesar recomendación
                song_data, cluster, similar_songs, cluster_probs = find_similar_by_id(
                    song_id, df, scaler, kmeans, similarity_metric, 5, has_clusters, pca, logger
                )
                
                if similar_songs is not None:
                    print_recommendations(song_data, cluster, similar_songs, cluster_probs, similarity_metric)
            
            elif command.startswith('id '):
                song_id = command[3:].strip()
                
                # Procesar recomendación
                song_data, cluster, similar_songs, cluster_probs = find_similar_by_id(
                    song_id, df, scaler, kmeans, similarity_metric, 5, has_clusters, pca, logger
                )
                
                if similar_songs is not None:
                    print_recommendations(song_data, cluster, similar_songs, cluster_probs, similarity_metric)
            
            elif command.startswith('search '):
                search_term = command[7:].strip().lower()
                
                # Buscar canciones que coincidan
                name_matches = df[df['name'].str.lower().str.contains(search_term, na=False)]
                artist_matches = df[df['artists'].str.lower().str.contains(search_term, na=False)]
                matches = pd.concat([name_matches, artist_matches]).drop_duplicates()
                
                if matches.empty:
                    print(f"❌ No se encontraron canciones con: '{search_term}'")
                else:
                    print(f"🔍 Canciones encontradas ({len(matches)}):")
                    for i, (_, song) in enumerate(matches.head(10).iterrows(), 1):
                        print(f"  {i}. {song['id']} - \"{song['name']}\" - {song['artists']}")
                    
                    if len(matches) > 10:
                        print(f"    ... y {len(matches) - 10} más")
            
            else:
                print("❌ Comando no reconocido. Usa: id <song_id>, search <término>, random, o quit")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Función principal"""
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    logger.info("🎵 INICIANDO SISTEMA DE RECOMENDACIÓN MUSICAL")
    logger.info("=" * 60)
    
    try:
        # 1. Cargar modelos
        scaler, kmeans, pca = load_models(args.models_dir, logger)
        if scaler is None or kmeans is None:
            return 1
        
        # 2. Cargar dataset
        df, has_clusters = load_clustered_dataset(args.dataset, args.clustered_data, logger)
        if df is None:
            return 1
        
        # 3. Procesar según modo de entrada
        if args.interactive:
            interactive_mode(scaler, kmeans, df, has_clusters, args.similarity_metric, pca, logger)
        
        elif args.song_id:
            # Buscar por ID de canción existente
            song_data, cluster, similar_songs, cluster_probs = find_similar_by_id(
                args.song_id, df, scaler, kmeans, args.similarity_metric, 
                args.top_n, has_clusters, pca, logger
            )
            
            if similar_songs is not None:
                print_recommendations(song_data, cluster, similar_songs, cluster_probs, args.similarity_metric)
        
        elif args.song_features:
            # Cargar características desde JSON
            features = load_song_features_from_json(args.song_features, logger)
            if features is None:
                return 1
            
            # Predecir cluster y similitudes
            cluster, similar_songs, cluster_probs = predict_cluster_and_similarities(
                features, scaler, kmeans, df, args.similarity_metric, 
                args.top_n, has_clusters, pca, logger
            )
            
            print_recommendations(None, cluster, similar_songs, cluster_probs, args.similarity_metric)
        
        else:
            logger.error("❌ Debe especificar --song-id, --song-features, o --interactive")
            return 1
        
        logger.info("🎵 ¡RECOMENDACIÓN COMPLETADA EXITOSAMENTE! 🎉")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error durante la ejecución: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)