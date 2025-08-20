#!/usr/bin/env python3
"""
TEST DE RECOMENDACIONES SEMÁNTICAS CON CANCIÓN ESPECÍFICA
Permite probar el sistema de recomendaciones con una canción elegida por el usuario.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_vectorization_and_dataset():
    """Carga datos de vectorización y dataset original."""
    print("🔄 Cargando sistema de vectorización...")
    
    # Cargar vectorización
    base_dir = Path(__file__).parent.parent / "vectorization_complete_output"
    timestamp = "20250819_194820"
    
    embeddings = np.load(base_dir / f"embeddings_complete_{timestamp}.npy")
    track_ids = np.load(base_dir / f"track_ids_complete_{timestamp}.npy")
    
    with open(base_dir / f"similarity_index_{timestamp}.pkl", 'rb') as f:
        similarity_index = pickle.load(f)
    
    # Filtrar embeddings válidos
    valid_mask = np.any(embeddings != 0, axis=1)
    valid_embeddings = embeddings[valid_mask]
    valid_track_ids = track_ids[valid_mask]
    
    print(f"✅ Vectorización cargada: {len(valid_embeddings)} embeddings válidos")
    
    # Cargar dataset original para obtener nombres de canciones
    dataset_path = Path(__file__).parent.parent / "data" / "final_data" / "picked_data_optimal.csv"
    
    try:
        # Probar diferentes separadores
        df = pd.read_csv(dataset_path, sep='^', encoding='utf-8')
        print(f"✅ Dataset cargado: {len(df)} canciones")
    except:
        try:
            df = pd.read_csv(dataset_path, sep='@@', encoding='utf-8')
            print(f"✅ Dataset cargado: {len(df)} canciones (separador @@)")
        except Exception as e:
            print(f"❌ Error cargando dataset: {e}")
            return None, None, None, None
    
    return valid_embeddings, valid_track_ids, similarity_index, df

def search_songs(df, query, max_results=10):
    """Busca canciones por nombre o artista."""
    print(f"\n🔍 Buscando canciones con '{query}'...")
    
    # Verificar columnas disponibles
    possible_cols = ['name', 'track_name', 'song_name', 'title', 'artist', 'artist_name', 'artists']
    name_col = None
    artist_col = None
    
    for col in df.columns:
        if any(x in col.lower() for x in ['name', 'title', 'song', 'track']) and name_col is None:
            name_col = col
        if 'artist' in col.lower() and artist_col is None:
            artist_col = col
    
    if name_col is None:
        print("❌ No se encontró columna de nombre de canción")
        print(f"Columnas disponibles: {list(df.columns)}")
        return []
    
    print(f"📊 Usando columnas: {name_col}, {artist_col}")
    
    # Buscar en nombre de canción
    query_lower = query.lower()
    name_matches = df[df[name_col].str.lower().str.contains(query_lower, na=False)]
    
    # Buscar en artista si existe la columna
    artist_matches = pd.DataFrame()
    if artist_col is not None:
        artist_matches = df[df[artist_col].str.lower().str.contains(query_lower, na=False)]
    
    # Combinar resultados
    all_matches = pd.concat([name_matches, artist_matches]).drop_duplicates()
    
    if len(all_matches) == 0:
        print(f"❌ No se encontraron canciones con '{query}'")
        return []
    
    # Mostrar resultados
    print(f"✅ Encontradas {len(all_matches)} canciones:")
    results = []
    
    for idx, (_, row) in enumerate(all_matches.head(max_results).iterrows()):
        artist_info = f" - {row[artist_col]}" if artist_col is not None else ""
        print(f"  {idx}: {row[name_col]}{artist_info}")
        results.append({
            'index': idx,
            'track_id': row.get('id', row.get('track_id', 'unknown')),
            'name': row[name_col],
            'artist': row[artist_col] if artist_col is not None else 'Unknown',
            'df_index': row.name
        })
    
    return results

def get_song_recommendations(embeddings, track_ids, similarity_index, target_track_id, n_recommendations=10):
    """Obtiene recomendaciones para una canción específica."""
    print(f"\n🎵 Generando recomendaciones para track: {target_track_id}")
    
    # Encontrar índice de la canción
    track_indices = np.where(track_ids == target_track_id)[0]
    
    if len(track_indices) == 0:
        print(f"❌ Canción {target_track_id} no encontrada en vectorización")
        return None
    
    track_idx = track_indices[0]
    print(f"✅ Canción encontrada en índice {track_idx}")
    
    # Obtener embedding de la canción
    target_embedding = embeddings[track_idx:track_idx+1]
    
    # Método 1: Usar índice de similitud precomputado
    if 'model' in similarity_index:
        model = similarity_index['model']
        distances, indices = model.kneighbors(target_embedding, n_neighbors=n_recommendations+1)
        
        recommendations = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if i == 0:  # Skip la misma canción
                continue
            if idx < len(track_ids):
                similarity = 1 - dist
                recommendations.append({
                    'rank': i,
                    'track_id': track_ids[idx],
                    'similarity': similarity,
                    'distance': dist
                })
    else:
        # Método 2: Calcular similitudes directamente
        similarities = cosine_similarity(target_embedding, embeddings)[0]
        
        # Obtener índices ordenados por similitud (excluyendo la misma canción)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        recommendations = []
        for i, idx in enumerate(similar_indices):
            recommendations.append({
                'rank': i + 1,
                'track_id': track_ids[idx],
                'similarity': similarities[idx],
                'distance': 1 - similarities[idx]
            })
    
    print(f"✅ Generadas {len(recommendations)} recomendaciones")
    return recommendations

def get_clustering_info(embeddings, track_ids, target_track_id):
    """Obtiene información de clustering para la canción."""
    print(f"\n🎯 Analizando clusters para track: {target_track_id}")
    
    # Encontrar índice de la canción
    track_indices = np.where(track_ids == target_track_id)[0]
    if len(track_indices) == 0:
        return None
    
    track_idx = track_indices[0]
    
    # Clustering K-Means K=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    target_cluster = cluster_labels[track_idx]
    cluster_sizes = np.bincount(cluster_labels)
    
    cluster_info = {
        'target_cluster': int(target_cluster),
        'cluster_size': int(cluster_sizes[target_cluster]),
        'cluster_percentage': cluster_sizes[target_cluster] / len(cluster_labels) * 100,
        'other_cluster_size': int(cluster_sizes[1 - target_cluster]),
        'cluster_interpretation': 'Introspectivo' if target_cluster == 0 else 'Extrovertido'
    }
    
    print(f"✅ Canción en cluster {target_cluster} ({cluster_info['cluster_interpretation']})")
    print(f"   Tamaño del cluster: {cluster_info['cluster_size']} canciones ({cluster_info['cluster_percentage']:.1f}%)")
    
    return cluster_info, cluster_labels

def enhance_recommendations_with_names(recommendations, df, name_col, artist_col):
    """Añade nombres de canciones y artistas a las recomendaciones."""
    enhanced = []
    
    for rec in recommendations:
        # Buscar en dataset
        song_info = df[df.get('id', df.get('track_id', pd.Series())) == rec['track_id']]
        
        if len(song_info) > 0:
            song_data = song_info.iloc[0]
            enhanced.append({
                **rec,
                'name': song_data[name_col],
                'artist': song_data[artist_col] if artist_col is not None else 'Unknown'
            })
        else:
            enhanced.append({
                **rec,
                'name': 'Unknown Song',
                'artist': 'Unknown Artist'
            })
    
    return enhanced

def display_recommendations(recommendations, cluster_info=None):
    """Muestra las recomendaciones de forma organizada."""
    print(f"\n🎵 TOP {len(recommendations)} RECOMENDACIONES SEMÁNTICAS:")
    print("="*80)
    
    if cluster_info:
        print(f"🎯 Canción base: Cluster {cluster_info['target_cluster']} ({cluster_info['cluster_interpretation']})")
        print()
    
    for rec in recommendations:
        similarity_pct = rec['similarity'] * 100
        print(f"{rec['rank']:2d}. {rec['name']}")
        print(f"    👤 {rec['artist']}")
        print(f"    📊 Similitud: {similarity_pct:.1f}% (distancia: {rec['distance']:.3f})")
        print(f"    🔗 ID: {rec['track_id']}")
        print()

def main():
    """Función principal de testing."""
    print("🎵 SISTEMA DE RECOMENDACIONES SEMÁNTICAS - TEST INTERACTIVO")
    print("="*70)
    
    # Cargar datos
    embeddings, track_ids, similarity_index, df = load_vectorization_and_dataset()
    
    if df is None:
        print("❌ No se pudo cargar el dataset")
        return
    
    # Detectar columnas
    possible_name_cols = [col for col in df.columns if any(x in col.lower() for x in ['name', 'title', 'song', 'track'])]
    possible_artist_cols = [col for col in df.columns if 'artist' in col.lower()]
    
    name_col = possible_name_cols[0] if possible_name_cols else None
    artist_col = possible_artist_cols[0] if possible_artist_cols else None
    
    print(f"\n📊 Dataset info:")
    print(f"   Columnas detectadas: nombre='{name_col}', artista='{artist_col}'")
    print(f"   Total canciones: {len(df)}")
    print(f"   Embeddings disponibles: {len(track_ids)}")
    
    while True:
        print(f"\n" + "="*50)
        query = input("🔍 Ingresa nombre de canción o artista (o 'quit' para salir): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 ¡Gracias por probar el sistema!")
            break
        
        if not query:
            continue
        
        # Buscar canciones
        search_results = search_songs(df, query)
        
        if not search_results:
            continue
        
        # Seleccionar canción
        try:
            if len(search_results) == 1:
                selected_idx = 0
            else:
                selected_idx = int(input(f"\n✏️  Selecciona canción (0-{len(search_results)-1}): "))
                
            if selected_idx < 0 or selected_idx >= len(search_results):
                print("❌ Índice inválido")
                continue
                
            selected_song = search_results[selected_idx]
            print(f"\n🎵 Canción seleccionada: {selected_song['name']} - {selected_song['artist']}")
            
        except ValueError:
            print("❌ Por favor ingresa un número válido")
            continue
        
        # Generar recomendaciones
        recommendations = get_song_recommendations(
            embeddings, track_ids, similarity_index, 
            selected_song['track_id'], n_recommendations=10
        )
        
        if recommendations is None:
            print("❌ No se pudieron generar recomendaciones")
            continue
        
        # Obtener información de clustering
        cluster_info, cluster_labels = get_clustering_info(
            embeddings, track_ids, selected_song['track_id']
        )
        
        # Enriquecer recomendaciones con nombres
        enhanced_recommendations = enhance_recommendations_with_names(
            recommendations, df, name_col, artist_col
        )
        
        # Mostrar resultados
        display_recommendations(enhanced_recommendations, cluster_info)

if __name__ == "__main__":
    main()