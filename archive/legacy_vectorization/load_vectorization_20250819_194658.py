#!/usr/bin/env python3
"""
Script de carga rápida - Vectorización Completa 20250819_194658
Generado automáticamente para cargar embeddings y índice de similitud.
"""

import numpy as np
import pickle
from pathlib import Path

def load_complete_vectorization():
    """Carga vectorización completa."""
    base_dir = Path(__file__).parent
    
    # Cargar componentes
    embeddings = np.load(base_dir / "embeddings_complete_20250819_194658.npy")
    track_ids = np.load(base_dir / "track_ids_complete_20250819_194658.npy")
    
    with open(base_dir / "similarity_index_20250819_194658.pkl", 'rb') as f:
        similarity_index = pickle.load(f)
    
    return {
        "embeddings": embeddings,
        "track_ids": track_ids,
        "similarity_index": similarity_index,
        "timestamp": "20250819_194658"
    }

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
    print(f"Vectorización cargada: {len(data['track_ids'])} canciones")
    print(f"Dimensiones embeddings: {data['embeddings'].shape}")
    
    # Test similitud
    sample_track = data["track_ids"][0]
    similar = find_similar_songs(sample_track, 5)
    print(f"\nCanciones similares a {sample_track}:")
    for track, distance in similar:
        print(f"  {track}: {distance:.3f}")
