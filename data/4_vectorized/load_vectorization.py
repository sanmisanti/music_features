#!/usr/bin/env python3
"""
Script de carga rapida - Embeddings BERT de Letras
Actualizado para estructura reorganizada de data/4_vectorized/
"""

import numpy as np
import pickle
from pathlib import Path

def load_complete_vectorization():
    """Carga vectorizacion completa con embeddings BERT."""
    base_dir = Path(__file__).parent

    # Cargar componentes (nombres actualizados)
    embeddings = np.load(base_dir / "embeddings_bert_9753x384.npy")
    track_ids = np.load(base_dir / "track_ids_9753.npy", allow_pickle=True)

    with open(base_dir / "similarity_index.pkl", 'rb') as f:
        similarity_index = pickle.load(f)

    return {
        "embeddings": embeddings,
        "track_ids": track_ids,
        "similarity_index": similarity_index,
        "n_total": len(track_ids),
        "n_valid": np.sum(np.any(embeddings != 0, axis=1)),
        "dimensions": embeddings.shape[1]
    }

def load_valid_embeddings():
    """Carga solo embeddings validos (no-zero)."""
    data = load_complete_vectorization()

    # Filtrar embeddings validos
    valid_mask = np.any(data["embeddings"] != 0, axis=1)

    return {
        "embeddings": data["embeddings"][valid_mask],
        "track_ids": data["track_ids"][valid_mask],
        "n_valid": np.sum(valid_mask),
        "n_filtered": np.sum(~valid_mask)
    }

def find_similar_songs(track_id, n_neighbors=10):
    """Encuentra canciones similares a una dada."""
    data = load_complete_vectorization()

    # Encontrar indice del track
    track_idx = np.where(data["track_ids"] == track_id)[0]
    if len(track_idx) == 0:
        return None

    # Verificar que el embedding sea valido
    if np.all(data["embeddings"][track_idx[0]] == 0):
        print(f"ADVERTENCIA: Track {track_id} tiene embedding invalido (zero vector)")
        return None

    # Buscar similares
    query_embedding = data["embeddings"][track_idx[0]:track_idx[0]+1]
    distances, indices = data["similarity_index"]["model"].kneighbors(
        query_embedding, n_neighbors=n_neighbors+1
    )

    # Retornar resultados (skip el primero que es la misma cancion)
    similar_tracks = data["track_ids"][indices[0][1:]]
    similar_distances = distances[0][1:]

    return list(zip(similar_tracks, similar_distances))

if __name__ == "__main__":
    # Demo de uso
    print("=== Carga de Vectorizacion BERT ===\n")

    data = load_complete_vectorization()
    print(f"Total canciones: {data['n_total']}")
    print(f"Embeddings validos: {data['n_valid']}")
    print(f"Embeddings fallidos: {data['n_total'] - data['n_valid']}")
    print(f"Dimensiones: {data['dimensions']}")

    # Cargar solo validos
    print("\n=== Embeddings Validos ===")
    valid_data = load_valid_embeddings()
    print(f"Cargados: {valid_data['n_valid']} embeddings")
    print(f"Filtrados: {valid_data['n_filtered']} (zero vectors)")

    # Test similitud con un track valido
    print("\n=== Test de Similitud ===")
    # Buscar un track con embedding valido
    valid_mask = np.any(data["embeddings"] != 0, axis=1)
    valid_tracks = data["track_ids"][valid_mask]

    if len(valid_tracks) > 0:
        sample_track = valid_tracks[0]
        similar = find_similar_songs(sample_track, 5)
        if similar:
            print(f"Canciones similares a {sample_track}:")
            for track, distance in similar:
                print(f"  {track}: {distance:.4f}")
