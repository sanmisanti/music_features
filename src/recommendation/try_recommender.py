"""
Script interactivo para probar el recomendador (Etapa 5).

Ejecutar con: python -m src.recommendation.try_recommender

Permite buscar canciones por nombre/artista y obtener recomendaciones
al alpha optimo (0.80) o a cualquier alpha personalizado.
"""

import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.config import (
    NUMPY_SEED,
    RECOMMENDATION_TOP_K,
    UNIFIED_DATASET,
)
from src.recommendation.engine import (
    compute_sigma,
    precompute_similarities,
    recommend_all,
)


def search_songs(track_names, track_artists, query, max_results=20):
    """Busca canciones por nombre o artista (case-insensitive)."""
    query_lower = query.lower()
    matches = []
    for i, (name, artist) in enumerate(zip(track_names, track_artists)):
        name_str = str(name).lower()
        artist_str = str(artist).lower()
        if query_lower in name_str or query_lower in artist_str:
            matches.append((i, str(name), str(artist)))
            if len(matches) >= max_results:
                break
    return matches


def print_recommendation(rec, track_names, track_artists, genre_labels):
    """Imprime una recomendacion legible."""
    print(f"\n{'='*80}")
    print(f"Query: \"{rec.query_track_id}\"")
    print(f"  Nombre: {track_names[rec.query_idx]}")
    print(f"  Artista: {track_artists[rec.query_idx]}")
    print(f"  Genero: {rec.query_genre}")
    print(f"{'='*80}")
    print(f"{'#':<3} {'Score':<8} {'Sem':<8} {'Mus':<8} {'Genero':<8} {'Cancion':<45} {'Artista':<25}")
    print(f"{'-'*80}")
    for rank, (idx, score, s_sem, s_mus) in enumerate(zip(
        rec.recommended_indices, rec.scores,
        rec.scores_semantic, rec.scores_musical,
    ), start=1):
        name = str(track_names[idx])[:43]
        artist = str(track_artists[idx])[:23]
        same_genre = "OK" if rec.recommended_genres[rank-1] == rec.query_genre else "--"
        print(f"{rank:<3} {score:.4f}  {s_sem:.4f}  {s_mus:.4f}  "
              f"{str(rec.recommended_genres[rank-1]):<7} "
              f"{same_genre} {name:<43} {artist:<25}")


def main():
    np.random.seed(NUMPY_SEED)

    print("Cargando dataset unificado...")
    data = np.load(str(UNIFIED_DATASET), allow_pickle=True)
    semantic_embeddings = data["semantic_embeddings"]
    musical_features = data["musical_features"]
    genre_labels = data["genre_labels"]
    track_ids = data["track_ids"]
    track_names = data["track_names"]
    track_artists = data["track_artists"]

    n_songs = len(semantic_embeddings)
    print(f"Dataset: {n_songs} canciones cargadas.")

    print("Calculando sigma (puede tardar ~5 seg)...")
    sigma = compute_sigma(musical_features)
    print(f"Sigma = {sigma:.4f}")

    print("Pre-computando matrices de similitud (puede tardar ~15 seg)...")
    sim_semantic, sim_musical = precompute_similarities(
        semantic_embeddings, musical_features, sigma,
    )
    print("Listo.\n")

    alpha = 0.80  # optimo por grid search
    top_k = RECOMMENDATION_TOP_K

    print("="*80)
    print("RECOMENDADOR INTERACTIVO")
    print("="*80)
    print(f"Alpha actual: {alpha:.2f} (optimo)")
    print(f"Top-K: {top_k}")
    print("Comandos:")
    print("  buscar <query>  - busca canciones por nombre o artista")
    print("  rec <indice>    - recomienda para la cancion en ese indice")
    print("  alpha <valor>   - cambia alpha (0.0 a 1.0)")
    print("  k <valor>       - cambia top_k")
    print("  salir           - terminar")
    print("="*80)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "salir":
            break

        elif cmd == "buscar":
            if len(parts) < 2:
                print("Uso: buscar <query>")
                continue
            query = parts[1]
            matches = search_songs(track_names, track_artists, query)
            if not matches:
                print(f"Sin resultados para \"{query}\"")
                continue
            print(f"\n{len(matches)} resultados para \"{query}\":")
            for idx, name, artist in matches:
                genre = genre_labels[idx]
                print(f"  [{idx:>5}] {name[:50]:<50} - {artist[:25]:<25} ({genre})")

        elif cmd == "rec":
            if len(parts) < 2:
                print("Uso: rec <indice>")
                continue
            try:
                idx = int(parts[1])
                if idx < 0 or idx >= n_songs:
                    print(f"Indice fuera de rango (0 a {n_songs-1})")
                    continue
            except ValueError:
                print("El indice debe ser un numero entero")
                continue

            # Generar recomendaciones solo para este indice
            fused_row = alpha * sim_semantic[idx] + (1 - alpha) * sim_musical[idx]
            fused_row[idx] = -np.inf  # excluir self

            top_k_indices = np.argpartition(fused_row, -top_k)[-top_k:]
            candidate_scores = fused_row[top_k_indices]
            order = np.argsort(-candidate_scores)
            sorted_candidates = top_k_indices[order]
            sorted_scores = candidate_scores[order]

            # Armar un "result" simplificado
            class SimpleResult:
                pass
            rec = SimpleResult()
            rec.query_idx = idx
            rec.query_track_id = str(track_ids[idx])
            rec.query_genre = str(genre_labels[idx])
            rec.recommended_indices = sorted_candidates
            rec.recommended_genres = genre_labels[sorted_candidates]
            rec.scores = sorted_scores
            rec.scores_semantic = sim_semantic[idx, sorted_candidates]
            rec.scores_musical = sim_musical[idx, sorted_candidates]

            print_recommendation(rec, track_names, track_artists, genre_labels)

        elif cmd == "alpha":
            if len(parts) < 2:
                print(f"Alpha actual: {alpha:.2f}")
                continue
            try:
                new_alpha = float(parts[1])
                if not (0.0 <= new_alpha <= 1.0):
                    print("Alpha debe estar entre 0.0 y 1.0")
                    continue
                alpha = new_alpha
                print(f"Alpha = {alpha:.2f}")
            except ValueError:
                print("Alpha debe ser un numero")

        elif cmd == "k":
            if len(parts) < 2:
                print(f"Top-K actual: {top_k}")
                continue
            try:
                new_k = int(parts[1])
                if new_k < 1 or new_k > 100:
                    print("K debe estar entre 1 y 100")
                    continue
                top_k = new_k
                print(f"Top-K = {top_k}")
            except ValueError:
                print("K debe ser un numero entero")

        else:
            print(f"Comando desconocido: {cmd}")


if __name__ == "__main__":
    main()
