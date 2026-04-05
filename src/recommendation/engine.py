"""
Motor de recomendacion hibrido (Etapa 5).

Implementa k-NN directo con fusion tardia de similitudes semanticas
(coseno) y musicales (euclidea con kernel gaussiano). El peso de
fusion alpha se optimiza externamente via grid search.

Metricas por espacio:
- Semantico (384D, L2-normalizado): similitud coseno = producto punto.
- Musical (13D, z-score): distancia euclidea convertida a similitud
  via kernel gaussiano con sigma = mediana de distancias pairwise.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist, pdist

from src.config import (
    NUMPY_SEED,
    RECOMMENDATION_SIGMA_SAMPLE_SIZE,
    RECOMMENDATION_TOP_K,
)

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Resultado de recomendacion para una cancion query."""

    query_idx: int
    query_track_id: str
    query_genre: str
    recommended_indices: np.ndarray      # [K]
    recommended_track_ids: np.ndarray    # [K]
    recommended_genres: np.ndarray       # [K]
    scores: np.ndarray                   # [K] scores de fusion, descendente
    scores_semantic: np.ndarray          # [K] componente semantica
    scores_musical: np.ndarray           # [K] componente musical


def compute_sigma(
    musical_features: np.ndarray,
    sample_size: int = RECOMMENDATION_SIGMA_SAMPLE_SIZE,
    seed: int = NUMPY_SEED,
) -> float:
    """
    Calcula sigma para el kernel gaussiano como mediana de distancias
    pairwise euclideas sobre una muestra aleatoria.

    Parameters
    ----------
    musical_features : np.ndarray
        Features musicales, shape [N, D].
    sample_size : int
        Tamano de la muestra (N^2 pairwise es inviable para N=17964).
    seed : int
        Seed para reproducibilidad de la muestra.

    Returns
    -------
    sigma : float
        Mediana de distancias pairwise.
    """
    n_total = len(musical_features)
    rng = np.random.RandomState(seed)

    if sample_size >= n_total:
        sample = musical_features
    else:
        indices = rng.choice(n_total, size=sample_size, replace=False)
        sample = musical_features[indices]

    distances = pdist(sample, metric="euclidean")
    sigma = float(np.median(distances))

    logger.info(
        "Sigma calculado: %.4f (mediana de %d distancias pairwise, "
        "muestra de %d canciones)",
        sigma, len(distances), len(sample),
    )

    return sigma


def gaussian_kernel(distances: np.ndarray, sigma: float) -> np.ndarray:
    """
    Convierte distancias euclideas a similitudes via kernel gaussiano.

    Parameters
    ----------
    distances : np.ndarray
        Distancias euclideas (>= 0).
    sigma : float
        Ancho de banda del kernel.

    Returns
    -------
    similarities : np.ndarray
        Valores en (0, 1], donde 1 = distancia 0.
    """
    return np.exp(-distances ** 2 / (2 * sigma ** 2))


def precompute_similarities(
    semantic_embeddings: np.ndarray,
    musical_features: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-computa matrices completas de similitud para ambas modalidades.

    Parameters
    ----------
    semantic_embeddings : np.ndarray
        Embeddings semanticos L2-normalizados, shape [N, 384].
    musical_features : np.ndarray
        Features musicales z-score, shape [N, 13].
    sigma : float
        Sigma del kernel gaussiano.

    Returns
    -------
    sim_semantic : np.ndarray
        Similitudes coseno [N, N], float32.
    sim_musical : np.ndarray
        Similitudes gaussianas [N, N], float32.
    """
    n = len(semantic_embeddings)

    # Semantico: coseno = dot product para vectores L2-normalizados
    logger.info("Calculando matriz de similitud semantica (%d x %d)...", n, n)
    sim_semantic = (semantic_embeddings @ semantic_embeddings.T).astype(np.float32)

    # Musical: distancia euclidea -> kernel gaussiano
    logger.info("Calculando matriz de distancias musicales (%d x %d)...", n, n)
    dist_musical = cdist(musical_features, musical_features, metric="euclidean")
    sim_musical = gaussian_kernel(dist_musical, sigma).astype(np.float32)
    del dist_musical  # liberar memoria

    # Verificaciones
    logger.info(
        "  Sim semantica: min=%.4f, max=%.4f, media=%.4f",
        sim_semantic.min(), sim_semantic.max(), sim_semantic.mean(),
    )
    logger.info(
        "  Sim musical: min=%.4f, max=%.4f, media=%.4f",
        sim_musical.min(), sim_musical.max(), sim_musical.mean(),
    )

    return sim_semantic, sim_musical


def recommend_all(
    sim_semantic: np.ndarray,
    sim_musical: np.ndarray,
    alpha: float,
    top_k: int,
    track_ids: np.ndarray,
    genre_labels: np.ndarray,
) -> List[RecommendationResult]:
    """
    Genera recomendaciones para todas las canciones del dataset.

    Parameters
    ----------
    sim_semantic : np.ndarray
        Matriz de similitud semantica [N, N].
    sim_musical : np.ndarray
        Matriz de similitud musical [N, N].
    alpha : float
        Peso de la componente semantica (0 = solo musical, 1 = solo semantico).
    top_k : int
        Numero de recomendaciones por query.
    track_ids : np.ndarray
        IDs de canciones, shape [N].
    genre_labels : np.ndarray
        Etiquetas de genero, shape [N].

    Returns
    -------
    results : list of RecommendationResult
    """
    n = len(sim_semantic)

    # Fusion
    fused = alpha * sim_semantic + (1 - alpha) * sim_musical

    # Excluir self-recommendation: poner diagonal a -inf
    np.fill_diagonal(fused, -np.inf)

    # Top-K por argpartition (O(N) per row, mas eficiente que argsort completo)
    # argpartition devuelve indices de los top_k mayores, sin orden
    top_k_indices = np.argpartition(fused, -top_k, axis=1)[:, -top_k:]

    results = []
    for i in range(n):
        candidates = top_k_indices[i]
        candidate_scores = fused[i, candidates]

        # Ordenar los top_k candidatos por score descendente
        order = np.argsort(-candidate_scores)
        sorted_candidates = candidates[order]
        sorted_scores = candidate_scores[order]

        results.append(RecommendationResult(
            query_idx=i,
            query_track_id=str(track_ids[i]),
            query_genre=str(genre_labels[i]),
            recommended_indices=sorted_candidates,
            recommended_track_ids=track_ids[sorted_candidates],
            recommended_genres=genre_labels[sorted_candidates],
            scores=sorted_scores,
            scores_semantic=sim_semantic[i, sorted_candidates],
            scores_musical=sim_musical[i, sorted_candidates],
        ))

    return results
