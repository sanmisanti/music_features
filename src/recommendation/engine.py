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
import re
import unicodedata
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

# Sobre-muestreo para deduplicacion: se recuperan mas candidatos que top_k
# para poder colapsar duplicados y rellenar hasta top_k canciones unicas.
DEDUP_OVERFETCH_FACTOR = 10
DEDUP_OVERFETCH_MIN = 100


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


def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparacion (minusculas, sin acentos ni puntuacion).

    Se usa para agrupar duplicados: dos entradas con el mismo nombre y artista
    normalizados se consideran la misma grabacion.
    """
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_duplicate_groups(
    track_names: np.ndarray,
    track_artists: np.ndarray,
) -> np.ndarray:
    """
    Asigna un identificador de grupo a cada cancion, de modo que las entradas
    con el mismo (nombre, artista) normalizados comparten identificador.

    El dataset fuente contiene la misma grabacion con `track_id` distintos
    (recolectada desde playlists diferentes). Agrupar por (nombre, artista)
    permite deduplicar la salida del recomendador sin colapsar variantes
    legitimas (versiones en vivo, remixes o traducciones tienen nombres
    distintos y quedan en grupos separados).

    Parameters
    ----------
    track_names : np.ndarray
        Nombres de las canciones, shape [N].
    track_artists : np.ndarray
        Artistas, shape [N].

    Returns
    -------
    groups : np.ndarray
        Identificadores de grupo, shape [N], dtype int64.
    """
    keys: dict = {}
    groups = np.empty(len(track_names), dtype=np.int64)
    for i in range(len(track_names)):
        key = normalize_text(track_names[i]) + "|" + normalize_text(track_artists[i])
        gid = keys.get(key)
        if gid is None:
            gid = len(keys)
            keys[key] = gid
        groups[i] = gid
    n_dupes = len(track_names) - len(keys)
    logger.info(
        "Grupos de duplicados: %d grabaciones unicas, %d entradas duplicadas "
        "sobre %d canciones",
        len(keys), n_dupes, len(track_names),
    )
    return groups


def _build_result(
    i: int,
    sorted_candidates: np.ndarray,
    fused: np.ndarray,
    sim_semantic: np.ndarray,
    sim_musical: np.ndarray,
    track_ids: np.ndarray,
    genre_labels: np.ndarray,
) -> "RecommendationResult":
    """Construye un RecommendationResult a partir de indices ya ordenados."""
    return RecommendationResult(
        query_idx=i,
        query_track_id=str(track_ids[i]),
        query_genre=str(genre_labels[i]),
        recommended_indices=sorted_candidates,
        recommended_track_ids=track_ids[sorted_candidates],
        recommended_genres=genre_labels[sorted_candidates],
        scores=fused[i, sorted_candidates],
        scores_semantic=sim_semantic[i, sorted_candidates],
        scores_musical=sim_musical[i, sorted_candidates],
    )


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
    dup_groups: Optional[np.ndarray] = None,
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
    dup_groups : np.ndarray, optional
        Identificadores de grupo de duplicados por cancion (ver
        `compute_duplicate_groups`). Si se provee, la lista de cada query se
        deduplica en post-procesamiento: se conserva una sola grabacion por
        grupo (la de mayor score) y se excluye el grupo de la propia query,
        rellenando con los siguientes candidatos hasta completar top_k.

    Returns
    -------
    results : list of RecommendationResult
    """
    n = len(sim_semantic)

    # Fusion
    fused = alpha * sim_semantic + (1 - alpha) * sim_musical

    # Excluir self-recommendation: poner diagonal a -inf
    np.fill_diagonal(fused, -np.inf)

    if dup_groups is None:
        # Sin deduplicacion: top_k directo por argpartition
        cand_indices = np.argpartition(fused, -top_k, axis=1)[:, -top_k:]
        results = []
        for i in range(n):
            candidates = cand_indices[i]
            sorted_candidates = candidates[np.argsort(-fused[i, candidates])]
            results.append(_build_result(
                i, sorted_candidates, fused, sim_semantic, sim_musical,
                track_ids, genre_labels,
            ))
        return results

    # Con deduplicacion: sobre-muestrear y colapsar por grupo
    over_k = min(n - 1, max(top_k * DEDUP_OVERFETCH_FACTOR, DEDUP_OVERFETCH_MIN))
    cand_indices = np.argpartition(fused, -over_k, axis=1)[:, -over_k:]

    results = []
    for i in range(n):
        candidates = cand_indices[i]
        candidates = candidates[np.argsort(-fused[i, candidates])]  # desc
        q_group = dup_groups[i]
        seen_groups = set()
        seen_sigs = set()
        picked = []
        for j in candidates:
            # Misma grabacion que la consulta (letra identica y audio
            # practicamente identico): excluir aunque tenga metadatos distintos.
            if sim_semantic[i, j] >= 0.9999 and sim_musical[i, j] >= 0.999:
                continue
            g = dup_groups[j]
            if g == q_group or g in seen_groups:
                continue
            # Grabacion identica con otro rotulo: misma firma de similitud en
            # ambas modalidades. Una variante legitima difiere en lo musical.
            sig = (round(float(sim_semantic[i, j]), 5),
                   round(float(sim_musical[i, j]), 5))
            if sig in seen_sigs:
                continue
            seen_groups.add(g)
            seen_sigs.add(sig)
            picked.append(j)
            if len(picked) == top_k:
                break
        sorted_candidates = np.asarray(picked, dtype=np.int64)
        results.append(_build_result(
            i, sorted_candidates, fused, sim_semantic, sim_musical,
            track_ids, genre_labels,
        ))

    return results
