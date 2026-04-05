"""
Evaluacion del sistema de recomendacion (Etapa 5).

Metricas implementadas:
- Precision@K con genero como proxy de ground truth (declarado upfront).
- Intra-List Diversity (ILD): diversidad dentro de cada lista de recomendaciones.
- Cobertura del catalogo: fraccion del dataset que aparece en alguna recomendacion.
- Cobertura por genero: fraccion por genero que aparece en recomendaciones.
- Distribucion de clusters: metrica descriptiva, no se usa para optimizacion.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.recommendation.engine import RecommendationResult

logger = logging.getLogger(__name__)


@dataclass
class RecommendationMetrics:
    """Metricas agregadas para una configuracion de recomendacion."""

    alpha: float
    top_k: int
    n_queries: int

    # Precision@K con genero como proxy
    precision_at_k: float = 0.0
    precision_at_k_std: float = 0.0
    precision_per_genre: Dict[str, float] = field(default_factory=dict)

    # Diversidad intra-lista
    ild_semantic: float = 0.0
    ild_semantic_std: float = 0.0
    ild_musical: float = 0.0
    ild_musical_std: float = 0.0

    # Cobertura
    catalog_coverage: float = 0.0
    genre_coverage: Dict[str, float] = field(default_factory=dict)

    # Distribucion de clusters (descriptiva)
    cluster_distribution: Dict[str, Dict[int, float]] = field(default_factory=dict)


def precision_at_k(
    results: List[RecommendationResult],
    k: int,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Precision@K usando genero como proxy de ground truth.

    Una recomendacion es "relevante" si comparte genero con la query.
    Metrica proxy declarada explicitamente en el informe.

    Parameters
    ----------
    results : list of RecommendationResult
    k : int
        Numero de recomendaciones a evaluar.

    Returns
    -------
    mean_precision : float
    std_precision : float
    per_genre : dict
        Precision media por genero de la query.
    """
    precisions = []
    genre_precisions: Dict[str, List[float]] = {}

    for r in results:
        # Tomar solo los primeros k
        rec_genres = r.recommended_genres[:k]
        n_relevant = int(np.sum(rec_genres == r.query_genre))
        p = n_relevant / k
        precisions.append(p)

        if r.query_genre not in genre_precisions:
            genre_precisions[r.query_genre] = []
        genre_precisions[r.query_genre].append(p)

    mean_p = float(np.mean(precisions))
    std_p = float(np.std(precisions))
    per_genre = {g: float(np.mean(ps)) for g, ps in genre_precisions.items()}

    return mean_p, std_p, per_genre


def intra_list_diversity(
    results: List[RecommendationResult],
    data: np.ndarray,
    metric: str = "cosine",
) -> Tuple[float, float]:
    """
    Intra-List Diversity (ILD): distancia pairwise media entre recomendaciones.

    Parameters
    ----------
    results : list of RecommendationResult
    data : np.ndarray
        Embeddings o features, shape [N, D].
    metric : str
        "cosine" para semantico (ILD = 1 - coseno),
        "euclidean" para musical.

    Returns
    -------
    mean_ild : float
    std_ild : float
    """
    ilds = []

    for r in results:
        rec_data = data[r.recommended_indices]
        k = len(rec_data)

        if k < 2:
            ilds.append(0.0)
            continue

        if metric == "cosine":
            # Para L2-normalizados: coseno = dot product
            sim_matrix = rec_data @ rec_data.T
            # ILD = 1 - similitud coseno (promedio de pares unicos)
            n_pairs = k * (k - 1) / 2
            # Sumar triangulo superior de (1 - sim)
            triu_indices = np.triu_indices(k, k=1)
            pairwise_dist = 1.0 - sim_matrix[triu_indices]
            ild = float(pairwise_dist.mean())
        else:
            # Euclidea
            from scipy.spatial.distance import pdist
            dists = pdist(rec_data, metric="euclidean")
            ild = float(dists.mean())

        ilds.append(ild)

    return float(np.mean(ilds)), float(np.std(ilds))


def catalog_coverage(
    results: List[RecommendationResult],
    n_total: int,
) -> float:
    """
    Fraccion del catalogo que aparece en al menos una recomendacion.

    Parameters
    ----------
    results : list of RecommendationResult
    n_total : int
        Tamano total del catalogo.

    Returns
    -------
    coverage : float
        Valor en [0, 1].
    """
    all_recommended = set()
    for r in results:
        all_recommended.update(r.recommended_indices.tolist())

    return len(all_recommended) / n_total


def genre_coverage(
    results: List[RecommendationResult],
    genre_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Cobertura por genero: fraccion de canciones de cada genero que
    aparecen en al menos una recomendacion.

    Parameters
    ----------
    results : list of RecommendationResult
    genre_labels : np.ndarray
        Etiquetas de genero, shape [N].

    Returns
    -------
    coverage : dict
        {genero: fraccion}.
    """
    all_recommended = set()
    for r in results:
        all_recommended.update(r.recommended_indices.tolist())

    unique_genres = sorted(set(genre_labels))
    coverage = {}
    for g in unique_genres:
        genre_indices = set(np.where(genre_labels == g)[0].tolist())
        recommended_from_genre = genre_indices & all_recommended
        coverage[str(g)] = len(recommended_from_genre) / len(genre_indices)

    return coverage


def cluster_distribution_in_recommendations(
    results: List[RecommendationResult],
    cluster_labels: np.ndarray,
    space_name: str,
) -> Dict[int, float]:
    """
    Distribucion de clusters entre los items recomendados (descriptiva).

    Parameters
    ----------
    results : list of RecommendationResult
    cluster_labels : np.ndarray
        Asignaciones de cluster, shape [N]. -1 = noise.
    space_name : str
        Nombre del espacio (para logging).

    Returns
    -------
    distribution : dict
        {cluster_id: fraccion}.
    """
    all_recommended = []
    for r in results:
        all_recommended.extend(r.recommended_indices.tolist())

    rec_labels = cluster_labels[all_recommended]
    unique, counts = np.unique(rec_labels, return_counts=True)
    total = len(rec_labels)

    distribution = {int(c): round(float(n) / total, 4) for c, n in zip(unique, counts)}

    logger.info(
        "Distribucion de clusters en recomendaciones (%s): %s",
        space_name, distribution,
    )

    return distribution


def evaluate_recommendations(
    results: List[RecommendationResult],
    alpha: float,
    top_k: int,
    semantic_embeddings: np.ndarray,
    musical_features: np.ndarray,
    genre_labels: np.ndarray,
    cluster_labels_sem: Optional[np.ndarray] = None,
    cluster_labels_mus: Optional[np.ndarray] = None,
) -> RecommendationMetrics:
    """
    Evaluacion completa de un conjunto de recomendaciones.

    Parameters
    ----------
    results : list of RecommendationResult
    alpha : float
        Peso semantico utilizado.
    top_k : int
        Numero de recomendaciones evaluadas.
    semantic_embeddings : np.ndarray
        Para calcular ILD semantica.
    musical_features : np.ndarray
        Para calcular ILD musical.
    genre_labels : np.ndarray
        Para Precision@K y cobertura por genero.
    cluster_labels_sem : np.ndarray, optional
        Labels de clustering semantico para distribucion descriptiva.
    cluster_labels_mus : np.ndarray, optional
        Labels de clustering musical para distribucion descriptiva.

    Returns
    -------
    RecommendationMetrics
    """
    n_total = len(semantic_embeddings)

    # Precision@K
    mean_p, std_p, per_genre = precision_at_k(results, top_k)

    # ILD
    ild_sem, ild_sem_std = intra_list_diversity(
        results, semantic_embeddings, metric="cosine",
    )
    ild_mus, ild_mus_std = intra_list_diversity(
        results, musical_features, metric="euclidean",
    )

    # Cobertura
    cov = catalog_coverage(results, n_total)
    g_cov = genre_coverage(results, genre_labels)

    # Clusters (descriptivo)
    cluster_dist = {}
    if cluster_labels_sem is not None:
        cluster_dist["semantic"] = cluster_distribution_in_recommendations(
            results, cluster_labels_sem, "semantic",
        )
    if cluster_labels_mus is not None:
        cluster_dist["musical"] = cluster_distribution_in_recommendations(
            results, cluster_labels_mus, "musical",
        )

    metrics = RecommendationMetrics(
        alpha=alpha,
        top_k=top_k,
        n_queries=len(results),
        precision_at_k=round(mean_p, 6),
        precision_at_k_std=round(std_p, 6),
        precision_per_genre={g: round(v, 6) for g, v in per_genre.items()},
        ild_semantic=round(ild_sem, 6),
        ild_semantic_std=round(ild_sem_std, 6),
        ild_musical=round(ild_mus, 6),
        ild_musical_std=round(ild_mus_std, 6),
        catalog_coverage=round(cov, 6),
        genre_coverage={g: round(v, 6) for g, v in g_cov.items()},
        cluster_distribution=cluster_dist,
    )

    logger.info(
        "Evaluacion alpha=%.2f K=%d: P@K=%.4f, ILD_sem=%.4f, "
        "ILD_mus=%.4f, cobertura=%.4f",
        alpha, top_k, mean_p, ild_sem, ild_mus, cov,
    )

    return metrics
