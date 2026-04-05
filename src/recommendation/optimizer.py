"""
Optimizacion de pesos de fusion (Etapa 5).

Grid search sobre alpha para encontrar el peso optimo de fusion
entre similitud semantica y musical. Se optimiza Precision@K con
genero como proxy de ground truth (declarado upfront).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.config import (
    RECOMMENDATION_ALPHA_RANGE,
    RECOMMENDATION_K_VALUES,
    RECOMMENDATION_TOP_K,
)
from src.recommendation.engine import recommend_all
from src.recommendation.evaluation import precision_at_k

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Resultado del grid search sobre alpha."""

    alpha_range: List[float]
    k_values: List[int]

    # Resultados: {alpha: {k: precision}}
    results: Dict[float, Dict[int, float]] = field(default_factory=dict)

    # Mejor configuracion
    best_alpha: float = 0.0
    best_precision: float = 0.0
    primary_k: int = RECOMMENDATION_TOP_K

    sigma: float = 0.0
    elapsed_seconds: float = 0.0


def grid_search_alpha(
    sim_semantic: np.ndarray,
    sim_musical: np.ndarray,
    track_ids: np.ndarray,
    genre_labels: np.ndarray,
    alpha_range: List[float] = None,
    k_values: List[int] = None,
    primary_k: int = RECOMMENDATION_TOP_K,
    sigma: float = 0.0,
) -> OptimizationResult:
    """
    Grid search sobre alpha para optimizar Precision@K.

    Pre-computa recomendaciones para cada alpha y evalua P@K
    para multiples valores de K. Selecciona el alpha que maximiza
    P@primary_k.

    Parameters
    ----------
    sim_semantic : np.ndarray
        Matriz de similitud semantica [N, N].
    sim_musical : np.ndarray
        Matriz de similitud musical [N, N].
    track_ids : np.ndarray
        IDs de canciones, shape [N].
    genre_labels : np.ndarray
        Etiquetas de genero, shape [N].
    alpha_range : list of float
        Valores de alpha a evaluar.
    k_values : list of int
        Valores de K para Precision@K.
    primary_k : int
        K principal para seleccion del mejor alpha.
    sigma : float
        Sigma utilizado (se almacena en resultado para referencia).

    Returns
    -------
    OptimizationResult
    """
    if alpha_range is None:
        alpha_range = RECOMMENDATION_ALPHA_RANGE
    if k_values is None:
        k_values = RECOMMENDATION_K_VALUES

    max_k = max(k_values)
    n_alphas = len(alpha_range)

    logger.info(
        "Grid search: %d valores de alpha, K=%s, primary_k=%d",
        n_alphas, k_values, primary_k,
    )

    start_time = time.time()
    all_results: Dict[float, Dict[int, float]] = {}
    best_alpha = 0.0
    best_precision = -1.0

    for i, alpha in enumerate(alpha_range):
        # Generar recomendaciones con este alpha
        recs = recommend_all(
            sim_semantic, sim_musical, alpha, max_k,
            track_ids, genre_labels,
        )

        # Evaluar P@K para cada K
        precisions_by_k = {}
        for k in k_values:
            mean_p, _, _ = precision_at_k(recs, k)
            precisions_by_k[k] = round(mean_p, 6)

        all_results[alpha] = precisions_by_k

        # Actualizar mejor
        p_primary = precisions_by_k[primary_k]
        if p_primary > best_precision:
            best_precision = p_primary
            best_alpha = alpha

        if (i + 1) % 5 == 0 or i == 0 or i == n_alphas - 1:
            logger.info(
                "  alpha=%.2f: P@%d=%.4f %s [%d/%d]",
                alpha, primary_k, p_primary,
                "<-- mejor" if alpha == best_alpha and i > 0 else "",
                i + 1, n_alphas,
            )

    elapsed = time.time() - start_time

    logger.info(
        "Grid search completado en %.1f s. Mejor alpha=%.2f, P@%d=%.4f",
        elapsed, best_alpha, primary_k, best_precision,
    )

    return OptimizationResult(
        alpha_range=alpha_range,
        k_values=k_values,
        results=all_results,
        best_alpha=best_alpha,
        best_precision=round(best_precision, 6),
        primary_k=primary_k,
        sigma=sigma,
        elapsed_seconds=round(elapsed, 1),
    )
