"""
Modulo de algoritmos de clustering (Etapa 4, Paso 2).

Wrapper fino sobre scikit-learn y hdbscan que ejecuta un algoritmo
y devuelve las asignaciones. Funciones puras, sin evaluacion ni I/O.

Nota sobre distancia coseno y datos L2-normalizados:
Para datos en la hiperesfera unitaria (||x||=1), la distancia euclidiana
y la distancia coseno son equivalentes:
    ||a - b||^2 = 2 - 2*cos(a, b)  cuando ||a|| = ||b|| = 1
Por lo tanto, K-Means y Ward con distancia euclidiana son correctos
para el espacio semantico L2-normalizado.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from src.config import (
    CLUSTERING_K_RANGE,
    HDBSCAN_MIN_CLUSTER_SIZES,
    MIN_CLUSTER_PCT,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Resultado de un algoritmo de clustering."""

    algorithm: str  # "kmeans", "ward", "hdbscan"
    space_name: str  # "semantic_384d", "musical_13d", "semantic_umap"
    k: int  # clusters pedidos (para HDBSCAN: clusters encontrados)
    labels: np.ndarray = field(repr=False)  # [N] asignaciones (-1 = noise)
    n_clusters_found: int = 0
    n_noise: int = 0  # puntos marcados como noise (solo HDBSCAN)
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    has_degenerate_clusters: bool = False  # algun cluster < MIN_CLUSTER_PCT
    elapsed_seconds: float = 0.0
    params: Dict = field(default_factory=dict)


def _compute_cluster_info(
    labels: np.ndarray, n_total: int,
) -> tuple:
    """Calcula tamanos de clusters y detecta clusters degenerados."""
    unique_labels = np.unique(labels[labels >= 0])  # excluir noise (-1)
    sizes = {int(c): int((labels == c).sum()) for c in unique_labels}
    n_clusters = len(unique_labels)
    n_noise = int((labels == -1).sum())
    has_degenerate = any(
        count < n_total * MIN_CLUSTER_PCT for count in sizes.values()
    )
    return sizes, n_clusters, n_noise, has_degenerate


def run_kmeans(
    data: np.ndarray,
    k: int,
    space_name: str,
    random_state: int = RANDOM_SEED,
) -> ClusteringResult:
    """
    K-Means++ con distancia euclidiana.

    Parameters
    ----------
    data : np.ndarray
        Datos, shape [N, D].
    k : int
        Numero de clusters.
    space_name : str
        Nombre del espacio (para reporte).
    random_state : int
        Seed.

    Returns
    -------
    ClusteringResult
    """
    t0 = time.time()
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        max_iter=500,
        tol=1e-6,
        random_state=random_state,
    )
    labels = model.fit_predict(data)
    elapsed = time.time() - t0

    sizes, n_found, n_noise, has_deg = _compute_cluster_info(labels, len(data))

    logger.info(
        "  K-Means++ k=%d on %s: %d clusters, %.1fs, degenerate=%s",
        k, space_name, n_found, elapsed, has_deg,
    )

    return ClusteringResult(
        algorithm="kmeans",
        space_name=space_name,
        k=k,
        labels=labels,
        n_clusters_found=n_found,
        n_noise=n_noise,
        cluster_sizes=sizes,
        has_degenerate_clusters=has_deg,
        elapsed_seconds=round(elapsed, 2),
        params={"init": "k-means++", "n_init": 20, "max_iter": 500},
    )


def run_ward(
    data: np.ndarray,
    k: int,
    space_name: str,
) -> ClusteringResult:
    """
    Clustering jerarquico aglomerativo con criterio de Ward.

    Usa Ward2 (Lance-Williams) via sklearn AgglomerativeClustering.
    Solo admite distancia euclidiana (parte de la definicion del criterio).

    Parameters
    ----------
    data : np.ndarray
        Datos, shape [N, D].
    k : int
        Numero de clusters.
    space_name : str
        Nombre del espacio.

    Returns
    -------
    ClusteringResult
    """
    t0 = time.time()
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    )
    labels = model.fit_predict(data)
    elapsed = time.time() - t0

    sizes, n_found, n_noise, has_deg = _compute_cluster_info(labels, len(data))

    logger.info(
        "  Ward k=%d on %s: %d clusters, %.1fs, degenerate=%s",
        k, space_name, n_found, elapsed, has_deg,
    )

    return ClusteringResult(
        algorithm="ward",
        space_name=space_name,
        k=k,
        labels=labels,
        n_clusters_found=n_found,
        n_noise=n_noise,
        cluster_sizes=sizes,
        has_degenerate_clusters=has_deg,
        elapsed_seconds=round(elapsed, 2),
        params={"linkage": "ward"},
    )


def run_hdbscan(
    data: np.ndarray,
    min_cluster_size: int,
    space_name: str,
    metric: str = "euclidean",
) -> ClusteringResult:
    """
    HDBSCAN con min_cluster_size configurable.

    Parameters
    ----------
    data : np.ndarray
        Datos, shape [N, D].
    min_cluster_size : int
        Tamano minimo de cluster.
    space_name : str
        Nombre del espacio.
    metric : str
        Metrica de distancia.

    Returns
    -------
    ClusteringResult
    """
    from sklearn.cluster import HDBSCAN

    t0 = time.time()
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric,
        n_jobs=1,
    )
    labels = model.fit_predict(data)
    elapsed = time.time() - t0

    sizes, n_found, n_noise, has_deg = _compute_cluster_info(labels, len(data))

    noise_pct = round(100.0 * n_noise / len(data), 1)
    logger.info(
        "  HDBSCAN min_size=%d on %s: %d clusters, %d noise (%.1f%%), "
        "%.1fs, degenerate=%s",
        min_cluster_size, space_name, n_found, n_noise, noise_pct,
        elapsed, has_deg,
    )

    return ClusteringResult(
        algorithm="hdbscan",
        space_name=space_name,
        k=n_found,
        labels=labels,
        n_clusters_found=n_found,
        n_noise=n_noise,
        cluster_sizes=sizes,
        has_degenerate_clusters=has_deg,
        elapsed_seconds=round(elapsed, 2),
        params={
            "min_cluster_size": min_cluster_size,
            "metric": metric,
        },
    )


def run_all_algorithms(
    data: np.ndarray,
    space_name: str,
    k_range: Optional[List[int]] = None,
    hdbscan_min_cluster_sizes: Optional[List[int]] = None,
    hdbscan_metric: str = "euclidean",
    random_state: int = RANDOM_SEED,
) -> List[ClusteringResult]:
    """
    Ejecuta K-Means, Ward y HDBSCAN sobre un espacio.

    K-Means y Ward se ejecutan para cada k en k_range.
    HDBSCAN se ejecuta para cada min_cluster_size.

    Parameters
    ----------
    data : np.ndarray
        Datos, shape [N, D].
    space_name : str
        Nombre del espacio.
    k_range : list of int, optional
        Rango de K para K-Means y Ward. Default: CLUSTERING_K_RANGE.
    hdbscan_min_cluster_sizes : list of int, optional
        Tamanos minimos para HDBSCAN. Default: HDBSCAN_MIN_CLUSTER_SIZES.
    hdbscan_metric : str
        Metrica para HDBSCAN.
    random_state : int
        Seed para K-Means.

    Returns
    -------
    list of ClusteringResult
        Todos los resultados (incluyendo configuraciones degeneradas).
    """
    if k_range is None:
        k_range = CLUSTERING_K_RANGE
    if hdbscan_min_cluster_sizes is None:
        hdbscan_min_cluster_sizes = HDBSCAN_MIN_CLUSTER_SIZES

    results = []

    logger.info(
        "Ejecutando algoritmos sobre %s: shape=%s", space_name, data.shape,
    )

    # K-Means para cada K
    logger.info("  --- K-Means++ (K=%s) ---", k_range)
    for k in k_range:
        results.append(run_kmeans(data, k, space_name, random_state))

    # Ward para cada K
    logger.info("  --- Ward (K=%s) ---", k_range)
    for k in k_range:
        results.append(run_ward(data, k, space_name))

    # HDBSCAN para cada min_cluster_size
    logger.info(
        "  --- HDBSCAN (min_cluster_size=%s) ---",
        hdbscan_min_cluster_sizes,
    )
    for mcs in hdbscan_min_cluster_sizes:
        results.append(
            run_hdbscan(data, mcs, space_name, metric=hdbscan_metric)
        )

    n_valid = sum(1 for r in results if not r.has_degenerate_clusters)
    n_degenerate = len(results) - n_valid
    logger.info(
        "  Total: %d configuraciones (%d validas, %d degeneradas)",
        len(results), n_valid, n_degenerate,
    )

    return results
