"""
Modulo de purificacion post-clustering (Etapa 4, Paso 4).

Identifica y elimina puntos mal asignados a su cluster para mejorar
la cohesion interna. Dos estrategias complementarias:
1. Silhouette negativo: el punto esta mas cerca de otro cluster.
2. Distancia al centroide: el punto es outlier dentro de su cluster.

La estrategia combinada aplica la union de ambos criterios, con un
cap configurable en el porcentaje de remocion (default 15%).

La purificacion post-clustering no tiene framework formal en la
literatura (Gan & Ng 2017, Dinh & Huynh 2021). Este modulo formaliza
un procedimiento reproducible con metricas pre/post.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_samples,
)

from src.config import (
    PURIFICATION_MAX_REMOVAL_PCT,
    PURIFICATION_SIGMA_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class PurificationReport:
    """Reporte de purificacion de un clustering."""

    algorithm: str
    space_name: str
    k: int
    strategy: str  # "negative_silhouette", "centroid_distance", "combined"

    # Pre-purificacion
    n_samples_before: int = 0
    silhouette_macro_before: float = 0.0
    silhouette_micro_before: float = 0.0
    davies_bouldin_before: float = 0.0
    cluster_sizes_before: Dict[int, int] = field(default_factory=dict)

    # Post-purificacion
    n_samples_after: int = 0
    n_removed: int = 0
    removal_pct: float = 0.0
    silhouette_macro_after: float = 0.0
    silhouette_micro_after: float = 0.0
    davies_bouldin_after: float = 0.0
    cluster_sizes_after: Dict[int, int] = field(default_factory=dict)

    # Mejoras
    silhouette_improvement: float = 0.0  # absoluto
    silhouette_improvement_pct: float = 0.0  # porcentual
    db_improvement: float = 0.0  # absoluto (negativo = mejora)
    within_removal_budget: bool = True

    # Indices removidos
    removed_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), repr=False)


def _compute_silhouette_stats(
    data: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> Tuple[float, float, Dict[int, float], np.ndarray]:
    """Calcula Silhouette micro, macro, per-cluster y per-sample."""
    sil_samples = silhouette_samples(data, labels, metric=metric)
    sil_micro = float(sil_samples.mean())

    unique_labels = np.unique(labels[labels >= 0])
    per_cluster = {}
    for c in unique_labels:
        per_cluster[int(c)] = float(sil_samples[labels == c].mean())
    sil_macro = float(np.mean(list(per_cluster.values())))

    return sil_micro, sil_macro, per_cluster, sil_samples


def purify_negative_silhouette(
    data: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifica puntos con silhouette individual < 0.

    Un punto con silhouette negativo esta, en promedio, mas cerca de
    los puntos de otro cluster que de los de su propio cluster.

    Parameters
    ----------
    data : np.ndarray
        Datos, shape [N, D].
    labels : np.ndarray
        Asignaciones de cluster, shape [N].
    metric : str
        Metrica de distancia.

    Returns
    -------
    mask_keep : np.ndarray
        Boolean mask, True = mantener, shape [N].
    sil_per_point : np.ndarray
        Silhouette por punto, shape [N].
    """
    sil_per_point = silhouette_samples(data, labels, metric=metric)
    mask_keep = sil_per_point >= 0
    n_negative = int((~mask_keep).sum())
    logger.info(
        "  Silhouette negativo: %d puntos flagged (%.1f%%)",
        n_negative, 100.0 * n_negative / len(data),
    )
    return mask_keep, sil_per_point


def purify_centroid_distance(
    data: np.ndarray,
    labels: np.ndarray,
    sigma_threshold: float = PURIFICATION_SIGMA_THRESHOLD,
) -> np.ndarray:
    """
    Identifica puntos cuya distancia al centroide de su cluster excede
    sigma_threshold * std del cluster.

    Parameters
    ----------
    data : np.ndarray
        Datos, shape [N, D].
    labels : np.ndarray
        Asignaciones de cluster, shape [N].
    sigma_threshold : float
        Umbral en desviaciones estandar.

    Returns
    -------
    mask_keep : np.ndarray
        Boolean mask, True = mantener, shape [N].
    """
    mask_keep = np.ones(len(data), dtype=bool)
    unique_labels = np.unique(labels[labels >= 0])
    total_flagged = 0

    for c in unique_labels:
        cluster_mask = labels == c
        cluster_data = data[cluster_mask]
        centroid = cluster_data.mean(axis=0)
        distances = np.linalg.norm(cluster_data - centroid, axis=1)
        threshold = distances.mean() + sigma_threshold * distances.std()
        outlier_mask = distances > threshold
        n_flagged = int(outlier_mask.sum())
        total_flagged += n_flagged

        # Aplicar al mask global
        cluster_indices = np.where(cluster_mask)[0]
        mask_keep[cluster_indices[outlier_mask]] = False

    logger.info(
        "  Distancia centroide (> %.1f sigma): %d puntos flagged (%.1f%%)",
        sigma_threshold, total_flagged, 100.0 * total_flagged / len(data),
    )
    return mask_keep


def purify_combined(
    data: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    sigma_threshold: float = PURIFICATION_SIGMA_THRESHOLD,
    max_removal_pct: float = PURIFICATION_MAX_REMOVAL_PCT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrategia combinada: union de puntos flagged por silhouette negativo
    y distancia al centroide. Si la remocion excede max_removal_pct,
    prioriza los puntos con silhouette mas negativo hasta el umbral.

    Parameters
    ----------
    data : np.ndarray
        Datos, shape [N, D].
    labels : np.ndarray
        Asignaciones de cluster, shape [N].
    metric : str
        Metrica de distancia para silhouette.
    sigma_threshold : float
        Umbral para distancia al centroide.
    max_removal_pct : float
        Porcentaje maximo de remocion (0-1).

    Returns
    -------
    mask_keep : np.ndarray
        Boolean mask, True = mantener, shape [N].
    sil_per_point : np.ndarray
        Silhouette por punto, shape [N].
    """
    n_total = len(data)
    max_removal = int(n_total * max_removal_pct)

    # Obtener flags de ambas estrategias
    mask_sil, sil_per_point = purify_negative_silhouette(data, labels, metric)
    mask_centroid = purify_centroid_distance(data, labels, sigma_threshold)

    # Union: flagged si cualquiera de las dos lo marca
    mask_combined = mask_sil & mask_centroid
    n_flagged = int((~mask_combined).sum())

    logger.info(
        "  Combinado: %d puntos flagged (%.1f%%)",
        n_flagged, 100.0 * n_flagged / n_total,
    )

    if n_flagged <= max_removal:
        logger.info(
            "  Dentro del presupuesto de remocion (max %d = %.0f%%)",
            max_removal, max_removal_pct * 100,
        )
        return mask_combined, sil_per_point

    # Excede el presupuesto: priorizar los de silhouette mas negativo
    logger.info(
        "  Excede presupuesto (%d > %d). Priorizando por silhouette...",
        n_flagged, max_removal,
    )
    flagged_indices = np.where(~mask_combined)[0]
    flagged_sils = sil_per_point[flagged_indices]

    # Ordenar por silhouette ascendente (los mas negativos primero)
    priority_order = np.argsort(flagged_sils)
    indices_to_remove = flagged_indices[priority_order[:max_removal]]

    mask_keep = np.ones(n_total, dtype=bool)
    mask_keep[indices_to_remove] = False

    logger.info(
        "  Remocion ajustada: %d puntos (%.1f%%)",
        max_removal, 100.0 * max_removal / n_total,
    )

    return mask_keep, sil_per_point


def run_purification(
    data: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    space_name: str,
    k: int,
    metric: str = "euclidean",
    sigma_threshold: float = PURIFICATION_SIGMA_THRESHOLD,
    max_removal_pct: float = PURIFICATION_MAX_REMOVAL_PCT,
) -> PurificationReport:
    """
    Ejecuta purificacion combinada y calcula metricas pre/post.

    Parameters
    ----------
    data : np.ndarray
        Datos, shape [N, D].
    labels : np.ndarray
        Asignaciones de cluster, shape [N]. -1 = noise (excluidos).
    algorithm : str
        Nombre del algoritmo.
    space_name : str
        Nombre del espacio.
    k : int
        Numero de clusters.
    metric : str
        Metrica de distancia.
    sigma_threshold : float
        Umbral para distancia al centroide.
    max_removal_pct : float
        Porcentaje maximo de remocion.

    Returns
    -------
    PurificationReport
    """
    # Excluir noise points para purificacion
    assigned_mask = labels >= 0
    data_assigned = data[assigned_mask]
    labels_assigned = labels[assigned_mask]
    n_assigned = len(data_assigned)

    logger.info(
        "Purificacion: %s %s k=%d, %d puntos asignados (excluidos %d noise)",
        algorithm, space_name, k, n_assigned, int((~assigned_mask).sum()),
    )

    # --- Metricas PRE-purificacion ---
    sil_micro_before, sil_macro_before, _, _ = _compute_silhouette_stats(
        data_assigned, labels_assigned, metric,
    )
    db_before = float(davies_bouldin_score(data_assigned, labels_assigned))
    sizes_before = {
        int(c): int((labels_assigned == c).sum())
        for c in np.unique(labels_assigned)
    }

    logger.info(
        "  Pre-purificacion: Sil_macro=%.4f, Sil_micro=%.4f, DB=%.4f",
        sil_macro_before, sil_micro_before, db_before,
    )

    # --- Purificacion combinada ---
    mask_keep, _ = purify_combined(
        data_assigned, labels_assigned, metric,
        sigma_threshold, max_removal_pct,
    )

    data_purified = data_assigned[mask_keep]
    labels_purified = labels_assigned[mask_keep]
    n_removed = n_assigned - len(data_purified)
    removal_pct = n_removed / n_assigned if n_assigned > 0 else 0.0

    # Indices removidos (relativos al array original con noise)
    assigned_indices = np.where(assigned_mask)[0]
    removed_indices = assigned_indices[~mask_keep]

    # --- Metricas POST-purificacion ---
    unique_post = np.unique(labels_purified)
    if len(unique_post) < 2:
        logger.warning(
            "  Post-purificacion: menos de 2 clusters, metricas no calculables",
        )
        return PurificationReport(
            algorithm=algorithm, space_name=space_name, k=k,
            strategy="combined",
            n_samples_before=n_assigned,
            silhouette_macro_before=round(sil_macro_before, 6),
            silhouette_micro_before=round(sil_micro_before, 6),
            davies_bouldin_before=round(db_before, 6),
            cluster_sizes_before=sizes_before,
            n_samples_after=len(data_purified),
            n_removed=n_removed,
            removal_pct=round(removal_pct, 6),
            within_removal_budget=removal_pct <= max_removal_pct,
            removed_indices=removed_indices,
        )

    sil_micro_after, sil_macro_after, _, _ = _compute_silhouette_stats(
        data_purified, labels_purified, metric,
    )
    db_after = float(davies_bouldin_score(data_purified, labels_purified))
    sizes_after = {
        int(c): int((labels_purified == c).sum())
        for c in np.unique(labels_purified)
    }

    sil_improvement = sil_macro_after - sil_macro_before
    sil_improvement_pct = (
        100.0 * sil_improvement / abs(sil_macro_before)
        if sil_macro_before != 0 else 0.0
    )
    db_improvement = db_after - db_before  # negativo = mejora

    logger.info(
        "  Post-purificacion: Sil_macro=%.4f, Sil_micro=%.4f, DB=%.4f",
        sil_macro_after, sil_micro_after, db_after,
    )
    logger.info(
        "  Remocion: %d puntos (%.1f%%), mejora Sil_macro: %+.4f (%+.1f%%)",
        n_removed, removal_pct * 100, sil_improvement, sil_improvement_pct,
    )

    return PurificationReport(
        algorithm=algorithm,
        space_name=space_name,
        k=k,
        strategy="combined",
        n_samples_before=n_assigned,
        silhouette_macro_before=round(sil_macro_before, 6),
        silhouette_micro_before=round(sil_micro_before, 6),
        davies_bouldin_before=round(db_before, 6),
        cluster_sizes_before=sizes_before,
        n_samples_after=len(data_purified),
        n_removed=n_removed,
        removal_pct=round(removal_pct, 6),
        silhouette_macro_after=round(sil_macro_after, 6),
        silhouette_micro_after=round(sil_micro_after, 6),
        davies_bouldin_after=round(db_after, 6),
        cluster_sizes_after=sizes_after,
        silhouette_improvement=round(sil_improvement, 6),
        silhouette_improvement_pct=round(sil_improvement_pct, 2),
        db_improvement=round(db_improvement, 6),
        within_removal_budget=removal_pct <= max_removal_pct,
        removed_indices=removed_indices,
    )
