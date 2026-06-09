"""
Modulo de evaluacion de clustering (Etapa 4, Paso 2).

Calcula metricas internas (Silhouette, Davies-Bouldin, Calinski-Harabasz)
y externas (ARI, NMI contra genero como proxy) para cada configuracion
de clustering. Funciones puras, sin I/O.

Silhouette macro-averaged: promedio de la silhouette media de cada cluster,
ponderando cada cluster igualmente. Evita que un cluster dominante oculte
la mala calidad de clusters pequenos. Se usa como metrica primaria de
seleccion segun Chicco et al. (2025).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusteringMetrics:
    """Metricas de evaluacion para una configuracion de clustering."""

    algorithm: str
    space_name: str
    k: int
    n_samples: int
    n_clusters_found: int
    n_noise: int
    has_degenerate_clusters: bool
    cluster_sizes: Dict[int, int]

    # Metricas internas
    silhouette_micro: float = 0.0  # promedio por punto (sklearn default)
    silhouette_macro: float = 0.0  # promedio por cluster (metrica primaria)
    davies_bouldin: float = 0.0  # menor = mejor
    calinski_harabasz: float = 0.0  # mayor = mejor

    # Metricas externas (proxy: genero)
    ari_genre: float = 0.0
    nmi_genre: float = 0.0

    # Diagnostico per-cluster
    silhouette_per_cluster: Dict[int, float] = field(default_factory=dict)

    # Parametros del algoritmo
    params: Dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def evaluate_clustering(
    data: np.ndarray,
    labels: np.ndarray,
    genre_labels: np.ndarray,
    algorithm: str,
    space_name: str,
    k: int,
    n_noise: int = 0,
    has_degenerate_clusters: bool = False,
    cluster_sizes: Optional[Dict[int, int]] = None,
    params: Optional[Dict] = None,
    elapsed_seconds: float = 0.0,
    metric: str = "euclidean",
) -> ClusteringMetrics:
    """
    Calcula todas las metricas para una configuracion de clustering.

    Para configuraciones con noise points (HDBSCAN, labels=-1), las metricas
    se calculan solo sobre los puntos asignados. Las metricas externas (ARI,
    NMI) incluyen todos los puntos.

    Parameters
    ----------
    data : np.ndarray
        Datos, shape [N, D].
    labels : np.ndarray
        Asignaciones de cluster, shape [N]. -1 = noise.
    genre_labels : np.ndarray
        Etiquetas de genero, shape [N].
    algorithm : str
        Nombre del algoritmo.
    space_name : str
        Nombre del espacio.
    k : int
        Clusters pedidos/encontrados.
    n_noise : int
        Puntos de ruido.
    has_degenerate_clusters : bool
        Si algun cluster es < MIN_CLUSTER_PCT.
    cluster_sizes : dict, optional
        Tamanos de clusters.
    params : dict, optional
        Parametros del algoritmo.
    elapsed_seconds : float
        Tiempo de ejecucion del algoritmo.
    metric : str
        Metrica de distancia para Silhouette.

    Returns
    -------
    ClusteringMetrics
    """
    if cluster_sizes is None:
        cluster_sizes = {}
    if params is None:
        params = {}

    n_samples = len(data)
    unique_labels = np.unique(labels[labels >= 0])
    n_clusters_found = len(unique_labels)

    # Mask para puntos asignados (excluir noise)
    assigned_mask = labels >= 0
    data_assigned = data[assigned_mask]
    labels_assigned = labels[assigned_mask]

    # Si hay menos de 2 clusters asignados, no se pueden calcular metricas
    if n_clusters_found < 2:
        logger.warning(
            "  %s k=%d on %s: solo %d cluster(s), metricas no calculables",
            algorithm, k, space_name, n_clusters_found,
        )
        return ClusteringMetrics(
            algorithm=algorithm,
            space_name=space_name,
            k=k,
            n_samples=n_samples,
            n_clusters_found=n_clusters_found,
            n_noise=n_noise,
            has_degenerate_clusters=True,
            cluster_sizes=cluster_sizes,
            params=params,
            elapsed_seconds=elapsed_seconds,
        )

    # --- Silhouette micro (sklearn default: promedio por punto) ---
    sil_micro = float(
        silhouette_score(data_assigned, labels_assigned, metric=metric)
    )

    # --- Silhouette macro (promedio por cluster) ---
    sil_samples = silhouette_samples(
        data_assigned, labels_assigned, metric=metric
    )
    sil_per_cluster = {}
    for c in unique_labels:
        mask = labels_assigned == c
        sil_per_cluster[int(c)] = float(sil_samples[mask].mean())
    sil_macro = float(np.mean(list(sil_per_cluster.values())))

    # --- Davies-Bouldin (menor = mejor) ---
    db = float(davies_bouldin_score(data_assigned, labels_assigned))

    # --- Calinski-Harabasz (mayor = mejor) ---
    ch = float(calinski_harabasz_score(data_assigned, labels_assigned))

    # --- Metricas externas (sobre TODOS los puntos, incluyendo noise) ---
    # Para HDBSCAN, noise points (-1) se tratan como su propio "cluster"
    ari = float(adjusted_rand_score(genre_labels, labels))
    nmi = float(
        normalized_mutual_info_score(
            genre_labels, labels, average_method="arithmetic"
        )
    )

    logger.info(
        "  %s k=%d %s: Sil_macro=%.4f, Sil_micro=%.4f, DB=%.4f, "
        "CH=%.1f, ARI=%.4f, NMI=%.4f",
        algorithm, k, space_name,
        sil_macro, sil_micro, db, ch, ari, nmi,
    )

    return ClusteringMetrics(
        algorithm=algorithm,
        space_name=space_name,
        k=k,
        n_samples=n_samples,
        n_clusters_found=n_clusters_found,
        n_noise=n_noise,
        has_degenerate_clusters=has_degenerate_clusters,
        cluster_sizes=cluster_sizes,
        silhouette_micro=round(sil_micro, 6),
        silhouette_macro=round(sil_macro, 6),
        davies_bouldin=round(db, 6),
        calinski_harabasz=round(ch, 2),
        ari_genre=round(ari, 6),
        nmi_genre=round(nmi, 6),
        silhouette_per_cluster=sil_per_cluster,
        params=params,
        elapsed_seconds=elapsed_seconds,
    )


def compute_cross_modal_nmi(
    labels_semantic: np.ndarray,
    labels_musical: np.ndarray,
) -> float:
    """
    NMI entre las mejores asignaciones de clustering semantico y musical.

    NMI bajo = los clusters capturan informacion complementaria
    (justifica fusion multimodal).

    Parameters
    ----------
    labels_semantic : np.ndarray
        Labels del mejor clustering semantico.
    labels_musical : np.ndarray
        Labels del mejor clustering musical.

    Returns
    -------
    float
        NMI en [0, 1].
    """
    nmi = float(
        normalized_mutual_info_score(
            labels_semantic, labels_musical, average_method="arithmetic"
        )
    )
    logger.info("NMI cross-modal: %.4f", nmi)
    return nmi


def select_best_configuration(
    metrics_list: List[ClusteringMetrics],
    max_noise_pct: float = 0.5,
) -> Optional[ClusteringMetrics]:
    """
    Selecciona la mejor configuracion por triangulacion.

    Criterio:
    1. Filtrar configuraciones con has_degenerate_clusters=True
    2. Filtrar configuraciones con n_clusters_found < 2
    3. Filtrar configuraciones con cobertura insuficiente (ruido > max_noise_pct)
    4. Ordenar por silhouette_macro descendente
    5. En empate (diferencia < 0.01): preferir menor davies_bouldin

    El filtro de cobertura evita seleccionar configuraciones que descartan la
    mayoria de los puntos como ruido: un HDBSCAN que agrupa solo el subconjunto
    mas denso obtiene un Silhouette alto sobre esos pocos puntos, pero no
    representa la estructura global del espacio. Sin este filtro, una particion
    que cubre el 16% del catalogo puede desplazar a una de cobertura total con
    estructura mas debil pero representativa.

    Parameters
    ----------
    metrics_list : list of ClusteringMetrics
        Todas las configuraciones evaluadas para un espacio.
    max_noise_pct : float
        Fraccion maxima de puntos clasificados como ruido admitida (0-1).

    Returns
    -------
    ClusteringMetrics or None
        Mejor configuracion, o None si todas son degeneradas.
    """
    valid = [
        m for m in metrics_list
        if not m.has_degenerate_clusters and m.n_clusters_found >= 2
    ]

    if not valid:
        logger.warning("Ninguna configuracion valida encontrada")
        return None

    # Filtrar por cobertura: descartar configs con demasiado ruido
    covered = [
        m for m in valid
        if m.n_samples == 0 or (m.n_noise / m.n_samples) <= max_noise_pct
    ]
    if covered:
        n_descartadas = len(valid) - len(covered)
        if n_descartadas:
            logger.info(
                "  %d configuracion(es) descartada(s) por ruido > %.0f%%",
                n_descartadas, max_noise_pct * 100,
            )
        valid = covered
    else:
        logger.warning(
            "  Todas las configuraciones superan el umbral de ruido %.0f%%; "
            "se selecciona sin filtrar por cobertura.",
            max_noise_pct * 100,
        )

    # Ordenar por silhouette_macro desc, luego davies_bouldin asc
    valid.sort(key=lambda m: (-m.silhouette_macro, m.davies_bouldin))

    best = valid[0]

    # Log de candidatos cercanos (dentro de 0.01 de silhouette)
    close = [
        m for m in valid[1:]
        if abs(m.silhouette_macro - best.silhouette_macro) < 0.01
    ]
    if close:
        logger.info(
            "  Mejor: %s k=%d (Sil_macro=%.4f, DB=%.4f). "
            "%d candidatos cercanos (dentro de 0.01 Sil).",
            best.algorithm, best.k, best.silhouette_macro,
            best.davies_bouldin, len(close),
        )
    else:
        logger.info(
            "  Mejor: %s k=%d (Sil_macro=%.4f, DB=%.4f). "
            "Sin candidatos cercanos.",
            best.algorithm, best.k, best.silhouette_macro,
            best.davies_bouldin,
        )

    return best
