"""
Modulo de reduccion dimensional y preparacion de espacios (Etapa 4).

Proporciona funciones para:
1. Reduccion dimensional con UMAP (para preprocesamiento y visualizacion).
2. Preparacion de los 4 espacios sobre los que se evalua Hopkins:
   semantico 384D, semantico UMAP, musical 13D, concatenado.

UMAP tiene dos usos distintos que no deben confundirse:
- Visualizacion (n_components=2): para figuras del informe.
- Preprocesamiento (n_components=30): reduccion antes de clustering o
  Hopkins, donde la alta dimensionalidad degrada las metricas.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.config import (
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_PREPROCESSING_N_COMPONENTS,
)

logger = logging.getLogger(__name__)


@dataclass
class UMAPReport:
    """Reporte de una reduccion UMAP."""

    input_dim: int
    output_dim: int
    n_samples: int
    n_neighbors: int
    min_dist: float
    metric: str
    random_state: int
    elapsed_seconds: float


def reduce_umap(
    data: np.ndarray,
    n_components: int,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    metric: str = "cosine",
    random_state: int = 42,
) -> Tuple[np.ndarray, UMAPReport]:
    """
    Aplica UMAP para reduccion dimensional.

    Parameters
    ----------
    data : np.ndarray
        Datos de entrada, shape [N, D].
    n_components : int
        Dimensionalidad de salida.
    n_neighbors : int
        Numero de vecinos para construir el grafo local.
    min_dist : float
        Distancia minima entre puntos en el espacio reducido.
    metric : str
        Metrica de distancia en el espacio original.
    random_state : int
        Seed para reproducibilidad.

    Returns
    -------
    reduced : np.ndarray
        Datos reducidos, shape [N, n_components], float32.
    report : UMAPReport
        Reporte con parametros y tiempo de ejecucion.
    """
    import umap

    n_samples, input_dim = data.shape
    logger.info(
        "UMAP: [%d, %d] -> [%d, %d], metric=%s, n_neighbors=%d, min_dist=%.2f",
        n_samples, input_dim, n_samples, n_components,
        metric, n_neighbors, min_dist,
    )

    t0 = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(data).astype(np.float32)
    elapsed = time.time() - t0

    logger.info("  UMAP completado en %.1f s", elapsed)

    report = UMAPReport(
        input_dim=input_dim,
        output_dim=n_components,
        n_samples=n_samples,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        elapsed_seconds=round(elapsed, 2),
    )

    return reduced, report


def reduce_for_visualization(
    data: np.ndarray,
    metric: str = "cosine",
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    random_state: int = 42,
) -> Tuple[np.ndarray, UMAPReport]:
    """
    Wrapper de reduce_umap con n_components=2 para visualizacion.

    Parameters
    ----------
    data : np.ndarray
        Datos de entrada, shape [N, D].
    metric : str
        Metrica de distancia.
    n_neighbors : int
        Numero de vecinos.
    min_dist : float
        Distancia minima en 2D.
    random_state : int
        Seed.

    Returns
    -------
    embedding_2d : np.ndarray
        Proyeccion 2D, shape [N, 2], float32.
    report : UMAPReport
        Reporte.
    """
    return reduce_umap(
        data,
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )


def prepare_spaces_for_hopkins(
    semantic_embeddings: np.ndarray,
    musical_features: np.ndarray,
    umap_n_components: int = UMAP_PREPROCESSING_N_COMPONENTS,
    umap_n_neighbors: int = UMAP_N_NEIGHBORS,
    umap_min_dist: float = UMAP_MIN_DIST,
    random_state: int = 42,
) -> Tuple[Dict[str, Tuple[np.ndarray, str, str]], UMAPReport]:
    """
    Prepara los 4 espacios para el calculo de Hopkins.

    Para cada espacio, retorna los datos, la metrica de distancia apropiada,
    y el tipo de distribucion de referencia para Hopkins:
    - semantic_384d: coseno + hiperesfera (datos L2-normalizados)
    - semantic_umap: euclidiana + hipercubo (post-UMAP, ya no en hiperesfera)
    - musical_13d: euclidiana + hipercubo (z-score)
    - concatenated: euclidiana + hipercubo (min-max por bloque)

    El espacio concatenado normaliza cada bloque al rango [0, 1] con min-max
    antes de concatenar, para que ambas modalidades contribuyan con rango
    comparable a la distancia euclidiana.

    Parameters
    ----------
    semantic_embeddings : np.ndarray
        Embeddings semanticos, shape [N, 384], L2-normalizados.
    musical_features : np.ndarray
        Features musicales, shape [N, 13], z-score.
    umap_n_components : int
        Dimensionalidad de la proyeccion UMAP.
    umap_n_neighbors : int
        Vecinos para UMAP.
    umap_min_dist : float
        Distancia minima para UMAP.
    random_state : int
        Seed.

    Returns
    -------
    spaces : dict
        {space_name: (data, metric, reference_distribution)}
    umap_report : UMAPReport
        Reporte de la reduccion UMAP aplicada.
    """
    logger.info("Preparando 4 espacios para Hopkins...")

    # 1. Semantico original (384D, coseno, hiperesfera)
    spaces = {
        "semantic_384d": (semantic_embeddings, "cosine", "hypersphere"),
    }
    logger.info(
        "  semantic_384d: shape=%s, metric=cosine, ref=hypersphere",
        semantic_embeddings.shape,
    )

    # 2. Semantico UMAP (reducido, euclidiana, hipercubo)
    semantic_umap, umap_report = reduce_umap(
        semantic_embeddings,
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=random_state,
    )
    spaces["semantic_umap"] = (semantic_umap, "euclidean", "hypercube")
    logger.info(
        "  semantic_umap: shape=%s, metric=euclidean, ref=hypercube",
        semantic_umap.shape,
    )

    # 3. Musical (13D, euclidiana, hipercubo)
    spaces["musical_13d"] = (musical_features, "euclidean", "hypercube")
    logger.info(
        "  musical_13d: shape=%s, metric=euclidean, ref=hypercube",
        musical_features.shape,
    )

    # 4. Concatenado (397D, euclidiana, hipercubo)
    # Min-max por bloque para igualar escalas antes de concatenar
    sem_min = semantic_embeddings.min(axis=0, keepdims=True)
    sem_max = semantic_embeddings.max(axis=0, keepdims=True)
    sem_range = sem_max - sem_min
    sem_range[sem_range == 0] = 1.0  # evitar division por cero
    sem_scaled = (semantic_embeddings - sem_min) / sem_range

    mus_min = musical_features.min(axis=0, keepdims=True)
    mus_max = musical_features.max(axis=0, keepdims=True)
    mus_range = mus_max - mus_min
    mus_range[mus_range == 0] = 1.0
    mus_scaled = (musical_features - mus_min) / mus_range

    concatenated = np.concatenate(
        [sem_scaled, mus_scaled], axis=1
    ).astype(np.float32)
    spaces["concatenated"] = (concatenated, "euclidean", "hypercube")
    logger.info(
        "  concatenated: shape=%s, metric=euclidean, ref=hypercube",
        concatenated.shape,
    )

    return spaces, umap_report
