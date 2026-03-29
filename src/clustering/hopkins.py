"""
Modulo de calculo del estadistico de Hopkins (Etapa 4, Paso 1).

El estadistico de Hopkins mide la tendencia al agrupamiento de un dataset:
compara las distancias nearest-neighbor de puntos reales con las de puntos
generados uniformemente. Un valor cercano a 0.5 indica distribucion aleatoria
(sin estructura), mientras que valores superiores a 0.7 sugieren estructura
de clusters significativa.

Detalle critico sobre la distribucion de referencia:
- Para datos L2-normalizados (hiperesfera unitaria), los puntos uniformes
  se generan sobre la hiperesfera, no en el hipercubo. Un hipercubo en alta
  dimensionalidad genera puntos mayoritariamente fuera de la hiperesfera,
  inflando Hopkins artificialmente.
- Para datos sin restriccion geometrica (z-score), los puntos uniformes
  se generan en el hipercubo definido por [min, max] por dimension.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.config import HOPKINS_ITERATIONS, HOPKINS_SAMPLE_SIZE, HOPKINS_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class HopkinsReport:
    """Reporte del test de Hopkins sobre un espacio de features."""

    space_name: str
    n_samples: int
    n_dimensions: int
    sample_size: int
    n_iterations: int
    metric: str
    reference_distribution: str  # "hypersphere" o "hypercube"
    hopkins_values: np.ndarray = field(repr=False)
    mean: float = 0.0
    std: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    exceeds_threshold: bool = False


def _generate_uniform_hypercube(
    data: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Genera puntos uniformes en el hipercubo [min, max] por dimension.

    Apropiado para datos sin restriccion geometrica (z-score, features
    musicales). Cada dimension se muestrea independientemente en el rango
    observado de los datos reales.

    Parameters
    ----------
    data : np.ndarray
        Datos reales, shape [N, D].
    n_points : int
        Cantidad de puntos a generar.
    rng : np.random.Generator
        Generador de numeros aleatorios.

    Returns
    -------
    np.ndarray
        Puntos uniformes, shape [n_points, D].
    """
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    return rng.uniform(mins, maxs, size=(n_points, data.shape[1])).astype(
        data.dtype
    )


def _generate_uniform_hypersphere(
    n_dimensions: int,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Genera puntos uniformes sobre la hiperesfera unitaria.

    Apropiado para datos L2-normalizados (embeddings semanticos). Se generan
    vectores de una distribucion normal multivariada y se L2-normalizan,
    lo que produce una distribucion uniforme sobre la superficie de la
    hiperesfera (propiedad de la simetria esferica de la normal).

    Parameters
    ----------
    n_dimensions : int
        Dimensionalidad del espacio.
    n_points : int
        Cantidad de puntos a generar.
    rng : np.random.Generator
        Generador de numeros aleatorios.

    Returns
    -------
    np.ndarray
        Puntos uniformes sobre la hiperesfera, shape [n_points, D], float32.
    """
    vectors = rng.standard_normal(size=(n_points, n_dimensions)).astype(
        np.float32
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def compute_hopkins_statistic(
    data: np.ndarray,
    sample_size: int = HOPKINS_SAMPLE_SIZE,
    metric: str = "euclidean",
    reference: Literal["hypercube", "hypersphere"] = "hypercube",
    random_state: int = 42,
) -> float:
    """
    Calcula una iteracion del estadistico de Hopkins.

    H = sum(u_i) / (sum(u_i) + sum(w_i))

    donde:
    - u_i = distancia del i-esimo punto uniforme a su vecino mas cercano
      en los datos reales
    - w_i = distancia del i-esimo punto real (subsample) a su vecino mas
      cercano en los datos reales (excluyendose a si mismo)

    Parameters
    ----------
    data : np.ndarray
        Datos reales, shape [N, D].
    sample_size : int
        Cantidad de puntos a muestrear (m). Debe ser << N.
    metric : str
        Metrica de distancia ("euclidean" o "cosine").
    reference : {"hypercube", "hypersphere"}
        Distribucion de referencia para los puntos uniformes.
    random_state : int
        Seed para reproducibilidad.

    Returns
    -------
    float
        Valor de Hopkins en [0, 1].
    """
    n_samples, n_dims = data.shape
    rng = np.random.default_rng(random_state)

    if sample_size >= n_samples:
        sample_size = n_samples // 10
        logger.warning(
            "sample_size >= n_samples, reducido a %d", sample_size
        )

    # Ajustar nearest neighbor: k=1 para uniformes, k=2 para reales
    # (el mas cercano de un punto real a si mismo es 0)
    nn_model = NearestNeighbors(n_neighbors=2, metric=metric, algorithm="auto")
    nn_model.fit(data)

    # --- u_i: distancias de puntos uniformes al vecino mas cercano ---
    if reference == "hypersphere":
        uniform_points = _generate_uniform_hypersphere(n_dims, sample_size, rng)
    else:
        uniform_points = _generate_uniform_hypercube(data, sample_size, rng)

    u_distances, _ = nn_model.kneighbors(uniform_points, n_neighbors=1)
    u_sum = float(u_distances.sum())

    # --- w_i: distancias de puntos reales al segundo vecino mas cercano ---
    sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
    sample_points = data[sample_indices]

    w_distances, _ = nn_model.kneighbors(sample_points, n_neighbors=2)
    # El segundo vecino (indice 1) es el mas cercano excluyendo a si mismo
    w_sum = float(w_distances[:, 1].sum())

    denominator = u_sum + w_sum
    if denominator == 0:
        logger.warning("Hopkins: denominador es 0, retornando 0.5")
        return 0.5

    return u_sum / denominator


def compute_hopkins_bootstrap(
    data: np.ndarray,
    space_name: str,
    sample_size: int = HOPKINS_SAMPLE_SIZE,
    n_iterations: int = HOPKINS_ITERATIONS,
    metric: str = "euclidean",
    reference: Literal["hypercube", "hypersphere"] = "hypercube",
    random_state: int = 42,
) -> HopkinsReport:
    """
    Ejecuta Hopkins con bootstrap: n_iterations repeticiones con seeds
    incrementales. Calcula media, desviacion estandar e intervalo de
    confianza al 95% (percentiles 2.5 y 97.5).

    Parameters
    ----------
    data : np.ndarray
        Datos reales, shape [N, D].
    space_name : str
        Nombre del espacio (para reporte).
    sample_size : int
        Puntos a muestrear por iteracion.
    n_iterations : int
        Numero de repeticiones bootstrap.
    metric : str
        Metrica de distancia.
    reference : {"hypercube", "hypersphere"}
        Distribucion de referencia para puntos uniformes.
    random_state : int
        Seed base. Cada iteracion usa random_state + i.

    Returns
    -------
    HopkinsReport
        Reporte con estadisticas agregadas.
    """
    n_samples, n_dims = data.shape
    logger.info(
        "Hopkins bootstrap: space=%s, dims=%d, samples=%d, "
        "iterations=%d, metric=%s, reference=%s",
        space_name, n_dims, n_samples, n_iterations, metric, reference,
    )

    values = np.zeros(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        values[i] = compute_hopkins_statistic(
            data,
            sample_size=sample_size,
            metric=metric,
            reference=reference,
            random_state=random_state + i,
        )

    mean = float(values.mean())
    std = float(values.std())
    ci_lower = float(np.percentile(values, 2.5))
    ci_upper = float(np.percentile(values, 97.5))

    report = HopkinsReport(
        space_name=space_name,
        n_samples=n_samples,
        n_dimensions=n_dims,
        sample_size=sample_size,
        n_iterations=n_iterations,
        metric=metric,
        reference_distribution=reference,
        hopkins_values=values,
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        min_value=float(values.min()),
        max_value=float(values.max()),
        exceeds_threshold=mean > HOPKINS_THRESHOLD,
    )

    logger.info(
        "  %s: H=%.4f +/- %.4f, IC95%%=[%.4f, %.4f], > %.1f: %s",
        space_name, mean, std, ci_lower, ci_upper,
        HOPKINS_THRESHOLD, report.exceeds_threshold,
    )

    return report
