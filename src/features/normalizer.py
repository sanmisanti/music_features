"""
Modulo de extraccion y normalizacion de features musicales (Etapa 3, Paso 3).

Extrae las 12 features musicales de Spotify del dataset seleccionado,
aplica normalizacion z-score por columna, y genera estadisticas pre/post
para documentacion y verificacion.

Nota sobre key y mode: key (0-11, categorica ordinal) y mode (0/1, binaria)
no son estrictamente continuas. Aplicar z-score a estas variables es una
simplificacion documentada. La alternativa (one-hot encoding) agregaria 12+
dimensiones sin beneficio proporcional para un espacio de solo 12D.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import MUSICAL_FEATURES

logger = logging.getLogger(__name__)


@dataclass
class NormalizationReport:
    """Reporte de la normalizacion de features musicales."""

    n_samples: int
    n_features: int
    feature_names: List[str]
    method: str  # "zscore"
    raw_means: np.ndarray = field(repr=False)
    raw_stds: np.ndarray = field(repr=False)
    raw_mins: np.ndarray = field(repr=False)
    raw_maxs: np.ndarray = field(repr=False)
    post_means: np.ndarray = field(repr=False)
    post_stds: np.ndarray = field(repr=False)
    n_nan_before: int = 0
    n_nan_after: int = 0


def extract_musical_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Extrae las 12 features musicales del DataFrame seleccionado.

    Verifica que todas las columnas definidas en MUSICAL_FEATURES existen
    y que no contienen valores NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset seleccionado con las 25 columnas originales.

    Returns
    -------
    features : np.ndarray
        Matriz de features musicales, shape [N, 12], dtype float32.
    feature_names : list of str
        Nombres de las 12 features en el orden extraido.

    Raises
    ------
    KeyError
        Si alguna columna de MUSICAL_FEATURES no existe en el DataFrame.
    ValueError
        Si se detectan valores NaN en las features extraidas.
    """
    missing = [col for col in MUSICAL_FEATURES if col not in df.columns]
    if missing:
        raise KeyError(
            f"Columnas faltantes en el DataFrame: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    features_df = df[MUSICAL_FEATURES]
    n_nan = int(features_df.isnull().sum().sum())

    if n_nan > 0:
        nan_per_col = features_df.isnull().sum()
        cols_with_nan = nan_per_col[nan_per_col > 0]
        raise ValueError(
            f"Se detectaron {n_nan} valores NaN en features musicales. "
            f"Columnas afectadas: {dict(cols_with_nan)}"
        )

    features = features_df.to_numpy(dtype=np.float32)
    feature_names = list(MUSICAL_FEATURES)

    logger.info(
        "Features extraidas: %d muestras x %d features, dtype=%s, NaN=%d",
        features.shape[0], features.shape[1], features.dtype, n_nan,
    )

    return features, feature_names


def normalize_features(
    features: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, NormalizationReport]:
    """
    Aplica normalizacion z-score (estandarizacion) por columna.

    La transformacion es: z = (x - media) / desviacion_estandar
    para cada feature independientemente.

    Parameters
    ----------
    features : np.ndarray
        Matriz de features crudas, shape [N, D], dtype float32.
    feature_names : list of str
        Nombres de las D features.

    Returns
    -------
    normalized : np.ndarray
        Features normalizadas, shape [N, D], dtype float32.
    report : NormalizationReport
        Estadisticas pre y post normalizacion.
    """
    n_nan_before = int(np.isnan(features).sum())

    # Estadisticas pre-normalizacion
    raw_means = features.mean(axis=0)
    raw_stds = features.std(axis=0)
    raw_mins = features.min(axis=0)
    raw_maxs = features.max(axis=0)

    logger.info("Estadisticas pre-normalizacion calculadas para %d features", len(feature_names))
    for i, name in enumerate(feature_names):
        logger.info(
            "  %s: media=%.4f, std=%.4f, min=%.4f, max=%.4f",
            name, raw_means[i], raw_stds[i], raw_mins[i], raw_maxs[i],
        )

    # Z-score: (x - media) / std
    normalized = (features - raw_means) / raw_stds
    normalized = normalized.astype(np.float32)

    # Estadisticas post-normalizacion
    post_means = normalized.mean(axis=0)
    post_stds = normalized.std(axis=0)
    n_nan_after = int(np.isnan(normalized).sum())

    logger.info("Normalizacion z-score aplicada")
    logger.info(
        "  Post-normalizacion: media_max=%.2e, std_rango=[%.6f, %.6f]",
        float(np.abs(post_means).max()),
        float(post_stds.min()),
        float(post_stds.max()),
    )

    report = NormalizationReport(
        n_samples=features.shape[0],
        n_features=features.shape[1],
        feature_names=feature_names,
        method="zscore",
        raw_means=raw_means,
        raw_stds=raw_stds,
        raw_mins=raw_mins,
        raw_maxs=raw_maxs,
        post_means=post_means,
        post_stds=post_stds,
        n_nan_before=n_nan_before,
        n_nan_after=n_nan_after,
    )

    return normalized, report
