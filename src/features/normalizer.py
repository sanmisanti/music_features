"""
Modulo de extraccion y normalizacion de features musicales (Etapa 3, Paso 3).

Extrae las 12 features musicales de Spotify del dataset seleccionado,
aplica codificacion circular a key (12D -> 13D: key_sin + key_cos),
normaliza con z-score por columna, y genera estadisticas pre/post
para documentacion y verificacion.

Tratamiento de variables categoricas:
- key (0-11, tonalidad): codificacion circular sin/cos para preservar
  la relacion de adyacencia entre tonalidades (Do y Si son vecinas).
- mode (0/1, modalidad): z-score directo. Para variables binarias,
  z-score es matematicamente equivalente a una transformacion lineal
  que centra y escala los dos valores posibles.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import KEY_PERIOD, MUSICAL_FEATURES

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


def circular_encode_key(
    values: np.ndarray, period: int = KEY_PERIOD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Codifica la tonalidad (key) como dos componentes circulares sin/cos.

    La tonalidad es una variable categorica circular: los valores 0 (Do) y
    11 (Si) son adyacentes en el circulo cromatico, pero su representacion
    numerica los situa a maxima distancia. La codificacion circular preserva
    esta relacion de adyacencia mapeando cada valor a un punto en el circulo
    unitario.

    Parameters
    ----------
    values : np.ndarray
        Valores enteros de key (0-11).
    period : int
        Periodo de la codificacion circular (12 para tonalidades).

    Returns
    -------
    key_sin : np.ndarray
        Componente seno, shape [N], dtype float32.
    key_cos : np.ndarray
        Componente coseno, shape [N], dtype float32.
    """
    angles = 2.0 * np.pi * values / period
    return np.sin(angles).astype(np.float32), np.cos(angles).astype(np.float32)


def extract_musical_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Extrae las features musicales del DataFrame y aplica codificacion circular a key.

    Extrae las 12 columnas definidas en MUSICAL_FEATURES, reemplaza la columna
    ``key`` (valores enteros 0-11) por dos columnas ``key_sin`` y ``key_cos``
    mediante codificacion circular, resultando en 13 features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset seleccionado con las 25 columnas originales.

    Returns
    -------
    features : np.ndarray
        Matriz de features musicales, shape [N, 13], dtype float32.
    feature_names : list of str
        Nombres de las 13 features en el orden extraido.

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

    # --- Codificacion circular de key ---
    key_idx = feature_names.index("key")
    key_values = features[:, key_idx]
    key_sin, key_cos = circular_encode_key(key_values)

    logger.info(
        "Codificacion circular de key: %d valores, "
        "key_sin rango=[%.4f, %.4f], key_cos rango=[%.4f, %.4f]",
        len(key_values),
        float(key_sin.min()), float(key_sin.max()),
        float(key_cos.min()), float(key_cos.max()),
    )

    # Reemplazar columna key por key_sin y key_cos
    # Resultado: columnas antes de key + key_sin + key_cos + columnas despues de key
    features = np.concatenate([
        features[:, :key_idx],           # columnas antes de key
        key_sin.reshape(-1, 1),          # key_sin
        key_cos.reshape(-1, 1),          # key_cos
        features[:, key_idx + 1:],       # columnas despues de key
    ], axis=1)

    feature_names = (
        feature_names[:key_idx]
        + ["key_sin", "key_cos"]
        + feature_names[key_idx + 1:]
    )

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
