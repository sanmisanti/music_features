"""
Modulo de unificacion de features semanticas y musicales (Etapa 3, Paso 4).

Empaqueta los embeddings semanticos (384D), features musicales normalizadas (12D),
y metadatos de evaluacion (genero, nombre, artista) en un unico archivo NPZ.
Verifica alineacion cruzada entre los NPY de 4_vectorized/ y el CSV de 3_selected/.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from src.config import (
    SELECTED_DATASET,
    SOURCE_SEPARATOR,
    UNIFIED_DATASET,
    VECTORIZED_EMBEDDINGS,
    VECTORIZED_FEATURE_NAMES,
    VECTORIZED_MUSICAL_FEATURES,
    VECTORIZED_TOKEN_COUNTS,
    VECTORIZED_TRACK_IDS,
)

logger = logging.getLogger(__name__)


@dataclass
class UnificationReport:
    """Reporte de la unificacion de features en dataset NPZ."""

    n_samples: int
    semantic_dim: int
    musical_dim: int
    n_genres: int
    genre_distribution: Dict[str, int]
    semantic_l2_mean: float
    semantic_l2_std: float
    musical_mean_abs_mean: float
    musical_mean_std: float
    n_nan_semantic: int
    n_nan_musical: int
    track_ids_aligned: bool
    npz_size_mb: float
    n_arrays: int


def load_feature_components() -> Dict[str, np.ndarray]:
    """
    Carga los 5 archivos NPY de data/4_vectorized/.

    Verifica que todos existen y que la dimension 0 (numero de muestras)
    es consistente entre embeddings, musical_features, track_ids y token_counts.

    Returns
    -------
    dict
        Diccionario con keys: embeddings, musical_features, track_ids,
        feature_names, token_counts.

    Raises
    ------
    FileNotFoundError
        Si alguno de los archivos NPY no existe.
    ValueError
        Si la dimension 0 no es consistente entre arrays.
    """
    paths = {
        "embeddings": VECTORIZED_EMBEDDINGS,
        "musical_features": VECTORIZED_MUSICAL_FEATURES,
        "track_ids": VECTORIZED_TRACK_IDS,
        "feature_names": VECTORIZED_FEATURE_NAMES,
        "token_counts": VECTORIZED_TOKEN_COUNTS,
    }

    # Verificar existencia
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Archivo NPY no encontrado: {path} ({name})"
            )

    # Cargar arrays
    components = {}
    for name, path in paths.items():
        components[name] = np.load(path, allow_pickle=True)
        logger.info(
            "Cargado %s: shape=%s, dtype=%s",
            name, components[name].shape, components[name].dtype,
        )

    # Verificar dimension 0 consistente (excluir feature_names que tiene shape (12,))
    n_samples = components["embeddings"].shape[0]
    for name in ["musical_features", "track_ids", "token_counts"]:
        if components[name].shape[0] != n_samples:
            raise ValueError(
                f"Dimension 0 inconsistente: embeddings={n_samples}, "
                f"{name}={components[name].shape[0]}"
            )

    logger.info(
        "Componentes cargados: %d muestras, dimension 0 consistente",
        n_samples,
    )

    return components


def load_metadata(
    track_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Carga metadatos del CSV seleccionado y verifica alineacion con track_ids.

    Extrae playlist_genre, track_name y track_artist del DataFrame,
    verificando que los track_ids del CSV coinciden con los del NPY
    en el mismo orden.

    Parameters
    ----------
    track_ids : np.ndarray
        Array de track_ids de los archivos NPY (referencia de orden).

    Returns
    -------
    dict
        Diccionario con keys: genre_labels, track_names, track_artists.

    Raises
    ------
    ValueError
        Si los track_ids no coinciden en cantidad u orden.
    """
    logger.info("Cargando metadatos desde %s", SELECTED_DATASET)
    df = pd.read_csv(
        SELECTED_DATASET,
        sep=SOURCE_SEPARATOR,
        encoding="utf-8",
        engine="python",
    )
    logger.info("CSV cargado: %d filas, %d columnas", len(df), len(df.columns))

    # Verificar alineacion de track_ids
    csv_track_ids = df["track_id"].values

    if len(csv_track_ids) != len(track_ids):
        raise ValueError(
            f"Cantidad de track_ids no coincide: CSV={len(csv_track_ids)}, "
            f"NPY={len(track_ids)}"
        )

    if not np.array_equal(csv_track_ids, track_ids):
        # Determinar cuantos difieren para diagnostico
        n_diff = int(np.sum(csv_track_ids != track_ids))
        raise ValueError(
            f"Track IDs del CSV y NPY no coinciden en orden: "
            f"{n_diff} posiciones difieren de {len(track_ids)}"
        )

    logger.info("Track IDs alineados: %d coinciden en orden", len(track_ids))

    # Extraer metadatos
    metadata = {
        "genre_labels": df["playlist_genre"].values,
        "track_names": df["track_name"].values,
        "track_artists": df["track_artist"].values,
    }

    for name, arr in metadata.items():
        n_null = int(pd.Series(arr).isnull().sum())
        logger.info("  %s: %d valores, %d nulos", name, len(arr), n_null)

    return metadata


def build_unified_dataset(
    components: Dict[str, np.ndarray],
    metadata: Dict[str, np.ndarray],
) -> UnificationReport:
    """
    Construye el dataset unificado NPZ a partir de componentes y metadatos.

    Ejecuta verificaciones de integridad cruzada, guarda el archivo NPZ
    con np.savez (sin compresion, para carga rapida), y genera el reporte.

    Parameters
    ----------
    components : dict
        Diccionario de load_feature_components(): embeddings, musical_features,
        track_ids, feature_names, token_counts.
    metadata : dict
        Diccionario de load_metadata(): genre_labels, track_names, track_artists.

    Returns
    -------
    UnificationReport
        Reporte con estadisticas del dataset unificado.

    Raises
    ------
    AssertionError
        Si alguna verificacion de integridad falla.
    """
    embeddings = components["embeddings"]
    musical = components["musical_features"]
    track_ids = components["track_ids"]
    feature_names = components["feature_names"]
    token_counts = components["token_counts"]

    genre_labels = metadata["genre_labels"]
    track_names = metadata["track_names"]
    track_artists = metadata["track_artists"]

    n_samples = embeddings.shape[0]

    # --- Verificaciones de integridad ---
    logger.info("Ejecutando verificaciones de integridad cruzada...")

    # NaN en matrices numericas
    n_nan_semantic = int(np.isnan(embeddings).sum())
    n_nan_musical = int(np.isnan(musical).sum())
    assert n_nan_semantic == 0, f"NaN en embeddings: {n_nan_semantic}"
    assert n_nan_musical == 0, f"NaN en musical_features: {n_nan_musical}"
    logger.info("  NaN: semantic=%d, musical=%d", n_nan_semantic, n_nan_musical)

    # Norma L2 de embeddings
    norms = np.linalg.norm(embeddings, axis=1)
    l2_mean = float(norms.mean())
    l2_std = float(norms.std())
    assert abs(l2_mean - 1.0) < 0.01, (
        f"Norma L2 media fuera de tolerancia: {l2_mean:.6f} (esperado ~1.0)"
    )
    logger.info("  Norma L2: media=%.6f, std=%.6f", l2_mean, l2_std)

    # Z-score de features musicales
    musical_means = musical.mean(axis=0)
    musical_stds = musical.std(axis=0)
    mean_abs_mean = float(np.abs(musical_means).mean())
    mean_std = float(musical_stds.mean())
    assert mean_abs_mean < 1e-5, (
        f"Media absoluta de features musicales fuera de tolerancia: {mean_abs_mean:.2e}"
    )
    assert abs(mean_std - 1.0) < 1e-5, (
        f"Media de DE de features musicales fuera de tolerancia: {mean_std:.6f}"
    )
    logger.info(
        "  Z-score: media_abs_media=%.2e, media_std=%.6f",
        mean_abs_mean, mean_std,
    )

    # Generos unicos
    unique_genres = sorted(set(str(g) for g in genre_labels))
    n_genres = len(unique_genres)
    assert n_genres == 6, f"Generos unicos esperados: 6, encontrados: {n_genres} ({unique_genres})"
    logger.info("  Generos: %d unicos %s", n_genres, unique_genres)

    # Valores nulos en metadatos de texto
    n_null_names = int(pd.Series(track_names).isnull().sum())
    n_null_artists = int(pd.Series(track_artists).isnull().sum())
    assert n_null_names == 0, f"Valores nulos en track_names: {n_null_names}"
    assert n_null_artists == 0, f"Valores nulos en track_artists: {n_null_artists}"
    logger.info(
        "  Metadatos: 0 nulos en track_names, 0 nulos en track_artists",
    )

    logger.info("Todas las verificaciones de integridad pasaron correctamente")

    # --- Guardar NPZ ---
    UNIFIED_DATASET.parent.mkdir(parents=True, exist_ok=True)

    # Distribucion de generos
    genre_series = pd.Series(genre_labels)
    genre_dist = dict(genre_series.value_counts().sort_index())

    np.savez(
        UNIFIED_DATASET,
        semantic_embeddings=embeddings,
        musical_features=musical,
        track_ids=track_ids,
        feature_names=feature_names,
        genre_labels=genre_labels,
        track_names=track_names,
        track_artists=track_artists,
        token_counts=token_counts,
    )

    npz_size_mb = UNIFIED_DATASET.stat().st_size / (1024 * 1024)
    logger.info(
        "Dataset unificado guardado: %s (%.2f MB, 8 arrays)",
        UNIFIED_DATASET, npz_size_mb,
    )

    # --- Verificar NPZ post-guardado ---
    with np.load(UNIFIED_DATASET, allow_pickle=True) as npz:
        n_arrays = len(npz.files)
        logger.info(
            "Verificacion post-guardado: %d arrays en NPZ %s",
            n_arrays, sorted(npz.files),
        )

    report = UnificationReport(
        n_samples=n_samples,
        semantic_dim=embeddings.shape[1],
        musical_dim=musical.shape[1],
        n_genres=n_genres,
        genre_distribution=genre_dist,
        semantic_l2_mean=l2_mean,
        semantic_l2_std=l2_std,
        musical_mean_abs_mean=mean_abs_mean,
        musical_mean_std=mean_std,
        n_nan_semantic=n_nan_semantic,
        n_nan_musical=n_nan_musical,
        track_ids_aligned=True,
        npz_size_mb=npz_size_mb,
        n_arrays=n_arrays,
    )

    return report
