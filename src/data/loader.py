"""
Módulo de carga y validación del dataset fuente.

Provee funciones para cargar el CSV con separador '@@',
validar integridad de features musicales y calidad de letras.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

from src.config import (
    MUSICAL_FEATURES,
    SOURCE_DATASET,
    SOURCE_SEPARATOR,
)

logger = logging.getLogger(__name__)

# Rangos válidos para cada feature musical de Spotify
# Fuente: Spotify Web API Reference
FEATURE_RANGES = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "key": (0, 11),  # 0=C, 1=C#, ..., 11=B; -1=no detectada
    "loudness": (-60.0, 0.0),
    "mode": (0, 1),  # 0=menor, 1=mayor
    "speechiness": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "tempo": (0.0, 300.0),
    "duration_ms": (0, 3_600_000),  # hasta 1 hora
}


@dataclass
class LoadReport:
    """Reporte de carga del dataset fuente."""

    total_rows: int = 0
    total_columns: int = 0
    memory_mb: float = 0.0
    column_names: List[str] = field(default_factory=list)
    missing_by_column: Dict[str, int] = field(default_factory=dict)
    duplicate_track_ids: int = 0
    warnings: List[str] = field(default_factory=list)


def load_source_dataset(
    path=None, validate: bool = True
) -> Tuple[pd.DataFrame, LoadReport]:
    """
    Carga el dataset fuente desde CSV con separador '@@'.

    Parameters
    ----------
    path : Path, optional
        Ruta al archivo CSV. Si no se especifica, usa SOURCE_DATASET de config.
    validate : bool
        Si True, ejecuta validaciones básicas y las registra en el reporte.

    Returns
    -------
    Tuple[pd.DataFrame, LoadReport]
        DataFrame cargado y reporte de carga.
    """
    dataset_path = path or SOURCE_DATASET
    report = LoadReport()

    logger.info("Cargando dataset desde %s", dataset_path)

    df = pd.read_csv(
        dataset_path,
        sep=SOURCE_SEPARATOR,
        encoding="utf-8",
        on_bad_lines="skip",
        engine="python",
    )

    report.total_rows = len(df)
    report.total_columns = len(df.columns)
    report.memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    report.column_names = df.columns.tolist()

    # Valores faltantes por columna
    missing = df.isnull().sum()
    report.missing_by_column = {
        col: int(count) for col, count in missing.items() if count > 0
    }

    # Track IDs duplicados
    if "track_id" in df.columns:
        report.duplicate_track_ids = int(df["track_id"].duplicated().sum())
        if report.duplicate_track_ids > 0:
            report.warnings.append(
                f"{report.duplicate_track_ids} track_id duplicados detectados"
            )

    logger.info(
        "Dataset cargado: %d filas, %d columnas, %.2f MB",
        report.total_rows,
        report.total_columns,
        report.memory_mb,
    )

    if validate:
        feat_report = validate_musical_features(df)
        for feature, info in feat_report.items():
            if info["out_of_range"] > 0:
                msg = (
                    f"{feature}: {info['out_of_range']} valores fuera de rango "
                    f"{info['expected_range']}"
                )
                report.warnings.append(msg)
                logger.warning(msg)

        lyrics_report = validate_lyrics(df)
        if lyrics_report["null_count"] > 0:
            msg = f"Letras nulas: {lyrics_report['null_count']}"
            report.warnings.append(msg)
            logger.warning(msg)

    if report.warnings:
        logger.info("Advertencias de carga: %d", len(report.warnings))
    else:
        logger.info("Carga completada sin advertencias")

    return df, report


def validate_musical_features(df: pd.DataFrame) -> Dict:
    """
    Valida que las 12 features musicales estén presentes y dentro de rangos esperados.

    Returns
    -------
    Dict[str, Dict]
        Para cada feature: missing, out_of_range, expected_range.
    """
    results = {}
    for feature in MUSICAL_FEATURES:
        info = {
            "present": feature in df.columns,
            "missing": 0,
            "out_of_range": 0,
            "expected_range": FEATURE_RANGES.get(feature, (None, None)),
        }
        if feature in df.columns:
            col = df[feature]
            info["missing"] = int(col.isnull().sum())
            lo, hi = info["expected_range"]
            if lo is not None and hi is not None:
                valid = col.dropna()
                out = ((valid < lo) | (valid > hi)).sum()
                info["out_of_range"] = int(out)
        results[feature] = info
    return results


def validate_lyrics(df: pd.DataFrame) -> Dict:
    """
    Valida presencia y calidad básica de la columna de letras.

    Returns
    -------
    Dict
        null_count, empty_count, total, coverage_pct.
    """
    if "lyrics" not in df.columns:
        return {
            "null_count": len(df),
            "empty_count": 0,
            "total": len(df),
            "coverage_pct": 0.0,
        }

    lyrics = df["lyrics"]
    null_count = int(lyrics.isnull().sum())
    empty_count = int((lyrics.fillna("").str.strip() == "").sum()) - null_count
    empty_count = max(empty_count, 0)
    valid = len(df) - null_count - empty_count

    return {
        "null_count": null_count,
        "empty_count": empty_count,
        "total": len(df),
        "valid_count": valid,
        "coverage_pct": round(100.0 * valid / len(df), 2) if len(df) > 0 else 0.0,
    }
