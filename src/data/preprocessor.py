"""
Modulo de preprocesamiento y seleccion del dataset.

Aplica filtros de calidad justificados sobre el dataset fuente
para producir un dataset limpio apto para vectorizacion semantica.

Cada filtro es una funcion pura que recibe y retorna un DataFrame.
La funcion preprocess_dataset() orquesta la secuencia completa.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import (
    LYRICS_MAX_WORDS,
    LYRICS_MIN_WORDS,
    SOURCE_SEPARATOR,
)
from src.data.eda import compute_data_loss_budget

logger = logging.getLogger(__name__)


def filter_null_lyrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con letras nulas o compuestas unicamente por whitespace.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columna 'lyrics'.

    Returns
    -------
    pd.DataFrame
        Dataset sin filas con letras nulas o vacias.
    """
    mask = df["lyrics"].notna() & (df["lyrics"].str.strip() != "")
    result = df[mask].copy()
    removed = len(df) - len(result)
    logger.info(
        "Filtro letras nulas/vacias: %d filas eliminadas (%d -> %d)",
        removed, len(df), len(result),
    )
    return result


def filter_by_word_count(
    df: pd.DataFrame,
    min_words: int = LYRICS_MIN_WORDS,
    max_words: int = LYRICS_MAX_WORDS,
) -> pd.DataFrame:
    """
    Elimina filas cuyas letras tienen un conteo de palabras fuera del rango permitido.

    Precondicion: las letras no deben contener valores nulos (aplicar
    filter_null_lyrics previamente).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columna 'lyrics' sin valores nulos.
    min_words : int
        Minimo de palabras requerido (inclusive).
    max_words : int
        Maximo de palabras permitido (inclusive).

    Returns
    -------
    pd.DataFrame
        Dataset filtrado al rango [min_words, max_words].
    """
    word_counts = df["lyrics"].str.split().str.len()
    mask = (word_counts >= min_words) & (word_counts <= max_words)
    result = df[mask].copy()

    too_short = int((word_counts < min_words).sum())
    too_long = int((word_counts > max_words).sum())
    logger.info(
        "Filtro por conteo de palabras [%d, %d]: "
        "%d cortas + %d largas = %d eliminadas (%d -> %d)",
        min_words, max_words, too_short, too_long,
        too_short + too_long, len(df), len(result),
    )
    return result


def clip_loudness_anomalies(
    df: pd.DataFrame, max_db: float = 0.0
) -> Tuple[pd.DataFrame, int]:
    """
    Corrige valores de loudness superiores al maximo fisicamente valido.

    Valores de loudness por encima de max_db se recortan (clip) a max_db.
    No se eliminan filas; es una correccion in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columna 'loudness'.
    max_db : float
        Valor maximo permitido en dB (por defecto, 0.0 dB).

    Returns
    -------
    Tuple[pd.DataFrame, int]
        (DataFrame corregido, cantidad de valores recortados).
    """
    result = df.copy()
    anomaly_mask = result["loudness"] > max_db
    n_clipped = int(anomaly_mask.sum())

    if n_clipped > 0:
        result.loc[anomaly_mask, "loudness"] = max_db
        logger.info(
            "Loudness: %d valores > %.1f dB recortados a %.1f dB",
            n_clipped, max_db, max_db,
        )
    else:
        logger.info("Loudness: sin valores anomalos (todos <= %.1f dB)", max_db)

    return result, n_clipped


def preprocess_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict], int]:
    """
    Aplica la secuencia completa de filtros de calidad sobre el dataset fuente.

    Secuencia de filtros:
    1. Eliminacion de letras nulas o vacias
    2. Eliminacion de letras con < LYRICS_MIN_WORDS palabras
    3. Eliminacion de letras con > LYRICS_MAX_WORDS palabras
    4. Correccion de valores de loudness > 0 dB (clip, sin remocion)

    Cada paso se documenta mediante compute_data_loss_budget().

    Parameters
    ----------
    df : pd.DataFrame
        Dataset fuente completo.

    Returns
    -------
    Tuple[pd.DataFrame, List[Dict], int]
        (DataFrame limpio, lista de registros de data loss budget,
         cantidad de valores de loudness recortados).
    """
    budget_entries: List[Dict] = []
    original_count = len(df)

    logger.info("Inicio de preprocesamiento: %d registros", original_count)

    # --- Filtro 1: Letras nulas/vacias ---
    df_filtered = filter_null_lyrics(df)
    budget_entries.append(
        compute_data_loss_budget(
            df, df_filtered,
            filter_name="Letras nulas o vacías",
            reason="Sin contenido textual para vectorización semántica",
        )
    )

    # --- Filtro 2: Letras demasiado cortas ---
    df_before_short = df_filtered
    # Aplicar solo filtro de minimo primero para registrar por separado
    word_counts = df_filtered["lyrics"].str.split().str.len()
    df_filtered = df_filtered[word_counts >= LYRICS_MIN_WORDS].copy()
    budget_entries.append(
        compute_data_loss_budget(
            df_before_short, df_filtered,
            filter_name=f"Letras < {LYRICS_MIN_WORDS} palabras",
            reason="Contenido semántico insuficiente para embedding significativo",
        )
    )

    # --- Filtro 3: Letras excesivamente largas ---
    df_before_long = df_filtered
    word_counts = df_filtered["lyrics"].str.split().str.len()
    df_filtered = df_filtered[word_counts <= LYRICS_MAX_WORDS].copy()
    budget_entries.append(
        compute_data_loss_budget(
            df_before_long, df_filtered,
            filter_name=f"Letras > {LYRICS_MAX_WORDS} palabras",
            reason="Probables artefactos de scraping; canción típica: 200-500 palabras",
        )
    )

    # --- Correccion 4: Loudness anomalo (clip, no remocion) ---
    df_filtered, n_loudness_clipped = clip_loudness_anomalies(df_filtered)

    final_count = len(df_filtered)
    total_lost = original_count - final_count
    total_pct = round(100.0 * total_lost / original_count, 2) if original_count > 0 else 0.0

    logger.info(
        "Preprocesamiento finalizado: %d -> %d (-%d, -%.2f%%)",
        original_count, final_count, total_lost, total_pct,
    )

    return df_filtered, budget_entries, n_loudness_clipped


def save_selected_dataset(df: pd.DataFrame, path: Path) -> None:
    """
    Guarda el dataset seleccionado en CSV con separador '@@'.

    pandas.to_csv() no soporta delimitadores multi-caracter (el modulo csv
    de Python solo acepta un unico caracter). Se utiliza un caracter nulo
    como placeholder intermedio y se reemplaza por '@@' en el resultado final.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio a guardar.
    path : Path
        Ruta de destino del archivo CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    placeholder = "\x00"
    content = df.to_csv(index=False, sep=placeholder)
    content = content.replace(placeholder, SOURCE_SEPARATOR)
    path.write_text(content, encoding="utf-8")
    size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    logger.info("Dataset guardado: %s (%d filas, %.2f MB)", path, len(df), size_mb)
