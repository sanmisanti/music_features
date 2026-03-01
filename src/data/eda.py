"""
Módulo de análisis exploratorio de datos (EDA).

Contiene funciones para estadísticas descriptivas, análisis de letras,
correlaciones, y análisis de sesgo del dataset.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    LYRICS_MAX_WORDS,
    LYRICS_MIN_WORDS,
    MUSICAL_FEATURES,
    PLAYLIST_GENRES,
    BERT_TARGET_TOKENS,
)

logger = logging.getLogger(__name__)

# Factor de estimación: 1 token ~ 0.75 palabras para texto en inglés/español
TOKENS_PER_WORD_ESTIMATE = 1.33


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class FeatureProfile:
    """Perfil estadístico completo de una feature numérica."""

    name: str
    count: int
    missing: int
    missing_pct: float
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    skewness: float
    kurtosis: float
    iqr: float
    outliers_iqr: int
    outliers_pct: float


# =============================================================================
# 1. ESTADÍSTICAS DESCRIPTIVAS
# =============================================================================


def compute_feature_profiles(
    df: pd.DataFrame, features: Optional[List[str]] = None
) -> Dict[str, FeatureProfile]:
    """
    Calcula perfil estadístico completo para cada feature numérica.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con las features.
    features : List[str], optional
        Lista de columnas a analizar. Por defecto, MUSICAL_FEATURES.

    Returns
    -------
    Dict[str, FeatureProfile]
        Diccionario feature_name -> FeatureProfile.
    """
    if features is None:
        features = MUSICAL_FEATURES

    profiles = {}
    for feat in features:
        if feat not in df.columns:
            logger.warning("Feature '%s' no encontrada en el DataFrame", feat)
            continue

        col = df[feat].dropna()
        n_total = len(df)
        n_missing = n_total - len(col)
        q1 = float(col.quantile(0.25))
        q3 = float(col.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = int(((col < lower_fence) | (col > upper_fence)).sum())

        profiles[feat] = FeatureProfile(
            name=feat,
            count=len(col),
            missing=n_missing,
            missing_pct=round(100.0 * n_missing / n_total, 2) if n_total > 0 else 0.0,
            mean=round(float(col.mean()), 4),
            std=round(float(col.std()), 4),
            min=round(float(col.min()), 4),
            q1=round(q1, 4),
            median=round(float(col.median()), 4),
            q3=round(q3, 4),
            max=round(float(col.max()), 4),
            skewness=round(float(col.skew()), 4),
            kurtosis=round(float(col.kurtosis()), 4),
            iqr=round(iqr, 4),
            outliers_iqr=outliers,
            outliers_pct=round(100.0 * outliers / len(col), 2) if len(col) > 0 else 0.0,
        )

    logger.info("Perfiles calculados para %d features", len(profiles))
    return profiles


# =============================================================================
# 2. ANÁLISIS DE LETRAS
# =============================================================================


def analyze_lyrics_quality(df: pd.DataFrame) -> Dict:
    """
    Analiza la calidad y distribución de longitudes del campo 'lyrics'.

    Returns
    -------
    Dict
        Estadísticas de cobertura, longitudes (caracteres, palabras, tokens estimados),
        cobertura de la ventana de tokens del modelo E5, y ratio tipo/token.
    """
    if "lyrics" not in df.columns:
        logger.error("Columna 'lyrics' no encontrada")
        return {}

    lyrics = df["lyrics"].copy()
    total = len(lyrics)
    null_mask = lyrics.isnull()
    n_null = int(null_mask.sum())

    # Reemplazar nulos para cálculos de longitud
    lyrics_filled = lyrics.fillna("")

    # Longitudes
    char_lengths = lyrics_filled.str.len()
    word_counts = lyrics_filled.str.split().str.len().fillna(0).astype(int)
    token_estimates = (word_counts * TOKENS_PER_WORD_ESTIMATE).astype(int)

    # Mascara de letras válidas (no nulas, con contenido)
    valid_mask = ~null_mask & (char_lengths > 0)
    n_valid = int(valid_mask.sum())
    n_empty = total - n_null - n_valid

    # Estadísticas solo sobre letras válidas
    valid_words = word_counts[valid_mask]
    valid_tokens = token_estimates[valid_mask]
    valid_chars = char_lengths[valid_mask]

    # Cobertura de ventana de tokens
    within_window = int((valid_tokens <= BERT_TARGET_TOKENS).sum())
    window_coverage_pct = (
        round(100.0 * within_window / n_valid, 2) if n_valid > 0 else 0.0
    )

    # Filtros de calidad por palabras
    too_short = int((valid_words < LYRICS_MIN_WORDS).sum())
    too_long = int((valid_words > LYRICS_MAX_WORDS).sum())
    in_range = n_valid - too_short - too_long

    # Type-token ratio (vocabulario / total palabras) sobre muestra
    ttr_values = []
    sample_lyrics = lyrics_filled[valid_mask]
    if len(sample_lyrics) > 0:
        # Calcular TTR sobre cada letra individualmente, luego promediar
        for text in sample_lyrics.head(5000):
            words = text.lower().split()
            if len(words) > 0:
                ttr_values.append(len(set(words)) / len(words))

    avg_ttr = round(float(np.mean(ttr_values)), 4) if ttr_values else 0.0

    result = {
        "total": total,
        "null_count": n_null,
        "empty_count": n_empty,
        "valid_count": n_valid,
        "coverage_pct": round(100.0 * n_valid / total, 2) if total > 0 else 0.0,
        "char_length": {
            "mean": round(float(valid_chars.mean()), 1) if n_valid > 0 else 0,
            "median": round(float(valid_chars.median()), 1) if n_valid > 0 else 0,
            "std": round(float(valid_chars.std()), 1) if n_valid > 0 else 0,
            "min": int(valid_chars.min()) if n_valid > 0 else 0,
            "max": int(valid_chars.max()) if n_valid > 0 else 0,
        },
        "word_count": {
            "mean": round(float(valid_words.mean()), 1) if n_valid > 0 else 0,
            "median": round(float(valid_words.median()), 1) if n_valid > 0 else 0,
            "std": round(float(valid_words.std()), 1) if n_valid > 0 else 0,
            "min": int(valid_words.min()) if n_valid > 0 else 0,
            "max": int(valid_words.max()) if n_valid > 0 else 0,
        },
        "token_estimate": {
            "mean": round(float(valid_tokens.mean()), 1) if n_valid > 0 else 0,
            "median": round(float(valid_tokens.median()), 1) if n_valid > 0 else 0,
            "std": round(float(valid_tokens.std()), 1) if n_valid > 0 else 0,
            "min": int(valid_tokens.min()) if n_valid > 0 else 0,
            "max": int(valid_tokens.max()) if n_valid > 0 else 0,
        },
        "token_window_coverage": {
            "window_size": BERT_TARGET_TOKENS,
            "within_window": within_window,
            "coverage_pct": window_coverage_pct,
        },
        "quality_filter": {
            "min_words": LYRICS_MIN_WORDS,
            "max_words": LYRICS_MAX_WORDS,
            "too_short": too_short,
            "too_long": too_long,
            "in_range": in_range,
            "in_range_pct": round(100.0 * in_range / n_valid, 2) if n_valid > 0 else 0.0,
        },
        "avg_type_token_ratio": avg_ttr,
    }

    logger.info(
        "Análisis de letras: %d válidas de %d (%.1f%%), cobertura ventana %d tokens: %.1f%%",
        n_valid,
        total,
        result["coverage_pct"],
        BERT_TARGET_TOKENS,
        window_coverage_pct,
    )
    return result


# =============================================================================
# 3. CORRELACIONES
# =============================================================================


def compute_correlation_matrix(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Calcula la matriz de correlación entre features numéricas.

    Parameters
    ----------
    df : pd.DataFrame
    features : List[str], optional
        Por defecto, MUSICAL_FEATURES.
    method : str
        Método de correlación ('pearson', 'spearman', 'kendall').

    Returns
    -------
    pd.DataFrame
        Matriz de correlación NxN.
    """
    if features is None:
        features = MUSICAL_FEATURES

    available = [f for f in features if f in df.columns]
    corr = df[available].corr(method=method)
    logger.info("Matriz de correlación %s: %dx%d", method, len(available), len(available))
    return corr


def identify_high_correlations(
    corr_matrix: pd.DataFrame, threshold: float = 0.5
) -> List[Dict]:
    """
    Identifica pares de features con correlación superior al umbral.

    Returns
    -------
    List[Dict]
        Lista de dicts con feature_1, feature_2, correlation, abs_correlation.
    """
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append(
                    {
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "correlation": round(float(val), 4),
                        "abs_correlation": round(abs(float(val)), 4),
                    }
                )

    pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    logger.info(
        "Correlaciones >= %.2f: %d pares encontrados", threshold, len(pairs)
    )
    return pairs


# =============================================================================
# 4. ANÁLISIS DE SESGO
# =============================================================================


def analyze_genre_balance(df: pd.DataFrame) -> Dict:
    """
    Analiza la distribución por género y subgénero del dataset.

    Calcula frecuencias absolutas, relativas, y entropía normalizada
    como medida de balance (1.0 = perfectamente balanceado).

    Returns
    -------
    Dict
        genre_distribution, subgenre_distribution, normalized_entropy,
        balance_assessment.
    """
    result = {"genre_distribution": {}, "subgenre_distribution": {}}

    if "playlist_genre" in df.columns:
        genre_counts = df["playlist_genre"].value_counts()
        total = genre_counts.sum()
        result["genre_distribution"] = {
            genre: {
                "count": int(count),
                "pct": round(100.0 * count / total, 2),
            }
            for genre, count in genre_counts.items()
        }

        # Entropía normalizada: H / log(n_classes)
        probs = genre_counts.values / total
        entropy = float(stats.entropy(probs, base=2))
        max_entropy = np.log2(len(genre_counts))
        normalized_entropy = round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0

        result["n_genres"] = len(genre_counts)
        result["normalized_entropy"] = normalized_entropy
        result["balance_assessment"] = (
            "alto" if normalized_entropy >= 0.95
            else "moderado" if normalized_entropy >= 0.85
            else "bajo"
        )

    if "playlist_subgenre" in df.columns:
        subgenre_counts = df["playlist_subgenre"].value_counts()
        total_sub = subgenre_counts.sum()
        result["subgenre_distribution"] = {
            sg: {"count": int(c), "pct": round(100.0 * c / total_sub, 2)}
            for sg, c in subgenre_counts.items()
        }
        result["n_subgenres"] = len(subgenre_counts)

    logger.info(
        "Balance de género: %d géneros, entropía normalizada=%.4f (%s)",
        result.get("n_genres", 0),
        result.get("normalized_entropy", 0),
        result.get("balance_assessment", "N/A"),
    )
    return result


def analyze_language_distribution(df: pd.DataFrame) -> Dict:
    """
    Analiza la distribución lingüística del dataset.

    Returns
    -------
    Dict
        Distribución por idioma, porcentaje de inglés, entropía.
    """
    if "language" not in df.columns:
        logger.warning("Columna 'language' no encontrada")
        return {}

    lang = df["language"].fillna("unknown")
    lang_counts = lang.value_counts()
    total = lang_counts.sum()

    distribution = {
        language: {"count": int(count), "pct": round(100.0 * count / total, 2)}
        for language, count in lang_counts.items()
    }

    english_pct = distribution.get("en", {}).get("pct", 0.0)

    probs = lang_counts.values / total
    entropy = float(stats.entropy(probs, base=2))
    max_entropy = np.log2(len(lang_counts)) if len(lang_counts) > 1 else 1.0
    normalized_entropy = round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0

    result = {
        "distribution": distribution,
        "n_languages": len(lang_counts),
        "english_pct": english_pct,
        "normalized_entropy": normalized_entropy,
        "dominant_language": lang_counts.index[0] if len(lang_counts) > 0 else "N/A",
    }

    logger.info(
        "Distribución lingüística: %d idiomas, inglés=%.1f%%, entropía=%.4f",
        result["n_languages"],
        english_pct,
        normalized_entropy,
    )
    return result


def analyze_instrumentalness_distribution(df: pd.DataFrame) -> Dict:
    """
    Analiza la distribución de instrumentalidad como indicador de sesgo.

    Spotify define instrumentalness > 0.5 como probable instrumental.

    Returns
    -------
    Dict
        Proporción vocal vs instrumental, estadísticas descriptivas.
    """
    if "instrumentalness" not in df.columns:
        logger.warning("Columna 'instrumentalness' no encontrada")
        return {}

    col = df["instrumentalness"].dropna()
    threshold = 0.5
    n_instrumental = int((col > threshold).sum())
    n_vocal = int((col <= threshold).sum())
    total = len(col)

    result = {
        "total": total,
        "n_instrumental": n_instrumental,
        "n_vocal": n_vocal,
        "instrumental_pct": round(100.0 * n_instrumental / total, 2) if total > 0 else 0.0,
        "vocal_pct": round(100.0 * n_vocal / total, 2) if total > 0 else 0.0,
        "threshold": threshold,
        "mean": round(float(col.mean()), 4),
        "median": round(float(col.median()), 4),
        "std": round(float(col.std()), 4),
        "pct_near_zero": round(100.0 * (col < 0.01).sum() / total, 2) if total > 0 else 0.0,
    }

    logger.info(
        "Instrumentalidad: %.1f%% vocal, %.1f%% instrumental (umbral=%.1f)",
        result["vocal_pct"],
        result["instrumental_pct"],
        threshold,
    )
    return result


def compute_data_loss_budget(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    filter_name: str,
    reason: str,
) -> Dict:
    """
    Registra explícitamente la pérdida de datos producida por un filtrado.

    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame antes del filtrado.
    df_after : pd.DataFrame
        DataFrame después del filtrado.
    filter_name : str
        Nombre identificador del filtro aplicado.
    reason : str
        Justificación técnica de la exclusión.

    Returns
    -------
    Dict
        Registro del presupuesto de pérdida.
    """
    n_before = len(df_before)
    n_after = len(df_after)
    n_lost = n_before - n_after

    entry = {
        "filter_name": filter_name,
        "reason": reason,
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_lost": n_lost,
        "loss_pct": round(100.0 * n_lost / n_before, 2) if n_before > 0 else 0.0,
        "cumulative_remaining_pct": (
            round(100.0 * n_after / n_before, 2) if n_before > 0 else 0.0
        ),
    }

    logger.info(
        "Data loss budget [%s]: %d -> %d (-%d, -%.1f%%) | Razón: %s",
        filter_name,
        n_before,
        n_after,
        n_lost,
        entry["loss_pct"],
        reason,
    )
    return entry


def compute_dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Genera un resumen general del dataset para la tabla de overview.

    Returns
    -------
    Dict
        Resumen con métricas generales del dataset.
    """
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "n_musical_features": sum(1 for f in MUSICAL_FEATURES if f in df.columns),
        "n_metadata_columns": len(df.columns) - sum(
            1 for f in MUSICAL_FEATURES if f in df.columns
        ),
        "total_missing": int(df.isnull().sum().sum()),
        "total_missing_pct": round(
            100.0 * df.isnull().sum().sum() / (len(df) * len(df.columns)), 2
        ),
        "duplicate_track_ids": (
            int(df["track_id"].duplicated().sum()) if "track_id" in df.columns else 0
        ),
    }

    # Cobertura de letras
    if "lyrics" in df.columns:
        lyrics = df["lyrics"]
        valid_lyrics = lyrics.dropna()
        valid_lyrics = valid_lyrics[valid_lyrics.str.strip() != ""]
        summary["lyrics_coverage_pct"] = round(
            100.0 * len(valid_lyrics) / len(df), 2
        )
    else:
        summary["lyrics_coverage_pct"] = 0.0

    # Géneros
    if "playlist_genre" in df.columns:
        summary["n_genres"] = df["playlist_genre"].nunique()
    if "playlist_subgenre" in df.columns:
        summary["n_subgenres"] = df["playlist_subgenre"].nunique()

    return summary
