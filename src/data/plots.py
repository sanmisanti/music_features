"""
Módulo de visualización para el EDA.

Genera 7 figuras en formato PDF para el informe de tesis.
Cada figura se guarda en results/figures/ y thesis/figures/.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURES_DIR,
    MUSICAL_FEATURES,
    THESIS_FIGURES_DIR,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURACIÓN DE ESTILO
# =============================================================================


def setup_thesis_style():
    """Configura estilo de matplotlib para figuras de tesis académica."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_palette("muted")


def save_figure(fig: plt.Figure, name: str):
    """
    Guarda una figura en results/figures/ y thesis/figures/.

    Parameters
    ----------
    fig : plt.Figure
    name : str
        Nombre base del archivo (sin extensión).
    """
    for target_dir in [FIGURES_DIR, THESIS_FIGURES_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{name}.{FIGURE_FORMAT}"
        fig.savefig(filepath, format=FIGURE_FORMAT, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info("Figura guardada: %s", filepath)

    plt.close(fig)


# =============================================================================
# FIGURAS
# =============================================================================


def plot_feature_distributions(df: pd.DataFrame) -> plt.Figure:
    """
    F1: Histogramas de distribución de las 12 features musicales.
    """
    setup_thesis_style()
    features = [f for f in MUSICAL_FEATURES if f in df.columns]
    n = len(features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data = df[feat].dropna()
        ax.hist(data, bins=50, color=sns.color_palette("muted")[0], edgecolor="white", linewidth=0.5)
        ax.set_title(feat, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia")
        # Línea de media
        mean_val = data.mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1, alpha=0.7)

    # Ocultar ejes sobrantes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribución de features musicales", fontweight="bold", y=1.02)
    fig.tight_layout()

    save_figure(fig, "feature_distributions")
    return fig


def plot_feature_boxplots(df: pd.DataFrame) -> plt.Figure:
    """
    F2: Boxplots de las 12 features musicales (normalizadas para comparación).
    """
    setup_thesis_style()
    features = [f for f in MUSICAL_FEATURES if f in df.columns]

    # Normalizar min-max para visualización comparativa
    data_norm = pd.DataFrame()
    for feat in features:
        col = df[feat].dropna()
        min_val, max_val = col.min(), col.max()
        if max_val > min_val:
            data_norm[feat] = (col - min_val) / (max_val - min_val)
        else:
            data_norm[feat] = 0.0

    fig, ax = plt.subplots(figsize=(14, 6))
    melted = data_norm.melt(var_name="Feature", value_name="Valor normalizado")
    sns.boxplot(
        data=melted,
        x="Feature",
        y="Valor normalizado",
        ax=ax,
        fliersize=2,
        linewidth=0.8,
    )
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.set_title("Boxplots de features musicales (min-max normalizado)", fontweight="bold")
    ax.set_xlabel("")
    fig.tight_layout()

    save_figure(fig, "feature_boxplots")
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    """
    F3: Mapa de calor de la matriz de correlación.
    """
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlación"},
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Matriz de correlación de features musicales", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    save_figure(fig, "correlation_matrix")
    return fig


def plot_genre_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    F4: Distribución de canciones por género (barras horizontales).
    """
    setup_thesis_style()

    if "playlist_genre" not in df.columns:
        logger.warning("Columna 'playlist_genre' no encontrada")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin datos de género", ha="center", va="center")
        save_figure(fig, "genre_distribution")
        return fig

    genre_counts = df["playlist_genre"].value_counts().sort_values()
    total = genre_counts.sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("muted", n_colors=len(genre_counts))
    bars = ax.barh(genre_counts.index, genre_counts.values, color=colors, edgecolor="white")

    # Anotar con conteo y porcentaje
    for bar, count in zip(bars, genre_counts.values):
        pct = 100.0 * count / total
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Número de canciones")
    ax.set_title("Distribución por género musical", fontweight="bold")
    ax.set_xlim(0, genre_counts.max() * 1.18)
    fig.tight_layout()

    save_figure(fig, "genre_distribution")
    return fig


def plot_lyrics_length_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    F5: Distribución de longitud de letras (palabras) con línea de ventana de tokens.
    """
    setup_thesis_style()

    if "lyrics" not in df.columns:
        logger.warning("Columna 'lyrics' no encontrada")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin datos de letras", ha="center", va="center")
        save_figure(fig, "lyrics_length_distribution")
        return fig

    lyrics = df["lyrics"].dropna()
    lyrics = lyrics[lyrics.str.strip() != ""]
    word_counts = lyrics.str.split().str.len()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(
        word_counts,
        bins=100,
        color=sns.color_palette("muted")[1],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
    )

    # Línea de ventana de tokens (~385 palabras para 512 tokens)
    window_words = int(512 / 1.33)
    ax.axvline(
        window_words,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Ventana E5 (~{window_words} palabras / 512 tokens)",
    )

    ax.set_xlabel("Número de palabras")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de longitud de letras", fontweight="bold")
    ax.legend()

    # Estadísticas en texto
    stats_text = (
        f"N={len(word_counts):,}\n"
        f"Media={word_counts.mean():.0f}\n"
        f"Mediana={word_counts.median():.0f}"
    )
    ax.text(
        0.97,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()

    save_figure(fig, "lyrics_length_distribution")
    return fig


def plot_language_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    F6: Distribución de idiomas (top 10 + otros).
    """
    setup_thesis_style()

    if "language" not in df.columns:
        logger.warning("Columna 'language' no encontrada")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin datos de idioma", ha="center", va="center")
        save_figure(fig, "language_distribution")
        return fig

    lang = df["language"].fillna("unknown")
    lang_counts = lang.value_counts()
    total = lang_counts.sum()

    # Top 10 + otros
    if len(lang_counts) > 10:
        top10 = lang_counts.head(10)
        others = lang_counts.iloc[10:].sum()
        plot_data = pd.concat([top10, pd.Series({"otros": others})])
    else:
        plot_data = lang_counts

    plot_data = plot_data.sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("muted", n_colors=len(plot_data))
    bars = ax.barh(plot_data.index, plot_data.values, color=colors, edgecolor="white")

    for bar, count in zip(bars, plot_data.values):
        pct = 100.0 * count / total
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Número de canciones")
    ax.set_title("Distribución por idioma", fontweight="bold")
    ax.set_xlim(0, plot_data.max() * 1.18)
    fig.tight_layout()

    save_figure(fig, "language_distribution")
    return fig


def plot_features_by_genre(df: pd.DataFrame) -> plt.Figure:
    """
    F7: Comparación de features musicales por género (violin plots).
    """
    setup_thesis_style()

    if "playlist_genre" not in df.columns:
        logger.warning("Columna 'playlist_genre' no encontrada")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin datos de género", ha="center", va="center")
        save_figure(fig, "features_by_genre")
        return fig

    # Seleccionar features 0-1 para comparación directa
    bounded_features = [
        f for f in MUSICAL_FEATURES
        if f in df.columns and f not in ("key", "loudness", "mode", "tempo", "duration_ms")
    ]

    n = len(bounded_features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(bounded_features):
        ax = axes[i]
        sns.violinplot(
            data=df,
            x="playlist_genre",
            y=feat,
            ax=ax,
            inner="quartile",
            linewidth=0.8,
            cut=0,
        )
        ax.set_title(feat, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distribución de features musicales por género",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    save_figure(fig, "features_by_genre")
    return fig
