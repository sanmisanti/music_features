"""
Visualizacion para la Etapa 5 (recomendacion).

Genera figuras en formato PDF para el informe de tesis.
Cada figura se guarda en results/figures/ y thesis/figures/.
"""

import logging
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURES_DIR,
    THESIS_FIGURES_DIR,
)
from src.recommendation.evaluation import RecommendationMetrics
from src.recommendation.optimizer import OptimizationResult

logger = logging.getLogger(__name__)


def _setup_thesis_style():
    """Configura estilo de matplotlib para figuras de tesis academica."""
    plt.rcParams.update({
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
    })
    sns.set_palette("muted")


def _save_figure(fig: plt.Figure, name: str):
    """Guarda una figura en results/figures/ y thesis/figures/."""
    for target_dir in [FIGURES_DIR, THESIS_FIGURES_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{name}.{FIGURE_FORMAT}"
        fig.savefig(
            filepath, format=FIGURE_FORMAT, dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        logger.info("Figura guardada: %s", filepath)


def plot_alpha_vs_precision(opt: OptimizationResult) -> plt.Figure:
    """
    Lineas de Precision@K en funcion de alpha para distintos K.

    Parameters
    ----------
    opt : OptimizationResult

    Returns
    -------
    fig : plt.Figure
    """
    _setup_thesis_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    alphas = opt.alpha_range
    colors = sns.color_palette("muted", len(opt.k_values))

    for i, k in enumerate(sorted(opt.k_values)):
        precisions = [opt.results[a][k] for a in alphas]
        ax.plot(alphas, precisions, marker="o", markersize=3,
                color=colors[i], label=f"P@{k}", linewidth=1.5)

    ax.axvline(opt.best_alpha, color="gray", linestyle="--", linewidth=1,
               label=f"Óptimo ($\\alpha = {opt.best_alpha:.2f}$)")

    ax.set_xlabel("$\\alpha$ (peso semántico)")
    ax.set_ylabel("Precision@$K$")
    ax.set_title("Precisión en función del peso de fusión")
    ax.legend(loc="best")
    ax.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    _save_figure(fig, "recommendation_alpha_precision")
    return fig


def plot_precision_per_genre(metrics: RecommendationMetrics) -> plt.Figure:
    """
    Barplot horizontal de Precision@K por genero.

    Parameters
    ----------
    metrics : RecommendationMetrics

    Returns
    -------
    fig : plt.Figure
    """
    _setup_thesis_style()

    genres = sorted([g for g in metrics.precision_per_genre.keys()
                     if not g.startswith("_")])
    values = [metrics.precision_per_genre[g] for g in genres]

    fig, ax = plt.subplots(figsize=(7, 4))

    colors = sns.color_palette("muted", len(genres))
    y_pos = range(len(genres))

    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genres)
    ax.set_xlabel(f"Precision@{metrics.top_k}")
    ax.set_title(f"Precision@{metrics.top_k} por género ($\\alpha = {metrics.alpha:.2f}$)")

    # Linea de media
    ax.axvline(metrics.precision_at_k, color="gray", linestyle="--",
               linewidth=1, label=f"Media ({metrics.precision_at_k:.3f})")
    ax.legend(loc="best")

    fig.tight_layout()
    _save_figure(fig, "recommendation_precision_genre")
    return fig


def plot_score_distributions(
    scores_relevant: np.ndarray,
    scores_non_relevant: np.ndarray,
) -> plt.Figure:
    """
    Histogramas de scores de fusion para recomendaciones relevantes
    (mismo genero) vs no relevantes.

    Parameters
    ----------
    scores_relevant : np.ndarray
        Scores de fusion de recomendaciones que comparten genero con la query.
    scores_non_relevant : np.ndarray
        Scores de recomendaciones que no comparten genero.

    Returns
    -------
    fig : plt.Figure
    """
    _setup_thesis_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(scores_non_relevant, bins=80, alpha=0.5, density=True,
            label="No relevantes", color="lightcoral", edgecolor="none")
    ax.hist(scores_relevant, bins=80, alpha=0.6, density=True,
            label="Relevantes (mismo género)", color="mediumseagreen",
            edgecolor="none")

    ax.set_xlabel("Score de fusión")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de scores: relevantes vs. no relevantes")
    ax.legend(loc="best")

    fig.tight_layout()
    _save_figure(fig, "recommendation_score_distributions")
    return fig


def plot_diversity_coverage(
    opt: OptimizationResult,
    metrics_per_alpha: Dict[float, RecommendationMetrics],
) -> plt.Figure:
    """
    ILD y cobertura del catalogo en funcion de alpha.

    Parameters
    ----------
    opt : OptimizationResult
    metrics_per_alpha : dict
        {alpha: RecommendationMetrics} para un subconjunto de alphas.

    Returns
    -------
    fig : plt.Figure
    """
    _setup_thesis_style()

    alphas = sorted(metrics_per_alpha.keys())
    ild_sem = [metrics_per_alpha[a].ild_semantic for a in alphas]
    coverage = [metrics_per_alpha[a].catalog_coverage for a in alphas]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = sns.color_palette("muted")[0]
    color2 = sns.color_palette("muted")[2]

    ax1.plot(alphas, ild_sem, marker="o", markersize=3, color=color1,
             label="ILD semántica", linewidth=1.5)
    ax1.set_xlabel("$\\alpha$ (peso semántico)")
    ax1.set_ylabel("ILD semántica", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(alphas, coverage, marker="s", markersize=3, color=color2,
             label="Cobertura", linewidth=1.5)
    ax2.set_ylabel("Cobertura del catálogo", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.axvline(opt.best_alpha, color="gray", linestyle="--", linewidth=1)

    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title("Diversidad y cobertura en función del peso de fusión")
    ax1.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    _save_figure(fig, "recommendation_diversity_coverage")
    return fig
