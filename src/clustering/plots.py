"""
Modulo de visualizacion para la Etapa 4 (clustering).

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
    HOPKINS_THRESHOLD,
    THESIS_FIGURES_DIR,
)
from src.clustering.hopkins import HopkinsReport
from src.clustering.evaluation import ClusteringMetrics
from src.clustering.purification import PurificationReport

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
    plt.close(fig)


def plot_hopkins_barplot(reports: List[HopkinsReport]) -> plt.Figure:
    """
    Barplot horizontal con Hopkins por espacio.

    Incluye barras de error (IC 95%), linea vertical en H=0.7
    (threshold de tendencia significativa) y H=0.5 (aleatorio).

    Parameters
    ----------
    reports : list of HopkinsReport
        Reportes de Hopkins para cada espacio.

    Returns
    -------
    fig : plt.Figure
        Figura generada.
    """
    _setup_thesis_style()

    display_names = {
        "semantic_384d": "Semántico (384D)",
        "semantic_umap": "Semántico (UMAP)",
        "musical_13d": "Musical (13D)",
        "concatenated": "Concatenado (397D)",
    }

    names = [display_names.get(r.space_name, r.space_name) for r in reports]
    means = [r.mean for r in reports]
    ci_low = [r.mean - r.ci_lower for r in reports]
    ci_high = [r.ci_upper - r.mean for r in reports]

    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(reports))
    colors = sns.color_palette("muted", len(reports))

    ax.barh(
        y_pos, means, xerr=[ci_low, ci_high],
        color=colors, edgecolor="black", linewidth=0.5,
        capsize=4, height=0.6,
    )

    # Lineas de referencia
    ax.axvline(
        x=0.5, color="gray", linestyle="--", linewidth=1,
        label="$H = 0{,}5$ (aleatorio)",
    )
    ax.axvline(
        x=HOPKINS_THRESHOLD, color="red", linestyle="--", linewidth=1,
        label=f"$H = {HOPKINS_THRESHOLD}$ (umbral)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Estadístico de Hopkins ($H$)")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Tendencia al agrupamiento por espacio de representación")

    fig.tight_layout()
    _save_figure(fig, "hopkins_results")
    return fig


# =========================================================================
# FIGURAS DE FASE B: MULTI-ALGORITMO
# =========================================================================

_SPACE_DISPLAY = {
    "semantic_384d": "Semántico (384D)",
    "semantic_umap": "Semántico (UMAP 30D)",
    "musical_13d": "Musical (13D)",
    "concatenated": "Concatenado (397D)",
}


def plot_silhouette_comparison(
    metrics_list: List[ClusteringMetrics],
    space_name: str,
) -> plt.Figure:
    """
    Grouped barplot: eje X = K, barras agrupadas por algoritmo,
    eje Y = silhouette macro.

    Solo incluye configuraciones no degeneradas con >= 2 clusters.

    Parameters
    ----------
    metrics_list : list of ClusteringMetrics
        Metricas de todas las configuraciones de un espacio.
    space_name : str
        Nombre del espacio.

    Returns
    -------
    fig : plt.Figure
    """
    _setup_thesis_style()

    # Separar por algoritmo (excluir degeneradas y < 2 clusters)
    valid = [
        m for m in metrics_list
        if not m.has_degenerate_clusters and m.n_clusters_found >= 2
    ]

    algorithms = sorted(set(m.algorithm for m in valid))
    algo_colors = dict(zip(algorithms, sns.color_palette("muted", len(algorithms))))

    fig, ax = plt.subplots(figsize=(9, 5))

    # Para K-Means y Ward: agrupar por K
    # Para HDBSCAN: usar n_clusters_found como eje X
    partition_algos = [a for a in algorithms if a in ("kmeans", "ward")]
    density_algos = [a for a in algorithms if a == "hdbscan"]

    # Obtener todos los K para algoritmos de particion
    k_values = sorted(set(
        m.k for m in valid if m.algorithm in partition_algos
    ))

    if k_values:
        n_algos = len(partition_algos)
        bar_width = 0.35
        x = np.arange(len(k_values))

        for i, algo in enumerate(partition_algos):
            algo_metrics = [m for m in valid if m.algorithm == algo]
            sils = []
            for k in k_values:
                match = [m for m in algo_metrics if m.k == k]
                sils.append(match[0].silhouette_macro if match else 0)

            offset = (i - (n_algos - 1) / 2) * bar_width
            label = algo.upper() if algo == "hdbscan" else algo.capitalize()
            ax.bar(
                x + offset, sils, bar_width,
                label=label, color=algo_colors[algo],
                edgecolor="black", linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_values])

    # HDBSCAN como puntos separados si hay
    for algo in density_algos:
        hdb_metrics = sorted(
            [m for m in valid if m.algorithm == algo],
            key=lambda m: m.k,
        )
        if hdb_metrics:
            hdb_ks = [m.k for m in hdb_metrics]
            hdb_sils = [m.silhouette_macro for m in hdb_metrics]
            ax.scatter(
                [len(k_values) + 0.5 + i * 0.3 for i in range(len(hdb_ks))],
                hdb_sils,
                color=algo_colors[algo], marker="D", s=60, zorder=5,
                label="HDBSCAN", edgecolors="black", linewidth=0.5,
            )
            for i, (hk, hs) in enumerate(zip(hdb_ks, hdb_sils)):
                ax.annotate(
                    f"k={hk}", (len(k_values) + 0.5 + i * 0.3, hs),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, ha="center",
                )

    space_display = _SPACE_DISPLAY.get(space_name, space_name)
    ax.set_xlabel("Número de clusters ($K$)")
    ax.set_ylabel("Silhouette macro-averaged")
    ax.set_title(f"Comparación de algoritmos --- {space_display}")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    _save_figure(fig, f"silhouette_comparison_{space_name}")
    return fig


def plot_cluster_sizes(
    cluster_sizes: Dict[int, int],
    algorithm: str,
    space_name: str,
    k: int,
) -> plt.Figure:
    """
    Barplot de distribucion de tamanos de clusters.

    Parameters
    ----------
    cluster_sizes : dict
        {cluster_id: count}
    algorithm : str
        Nombre del algoritmo.
    space_name : str
        Nombre del espacio.
    k : int
        Numero de clusters.

    Returns
    -------
    fig : plt.Figure
    """
    _setup_thesis_style()

    sorted_items = sorted(cluster_sizes.items())
    labels_c = [f"C{c}" for c, _ in sorted_items]
    counts = [count for _, count in sorted_items]
    total = sum(counts)
    pcts = [100.0 * c / total for c in counts]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("muted", len(labels_c))

    bars = ax.bar(
        labels_c, counts, color=colors,
        edgecolor="black", linewidth=0.5,
    )

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=8,
        )

    algo_display = algorithm.upper() if algorithm == "hdbscan" else algorithm.capitalize()
    space_display = _SPACE_DISPLAY.get(space_name, space_name)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Número de canciones")
    ax.set_title(f"Distribución de clusters --- {algo_display} $K={k}$, {space_display}")

    fig.tight_layout()
    name = f"cluster_sizes_{algorithm}_{space_name}_k{k}"
    _save_figure(fig, name)
    return fig


# =========================================================================
# FIGURAS DE FASE C: PURIFICACION Y VISUALIZACION UMAP
# =========================================================================


def plot_umap_clusters(
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    space_name: str,
    algorithm: str,
    k: int,
) -> plt.Figure:
    """
    Scatter plot UMAP 2D coloreado por asignacion de cluster.

    Parameters
    ----------
    embedding_2d : np.ndarray
        Proyeccion 2D, shape [N, 2].
    labels : np.ndarray
        Asignaciones de cluster, shape [N]. -1 = noise.
    space_name : str
        Nombre del espacio original.
    algorithm : str
        Nombre del algoritmo.
    k : int
        Numero de clusters.

    Returns
    -------
    fig : plt.Figure
    """
    _setup_thesis_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(set(labels))
    colors = sns.color_palette("muted", max(len(unique_labels), 1))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                c="lightgray", s=3, alpha=0.3, label="Noise",
            )
        else:
            ax.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                c=[colors[i % len(colors)]], s=5, alpha=0.5,
                label=f"C{label} ({int(mask.sum()):,})",
            )

    algo_display = algorithm.upper() if algorithm == "hdbscan" else algorithm.capitalize()
    space_display = _SPACE_DISPLAY.get(space_name, space_name)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Clusters {algo_display} $K={k}$ --- {space_display}")
    ax.legend(loc="best", fontsize=7, markerscale=3)

    fig.tight_layout()
    _save_figure(fig, f"umap_clusters_{space_name}")
    return fig


def plot_umap_genres(
    embedding_2d: np.ndarray,
    genre_labels: np.ndarray,
    space_name: str,
) -> plt.Figure:
    """
    Scatter plot UMAP 2D coloreado por genero (validacion visual).

    Parameters
    ----------
    embedding_2d : np.ndarray
        Proyeccion 2D, shape [N, 2].
    genre_labels : np.ndarray
        Etiquetas de genero, shape [N].
    space_name : str
        Nombre del espacio original.

    Returns
    -------
    fig : plt.Figure
    """
    _setup_thesis_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    unique_genres = sorted(set(genre_labels))
    colors = sns.color_palette("muted", len(unique_genres))

    for i, genre in enumerate(unique_genres):
        mask = genre_labels == genre
        ax.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=[colors[i]], s=5, alpha=0.4,
            label=f"{genre} ({int(mask.sum()):,})",
        )

    space_display = _SPACE_DISPLAY.get(space_name, space_name)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Géneros musicales --- {space_display}")
    ax.legend(loc="best", fontsize=7, markerscale=3)

    fig.tight_layout()
    _save_figure(fig, f"umap_genres_{space_name}")
    return fig


def plot_purification_comparison(
    reports: List[PurificationReport],
) -> plt.Figure:
    """
    Barplot side-by-side pre/post purificacion para Silhouette macro y DB.

    Parameters
    ----------
    reports : list of PurificationReport

    Returns
    -------
    fig : plt.Figure
    """
    _setup_thesis_style()

    n_reports = len(reports)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(n_reports)
    width = 0.35
    space_labels = [
        _SPACE_DISPLAY.get(r.space_name, r.space_name) for r in reports
    ]

    # Silhouette macro
    ax = axes[0]
    pre_sil = [r.silhouette_macro_before for r in reports]
    post_sil = [r.silhouette_macro_after for r in reports]
    ax.bar(x - width / 2, pre_sil, width, label="Pre", color="lightcoral", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, post_sil, width, label="Post", color="mediumseagreen", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(space_labels, fontsize=8)
    ax.set_ylabel("Silhouette macro")
    ax.set_title("Silhouette macro-averaged")
    ax.legend(fontsize=8)

    # Davies-Bouldin
    ax = axes[1]
    pre_db = [r.davies_bouldin_before for r in reports]
    post_db = [r.davies_bouldin_after for r in reports]
    ax.bar(x - width / 2, pre_db, width, label="Pre", color="lightcoral", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, post_db, width, label="Post", color="mediumseagreen", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(space_labels, fontsize=8)
    ax.set_ylabel("Davies-Bouldin")
    ax.set_title("Davies-Bouldin (menor = mejor)")
    ax.legend(fontsize=8)

    fig.suptitle("Efecto de la purificación", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_figure(fig, "purification_comparison")
    return fig
