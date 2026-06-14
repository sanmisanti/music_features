"""
Figuras ilustrativas para el Marco Teorico (Cap. 4).

Genera dos figuras con datos sinteticos:
  - clustering_dendrogram.pdf  : dendrograma del metodo de Ward
  - clustering_shapes.pdf      : comparacion K-Means vs HDBSCAN en distintas geometrias

Uso:
    python -m src.reporting.plot_marco_teorico
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
import hdbscan

from src.config import FIGURES_DIR, THESIS_FIGURES_DIR, RANDOM_SEED

FIGURE_FORMAT = "pdf"
FIGURE_DPI = 150


def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def _save(fig: plt.Figure, name: str):
    for d in [FIGURES_DIR, THESIS_FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.{FIGURE_FORMAT}", format=FIGURE_FORMAT, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardada: {name}.{FIGURE_FORMAT}")


# Paleta de colores discreta consistente con el resto del proyecto
CLUSTER_COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"]
NOISE_COLOR = "#CCCCCC"


def plot_dendrogram():
    """
    Dendrograma de Ward sobre datos sinteticos con 3 clusters bien definidos.
    Muestra el concepto de corte horizontal para obtener K clusters.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    centers = [[-2.5, 0], [2.5, 0.5], [0, 3]]
    X = np.vstack([
        rng.multivariate_normal(c, [[0.25, 0], [0, 0.25]], size=12)
        for c in centers
    ])

    Z = linkage(X, method="ward")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel izquierdo: scatter de los puntos coloreados por cluster real
    ax_scatter = axes[0]
    colors_real = (
        [CLUSTER_COLORS[0]] * 12
        + [CLUSTER_COLORS[1]] * 12
        + [CLUSTER_COLORS[2]] * 12
    )
    ax_scatter.scatter(X[:, 0], X[:, 1], c=colors_real, s=50, edgecolors="white", linewidths=0.5)
    ax_scatter.set_title("Datos originales")
    ax_scatter.set_xlabel("Feature 1")
    ax_scatter.set_ylabel("Feature 2")
    ax_scatter.set_aspect("equal")

    # Panel derecho: dendrograma con linea de corte
    ax_dend = axes[1]
    cut_height = 3.8
    dendrogram(
        Z,
        ax=ax_dend,
        color_threshold=cut_height,
        above_threshold_color="#888888",
        link_color_func=lambda k: "#888888",
        no_labels=True,
    )
    ax_dend.axhline(y=cut_height, color="#D65F5F", linestyle="--", linewidth=1.4, label=f"Corte → 3 clusters")
    ax_dend.set_title("Dendrograma (método de Ward)")
    ax_dend.set_xlabel("Observaciones")
    ax_dend.set_ylabel("Distancia de fusión")
    ax_dend.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.suptitle("Clustering jerárquico: dendrograma y corte", y=1.01)
    fig.tight_layout()
    _save(fig, "clustering_dendrogram")


def plot_shapes_comparison():
    """
    Comparacion K-Means vs HDBSCAN en tres geometrias de datos sinteticos:
      - Blobs esfericos (ambos funcionan)
      - Lunas (K-Means falla, HDBSCAN acierta)
      - Blobs con ruido/outliers (K-Means asigna todo, HDBSCAN identifica ruido)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Dataset 1: blobs esfericos
    X_blobs, _ = make_blobs(n_samples=220, centers=3, cluster_std=0.55, random_state=RANDOM_SEED)

    # Dataset 2: lunas entrelazadas
    X_moons, _ = make_moons(n_samples=240, noise=0.07, random_state=RANDOM_SEED)

    # Dataset 3: blobs con outliers dispersos
    X_core, _ = make_blobs(n_samples=180, centers=2, cluster_std=0.5, random_state=RANDOM_SEED)
    X_noise = rng.uniform(low=-5, high=5, size=(30, 2))
    X_noise_data = np.vstack([X_core, X_noise])

    datasets = [
        (X_blobs,      "Blobs esféricos",         2, 3),
        (X_moons,      "Formas no convexas",       2, 2),
        (X_noise_data, "Datos con outliers",       2, 2),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    row_labels = ["K-Means", "HDBSCAN"]

    for col, (X, title, min_samples_h, k) in enumerate(datasets):
        # --- K-Means ---
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels_km = km.fit_predict(X)

        ax = axes[0, col]
        for c in np.unique(labels_km):
            mask = labels_km == c
            ax.scatter(X[mask, 0], X[mask, 1], c=CLUSTER_COLORS[c], s=18,
                       edgecolors="none", alpha=0.85)
        centroids = km.cluster_centers_
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=120,
                   c="black", zorder=5, label="Centroides")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("K-Means", fontsize=11, fontweight="bold")
        if col == 0:
            ax.legend(loc="lower right", fontsize=8, frameon=True)

        # --- HDBSCAN ---
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=min_samples_h)
        labels_h = clusterer.fit_predict(X)

        ax = axes[1, col]
        unique = np.unique(labels_h)
        color_idx = 0
        for c in unique:
            mask = labels_h == c
            if c == -1:
                ax.scatter(X[mask, 0], X[mask, 1], c=NOISE_COLOR, s=18,
                           edgecolors="none", alpha=0.6, label="Ruido")
            else:
                ax.scatter(X[mask, 0], X[mask, 1], c=CLUSTER_COLORS[color_idx % len(CLUSTER_COLORS)],
                           s=18, edgecolors="none", alpha=0.85)
                color_idx += 1
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("HDBSCAN", fontsize=11, fontweight="bold")
        noise_count = np.sum(labels_h == -1)
        if noise_count > 0:
            ax.legend(loc="lower right", fontsize=8, frameon=True)

    fig.suptitle("K-Means vs HDBSCAN: efecto de la geometría de los datos", y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, "clustering_shapes")


if __name__ == "__main__":
    _setup_style()
    print("Generando figuras del Marco Teórico...")
    plot_dendrogram()
    plot_shapes_comparison()
    print("Listo.")
