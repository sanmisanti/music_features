# src/clustering/ - Clustering multimodal: Hopkins, multi-algoritmo, purificacion

from src.clustering.hopkins import (
    HopkinsReport,
    compute_hopkins_bootstrap,
    compute_hopkins_statistic,
)
from src.clustering.reduction import (
    UMAPReport,
    prepare_spaces_for_hopkins,
    reduce_for_visualization,
    reduce_umap,
)
from src.clustering.algorithms import (
    ClusteringResult,
    run_all_algorithms,
    run_hdbscan,
    run_kmeans,
    run_ward,
)
from src.clustering.evaluation import (
    ClusteringMetrics,
    compute_cross_modal_nmi,
    evaluate_clustering,
    select_best_configuration,
)
from src.clustering.purification import (
    PurificationReport,
    purify_centroid_distance,
    purify_combined,
    purify_negative_silhouette,
    run_purification,
)
from src.clustering.tables import (
    generate_best_configurations_table,
    generate_clustering_comparison_table,
    generate_cross_modal_nmi_table,
    generate_hopkins_table,
    generate_purification_table,
)
from src.clustering.plots import (
    plot_cluster_sizes,
    plot_hopkins_barplot,
    plot_purification_comparison,
    plot_silhouette_comparison,
    plot_umap_clusters,
    plot_umap_genres,
)

__all__ = [
    "HopkinsReport",
    "compute_hopkins_bootstrap",
    "compute_hopkins_statistic",
    "UMAPReport",
    "prepare_spaces_for_hopkins",
    "reduce_for_visualization",
    "reduce_umap",
    "ClusteringResult",
    "run_all_algorithms",
    "run_hdbscan",
    "run_kmeans",
    "run_ward",
    "ClusteringMetrics",
    "compute_cross_modal_nmi",
    "evaluate_clustering",
    "select_best_configuration",
    "PurificationReport",
    "purify_centroid_distance",
    "purify_combined",
    "purify_negative_silhouette",
    "run_purification",
    "generate_best_configurations_table",
    "generate_clustering_comparison_table",
    "generate_cross_modal_nmi_table",
    "generate_hopkins_table",
    "generate_purification_table",
    "plot_cluster_sizes",
    "plot_hopkins_barplot",
    "plot_purification_comparison",
    "plot_silhouette_comparison",
    "plot_umap_clusters",
    "plot_umap_genres",
]
