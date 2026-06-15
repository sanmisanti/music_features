"""
Modulo de recomendacion hibrida (Etapa 5).

Sistema k-NN directo con fusion tardia de similitudes semanticas
(coseno) y musicales (euclidea con kernel gaussiano).
"""

from src.recommendation.engine import (
    RecommendationResult,
    compute_sigma,
    gaussian_kernel,
    precompute_similarities,
    recommend_all,
)
from src.recommendation.evaluation import (
    RecommendationMetrics,
    catalog_coverage,
    cluster_distribution_in_recommendations,
    evaluate_recommendations,
    genre_coverage,
    intra_list_diversity,
    precision_at_k,
)
from src.recommendation.optimizer import (
    OptimizationResult,
    grid_search_alpha,
)
from src.recommendation.tables import (
    generate_fusion_strategy_table,
    generate_genre_coverage_table,
    generate_grid_search_table,
    generate_optimal_metrics_table,
    generate_precision_per_genre_table,
    generate_v1_comparison_table,
)
from src.recommendation.plots import (
    plot_alpha_vs_precision,
    plot_diversity_coverage,
    plot_precision_per_genre,
    plot_score_distributions,
)
