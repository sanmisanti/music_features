"""
Script orquestador de recomendacion (Etapa 5).

Ejecutar con: python -m src.recommendation.run_recommendation

Secuencia:
1.  Fijar seeds de reproducibilidad
2.  Cargar dataset unificado
3.  Calcular sigma (mediana de distancias pairwise musicales)
4.  Pre-computar matrices de similitud (semantica + musical)
5.  Grid search sobre alpha (21 valores, P@K para K=5,10,20)
6.  Seleccionar alpha optimo (max P@10)
7.  Evaluacion completa al alpha optimo
8.  Generar tablas LaTeX
9.  Generar figuras PDF
10. Guardar metricas JSON
11. Log resumen final
"""

import json
import logging
import time
from collections import Counter

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("src.recommendation.run_recommendation")

from src.config import (
    METRICS_DIR,
    NUMPY_SEED,
    RANDOM_SEED,
    RECOMMENDATION_ALPHA_RANGE,
    RECOMMENDATION_K_VALUES,
    RECOMMENDATION_TOP_K,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_PREPROCESSING_N_COMPONENTS,
    UNIFIED_DATASET,
)
from src.recommendation.engine import (
    compute_sigma,
    precompute_similarities,
    recommend_all,
)
from src.recommendation.evaluation import (
    evaluate_recommendations,
    precision_at_k,
)
from src.recommendation.optimizer import grid_search_alpha
from src.recommendation.plots import (
    plot_alpha_vs_precision,
    plot_diversity_coverage,
    plot_precision_per_genre,
    plot_score_distributions,
)
from src.recommendation.tables import (
    generate_fusion_strategy_table,
    generate_grid_search_table,
    generate_optimal_metrics_table,
    generate_precision_per_genre_table,
    generate_v1_comparison_table,
)


def main():
    """Ejecuta el pipeline de recomendacion."""
    start_time = time.time()

    # =========================================================================
    # PASO 1: REPRODUCIBILIDAD
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 1: Fijar seeds de reproducibilidad")
    logger.info("=" * 60)

    np.random.seed(NUMPY_SEED)
    logger.info(
        "Seeds fijados: RANDOM_SEED=%d, NUMPY_SEED=%d",
        RANDOM_SEED, NUMPY_SEED,
    )

    # =========================================================================
    # PASO 2: CARGA DEL DATASET UNIFICADO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 2: Carga del dataset unificado")
    logger.info("=" * 60)

    data = np.load(str(UNIFIED_DATASET), allow_pickle=True)
    semantic_embeddings = data["semantic_embeddings"]
    musical_features = data["musical_features"]
    genre_labels = data["genre_labels"]
    track_ids = data["track_ids"]

    n_songs = len(semantic_embeddings)
    logger.info("Dataset: %d canciones, %dD semantico, %dD musical",
                n_songs, semantic_embeddings.shape[1], musical_features.shape[1])

    # =========================================================================
    # PASO 3: CALCULAR SIGMA
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 3: Calcular sigma (kernel gaussiano)")
    logger.info("=" * 60)

    t0 = time.time()
    sigma = compute_sigma(musical_features)
    sigma_time = time.time() - t0
    logger.info("Sigma = %.4f (calculado en %.1f s)", sigma, sigma_time)

    # =========================================================================
    # PASO 4: PRE-COMPUTAR MATRICES DE SIMILITUD
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 4: Pre-computar matrices de similitud")
    logger.info("=" * 60)

    t0 = time.time()
    sim_semantic, sim_musical = precompute_similarities(
        semantic_embeddings, musical_features, sigma,
    )
    sim_time = time.time() - t0
    logger.info("Matrices pre-computadas en %.1f s", sim_time)

    # =========================================================================
    # PASO 5: GRID SEARCH ALPHA
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 5: Grid search sobre alpha")
    logger.info("=" * 60)

    opt_result = grid_search_alpha(
        sim_semantic, sim_musical,
        track_ids, genre_labels,
        alpha_range=RECOMMENDATION_ALPHA_RANGE,
        k_values=RECOMMENDATION_K_VALUES,
        primary_k=RECOMMENDATION_TOP_K,
        sigma=sigma,
    )

    # =========================================================================
    # PASO 6: SELECCIONAR ALPHA OPTIMO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 6: Alpha optimo seleccionado")
    logger.info("=" * 60)

    logger.info(
        "Mejor alpha = %.2f, P@%d = %.4f",
        opt_result.best_alpha, opt_result.primary_k, opt_result.best_precision,
    )

    # =========================================================================
    # PASO 7: EVALUACION COMPLETA AL ALPHA OPTIMO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 7: Evaluacion completa al alpha optimo")
    logger.info("=" * 60)

    # Generar recomendaciones al alpha optimo
    max_k = max(RECOMMENDATION_K_VALUES)
    best_recs = recommend_all(
        sim_semantic, sim_musical, opt_result.best_alpha, max_k,
        track_ids, genre_labels,
    )

    # Obtener cluster labels (re-ejecutar mejores configs de Etapa 4)
    cluster_labels_sem = None
    cluster_labels_mus = None
    try:
        from src.clustering.algorithms import run_hdbscan
        from src.clustering.reduction import reduce_umap

        logger.info("  Computando cluster labels para metricas descriptivas...")

        # Musical: HDBSCAN min_size=200
        mus_result = run_hdbscan(
            musical_features, min_cluster_size=200,
            space_name="musical_13d", metric="euclidean",
        )
        cluster_labels_mus = mus_result.labels

        # Semantico: UMAP 30D + HDBSCAN min_size=200
        semantic_umap, _ = reduce_umap(
            semantic_embeddings,
            n_components=UMAP_PREPROCESSING_N_COMPONENTS,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=RANDOM_SEED,
        )
        sem_result = run_hdbscan(
            semantic_umap, min_cluster_size=200,
            space_name="semantic_umap", metric="euclidean",
        )
        cluster_labels_sem = sem_result.labels

    except Exception as e:
        logger.warning("No se pudieron obtener cluster labels: %s", e)

    # Evaluacion completa
    optimal_metrics = evaluate_recommendations(
        best_recs, opt_result.best_alpha, RECOMMENDATION_TOP_K,
        semantic_embeddings, musical_features, genre_labels,
        cluster_labels_sem, cluster_labels_mus,
    )

    # Precision para otros K
    p5, _, p5_genre = precision_at_k(best_recs, 5)
    p20, _, p20_genre = precision_at_k(best_recs, 20)
    optimal_metrics.precision_per_genre["_p5"] = round(p5, 6)
    optimal_metrics.precision_per_genre["_p20"] = round(p20, 6)

    # P@10 al alpha equivalente a v1 (0.45 semantico)
    v1_equiv_recs = recommend_all(
        sim_semantic, sim_musical, 0.45, RECOMMENDATION_TOP_K,
        track_ids, genre_labels,
    )
    v2_at_v1_p10, _, _ = precision_at_k(v1_equiv_recs, RECOMMENDATION_TOP_K)

    logger.info("P@10 al alpha optimo: %.4f", optimal_metrics.precision_at_k)
    logger.info("P@10 al alpha=0.45 (equiv. v1): %.4f", v2_at_v1_p10)

    # Recopilar scores para histograma
    scores_relevant = []
    scores_non_relevant = []
    for r in best_recs:
        for j, score in enumerate(r.scores[:RECOMMENDATION_TOP_K]):
            if r.recommended_genres[j] == r.query_genre:
                scores_relevant.append(score)
            else:
                scores_non_relevant.append(score)

    scores_relevant = np.array(scores_relevant)
    scores_non_relevant = np.array(scores_non_relevant)

    # =========================================================================
    # PASO 8: TABLAS LaTeX
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 8: Generacion de tablas LaTeX")
    logger.info("=" * 60)

    genre_counts = dict(Counter(genre_labels))

    generate_fusion_strategy_table()
    generate_grid_search_table(opt_result)
    generate_optimal_metrics_table(optimal_metrics)
    generate_precision_per_genre_table(optimal_metrics, genre_counts)
    generate_v1_comparison_table(optimal_metrics, v2_at_v1_p10)

    # =========================================================================
    # PASO 9: FIGURAS PDF
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 9: Generacion de figuras PDF")
    logger.info("=" * 60)

    plot_alpha_vs_precision(opt_result)
    plot_precision_per_genre(optimal_metrics)
    plot_score_distributions(scores_relevant, scores_non_relevant)

    # Diversidad y cobertura para un subconjunto de alphas
    logger.info("  Calculando metricas de diversidad por alpha...")
    diversity_alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    metrics_per_alpha = {}
    for a in diversity_alphas:
        recs_a = recommend_all(
            sim_semantic, sim_musical, a, RECOMMENDATION_TOP_K,
            track_ids, genre_labels,
        )
        m = evaluate_recommendations(
            recs_a, a, RECOMMENDATION_TOP_K,
            semantic_embeddings, musical_features, genre_labels,
        )
        metrics_per_alpha[a] = m

    plot_diversity_coverage(opt_result, metrics_per_alpha)

    # =========================================================================
    # PASO 10: METRICAS JSON
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 10: Guardado de metricas JSON")
    logger.info("=" * 60)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_json = {
        "sigma": round(sigma, 6),
        "sigma_method": "median_pairwise_euclidean",
        "sigma_sample_size": 5000,
        "grid_search": {
            "alpha_range": opt_result.alpha_range,
            "k_values": opt_result.k_values,
            "results": {
                str(a): {str(k): v for k, v in precs.items()}
                for a, precs in opt_result.results.items()
            },
            "best_alpha": opt_result.best_alpha,
            "best_p_at_10": opt_result.best_precision,
            "elapsed_seconds": opt_result.elapsed_seconds,
        },
        "optimal_evaluation": {
            "alpha": optimal_metrics.alpha,
            "precision": {
                "p_at_5": optimal_metrics.precision_per_genre.get("_p5", 0),
                "p_at_10": optimal_metrics.precision_at_k,
                "p_at_20": optimal_metrics.precision_per_genre.get("_p20", 0),
            },
            "precision_std": optimal_metrics.precision_at_k_std,
            "precision_per_genre": {
                k: v for k, v in optimal_metrics.precision_per_genre.items()
                if not k.startswith("_")
            },
            "diversity": {
                "ild_semantic": optimal_metrics.ild_semantic,
                "ild_semantic_std": optimal_metrics.ild_semantic_std,
                "ild_musical": optimal_metrics.ild_musical,
                "ild_musical_std": optimal_metrics.ild_musical_std,
            },
            "coverage": {
                "catalog": optimal_metrics.catalog_coverage,
                "per_genre": optimal_metrics.genre_coverage,
            },
            "cluster_distribution": optimal_metrics.cluster_distribution,
        },
        "benchmarks": {
            "v1_p_at_10": 0.398,
            "v2_at_v1_weights_p_at_10": round(v2_at_v1_p10, 6),
            "v2_optimal_p_at_10": optimal_metrics.precision_at_k,
        },
        "timing": {
            "sigma_seconds": round(sigma_time, 1),
            "similarity_matrices_seconds": round(sim_time, 1),
            "grid_search_seconds": opt_result.elapsed_seconds,
            "total_seconds": round(time.time() - start_time, 1),
        },
    }

    metrics_path = METRICS_DIR / "etapa5_recommendation.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    logger.info("Metricas guardadas: %s", metrics_path)

    # =========================================================================
    # PASO 11: RESUMEN FINAL
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("RECOMENDACION COMPLETADA en %.1f segundos", elapsed)
    logger.info("=" * 60)

    logger.info("  Sigma: %.4f", sigma)
    logger.info("  Alpha optimo: %.2f", opt_result.best_alpha)
    logger.info("  P@5:  %.4f", optimal_metrics.precision_per_genre.get("_p5", 0))
    logger.info("  P@10: %.4f", optimal_metrics.precision_at_k)
    logger.info("  P@20: %.4f", optimal_metrics.precision_per_genre.get("_p20", 0))
    logger.info("  ILD semantica: %.4f", optimal_metrics.ild_semantic)
    logger.info("  ILD musical: %.4f", optimal_metrics.ild_musical)
    logger.info("  Cobertura catalogo: %.4f", optimal_metrics.catalog_coverage)
    logger.info("  P@10 v1 equiv (alpha=0.45): %.4f", v2_at_v1_p10)
    logger.info("  P@10 v1 referencia: 0.398")
    logger.info("Artefactos generados:")
    logger.info("  - 5 tablas LaTeX: results/tables/ + thesis/tables/")
    logger.info("  - 4 figuras PDF: results/figures/ + thesis/figures/")
    logger.info("  - Metricas JSON: results/metrics/etapa5_recommendation.json")


if __name__ == "__main__":
    main()
