"""
Script orquestador de purificacion y visualizacion (Etapa 4, Pasos 4-5).

Ejecutar con: python -m src.clustering.run_purification

Secuencia:
1.  Fijar seeds de reproducibilidad
2.  Cargar dataset unificado
3.  Preparar espacios (musical 13D, semantico UMAP 30D)
4.  Re-ejecutar mejores algoritmos para obtener labels
5.  Purificacion combinada sobre cada configuracion
6.  Evaluar hipotesis H4
7.  Generar UMAP 2D para visualizacion de ambos espacios
8.  Generar figuras: UMAP clusters, UMAP generos, purification comparison
9.  Generar tabla purification_results.tex
10. Guardar metricas JSON (etapa4_purification.json)
11. Log resumen final
"""

import json
import logging
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("src.clustering.run_purification")

from src.config import (
    METRICS_DIR,
    NUMPY_SEED,
    PURIFICATION_MAX_REMOVAL_PCT,
    PURIFICATION_SIGMA_THRESHOLD,
    RANDOM_SEED,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_PREPROCESSING_N_COMPONENTS,
    UNIFIED_DATASET,
)
from src.clustering.algorithms import run_hdbscan, run_kmeans
from src.clustering.plots import (
    plot_purification_comparison,
    plot_umap_clusters,
    plot_umap_genres,
)
from src.clustering.purification import run_purification
from src.clustering.reduction import reduce_for_visualization, reduce_umap
from src.clustering.tables import generate_purification_table


def main():
    """Ejecuta el pipeline de purificacion y visualizacion."""
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

    logger.info("Dataset: %d muestras", len(semantic_embeddings))

    # =========================================================================
    # PASO 3: PREPARAR ESPACIOS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 3: Preparar espacios")
    logger.info("=" * 60)

    # UMAP 30D para clustering semantico
    semantic_umap, umap_report = reduce_umap(
        semantic_embeddings,
        n_components=UMAP_PREPROCESSING_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    # =========================================================================
    # PASO 4: RE-EJECUTAR MEJORES ALGORITMOS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 4: Re-ejecutar mejores algoritmos")
    logger.info("=" * 60)

    # Musical: K-Means k=5 (mejor particion; HDBSCAN k=2 con 84% noise
    # no es util para purificacion)
    logger.info("  Musical: K-Means++ k=5")
    musical_result = run_kmeans(
        musical_features, k=5, space_name="musical_13d",
        random_state=RANDOM_SEED,
    )

    # Semantico UMAP: HDBSCAN min_size=200 k=3
    logger.info("  Semantico UMAP: HDBSCAN min_cluster_size=200")
    semantic_result = run_hdbscan(
        semantic_umap, min_cluster_size=200,
        space_name="semantic_umap", metric="euclidean",
    )

    # =========================================================================
    # PASO 5: PURIFICACION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 5: Purificacion combinada")
    logger.info("=" * 60)

    purification_reports = []

    # Musical
    logger.info("--- Musical 13D ---")
    musical_purif = run_purification(
        data=musical_features,
        labels=musical_result.labels,
        algorithm="kmeans",
        space_name="musical_13d",
        k=5,
        metric="euclidean",
        sigma_threshold=PURIFICATION_SIGMA_THRESHOLD,
        max_removal_pct=PURIFICATION_MAX_REMOVAL_PCT,
    )
    purification_reports.append(musical_purif)

    # Semantico UMAP
    logger.info("--- Semantico UMAP 30D ---")
    semantic_purif = run_purification(
        data=semantic_umap,
        labels=semantic_result.labels,
        algorithm="hdbscan",
        space_name="semantic_umap",
        k=3,
        metric="euclidean",
        sigma_threshold=PURIFICATION_SIGMA_THRESHOLD,
        max_removal_pct=PURIFICATION_MAX_REMOVAL_PCT,
    )
    purification_reports.append(semantic_purif)

    # =========================================================================
    # PASO 6: EVALUACION H4
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 6: Evaluacion de hipotesis H4")
    logger.info("=" * 60)

    # H4: Purificacion mejora Silhouette sin eliminar > 15%
    h4_results = []
    for r in purification_reports:
        improved = r.silhouette_improvement > 0
        within_budget = r.within_removal_budget
        h4_results.append({
            "space": r.space_name,
            "improved": improved,
            "within_budget": within_budget,
            "confirmed": improved and within_budget,
        })
        logger.info(
            "  %s: mejora=%.4f (%+.1f%%), remocion=%.1f%%, "
            "dentro de presupuesto=%s -> %s",
            r.space_name, r.silhouette_improvement,
            r.silhouette_improvement_pct, r.removal_pct * 100,
            within_budget,
            "CONFIRMED" if improved and within_budget else "REJECTED",
        )

    all_confirmed = all(h["confirmed"] for h in h4_results)
    any_confirmed = any(h["confirmed"] for h in h4_results)

    if all_confirmed:
        h4_verdict = "CONFIRMED"
    elif any_confirmed:
        h4_verdict = "PARTIALLY_CONFIRMED"
    else:
        h4_verdict = "REJECTED"

    logger.info("Veredicto H4 global: %s", h4_verdict)

    # =========================================================================
    # PASO 7: UMAP 2D PARA VISUALIZACION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 7: UMAP 2D para visualizacion")
    logger.info("=" * 60)

    # Semantico -> UMAP 2D
    logger.info("  UMAP 2D del espacio semantico...")
    semantic_2d, _ = reduce_for_visualization(
        semantic_embeddings, metric="cosine", random_state=RANDOM_SEED,
    )

    # Musical -> UMAP 2D
    logger.info("  UMAP 2D del espacio musical...")
    musical_2d, _ = reduce_for_visualization(
        musical_features, metric="euclidean", random_state=RANDOM_SEED,
    )

    # =========================================================================
    # PASO 8: FIGURAS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 8: Generacion de figuras PDF")
    logger.info("=" * 60)

    # UMAP clusters - semantico
    plot_umap_clusters(
        semantic_2d, semantic_result.labels,
        "semantic_umap", "hdbscan", 3,
    )

    # UMAP clusters - musical
    plot_umap_clusters(
        musical_2d, musical_result.labels,
        "musical_13d", "kmeans", 5,
    )

    # UMAP generos - semantico
    plot_umap_genres(semantic_2d, genre_labels, "semantic_umap")

    # UMAP generos - musical
    plot_umap_genres(musical_2d, genre_labels, "musical_13d")

    # Purification comparison
    plot_purification_comparison(purification_reports)

    # =========================================================================
    # PASO 9: TABLA
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 9: Generacion de tabla LaTeX")
    logger.info("=" * 60)

    generate_purification_table(purification_reports)

    # =========================================================================
    # PASO 10: METRICAS JSON
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 10: Guardado de metricas JSON")
    logger.info("=" * 60)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    def _report_to_dict(r):
        return {
            "algorithm": r.algorithm,
            "space_name": r.space_name,
            "k": r.k,
            "strategy": r.strategy,
            "n_samples_before": r.n_samples_before,
            "n_samples_after": r.n_samples_after,
            "n_removed": r.n_removed,
            "removal_pct": round(r.removal_pct, 6),
            "silhouette_macro_before": r.silhouette_macro_before,
            "silhouette_macro_after": r.silhouette_macro_after,
            "silhouette_micro_before": r.silhouette_micro_before,
            "silhouette_micro_after": r.silhouette_micro_after,
            "davies_bouldin_before": r.davies_bouldin_before,
            "davies_bouldin_after": r.davies_bouldin_after,
            "silhouette_improvement": r.silhouette_improvement,
            "silhouette_improvement_pct": r.silhouette_improvement_pct,
            "db_improvement": r.db_improvement,
            "within_removal_budget": r.within_removal_budget,
            "cluster_sizes_before": {str(k): v for k, v in r.cluster_sizes_before.items()},
            "cluster_sizes_after": {str(k): v for k, v in r.cluster_sizes_after.items()},
        }

    metrics = {
        "purifications": [_report_to_dict(r) for r in purification_reports],
        "hypothesis_H4": {
            "statement": "Purificacion mejora Silhouette sin eliminar >15% datos",
            "per_space": h4_results,
            "verdict": h4_verdict,
        },
        "benchmarks_v1": {
            "silhouette_post_purification": 0.2893,
        },
    }

    metrics["total_elapsed_seconds"] = round(time.time() - start_time, 1)

    metrics_path = METRICS_DIR / "etapa4_purification.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metricas guardadas: %s", metrics_path)

    # =========================================================================
    # PASO 11: RESUMEN FINAL
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PURIFICACION COMPLETADA en %.1f segundos", elapsed)
    logger.info("=" * 60)

    for r in purification_reports:
        logger.info(
            "  %s (%s k=%d): Sil_macro %.4f -> %.4f (%+.4f, %+.1f%%), "
            "removidos %d (%.1f%%)",
            r.space_name, r.algorithm, r.k,
            r.silhouette_macro_before, r.silhouette_macro_after,
            r.silhouette_improvement, r.silhouette_improvement_pct,
            r.n_removed, r.removal_pct * 100,
        )

    logger.info("Veredicto H4: %s", h4_verdict)
    logger.info("Artefactos generados:")
    logger.info("  - Tabla LaTeX: results/tables/purification_results.tex + thesis/tables/")
    logger.info("  - Figuras PDF: results/figures/ + thesis/figures/")
    logger.info("  - Metricas JSON: results/metrics/etapa4_purification.json")


if __name__ == "__main__":
    main()
