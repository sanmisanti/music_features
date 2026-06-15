"""
Script orquestador del clustering multi-algoritmo (Etapa 4, Pasos 2-3).

Ejecutar con: python -m src.clustering.run_clustering

Secuencia:
1.  Fijar seeds de reproducibilidad
2.  Cargar dataset unificado
3.  Preparar espacios de clustering (semantico 384D, UMAP 30D, musical 13D)
4.  Ejecutar multi-algoritmo sobre espacio musical
5.  Ejecutar multi-algoritmo sobre espacio semantico 384D
6.  Ejecutar multi-algoritmo sobre espacio semantico UMAP
7.  Evaluar todas las configuraciones
8.  Seleccionar mejor configuracion por espacio
9.  Calcular NMI cross-modal (H3)
10. Evaluar hipotesis H2 y H3
11. Generar tablas LaTeX y figuras PDF
12. Guardar metricas JSON (etapa4_clustering.json)
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
logger = logging.getLogger("src.clustering.run_clustering")

from src.config import (
    CLUSTERING_K_RANGE,
    HDBSCAN_MIN_CLUSTER_SIZES,
    MAX_NOISE_PCT,
    METRICS_DIR,
    NUMPY_SEED,
    RANDOM_SEED,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_PREPROCESSING_N_COMPONENTS,
    UNIFIED_DATASET,
)
from src.clustering.algorithms import ClusteringResult, run_all_algorithms
from src.clustering.evaluation import (
    ClusteringMetrics,
    compute_cross_modal_nmi,
    evaluate_clustering,
    select_best_configuration,
)
from src.clustering.plots import plot_cluster_sizes, plot_silhouette_comparison
from src.clustering.reduction import reduce_umap
from src.clustering.tables import (
    generate_best_configurations_table,
    generate_clustering_comparison_table,
    generate_cross_modal_nmi_table,
)


def _evaluate_all_results(
    results: list,
    data: np.ndarray,
    genre_labels: np.ndarray,
    metric: str = "euclidean",
) -> list:
    """Evalua todas las configuraciones de clustering de un espacio."""
    metrics_list = []
    for r in results:
        m = evaluate_clustering(
            data=data,
            labels=r.labels,
            genre_labels=genre_labels,
            algorithm=r.algorithm,
            space_name=r.space_name,
            k=r.k,
            n_noise=r.n_noise,
            has_degenerate_clusters=r.has_degenerate_clusters,
            cluster_sizes=r.cluster_sizes,
            params=r.params,
            elapsed_seconds=r.elapsed_seconds,
            metric=metric,
        )
        metrics_list.append(m)
    return metrics_list


def main():
    """Ejecuta el pipeline completo de clustering multi-algoritmo."""
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

    logger.info(
        "Dataset: %d muestras, semantico=%s, musical=%s, generos=%d",
        len(semantic_embeddings), semantic_embeddings.shape,
        musical_features.shape, len(np.unique(genre_labels)),
    )

    # =========================================================================
    # PASO 3: PREPARAR ESPACIOS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 3: Preparar espacios de clustering")
    logger.info("=" * 60)

    # UMAP del espacio semantico
    semantic_umap, umap_report = reduce_umap(
        semantic_embeddings,
        n_components=UMAP_PREPROCESSING_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    spaces = {
        "musical_13d": (musical_features, "euclidean"),
        "semantic_384d": (semantic_embeddings, "euclidean"),
        "semantic_umap": (semantic_umap, "euclidean"),
    }

    logger.info("3 espacios preparados:")
    for name, (d, m) in spaces.items():
        logger.info("  %s: shape=%s, metric=%s", name, d.shape, m)

    # =========================================================================
    # PASOS 4-6: MULTI-ALGORITMO SOBRE CADA ESPACIO
    # =========================================================================

    all_results = {}  # space_name -> list of ClusteringResult
    all_metrics = {}  # space_name -> list of ClusteringMetrics

    for step_num, (space_name, (space_data, space_metric)) in enumerate(
        spaces.items(), start=4
    ):
        logger.info("=" * 60)
        logger.info(
            "PASO %d: Multi-algoritmo sobre %s", step_num, space_name,
        )
        logger.info("=" * 60)

        # Ejecutar algoritmos
        results = run_all_algorithms(
            data=space_data,
            space_name=space_name,
            k_range=CLUSTERING_K_RANGE,
            hdbscan_min_cluster_sizes=HDBSCAN_MIN_CLUSTER_SIZES,
            hdbscan_metric=space_metric,
            random_state=RANDOM_SEED,
        )
        all_results[space_name] = results

        # Evaluar
        logger.info("  Evaluando %d configuraciones...", len(results))
        metrics = _evaluate_all_results(
            results, space_data, genre_labels, metric=space_metric,
        )
        all_metrics[space_name] = metrics

    # =========================================================================
    # PASO 7: SELECCIONAR MEJOR CONFIGURACION POR ESPACIO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 7: Seleccion de mejor configuracion por espacio")
    logger.info("=" * 60)

    best_per_space = {}
    best_results_per_space = {}  # para acceder a labels

    for space_name in spaces:
        logger.info("  --- %s ---", space_name)
        best = select_best_configuration(
            all_metrics[space_name], max_noise_pct=MAX_NOISE_PCT,
        )
        best_per_space[space_name] = best

        if best is not None:
            # Encontrar el ClusteringResult correspondiente
            for r in all_results[space_name]:
                if (r.algorithm == best.algorithm and r.k == best.k):
                    best_results_per_space[space_name] = r
                    break

    # =========================================================================
    # PASO 8: NMI CROSS-MODAL (H3)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 8: NMI cross-modal")
    logger.info("=" * 60)

    # NMI cross-modal: comparar clusterings con estructura REAL. El clustering
    # semantico en 384D es inutilizable (Silhouette ~0.04; HDBSCAN colapsa a
    # 100% ruido), por lo que una particion arbitraria daria un NMI ~0 trivial.
    # Se elige, entre los espacios semanticos disponibles, el de mayor
    # Silhouette macro (UMAP), de modo que el NMI compare dos agrupamientos
    # genuinos (semantico UMAP vs musical) y la complementariedad medida no sea
    # un artefacto de una particion sin estructura.
    sem_candidates = []
    for name in ("semantic_umap", "semantic_384d"):
        cfg = best_per_space.get(name)
        res = best_results_per_space.get(name)
        if cfg is not None and res is not None:
            sem_candidates.append((name, cfg, res))

    best_sem = None
    best_sem_result = None
    sem_space_used = None
    if sem_candidates:
        sem_space_used, best_sem, best_sem_result = max(
            sem_candidates, key=lambda t: t[1].silhouette_macro,
        )
        logger.info(
            "  Espacio semantico para NMI: %s (%s k=%d, Sil_macro=%.4f)",
            sem_space_used, best_sem.algorithm, best_sem.k,
            best_sem.silhouette_macro,
        )

    best_mus = best_per_space.get("musical_13d")
    best_mus_result = best_results_per_space.get("musical_13d")

    nmi_cross = 0.0
    if best_sem_result is not None and best_mus_result is not None:
        nmi_cross = compute_cross_modal_nmi(
            best_sem_result.labels, best_mus_result.labels,
        )
    else:
        logger.warning("No se puede calcular NMI cross-modal: faltan configs")

    # =========================================================================
    # PASO 9: EVALUACION DE HIPOTESIS H2 Y H3
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 9: Evaluacion de hipotesis H2 y H3")
    logger.info("=" * 60)

    # H2: Silhouette musical > Silhouette semantico
    sil_musical = best_per_space["musical_13d"].silhouette_macro if best_per_space.get("musical_13d") else 0.0
    sil_semantic = best_per_space["semantic_384d"].silhouette_macro if best_per_space.get("semantic_384d") else 0.0

    h2_confirmed = sil_musical > sil_semantic
    h2_verdict = "CONFIRMED" if h2_confirmed else "REJECTED"

    logger.info("H2: 'Silhouette musical > Silhouette semantico'")
    logger.info(
        "  Musical: %.4f vs Semantico: %.4f -> %s",
        sil_musical, sil_semantic, h2_verdict,
    )

    # H3: NMI cross-modal < 0.15
    h3_confirmed = nmi_cross < 0.15
    h3_verdict = "CONFIRMED" if h3_confirmed else "REJECTED"

    logger.info("H3: 'NMI cross-modal < 0.15'")
    logger.info("  NMI=%.4f -> %s", nmi_cross, h3_verdict)

    # =========================================================================
    # PASO 10: TABLAS Y FIGURAS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 10: Generacion de tablas LaTeX y figuras PDF")
    logger.info("=" * 60)

    # Tablas comparativas por espacio
    for space_name in spaces:
        generate_clustering_comparison_table(
            all_metrics[space_name], space_name,
        )
        plot_silhouette_comparison(all_metrics[space_name], space_name)

    # Tabla de mejores configuraciones
    generate_best_configurations_table(best_per_space)

    # Tabla NMI cross-modal
    generate_cross_modal_nmi_table(nmi_cross, best_sem, best_mus)

    # Figuras de distribucion de clusters para las mejores configs
    for space_name, best in best_per_space.items():
        if best is not None:
            plot_cluster_sizes(
                best.cluster_sizes, best.algorithm, space_name, best.k,
            )

    # =========================================================================
    # PASO 11: METRICAS JSON
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 11: Guardado de metricas JSON")
    logger.info("=" * 60)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    def _metrics_to_dict(m: ClusteringMetrics) -> dict:
        return {
            "algorithm": m.algorithm,
            "space_name": m.space_name,
            "k": m.k,
            "n_clusters_found": m.n_clusters_found,
            "n_noise": m.n_noise,
            "has_degenerate_clusters": m.has_degenerate_clusters,
            "cluster_sizes": {str(k): v for k, v in m.cluster_sizes.items()},
            "silhouette_micro": m.silhouette_micro,
            "silhouette_macro": m.silhouette_macro,
            "davies_bouldin": m.davies_bouldin,
            "calinski_harabasz": m.calinski_harabasz,
            "ari_genre": m.ari_genre,
            "nmi_genre": m.nmi_genre,
            "params": m.params,
            "elapsed_seconds": m.elapsed_seconds,
        }

    metrics = {
        "configurations": {},
        "best_per_space": {},
        "cross_modal_nmi": {
            "value": round(nmi_cross, 6),
            "semantic_space_used": sem_space_used,
            "semantic_config": (
                f"{best_sem.algorithm}_k{best_sem.k}"
                if best_sem else None
            ),
            "musical_config": (
                f"{best_mus.algorithm}_k{best_mus.k}"
                if best_mus else None
            ),
        },
        "hypothesis_H2": {
            "statement": "Silhouette musical > Silhouette semantico",
            "silhouette_musical": round(sil_musical, 6),
            "silhouette_semantic": round(sil_semantic, 6),
            "verdict": h2_verdict,
        },
        "hypothesis_H3": {
            "statement": "NMI cross-modal < 0.15",
            "nmi_value": round(nmi_cross, 6),
            "verdict": h3_verdict,
        },
        "umap_params": {
            "n_components": umap_report.output_dim,
            "n_neighbors": umap_report.n_neighbors,
            "min_dist": umap_report.min_dist,
            "elapsed_seconds": umap_report.elapsed_seconds,
        },
        "benchmarks_v1": {
            "nmi_cross_modal": 0.0567,
            "silhouette_post_purification": 0.2893,
        },
    }

    for space_name in spaces:
        metrics["configurations"][space_name] = [
            _metrics_to_dict(m) for m in all_metrics[space_name]
        ]
        best = best_per_space[space_name]
        if best is not None:
            metrics["best_per_space"][space_name] = {
                "algorithm": best.algorithm,
                "k": best.k,
                "silhouette_macro": best.silhouette_macro,
                "davies_bouldin": best.davies_bouldin,
                "nmi_genre": best.nmi_genre,
            }

    metrics["total_elapsed_seconds"] = round(time.time() - start_time, 1)

    metrics_path = METRICS_DIR / "etapa4_clustering.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metricas guardadas: %s", metrics_path)

    # =========================================================================
    # PASO 12: RESUMEN FINAL
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("CLUSTERING COMPLETADO en %.1f segundos", elapsed)
    logger.info("=" * 60)

    logger.info("Mejor configuracion por espacio:")
    for space_name, best in best_per_space.items():
        if best is not None:
            logger.info(
                "  %-20s: %s k=%d, Sil_macro=%.4f, DB=%.4f, NMI_genre=%.4f",
                space_name, best.algorithm, best.k,
                best.silhouette_macro, best.davies_bouldin, best.nmi_genre,
            )
        else:
            logger.info("  %-20s: ninguna configuracion valida", space_name)

    logger.info("NMI cross-modal: %.4f", nmi_cross)
    logger.info("Veredicto H2: %s", h2_verdict)
    logger.info("Veredicto H3: %s", h3_verdict)
    logger.info("Artefactos generados:")
    logger.info("  - Tablas LaTeX: results/tables/ + thesis/tables/")
    logger.info("  - Figuras PDF: results/figures/ + thesis/figures/")
    logger.info("  - Metricas JSON: results/metrics/etapa4_clustering.json")


if __name__ == "__main__":
    main()
