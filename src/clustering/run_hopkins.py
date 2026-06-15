"""
Script orquestador del analisis de Hopkins (Etapa 4, Paso 1).

Ejecutar con: python -m src.clustering.run_hopkins

Secuencia:
1. Fijar seeds de reproducibilidad
2. Cargar dataset unificado (data/5_unified/unified_dataset.npz)
3. Preparar 4 espacios (semantic_384d, semantic_umap, musical_13d, concatenated)
4. Calcular Hopkins sobre cada espacio (30 iteraciones x 4 espacios)
5. Evaluar hipotesis H1 (ambos espacios > 0.7)
6. Generar tabla LaTeX (hopkins_results.tex)
7. Generar figura PDF (hopkins_results.pdf)
8. Guardar metricas JSON (etapa4_hopkins.json)
9. Log resumen final con veredicto H1
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
logger = logging.getLogger("src.clustering.run_hopkins")

from src.config import (
    HOPKINS_ITERATIONS,
    HOPKINS_SAMPLE_SIZE,
    HOPKINS_THRESHOLD,
    METRICS_DIR,
    NUMPY_SEED,
    RANDOM_SEED,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_PREPROCESSING_N_COMPONENTS,
    UNIFIED_DATASET,
)
from src.clustering.hopkins import HopkinsReport, compute_hopkins_bootstrap
from src.clustering.plots import plot_hopkins_barplot
from src.clustering.reduction import prepare_spaces_for_hopkins
from src.clustering.tables import generate_hopkins_table


def main():
    """Ejecuta el pipeline completo de analisis de Hopkins."""
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

    logger.info(
        "Dataset cargado: %d muestras", semantic_embeddings.shape[0],
    )
    logger.info(
        "  Semantico: shape=%s, dtype=%s",
        semantic_embeddings.shape, semantic_embeddings.dtype,
    )
    logger.info(
        "  Musical: shape=%s, dtype=%s",
        musical_features.shape, musical_features.dtype,
    )

    # Verificacion de integridad basica
    l2_norms = np.linalg.norm(semantic_embeddings, axis=1)
    logger.info(
        "  Norma L2 semantica: media=%.6f, std=%.6f (esperado ~1.0)",
        l2_norms.mean(), l2_norms.std(),
    )

    # =========================================================================
    # PASO 3: PREPARACION DE ESPACIOS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 3: Preparacion de 4 espacios para Hopkins")
    logger.info("=" * 60)

    spaces, umap_report = prepare_spaces_for_hopkins(
        semantic_embeddings,
        musical_features,
        umap_n_components=UMAP_PREPROCESSING_N_COMPONENTS,
        umap_n_neighbors=UMAP_N_NEIGHBORS,
        umap_min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_SEED,
    )

    # =========================================================================
    # PASO 4: CALCULO DE HOPKINS SOBRE CADA ESPACIO
    # =========================================================================
    logger.info("=" * 60)
    logger.info(
        "PASO 4: Hopkins bootstrap (%d iteraciones x %d espacios)",
        HOPKINS_ITERATIONS, len(spaces),
    )
    logger.info("=" * 60)

    # Orden fijo para reproducibilidad y presentacion
    space_order = ["semantic_384d", "semantic_umap", "musical_13d", "concatenated"]
    reports = []

    for space_name in space_order:
        space_data, metric, reference = spaces[space_name]
        report = compute_hopkins_bootstrap(
            data=space_data,
            space_name=space_name,
            sample_size=HOPKINS_SAMPLE_SIZE,
            n_iterations=HOPKINS_ITERATIONS,
            metric=metric,
            reference=reference,
            random_state=RANDOM_SEED,
        )
        reports.append(report)

    # =========================================================================
    # PASO 5: EVALUACION DE HIPOTESIS H1
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 5: Evaluacion de hipotesis H1")
    logger.info("=" * 60)

    # H1: Ambos espacios (semantico y musical) exhiben Hopkins > 0.7
    h1_semantic = next(r for r in reports if r.space_name == "semantic_384d")
    h1_musical = next(r for r in reports if r.space_name == "musical_13d")

    h1_sem_pass = h1_semantic.exceeds_threshold
    h1_mus_pass = h1_musical.exceeds_threshold

    if h1_sem_pass and h1_mus_pass:
        h1_verdict = "CONFIRMED"
    elif h1_sem_pass or h1_mus_pass:
        h1_verdict = "PARTIALLY_CONFIRMED"
    else:
        h1_verdict = "REJECTED"

    logger.info(
        "H1: 'Ambos espacios exhiben Hopkins > %.1f'", HOPKINS_THRESHOLD,
    )
    logger.info(
        "  Semantico 384D: H=%.4f, > %.1f: %s",
        h1_semantic.mean, HOPKINS_THRESHOLD, h1_sem_pass,
    )
    logger.info(
        "  Musical 13D: H=%.4f, > %.1f: %s",
        h1_musical.mean, HOPKINS_THRESHOLD, h1_mus_pass,
    )
    logger.info("  Veredicto H1: %s", h1_verdict)

    # Comparacion semantico 384D vs UMAP (diagnostico curse of dim)
    h_umap = next(r for r in reports if r.space_name == "semantic_umap")
    diff_384_umap = h1_semantic.mean - h_umap.mean
    logger.info(
        "  Diagnostico curse-of-dim: H(384D)=%.4f vs H(UMAP %dD)=%.4f, "
        "diff=%.4f",
        h1_semantic.mean, umap_report.output_dim, h_umap.mean,
        diff_384_umap,
    )
    if diff_384_umap > 0.05:
        logger.warning(
            "  ATENCION: Hopkins 384D es %.4f mayor que UMAP, "
            "posible inflacion por curse of dimensionality",
            diff_384_umap,
        )

    # =========================================================================
    # PASO 6: TABLA LaTeX
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 6: Generacion de tabla LaTeX")
    logger.info("=" * 60)

    generate_hopkins_table(reports)

    # =========================================================================
    # PASO 7: FIGURA PDF
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 7: Generacion de figura PDF")
    logger.info("=" * 60)

    plot_hopkins_barplot(reports)

    # =========================================================================
    # PASO 8: METRICAS JSON
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 8: Guardado de metricas JSON")
    logger.info("=" * 60)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "spaces": {},
        "umap_params": {
            "n_components": umap_report.output_dim,
            "n_neighbors": umap_report.n_neighbors,
            "min_dist": umap_report.min_dist,
            "metric": umap_report.metric,
            "elapsed_seconds": umap_report.elapsed_seconds,
        },
        "hypothesis_H1": {
            "statement": (
                f"Ambos espacios (semantico y musical) exhiben "
                f"Hopkins > {HOPKINS_THRESHOLD}"
            ),
            "semantic_384d_exceeds": h1_sem_pass,
            "musical_13d_exceeds": h1_mus_pass,
            "verdict": h1_verdict,
        },
        "curse_of_dimensionality_diagnostic": {
            "hopkins_384d": round(h1_semantic.mean, 6),
            "hopkins_umap": round(h_umap.mean, 6),
            "difference": round(diff_384_umap, 6),
            "possible_inflation": diff_384_umap > 0.05,
        },
        "benchmarks_v1": {
            "hopkins_semantic_384d": 0.7752,
            "hopkins_musical_12d": 0.7871,
        },
    }

    for r in reports:
        metrics["spaces"][r.space_name] = {
            "n_samples": r.n_samples,
            "n_dimensions": r.n_dimensions,
            "metric": r.metric,
            "reference_distribution": r.reference_distribution,
            "sample_size": r.sample_size,
            "n_iterations": r.n_iterations,
            "hopkins_mean": round(r.mean, 6),
            "hopkins_std": round(r.std, 6),
            "ci_lower": round(r.ci_lower, 6),
            "ci_upper": round(r.ci_upper, 6),
            "min_value": round(r.min_value, 6),
            "max_value": round(r.max_value, 6),
            "exceeds_threshold": r.exceeds_threshold,
        }

    metrics["total_elapsed_seconds"] = round(time.time() - start_time, 1)

    metrics_path = METRICS_DIR / "etapa4_hopkins.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metricas guardadas: %s", metrics_path)

    # =========================================================================
    # PASO 9: RESUMEN FINAL
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("HOPKINS COMPLETADO en %.1f segundos", elapsed)
    logger.info("=" * 60)

    logger.info("Resultados por espacio:")
    for r in reports:
        logger.info(
            "  %-20s: H=%.4f +/- %.4f, IC95%%=[%.4f, %.4f], > %.1f: %s",
            r.space_name, r.mean, r.std,
            r.ci_lower, r.ci_upper,
            HOPKINS_THRESHOLD, r.exceeds_threshold,
        )

    logger.info("Veredicto H1: %s", h1_verdict)
    logger.info("Artefactos generados:")
    logger.info("  - Tabla LaTeX: results/tables/hopkins_results.tex + thesis/tables/")
    logger.info("  - Figura PDF: results/figures/hopkins_results.pdf + thesis/figures/")
    logger.info("  - Metricas JSON: results/metrics/etapa4_hopkins.json")


if __name__ == "__main__":
    main()
