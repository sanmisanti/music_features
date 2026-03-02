"""
Script orquestador de la unificacion de features (Etapa 3, Paso 4).

Ejecutar con: python -m src.features.run_unification

Secuencia:
1. Fijar seeds de reproducibilidad
2. Cargar componentes de features (5 NPY de data/4_vectorized/)
3. Cargar metadatos del CSV (data/3_selected/selected_dataset.csv)
4. Verificaciones de integridad cruzada + guardado NPZ
5. Guardar artefactos complementarios (tabla LaTeX + metricas JSON)
6. Log resumen final
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
logger = logging.getLogger("src.features.run_unification")

from src.config import (
    METRICS_DIR,
    NUMPY_SEED,
    RANDOM_SEED,
    UNIFIED_DATASET,
)
from src.features.unifier import (
    build_unified_dataset,
    load_feature_components,
    load_metadata,
)
from src.features.tables import generate_unification_table


def main():
    """Ejecuta el pipeline completo de unificacion de features."""
    start_time = time.time()

    # =========================================================================
    # PASO 1: REPRODUCIBILIDAD
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 1: Fijar seeds de reproducibilidad")
    logger.info("=" * 60)

    np.random.seed(NUMPY_SEED)
    logger.info("Seeds fijados: RANDOM_SEED=%d, NUMPY_SEED=%d", RANDOM_SEED, NUMPY_SEED)

    # =========================================================================
    # PASO 2: CARGA DE COMPONENTES DE FEATURES
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 2: Carga de componentes de features (5 NPY)")
    logger.info("=" * 60)

    components = load_feature_components()

    # =========================================================================
    # PASO 3: CARGA DE METADATOS DEL CSV
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 3: Carga de metadatos del CSV seleccionado")
    logger.info("=" * 60)

    metadata = load_metadata(components["track_ids"])

    # =========================================================================
    # PASO 4: VERIFICACIONES + GUARDADO NPZ
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 4: Verificaciones de integridad y guardado NPZ")
    logger.info("=" * 60)

    report = build_unified_dataset(components, metadata)

    # =========================================================================
    # PASO 5: ARTEFACTOS COMPLEMENTARIOS (TABLA LATEX + METRICAS JSON)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 5: Guardado de artefactos complementarios")
    logger.info("=" * 60)

    # 5a. Tabla LaTeX
    generate_unification_table(report)

    # 5b. Metricas JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "unification": {
            "n_samples": report.n_samples,
            "semantic_dim": report.semantic_dim,
            "musical_dim": report.musical_dim,
            "n_genres": report.n_genres,
            "genre_distribution": report.genre_distribution,
            "n_arrays": report.n_arrays,
            "npz_size_mb": round(report.npz_size_mb, 2),
        },
        "integrity_checks": {
            "n_nan_semantic": report.n_nan_semantic,
            "n_nan_musical": report.n_nan_musical,
            "semantic_l2_mean": round(report.semantic_l2_mean, 6),
            "semantic_l2_std": round(report.semantic_l2_std, 6),
            "musical_mean_abs_mean": round(report.musical_mean_abs_mean, 10),
            "musical_mean_std": round(report.musical_mean_std, 10),
            "track_ids_aligned": report.track_ids_aligned,
        },
        "output_paths": {
            "unified_dataset": str(UNIFIED_DATASET),
        },
    }

    metrics_path = METRICS_DIR / "etapa3_unification.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metricas guardadas: %s", metrics_path)

    # =========================================================================
    # PASO 6: RESUMEN FINAL
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("UNIFICACION COMPLETADA en %.1f segundos", elapsed)
    logger.info("=" * 60)
    logger.info("Resultados:")
    logger.info(
        "  Dataset: %d muestras, %d dim semanticas + %d dim musicales",
        report.n_samples, report.semantic_dim, report.musical_dim,
    )
    logger.info(
        "  Generos: %d unicos, distribucion: %s",
        report.n_genres, report.genre_distribution,
    )
    logger.info(
        "  Integridad: NaN_sem=%d, NaN_mus=%d, L2_mean=%.4f, aligned=%s",
        report.n_nan_semantic, report.n_nan_musical,
        report.semantic_l2_mean, report.track_ids_aligned,
    )
    logger.info(
        "  NPZ: %.2f MB, %d arrays",
        report.npz_size_mb, report.n_arrays,
    )
    logger.info("Artefactos generados:")
    logger.info("  - Dataset NPZ: %s", UNIFIED_DATASET)
    logger.info("  - Tabla LaTeX: results/tables/unified_summary.tex + thesis/tables/")
    logger.info("  - Metricas JSON: results/metrics/etapa3_unification.json")


if __name__ == "__main__":
    main()
