"""
Script orquestador del EDA (Etapa 2).

Ejecutar con: python -m src.data.run_eda

Secuencia:
1. Carga y validación del dataset fuente
2. Estadísticas descriptivas de features musicales
3. Análisis de calidad de letras
4. Matriz de correlaciones
5. Análisis de sesgo (género, idioma, instrumentalidad)
6. Generación de 7 figuras PDF
7. Generación de 5 tablas LaTeX
8. Guardado de métricas consolidadas en JSON
"""

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Configurar logging antes de cualquier import del proyecto
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("src.data.run_eda")

from src.config import (
    METRICS_DIR,
    MUSICAL_FEATURES,
    NUMPY_SEED,
    RANDOM_SEED,
)
from src.data.loader import load_source_dataset
from src.data.eda import (
    analyze_genre_balance,
    analyze_instrumentalness_distribution,
    analyze_language_distribution,
    analyze_lyrics_quality,
    compute_correlation_matrix,
    compute_dataset_summary,
    compute_feature_profiles,
    identify_high_correlations,
)
from src.data.plots import (
    plot_correlation_heatmap,
    plot_feature_boxplots,
    plot_feature_distributions,
    plot_features_by_genre,
    plot_genre_distribution,
    plot_language_distribution,
    plot_lyrics_length_distribution,
)
from src.data.tables import (
    generate_correlation_table,
    generate_dataset_overview_table,
    generate_descriptive_stats_table,
    generate_genre_distribution_table,
    generate_lyrics_quality_table,
)


def main():
    """Ejecuta el pipeline completo de EDA."""
    start_time = time.time()

    # Reproducibilidad
    np.random.seed(NUMPY_SEED)
    logger.info("Seeds fijados: RANDOM_SEED=%d, NUMPY_SEED=%d", RANDOM_SEED, NUMPY_SEED)

    # =========================================================================
    # 1. CARGA Y VALIDACIÓN
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 1: Carga y validación del dataset fuente")
    logger.info("=" * 60)

    df, load_report = load_source_dataset(validate=True)
    logger.info(
        "Reporte de carga: %d filas, %d columnas, %.2f MB, %d advertencias",
        load_report.total_rows,
        load_report.total_columns,
        load_report.memory_mb,
        len(load_report.warnings),
    )
    for w in load_report.warnings:
        logger.warning("  -> %s", w)

    # =========================================================================
    # 2. ESTADÍSTICAS DESCRIPTIVAS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 2: Estadísticas descriptivas de features musicales")
    logger.info("=" * 60)

    profiles = compute_feature_profiles(df, MUSICAL_FEATURES)
    for name, p in profiles.items():
        logger.info(
            "  %s: media=%.4f, DE=%.4f, outliers=%.1f%%",
            name, p.mean, p.std, p.outliers_pct,
        )

    # =========================================================================
    # 3. ANÁLISIS DE CALIDAD DE LETRAS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 3: Análisis de calidad de letras")
    logger.info("=" * 60)

    lyrics_stats = analyze_lyrics_quality(df)

    # =========================================================================
    # 4. MATRIZ DE CORRELACIONES
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 4: Matriz de correlaciones")
    logger.info("=" * 60)

    corr_matrix = compute_correlation_matrix(df, MUSICAL_FEATURES)
    high_corrs = identify_high_correlations(corr_matrix, threshold=0.5)
    for pair in high_corrs:
        logger.info(
            "  Correlación alta: %s <-> %s (r=%.4f)",
            pair["feature_1"], pair["feature_2"], pair["correlation"],
        )

    # =========================================================================
    # 5. ANÁLISIS DE SESGO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 5: Análisis de sesgo")
    logger.info("=" * 60)

    genre_balance = analyze_genre_balance(df)
    language_dist = analyze_language_distribution(df)
    instrumentalness_dist = analyze_instrumentalness_distribution(df)

    # =========================================================================
    # 6. GENERACIÓN DE FIGURAS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 6: Generación de 7 figuras PDF")
    logger.info("=" * 60)

    plot_feature_distributions(df)
    plot_feature_boxplots(df)
    plot_correlation_heatmap(corr_matrix)
    plot_genre_distribution(df)
    plot_lyrics_length_distribution(df)
    plot_language_distribution(df)
    plot_features_by_genre(df)

    logger.info("7 figuras generadas correctamente")

    # =========================================================================
    # 7. GENERACIÓN DE TABLAS LaTeX
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 7: Generación de 5 tablas LaTeX")
    logger.info("=" * 60)

    summary = compute_dataset_summary(df)

    generate_descriptive_stats_table(profiles)
    generate_genre_distribution_table(genre_balance)
    generate_lyrics_quality_table(lyrics_stats)
    generate_correlation_table(high_corrs)
    generate_dataset_overview_table(summary)

    logger.info("5 tablas LaTeX generadas correctamente")

    # =========================================================================
    # 8. GUARDADO DE MÉTRICAS CONSOLIDADAS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 8: Guardado de métricas en JSON")
    logger.info("=" * 60)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "dataset_summary": summary,
        "load_report": {
            "total_rows": load_report.total_rows,
            "total_columns": load_report.total_columns,
            "memory_mb": load_report.memory_mb,
            "missing_by_column": load_report.missing_by_column,
            "duplicate_track_ids": load_report.duplicate_track_ids,
            "warnings": load_report.warnings,
        },
        "feature_profiles": {
            name: asdict(p) for name, p in profiles.items()
        },
        "lyrics_quality": lyrics_stats,
        "correlation_analysis": {
            "method": "pearson",
            "high_correlations": high_corrs,
        },
        "bias_analysis": {
            "genre_balance": genre_balance,
            "language_distribution": language_dist,
            "instrumentalness": instrumentalness_dist,
        },
    }

    metrics["total_elapsed_seconds"] = round(time.time() - start_time, 1)

    metrics_path = METRICS_DIR / "etapa2_eda.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métricas guardadas: %s", metrics_path)

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("EDA COMPLETADO en %.1f segundos", elapsed)
    logger.info("=" * 60)
    logger.info("Artefactos generados:")
    logger.info("  - 7 figuras PDF en results/figures/ y thesis/figures/")
    logger.info("  - 5 tablas LaTeX en results/tables/ y thesis/tables/")
    logger.info("  - Métricas JSON en results/metrics/etapa2_eda.json")


if __name__ == "__main__":
    main()
