"""
Script orquestador de la normalizacion de features musicales (Etapa 3, Paso 3).

Ejecutar con: python -m src.features.run_normalization

Secuencia:
1. Fijar seeds de reproducibilidad
2. Cargar dataset seleccionado desde data/3_selected/selected_dataset.csv
3. Extraer features musicales (12 columnas CSV -> 13 features con codificacion circular de key)
4. Normalizar (z-score)
5. Verificaciones post-normalizacion
6. Guardar artefactos (2 NPY + tabla LaTeX + metricas JSON)
7. Log resumen final
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
logger = logging.getLogger("src.features.run_normalization")

import pandas as pd

from src.config import (
    METRICS_DIR,
    NUMPY_SEED,
    RANDOM_SEED,
    SELECTED_DATASET,
    SOURCE_SEPARATOR,
    VECTORIZED_DIR,
    VECTORIZED_FEATURE_NAMES,
    VECTORIZED_MUSICAL_FEATURES,
    VECTORIZED_TRACK_IDS,
)
from src.features.normalizer import extract_musical_features, normalize_features
from src.features.tables import generate_normalization_table


def main():
    """Ejecuta el pipeline completo de normalizacion de features musicales."""
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
    # PASO 2: CARGA DEL DATASET SELECCIONADO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 2: Carga del dataset seleccionado")
    logger.info("=" * 60)

    logger.info("Cargando dataset desde %s", SELECTED_DATASET)
    df = pd.read_csv(
        SELECTED_DATASET,
        sep=SOURCE_SEPARATOR,
        encoding="utf-8",
        engine="python",
    )
    logger.info(
        "Dataset cargado: %d filas, %d columnas",
        len(df), len(df.columns),
    )

    # =========================================================================
    # PASO 3: EXTRACCION DE FEATURES MUSICALES
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 3: Extraccion de features musicales")
    logger.info("=" * 60)

    features_raw, feature_names = extract_musical_features(df)

    # =========================================================================
    # PASO 4: NORMALIZACION Z-SCORE
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 4: Normalizacion z-score")
    logger.info("=" * 60)

    features_normalized, norm_report = normalize_features(features_raw, feature_names)

    # =========================================================================
    # PASO 5: VERIFICACIONES POST-NORMALIZACION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 5: Verificaciones post-normalizacion")
    logger.info("=" * 60)

    # 5a. Shape y dtype
    expected_shape = (len(df), len(feature_names))
    assert features_normalized.shape == expected_shape, (
        f"Shape incorrecto: {features_normalized.shape} vs esperado {expected_shape}"
    )
    assert features_normalized.dtype == np.float32, (
        f"Dtype incorrecto: {features_normalized.dtype} vs esperado float32"
    )
    logger.info(
        "Verificacion: shape=%s, dtype=%s",
        features_normalized.shape, features_normalized.dtype,
    )

    # 5b. NaN
    n_nan = int(np.isnan(features_normalized).sum())
    assert n_nan == 0, f"Se encontraron {n_nan} valores NaN post-normalizacion"
    logger.info("Verificacion: 0 valores NaN")

    # 5c. Media ~0 por feature (tolerancia 1e-6)
    post_means = features_normalized.mean(axis=0)
    max_mean_deviation = float(np.abs(post_means).max())
    assert max_mean_deviation < 1e-6, (
        f"Media maxima post-normalizacion: {max_mean_deviation:.2e} (tolerancia 1e-6)"
    )
    logger.info(
        "Verificacion: media maxima por feature = %.2e (tolerancia 1e-6)",
        max_mean_deviation,
    )

    # 5d. DE ~1 por feature (tolerancia 1e-6)
    post_stds = features_normalized.std(axis=0)
    max_std_deviation = float(np.abs(post_stds - 1.0).max())
    assert max_std_deviation < 1e-6, (
        f"Desviacion maxima de std respecto a 1.0: {max_std_deviation:.2e} (tolerancia 1e-6)"
    )
    logger.info(
        "Verificacion: desviacion maxima de DE respecto a 1.0 = %.2e (tolerancia 1e-6)",
        max_std_deviation,
    )

    # 5e. Alineacion con track_ids de embeddings
    track_ids_path = VECTORIZED_TRACK_IDS
    if track_ids_path.exists():
        embedding_track_ids = np.load(track_ids_path, allow_pickle=True)
        dataset_track_ids = df["track_id"].values

        assert len(dataset_track_ids) == len(embedding_track_ids), (
            f"Cantidad de track_ids no coincide: dataset={len(dataset_track_ids)} "
            f"vs embeddings={len(embedding_track_ids)}"
        )
        assert np.array_equal(dataset_track_ids, embedding_track_ids), (
            "Track IDs del dataset y embeddings no coinciden en orden"
        )
        logger.info(
            "Verificacion: %d track_ids alineados con embeddings existentes",
            len(embedding_track_ids),
        )
    else:
        logger.warning(
            "Archivo de track_ids no encontrado en %s — "
            "la verificacion de alineacion se omitio",
            track_ids_path,
        )

    logger.info("Todas las verificaciones post-normalizacion pasaron correctamente")

    # =========================================================================
    # PASO 6: GUARDADO DE ARTEFACTOS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 6: Guardado de artefactos")
    logger.info("=" * 60)

    # 6a. Archivos NPY
    VECTORIZED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(VECTORIZED_MUSICAL_FEATURES, features_normalized)
    logger.info(
        "Features musicales guardadas: %s (shape=%s, %.2f MB)",
        VECTORIZED_MUSICAL_FEATURES, features_normalized.shape,
        VECTORIZED_MUSICAL_FEATURES.stat().st_size / (1024 * 1024),
    )

    np.save(VECTORIZED_FEATURE_NAMES, np.array(feature_names))
    logger.info(
        "Nombres de features guardados: %s (%d features)",
        VECTORIZED_FEATURE_NAMES, len(feature_names),
    )

    # 6b. Tabla LaTeX de normalizacion
    generate_normalization_table(norm_report)

    # 6c. Metricas JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "normalization": {
            "method": norm_report.method,
            "n_samples": norm_report.n_samples,
            "n_features": norm_report.n_features,
            "feature_names": norm_report.feature_names,
            "nan_before": norm_report.n_nan_before,
            "nan_after": norm_report.n_nan_after,
        },
        "raw_statistics": {
            name: {
                "mean": round(float(norm_report.raw_means[i]), 6),
                "std": round(float(norm_report.raw_stds[i]), 6),
                "min": round(float(norm_report.raw_mins[i]), 6),
                "max": round(float(norm_report.raw_maxs[i]), 6),
            }
            for i, name in enumerate(norm_report.feature_names)
        },
        "post_statistics": {
            name: {
                "mean": round(float(norm_report.post_means[i]), 10),
                "std": round(float(norm_report.post_stds[i]), 10),
            }
            for i, name in enumerate(norm_report.feature_names)
        },
        "postconditions": {
            "shape": list(features_normalized.shape),
            "dtype": str(features_normalized.dtype),
            "nan_count": n_nan,
            "max_mean_deviation": round(max_mean_deviation, 12),
            "max_std_deviation": round(max_std_deviation, 12),
            "track_ids_aligned": track_ids_path.exists(),
        },
        "output_paths": {
            "musical_features": str(VECTORIZED_MUSICAL_FEATURES),
            "feature_names": str(VECTORIZED_FEATURE_NAMES),
        },
    }

    metrics_path = METRICS_DIR / "etapa3_normalization.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metricas guardadas: %s", metrics_path)

    # =========================================================================
    # PASO 7: RESUMEN FINAL
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("NORMALIZACION COMPLETADA en %.1f segundos", elapsed)
    logger.info("=" * 60)
    logger.info("Resultados:")
    logger.info("  Metodo: %s", norm_report.method)
    logger.info(
        "  Features: %d muestras x %d features",
        norm_report.n_samples, norm_report.n_features,
    )
    logger.info(
        "  NaN: %d antes, %d despues",
        norm_report.n_nan_before, norm_report.n_nan_after,
    )
    logger.info(
        "  Post-normalizacion: media_max=%.2e, std_max_dev=%.2e",
        max_mean_deviation, max_std_deviation,
    )
    logger.info("Artefactos generados:")
    logger.info("  - Features: %s", VECTORIZED_MUSICAL_FEATURES)
    logger.info("  - Nombres: %s", VECTORIZED_FEATURE_NAMES)
    logger.info("  - Tabla LaTeX: results/tables/normalization_stats.tex + thesis/tables/")
    logger.info("  - Metricas JSON: results/metrics/etapa3_normalization.json")


if __name__ == "__main__":
    main()
