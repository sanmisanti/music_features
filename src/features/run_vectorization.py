"""
Script orquestador de la vectorizacion semantica (Etapa 3, Paso 2).

Ejecutar con: python -m src.features.run_vectorization

Secuencia:
1. Fijar seeds de reproducibilidad (numpy + torch)
2. Cargar dataset seleccionado desde data/3_selected/selected_dataset.csv
3. Preparar letras (normalizacion minima de artefactos)
4. Medir cobertura real de tokens con tokenizer E5
5. Vectorizar letras (17,964 x 384 embeddings)
6. Verificaciones post-vectorizacion
7. Guardar artefactos (3 NPY + tabla LaTeX + metricas JSON)
8. Log resumen final
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
logger = logging.getLogger("src.features.run_vectorization")

from src.config import (
    BERT_BATCH_SIZE,
    BERT_EMBEDDING_DIM,
    BERT_MODEL_NAME,
    BERT_TARGET_TOKENS,
    METRICS_DIR,
    NUMPY_SEED,
    RANDOM_SEED,
    SELECTED_DATASET,
    SOURCE_SEPARATOR,
    VECTORIZED_DIR,
    VECTORIZED_EMBEDDINGS,
    VECTORIZED_TOKEN_COUNTS,
    VECTORIZED_TRACK_IDS,
)
from src.features.tables import generate_token_coverage_table
from src.features.vectorizer import (
    measure_token_coverage,
    prepare_lyrics_for_encoding,
    vectorize_lyrics,
)


def main():
    """Ejecuta el pipeline completo de vectorizacion semantica."""
    start_time = time.time()

    # =========================================================================
    # PASO 1: REPRODUCIBILIDAD
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 1: Fijar seeds de reproducibilidad")
    logger.info("=" * 60)

    np.random.seed(NUMPY_SEED)
    try:
        import torch
        torch.manual_seed(NUMPY_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(NUMPY_SEED)
        logger.info(
            "Seeds fijados: RANDOM_SEED=%d, NUMPY_SEED=%d, torch=%d",
            RANDOM_SEED, NUMPY_SEED, NUMPY_SEED,
        )
        logger.info(
            "Dispositivo: %s",
            "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        )
    except ImportError:
        logger.info(
            "Seeds fijados: RANDOM_SEED=%d, NUMPY_SEED=%d (torch no disponible)",
            RANDOM_SEED, NUMPY_SEED,
        )

    # =========================================================================
    # PASO 2: CARGA DEL DATASET SELECCIONADO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 2: Carga del dataset seleccionado")
    logger.info("=" * 60)

    import pandas as pd

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

    # Extraer columnas necesarias
    track_ids = df["track_id"].values
    lyrics = df["lyrics"]

    assert lyrics.isnull().sum() == 0, "Se encontraron letras nulas en el dataset seleccionado"
    logger.info("Verificacion: 0 letras nulas en dataset seleccionado")

    # =========================================================================
    # PASO 3: PREPARACION DE LETRAS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 3: Preparacion de letras (normalizacion minima)")
    logger.info("=" * 60)

    lyrics_prepared = prepare_lyrics_for_encoding(lyrics)

    # =========================================================================
    # PASO 4: COBERTURA REAL DE TOKENS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 4: Medicion de cobertura real de tokens")
    logger.info("=" * 60)

    token_report = measure_token_coverage(lyrics_prepared)

    # =========================================================================
    # PASO 5: VECTORIZACION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 5: Vectorizacion de letras con %s", BERT_MODEL_NAME)
    logger.info("=" * 60)

    embeddings, vec_report = vectorize_lyrics(lyrics_prepared)

    # =========================================================================
    # PASO 6: VERIFICACIONES POST-VECTORIZACION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 6: Verificaciones post-vectorizacion")
    logger.info("=" * 60)

    # 6a. Shape y dtype
    expected_shape = (len(df), BERT_EMBEDDING_DIM)
    assert embeddings.shape == expected_shape, (
        f"Shape incorrecto: {embeddings.shape} vs esperado {expected_shape}"
    )
    assert embeddings.dtype == np.float32, (
        f"Dtype incorrecto: {embeddings.dtype} vs esperado float32"
    )
    logger.info(
        "Verificacion: shape=%s, dtype=%s",
        embeddings.shape, embeddings.dtype,
    )

    # 6b. NaN
    n_nan = int(np.isnan(embeddings).sum())
    assert n_nan == 0, f"Se encontraron {n_nan} valores NaN en embeddings"
    logger.info("Verificacion: 0 valores NaN")

    # 6c. Vectores cero
    norms = np.linalg.norm(embeddings, axis=1)
    n_zero = int((norms < 1e-8).sum())
    logger.info(
        "Verificacion: %d vectores cero (%.2f%%)",
        n_zero, 100.0 * n_zero / len(embeddings),
    )

    # 6d. Norma L2 media
    nonzero_norms = norms[norms >= 1e-8]
    if len(nonzero_norms) > 0:
        mean_norm = float(np.mean(nonzero_norms))
        logger.info(
            "Verificacion: norma L2 media=%.4f (esperado ~1.0 por normalize_embeddings=True)",
            mean_norm,
        )
        assert abs(mean_norm - 1.0) < 0.01, (
            f"Norma L2 media={mean_norm:.4f}, esperado ~1.0"
        )

    # 6e. Track IDs coinciden
    assert len(track_ids) == embeddings.shape[0], (
        f"Longitud track_ids ({len(track_ids)}) != filas embeddings ({embeddings.shape[0]})"
    )
    logger.info("Verificacion: track_ids alineados (%d)", len(track_ids))

    logger.info("Todas las verificaciones post-vectorizacion pasaron correctamente")

    # =========================================================================
    # PASO 7: GUARDADO DE ARTEFACTOS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 7: Guardado de artefactos")
    logger.info("=" * 60)

    # 7a. Archivos NPY
    VECTORIZED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(VECTORIZED_EMBEDDINGS, embeddings)
    logger.info(
        "Embeddings guardados: %s (shape=%s, %.2f MB)",
        VECTORIZED_EMBEDDINGS, embeddings.shape,
        VECTORIZED_EMBEDDINGS.stat().st_size / (1024 * 1024),
    )

    np.save(VECTORIZED_TRACK_IDS, track_ids)
    logger.info("Track IDs guardados: %s (%d)", VECTORIZED_TRACK_IDS, len(track_ids))

    np.save(VECTORIZED_TOKEN_COUNTS, token_report.token_counts)
    logger.info(
        "Token counts guardados: %s (%d)",
        VECTORIZED_TOKEN_COUNTS, len(token_report.token_counts),
    )

    # 7b. Tabla LaTeX de cobertura de tokens
    generate_token_coverage_table(token_report)

    # 7c. Metricas JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": {
            "name": BERT_MODEL_NAME,
            "embedding_dim": BERT_EMBEDDING_DIM,
            "max_seq_length": BERT_TARGET_TOKENS,
            "batch_size": BERT_BATCH_SIZE,
        },
        "token_coverage": {
            "total_songs": token_report.total_songs,
            "window_size": token_report.window_size,
            "within_window": token_report.within_window,
            "coverage_pct": token_report.coverage_pct,
            "mean_tokens": token_report.mean_tokens,
            "median_tokens": token_report.median_tokens,
            "std_tokens": token_report.std_tokens,
            "min_tokens": token_report.min_tokens,
            "max_tokens": token_report.max_tokens,
            "p90_tokens": token_report.p90_tokens,
            "p95_tokens": token_report.p95_tokens,
            "p99_tokens": token_report.p99_tokens,
            "heuristic_coverage_pct": 58.8,
        },
        "vectorization": {
            "total_songs": vec_report.total_songs,
            "successful": vec_report.successful,
            "failed": vec_report.failed,
            "failure_indices": vec_report.failure_indices,
            "embedding_shape": list(vec_report.embedding_shape),
            "dtype": vec_report.dtype,
            "elapsed_seconds": vec_report.elapsed_seconds,
            "throughput_songs_per_sec": vec_report.throughput_songs_per_sec,
            "model_load_seconds": vec_report.model_load_seconds,
        },
        "embedding_stats": vec_report.embedding_stats,
        "postconditions": {
            "nan_count": n_nan,
            "zero_vectors": n_zero,
            "l2_norm_mean": round(float(np.mean(norms)), 6),
            "track_ids_aligned": True,
        },
        "output_paths": {
            "embeddings": str(VECTORIZED_EMBEDDINGS),
            "track_ids": str(VECTORIZED_TRACK_IDS),
            "token_counts": str(VECTORIZED_TOKEN_COUNTS),
        },
    }

    metrics_path = METRICS_DIR / "etapa3_vectorization.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metricas guardadas: %s", metrics_path)

    # =========================================================================
    # PASO 8: RESUMEN FINAL
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("VECTORIZACION COMPLETADA en %.1f segundos", elapsed)
    logger.info("=" * 60)
    logger.info("Resultados:")
    logger.info("  Modelo: %s (%dD)", BERT_MODEL_NAME, BERT_EMBEDDING_DIM)
    logger.info(
        "  Cobertura de tokens: %.1f%% (heuristica EDA: 58.8%%)",
        token_report.coverage_pct,
    )
    logger.info(
        "  Embeddings: %d canciones x %d dimensiones (%.2f MB)",
        embeddings.shape[0], embeddings.shape[1],
        VECTORIZED_EMBEDDINGS.stat().st_size / (1024 * 1024),
    )
    logger.info(
        "  Vectorizacion: %d exitosas, %d fallidas (%.1f songs/s)",
        vec_report.successful, vec_report.failed,
        vec_report.throughput_songs_per_sec,
    )
    logger.info(
        "  Norma L2: media=%.4f, std=%.4f",
        vec_report.embedding_stats["l2_norm_mean"],
        vec_report.embedding_stats["l2_norm_std"],
    )
    logger.info("Artefactos generados:")
    logger.info("  - Embeddings: %s", VECTORIZED_EMBEDDINGS)
    logger.info("  - Track IDs: %s", VECTORIZED_TRACK_IDS)
    logger.info("  - Token counts: %s", VECTORIZED_TOKEN_COUNTS)
    logger.info("  - Tabla LaTeX: results/tables/token_coverage.tex + thesis/tables/")
    logger.info("  - Metricas JSON: results/metrics/etapa3_vectorization.json")


if __name__ == "__main__":
    main()
