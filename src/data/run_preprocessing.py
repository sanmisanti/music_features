"""
Script orquestador del preprocesamiento y seleccion (Etapa 3, Paso 1).

Ejecutar con: python -m src.data.run_preprocessing

Secuencia:
1. Fijar seeds de reproducibilidad
2. Cargar dataset fuente via load_source_dataset()
3. Aplicar filtros de calidad (preprocess_dataset)
4. Guardar dataset seleccionado en data/3_selected/selected_dataset.csv
5. Generar tabla LaTeX del data loss budget
6. Guardar metricas JSON en results/metrics/etapa3_preprocessing.json
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
logger = logging.getLogger("src.data.run_preprocessing")

from src.config import (
    LYRICS_MAX_WORDS,
    LYRICS_MIN_WORDS,
    METRICS_DIR,
    NUMPY_SEED,
    RANDOM_SEED,
    SELECTED_DATASET,
    TABLES_DIR,
    THESIS_TABLES_DIR,
)
from src.data.loader import load_source_dataset
from src.data.preprocessor import preprocess_dataset, save_selected_dataset


def _fmt(value, decimals=2) -> str:
    """Formatea un numero con separador de miles y decimales."""
    if isinstance(value, int):
        return f"{value:,}".replace(",", "\\,")
    if isinstance(value, float):
        return f"{value:,.{decimals}f}".replace(",", "\\,")
    return str(value)


def generate_data_loss_budget_table(budget_entries, n_loudness_clipped, original_count):
    """
    Genera tabla LaTeX del data loss budget con formato booktabs.

    La tabla documenta cada filtro aplicado, la cantidad de registros
    antes y despues, la perdida absoluta y porcentual, y la justificacion.
    """
    header = r"""\begin{table}[htbp]
\centering
\caption{Presupuesto de pérdida de datos (\textit{data loss budget}) del preprocesamiento.}
\label{tab:data_loss_budget}
\small
\begin{tabular}{lrrrrp{6cm}}
\toprule
Filtro & Antes & Después & Pérdida & \% & Justificación \\
\midrule
"""
    rows = []
    for entry in budget_entries:
        name = entry["filter_name"]
        rows.append(
            f"{name} & "
            f"{_fmt(entry['rows_before'])} & "
            f"{_fmt(entry['rows_after'])} & "
            f"{_fmt(entry['rows_lost'])} & "
            f"{_fmt(entry['loss_pct'], 2)} & "
            f"{entry['reason']} \\\\"
        )

    # Fila de total
    if budget_entries:
        final_count = budget_entries[-1]["rows_after"]
        total_lost = original_count - final_count
        total_pct = round(100.0 * total_lost / original_count, 2) if original_count > 0 else 0.0
        rows.append("\\midrule")
        rows.append(
            f"\\textbf{{Total}} & "
            f"{_fmt(original_count)} & "
            f"{_fmt(final_count)} & "
            f"{_fmt(total_lost)} & "
            f"{_fmt(total_pct, 2)} & --- \\\\"
        )

    footer = r"""\bottomrule
\end{tabular}"""

    # Nota al pie sobre loudness
    footnote = ""
    if n_loudness_clipped > 0:
        footnote = (
            f"\n\\vspace{{0.5em}}\n"
            f"\\footnotesize{{Nota: {n_loudness_clipped} valores de "
            f"\\texttt{{loudness}} $> 0$ dB fueron recortados a 0 dB "
            f"(corrección sin eliminación de filas).}}"
        )

    closing = "\n\\end{table}"

    content = header + "\n".join(rows) + "\n" + footer + footnote + closing

    for target_dir in [TABLES_DIR, THESIS_TABLES_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / "data_loss_budget.tex"
        filepath.write_text(content, encoding="utf-8")
        logger.info("Tabla guardada: %s", filepath)

    return content


def main():
    """Ejecuta el pipeline completo de preprocesamiento y seleccion."""
    start_time = time.time()

    # Reproducibilidad
    np.random.seed(NUMPY_SEED)
    logger.info("Seeds fijados: RANDOM_SEED=%d, NUMPY_SEED=%d", RANDOM_SEED, NUMPY_SEED)

    # =========================================================================
    # 1. CARGA DEL DATASET FUENTE
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 1: Carga del dataset fuente")
    logger.info("=" * 60)

    df, load_report = load_source_dataset(validate=False)
    logger.info(
        "Dataset cargado: %d filas, %d columnas, %.2f MB",
        load_report.total_rows,
        load_report.total_columns,
        load_report.memory_mb,
    )

    # =========================================================================
    # 2. PREPROCESAMIENTO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 2: Aplicación de filtros de calidad")
    logger.info("=" * 60)

    original_count = len(df)
    df_clean, budget_entries, n_loudness_clipped = preprocess_dataset(df)

    # =========================================================================
    # 3. VERIFICACIONES POST-FILTRADO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 3: Verificaciones post-filtrado")
    logger.info("=" * 60)

    # Verificar que no quedan letras nulas
    null_lyrics = df_clean["lyrics"].isnull().sum()
    assert null_lyrics == 0, f"Quedan {null_lyrics} letras nulas tras preprocesamiento"
    logger.info("Verificación: 0 letras nulas")

    # Verificar rango de palabras
    word_counts = df_clean["lyrics"].str.split().str.len()
    below_min = (word_counts < LYRICS_MIN_WORDS).sum()
    above_max = (word_counts > LYRICS_MAX_WORDS).sum()
    assert below_min == 0, f"{below_min} letras debajo del mínimo"
    assert above_max == 0, f"{above_max} letras encima del máximo"
    logger.info(
        "Verificación: todas las letras en rango [%d, %d] palabras",
        LYRICS_MIN_WORDS, LYRICS_MAX_WORDS,
    )

    # Verificar loudness
    loudness_above_zero = (df_clean["loudness"] > 0.0).sum()
    assert loudness_above_zero == 0, f"{loudness_above_zero} valores de loudness > 0 dB"
    logger.info("Verificación: 0 valores de loudness > 0 dB")

    # Verificar que las columnas se preservan
    assert len(df_clean.columns) == len(df.columns), "Se perdieron columnas"
    logger.info("Verificación: %d columnas preservadas", len(df_clean.columns))

    # =========================================================================
    # 4. GUARDADO DEL DATASET SELECCIONADO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 4: Guardado del dataset seleccionado")
    logger.info("=" * 60)

    save_selected_dataset(df_clean, SELECTED_DATASET)

    # =========================================================================
    # 5. GENERACIÓN DE TABLA LaTeX
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 5: Generación de tabla LaTeX (data loss budget)")
    logger.info("=" * 60)

    generate_data_loss_budget_table(budget_entries, n_loudness_clipped, original_count)

    # =========================================================================
    # 6. GUARDADO DE MÉTRICAS JSON
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PASO 6: Guardado de métricas en JSON")
    logger.info("=" * 60)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    final_count = len(df_clean)
    total_lost = original_count - final_count

    metrics = {
        "original_count": original_count,
        "final_count": final_count,
        "total_lost": total_lost,
        "total_loss_pct": round(100.0 * total_lost / original_count, 2),
        "loudness_clipped": n_loudness_clipped,
        "filters_applied": budget_entries,
        "postconditions": {
            "null_lyrics": 0,
            "below_min_words": 0,
            "above_max_words": 0,
            "loudness_above_zero": 0,
            "columns_preserved": len(df_clean.columns),
        },
        "parameters": {
            "min_words": LYRICS_MIN_WORDS,
            "max_words": LYRICS_MAX_WORDS,
            "loudness_clip_db": 0.0,
        },
        "output_path": str(SELECTED_DATASET),
    }

    metrics["total_elapsed_seconds"] = round(time.time() - start_time, 1)

    metrics_path = METRICS_DIR / "etapa3_preprocessing.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métricas guardadas: %s", metrics_path)

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PREPROCESAMIENTO COMPLETADO en %.1f segundos", elapsed)
    logger.info("=" * 60)
    logger.info("Resultados:")
    logger.info("  Dataset: %d -> %d registros (%.2f%% pérdida)", original_count, final_count, metrics["total_loss_pct"])
    logger.info("  Loudness corregidos: %d valores", n_loudness_clipped)
    logger.info("Artefactos generados:")
    logger.info("  - Dataset: %s", SELECTED_DATASET)
    logger.info("  - Tabla LaTeX: results/tables/data_loss_budget.tex + thesis/tables/")
    logger.info("  - Métricas JSON: results/metrics/etapa3_preprocessing.json")


if __name__ == "__main__":
    main()
