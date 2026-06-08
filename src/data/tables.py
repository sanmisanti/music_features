"""
Módulo de generación de tablas LaTeX para el EDA.

Genera 5 tablas en formato booktabs, guardadas en results/tables/ y thesis/tables/.
"""

import logging
from pathlib import Path
from typing import Dict, List

from src.config import TABLES_DIR, THESIS_TABLES_DIR
from src.data.eda import FeatureProfile

logger = logging.getLogger(__name__)


def _save_table(content: str, name: str):
    """Guarda una tabla LaTeX en results/tables/ y thesis/tables/."""
    for target_dir in [TABLES_DIR, THESIS_TABLES_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{name}.tex"
        filepath.write_text(content, encoding="utf-8")
        logger.info("Tabla guardada: %s", filepath)


def _fmt(value, decimals=2) -> str:
    """Formatea un número con separador de miles y decimales."""
    if isinstance(value, int):
        return f"{value:,}".replace(",", "\\,")
    if isinstance(value, float):
        return f"{value:,.{decimals}f}".replace(",", "\\,")
    return str(value)


# =============================================================================
# T1: ESTADÍSTICAS DESCRIPTIVAS DE FEATURES
# =============================================================================


def generate_descriptive_stats_table(profiles: Dict[str, FeatureProfile]) -> str:
    """
    Genera tabla LaTeX con estadísticas descriptivas de las 12 features.
    """
    header = r"""\begin{table}[htbp]
\centering
\caption{Estadísticas descriptivas de las features musicales del dataset fuente.}
\label{tab:feature_descriptive_stats}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
Feature & N & Media & DE & Mín & Q1 & Mediana & Q3 & Máx & Outliers (\%) \\
\midrule
"""
    rows = []
    for name, p in profiles.items():
        display_name = name.replace("_", "\\_")
        row = (
            f"{display_name} & "
            f"{_fmt(p.count)} & "
            f"{_fmt(p.mean)} & "
            f"{_fmt(p.std)} & "
            f"{_fmt(p.min)} & "
            f"{_fmt(p.q1)} & "
            f"{_fmt(p.median)} & "
            f"{_fmt(p.q3)} & "
            f"{_fmt(p.max)} & "
            f"{_fmt(p.outliers_pct, 1)} \\\\"
        )
        rows.append(row)

    footer = r"""\bottomrule
\end{tabular}%
}
\end{table}"""

    content = header + "\n".join(rows) + "\n" + footer
    _save_table(content, "feature_descriptive_stats")
    return content


# =============================================================================
# T2: DISTRIBUCIÓN POR GÉNERO Y SUBGÉNERO
# =============================================================================


def generate_genre_distribution_table(genre_data: Dict) -> str:
    """
    Genera tabla LaTeX con distribución por género.
    """
    header = r"""\begin{table}[htbp]
\centering
\caption{Distribución de canciones por género musical.}
\label{tab:genre_distribution}
\begin{tabular}{lrr}
\toprule
Género & N & \% \\
\midrule
"""
    rows = []
    genre_dist = genre_data.get("genre_distribution", {})
    total = 0
    for genre, info in sorted(genre_dist.items(), key=lambda x: -x[1]["count"]):
        display = genre.replace("&", "\\&")
        rows.append(f"{display} & {_fmt(info['count'])} & {_fmt(info['pct'], 1)} \\\\")
        total += info["count"]

    footer_line = f"\\midrule\nTotal & {_fmt(total)} & 100.0 \\\\"
    entropy_note = ""
    if "normalized_entropy" in genre_data:
        entropy_note = (
            f"\n\\multicolumn{{3}}{{l}}"
            f"{{Entropía normalizada: {_fmt(genre_data['normalized_entropy'], 4)} "
            f"(balance {genre_data.get('balance_assessment', 'N/A')})}} \\\\"
        )

    footer = r"""
\bottomrule
\end{tabular}
\end{table}"""

    content = header + "\n".join(rows) + "\n" + footer_line + entropy_note + footer
    _save_table(content, "genre_subgenre_distribution")
    return content


# =============================================================================
# T3: CALIDAD DE LETRAS
# =============================================================================


def generate_lyrics_quality_table(lyrics_stats: Dict) -> str:
    """
    Genera tabla LaTeX con resumen de calidad de letras.
    """
    wc = lyrics_stats.get("word_count", {})
    tc = lyrics_stats.get("token_estimate", {})
    tw = lyrics_stats.get("token_window_coverage", {})
    qf = lyrics_stats.get("quality_filter", {})

    content = r"""\begin{table}[htbp]
\centering
\caption{Resumen de calidad del campo de letras.}
\label{tab:lyrics_quality}
\begin{tabular}{lr}
\toprule
Métrica & Valor \\
\midrule
"""
    metric_rows = [
        ("Total de canciones", _fmt(lyrics_stats.get("total", 0))),
        ("Letras válidas", _fmt(lyrics_stats.get("valid_count", 0))),
        ("Cobertura (\\%)", _fmt(lyrics_stats.get("coverage_pct", 0), 1)),
        ("\\midrule", ""),
        ("Palabras --- media", _fmt(wc.get("mean", 0), 1)),
        ("Palabras --- mediana", _fmt(wc.get("median", 0), 1)),
        ("Palabras --- DE", _fmt(wc.get("std", 0), 1)),
        ("Palabras --- mín", _fmt(wc.get("min", 0))),
        ("Palabras --- máx", _fmt(wc.get("max", 0))),
        ("\\midrule", ""),
        ("Tokens estimados --- media", _fmt(tc.get("mean", 0), 1)),
        ("Tokens estimados --- mediana", _fmt(tc.get("median", 0), 1)),
        (f"Cobertura ventana {tw.get('window_size', 512)} tokens (\\%)", _fmt(tw.get("coverage_pct", 0), 1)),
        ("\\midrule", ""),
        (f"Demasiado cortas ($<${qf.get('min_words', 10)} palabras)", _fmt(qf.get("too_short", 0))),
        (f"Demasiado largas ($>${qf.get('max_words', 2000)} palabras)", _fmt(qf.get("too_long", 0))),
        ("En rango (\\%)", _fmt(qf.get("in_range_pct", 0), 1)),
        ("\\midrule", ""),
        ("Type-token ratio promedio", _fmt(lyrics_stats.get("avg_type_token_ratio", 0), 4)),
    ]

    for label, value in metric_rows:
        if label == "\\midrule":
            content += "\\midrule\n"
        else:
            content += f"{label} & {value} \\\\\n"

    content += r"""\bottomrule
\end{tabular}
\end{table}"""

    _save_table(content, "lyrics_quality_summary")
    return content


# =============================================================================
# T4: CORRELACIONES ALTAS
# =============================================================================


def generate_correlation_table(high_corrs: List[Dict]) -> str:
    """
    Genera tabla LaTeX con pares de features altamente correlacionadas.
    """
    header = r"""\begin{table}[htbp]
\centering
\caption{Pares de features con correlación $|r| \geq 0.5$.}
\label{tab:high_correlations}
\begin{tabular}{llr}
\toprule
Feature 1 & Feature 2 & $r$ \\
\midrule
"""
    rows = []
    if not high_corrs:
        rows.append(
            r"\multicolumn{3}{c}{No se encontraron correlaciones $|r| \geq 0.5$} \\"
        )
    else:
        for pair in high_corrs:
            f1 = pair["feature_1"].replace("_", "\\_")
            f2 = pair["feature_2"].replace("_", "\\_")
            r_val = pair["correlation"]
            rows.append(f"{f1} & {f2} & {r_val:+.4f} \\\\")

    footer = r"""\bottomrule
\end{tabular}
\end{table}"""

    content = header + "\n".join(rows) + "\n" + footer
    _save_table(content, "high_correlations")
    return content


# =============================================================================
# T5: OVERVIEW DEL DATASET
# =============================================================================


def generate_dataset_overview_table(summary: Dict) -> str:
    """
    Genera tabla LaTeX con overview general del dataset.
    """
    content = r"""\begin{table}[htbp]
\centering
\caption{Resumen general del dataset fuente.}
\label{tab:dataset_overview}
\begin{tabular}{lr}
\toprule
Propiedad & Valor \\
\midrule
"""
    metric_rows = [
        ("Total de registros", _fmt(summary.get("total_rows", 0))),
        ("Total de columnas", _fmt(summary.get("total_columns", 0))),
        ("Features musicales", _fmt(summary.get("n_musical_features", 0))),
        ("Columnas de metadatos", _fmt(summary.get("n_metadata_columns", 0))),
        ("\\midrule", ""),
        ("Valores faltantes totales", _fmt(summary.get("total_missing", 0))),
        ("Valores faltantes (\\%)", _fmt(summary.get("total_missing_pct", 0), 2)),
        ("Track IDs duplicados", _fmt(summary.get("duplicate_track_ids", 0))),
        ("\\midrule", ""),
        ("Cobertura de letras (\\%)", _fmt(summary.get("lyrics_coverage_pct", 0), 1)),
        ("Géneros", _fmt(summary.get("n_genres", 0))),
        ("Subgéneros", _fmt(summary.get("n_subgenres", 0))),
    ]

    for label, value in metric_rows:
        if label == "\\midrule":
            content += "\\midrule\n"
        else:
            content += f"{label} & {value} \\\\\n"

    content += r"""\bottomrule
\end{tabular}
\end{table}"""

    _save_table(content, "dataset_overview")
    return content
