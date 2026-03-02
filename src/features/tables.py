"""
Modulo de generacion de tablas LaTeX para la Etapa 3 (features).

Genera tablas en formato booktabs, guardadas en results/tables/ y thesis/tables/.
"""

import logging

from src.config import TABLES_DIR, THESIS_TABLES_DIR
from src.features.vectorizer import TokenCoverageReport

logger = logging.getLogger(__name__)


def _save_table(content: str, name: str):
    """Guarda una tabla LaTeX en results/tables/ y thesis/tables/."""
    for target_dir in [TABLES_DIR, THESIS_TABLES_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{name}.tex"
        filepath.write_text(content, encoding="utf-8")
        logger.info("Tabla guardada: %s", filepath)


def _fmt(value, decimals=2) -> str:
    """Formatea un numero con separador de miles y decimales."""
    if isinstance(value, int):
        return f"{value:,}".replace(",", "\\,")
    if isinstance(value, float):
        return f"{value:,.{decimals}f}".replace(",", "\\,")
    return str(value)


def generate_token_coverage_table(
    report: TokenCoverageReport,
    heuristic_coverage_pct: float = 58.8,
) -> str:
    """
    Genera tabla LaTeX con la cobertura real de tokens del tokenizer E5.

    Incluye comparacion con la estimacion heuristica del EDA (1.33 tok/palabra).

    Parameters
    ----------
    report : TokenCoverageReport
        Reporte de cobertura de tokens del tokenizer real.
    heuristic_coverage_pct : float
        Cobertura estimada con la heuristica del EDA para comparacion.

    Returns
    -------
    str
        Contenido LaTeX de la tabla.
    """
    truncated = report.total_songs - report.within_window
    truncated_pct = round(100.0 - report.coverage_pct, 2)

    content = r"""\begin{table}[htbp]
\centering
\caption{Cobertura de la ventana de contexto del tokenizer \texttt{multilingual-e5-small}.}
\label{tab:token_coverage}
\begin{tabular}{lr}
\toprule
Métrica & Valor \\
\midrule
"""
    rows = [
        ("Total de canciones", _fmt(report.total_songs)),
        ("Ventana de contexto", f"{_fmt(report.window_size)} tokens"),
        (f"Canciones dentro de ventana", f"{_fmt(report.within_window)} ({_fmt(report.coverage_pct, 1)}\\%)"),
        (f"Canciones truncadas", f"{_fmt(truncated)} ({_fmt(truncated_pct, 1)}\\%)"),
    ]

    for label, value in rows:
        content += f"{label} & {value} \\\\\n"

    content += "\\midrule\n"
    content += "\\multicolumn{2}{l}{\\textit{Distribución de tokens}} \\\\\n"
    content += "\\midrule\n"

    dist_rows = [
        ("Media", _fmt(report.mean_tokens, 1)),
        ("Mediana", _fmt(report.median_tokens, 1)),
        ("Desviación estándar", _fmt(report.std_tokens, 1)),
        ("Mínimo", _fmt(report.min_tokens)),
        ("Máximo", _fmt(report.max_tokens)),
        ("Percentil 90", _fmt(report.p90_tokens)),
        ("Percentil 95", _fmt(report.p95_tokens)),
        ("Percentil 99", _fmt(report.p99_tokens)),
    ]

    for label, value in dist_rows:
        content += f"{label} & {value} \\\\\n"

    content += "\\midrule\n"
    content += "\\multicolumn{2}{l}{\\textit{Comparación con heurística}} \\\\\n"
    content += "\\midrule\n"

    cmp_rows = [
        ("Cobertura estimada (1.33 tok/palabra)", f"{_fmt(heuristic_coverage_pct, 1)}\\%"),
        ("Cobertura real (tokenizer)", f"{_fmt(report.coverage_pct, 1)}\\%"),
    ]

    for label, value in cmp_rows:
        content += f"{label} & {value} \\\\\n"

    content += r"""\bottomrule
\end{tabular}
\end{table}"""

    _save_table(content, "token_coverage")
    return content
