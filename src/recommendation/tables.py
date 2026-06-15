"""
Generacion de tablas LaTeX para la Etapa 5 (recomendacion).

Genera tablas en formato booktabs, guardadas en results/tables/ y thesis/tables/.
"""

import logging
from typing import Dict, List

from src.config import TABLES_DIR, THESIS_TABLES_DIR
from src.recommendation.evaluation import RecommendationMetrics
from src.recommendation.optimizer import OptimizationResult

logger = logging.getLogger(__name__)


def _save_table(content: str, name: str):
    """Guarda una tabla LaTeX en results/tables/ y thesis/tables/."""
    for target_dir in [TABLES_DIR, THESIS_TABLES_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{name}.tex"
        filepath.write_text(content, encoding="utf-8")
        logger.info("Tabla guardada: %s", filepath)


def generate_fusion_strategy_table() -> str:
    """Tabla de estrategia de fusion (semantico vs musical)."""
    content = (
        r"\begin{table}[htbp]" "\n"
        r"  \centering" "\n"
        r"  \caption{Estrategia de fusión por modalidad}" "\n"
        r"  \label{tab:recommendation_fusion_strategy}" "\n"
        r"  \begin{tabular}{llll}" "\n"
        r"    \toprule" "\n"
        r"    Componente & Métrica & Conversión & Justificación \\" "\n"
        r"    \midrule" "\n"
        r"    Semántico (384D) & Coseno & Directa $[0, 1]$ & "
        r"Embeddings L2-normalizados \\" "\n"
        r"    Musical (13D) & Euclídea & Kernel gaussiano & "
        r"Features \textit{z-score}, preserva magnitud \\" "\n"
        r"    \bottomrule" "\n"
        r"  \end{tabular}" "\n"
        r"  \begin{tablenotes}[flushleft]" "\n"
        r"    \small" "\n"
        r"    \item Fusión tardía: $\text{score} = \alpha \cdot s_{\text{sem}} "
        r"+ (1 - \alpha) \cdot s_{\text{mus}}$. "
        r"El parámetro $\sigma$ del kernel se fija como la mediana de las "
        r"distancias euclídeas pairwise." "\n"
        r"  \end{tablenotes}" "\n"
        r"\end{table}"
    )
    _save_table(content, "recommendation_fusion_strategy")
    return content


def generate_grid_search_table(opt: OptimizationResult) -> str:
    """Tabla de resultados del grid search sobre alpha."""
    k_values = sorted(opt.k_values)
    k_headers = " & ".join([f"P@{k}" for k in k_values])

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Precision@$K$ en función del peso de fusión $\alpha$}",
        r"  \label{tab:recommendation_grid_search}",
        r"  \begin{tabular}{r" + "r" * len(k_values) + "}",
        r"    \toprule",
        f"    $\\alpha$ & {k_headers} \\\\",
        r"    \midrule",
    ]

    for alpha in opt.alpha_range:
        precisions = opt.results[alpha]
        values = " & ".join([f"{precisions[k]:.4f}" for k in k_values])
        marker = r" $\leftarrow$" if alpha == opt.best_alpha else ""
        lines.append(f"    {alpha:.2f} & {values}{marker} \\\\")

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}[flushleft]",
        r"    \small",
        f"    \\item $\\alpha$ = peso semántico. "
        f"Mejor configuración: $\\alpha = {opt.best_alpha:.2f}$ "
        f"(P@{opt.primary_k} = {opt.best_precision:.4f}).",
        r"  \end{tablenotes}",
        r"\end{table}",
    ])

    content = "\n".join(lines)
    _save_table(content, "recommendation_grid_search")
    return content


def generate_optimal_metrics_table(metrics: RecommendationMetrics) -> str:
    """Tabla resumen de metricas al alpha optimo."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Métricas del sistema de recomendación al peso óptimo}",
        r"  \label{tab:recommendation_optimal_metrics}",
        r"  \begin{tabular}{lr}",
        r"    \toprule",
        r"    Métrica & Valor \\",
        r"    \midrule",
        f"    Peso óptimo ($\\alpha$) & {metrics.alpha:.2f} \\\\",
        f"    P@5 & {metrics.precision_per_genre.get('_p5', 0.0):.4f} \\\\",
        f"    P@10 & {metrics.precision_at_k:.4f} \\\\",
        f"    P@20 & {metrics.precision_per_genre.get('_p20', 0.0):.4f} \\\\",
        r"    \midrule",
        f"    ILD semántica & {metrics.ild_semantic:.4f} \\\\",
        f"    ILD musical & {metrics.ild_musical:.4f} \\\\",
        r"    \midrule",
        f"    Cobertura del catálogo & {metrics.catalog_coverage:.4f} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    content = "\n".join(lines)
    _save_table(content, "recommendation_optimal_metrics")
    return content


def generate_precision_per_genre_table(
    metrics: RecommendationMetrics,
    genre_counts: Dict[str, int],
) -> str:
    """Tabla de Precision@K desglosada por genero."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Precision@10 por género musical}",
        r"  \label{tab:recommendation_precision_per_genre}",
        r"  \begin{tabular}{lrr}",
        r"    \toprule",
        r"    Género & $N$ canciones & P@10 \\",
        r"    \midrule",
    ]

    for genre in sorted(metrics.precision_per_genre.keys()):
        if genre.startswith("_"):
            continue
        n = genre_counts.get(genre, 0)
        p = metrics.precision_per_genre[genre]
        genre_escaped = genre.replace("&", r"\&")
        lines.append(f"    {genre_escaped} & {n:,} & {p:.4f} \\\\")

    lines.extend([
        r"    \midrule",
        f"    \\textbf{{Media}} & {metrics.n_queries:,} "
        f"& \\textbf{{{metrics.precision_at_k:.4f}}} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    content = "\n".join(lines)
    _save_table(content, "recommendation_precision_per_genre")
    return content


def generate_genre_coverage_table(metrics: RecommendationMetrics) -> str:
    """Tabla de cobertura del catalogo desglosada por genero."""
    coverage = {
        g: v for g, v in metrics.genre_coverage.items()
        if not g.startswith("_")
    }
    ordered = sorted(coverage.items(), key=lambda kv: kv[1], reverse=True)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Cobertura del catálogo por género al $\alpha$ óptimo}",
        r"  \label{tab:coverage_per_genre}",
        r"  \begin{tabular}{lr}",
        r"    \toprule",
        r"    Género & Cobertura \\",
        r"    \midrule",
    ]

    for genre, value in ordered:
        genre_escaped = genre.replace("&", r"\&")
        lines.append(f"    {genre_escaped} & {value * 100:.2f}\\% \\\\")

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    content = "\n".join(lines)
    _save_table(content, "coverage_per_genre")
    return content


def generate_v1_comparison_table(
    v2_metrics: RecommendationMetrics,
    v2_at_v1_weights: float,
    v1_precision: float = 0.398,
) -> str:
    """Tabla comparativa v1 vs v2."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Comparación con la primera ejecución del sistema}",
        r"  \label{tab:recommendation_v1_comparison}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Propiedad & v1 & v2 ($\alpha = 0{,}45$) & v2 (óptimo) \\",
        r"    \midrule",
        f"    Canciones & 7\\,811 & 17\\,964 & 17\\,964 \\\\",
        f"    Features musicales & 12D & 13D & 13D \\\\",
        f"    Modelo embeddings & MiniLM & E5-small & E5-small \\\\",
        f"    Métrica musical & coseno & euclídea & euclídea \\\\",
        f"    P@10 & {v1_precision:.3f} & {v2_at_v1_weights:.4f} "
        f"& {v2_metrics.precision_at_k:.4f} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}[flushleft]",
        r"    \small",
        r"    \item Comparación orientativa: datasets, modelos y métricas "
        r"difieren entre versiones.",
        r"  \end{tablenotes}",
        r"\end{table}",
    ]

    content = "\n".join(lines)
    _save_table(content, "recommendation_v1_comparison")
    return content
