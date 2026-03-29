"""
Modulo de generacion de tablas LaTeX para la Etapa 4 (clustering).

Genera tablas en formato booktabs, guardadas en results/tables/ y thesis/tables/.
"""

import logging
from typing import Dict, List, Optional

from src.config import TABLES_DIR, THESIS_TABLES_DIR
from src.clustering.hopkins import HopkinsReport
from src.clustering.evaluation import ClusteringMetrics
from src.clustering.purification import PurificationReport

logger = logging.getLogger(__name__)


def _save_table(content: str, name: str):
    """Guarda una tabla LaTeX en results/tables/ y thesis/tables/."""
    for target_dir in [TABLES_DIR, THESIS_TABLES_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{name}.tex"
        filepath.write_text(content, encoding="utf-8")
        logger.info("Tabla guardada: %s", filepath)


def generate_hopkins_table(reports: List[HopkinsReport]) -> str:
    """
    Genera tabla LaTeX con resultados de Hopkins por espacio.

    Columnas: Espacio, Dimensiones, Metrica, Media +/- DE, IC 95%, > 0.7.

    Parameters
    ----------
    reports : list of HopkinsReport
        Reportes de Hopkins para cada espacio.

    Returns
    -------
    str
        Contenido LaTeX de la tabla.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Estadístico de Hopkins por espacio de representación}",
        r"  \label{tab:hopkins_results}",
        r"  \begin{tabular}{llrrcc}",
        r"    \toprule",
        r"    Espacio & Métrica & Dims & $H$ (media $\pm$ DE) & IC 95\% & $> 0{,}7$ \\",
        r"    \midrule",
    ]

    # Nombres legibles para los espacios
    display_names = {
        "semantic_384d": "Semántico (original)",
        "semantic_umap": "Semántico (UMAP)",
        "musical_13d": "Musical",
        "concatenated": "Concatenado",
    }

    for r in reports:
        name = display_names.get(r.space_name, r.space_name)
        metric_display = "coseno" if r.metric == "cosine" else "euclídea"
        h_str = f"${r.mean:.4f} \\pm {r.std:.4f}$"
        ci_str = f"$[{r.ci_lower:.4f},\\; {r.ci_upper:.4f}]$"
        threshold_str = "Sí" if r.exceeds_threshold else "No"

        lines.append(
            f"    {name} & {metric_display} & {r.n_dimensions} "
            f"& {h_str} & {ci_str} & {threshold_str} \\\\"
        )

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}[flushleft]",
        r"    \small",
        r"    \item Nota: $H \approx 0{,}5$ indica distribución aleatoria; "
        r"$H > 0{,}7$ indica tendencia significativa al agrupamiento. "
        r"Cada valor es el promedio de 30 iteraciones bootstrap.",
        r"  \end{tablenotes}",
        r"\end{table}",
    ])

    content = "\n".join(lines)
    _save_table(content, "hopkins_results")
    return content


# =========================================================================
# TABLAS DE FASE B: MULTI-ALGORITMO
# =========================================================================

_SPACE_DISPLAY = {
    "semantic_384d": "Semántico (384D)",
    "semantic_umap": "Semántico (UMAP 30D)",
    "musical_13d": "Musical (13D)",
    "concatenated": "Concatenado (397D)",
}


def generate_clustering_comparison_table(
    metrics_list: List[ClusteringMetrics],
    space_name: str,
) -> str:
    """
    Tabla comparativa de algoritmos para un espacio.

    Columnas: Algoritmo, K, Sil-macro, Sil-micro, DB, CH, ARI, NMI, Deg.

    Parameters
    ----------
    metrics_list : list of ClusteringMetrics
        Metricas para todas las configuraciones de un espacio.
    space_name : str
        Nombre del espacio.

    Returns
    -------
    str
        Contenido LaTeX.
    """
    space_display = _SPACE_DISPLAY.get(space_name, space_name)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{Comparación de algoritmos de clustering --- {space_display}}}",
        f"  \\label{{tab:clustering_comparison_{space_name}}}",
        r"  \begin{tabular}{llrrrrrrc}",
        r"    \toprule",
        r"    Algoritmo & $K$ & Sil\textsubscript{macro} & Sil\textsubscript{micro} "
        r"& DB & CH & ARI & NMI & Deg \\",
        r"    \midrule",
    ]

    for m in sorted(metrics_list, key=lambda x: (x.algorithm, x.k)):
        algo = m.algorithm.upper() if m.algorithm == "hdbscan" else m.algorithm.capitalize()
        k_str = str(m.k)
        if m.algorithm == "hdbscan":
            mcs = m.params.get("min_cluster_size", "?")
            k_str = f"{m.n_clusters_found} ({mcs})"

        deg = "Sí" if m.has_degenerate_clusters else "---"

        if m.n_clusters_found < 2:
            lines.append(
                f"    {algo} & {k_str} & --- & --- & --- & --- "
                f"& --- & --- & {deg} \\\\"
            )
        else:
            lines.append(
                f"    {algo} & {k_str} "
                f"& {m.silhouette_macro:.4f} & {m.silhouette_micro:.4f} "
                f"& {m.davies_bouldin:.4f} & {m.calinski_harabasz:.0f} "
                f"& {m.ari_genre:.4f} & {m.nmi_genre:.4f} & {deg} \\\\"
            )

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}[flushleft]",
        r"    \small",
        r"    \item Sil\textsubscript{macro}: Silhouette macro-averaged (métrica primaria). "
        r"DB: Davies-Bouldin (menor = mejor). CH: Calinski-Harabasz (mayor = mejor). "
        r"ARI/NMI: contra género como proxy. Deg: cluster degenerado ($<1\%$ del dataset). "
        r"Para HDBSCAN, $K$ muestra clusters encontrados (min\_cluster\_size).",
        r"  \end{tablenotes}",
        r"\end{table}",
    ])

    content = "\n".join(lines)
    _save_table(content, f"clustering_comparison_{space_name}")
    return content


def generate_best_configurations_table(
    best_per_space: Dict[str, Optional[ClusteringMetrics]],
) -> str:
    """
    Tabla resumen con la mejor configuracion por espacio.

    Parameters
    ----------
    best_per_space : dict
        {space_name: ClusteringMetrics or None}

    Returns
    -------
    str
        Contenido LaTeX.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Mejor configuración de clustering por espacio}",
        r"  \label{tab:clustering_best_configs}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    Espacio & Algoritmo & $K$ & Sil\textsubscript{macro} & DB & NMI\textsubscript{género} \\",
        r"    \midrule",
    ]

    for space_name, m in best_per_space.items():
        space_display = _SPACE_DISPLAY.get(space_name, space_name)
        if m is None:
            lines.append(
                f"    {space_display} & --- & --- & --- & --- & --- \\\\"
            )
        else:
            algo = m.algorithm.upper() if m.algorithm == "hdbscan" else m.algorithm.capitalize()
            lines.append(
                f"    {space_display} & {algo} & {m.k} "
                f"& {m.silhouette_macro:.4f} & {m.davies_bouldin:.4f} "
                f"& {m.nmi_genre:.4f} \\\\"
            )

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    content = "\n".join(lines)
    _save_table(content, "clustering_best_configs")
    return content


def generate_cross_modal_nmi_table(
    nmi_value: float,
    semantic_config: Optional[ClusteringMetrics],
    musical_config: Optional[ClusteringMetrics],
) -> str:
    """
    Tabla con NMI cross-modal y las configuraciones comparadas.

    Parameters
    ----------
    nmi_value : float
        NMI entre clusters semanticos y musicales.
    semantic_config : ClusteringMetrics or None
        Mejor configuracion semantica.
    musical_config : ClusteringMetrics or None
        Mejor configuracion musical.

    Returns
    -------
    str
        Contenido LaTeX.
    """
    sem_desc = "---"
    mus_desc = "---"
    if semantic_config is not None:
        sem_algo = semantic_config.algorithm.capitalize()
        sem_desc = f"{sem_algo}, $K={semantic_config.k}$"
    if musical_config is not None:
        mus_algo = musical_config.algorithm.capitalize()
        mus_desc = f"{mus_algo}, $K={musical_config.k}$"

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Complementariedad cross-modal (NMI entre clusterings)}",
        r"  \label{tab:cross_modal_nmi}",
        r"  \begin{tabular}{lr}",
        r"    \toprule",
        r"    Propiedad & Valor \\",
        r"    \midrule",
        f"    Clustering semántico & {sem_desc} \\\\",
        f"    Clustering musical & {mus_desc} \\\\",
        f"    NMI cross-modal & ${nmi_value:.4f}$ \\\\",
        r"    \midrule",
        f"    Interpretación & {'Alta complementariedad' if nmi_value < 0.15 else 'Complementariedad moderada/baja'} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}[flushleft]",
        r"    \small",
        r"    \item NMI $< 0{,}15$ indica que los clusters semánticos y musicales capturan "
        r"información complementaria, justificando la fusión multimodal.",
        r"  \end{tablenotes}",
        r"\end{table}",
    ]

    content = "\n".join(lines)
    _save_table(content, "cross_modal_nmi")
    return content


# =========================================================================
# TABLAS DE FASE C: PURIFICACION
# =========================================================================


def generate_purification_table(reports: List[PurificationReport]) -> str:
    """
    Tabla pre/post purificacion para cada configuracion purificada.

    Parameters
    ----------
    reports : list of PurificationReport
        Reportes de purificacion.

    Returns
    -------
    str
        Contenido LaTeX.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Efecto de la purificación sobre la calidad de clusters}",
        r"  \label{tab:purification_results}",
        r"  \begin{tabular}{llrrrrrr}",
        r"    \toprule",
        r"    Espacio & Fase & $N$ & Sil\textsubscript{macro} "
        r"& Sil\textsubscript{micro} & DB & Removidos & \% \\",
        r"    \midrule",
    ]

    for r in reports:
        space_display = _SPACE_DISPLAY.get(r.space_name, r.space_name)

        lines.append(
            f"    {space_display} & Pre & {r.n_samples_before:,} "
            f"& {r.silhouette_macro_before:.4f} & {r.silhouette_micro_before:.4f} "
            f"& {r.davies_bouldin_before:.4f} & --- & --- \\\\".replace(",", "\\,")
        )
        lines.append(
            f"    & Post & {r.n_samples_after:,} "
            f"& {r.silhouette_macro_after:.4f} & {r.silhouette_micro_after:.4f} "
            f"& {r.davies_bouldin_after:.4f} & {r.n_removed:,} "
            f"& {r.removal_pct * 100:.1f}\\% \\\\".replace(",", "\\,")
        )
        # Linea de mejora
        sil_sign = "+" if r.silhouette_improvement >= 0 else ""
        db_sign = "+" if r.db_improvement >= 0 else ""
        lines.append(
            f"    & $\\Delta$ & --- "
            f"& ${sil_sign}{r.silhouette_improvement:.4f}$ & --- "
            f"& ${db_sign}{r.db_improvement:.4f}$ & --- & --- \\\\"
        )
        lines.append(r"    \midrule")

    # Quitar ultimo midrule y poner bottomrule
    if lines[-1] == r"    \midrule":
        lines[-1] = r"    \bottomrule"

    lines.extend([
        r"  \end{tabular}",
        r"  \begin{tablenotes}[flushleft]",
        r"    \small",
        r"    \item Estrategia combinada: eliminación de puntos con Silhouette $< 0$ "
        r"y outliers a $> 2\sigma$ del centroide. "
        r"Presupuesto máximo de remoción: 15\%.",
        r"  \end{tablenotes}",
        r"\end{table}",
    ])

    content = "\n".join(lines)
    _save_table(content, "purification_results")
    return content
