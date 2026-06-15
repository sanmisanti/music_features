"""
Generacion de la tabla LaTeX de tiempos de ejecucion del pipeline.

Agrega el tiempo total de cada etapa leyendo los archivos JSON de
results/metrics/. Cada orquestador (run_*.py) registra su tiempo total bajo
la clave de nivel superior 'total_elapsed_seconds'. Para las dos etapas que
ya persistian su tiempo con otra clave (vectorizacion y recomendacion) se leen
los campos preexistentes, de modo que no es necesario re-ejecutarlas.

Mientras una etapa no haya sido re-ejecutada con la instrumentacion vigente,
su tiempo no estara en el JSON; en ese caso se usa el valor medido y
documentado de la ejecucion previa (registrado en los logs) como respaldo
provisional. Tras una re-ejecucion completa del pipeline, todos los valores
provienen de los JSON y la dependencia de los respaldos desaparece.

Uso:
    python -m src.reporting.timing
"""

import json
import logging
from typing import List, Optional, Tuple

from src.config import METRICS_DIR, TABLES_DIR, THESIS_TABLES_DIR

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Definicion de fuentes por etapa
# -----------------------------------------------------------------------------
# Cada fuente es (archivo_json, [rutas_de_clave_candidatas], respaldo_segundos).
# Las rutas se prueban en orden; la primera presente se utiliza. El respaldo es
# el tiempo medido en la ejecucion previa (de logs), usado solo si el JSON aun
# no registra el tiempo de esa etapa.
_Source = Tuple[str, List[str], float]

# Cada fila de la tabla agrega (suma) una o mas fuentes.
_ROWS: List[Tuple[str, List[_Source]]] = [
    ("EDA y preprocesamiento", [
        ("etapa2_eda.json", ["total_elapsed_seconds"], 12.2),
        ("etapa3_preprocessing.json", ["total_elapsed_seconds"], 4.4),
    ]),
    ("Vectorización semántica (E5)", [
        ("etapa3_vectorization.json",
         ["total_elapsed_seconds", "vectorization.elapsed_seconds"], 7378.43),
    ]),
    ("Normalización y unificación", [
        ("etapa3_normalization.json", ["total_elapsed_seconds"], 0.5),
        ("etapa3_unification.json", ["total_elapsed_seconds"], 0.3),
    ]),
    ("Hopkins (30 iteraciones, 4 espacios)", [
        ("etapa4_hopkins.json", ["total_elapsed_seconds"], 90.0),
    ]),
    ("Multi-algoritmo de clustering", [
        ("etapa4_clustering.json", ["total_elapsed_seconds"], 2400.0),
    ]),
    ("Purificación combinada", [
        ("etapa4_purification.json", ["total_elapsed_seconds"], 88.0),
    ]),
    ("Recomendación completa", [
        ("etapa5_recommendation.json",
         ["total_elapsed_seconds", "timing.total_seconds"], 135.7),
    ]),
]


def _dig(data: dict, dotted_key: str) -> Optional[float]:
    """Devuelve el valor de una ruta de clave punteada, o None si no existe."""
    node = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, (int, float)) else None


def _resolve_source(source: _Source) -> Tuple[float, bool]:
    """Resuelve una fuente a (segundos, proviene_de_json)."""
    filename, key_paths, fallback = source
    path = METRICS_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in key_paths:
            value = _dig(data, key)
            if value is not None:
                return float(value), True
    return float(fallback), False


def _fmt_seconds(seconds: float) -> str:
    """Formatea segundos como cadena LaTeX legible."""
    if seconds < 1:
        return r"$<$ 1 s"
    s = int(round(seconds))
    sep = f"{s:,}".replace(",", r"\,")
    if s >= 3600:
        h = s // 3600
        m = round((s % 3600) / 60)
        return f"{sep} s ($\\approx$ {h} h {m:02d} min)"
    if s >= 120:
        m = round(s / 60)
        return f"{sep} s ($\\approx$ {m} min)"
    return f"{sep} s"


def _save_table(content: str, name: str):
    """Guarda una tabla LaTeX en results/tables/ y thesis/tables/."""
    for target_dir in [TABLES_DIR, THESIS_TABLES_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{name}.tex"
        filepath.write_text(content, encoding="utf-8")
        logger.info("Tabla guardada: %s", filepath)


def generate_timing_table() -> str:
    """Genera la tabla de tiempos de ejecucion del pipeline."""
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Tiempos de ejecución por etapa del pipeline (CPU, sin GPU).}",
        r"    \label{tab:tiempos_ejecucion}",
        r"    \begin{tabular}{lr}",
        r"        \toprule",
        r"        Etapa & Tiempo aproximado \\",
        r"        \midrule",
    ]

    n_fallback = 0
    for label, sources in _ROWS:
        total = 0.0
        for source in sources:
            value, from_json = _resolve_source(source)
            total += value
            if not from_json:
                n_fallback += 1
        lines.append(f"        {label} & {_fmt_seconds(total)} \\\\")

    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])

    if n_fallback:
        logger.warning(
            "%d fuente(s) de tiempo no estaban en los JSON; se usaron valores "
            "de respaldo de la ejecucion previa. Re-ejecute las etapas "
            "correspondientes para que la tabla se genere unicamente desde JSON.",
            n_fallback,
        )

    content = "\n".join(lines)
    _save_table(content, "pipeline_timing")
    return content


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    print(generate_timing_table())
