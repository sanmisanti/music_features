"""
Generacion de tablas LaTeX para el analisis cualitativo (Capitulo 6).

Lee results/metrics/etapa6_qualitative.json (producido por
src.recommendation.generate_qualitative) y emite tablas booktabs en
results/tables/ y thesis/tables/.

Ejecutar con: python -m src.recommendation.qualitative_tables

Las tablas se generan desde codigo (no se editan a mano). La limpieza de
titulos (comillas del CSV, escape LaTeX, truncado) es deterministica.
"""

import json
import logging
import re

from src.config import METRICS_DIR, TABLES_DIR, THESIS_TABLES_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

QUAL_JSON = METRICS_DIR / "etapa6_qualitative.json"

MAX_TITLE = 34
MAX_ARTIST = 22


# =============================================================================
# LIMPIEZA Y ESCAPE
# =============================================================================

def _clean_raw(text: str) -> str:
    """Colapsa comillas dobles del CSV y recorta comillas/espacios sobrantes."""
    t = str(text).strip()
    t = t.replace('""', '"')
    t = t.strip('"').strip()
    return t


def _escape_latex(text: str) -> str:
    """Escapa caracteres especiales de LaTeX."""
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in text)


def _trunc(text: str, n: int) -> str:
    text = text.strip()
    if len(text) > n:
        return text[: n - 1].rstrip() + "\\ldots"
    return text


def title(text: str, n: int = MAX_TITLE) -> str:
    return _escape_latex(_trunc(_clean_raw(text), n))


def artist(text: str, n: int = MAX_ARTIST) -> str:
    return _escape_latex(_trunc(_clean_raw(text), n))


def genre(text: str) -> str:
    return str(text).replace("&", r"\&")


def _save(content: str, name: str):
    for target in (TABLES_DIR, THESIS_TABLES_DIR):
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{name}.tex").write_text(content, encoding="utf-8")
        logger.info("Tabla guardada: %s", target / f"{name}.tex")


# =============================================================================
# TABLA: LISTA DE RECOMENDACIONES (estudio de caso)
# =============================================================================

def recommendation_table(block, name, caption, label, note=None):
    """
    Top-10 de una query con componentes semantica/musical.
    Marca duplicados con daga y anota generos que difieren del de la query.
    """
    q_genre = block["query_genre"]
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \begin{threeparttable}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \small",
        r"  \begin{tabular}{c >{\raggedright\arraybackslash}p{4.5cm} "
        r">{\raggedright\arraybackslash}p{3.4cm} l c c c}",
        r"    \toprule",
        r"    \# & Canción & Artista & Gén. & Idm. & $s_{\text{sem}}$ & $s_{\text{mus}}$ \\",
        r"    \midrule",
    ]
    for r in block["recommendations"]:
        cross = "" if r["same_genre"] else r"\textsuperscript{*}"
        lines.append(
            f"    {r['rank']} & {title(r['name'])} & {artist(r['artist'])} & "
            f"{genre(r['genre'])}{cross} & {r['language']} & "
            f"{r['s_sem']:.3f} & {r['s_mus']:.3f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}[flushleft]",
        r"    \small",
        f"    \\item Consulta: \\emph{{{title(block['query_name'], 40)}}} "
        f"({genre(q_genre)}, {block['query_language']}). "
        r"$s_{\text{sem}}$: similitud coseno semántica; "
        r"$s_{\text{mus}}$: similitud musical (kernel gaussiano). "
        r"\textsuperscript{*}: género distinto al de la consulta. "
        r"La lista se deduplica: una sola grabación por canción "
        r"(los duplicados exactos del catálogo se colapsan).",
    ]
    if note:
        lines.append(f"    \\item {note}")
    lines += [
        r"  \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}",
    ]
    content = "\n".join(lines)
    _save(content, name)
    return content


# =============================================================================
# TABLA: ABLACION POR ALPHA
# =============================================================================

def ablation_table(entry, name, caption, label):
    """Tres columnas (alpha=0.00, 0.80, 1.00) con titulo e idioma por rank."""
    cols = ["0.00", "0.80", "1.00"]
    blocks = {c: entry["by_alpha"][c] for c in cols}
    q = blocks["0.80"]

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \begin{threeparttable}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \footnotesize",
        r"  \begin{tabular}{c p{3.9cm} p{3.9cm} p{3.9cm}}",
        r"    \toprule",
        r"    \# & $\alpha=0{,}00$ (solo musical) & $\alpha=0{,}80$ (híbrido) & "
        r"$\alpha=1{,}00$ (solo semántico) \\",
        r"    \midrule",
    ]
    for i in range(len(q["recommendations"])):
        cells = []
        for c in cols:
            r = blocks[c]["recommendations"][i]
            cells.append(f"{title(r['name'], 26)} ({r['language']})")
        lines.append(f"    {i + 1} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}[flushleft]",
        r"    \small",
        f"    \\item Consulta: \\emph{{{title(q['query_name'], 40)}}} "
        f"({genre(q['query_genre'])}, {q['query_language']}). "
        r"Entre paréntesis, el idioma de cada recomendación. "
        r"El peso $\alpha$ controla la mezcla semántica/musical de la fusión tardía.",
        r"  \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}",
    ]
    content = "\n".join(lines)
    _save(content, name)
    return content


# =============================================================================
# TABLA: RESUMEN CROSS-LANGUAGE
# =============================================================================

def crosslanguage_table(rows, name, caption, label):
    """
    rows: lista de (block, etiqueta). Resume la distribucion de idiomas del
    top-10 y cuantas recomendaciones comparten idioma con la consulta.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \begin{threeparttable}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \small",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    Consulta & Idioma & \multicolumn{3}{c}{Idiomas del top-10} \\",
        r"    \cmidrule(lr){3-5}",
        r"     & consulta & inglés & español & otros \\",
        r"    \midrule",
    ]
    for block, _label in rows:
        dist = block["lang_distribution"]
        en = dist.get("en", 0)
        es = dist.get("es", 0)
        otros = sum(v for k, v in dist.items() if k not in ("en", "es"))
        lines.append(
            f"    {title(block['query_name'], 26)} & {block['query_language']} & "
            f"{en} & {es} & {otros} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}[flushleft]",
        r"    \small",
        r"    \item Distribución de idiomas de las 10 recomendaciones al peso óptimo "
        r"($\alpha=0{,}80$). Las consultas en inglés no recuperan canciones en otros "
        r"idiomas; las consultas en español recuperan mayoritariamente español.",
        r"  \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}",
    ]
    content = "\n".join(lines)
    _save(content, name)
    return content


# =============================================================================
# MAIN
# =============================================================================

def main():
    data = json.loads(QUAL_JSON.read_text(encoding="utf-8"))

    cases = {_clean_raw(b["query_name"]).lower(): b
             for b in data["case_studies"] if b.get("found")}

    def find_case(substr):
        for k, b in cases.items():
            if substr in k:
                return b
        raise KeyError(substr)

    # --- Estudios de caso ---
    recommendation_table(
        find_case("vuelve"),
        "qual_case_vuelve",
        "Recomendaciones para \\emph{Vuelve} (Ricky Martin) al peso óptimo",
        "tab:qual_case_vuelve",
        note=r"El sistema recupera baladas de pérdida amorosa en español "
             r"(Gloria Estefan, Alejandro Fernández) y en inglés "
             r"(\emph{The Scientist}), evidenciando la fusión de ambas modalidades.",
    )
    recommendation_table(
        find_case("lose yourself"),
        "qual_case_lose_yourself",
        "Recomendaciones para \\emph{Lose Yourself} (Eminem) al peso óptimo",
        "tab:qual_case_lose_yourself",
        note=r"El vecindario más cercano de la consulta contenía varias "
             r"grabaciones repetidas de las mismas canciones (naturales en "
             r"catálogos musicales); la lista final las colapsa y muestra diez "
             r"canciones distintas de rap motivacional, con \emph{'Till I Collapse} "
             r"entre ellas.",
    )

    # --- Ablacion por alpha ---
    abl = {_clean_raw(e["spec"]["name"]).lower(): e for e in data["ablation"]}
    abl_vuelve = next(e for k, e in abl.items() if "vuelve" in k)
    ablation_table(
        abl_vuelve,
        "qual_ablation_vuelve",
        "Ablación del peso de fusión $\\alpha$ para \\emph{Vuelve} (Ricky Martin)",
        "tab:qual_ablation_vuelve",
    )
    abl_lose = next(e for k, e in abl.items() if "lose yourself" in k)
    ablation_table(
        abl_lose,
        "qual_ablation_lose_yourself",
        "Ablación del peso de fusión $\\alpha$ para \\emph{Lose Yourself} (Eminem)",
        "tab:qual_ablation_lose_yourself",
    )
    abl_shape = next(e for k, e in abl.items() if "shape of you" in k)
    ablation_table(
        abl_shape,
        "qual_ablation_shape_of_you",
        "Ablación del peso de fusión $\\alpha$ para \\emph{Shape of You} (Ed Sheeran)",
        "tab:qual_ablation_shape_of_you",
    )

    # --- Probes tematicos ---
    fiesta = data["thematic"]["celebracion_fiesta"]
    igf = next(b for b in fiesta if b.get("found") and "gotta feeling" in _clean_raw(b["query_name"]).lower())
    recommendation_table(
        igf,
        "qual_probe_celebracion",
        "Probe temático de celebración: \\emph{I Gotta Feeling} (The Black Eyed Peas)",
        "tab:qual_probe_celebracion",
        note=r"Tema con firma léxica concreta: el sistema recupera otros himnos "
             r"de fiesta (\emph{Time of Our Lives}, \emph{This Is How We Do It}).",
    )
    duelo = data["thematic"]["duelo_muerte"]
    syg = next((b for b in duelo if b.get("found") and "see you again" in _clean_raw(b["query_name"]).lower()), None)
    if syg:
        recommendation_table(
            syg,
            "qual_probe_duelo",
            "Probe temático de duelo: \\emph{See You Again} (Wiz Khalifa)",
            "tab:qual_probe_duelo",
            note=r"Tema de registro afectivo sin firma léxica única: el sistema "
                 r"recupera un conglomerado emocional de pérdida y ausencia "
                 r"(\emph{Thru These Tears}, \emph{Twenty One}), con más cruce de "
                 r"género que un tema de léxico explícito.",
        )

    protesta = data["thematic"]["protesta_politica"]
    zombie = next((b for b in protesta if b.get("found") and "zombie" in _clean_raw(b["query_name"]).lower()), None)
    if zombie:
        recommendation_table(
            zombie,
            "qual_probe_protesta",
            "Probe temático de protesta: \\emph{Zombie} (The Cranberries)",
            "tab:qual_probe_protesta",
            note=r"Para temas abstractos el sistema recupera principalmente el "
                 r"registro sonoro y de género; el acierto temático (\emph{21 Guns}, "
                 r"antibélica) es minoritario frente a vecinos de \textit{rock} genérico.",
        )

    # --- Cross-language ---
    cl = data["cross_language"]
    def first_found(group):
        return next((b for b in cl[group] if b.get("found")), None)
    cross_rows = []
    for group in ("en_desamor", "en_amor", "en_fiesta"):
        b = first_found(group)
        if b:
            cross_rows.append((b, group))
    for b in cl["es_query"]:
        if b.get("found"):
            cross_rows.append((b, "es_query"))
    crosslanguage_table(
        cross_rows,
        "qual_crosslanguage",
        "Distribución de idiomas de las recomendaciones según el idioma de la consulta",
        "tab:qual_crosslanguage",
    )

    logger.info("\nTablas cualitativas generadas.")


if __name__ == "__main__":
    main()
