"""
Generacion de datos para el analisis cualitativo del sistema de recomendacion
(Capitulo 6: Resultados Experimentales).

Ejecutar con: python -m src.recommendation.generate_qualitative

Produce, de forma NO interactiva y reproducible, la evidencia cualitativa que
respalda el Capitulo 6:

- Estudios de caso: top-10 al alpha optimo (0.80) para queries representativas.
- Ablacion por alpha: top-10 a alpha en {0.0, 0.80, 1.0} para queries selectas.
- Probes tematicos: top-10 para queries de temas especificos, cruzando generos.
- Cross-language: top-10 con distribucion de idiomas de las recomendaciones.

Salidas:
- results/metrics/etapa6_qualitative.json  (datos estructurados)
- results/qualitative_report.txt            (reporte legible para analisis)

No genera tablas LaTeX: la seleccion de casos y el diseno de tablas se hacen
DESPUES de revisar este reporte, en un segundo paso.
"""

import json
import logging
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.config import (
    METRICS_DIR,
    NUMPY_SEED,
    RESULTS_DIR,
    SOURCE_DATASET,
    SOURCE_SEPARATOR,
    UNIFIED_DATASET,
)
from src.recommendation.engine import (
    compute_duplicate_groups,
    compute_sigma,
    gaussian_kernel,
    normalize_text,
)

ALPHA_OPTIMAL = 0.80  # optimo por grid search (Etapa 5)
TOP_K = 10

# =============================================================================
# DEFINICION DE QUERIES
# =============================================================================
# Cada query se resuelve por coincidencia (substring, case-insensitive) de
# nombre y (opcionalmente) artista. Si una no existe en el catalogo, se reporta
# como no encontrada y se omite. Se dan listas generosas de candidatos para los
# probes tematicos y cross-language; se analiza luego cuales resolvieron.

CASE_STUDIES = [
    {"name": "Vuelve", "artist": "Ricky Martin"},
    {"name": "Lose Yourself", "artist": "Eminem"},
    {"name": "Shape of You", "artist": "Ed Sheeran"},
    {"name": "Piano Man", "artist": "Billy Joel"},
    {"name": "Welcome To The Machine", "artist": "Pink Floyd"},
]

# Queries para la ablacion por alpha (subconjunto que exhibe bien el contraste
# musical vs semantico).
ABLATION_QUERIES = [
    {"name": "Vuelve", "artist": "Ricky Martin"},
    {"name": "Lose Yourself", "artist": "Eminem"},
    {"name": "Shape of You", "artist": "Ed Sheeran"},
]

# Probes tematicos: tema -> lista de canciones candidatas del mismo tema.
THEMATIC_PROBES = {
    "adiccion": [
        {"name": "When I'm Gone", "artist": "Eminem"},
        {"name": "Rehab", "artist": "Amy Winehouse"},
        {"name": "Otherside", "artist": "Macklemore"},
        {"name": "Under the Bridge", "artist": "Red Hot Chili Peppers"},
        {"name": "Sober", "artist": "P!nk"},
    ],
    "duelo_muerte": [
        {"name": "Tears in Heaven", "artist": "Eric Clapton"},
        {"name": "See You Again", "artist": "Wiz Khalifa"},
        {"name": "Hurt", "artist": "Johnny Cash"},
        {"name": "Fire and Rain", "artist": "James Taylor"},
    ],
    "protesta_politica": [
        {"name": "American Idiot", "artist": "Green Day"},
        {"name": "Killing In The Name", "artist": "Rage Against"},
        {"name": "Fortunate Son", "artist": "Creedence"},
        {"name": "Zombie", "artist": "Cranberries"},
        {"name": "Sunday Bloody Sunday", "artist": "U2"},
    ],
    "celebracion_fiesta": [
        {"name": "I Gotta Feeling", "artist": "Black Eyed Peas"},
        {"name": "Party Rock Anthem", "artist": "LMFAO"},
        {"name": "Uptown Funk", "artist": "Mark Ronson"},
        {"name": "Celebration", "artist": "Kool"},
    ],
}

# Cross-language: queries en ingles con temas universales y queries en espanol.
# Para cada una se reporta la distribucion de idiomas de las top-10.
CROSS_LANGUAGE = {
    "en_desamor": [
        {"name": "Someone Like You", "artist": "Adele"},
        {"name": "Nothing Compares", "artist": "Sinead"},
    ],
    "en_fiesta": [
        {"name": "I Gotta Feeling", "artist": "Black Eyed Peas"},
    ],
    "en_amor": [
        {"name": "Perfect", "artist": "Ed Sheeran"},
    ],
    "es_query": [
        {"name": "Vuelve", "artist": "Ricky Martin"},
        {"name": "Vivir Mi Vida", "artist": "Marc Anthony"},
        {"name": "La Camisa Negra", "artist": "Juanes"},
        {"name": "Bailando", "artist": "Enrique Iglesias"},
    ],
}


# =============================================================================
# UTILIDADES
# =============================================================================

# Normalizacion compartida con el motor (misma logica de agrupamiento de duplicados).
_norm = normalize_text


def resolve_query(spec, track_names, track_artists):
    """
    Resuelve una query {name, artist} al indice del catalogo.

    Devuelve (idx, n_matches, all_match_indices) o (None, 0, []) si no existe.
    Prefiere coincidencia exacta de nombre; ante empate, la primera aparicion.
    """
    name_q = _norm(spec["name"])
    artist_q = _norm(spec.get("artist", ""))

    exact, partial = [], []
    for i in range(len(track_names)):
        name_i = _norm(track_names[i])
        artist_i = _norm(track_artists[i])
        if artist_q and artist_q not in artist_i:
            continue
        if name_q == name_i:
            exact.append(i)
        elif name_q in name_i:
            partial.append(i)

    matches = exact if exact else partial
    if not matches:
        return None, 0, []
    return matches[0], len(matches), matches


def recommend(idx, alpha, top_k, embeddings, musical, sigma, dup_groups):
    """
    Genera top-K recomendaciones deduplicadas para `idx` al peso `alpha`.

    Deduplicacion consistente con el motor: sobre-muestrea candidatos, colapsa
    por grupo (una grabacion por grupo) y excluye el grupo de la propia query.
    """
    s_sem = embeddings @ embeddings[idx]
    d_mus = np.linalg.norm(musical - musical[idx], axis=1)
    s_mus = gaussian_kernel(d_mus, sigma)
    fused = alpha * s_sem + (1.0 - alpha) * s_mus
    fused[idx] = -np.inf

    over_k = min(len(fused) - 1, max(top_k * 10, 100))
    cand = np.argpartition(fused, -over_k)[-over_k:]
    cand = cand[np.argsort(-fused[cand])]

    q_group = dup_groups[idx]
    seen_groups = set()
    seen_sigs = set()
    picked = []
    for j in cand:
        # Misma grabacion que la consulta (letra identica y audio practicamente
        # identico): no recomendarla, aunque tenga metadatos distintos. Una
        # version legitima (remaster, vivo, cover) difiere mas en lo musical.
        if s_sem[j] >= 0.9999 and s_mus[j] >= 0.999:
            continue
        g = dup_groups[j]
        if g == q_group or g in seen_groups:
            continue
        # Grabacion identica con otro rotulo: identica similitud a la consulta
        # en ambas modalidades. Una variante legitima (remaster, vivo, cover)
        # difiere en el vector musical, por lo que su firma no coincide.
        sig = (round(float(s_sem[j]), 5), round(float(s_mus[j]), 5))
        if sig in seen_sigs:
            continue
        seen_groups.add(g)
        seen_sigs.add(sig)
        picked.append(j)
        if len(picked) == top_k:
            break
    top = np.asarray(picked, dtype=int)
    return top, fused[top], s_sem[top], s_mus[top]


def build_rec_records(idx, alpha, top_k, ctx):
    """Construye la lista de dicts de una recomendacion (para JSON/reporte)."""
    embeddings, musical, sigma = ctx["embeddings"], ctx["musical"], ctx["sigma"]
    names, artists = ctx["names"], ctx["artists"]
    genres, langs = ctx["genres"], ctx["langs"]

    top, scores, s_sem, s_mus = recommend(
        idx, alpha, top_k, embeddings, musical, sigma, ctx["dup_groups"]
    )
    q_genre = str(genres[idx])
    recs = []
    norm_seen = Counter()
    for rank, j in enumerate(top, start=1):
        norm_key = _norm(names[j]) + "|" + _norm(artists[j])
        norm_seen[norm_key] += 1
        recs.append({
            "rank": rank,
            "idx": int(j),
            "name": str(names[j]),
            "artist": str(artists[j]),
            "genre": str(genres[j]),
            "language": str(langs[j]),
            "score": round(float(scores[rank - 1]), 4),
            "s_sem": round(float(s_sem[rank - 1]), 4),
            "s_mus": round(float(s_mus[rank - 1]), 4),
            "same_genre": str(genres[j]) == q_genre,
        })
    # marcar duplicados por (nombre, artista) normalizados
    dup_keys = {k for k, c in norm_seen.items() if c > 1}
    for r in recs:
        r["is_duplicate"] = (_norm(r["name"]) + "|" + _norm(r["artist"])) in dup_keys
    return recs


def query_block(spec, ctx, alpha=ALPHA_OPTIMAL, top_k=TOP_K):
    """Resuelve una query y genera su recomendacion al peso `alpha`."""
    idx, n_matches, _ = resolve_query(spec, ctx["names"], ctx["artists"])
    if idx is None:
        return {"spec": spec, "found": False}
    recs = build_rec_records(idx, alpha, top_k, ctx)
    langs_top = Counter(r["language"] for r in recs)
    genres_top = Counter(r["genre"] for r in recs)
    return {
        "spec": spec,
        "found": True,
        "query_idx": int(idx),
        "query_name": str(ctx["names"][idx]),
        "query_artist": str(ctx["artists"][idx]),
        "query_genre": str(ctx["genres"][idx]),
        "query_language": str(ctx["langs"][idx]),
        "n_catalog_matches": n_matches,
        "alpha": alpha,
        "recommendations": recs,
        "lang_distribution": dict(langs_top),
        "genre_distribution": dict(genres_top),
        "n_duplicates": sum(1 for r in recs if r["is_duplicate"]),
    }


# =============================================================================
# REPORTE LEGIBLE
# =============================================================================

def fmt_block(block):
    lines = []
    if not block["found"]:
        s = block["spec"]
        lines.append(f"  [NO ENCONTRADA] {s['name']} - {s.get('artist', '')}")
        return "\n".join(lines)
    lines.append(
        f'  Query [{block["query_idx"]}] "{block["query_name"]}" - '
        f'{block["query_artist"]} '
        f'(genero={block["query_genre"]}, idioma={block["query_language"]}, '
        f'alpha={block["alpha"]:.2f}, coincidencias_catalogo={block["n_catalog_matches"]})'
    )
    lines.append(
        f"  {'#':<3}{'score':<8}{'sem':<8}{'mus':<8}{'gen':<7}{'idm':<6}"
        f"{'=g':<4}{'dup':<5}{'cancion':<40}{'artista'}"
    )
    for r in block["recommendations"]:
        same = "OK" if r["same_genre"] else "--"
        dup = "DUP" if r["is_duplicate"] else ""
        lines.append(
            f"  {r['rank']:<3}{r['score']:<8.4f}{r['s_sem']:<8.4f}{r['s_mus']:<8.4f}"
            f"{r['genre']:<7}{r['language']:<6}{same:<4}{dup:<5}"
            f"{r['name'][:38]:<40}{r['artist'][:24]}"
        )
    lines.append(
        f"  -> idiomas: {block['lang_distribution']} | "
        f"generos: {block['genre_distribution']} | "
        f"duplicados: {block['n_duplicates']}"
    )
    return "\n".join(lines)


def main():
    np.random.seed(NUMPY_SEED)

    print("Cargando dataset unificado...")
    data = np.load(str(UNIFIED_DATASET), allow_pickle=True)
    embeddings = data["semantic_embeddings"].astype(np.float32)
    musical = data["musical_features"].astype(np.float32)
    track_ids = data["track_ids"]
    names = data["track_names"]
    artists = data["track_artists"]
    genres = data["genre_labels"]
    n = len(embeddings)
    print(f"Dataset: {n} canciones.")

    # Idioma: unir por track_id desde el dataset fuente (separador @@).
    print("Cargando idiomas desde dataset fuente...")
    df = pd.read_csv(
        SOURCE_DATASET, sep=SOURCE_SEPARATOR, encoding="utf-8", engine="python",
    )
    lang_by_id = df.set_index("track_id")["language"]
    langs = np.array([
        str(lang_by_id.get(tid, "unknown")) for tid in track_ids
    ], dtype=object)
    # normalizar NaN -> unknown
    langs = np.array([
        "unknown" if (l is None or l == "nan" or l == "") else l for l in langs
    ], dtype=object)
    print(f"Idiomas distribucion global (top): "
          f"{Counter(langs).most_common(8)}")

    print("Calculando sigma...")
    sigma = compute_sigma(musical)
    print(f"Sigma = {sigma:.4f}")

    # Grupos de duplicados: las listas mostradas se deduplican (una grabacion
    # por grupo de nombre+artista). Esto afecta solo la demostracion cualitativa;
    # la evaluacion cuantitativa (Etapa 5) se computa sobre las listas sin filtrar.
    dup_groups = compute_duplicate_groups(names, artists)
    n_unique = int(len(np.unique(dup_groups)))
    print(f"Deduplicacion: {n_unique} grabaciones unicas "
          f"({n - n_unique} entradas duplicadas)")

    ctx = {
        "embeddings": embeddings, "musical": musical, "sigma": sigma,
        "names": names, "artists": artists, "genres": genres, "langs": langs,
        "dup_groups": dup_groups,
    }

    report = []
    payload = {
        "alpha_optimal": ALPHA_OPTIMAL,
        "top_k": TOP_K,
        "sigma": round(sigma, 6),
        "n_songs": n,
        "seed": NUMPY_SEED,
    }

    # --- Estudios de caso (alpha optimo) ---
    print("Generando estudios de caso...")
    report.append("=" * 90)
    report.append("SECCION 1 — ESTUDIOS DE CASO (alpha = 0.80)")
    report.append("=" * 90)
    case_blocks = [query_block(s, ctx) for s in CASE_STUDIES]
    payload["case_studies"] = case_blocks
    for b in case_blocks:
        report.append(fmt_block(b))
        report.append("")

    # --- Ablacion por alpha ---
    print("Generando ablacion por alpha...")
    report.append("=" * 90)
    report.append("SECCION 2 — ABLACION POR ALPHA (0.00 solo musical | 0.80 hibrido | 1.00 solo semantico)")
    report.append("=" * 90)
    ablation_blocks = []
    for s in ABLATION_QUERIES:
        entry = {"spec": s, "by_alpha": {}}
        for a in (0.0, ALPHA_OPTIMAL, 1.0):
            b = query_block(s, ctx, alpha=a)
            entry["by_alpha"][f"{a:.2f}"] = b
            report.append(f"--- alpha = {a:.2f} ---")
            report.append(fmt_block(b))
            report.append("")
        ablation_blocks.append(entry)
    payload["ablation"] = ablation_blocks

    # --- Probes tematicos ---
    print("Generando probes tematicos...")
    report.append("=" * 90)
    report.append("SECCION 3 — PROBES TEMATICOS")
    report.append("=" * 90)
    thematic = {}
    for theme, specs in THEMATIC_PROBES.items():
        report.append(f"\n### TEMA: {theme} ###")
        thematic[theme] = [query_block(s, ctx) for s in specs]
        for b in thematic[theme]:
            report.append(fmt_block(b))
            report.append("")
    payload["thematic"] = thematic

    # --- Cross-language ---
    print("Generando cross-language...")
    report.append("=" * 90)
    report.append("SECCION 4 — CROSS-LANGUAGE (distribucion de idiomas en top-10)")
    report.append("=" * 90)
    cross = {}
    for group, specs in CROSS_LANGUAGE.items():
        report.append(f"\n### GRUPO: {group} ###")
        cross[group] = [query_block(s, ctx) for s in specs]
        for b in cross[group]:
            report.append(fmt_block(b))
            report.append("")
    payload["cross_language"] = cross

    # --- Guardar ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "qualitative_report.txt"
    json_path = METRICS_DIR / "etapa6_qualitative.json"
    report_path.write_text("\n".join(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\nReporte legible: {report_path}")
    print(f"Datos JSON:      {json_path}")
    print("\nLISTO. Revisar el reporte para el analisis cualitativo del Capitulo 6.")


if __name__ == "__main__":
    main()
