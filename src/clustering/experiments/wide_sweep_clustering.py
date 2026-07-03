"""
Experimento confirmatorio: barrido ANCHO de hiperparametros de clustering.

Objetivo: verificar empiricamente que ampliar el rango de busqueda
(K de 2 a 20, min_cluster_size de 20 a 500) NO produce particiones de mayor
calidad que las reportadas en la seccion 5.6 del informe con los rangos de
produccion (K en {5,6,7,8}, min_cluster_size en {50,100,200,300,500}).

Responde a la pregunta metodologica: "y si con K=20 hay buenos numeros?".
Hallazgos (ver results/metrics/etapa4_wide_sweep.json y seccion 5.6.2):
- K alto NUNCA gana en ningun espacio: las configuraciones de mayor granularidad
  fragmentan los datos en micro-clusters < 1% que el filtro de degeneracion
  descarta. El techo del rango original (8) era mas que suficiente.
- El espacio operativo (semantic_umap, usado en el NMI cross-modal) mantiene su
  optimo hdbscan(mcs=200) k=3 Sil_macro=0.9137 sobre todo el rango ampliado.
- En espacios difusos (musical, semantico 384D), las particiones mas gruesas que
  el piso adoptado (K < 5) alcanzan un Silhouette ligeramente superior, coherente
  con su baja granularidad, pero no modifican ninguna conclusion del informe.

NO MODIFICA NADA DE PRODUCCION:
- No toca src/config.py.
- No regenera tablas LaTeX (thesis/tables/) ni figuras (thesis/figures/).
- No sobrescribe results/metrics/etapa4_clustering.json.
- Solo imprime un reporte y guarda results/metrics/etapa4_wide_sweep.json.

Reutiliza las mismas funciones, seeds y criterios de seleccion que
src/clustering/run_clustering.py, de modo que el resultado es comparable 1:1
con la seccion 5.6.

Ejecutar (desde la raiz del repositorio):
    python -m src.clustering.experiments.wide_sweep_clustering
"""

import json
import logging

import numpy as np

logging.basicConfig(
    level=logging.WARNING,  # silenciar el log por-configuracion; el reporte se imprime al final
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.config import (
    MAX_NOISE_PCT,
    METRICS_DIR,
    MIN_CLUSTER_PCT,
    NUMPY_SEED,
    RANDOM_SEED,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_PREPROCESSING_N_COMPONENTS,
    UNIFIED_DATASET,
)
from src.clustering.algorithms import run_all_algorithms
from src.clustering.evaluation import (
    evaluate_clustering,
    select_best_configuration,
)
from src.clustering.reduction import reduce_umap

# =============================================================================
# RANGOS ANCHOS (vs produccion: K={5,6,7,8}, mcs={50,100,200,300,500})
# =============================================================================
K_RANGE_WIDE = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
HDBSCAN_MCS_WIDE = [20, 30, 50, 100, 200, 300, 500]

# Rangos de produccion, para marcar que configs SI estaban en el barrido original
K_RANGE_PROD = {5, 6, 7, 8}
MCS_PROD = {50, 100, 200, 300, 500}


def _config_label(m):
    """Etiqueta legible de la configuracion."""
    if m.algorithm == "hdbscan":
        mcs = m.params.get("min_cluster_size", "?")
        return f"hdbscan(mcs={mcs})"
    return f"{m.algorithm}(K={m.k})"


def _in_production_range(m):
    if m.algorithm == "hdbscan":
        return m.params.get("min_cluster_size") in MCS_PROD
    return m.k in K_RANGE_PROD


def _min_cluster_pct(m):
    """Tamano del cluster mas pequeno (excluyendo ruido) como fraccion del total."""
    if not m.cluster_sizes or m.n_samples == 0:
        return 0.0
    # cluster_sizes puede incluir la etiqueta -1 (ruido); excluirla
    sizes = [v for k, v in m.cluster_sizes.items() if int(k) >= 0]
    if not sizes:
        return 0.0
    return min(sizes) / m.n_samples


def _is_valid(m):
    """Replica el criterio de select_best_configuration."""
    if m.has_degenerate_clusters or m.n_clusters_found < 2:
        return False
    if m.n_samples > 0 and (m.n_noise / m.n_samples) > MAX_NOISE_PCT:
        return False
    return True


def evaluate_space(name, data, metric, genre_labels):
    """Corre el barrido ancho sobre un espacio y evalua todas las configs."""
    results = run_all_algorithms(
        data=data,
        space_name=name,
        k_range=K_RANGE_WIDE,
        hdbscan_min_cluster_sizes=HDBSCAN_MCS_WIDE,
        hdbscan_metric=metric,
        random_state=RANDOM_SEED,
    )
    metrics = []
    for r in results:
        m = evaluate_clustering(
            data=data,
            labels=r.labels,
            genre_labels=genre_labels,
            algorithm=r.algorithm,
            space_name=r.space_name,
            k=r.k,
            n_noise=r.n_noise,
            has_degenerate_clusters=r.has_degenerate_clusters,
            cluster_sizes=r.cluster_sizes,
            params=r.params,
            elapsed_seconds=r.elapsed_seconds,
            metric=metric,
        )
        metrics.append(m)
    return metrics


def print_space_report(name, metrics):
    """Imprime todas las configs ordenadas por Silhouette macro."""
    print("\n" + "=" * 100)
    print(f"ESPACIO: {name}   (umbral cluster minimo = {MIN_CLUSTER_PCT:.0%}, "
          f"ruido maximo = {MAX_NOISE_PCT:.0%})")
    print("=" * 100)
    header = (f"{'config':<22}{'n_clust':>8}{'ruido%':>9}{'min_cl%':>9}"
              f"{'Sil_macro':>11}{'DB':>9}{'valida':>8}{'en_prod':>9}")
    print(header)
    print("-" * 100)

    # Ordenar por Sil_macro desc (los no calculables van al final con -inf)
    def sort_key(m):
        sm = m.silhouette_macro if m.silhouette_macro is not None else float("-inf")
        return -sm

    for m in sorted(metrics, key=sort_key):
        sm = m.silhouette_macro
        db = m.davies_bouldin
        sm_s = f"{sm:.4f}" if sm is not None else "  n/a"
        db_s = f"{db:.4f}" if db is not None else "  n/a"
        noise_pct = 100.0 * m.n_noise / m.n_samples if m.n_samples else 0.0
        mc_pct = 100.0 * _min_cluster_pct(m)
        valid = "si" if _is_valid(m) else "NO"
        in_prod = "si" if _in_production_range(m) else "-- nuevo"
        print(f"{_config_label(m):<22}{m.n_clusters_found:>8}{noise_pct:>8.1f}%"
              f"{mc_pct:>8.2f}%{sm_s:>11}{db_s:>9}{valid:>8}{in_prod:>9}")


def main():
    np.random.seed(NUMPY_SEED)

    print("Cargando dataset unificado...")
    data = np.load(str(UNIFIED_DATASET), allow_pickle=True)
    semantic_embeddings = data["semantic_embeddings"]
    musical_features = data["musical_features"]
    genre_labels = data["genre_labels"]
    print(f"  {len(semantic_embeddings)} muestras")

    print("Calculando proyeccion UMAP (mismos parametros que produccion)...")
    semantic_umap, _ = reduce_umap(
        semantic_embeddings,
        n_components=UMAP_PREPROCESSING_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    spaces = {
        "musical_13d": (musical_features, "euclidean"),
        "semantic_384d": (semantic_embeddings, "euclidean"),
        "semantic_umap": (semantic_umap, "euclidean"),
    }

    print(f"\nBarrido ANCHO: K en {K_RANGE_WIDE}")
    print(f"               min_cluster_size en {HDBSCAN_MCS_WIDE}")

    summary = {
        "k_range_wide": K_RANGE_WIDE,
        "hdbscan_mcs_wide": HDBSCAN_MCS_WIDE,
        "spaces": {},
    }

    for name, (d, metric) in spaces.items():
        print(f"\n>>> Procesando {name} (shape={d.shape})...")
        metrics = evaluate_space(name, d, metric, genre_labels)
        print_space_report(name, metrics)

        best = select_best_configuration(metrics, max_noise_pct=MAX_NOISE_PCT)
        # Mejor config valida que NO estaba en el rango de produccion (configs nuevas)
        new_valid = [
            m for m in metrics
            if _is_valid(m) and not _in_production_range(m)
        ]
        new_valid.sort(key=lambda m: (-m.silhouette_macro, m.davies_bouldin))
        best_new = new_valid[0] if new_valid else None

        print("\n  RESULTADO:")
        if best is not None:
            print(f"    Mejor config (rango ancho): {_config_label(best)}  "
                  f"Sil_macro={best.silhouette_macro:.4f}  "
                  f"DB={best.davies_bouldin:.4f}  "
                  f"{'[ya estaba en produccion]' if _in_production_range(best) else '[CONFIG NUEVA!]'}")
        else:
            print("    Ninguna configuracion valida.")
        if best_new is not None:
            print(f"    Mejor config NUEVA (fuera del rango prod): "
                  f"{_config_label(best_new)}  "
                  f"Sil_macro={best_new.silhouette_macro:.4f}  "
                  f"DB={best_new.davies_bouldin:.4f}")
        else:
            print("    Ninguna config NUEVA (fuera del rango prod) es valida.")

        summary["spaces"][name] = {
            "best_overall": None if best is None else {
                "config": _config_label(best),
                "silhouette_macro": best.silhouette_macro,
                "davies_bouldin": best.davies_bouldin,
                "in_production_range": _in_production_range(best),
            },
            "best_new_outside_prod": None if best_new is None else {
                "config": _config_label(best_new),
                "silhouette_macro": best_new.silhouette_macro,
                "davies_bouldin": best_new.davies_bouldin,
            },
            "all_configs": [
                {
                    "config": _config_label(m),
                    "n_clusters_found": m.n_clusters_found,
                    "noise_pct": round(100.0 * m.n_noise / m.n_samples, 2) if m.n_samples else 0.0,
                    "min_cluster_pct": round(100.0 * _min_cluster_pct(m), 3),
                    "silhouette_macro": m.silhouette_macro,
                    "davies_bouldin": m.davies_bouldin,
                    "valid": _is_valid(m),
                    "in_production_range": _in_production_range(m),
                }
                for m in metrics
            ],
        }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "etapa4_wide_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("CONCLUSION: para cada espacio, 'best_new_outside_prod' es None o tiene un")
    print("Sil_macro NO superior al mejor ya reportado en 5.6 para el espacio operativo.")
    print(f"Resumen JSON guardado en: {out_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
