"""
Baseline aleatorio de Intra-List Diversity (ILD).

Ejecutar con: python -m src.recommendation.baseline_ild

Proposito
---------
La ILD carece de escala absoluta: su magnitud depende por completo de la
distancia empleada (coseno acotada en el espacio semantico, euclidea sin cota
superior en el espacio musical). Por lo tanto, un valor observado (p. ej.
ILD musical = 2,6357) no es interpretable como "bueno" o "malo" en el vacio:
necesita un punto de referencia medido con la MISMA distancia.

Este script computa ese punto de referencia: la ILD esperada de un
recomendador que responde cada consulta con K canciones elegidas al azar del
catalogo. Contrastar la ILD del sistema contra este piso permite afirmar si
las listas reales son mas cohesivas (ILD menor) que una seleccion aleatoria.

Metodo
------
1.  Fija el seed de reproducibilidad (mismo NUMPY_SEED del sistema).
2.  Carga el dataset unificado (embeddings semanticos + features musicales).
3.  Genera N listas aleatorias de K canciones distintas cada una. Por defecto
    N = numero de canciones del catalogo, para igualar la escala de promediado
    de la evaluacion real (una lista por consulta).
4.  Computa la ILD de cada lista reutilizando `intra_list_diversity` del modulo
    de evaluacion (distancias identicas: coseno para semantico, euclidea para
    musical), y promedia sobre las N listas.
5.  Lee la ILD observada del sistema desde `etapa5_recommendation.json` y
    reporta el cociente observado/baseline y la separacion en desviaciones
    estandar.
6.  Persiste el resultado en `results/metrics/etapa5_ild_baseline.json`.

Nota: un recomendador aleatorio para la consulta q excluiria q de sus
candidatos; con un catalogo de ~18.000 canciones el efecto sobre la ILD
esperada es despreciable, por lo que las listas se muestrean del catalogo
completo sin ese ajuste.
"""

import json
import logging
import time
from types import SimpleNamespace

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("src.recommendation.baseline_ild")

from src.config import (
    METRICS_DIR,
    NUMPY_SEED,
    RECOMMENDATION_TOP_K,
    UNIFIED_DATASET,
)
from src.recommendation.evaluation import intra_list_diversity

# Numero de listas aleatorias a muestrear. None => usar n_songs (paridad de
# escala con la evaluacion real, que promedia una lista por consulta).
N_RANDOM_LISTS = None

# JSON de metricas de Etapa 5 desde donde se lee la ILD observada del sistema.
RECOMMENDATION_METRICS = METRICS_DIR / "etapa5_recommendation.json"
OUTPUT_METRICS = METRICS_DIR / "etapa5_ild_baseline.json"


def sample_random_lists(n_songs: int, k: int, n_lists: int, seed: int):
    """
    Genera `n_lists` listas de `k` indices distintos muestreados del catalogo.

    Returns
    -------
    list of SimpleNamespace
        Cada elemento expone `.recommended_indices` (np.ndarray [k]), la unica
        propiedad que `intra_list_diversity` consume.
    """
    rng = np.random.RandomState(seed)
    lists = []
    for _ in range(n_lists):
        idx = rng.choice(n_songs, size=k, replace=False)
        lists.append(SimpleNamespace(recommended_indices=idx))
    return lists


def load_observed_ild():
    """Lee la ILD observada del sistema (alpha optimo) desde el JSON de Etapa 5."""
    if not RECOMMENDATION_METRICS.exists():
        logger.warning(
            "No se encontro %s; se omite el contraste con la ILD observada.",
            RECOMMENDATION_METRICS,
        )
        return None
    with open(RECOMMENDATION_METRICS, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    diversity = metrics.get("optimal_evaluation", {}).get("diversity", {})
    if not diversity:
        logger.warning(
            "El JSON de Etapa 5 no contiene 'optimal_evaluation.diversity'."
        )
        return None
    return {
        "ild_semantic": diversity.get("ild_semantic"),
        "ild_musical": diversity.get("ild_musical"),
        "alpha": metrics.get("optimal_evaluation", {}).get("alpha"),
    }


def _compare(observed, base_mean, base_std):
    """Cociente observado/baseline y separacion en desviaciones estandar."""
    if observed is None or base_mean is None:
        return {"ratio_observed_over_baseline": None, "separation_std": None}
    ratio = observed / base_mean if base_mean else None
    separation = (base_mean - observed) / base_std if base_std else None
    return {
        "ratio_observed_over_baseline": round(ratio, 6) if ratio is not None else None,
        "separation_std": round(separation, 4) if separation is not None else None,
    }


def main():
    """Computa el baseline aleatorio de ILD y lo contrasta con el sistema."""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("BASELINE ALEATORIO DE ILD")
    logger.info("=" * 60)

    np.random.seed(NUMPY_SEED)

    # Carga del dataset unificado (mismas fuentes que run_recommendation).
    data = np.load(str(UNIFIED_DATASET), allow_pickle=True)
    semantic_embeddings = data["semantic_embeddings"]
    musical_features = data["musical_features"]

    n_songs = len(semantic_embeddings)
    k = RECOMMENDATION_TOP_K
    n_lists = N_RANDOM_LISTS if N_RANDOM_LISTS is not None else n_songs

    logger.info(
        "Catalogo: %d canciones | K=%d | listas aleatorias=%d | seed=%d",
        n_songs, k, n_lists, NUMPY_SEED,
    )

    # Muestreo de listas aleatorias.
    t0 = time.time()
    random_lists = sample_random_lists(n_songs, k, n_lists, seed=NUMPY_SEED)
    logger.info("Listas muestreadas en %.2fs", time.time() - t0)

    # ILD del baseline con las MISMAS distancias que la evaluacion real.
    base_sem_mean, base_sem_std = intra_list_diversity(
        random_lists, semantic_embeddings, metric="cosine"
    )
    base_mus_mean, base_mus_std = intra_list_diversity(
        random_lists, musical_features, metric="euclidean"
    )

    logger.info(
        "Baseline ILD semantica (coseno):  %.6f +/- %.6f",
        base_sem_mean, base_sem_std,
    )
    logger.info(
        "Baseline ILD musical (euclidea):  %.6f +/- %.6f",
        base_mus_mean, base_mus_std,
    )

    # Contraste con la ILD observada del sistema.
    observed = load_observed_ild()
    obs_sem = observed["ild_semantic"] if observed else None
    obs_mus = observed["ild_musical"] if observed else None

    cmp_sem = _compare(obs_sem, base_sem_mean, base_sem_std)
    cmp_mus = _compare(obs_mus, base_mus_mean, base_mus_std)

    if observed:
        logger.info("-" * 60)
        logger.info(
            "Observado (alpha=%s): ILD_sem=%.6f, ILD_mus=%.6f",
            observed["alpha"], obs_sem, obs_mus,
        )
        logger.info(
            "Semantica: observado/baseline=%.3f, separacion=%.2f sigma",
            cmp_sem["ratio_observed_over_baseline"], cmp_sem["separation_std"],
        )
        logger.info(
            "Musical:   observado/baseline=%.3f, separacion=%.2f sigma",
            cmp_mus["ratio_observed_over_baseline"], cmp_mus["separation_std"],
        )

    elapsed = round(time.time() - start_time, 2)

    result = {
        "config": {
            "n_songs": int(n_songs),
            "top_k": int(k),
            "n_random_lists": int(n_lists),
            "seed": int(NUMPY_SEED),
        },
        "baseline_random": {
            "ild_semantic": round(base_sem_mean, 6),
            "ild_semantic_std": round(base_sem_std, 6),
            "ild_musical": round(base_mus_mean, 6),
            "ild_musical_std": round(base_mus_std, 6),
        },
        "observed_system": {
            "alpha": observed["alpha"] if observed else None,
            "ild_semantic": obs_sem,
            "ild_musical": obs_mus,
        },
        "comparison": {
            "semantic": cmp_sem,
            "musical": cmp_mus,
        },
        "total_elapsed_seconds": elapsed,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Metricas guardadas en %s", OUTPUT_METRICS)
    logger.info("Tiempo total: %.2fs", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
