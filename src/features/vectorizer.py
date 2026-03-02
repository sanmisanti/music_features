"""
Modulo de vectorizacion semantica de letras con multilingual-e5-small.

Provee funciones para:
- Preparar texto de letras (normalizacion minima de artefactos)
- Medir cobertura real de tokens con el tokenizer del modelo
- Generar embeddings de 384 dimensiones via SentenceTransformer

El modelo E5 requiere el prefijo "passage: " para documentos/pasajes.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    BERT_BATCH_SIZE,
    BERT_EMBEDDING_DIM,
    BERT_MODEL_NAME,
    BERT_TARGET_TOKENS,
)

logger = logging.getLogger(__name__)

# Nombre completo del modelo en HuggingFace
E5_MODEL_ID = f"intfloat/{BERT_MODEL_NAME}"

# Prefijo obligatorio de la arquitectura E5 para documentos/pasajes
E5_PASSAGE_PREFIX = "passage: "

# Patron para detectar comillas envolventes (artefacto de scraping Genius)
_TRIPLE_QUOTE_RE = re.compile(r'^"{3}([\s\S]*?)"{3}$')
_SINGLE_QUOTE_RE = re.compile(r'^"([\s\S]*?)"$')
_MULTI_SPACE_RE = re.compile(r" {2,}")


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class TokenCoverageReport:
    """Reporte de cobertura de tokens del tokenizer real vs ventana del modelo."""

    total_songs: int = 0
    token_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    window_size: int = BERT_TARGET_TOKENS
    within_window: int = 0
    coverage_pct: float = 0.0
    mean_tokens: float = 0.0
    median_tokens: float = 0.0
    std_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    p90_tokens: int = 0
    p95_tokens: int = 0
    p99_tokens: int = 0


@dataclass
class VectorizationReport:
    """Reporte del proceso de vectorizacion de letras."""

    total_songs: int = 0
    successful: int = 0
    failed: int = 0
    failure_indices: List[int] = field(default_factory=list)
    embedding_shape: Tuple[int, int] = (0, 0)
    dtype: str = ""
    elapsed_seconds: float = 0.0
    throughput_songs_per_sec: float = 0.0
    model_load_seconds: float = 0.0
    embedding_stats: Dict = field(default_factory=dict)


# =============================================================================
# PREPARACION DE TEXTO
# =============================================================================


def prepare_lyrics_for_encoding(lyrics: pd.Series) -> pd.Series:
    """
    Aplica normalizacion minima a las letras antes de la vectorizacion.

    Dos transformaciones:
    1. Eliminar comillas envolventes (triples o simples), artefacto del
       scraping de Genius que desperdicia tokens sin aportar semantica.
    2. Normalizar espacios multiples consecutivos a espacio simple.

    No se aplica limpieza pesada: los modelos BERT/E5 manejan texto
    desordenado de forma nativa gracias a su preentrenamiento.

    Parameters
    ----------
    lyrics : pd.Series
        Serie con el texto de las letras.

    Returns
    -------
    pd.Series
        Letras normalizadas.
    """
    def _clean_single(text: str) -> str:
        if not isinstance(text, str):
            return text

        # Paso 1: eliminar comillas envolventes
        match = _TRIPLE_QUOTE_RE.match(text)
        if match:
            text = match.group(1)
        else:
            match = _SINGLE_QUOTE_RE.match(text)
            if match:
                text = match.group(1)

        # Paso 2: normalizar espacios multiples
        text = _MULTI_SPACE_RE.sub(" ", text)

        return text.strip()

    result = lyrics.apply(_clean_single)

    n_changed = int((result != lyrics).sum())
    logger.info(
        "Preparacion de letras: %d de %d textos modificados (%.1f%%)",
        n_changed, len(lyrics), 100.0 * n_changed / len(lyrics) if len(lyrics) > 0 else 0.0,
    )

    return result


# =============================================================================
# MEDICION DE COBERTURA DE TOKENS
# =============================================================================


def measure_token_coverage(
    lyrics: pd.Series,
    model_id: str = E5_MODEL_ID,
    window_size: int = BERT_TARGET_TOKENS,
) -> TokenCoverageReport:
    """
    Mide la cobertura real de tokens usando el tokenizer del modelo E5.

    Carga SOLO el tokenizer (no el modelo completo) para eficiencia.
    Tokeniza cada letra CON el prefijo "passage: " ya que asi sera
    procesada durante la vectorizacion.

    Parameters
    ----------
    lyrics : pd.Series
        Serie con el texto de las letras (ya preparadas).
    model_id : str
        Identificador del modelo en HuggingFace.
    window_size : int
        Tamano de la ventana de contexto del modelo en tokens.

    Returns
    -------
    TokenCoverageReport
        Estadisticas de cobertura incluyendo conteos por cancion.
    """
    from transformers import AutoTokenizer

    logger.info("Cargando tokenizer de '%s'...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info("Tokenizer cargado. Vocabulario: %d tokens", tokenizer.vocab_size)

    # Tokenizar cada letra con el prefijo E5
    token_counts = np.zeros(len(lyrics), dtype=np.int32)

    for i, text in enumerate(lyrics):
        prefixed = E5_PASSAGE_PREFIX + str(text)
        encoded = tokenizer.encode(prefixed, add_special_tokens=True)
        token_counts[i] = len(encoded)

        if (i + 1) % 5000 == 0:
            logger.info("  Tokenizadas %d / %d letras...", i + 1, len(lyrics))

    logger.info("Tokenizacion completada: %d letras procesadas", len(lyrics))

    # Calcular estadisticas
    within_window = int((token_counts <= window_size).sum())
    coverage_pct = round(100.0 * within_window / len(lyrics), 2) if len(lyrics) > 0 else 0.0

    report = TokenCoverageReport(
        total_songs=len(lyrics),
        token_counts=token_counts,
        window_size=window_size,
        within_window=within_window,
        coverage_pct=coverage_pct,
        mean_tokens=round(float(np.mean(token_counts)), 2),
        median_tokens=round(float(np.median(token_counts)), 2),
        std_tokens=round(float(np.std(token_counts)), 2),
        min_tokens=int(np.min(token_counts)),
        max_tokens=int(np.max(token_counts)),
        p90_tokens=int(np.percentile(token_counts, 90)),
        p95_tokens=int(np.percentile(token_counts, 95)),
        p99_tokens=int(np.percentile(token_counts, 99)),
    )

    logger.info(
        "Cobertura de tokens: %d / %d canciones dentro de ventana %d (%.2f%%)",
        report.within_window, report.total_songs, report.window_size, report.coverage_pct,
    )
    logger.info(
        "Distribucion: media=%.1f, mediana=%.1f, DE=%.1f, min=%d, max=%d",
        report.mean_tokens, report.median_tokens, report.std_tokens,
        report.min_tokens, report.max_tokens,
    )
    logger.info(
        "Percentiles: p90=%d, p95=%d, p99=%d",
        report.p90_tokens, report.p95_tokens, report.p99_tokens,
    )

    return report


# =============================================================================
# VECTORIZACION
# =============================================================================


def vectorize_lyrics(
    lyrics: pd.Series,
    model_id: str = E5_MODEL_ID,
    batch_size: int = BERT_BATCH_SIZE,
    target_dim: int = BERT_EMBEDDING_DIM,
    max_seq_length: int = BERT_TARGET_TOKENS,
) -> Tuple[np.ndarray, VectorizationReport]:
    """
    Genera embeddings semanticos de 384 dimensiones a partir de las letras.

    Carga el modelo SentenceTransformer completo, configura la ventana de
    contexto, y codifica todas las letras con el prefijo E5 "passage: ".

    Estrategia de errores: intento bulk primero; si falla, fallback
    lote por lote con vectores cero para letras individuales que fallen.

    Parameters
    ----------
    lyrics : pd.Series
        Serie con el texto de las letras (ya preparadas).
    model_id : str
        Identificador del modelo en HuggingFace.
    batch_size : int
        Tamano del lote para encode().
    target_dim : int
        Dimensionalidad esperada del embedding (384 para E5-small).
    max_seq_length : int
        Ventana de contexto maxima del modelo en tokens.

    Returns
    -------
    Tuple[np.ndarray, VectorizationReport]
        (embeddings float32 [N, 384] normalizados L2, reporte).
    """
    from sentence_transformers import SentenceTransformer

    report = VectorizationReport(total_songs=len(lyrics))

    # Cargar modelo
    logger.info("Cargando modelo SentenceTransformer '%s'...", model_id)
    model_start = time.time()
    model = SentenceTransformer(model_id)
    model.max_seq_length = max_seq_length
    report.model_load_seconds = round(time.time() - model_start, 2)
    logger.info(
        "Modelo cargado en %.1f s. max_seq_length=%d",
        report.model_load_seconds, model.max_seq_length,
    )

    # Preparar textos con prefijo E5
    texts = [E5_PASSAGE_PREFIX + str(text) for text in lyrics]

    # Intento bulk
    logger.info(
        "Vectorizando %d letras (batch_size=%d, normalize=True)...",
        len(texts), batch_size,
    )
    encode_start = time.time()

    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        report.successful = len(texts)
        report.failed = 0

    except Exception as e:
        logger.warning(
            "Encoding bulk fallo: %s. Ejecutando fallback lote por lote...", str(e)
        )
        embeddings = np.zeros((len(texts), target_dim), dtype=np.float32)
        report.failure_indices = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]

            try:
                batch_emb = model.encode(
                    batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                embeddings[start:end] = batch_emb
            except Exception:
                # Fallback individual dentro del lote
                for j, single_text in enumerate(batch):
                    idx = start + j
                    try:
                        single_emb = model.encode(
                            [single_text],
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )
                        embeddings[idx] = single_emb[0]
                    except Exception:
                        report.failure_indices.append(idx)
                        logger.warning("Fallo en cancion indice %d", idx)

            if (end) % 5000 == 0 or end == len(texts):
                logger.info("  Procesados %d / %d ...", end, len(texts))

        report.successful = len(texts) - len(report.failure_indices)
        report.failed = len(report.failure_indices)

    elapsed = time.time() - encode_start
    report.elapsed_seconds = round(elapsed, 2)
    report.throughput_songs_per_sec = round(len(texts) / elapsed, 2) if elapsed > 0 else 0.0
    report.embedding_shape = embeddings.shape
    report.dtype = str(embeddings.dtype)

    # Asegurar tipo float32
    embeddings = embeddings.astype(np.float32)

    # Estadisticas del espacio de embeddings
    report.embedding_stats = _compute_embedding_statistics(embeddings)

    logger.info(
        "Vectorizacion completada: %d exitosas, %d fallidas en %.1f s (%.1f songs/s)",
        report.successful, report.failed, report.elapsed_seconds,
        report.throughput_songs_per_sec,
    )

    return embeddings, report


# =============================================================================
# ESTADISTICAS DE EMBEDDINGS
# =============================================================================


def _compute_embedding_statistics(embeddings: np.ndarray) -> Dict:
    """
    Calcula estadisticas descriptivas del espacio de embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Matriz de embeddings [N, D] float32.

    Returns
    -------
    Dict
        Norma L2 media/std, varianza por dimension, porcentaje de vectores cero,
        rango de valores.
    """
    # Norma L2 de cada vector
    norms = np.linalg.norm(embeddings, axis=1)

    # Vectores cero (norma < epsilon)
    zero_mask = norms < 1e-8
    n_zero = int(zero_mask.sum())

    # Varianza por dimension (para detectar dimensiones muertas)
    var_per_dim = np.var(embeddings, axis=0)
    dead_dims = int((var_per_dim < 1e-10).sum())

    stats = {
        "l2_norm_mean": round(float(np.mean(norms)), 6),
        "l2_norm_std": round(float(np.std(norms)), 6),
        "l2_norm_min": round(float(np.min(norms)), 6),
        "l2_norm_max": round(float(np.max(norms)), 6),
        "zero_vectors": n_zero,
        "zero_vectors_pct": round(100.0 * n_zero / len(embeddings), 4) if len(embeddings) > 0 else 0.0,
        "value_min": round(float(np.min(embeddings)), 6),
        "value_max": round(float(np.max(embeddings)), 6),
        "value_mean": round(float(np.mean(embeddings)), 6),
        "variance_per_dim_mean": round(float(np.mean(var_per_dim)), 6),
        "variance_per_dim_min": round(float(np.min(var_per_dim)), 6),
        "variance_per_dim_max": round(float(np.max(var_per_dim)), 6),
        "dead_dimensions": dead_dims,
    }

    logger.info(
        "Estadisticas de embeddings: norma L2 media=%.4f (std=%.4f), "
        "vectores cero=%d, dimensiones muertas=%d",
        stats["l2_norm_mean"], stats["l2_norm_std"],
        stats["zero_vectors"], stats["dead_dimensions"],
    )

    return stats
