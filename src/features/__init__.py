# src/features/ - Ingeniería de features: musicales (Spotify 12D), semánticas (BERT 384D)

from src.features.normalizer import (
    NormalizationReport,
    extract_musical_features,
    normalize_features,
)
from src.features.tables import generate_normalization_table, generate_token_coverage_table
from src.features.vectorizer import (
    TokenCoverageReport,
    VectorizationReport,
    measure_token_coverage,
    prepare_lyrics_for_encoding,
    vectorize_lyrics,
)

__all__ = [
    "TokenCoverageReport",
    "VectorizationReport",
    "prepare_lyrics_for_encoding",
    "measure_token_coverage",
    "vectorize_lyrics",
    "generate_token_coverage_table",
    "NormalizationReport",
    "extract_musical_features",
    "normalize_features",
    "generate_normalization_table",
]
