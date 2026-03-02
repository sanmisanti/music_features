# src/features/ - Ingeniería de features: musicales (Spotify 12D), semánticas (BERT 384D)

from src.features.tables import generate_token_coverage_table
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
]
