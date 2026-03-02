# src/features/ - Ingeniería de features: musicales (Spotify 13D), semánticas (BERT 384D)

from src.features.normalizer import (
    NormalizationReport,
    circular_encode_key,
    extract_musical_features,
    normalize_features,
)
from src.features.tables import (
    generate_normalization_table,
    generate_token_coverage_table,
    generate_unification_table,
)
from src.features.unifier import (
    UnificationReport,
    build_unified_dataset,
    load_feature_components,
    load_metadata,
)
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
    "circular_encode_key",
    "extract_musical_features",
    "normalize_features",
    "generate_normalization_table",
    "UnificationReport",
    "load_feature_components",
    "load_metadata",
    "build_unified_dataset",
    "generate_unification_table",
]
