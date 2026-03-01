# src/data/ - Pipeline de datos: carga, limpieza, selección, unificación

from src.data.loader import LoadReport, load_source_dataset
from src.data.eda import (
    FeatureProfile,
    analyze_genre_balance,
    analyze_lyrics_quality,
    compute_correlation_matrix,
    compute_data_loss_budget,
    compute_dataset_summary,
    compute_feature_profiles,
)
from src.data.preprocessor import (
    preprocess_dataset,
    save_selected_dataset,
)

__all__ = [
    "load_source_dataset",
    "LoadReport",
    "FeatureProfile",
    "compute_feature_profiles",
    "compute_dataset_summary",
    "analyze_lyrics_quality",
    "compute_correlation_matrix",
    "analyze_genre_balance",
    "compute_data_loss_budget",
    "preprocess_dataset",
    "save_selected_dataset",
]
