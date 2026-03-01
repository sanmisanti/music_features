"""
Configuración centralizada del proyecto.

Todos los paths, seeds, y parámetros globales se definen aquí.
Ningún script debe definir estos valores localmente.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent

# Datos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset fuente
SOURCE_DATASET = DATA_DIR / "2_with_lyrics" / "spotify_songs_fixed.csv"
SOURCE_SEPARATOR = "@@"

# Dataset seleccionado (post-preprocesamiento)
SELECTED_DIR = DATA_DIR / "3_selected"
SELECTED_DATASET = SELECTED_DIR / "selected_dataset.csv"

# Resultados
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"

# Tesis (figuras y tablas generadas para LaTeX)
THESIS_DIR = PROJECT_ROOT / "thesis"
THESIS_FIGURES_DIR = THESIS_DIR / "figures"
THESIS_TABLES_DIR = THESIS_DIR / "tables"

# =============================================================================
# REPRODUCIBILIDAD
# =============================================================================

RANDOM_SEED = 42
NUMPY_SEED = 42
BOOTSTRAP_SEED = 123

# =============================================================================
# PARÁMETROS DEL DATASET
# =============================================================================

# Features musicales de Spotify utilizadas
MUSICAL_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# =============================================================================
# PARÁMETROS DE VECTORIZACIÓN BERT
# =============================================================================

BERT_MODEL_NAME = "multilingual-e5-small"
BERT_EMBEDDING_DIM = 384
BERT_BATCH_SIZE = 64
LYRICS_MIN_LENGTH = 50
LYRICS_MAX_LENGTH = 5000
BERT_TARGET_TOKENS = 512

# =============================================================================
# PARÁMETROS DE CLUSTERING
# =============================================================================

HOPKINS_ITERATIONS = 30
HOPKINS_SAMPLE_SIZE = 100

CLUSTERING_K_RANGE = [5, 6, 7, 8]
DBSCAN_EPS_RANGE = [0.1, 0.15, 0.2, 0.25, 0.3]

# Pesos de la función objetivo multi-criterio
OBJECTIVE_WEIGHTS = {
    "silhouette": 0.30,
    "balance": 0.30,
    "interpretability": 0.20,
    "cross_modal": 0.10,
    "granularity": 0.10,
}

# =============================================================================
# PARÁMETROS DE RECOMENDACIÓN
# =============================================================================

RECOMMENDATION_TOP_K = 10
RECOMMENDATION_CANDIDATES = 20

# =============================================================================
# PARÁMETROS EDA
# =============================================================================

# Columnas de metadatos (no numéricas / no features)
METADATA_COLUMNS = [
    "track_id",
    "track_name",
    "track_artist",
    "track_popularity",
    "track_album_id",
    "track_album_name",
    "track_album_release_date",
    "playlist_name",
    "playlist_id",
    "playlist_genre",
    "playlist_subgenre",
    "lyrics",
    "language",
]

# Géneros del dataset (playlists de Spotify)
PLAYLIST_GENRES = ["edm", "latin", "pop", "r&b", "rap", "rock"]

# Umbrales de calidad de letras (en palabras)
LYRICS_MIN_WORDS = 10
LYRICS_MAX_WORDS = 2000

# Parámetros de visualización
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
