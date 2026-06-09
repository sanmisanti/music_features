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

# Datos vectorizados
VECTORIZED_DIR = DATA_DIR / "4_vectorized"
VECTORIZED_EMBEDDINGS = VECTORIZED_DIR / "embeddings.npy"
VECTORIZED_TRACK_IDS = VECTORIZED_DIR / "track_ids.npy"
VECTORIZED_TOKEN_COUNTS = VECTORIZED_DIR / "token_counts.npy"
VECTORIZED_MUSICAL_FEATURES = VECTORIZED_DIR / "musical_features.npy"
VECTORIZED_FEATURE_NAMES = VECTORIZED_DIR / "feature_names.npy"

# Dataset unificado
UNIFIED_DIR = DATA_DIR / "5_unified"
UNIFIED_DATASET = UNIFIED_DIR / "unified_dataset.npz"

# Dataset clusterizado (Etapa 4)
CLUSTERED_DIR = DATA_DIR / "6_clustered"
CLUSTERED_DATASET = CLUSTERED_DIR / "clustered_dataset.npz"

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

# Codificacion circular de key (tonalidad)
KEY_PERIOD = 12  # periodo circular para codificacion sin/cos de key

# =============================================================================
# PARÁMETROS DE VECTORIZACIÓN BERT
# =============================================================================

BERT_MODEL_NAME = "multilingual-e5-small"
BERT_EMBEDDING_DIM = 384
BERT_BATCH_SIZE = 64
LYRICS_MIN_LENGTH = 50
LYRICS_MAX_LENGTH = 5000
BERT_TARGET_TOKENS = 512
BERT_CHUNK_TOKENS = 450
BERT_CHUNK_OVERLAP = 50

# =============================================================================
# PARÁMETROS DE CLUSTERING
# =============================================================================

# Hopkins statistic
HOPKINS_ITERATIONS = 30
HOPKINS_SAMPLE_SIZE = 100
HOPKINS_THRESHOLD = 0.7

# Algoritmos de clustering
CLUSTERING_K_RANGE = [5, 6, 7, 8]
HDBSCAN_MIN_CLUSTER_SIZES = [50, 100, 200, 300, 500]

# UMAP
UMAP_PREPROCESSING_N_COMPONENTS = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Umbrales de calidad de clustering
MIN_CLUSTER_PCT = 0.01  # rechazar clusters < 1% del dataset
MAX_NOISE_PCT = 0.50  # rechazar configs que dejan > 50% de puntos como ruido
#                       (cobertura insuficiente: el Silhouette se infla al
#                        medirse solo sobre los puntos mas densos)
PURIFICATION_MAX_REMOVAL_PCT = 0.15  # max 15% de datos eliminados
PURIFICATION_SIGMA_THRESHOLD = 2.0  # umbral para distancia al centroide

# Referencia v1 (no usar para seleccion; DBSCAN_EPS_RANGE y OBJECTIVE_WEIGHTS
# se conservan como referencia historica de la primera ejecucion)
DBSCAN_EPS_RANGE = [0.1, 0.15, 0.2, 0.25, 0.3]
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
RECOMMENDATION_K_VALUES = [5, 10, 20]  # valores de K para Precision@K

# Grid search del peso de fusion (alpha = peso semantico)
RECOMMENDATION_ALPHA_RANGE = [round(a * 0.05, 2) for a in range(21)]  # 0.00 a 1.00, paso 0.05

# Kernel gaussiano para conversion distancia euclidea -> similitud
RECOMMENDATION_SIGMA_SAMPLE_SIZE = 5000  # muestra para mediana de distancias pairwise

# Paths
RECOMMENDATION_DIR = DATA_DIR / "7_recommendation"

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
