"""
Configuración de rutas de datos y archivos para el módulo de clustering de letras.

Centraliza todas las rutas utilizadas por el sistema para facilitar 
mantenimiento y configuración.
"""

import os
from pathlib import Path

# === RUTAS BASE ===
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
CLUSTERING_ROOT = PROJECT_ROOT / "clustering"
LYRICS_MODULE_ROOT = CLUSTERING_ROOT / "algorithms" / "lyrics"

# === DATOS PRINCIPALES ===
DATA_PATHS = {
    # Dataset principal optimizado
    "main_dataset": PROJECT_ROOT / "data" / "3_selected/picked_data_optimal.csv",
    
    # Dataset fuente con letras
    "source_dataset": PROJECT_ROOT / "data" / "with_lyrics" / "spotify_songs_fixed.csv",
    
    # Datasets de prueba
    "test_subset": LYRICS_MODULE_ROOT / "data" / "sample_data" / "test_subset_1000.csv",
    "multilingual_test": LYRICS_MODULE_ROOT / "data" / "sample_data" / "multilingual_test.csv",
    
    # Stopwords por idioma
    "stopwords_dir": LYRICS_MODULE_ROOT / "data" / "stopwords",
    "stopwords_english": LYRICS_MODULE_ROOT / "data" / "stopwords" / "english.txt",
    "stopwords_spanish": LYRICS_MODULE_ROOT / "data" / "stopwords" / "spanish.txt",
    "stopwords_german": LYRICS_MODULE_ROOT / "data" / "stopwords" / "german.txt",
    "stopwords_portuguese": LYRICS_MODULE_ROOT / "data" / "stopwords" / "portuguese.txt"
}

# === CACHE Y MODELOS ===
CACHE_PATHS = {
    # Cache BERT embeddings
    "bert_cache_dir": LYRICS_MODULE_ROOT / "models" / "bert_cache",
    "embeddings_cache": LYRICS_MODULE_ROOT / "models" / "bert_cache" / "embeddings.cache",
    "similarity_cache": LYRICS_MODULE_ROOT / "models" / "bert_cache" / "similarities.cache",
    
    # Modelos de clustering
    "clustering_models_dir": LYRICS_MODULE_ROOT / "models" / "clustering_results",
    "best_model": LYRICS_MODULE_ROOT / "models" / "clustering_results" / "best_model.pkl",
    
    # Resultados de evaluación
    "evaluations_dir": LYRICS_MODULE_ROOT / "models" / "evaluations",
    "metrics_history": LYRICS_MODULE_ROOT / "models" / "evaluations" / "metrics_history.json"
}

# === OUTPUTS Y RESULTADOS ===
OUTPUT_PATHS = {
    # Resultados de clustering
    "clustering_results": PROJECT_ROOT / "outputs" / "lyrics_clustering",
    "cluster_assignments": PROJECT_ROOT / "outputs" / "lyrics_clustering" / "cluster_assignments.csv",
    "cluster_analysis": PROJECT_ROOT / "outputs" / "lyrics_clustering" / "cluster_analysis.json",
    
    # Visualizaciones
    "visualizations": PROJECT_ROOT / "outputs" / "lyrics_clustering" / "visualizations",
    "umap_plot": PROJECT_ROOT / "outputs" / "lyrics_clustering" / "visualizations" / "umap_clusters.png",
    "dendrogram": PROJECT_ROOT / "outputs" / "lyrics_clustering" / "visualizations" / "dendrogram.png",
    
    # Reports
    "reports": PROJECT_ROOT / "outputs" / "lyrics_clustering" / "reports",
    "evaluation_report": PROJECT_ROOT / "outputs" / "lyrics_clustering" / "reports" / "evaluation_report.html",
    "performance_report": PROJECT_ROOT / "outputs" / "lyrics_clustering" / "reports" / "performance_report.json"
}

# === LOGS ===
LOG_PATHS = {
    "main_log": LYRICS_MODULE_ROOT / "logs" / "lyrics_clustering.log",
    "error_log": LYRICS_MODULE_ROOT / "logs" / "errors.log",
    "performance_log": LYRICS_MODULE_ROOT / "logs" / "performance.log"
}

# === CONFIGURACIÓN ARCHIVO DATASET ===
DATASET_CONFIG = {
    "separator": "^",           # Separador principal dataset
    "decimal": ".",            # Separador decimal
    "encoding": "utf-8",       # Encoding archivo
    "lyrics_column": "lyrics", # Nombre columna letras
    "language_column": "language", # Nombre columna idioma
    "id_column": "track_id"    # Columna identificador único
}

def ensure_directories_exist():
    """
    Crea todos los directorios necesarios si no existen.
    """
    dirs_to_create = [
        LYRICS_MODULE_ROOT / "data" / "sample_data",
        LYRICS_MODULE_ROOT / "data" / "stopwords",
        LYRICS_MODULE_ROOT / "models" / "bert_cache",
        LYRICS_MODULE_ROOT / "models" / "clustering_results",
        LYRICS_MODULE_ROOT / "models" / "evaluations",
        LYRICS_MODULE_ROOT / "logs",
        PROJECT_ROOT / "outputs" / "lyrics_clustering" / "visualizations",
        PROJECT_ROOT / "outputs" / "lyrics_clustering" / "reports"
    ]
    
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)

def get_dataset_path(dataset_name: str = "main") -> Path:
    """
    Obtiene la ruta del dataset especificado.
    
    Args:
        dataset_name: "main", "source", "test", o "multilingual"
        
    Returns:
        Path al dataset solicitado
    """
    dataset_mapping = {
        "main": DATA_PATHS["main_dataset"],
        "source": DATA_PATHS["source_dataset"],
        "test": DATA_PATHS["test_subset"],
        "multilingual": DATA_PATHS["multilingual_test"]
    }
    
    if dataset_name not in dataset_mapping:
        raise ValueError(f"Dataset '{dataset_name}' no reconocido. Opciones: {list(dataset_mapping.keys())}")
    
    return dataset_mapping[dataset_name]

def get_cache_path(cache_type: str) -> Path:
    """
    Obtiene la ruta del cache especificado.
    
    Args:
        cache_type: "bert", "similarity", "models", "evaluations"
        
    Returns:
        Path al directorio de cache solicitado
    """
    cache_mapping = {
        "bert": CACHE_PATHS["bert_cache_dir"],
        "similarity": CACHE_PATHS["similarity_cache"],
        "models": CACHE_PATHS["clustering_models_dir"],
        "evaluations": CACHE_PATHS["evaluations_dir"]
    }
    
    if cache_type not in cache_mapping:
        raise ValueError(f"Cache type '{cache_type}' no reconocido. Opciones: {list(cache_mapping.keys())}")
    
    return cache_mapping[cache_type]

def get_models_path() -> Path:
    """
    Obtiene la ruta base de los modelos.
    
    Returns:
        Path al directorio de modelos
    """
    models_path = LYRICS_MODULE_ROOT / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    return models_path

def get_stopwords_path(language: str = None) -> Path:
    """
    Obtiene la ruta de stopwords para el idioma especificado.
    
    Args:
        language: "english", "spanish", "german", "portuguese" o None para directorio
        
    Returns:
        Path al archivo de stopwords o directorio
    """
    if language is None:
        return DATA_PATHS["stopwords_dir"]
    
    language_mapping = {
        "english": DATA_PATHS["stopwords_english"],
        "spanish": DATA_PATHS["stopwords_spanish"], 
        "german": DATA_PATHS["stopwords_german"],
        "portuguese": DATA_PATHS["stopwords_portuguese"]
    }
    
    if language not in language_mapping:
        raise ValueError(f"Idioma '{language}' no reconocido. Opciones: {list(language_mapping.keys())}")
    
    return language_mapping[language]