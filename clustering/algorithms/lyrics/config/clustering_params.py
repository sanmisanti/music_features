"""
Parámetros optimizados clustering semántico letras musicales.
Basado en literatura académica MIR + características dataset específico.
"""

from typing import Dict, Any, Tuple, List

# === ALGORITMOS CLUSTERING ESPECIALIZADOS ===
CLUSTERING_ALGORITHMS = {
    
    # K-Means cosine - Óptimo vectores BERT normalizados
    "kmeans_cosine": {
        "algorithm": "KMeans",
        "init": "k-means++",
        "n_init": 20,                    # Múltiples inicializaciones robustez
        "max_iter": 500,                 # Convergencia garantizada
        "tol": 1e-6,                     # Precisión alta convergencia
        "random_state": 42,
        "n_jobs": -1,                    # Usar todos CPU cores
        "performance": "fast",
        "quality": "high",
        "interpretability": "medium"
    },
    
    # Hierarchical - Mejor análisis jerárquico temas
    "hierarchical_ward": {
        "algorithm": "AgglomerativeClustering",
        "linkage": "ward",
        "metric": "euclidean",           # Compatible con Ward linkage
        "compute_full_tree": True,       # Para dendrogram analysis
        "performance": "medium",
        "quality": "high",
        "interpretability": "very_high"
    },
    
    # HDBSCAN - Detección automática clusters densos
    "hdbscan_density": {
        "algorithm": "HDBSCAN",
        "min_cluster_size": 50,          # Mín 50 canciones/cluster
        "min_samples": 25,               # Densidad mínima
        "metric": "cosine",              # Métrica optimizada texto
        "cluster_selection_method": "eom", # Excess of Mass
        "cluster_selection_epsilon": 0.1,
        "alpha": 1.0,
        "performance": "slow",
        "quality": "very_high",
        "handles_outliers": True
    }
}

# === OPTIMIZACIÓN K ===
K_OPTIMIZATION_CONFIG = {
    "k_range": (3, 15),                  # Rango evaluación K
    "methods": ["elbow", "silhouette", "gap_statistic", "semantic_coherence"],
    "default_k": 8,                      # Fallback si métodos fallan
    "hopkins_threshold": 0.75,           # Umbral clustering readiness
    "silhouette_threshold": 0.3,         # Umbral mínimo silhouette
    "coherence_threshold": 0.6,          # Umbral mínimo coherencia
    "max_iterations": 50,                # Máx iteraciones optimización
    "cross_validation_folds": 5          # K-fold para robustez
}

# === PREPROCESSING TEXTO MUSICAL ===
TEXT_PREPROCESSING_CONFIG = {
    "basic_cleaning": {
        "remove_urls": True,
        "remove_emails": True,
        "remove_numbers": False,         # Pueden ser semánticamente relevantes
        "lowercase": True,
        "strip_whitespace": True,
        "normalize_unicode": True
    },
    
    "music_specific": {
        "remove_common_intros": True,    # "Verse 1:", "Chorus:", etc.
        "handle_repetitions": "smart",   # Manejo inteligente repeticiones
        "preserve_emotional_words": True, # Preservar palabras emocionales
        "min_length_chars": 50,          # Mínimo caracteres válidos
        "max_length_chars": 5000,        # Máximo (truncar después)
        "target_length_tokens": 256      # Target para BERT optimization
    },
    
    "stopwords": {
        "use_stopwords": True,
        "custom_music_stopwords": [
            # Interjecciones musicales comunes
            "yeah", "oh", "ah", "eh", "uh", "mm", "hmm",
            "la", "da", "na", "ba", "sha", "doo", "woo",
            "hey", "yo", "whoa", "ooh", "aah",
            # Estructuras repetitivas
            "chorus", "verse", "bridge", "outro", "intro"
        ],
        "language_specific": {
            "en": ["gonna", "wanna", "gotta"],
            "es": ["ay", "eh", "uy"],
            "de": ["ja", "nein", "ach"],
            "pt": ["né", "ai", "ó"]
        }
    },
    
    "quality_validation": {
        "min_unique_words": 10,          # Mínimo palabras únicas
        "max_repetition_ratio": 0.7,    # Máx 70% repetición
        "min_text_diversity": 0.3,      # TTR mínimo
        "language_confidence": 0.8       # Confianza detección idioma
    }
}

# === VALIDACIÓN CROSS-LINGUAL ===
CROSS_LINGUAL_CONFIG = {
    "validation_enabled": True,
    "language_pairs": [
        ("en", "es"), ("en", "de"), ("en", "pt"), 
        ("es", "pt"), ("es", "de"), ("de", "pt")
    ],
    "similarity_thresholds": {
        "high": 0.85,                    # Muy similar cross-idiomas
        "medium": 0.65,                  # Moderadamente similar
        "low": 0.45                      # Baja pero válida similaridad
    },
    "sample_size_per_language": 100,     # Muestras para validación
    "consistency_metrics": [
        "cluster_assignment_agreement",
        "semantic_distance_correlation", 
        "topic_overlap_coefficient",
        "translation_equivalence_score"
    ],
    "min_consistency_score": 0.75        # Umbral mínimo consistencia
}

# === PERFORMANCE OPTIMIZATION ===
PERFORMANCE_CONFIG = {
    "cpu_optimization": {
        "use_all_cores": True,
        "batch_size_strategy": "adaptive", # Adaptativo según memoria
        "memory_limit_gb": 4.0,          # Límite memoria proceso
        "gc_frequency": "per_batch",     # Garbage collection frecuencia
        "progress_reporting": True
    },
    
    "memory_management": {
        "cache_embeddings": True,
        "max_cache_size_gb": 3.0,
        "cache_compression": True,
        "memory_monitoring": True,
        "swap_threshold": 0.9            # Swap a disco al 90% memoria
    },
    
    "error_handling": {
        "max_retries": 3,
        "retry_delay": 1.0,              # Segundos entre reintentos
        "fallback_batch_size": 16,       # Batch size si falla
        "continue_on_error": True,       # Continuar si errores menores
        "detailed_logging": True
    }
}

def get_algorithm_config(algorithm_name: str) -> Dict[str, Any]:
    """
    Obtiene configuración para algoritmo clustering específico.
    
    Args:
        algorithm_name: Nombre del algoritmo clustering
        
    Returns:
        Dict con configuración del algoritmo
        
    Raises:
        ValueError: Si algoritmo no soportado
    """
    if algorithm_name not in CLUSTERING_ALGORITHMS:
        available = list(CLUSTERING_ALGORITHMS.keys())
        raise ValueError(f"Algoritmo '{algorithm_name}' no soportado. Disponibles: {available}")
    
    return CLUSTERING_ALGORITHMS[algorithm_name].copy()

def get_optimal_k_range(dataset_size: int) -> Tuple[int, int]:
    """
    Determina rango K óptimo basado en tamaño dataset.
    
    Args:
        dataset_size: Número de canciones en dataset
        
    Returns:
        Tuple (k_min, k_max) para optimización
    """
    if dataset_size < 1000:
        return (2, 8)
    elif dataset_size < 5000:
        return (3, 12)
    elif dataset_size < 15000:
        return (5, 15)
    else:
        return (8, 20)

def get_preprocessing_config(language: str = None) -> Dict[str, Any]:
    """
    Obtiene configuración preprocessing para idioma específico.
    
    Args:
        language: Código idioma (en, es, de, pt). Si None, usa config universal
        
    Returns:
        Dict con configuración preprocessing optimizada
    """
    config = TEXT_PREPROCESSING_CONFIG.copy()
    
    if language and language in config["stopwords"]["language_specific"]:
        # Añadir stopwords específicas del idioma
        custom_stopwords = config["stopwords"]["custom_music_stopwords"]
        language_stopwords = config["stopwords"]["language_specific"][language]
        config["stopwords"]["custom_music_stopwords"] = custom_stopwords + language_stopwords
    
    return config

def validate_clustering_config(config: Dict[str, Any]) -> bool:
    """
    Valida configuración clustering para consistencia.
    
    Args:
        config: Configuración clustering a validar
        
    Returns:
        True si configuración válida, False otherwise
    """
    required_keys = ["algorithm", "random_state"] if "random_state" in config else ["algorithm"]
    
    for key in required_keys:
        if key not in config:
            return False
    
    # Validaciones específicas por algoritmo
    algorithm = config["algorithm"]
    
    if algorithm == "KMeans":
        if "n_clusters" not in config and "k_range" not in config:
            return False
    elif algorithm == "HDBSCAN":
        if "min_cluster_size" not in config:
            return False
    
    return True

# === CONFIGURACIÓN DEFAULT RECOMENDADA ===
DEFAULT_CONFIG = {
    "algorithm": "kmeans_cosine",
    "k_optimization": True,
    "cross_lingual_validation": True,
    "performance_optimization": True,
    "detailed_logging": True
}