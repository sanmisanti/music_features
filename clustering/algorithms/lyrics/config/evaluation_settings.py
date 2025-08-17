"""
Configuración métricas evaluación especializadas
para clustering semántico letras musicales multilingües.
"""

from typing import Dict, Any, List, Tuple

# === MÉTRICAS COHERENCIA TEMÁTICA ===
TOPIC_COHERENCE_CONFIG = {
    "coherence_measures": ["c_v", "c_npmi", "u_mass", "c_uci"],
    "gensim_params": {
        "num_topics": "auto",            # Auto-detect basado en clusters
        "window_size": 110,              # Ventana contexto palabras
        "topn": 20,                      # Top 20 palabras por tópico
        "dictionary_filter": {
            "no_below": 5,               # Palabra en ≥5 documentos
            "no_above": 0.7,             # Palabra en ≤70% documentos
            "keep_n": 2000               # Top 2000 palabras vocabulario
        },
        "alpha": "auto",                 # Concentración tópicos
        "eta": "auto",                   # Concentración palabras
        "passes": 20,                    # Iteraciones entrenamiento
        "iterations": 400,               # Iteraciones por documento
        "random_state": 42
    },
    "minimum_coherence": 0.4,            # Umbral aceptable coherencia
    "target_coherence": 0.6,             # Objetivo calidad alta
    "coherence_weights": {               # Pesos combinación métricas
        "c_v": 0.4,                      # Más interpretable humanamente
        "c_npmi": 0.3,                   # Normalizada robusta
        "u_mass": 0.2,                   # Rápida computación
        "c_uci": 0.1                     # Confirmación adicional
    }
}

# === VALIDACIÓN CROSS-LINGUAL ===
CROSS_LINGUAL_EVAL_CONFIG = {
    "language_pairs": [
        ("en", "es"), ("en", "de"), ("en", "pt"), 
        ("es", "pt"), ("es", "de"), ("de", "pt")
    ],
    "similarity_thresholds": {
        "excellent": 0.90,               # Excelente similaridad
        "very_good": 0.80,               # Muy buena
        "good": 0.70,                    # Buena
        "acceptable": 0.60,              # Aceptable
        "poor": 0.50                     # Pobre pero válida
    },
    "sample_sizes": {
        "quick_test": 50,                # Test rápido
        "standard_validation": 100,      # Validación estándar
        "comprehensive_test": 200        # Test comprensivo
    },
    "consistency_metrics": [
        "cluster_assignment_agreement",   # Acuerdo asignación clusters
        "semantic_distance_correlation",  # Correlación distancias
        "topic_overlap_coefficient",      # Overlap temas
        "translation_equivalence_score",  # Equivalencia traducción
        "cultural_coherence_index"        # Coherencia cultural
    ],
    "validation_strategies": {
        "random_sampling": True,          # Muestreo aleatorio
        "stratified_sampling": True,      # Muestreo estratificado
        "balanced_languages": True,       # Balance idiomas
        "genre_diversity": True           # Diversidad géneros
    }
}

# === DIVERSIDAD SEMÁNTICA ===
SEMANTIC_DIVERSITY_CONFIG = {
    "intra_cluster": {
        "method": "average_pairwise_distance",
        "metric": "cosine",
        "target_range": (0.2, 0.6),      # Rango óptimo diversidad
        "sample_size": 100,               # Muestras para cálculo
        "confidence_level": 0.95          # Nivel confianza estadística
    },
    
    "inter_cluster": {
        "method": "centroid_distances",
        "minimum_separation": 0.3,        # Separación mínima centroides
        "target_separation": 0.5,         # Separación target
        "outlier_threshold": 2.0,         # Threshold detección outliers
        "validation_method": "silhouette" # Método validación separación
    },
    
    "global_diversity": {
        "entropy_threshold": 0.8,         # Entropía mínima distribución
        "coverage_threshold": 0.9,        # Cobertura espacio semántico
        "balance_coefficient": 0.7,       # Balance tamaños clusters
        "redundancy_threshold": 0.1       # Máxima redundancia permitida
    }
}

# === INTERPRETABILIDAD CLUSTERS ===
INTERPRETABILITY_CONFIG = {
    "cluster_labeling": {
        "method": "hybrid",               # TF-IDF + LDA + embeddings
        "keywords_per_cluster": 10,       # Keywords por cluster
        "extraction_methods": {
            "tfidf": {
                "max_features": 1000,
                "ngram_range": (1, 2),
                "min_df": 2,
                "max_df": 0.8
            },
            "lda": {
                "n_components": "auto",   # Auto-detect
                "learning_method": "batch",
                "max_iter": 100
            },
            "embeddings": {
                "similarity_threshold": 0.7,
                "aggregation_method": "centroid"
            }
        }
    },
    
    "visualization": {
        "dimensionality_reduction": "umap", # UMAP vs t-SNE
        "umap_params": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "cosine",
            "n_components": 2,
            "random_state": 42,
            "spread": 1.0,
            "local_connectivity": 1.0
        },
        "tsne_params": {
            "perplexity": 30,
            "learning_rate": 200,
            "n_iter": 1000,
            "metric": "cosine",
            "random_state": 42
        },
        "plot_config": {
            "figsize": (12, 8),
            "point_size": 50,
            "alpha": 0.7,
            "colormap": "tab10"
        }
    },
    
    "representative_samples": {
        "selection_method": "centroid_proximity", # Más cercanas al centroide
        "samples_per_cluster": 5,         # Muestras representativas
        "diversity_factor": 0.3,          # Factor diversidad selección
        "quality_threshold": 0.8          # Umbral calidad muestras
    }
}

# === BENCHMARKING ===
BENCHMARKING_CONFIG = {
    "baseline_methods": {
        "tfidf_kmeans": {
            "vectorizer": {
                "max_features": 5000,
                "ngram_range": (1, 2),
                "min_df": 5,
                "max_df": 0.8,
                "stop_words": "english"     # Será expandido multilingüe
            },
            "clustering": {
                "algorithm": "KMeans",
                "init": "k-means++",
                "n_init": 10,
                "random_state": 42
            }
        },
        
        "word2vec_average": {
            "model_params": {
                "vector_size": 300,
                "window": 5,
                "min_count": 5,
                "workers": 4,
                "epochs": 100,
                "random_state": 42
            },
            "aggregation": "mean",          # Mean pooling embeddings
            "clustering": "kmeans_cosine"
        },
        
        "random_baseline": {
            "method": "uniform_random",
            "n_runs": 10,                   # Múltiples runs para robustez
            "seed_range": (1, 100)
        }
    },
    
    "comparison_metrics": [
        "silhouette_score",
        "adjusted_rand_index", 
        "normalized_mutual_info",
        "homogeneity_score",
        "completeness_score",
        "v_measure_score",
        "topic_coherence",
        "semantic_diversity",
        "cross_lingual_consistency"
    ],
    
    "statistical_tests": {
        "significance_level": 0.05,        # Alpha para tests
        "multiple_comparisons": "bonferroni", # Corrección múltiples tests
        "effect_size_threshold": 0.2,      # Tamaño efecto mínimo
        "bootstrap_samples": 1000          # Muestras bootstrap
    }
}

# === MÉTRICAS PERFORMANCE ===
PERFORMANCE_METRICS_CONFIG = {
    "time_metrics": {
        "preprocessing_time": True,
        "vectorization_time": True,
        "clustering_time": True,
        "evaluation_time": True,
        "total_pipeline_time": True
    },
    
    "memory_metrics": {
        "peak_memory_usage": True,
        "average_memory_usage": True,
        "cache_efficiency": True,
        "memory_leaks_detection": True
    },
    
    "quality_metrics": {
        "clustering_stability": True,      # Consistencia múltiples runs
        "convergence_analysis": True,      # Análisis convergencia
        "scalability_analysis": True,      # Performance vs dataset size
        "robustness_testing": True        # Robustez ante outliers
    },
    
    "thresholds": {
        "max_processing_time_minutes": 180, # 3 horas máximo
        "max_memory_usage_gb": 8.0,        # 8GB máximo memoria
        "min_cache_hit_rate": 0.8,         # 80% hit rate mínimo
        "max_convergence_iterations": 500  # Máx iteraciones convergencia
    }
}

def get_evaluation_config(evaluation_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Obtiene configuración evaluación para tipo específico.
    
    Args:
        evaluation_type: "quick", "standard", "comprehensive", "benchmark"
        
    Returns:
        Dict con configuración evaluación optimizada
    """
    if evaluation_type == "quick":
        config = {
            "topic_coherence": {"coherence_measures": ["c_v"]},
            "cross_lingual": {"sample_sizes": {"quick_test": 50}},
            "benchmarking": {"baseline_methods": ["random_baseline"]},
            "statistical_tests": {"bootstrap_samples": 100}
        }
    elif evaluation_type == "standard":
        config = {
            "topic_coherence": TOPIC_COHERENCE_CONFIG,
            "cross_lingual": CROSS_LINGUAL_EVAL_CONFIG,
            "semantic_diversity": SEMANTIC_DIVERSITY_CONFIG,
            "interpretability": INTERPRETABILITY_CONFIG
        }
    elif evaluation_type == "comprehensive":
        config = {
            "topic_coherence": TOPIC_COHERENCE_CONFIG,
            "cross_lingual": CROSS_LINGUAL_EVAL_CONFIG,
            "semantic_diversity": SEMANTIC_DIVERSITY_CONFIG,
            "interpretability": INTERPRETABILITY_CONFIG,
            "benchmarking": BENCHMARKING_CONFIG,
            "performance": PERFORMANCE_METRICS_CONFIG
        }
    elif evaluation_type == "benchmark":
        config = {
            "benchmarking": BENCHMARKING_CONFIG,
            "performance": PERFORMANCE_METRICS_CONFIG,
            "statistical_tests": BENCHMARKING_CONFIG["statistical_tests"]
        }
    else:
        raise ValueError(f"Evaluation type '{evaluation_type}' no reconocido. "
                        f"Opciones: quick, standard, comprehensive, benchmark")
    
    return config

def get_metric_weights(focus: str = "balanced") -> Dict[str, float]:
    """
    Obtiene pesos métricas para diferentes enfoques evaluación.
    
    Args:
        focus: "quality", "performance", "interpretability", "balanced"
        
    Returns:
        Dict con pesos para cada métrica
    """
    if focus == "quality":
        return {
            "topic_coherence": 0.4,
            "semantic_diversity": 0.3,
            "cross_lingual_consistency": 0.2,
            "silhouette_score": 0.1
        }
    elif focus == "performance":
        return {
            "processing_time": 0.4,
            "memory_efficiency": 0.3,
            "cache_performance": 0.2,
            "scalability": 0.1
        }
    elif focus == "interpretability":
        return {
            "cluster_labeling_quality": 0.4,
            "representative_samples": 0.3,
            "visualization_clarity": 0.2,
            "topic_coherence": 0.1
        }
    else:  # balanced
        return {
            "topic_coherence": 0.25,
            "semantic_diversity": 0.2,
            "cross_lingual_consistency": 0.15,
            "interpretability": 0.15,
            "performance": 0.15,
            "silhouette_score": 0.1
        }

# === CONFIGURACIÓN DEFAULT ===
DEFAULT_EVALUATION_CONFIG = get_evaluation_config("comprehensive")