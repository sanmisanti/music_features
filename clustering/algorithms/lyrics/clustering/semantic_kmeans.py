"""
Semantic K-Means Clustering para Letras Musicales

Implementación especializada de K-Means usando embeddings BERT 384D
con distancia cosine optimizada para clustering semántico de letras.

Características:
- Cosine distance para embeddings normalizados
- Optimización automática de K (Elbow + Silhouette)
- Inicialización inteligente K-means++
- Soporte multilingüe nativo
- Métricas especializadas para letras

Autor: Sistema de Clustering Musical
Fecha: Agosto 2025 - FASE 5
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import time
from pathlib import Path

try:
    from ..vectorization.bert_vectorizer import BertVectorizer
    from ..config.clustering_params import get_clustering_config
    from ..evaluation.cluster_evaluator import ClusterEvaluator
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    
    from vectorization.bert_vectorizer import BertVectorizer
    from config.clustering_params import get_clustering_config
    from evaluation.cluster_evaluator import ClusterEvaluator

# Setup logging
logger = logging.getLogger(__name__)


class SemanticKMeans:
    """
    K-Means semántico optimizado para clustering de letras musicales.
    
    Diseñado específicamente para embeddings BERT 384D con distancia
    cosine y optimizaciones para el contexto musical.
    """
    
    def __init__(self,
                 n_clusters: int = None,
                 max_k: int = 20,
                 random_state: int = 42,
                 metric: str = 'cosine',
                 auto_optimize_k: bool = True):
        """
        Inicializa clustering K-Means semántico.
        
        Args:
            n_clusters: Número clusters. Si None, optimización automática
            max_k: Máximo K para optimización automática
            random_state: Seed para reproducibilidad
            metric: Métrica distancia ('cosine', 'euclidean')
            auto_optimize_k: Optimizar K automáticamente
        """
        self.n_clusters = n_clusters
        self.max_k = max_k
        self.random_state = random_state
        self.metric = metric
        self.auto_optimize_k = auto_optimize_k
        
        # Estado interno
        self.model = None
        self.embeddings = None
        self.labels = None
        self.cluster_centers = None
        self.inertia_history = []
        self.silhouette_history = []
        self.optimal_k = None
        self.is_fitted = False
        
        # Configuración
        self.config = get_clustering_config()
        
        logger.info(f"🔬 SemanticKMeans inicializado:")
        logger.info(f"   Clusters: {n_clusters or 'auto-optimize'}")
        logger.info(f"   Métrica: {metric}")
        logger.info(f"   Max K: {max_k}")
    
    def fit(self, embeddings: np.ndarray, texts: List[str] = None) -> 'SemanticKMeans':
        """
        Entrena modelo K-Means semántico.
        
        Args:
            embeddings: Array embeddings BERT (N, 384)
            texts: Lista textos originales (opcional, para análisis)
            
        Returns:
            Self (fluent interface)
        """
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings no pueden ser None o vacíos")
        
        # Validar dimensiones
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings deben ser 2D, recibido: {embeddings.ndim}D")
        
        if embeddings.shape[1] != 384:
            logger.warning(f"Dimensión esperada 384, recibida: {embeddings.shape[1]}")
        
        self.embeddings = embeddings
        self.texts = texts or [f"text_{i}" for i in range(len(embeddings))]
        
        logger.info(f"🔄 Iniciando clustering semántico:")
        logger.info(f"   Samples: {len(embeddings)}")
        logger.info(f"   Dimensiones: {embeddings.shape[1]}")
        logger.info(f"   Métrica: {self.metric}")
        
        start_time = time.time()
        
        # Normalizar embeddings para cosine similarity
        if self.metric == 'cosine':
            self.embeddings = normalize(embeddings, norm='l2')
            logger.info("   ✅ Embeddings normalizados para cosine similarity")
        
        # Optimización automática de K si requerida
        if self.auto_optimize_k and self.n_clusters is None:
            self.optimal_k = self._optimize_k()
            self.n_clusters = self.optimal_k
            logger.info(f"   🎯 K óptimo encontrado: {self.optimal_k}")
        
        # Entrenamiento final
        self.model = self._create_kmeans_model(self.n_clusters)
        self.labels = self.model.fit_predict(self.embeddings)
        self.cluster_centers = self.model.cluster_centers_
        
        # Marcar como entrenado
        self.is_fitted = True
        
        training_time = time.time() - start_time
        
        # Log estadísticas finales
        self._log_clustering_stats(training_time)
        
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predice clusters para nuevos embeddings.
        
        Args:
            embeddings: Nuevos embeddings BERT
            
        Returns:
            Array de labels predichas
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado primero (llamar fit())")
        
        if self.metric == 'cosine':
            embeddings = normalize(embeddings, norm='l2')
        
        return self.model.predict(embeddings)
    
    def _create_kmeans_model(self, n_clusters: int) -> KMeans:
        """Crea modelo K-Means con configuración optimizada."""
        if self.metric == 'cosine':
            # Para cosine similarity, usar euclidean en embeddings normalizados
            return KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-4,
                random_state=self.random_state,
                algorithm='lloyd'  # Más estable para embeddings densos
            )
        else:
            return KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-4,
                random_state=self.random_state
            )
    
    def _optimize_k(self) -> int:
        """
        Optimización automática de K usando Elbow + Silhouette.
        
        Returns:
            K óptimo encontrado
        """
        logger.info("🔍 Optimizando K automáticamente...")
        
        # Configurar rango K
        min_k = max(2, int(len(self.embeddings) ** 0.5 / 10))  # Heurística mínima
        max_k = min(self.max_k, len(self.embeddings) // 10)    # Máximo práctico
        
        if max_k <= min_k:
            logger.warning(f"Dataset muy pequeño, usando K=2")
            return 2
        
        k_range = range(min_k, max_k + 1)
        
        # Evaluación para cada K
        inertias = []
        silhouettes = []
        
        for k in k_range:
            try:
                model = self._create_kmeans_model(k)
                labels = model.fit_predict(self.embeddings)
                
                # Inertia (para Elbow method)
                inertias.append(model.inertia_)
                
                # Silhouette score
                if len(set(labels)) > 1:  # Evitar error si solo hay 1 cluster
                    from sklearn.metrics import silhouette_score
                    sil_score = silhouette_score(self.embeddings, labels, metric='cosine' if self.metric == 'cosine' else 'euclidean')
                    silhouettes.append(sil_score)
                else:
                    silhouettes.append(0.0)
                
                logger.debug(f"   K={k}: inertia={inertias[-1]:.3f}, silhouette={silhouettes[-1]:.3f}")
                
            except Exception as e:
                logger.warning(f"Error evaluando K={k}: {e}")
                inertias.append(float('inf'))
                silhouettes.append(0.0)
        
        # Guardar historia
        self.inertia_history = list(zip(k_range, inertias))
        self.silhouette_history = list(zip(k_range, silhouettes))
        
        # Encontrar K óptimo
        optimal_k = self._find_optimal_k(k_range, inertias, silhouettes)
        
        logger.info(f"   📊 Evaluación K completada:")
        logger.info(f"   📈 Mejor Silhouette: K={k_range[np.argmax(silhouettes)]}, score={max(silhouettes):.3f}")
        logger.info(f"   🎯 K óptimo seleccionado: {optimal_k}")
        
        return optimal_k
    
    def _find_optimal_k(self, k_range: range, inertias: List[float], silhouettes: List[float]) -> int:
        """
        Encuentra K óptimo combinando Elbow method + Silhouette.
        
        Returns:
            K óptimo
        """
        # Método 1: Silhouette score máximo
        best_silhouette_idx = np.argmax(silhouettes)
        k_silhouette = k_range[best_silhouette_idx]
        
        # Método 2: Elbow method (segunda derivada)
        k_elbow = self._find_elbow(k_range, inertias)
        
        # Combinar métodos con ponderación
        if silhouettes[best_silhouette_idx] > 0.3:  # Silhouette bueno
            # Priorizar silhouette si es bueno
            optimal_k = k_silhouette
            logger.debug(f"   Seleccionado K={optimal_k} por Silhouette score alto")
        elif k_elbow is not None:
            # Usar elbow si silhouette no es concluyente
            optimal_k = k_elbow
            logger.debug(f"   Seleccionado K={optimal_k} por Elbow method")
        else:
            # Fallback: promedio de los dos métodos
            optimal_k = (k_silhouette + (k_elbow or k_silhouette)) // 2
            logger.debug(f"   Seleccionado K={optimal_k} por promedio de métodos")
        
        # Validar rango
        optimal_k = max(min(optimal_k, max(k_range)), min(k_range))
        
        return optimal_k
    
    def _find_elbow(self, k_range: range, inertias: List[float]) -> Optional[int]:
        """Encuentra codo en curva inertia usando segunda derivada."""
        if len(inertias) < 3:
            return None
        
        # Calcular segunda derivada
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        
        if not second_derivatives:
            return None
        
        # Encontrar máximo de segunda derivada (más pronunciado codo)
        max_deriv_idx = np.argmax(second_derivatives)
        elbow_k = k_range[max_deriv_idx + 1]  # Ajustar índice
        
        return elbow_k
    
    def _log_clustering_stats(self, training_time: float):
        """Log estadísticas finales del clustering."""
        if not self.is_fitted:
            return
        
        # Contar clusters no vacíos
        unique_labels = set(self.labels)
        n_clusters_actual = len(unique_labels)
        
        # Tamaño clusters
        cluster_sizes = [np.sum(self.labels == label) for label in unique_labels]
        
        logger.info("📊 CLUSTERING COMPLETADO:")
        logger.info(f"   ⏱️ Tiempo entrenamiento: {training_time:.2f}s")
        logger.info(f"   🎯 Clusters generados: {n_clusters_actual}/{self.n_clusters}")
        logger.info(f"   📏 Tamaño promedio cluster: {np.mean(cluster_sizes):.1f}")
        logger.info(f"   📊 Distribución clusters: min={min(cluster_sizes)}, max={max(cluster_sizes)}")
        
        # Inertia final
        if hasattr(self.model, 'inertia_'):
            logger.info(f"   🔵 Inertia final: {self.model.inertia_:.3f}")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Obtiene información detallada de los clusters.
        
        Returns:
            Dict con estadísticas y métricas de clusters
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado primero")
        
        unique_labels = set(self.labels)
        
        cluster_info = {
            "n_clusters_requested": self.n_clusters,
            "n_clusters_actual": len(unique_labels),
            "total_samples": len(self.labels),
            "clusters": {}
        }
        
        # Información por cluster
        for label in sorted(unique_labels):
            cluster_mask = self.labels == label
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_texts = [self.texts[i] for i in range(len(self.texts)) if cluster_mask[i]]
            
            # Calcular cohesión intra-cluster
            if len(cluster_embeddings) > 1:
                if self.metric == 'cosine':
                    distances = pairwise_distances(cluster_embeddings, metric='cosine')
                    avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                    cohesion = 1 - avg_distance  # Convertir a similitud
                else:
                    distances = pairwise_distances(cluster_embeddings, metric='euclidean')
                    avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                    cohesion = 1 / (1 + avg_distance)  # Normalizar
            else:
                cohesion = 1.0
            
            cluster_info["clusters"][int(label)] = {
                "size": int(np.sum(cluster_mask)),
                "percentage": float(np.mean(cluster_mask) * 100),
                "cohesion": float(cohesion),
                "sample_texts": cluster_texts[:3]  # Primeros 3 textos como ejemplo
            }
        
        # Agregar métricas globales
        if hasattr(self.model, 'inertia_'):
            cluster_info["inertia"] = float(self.model.inertia_)
        
        if self.inertia_history:
            cluster_info["optimization_history"] = {
                "inertia_history": self.inertia_history,
                "silhouette_history": self.silhouette_history,
                "optimal_k": self.optimal_k
            }
        
        return cluster_info
    
    def get_cluster_assignments(self) -> np.ndarray:
        """Retorna asignaciones de cluster."""
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado primero")
        return self.labels.copy()
    
    def get_cluster_centers(self) -> np.ndarray:
        """Retorna centros de clusters."""
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado primero")
        return self.cluster_centers.copy()


def cluster_lyrics_semantic(embeddings: np.ndarray, 
                           texts: List[str] = None,
                           n_clusters: int = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Función helper para clustering semántico rápido.
    
    Args:
        embeddings: Embeddings BERT de letras
        texts: Textos originales (opcional)
        n_clusters: Número clusters (None para auto)
        **kwargs: Argumentos adicionales para SemanticKMeans
        
    Returns:
        Dict con resultados clustering
    """
    clusterer = SemanticKMeans(n_clusters=n_clusters, **kwargs)
    clusterer.fit(embeddings, texts)
    
    return {
        "labels": clusterer.get_cluster_assignments(),
        "cluster_info": clusterer.get_cluster_info(),
        "model": clusterer
    }


if __name__ == "__main__":
    # Test básico
    print("🧪 Test SemanticKMeans:")
    
    # Generar embeddings fake para test
    import numpy as np
    
    np.random.seed(42)
    n_samples = 100
    embeddings = np.random.rand(n_samples, 384)
    texts = [f"sample_lyric_{i}" for i in range(n_samples)]
    
    # Test clustering
    clusterer = SemanticKMeans(n_clusters=5)
    clusterer.fit(embeddings, texts)
    
    info = clusterer.get_cluster_info()
    print(f"✅ Clusters creados: {info['n_clusters_actual']}")
    print(f"✅ Samples procesados: {info['total_samples']}")
    print(f"✅ Inertia: {info.get('inertia', 'N/A'):.3f}")