"""
Hierarchical Clustering para Letras Musicales

Implementación especializada de clustering jerárquico usando embeddings BERT
con linkage optimizado y dendrogramas interpretativos.

Características:
- Linkage methods optimizados (ward, average, complete)
- Distancia cosine para embeddings semánticos
- Determinación automática número clusters
- Dendrogramas con interpretación musical
- Métricas especializadas para jerarquías

Autor: Sistema de Clustering Musical
Fecha: Agosto 2025 - FASE 5
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import time
from pathlib import Path

try:
    from ..config.clustering_params import get_clustering_config
    from ..evaluation.cluster_evaluator import ClusterEvaluator
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    
    from config.clustering_params import get_clustering_config
    from evaluation.cluster_evaluator import ClusterEvaluator

# Setup logging
logger = logging.getLogger(__name__)


class HierarchicalClusterer:
    """
    Clustering jerárquico optimizado para análisis semántico de letras.
    
    Especializado para embeddings BERT con interpretación musical
    y visualización de jerarquías temáticas.
    """
    
    def __init__(self,
                 n_clusters: int = None,
                 linkage: str = 'average',
                 metric: str = 'cosine',
                 auto_clusters: bool = True,
                 inconsistency_threshold: float = 1.0):
        """
        Inicializa clustering jerárquico semántico.
        
        Args:
            n_clusters: Número clusters. Si None, determinación automática
            linkage: Método linkage ('ward', 'average', 'complete', 'single')
            metric: Métrica distancia ('cosine', 'euclidean')
            auto_clusters: Determinar automáticamente número clusters
            inconsistency_threshold: Umbral inconsistencia para auto clusters
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.auto_clusters = auto_clusters
        self.inconsistency_threshold = inconsistency_threshold
        
        # Validar compatibilidad linkage-metric
        if linkage == 'ward' and metric != 'euclidean':
            logger.warning("Ward linkage requiere métrica euclidean, cambiando métrica")
            self.metric = 'euclidean'
        
        # Estado interno
        self.model = None
        self.embeddings = None
        self.labels = None
        self.linkage_matrix = None
        self.distance_matrix = None
        self.is_fitted = False
        
        # Configuración
        self.config = get_clustering_config()
        
        logger.info(f"🌳 HierarchicalClusterer inicializado:")
        logger.info(f"   Clusters: {n_clusters or 'auto-detect'}")
        logger.info(f"   Linkage: {linkage}")
        logger.info(f"   Métrica: {metric}")
    
    def fit(self, embeddings: np.ndarray, texts: List[str] = None) -> 'HierarchicalClusterer':
        """
        Entrena modelo clustering jerárquico.
        
        Args:
            embeddings: Array embeddings BERT (N, 384)
            texts: Lista textos originales (opcional)
            
        Returns:
            Self (fluent interface)
        """
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings no pueden ser None o vacíos")
        
        if embeddings.shape[0] < 2:
            raise ValueError("Se requieren al menos 2 samples para clustering jerárquico")
        
        self.embeddings = embeddings
        self.texts = texts or [f"text_{i}" for i in range(len(embeddings))]
        
        logger.info(f"🔄 Iniciando clustering jerárquico:")
        logger.info(f"   Samples: {len(embeddings)}")
        logger.info(f"   Dimensiones: {embeddings.shape[1]}")
        logger.info(f"   Linkage: {self.linkage}")
        logger.info(f"   Métrica: {self.metric}")
        
        start_time = time.time()
        
        # Calcular matriz distancia
        self._compute_distance_matrix()
        
        # Construir jerarquía
        self._build_hierarchy()
        
        # Determinar número clusters si necesario
        if self.auto_clusters and self.n_clusters is None:
            self.n_clusters = self._auto_determine_clusters()
            logger.info(f"   🎯 Clusters determinados automáticamente: {self.n_clusters}")
        
        # Clustering final
        self._perform_clustering()
        
        self.is_fitted = True
        training_time = time.time() - start_time
        
        # Log estadísticas
        self._log_clustering_stats(training_time)
        
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predice clusters para nuevos embeddings.
        
        Nota: Para clustering jerárquico, esto requiere recalcular
        la jerarquía incluyendo los nuevos puntos.
        
        Args:
            embeddings: Nuevos embeddings
            
        Returns:
            Labels predichas (aproximación)
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado primero")
        
        logger.warning("Predicción en clustering jerárquico es aproximativa")
        
        # Aproximación: asignar al cluster del centro más cercano
        from sklearn.metrics import pairwise_distances
        
        # Calcular centros clusters actuales
        cluster_centers = self._compute_cluster_centers()
        
        # Encontrar cluster más cercano para cada nuevo punto
        distances = pairwise_distances(embeddings, cluster_centers, metric=self.metric)
        predicted_labels = np.argmin(distances, axis=1)
        
        return predicted_labels
    
    def _compute_distance_matrix(self):
        """Calcula matriz de distancias entre embeddings."""
        logger.debug("   📏 Calculando matriz distancias...")
        
        if self.metric == 'cosine':
            # Normalizar embeddings para cosine similarity
            from sklearn.preprocessing import normalize
            normalized_embeddings = normalize(self.embeddings, norm='l2')
            self.distance_matrix = pairwise_distances(normalized_embeddings, metric='cosine')
        else:
            self.distance_matrix = pairwise_distances(self.embeddings, metric=self.metric)
        
        logger.debug(f"   ✅ Matriz distancias: {self.distance_matrix.shape}")
    
    def _build_hierarchy(self):
        """Construye jerarquía usando scipy.cluster.hierarchy."""
        logger.debug("   🌳 Construyendo jerarquía...")
        
        # Convertir a vector condensado para scipy
        if self.metric == 'cosine':
            # Usar distancia cosine directamente
            condensed_distances = pdist(self.embeddings, metric='cosine')
        else:
            condensed_distances = pdist(self.embeddings, metric=self.metric)
        
        # Construir linkage matrix
        self.linkage_matrix = linkage(condensed_distances, method=self.linkage)
        
        logger.debug(f"   ✅ Jerarquía construida: {self.linkage_matrix.shape}")
    
    def _auto_determine_clusters(self) -> int:
        """
        Determina automáticamente número óptimo de clusters.
        
        Usa inconsistency statistics para encontrar cortes naturales.
        
        Returns:
            Número óptimo de clusters
        """
        logger.debug("   🔍 Determinando clusters automáticamente...")
        
        # Calcular estadísticas inconsistencia
        inconsistency_stats = inconsistent(self.linkage_matrix)
        
        # Encontrar grandes saltos en inconsistencia
        inconsistency_values = inconsistency_stats[:, 3]  # Columna inconsistencia
        
        # Método 1: Umbral inconsistencia
        n_clusters_inconsistency = len(inconsistency_values[inconsistency_values < self.inconsistency_threshold]) + 1
        
        # Método 2: Máxima diferencia en distancias linkage
        linkage_distances = self.linkage_matrix[:, 2]
        distance_diffs = np.diff(linkage_distances)
        max_diff_idx = np.argmax(distance_diffs)
        n_clusters_elbow = len(self.embeddings) - max_diff_idx - 1
        
        # Método 3: Heurística basada en tamaño dataset
        n_samples = len(self.embeddings)
        n_clusters_heuristic = max(2, min(int(np.sqrt(n_samples / 2)), 20))
        
        # Combinar métodos
        candidate_clusters = [n_clusters_inconsistency, n_clusters_elbow, n_clusters_heuristic]
        
        # Filtrar valores razonables
        valid_clusters = [n for n in candidate_clusters if 2 <= n <= n_samples // 2]
        
        if not valid_clusters:
            optimal_clusters = n_clusters_heuristic
        else:
            # Usar mediana de métodos válidos
            optimal_clusters = int(np.median(valid_clusters))
        
        # Validar rango final
        optimal_clusters = max(2, min(optimal_clusters, min(20, n_samples // 2)))
        
        logger.debug(f"   📊 Candidatos clusters: {candidate_clusters}")
        logger.debug(f"   🎯 Clusters seleccionados: {optimal_clusters}")
        
        return optimal_clusters
    
    def _perform_clustering(self):
        """Realiza clustering final con número clusters determinado."""
        logger.debug(f"   ✂️ Cortando dendrograma en {self.n_clusters} clusters...")
        
        # Usar fcluster para obtener labels
        self.labels = fcluster(self.linkage_matrix, self.n_clusters, criterion='maxclust')
        
        # Ajustar labels para empezar en 0
        self.labels = self.labels - 1
        
        # Crear modelo AgglomerativeClustering para compatibilidad
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric=self.metric,
            linkage=self.linkage
        )
        
        # "Entrenar" modelo con resultados ya calculados
        self.model.labels_ = self.labels
        self.model.n_clusters_ = self.n_clusters
        
        logger.debug(f"   ✅ Clustering completado: {len(set(self.labels))} clusters únicos")
    
    def _compute_cluster_centers(self) -> np.ndarray:
        """Calcula centros de clusters (centroides)."""
        unique_labels = set(self.labels)
        centers = []
        
        for label in sorted(unique_labels):
            cluster_mask = self.labels == label
            cluster_embeddings = self.embeddings[cluster_mask]
            
            # Calcular centroide
            center = np.mean(cluster_embeddings, axis=0)
            centers.append(center)
        
        return np.array(centers)
    
    def _log_clustering_stats(self, training_time: float):
        """Log estadísticas finales del clustering."""
        if not self.is_fitted:
            return
        
        unique_labels = set(self.labels)
        n_clusters_actual = len(unique_labels)
        cluster_sizes = [np.sum(self.labels == label) for label in unique_labels]
        
        logger.info("📊 CLUSTERING JERÁRQUICO COMPLETADO:")
        logger.info(f"   ⏱️ Tiempo entrenamiento: {training_time:.2f}s")
        logger.info(f"   🌳 Clusters generados: {n_clusters_actual}")
        logger.info(f"   📏 Tamaño promedio: {np.mean(cluster_sizes):.1f}")
        logger.info(f"   📊 Distribución: min={min(cluster_sizes)}, max={max(cluster_sizes)}")
    
    def plot_dendrogram(self, 
                       max_display: int = 30,
                       save_path: Optional[Path] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Genera dendrograma de la jerarquía.
        
        Args:
            max_display: Máximo número nodos mostrar
            save_path: Ruta para guardar imagen (opcional)
            figsize: Tamaño figura matplotlib
            
        Returns:
            Figura matplotlib
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado primero")
        
        plt.figure(figsize=figsize)
        
        # Crear dendrograma
        dendrogram_data = dendrogram(
            self.linkage_matrix,
            truncate_mode='lastp' if len(self.embeddings) > max_display else None,
            p=max_display if len(self.embeddings) > max_display else None,
            orientation='top',
            leaf_rotation=90,
            leaf_font_size=8
        )
        
        plt.title(f'Dendrograma - Clustering Jerárquico Letras Musicales\\n'
                 f'Linkage: {self.linkage}, Métrica: {self.metric}, Clusters: {self.n_clusters}')
        plt.xlabel('Índice Muestra o (Tamaño Cluster)')
        plt.ylabel('Distancia')
        
        # Línea horizontal indicando corte para n_clusters
        if self.n_clusters and hasattr(self, 'linkage_matrix'):
            # Encontrar altura de corte
            cut_height = self.linkage_matrix[-(self.n_clusters-1), 2]
            plt.axhline(y=cut_height, color='red', linestyle='--', alpha=0.7,
                       label=f'Corte para {self.n_clusters} clusters')
            plt.legend()
        
        plt.tight_layout()
        
        # Guardar si se especifica ruta
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Dendrograma guardado: {save_path}")
        
        return plt.gcf()
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Obtiene información detallada de los clusters jerárquicos.
        
        Returns:
            Dict con estadísticas y jerarquía
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado primero")
        
        unique_labels = set(self.labels)
        
        info = {
            "n_clusters": len(unique_labels),
            "total_samples": len(self.labels),
            "linkage_method": self.linkage,
            "distance_metric": self.metric,
            "clusters": {}
        }
        
        # Información por cluster
        for label in sorted(unique_labels):
            cluster_mask = self.labels == label
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_texts = [self.texts[i] for i in range(len(self.texts)) if cluster_mask[i]]
            
            # Cohesión intra-cluster
            if len(cluster_embeddings) > 1:
                intra_distances = pairwise_distances(cluster_embeddings, metric=self.metric)
                avg_intra_distance = np.mean(intra_distances[np.triu_indices_from(intra_distances, k=1)])
                cohesion = 1 / (1 + avg_intra_distance) if self.metric == 'euclidean' else 1 - avg_intra_distance
            else:
                cohesion = 1.0
            
            info["clusters"][int(label)] = {
                "size": int(np.sum(cluster_mask)),
                "percentage": float(np.mean(cluster_mask) * 100),
                "cohesion": float(cohesion),
                "sample_texts": cluster_texts[:3]
            }
        
        # Agregar información jerarquía
        if self.linkage_matrix is not None:
            info["hierarchy"] = {
                "linkage_matrix_shape": self.linkage_matrix.shape,
                "max_distance": float(np.max(self.linkage_matrix[:, 2])),
                "min_distance": float(np.min(self.linkage_matrix[:, 2])),
                "avg_distance": float(np.mean(self.linkage_matrix[:, 2]))
            }
        
        return info
    
    def get_cluster_assignments(self) -> np.ndarray:
        """Retorna asignaciones de cluster."""
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado primero")
        return self.labels.copy()


def cluster_lyrics_hierarchical(embeddings: np.ndarray,
                               texts: List[str] = None,
                               n_clusters: int = None,
                               **kwargs) -> Dict[str, Any]:
    """
    Función helper para clustering jerárquico rápido.
    
    Args:
        embeddings: Embeddings BERT de letras
        texts: Textos originales (opcional)
        n_clusters: Número clusters (None para auto)
        **kwargs: Argumentos adicionales para HierarchicalClusterer
        
    Returns:
        Dict con resultados clustering
    """
    clusterer = HierarchicalClusterer(n_clusters=n_clusters, **kwargs)
    clusterer.fit(embeddings, texts)
    
    return {
        "labels": clusterer.get_cluster_assignments(),
        "cluster_info": clusterer.get_cluster_info(),
        "model": clusterer
    }


if __name__ == "__main__":
    # Test básico
    print("🧪 Test HierarchicalClusterer:")
    
    # Generar embeddings fake para test
    import numpy as np
    
    np.random.seed(42)
    n_samples = 50  # Menor para jerarquico
    embeddings = np.random.rand(n_samples, 384)
    texts = [f"sample_lyric_{i}" for i in range(n_samples)]
    
    # Test clustering
    clusterer = HierarchicalClusterer(n_clusters=5, linkage='average')
    clusterer.fit(embeddings, texts)
    
    info = clusterer.get_cluster_info()
    print(f"✅ Clusters creados: {info['n_clusters']}")
    print(f"✅ Samples procesados: {info['total_samples']}")
    print(f"✅ Linkage method: {info['linkage_method']}")
    
    # Test dendrograma
    try:
        fig = clusterer.plot_dendrogram()
        print("✅ Dendrograma generado")
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Error dendrograma: {e}")