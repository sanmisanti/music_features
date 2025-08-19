"""
Evaluador de Clusters Semánticos para Letras Musicales

Sistema completo de evaluación automática de clustering semántico
con métricas especializadas para análisis musical.

Métricas implementadas:
- Silhouette Score (cohesión/separación)
- Davies-Bouldin Index (compacidad clusters)
- Calinski-Harabasz Index (varianza ratio)
- Semantic Coherence Score (coherencia temática)
- Cross-lingual Consistency (consistencia multilingüe)

Autor: Sistema de Clustering Musical
Fecha: Agosto 2025 - FASE 5
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score
)
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter, defaultdict
import time

try:
    from ..config.clustering_params import get_clustering_config
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    
    from config.clustering_params import get_clustering_config

# Setup logging
logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """
    Evaluador completo de calidad clustering semántico.
    
    Especializado para letras musicales con métricas tanto
    estándar como específicas del dominio musical.
    """
    
    def __init__(self, metric: str = 'cosine'):
        """
        Inicializa evaluador de clusters.
        
        Args:
            metric: Métrica distancia ('cosine', 'euclidean')
        """
        self.metric = metric
        self.config = get_clustering_config()
        
        # Cache para optimizar cálculos
        self._distance_matrix_cache = {}
        self._embeddings_cache_key = None
        
        logger.info(f"📊 ClusterEvaluator inicializado con métrica: {metric}")
    
    def evaluate_clustering(self,
                          embeddings: np.ndarray,
                          labels: np.ndarray,
                          texts: List[str] = None,
                          languages: List[str] = None,
                          detailed: bool = True) -> Dict[str, Any]:
        """
        Evaluación completa de clustering semántico.
        
        Args:
            embeddings: Array embeddings BERT (N, 384)
            labels: Array labels cluster (N,)
            texts: Lista textos originales (opcional)
            languages: Lista idiomas (opcional)
            detailed: Incluir análisis detallado por cluster
            
        Returns:
            Dict con todas las métricas de evaluación
        """
        logger.info(f"📊 Iniciando evaluación clustering:")
        logger.info(f"   Samples: {len(embeddings)}")
        logger.info(f"   Clusters únicos: {len(set(labels))}")
        logger.info(f"   Métrica: {self.metric}")
        
        start_time = time.time()
        
        # Validaciones básicas
        if len(embeddings) != len(labels):
            raise ValueError("Embeddings y labels deben tener la misma longitud")
        
        if len(set(labels)) < 2:
            raise ValueError("Se requieren al menos 2 clusters para evaluación")
        
        # Preparar cache matriz distancias
        self._prepare_distance_cache(embeddings)
        
        # Métricas estándar clustering
        standard_metrics = self._compute_standard_metrics(embeddings, labels)
        
        # Métricas especializadas música
        music_metrics = self._compute_music_specific_metrics(embeddings, labels, texts, languages)
        
        # Análisis por cluster si se solicita
        cluster_analysis = {}
        if detailed:
            cluster_analysis = self._analyze_clusters_detailed(embeddings, labels, texts, languages)
        
        # Consolidar resultados
        evaluation_time = time.time() - start_time
        
        results = {
            "standard_metrics": standard_metrics,
            "music_metrics": music_metrics,
            "cluster_analysis": cluster_analysis,
            "evaluation_info": {
                "n_samples": len(embeddings),
                "n_clusters": len(set(labels)),
                "metric": self.metric,
                "evaluation_time": evaluation_time,
                "timestamp": time.time()
            }
        }
        
        # Log resumen
        self._log_evaluation_summary(results)
        
        return results
    
    def _prepare_distance_cache(self, embeddings: np.ndarray):
        """Prepara cache de matriz distancias para optimizar cálculos."""
        cache_key = hash(embeddings.tobytes())
        
        if cache_key != self._embeddings_cache_key:
            logger.debug("   🔄 Calculando matriz distancias...")
            
            if self.metric == 'cosine':
                # Normalizar embeddings para cosine
                from sklearn.preprocessing import normalize
                normalized_embeddings = normalize(embeddings, norm='l2')
                distance_matrix = pairwise_distances(normalized_embeddings, metric='cosine')
            else:
                distance_matrix = pairwise_distances(embeddings, metric=self.metric)
            
            self._distance_matrix_cache[cache_key] = distance_matrix
            self._embeddings_cache_key = cache_key
            
            logger.debug(f"   ✅ Matriz distancias calculada: {distance_matrix.shape}")
    
    def _compute_standard_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calcula métricas estándar de clustering."""
        logger.debug("   📏 Calculando métricas estándar...")
        
        metrics = {}
        
        try:
            # Silhouette Score
            if self.metric == 'cosine':
                # Usar embeddings normalizados para cosine
                from sklearn.preprocessing import normalize
                normalized_embeddings = normalize(embeddings, norm='l2')
                silhouette_avg = silhouette_score(normalized_embeddings, labels, metric='cosine')
            else:
                silhouette_avg = silhouette_score(embeddings, labels, metric=self.metric)
            
            metrics["silhouette_score"] = float(silhouette_avg)
            
        except Exception as e:
            logger.warning(f"Error calculando Silhouette Score: {e}")
            metrics["silhouette_score"] = 0.0
        
        try:
            # Davies-Bouldin Index (menor es mejor)
            db_score = davies_bouldin_score(embeddings, labels)
            metrics["davies_bouldin_index"] = float(db_score)
            
        except Exception as e:
            logger.warning(f"Error calculando Davies-Bouldin: {e}")
            metrics["davies_bouldin_index"] = float('inf')
        
        try:
            # Calinski-Harabasz Index (mayor es mejor)
            ch_score = calinski_harabasz_score(embeddings, labels)
            metrics["calinski_harabasz_index"] = float(ch_score)
            
        except Exception as e:
            logger.warning(f"Error calculando Calinski-Harabasz: {e}")
            metrics["calinski_harabasz_index"] = 0.0
        
        # Métricas básicas distribución
        metrics.update(self._compute_distribution_metrics(labels))
        
        return metrics
    
    def _compute_distribution_metrics(self, labels: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de distribución de clusters."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Balance clusters (entropía normalizada)
        proportions = counts / len(labels)
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        max_entropy = np.log2(len(unique_labels))
        balance_score = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Estadísticas tamaño clusters
        size_stats = {
            "cluster_balance_score": float(balance_score),
            "avg_cluster_size": float(np.mean(counts)),
            "std_cluster_size": float(np.std(counts)),
            "min_cluster_size": int(np.min(counts)),
            "max_cluster_size": int(np.max(counts)),
            "cluster_size_ratio": float(np.max(counts) / np.min(counts))
        }
        
        return size_stats
    
    def _compute_music_specific_metrics(self,
                                      embeddings: np.ndarray,
                                      labels: np.ndarray,
                                      texts: List[str] = None,
                                      languages: List[str] = None) -> Dict[str, float]:
        """Calcula métricas específicas para clustering musical."""
        logger.debug("   🎵 Calculando métricas específicas música...")
        
        metrics = {}
        
        # Cohesión semántica promedio
        semantic_cohesion = self._compute_semantic_cohesion(embeddings, labels)
        metrics["semantic_cohesion"] = semantic_cohesion
        
        # Separación inter-cluster
        inter_cluster_separation = self._compute_inter_cluster_separation(embeddings, labels)
        metrics["inter_cluster_separation"] = inter_cluster_separation
        
        # Consistencia cross-lingüe si hay idiomas
        if languages:
            cross_lingual_consistency = self._compute_cross_lingual_consistency(labels, languages)
            metrics["cross_lingual_consistency"] = cross_lingual_consistency
        
        # Diversidad temática
        if texts:
            thematic_diversity = self._compute_thematic_diversity(texts, labels)
            metrics["thematic_diversity"] = thematic_diversity
        
        # Métrica compuesta calidad
        overall_quality = self._compute_overall_quality_score(metrics)
        metrics["overall_quality_score"] = overall_quality
        
        return metrics
    
    def _compute_semantic_cohesion(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calcula cohesión semántica promedio de todos los clusters."""
        unique_labels = set(labels)
        cohesions = []
        
        distance_matrix = self._distance_matrix_cache[self._embeddings_cache_key]
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) < 2:
                continue
            
            # Extraer submatriz distancias para este cluster
            cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            
            # Promedio distancias intra-cluster (excluyendo diagonal)
            mask = np.triu(np.ones_like(cluster_distances, dtype=bool), k=1)
            avg_intra_distance = np.mean(cluster_distances[mask])
            
            # Convertir a cohesión (1 - distancia para cosine, 1/(1+dist) para euclidean)
            if self.metric == 'cosine':
                cohesion = 1 - avg_intra_distance
            else:
                cohesion = 1 / (1 + avg_intra_distance)
            
            cohesions.append(cohesion)
        
        return float(np.mean(cohesions)) if cohesions else 0.0
    
    def _compute_inter_cluster_separation(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calcula separación promedio entre clusters."""
        unique_labels = sorted(set(labels))
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return 0.0
        
        # Calcular centros clusters
        cluster_centers = []
        for label in unique_labels:
            cluster_mask = labels == label
            center = np.mean(embeddings[cluster_mask], axis=0)
            cluster_centers.append(center)
        
        cluster_centers = np.array(cluster_centers)
        
        # Calcular distancias entre centros
        if self.metric == 'cosine':
            from sklearn.preprocessing import normalize
            normalized_centers = normalize(cluster_centers, norm='l2')
            center_distances = pairwise_distances(normalized_centers, metric='cosine')
        else:
            center_distances = pairwise_distances(cluster_centers, metric=self.metric)
        
        # Promedio distancias entre centros (excluyendo diagonal)
        mask = np.triu(np.ones_like(center_distances, dtype=bool), k=1)
        avg_separation = np.mean(center_distances[mask])
        
        return float(avg_separation)
    
    def _compute_cross_lingual_consistency(self, labels: np.ndarray, languages: List[str]) -> float:
        """Calcula consistencia cross-lingüe en clusters."""
        if len(languages) != len(labels):
            return 0.0
        
        unique_labels = set(labels)
        consistencies = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_languages = [languages[i] for i in range(len(languages)) if cluster_mask[i]]
            
            if len(cluster_languages) < 2:
                continue
            
            # Calcular diversidad idiomas en cluster
            lang_counts = Counter(cluster_languages)
            lang_proportions = np.array(list(lang_counts.values())) / len(cluster_languages)
            
            # Entropía idiomas (alta diversidad = baja consistencia)
            entropy = -np.sum(lang_proportions * np.log2(lang_proportions + 1e-10))
            max_entropy = np.log2(len(lang_counts))
            
            # Consistencia = 1 - entropía normalizada
            consistency = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
            consistencies.append(consistency)
        
        return float(np.mean(consistencies)) if consistencies else 0.0
    
    def _compute_thematic_diversity(self, texts: List[str], labels: np.ndarray) -> float:
        """Calcula diversidad temática usando análisis textual básico."""
        try:
            from collections import defaultdict
            import re
            
            # Extraer palabras clave por cluster
            cluster_words = defaultdict(list)
            
            for i, (text, label) in enumerate(zip(texts, labels)):
                if not text or not isinstance(text, str):
                    continue
                
                # Tokenización simple
                words = re.findall(r'\b\w+\b', text.lower())
                # Filtrar palabras muy cortas
                words = [w for w in words if len(w) > 2]
                cluster_words[label].extend(words)
            
            # Calcular diversidad vocabulario por cluster
            diversities = []
            for label, words in cluster_words.items():
                if len(words) < 2:
                    continue
                
                unique_words = len(set(words))
                total_words = len(words)
                
                # TTR (Type-Token Ratio) como medida diversidad
                diversity = unique_words / total_words if total_words > 0 else 0
                diversities.append(diversity)
            
            return float(np.mean(diversities)) if diversities else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculando diversidad temática: {e}")
            return 0.0
    
    def _compute_overall_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calcula score de calidad overall ponderado."""
        # Pesos para diferentes métricas (ajustables)
        weights = {
            "semantic_cohesion": 0.3,
            "inter_cluster_separation": 0.3,
            "cross_lingual_consistency": 0.2,
            "thematic_diversity": 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_sum += metrics[metric] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _analyze_clusters_detailed(self,
                                 embeddings: np.ndarray,
                                 labels: np.ndarray,
                                 texts: List[str] = None,
                                 languages: List[str] = None) -> Dict[str, Any]:
        """Análisis detallado por cluster individual."""
        logger.debug("   🔍 Analizando clusters individualmente...")
        
        unique_labels = sorted(set(labels))
        cluster_details = {}
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_embeddings = embeddings[cluster_mask]
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]] if texts else []
            cluster_languages = [languages[i] for i in range(len(languages)) if cluster_mask[i]] if languages else []
            
            # Análisis básico cluster
            cluster_info = {
                "size": int(np.sum(cluster_mask)),
                "percentage": float(np.mean(cluster_mask) * 100)
            }
            
            # Cohesión intra-cluster
            if len(cluster_embeddings) > 1:
                cluster_distances = pairwise_distances(cluster_embeddings, metric=self.metric)
                mask = np.triu(np.ones_like(cluster_distances, dtype=bool), k=1)
                avg_intra_distance = np.mean(cluster_distances[mask])
                
                if self.metric == 'cosine':
                    cohesion = 1 - avg_intra_distance
                else:
                    cohesion = 1 / (1 + avg_intra_distance)
                
                cluster_info["cohesion"] = float(cohesion)
            else:
                cluster_info["cohesion"] = 1.0
            
            # Análisis idiomas si disponible
            if cluster_languages:
                lang_counts = Counter(cluster_languages)
                cluster_info["languages"] = dict(lang_counts)
                cluster_info["dominant_language"] = max(lang_counts.items(), key=lambda x: x[1])[0]
                cluster_info["language_diversity"] = len(lang_counts)
            
            # Samples representativos
            if cluster_texts:
                cluster_info["sample_texts"] = cluster_texts[:3]
            
            cluster_details[int(label)] = cluster_info
        
        return cluster_details
    
    def _log_evaluation_summary(self, results: Dict[str, Any]):
        """Log resumen de evaluación."""
        standard = results["standard_metrics"]
        music = results["music_metrics"]
        info = results["evaluation_info"]
        
        logger.info("📊 EVALUACIÓN CLUSTERING COMPLETADA:")
        logger.info(f"   ⏱️ Tiempo evaluación: {info['evaluation_time']:.2f}s")
        logger.info(f"   🎯 Samples: {info['n_samples']}, Clusters: {info['n_clusters']}")
        
        # Métricas clave
        logger.info("📈 MÉTRICAS PRINCIPALES:")
        logger.info(f"   🔵 Silhouette Score: {standard['silhouette_score']:.3f}")
        logger.info(f"   🟡 Davies-Bouldin: {standard['davies_bouldin_index']:.3f}")
        logger.info(f"   🟢 Calinski-Harabasz: {standard['calinski_harabasz_index']:.1f}")
        
        if "overall_quality_score" in music:
            logger.info(f"   ⭐ Calidad Overall: {music['overall_quality_score']:.3f}")
    
    def compare_clustering_results(self, results1: Dict, results2: Dict) -> Dict[str, Any]:
        """
        Compara dos resultados de clustering.
        
        Args:
            results1: Primer resultado evaluación
            results2: Segundo resultado evaluación
            
        Returns:
            Dict con comparación detallada
        """
        comparison = {
            "method1_metrics": results1["standard_metrics"],
            "method2_metrics": results2["standard_metrics"],
            "improvements": {},
            "winner": {}
        }
        
        # Comparar métricas clave
        key_metrics = ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_index"]
        
        for metric in key_metrics:
            if metric in results1["standard_metrics"] and metric in results2["standard_metrics"]:
                val1 = results1["standard_metrics"][metric]
                val2 = results2["standard_metrics"][metric]
                
                # Para Davies-Bouldin, menor es mejor
                if metric == "davies_bouldin_index":
                    improvement = (val1 - val2) / val1 if val1 != 0 else 0
                    winner = "method1" if val2 < val1 else "method2" if val2 > val1 else "tie"
                else:
                    # Para otros, mayor es mejor
                    improvement = (val2 - val1) / val1 if val1 != 0 else 0
                    winner = "method2" if val2 > val1 else "method1" if val2 < val1 else "tie"
                
                comparison["improvements"][metric] = improvement
                comparison["winner"][metric] = winner
        
        return comparison


if __name__ == "__main__":
    # Test básico
    print("🧪 Test ClusterEvaluator:")
    
    # Generar datos fake para test
    import numpy as np
    
    np.random.seed(42)
    n_samples = 100
    embeddings = np.random.rand(n_samples, 384)
    labels = np.random.randint(0, 5, n_samples)
    texts = [f"sample_lyric_{i}" for i in range(n_samples)]
    languages = np.random.choice(["en", "es", "de"], n_samples).tolist()
    
    # Test evaluación
    evaluator = ClusterEvaluator(metric='cosine')
    results = evaluator.evaluate_clustering(embeddings, labels, texts, languages)
    
    print(f"✅ Evaluación completada:")
    print(f"   Silhouette: {results['standard_metrics']['silhouette_score']:.3f}")
    print(f"   Davies-Bouldin: {results['standard_metrics']['davies_bouldin_index']:.3f}")
    print(f"   Calinski-Harabasz: {results['standard_metrics']['calinski_harabasz_index']:.1f}")
    print(f"   Clusters analizados: {len(results['cluster_analysis'])}")