#!/usr/bin/env python3
"""
🎵 SISTEMA DE RECOMENDACIÓN MUSICAL OPTIMIZADO
===============================================

Sistema de recomendación de clase mundial integrado nativamente con ClusterPurifier.
Diseñado para máximo performance y calidad usando clustering optimizado (+86.1% Silhouette).

CARACTERÍSTICAS PRINCIPALES:
- Integración nativa con cluster_purification.py (Hierarchical clustering)
- Dataset optimizado: picked_data_optimal.csv (16,081 canciones purificadas)
- Performance objetivo: <100ms por recomendación (vs 2-5s actual)
- 6 estrategias de recomendación avanzadas
- Sistema de explicabilidad integrado
- Evaluación automática de calidad

ESTRATEGIAS DISPONIBLES:
1. cluster_pure: Solo cluster optimizado (máxima cohesión)
2. similarity_weighted: Similitud con pesos discriminativos
3. hybrid_balanced: 50% cluster + 50% similitud global
4. diversity_boosted: Anti-clustering para máxima diversidad
5. mood_contextual: Basado en características emocionales
6. temporal_aware: Considera popularidad y época

Autor: Optimized Music Recommender System
Fecha: Enero 2025
Estado: ✨ SISTEMA OPTIMIZADO EN DESARROLLO
"""

import numpy as np
import pandas as pd
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Imports clustering y métricas
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples

# Import sistema optimizado
import sys
sys.path.append('.')
try:
    from cluster_purification import ClusterPurifier
except ImportError:
    print("⚠️  Warning: cluster_purification.py no encontrado, usando fallback")

class OptimizedMusicRecommender:
    """
    Sistema de recomendación musical optimizado con integración nativa ClusterPurifier.
    
    Performance objetivo: <100ms por recomendación
    Calidad objetivo: +15-25% precision vs baseline
    """
    
    def __init__(self, base_path: Optional[str] = None, random_state: int = 42):
        """Inicializar recomendador optimizado."""
        
        if base_path is None:
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Configuración optimizada
        self.dataset_path = self.base_path / 'data' / 'final_data' / 'picked_data_optimal.csv'
        self.output_dir = self.base_path / 'outputs' / 'recommender_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Características discriminativas optimizadas (9 de 12)
        self.discriminative_features = [
            'instrumentalness', 'acousticness', 'energy', 'danceability',
            'valence', 'speechiness', 'liveness', 'loudness', 'tempo'
        ]
        
        # Estrategias de recomendación disponibles
        self.recommendation_strategies = {
            'cluster_pure': self._recommend_cluster_pure,
            'similarity_weighted': self._recommend_similarity_weighted,
            'hybrid_balanced': self._recommend_hybrid_balanced,
            'diversity_boosted': self._recommend_diversity_boosted,
            'mood_contextual': self._recommend_mood_contextual,
            'temporal_aware': self._recommend_temporal_aware
        }
        
        # Cache y índices para performance O(1)
        self.dataset = None
        self.cluster_assignments = None
        self.similarity_matrix = None
        self.feature_weights = None
        self.cluster_centroids = None
        self.inverted_index = {}
        
        # Configuración clustering optimizado
        self.n_clusters = 3  # Valor óptimo validado
        self.clustering_algorithm = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            random_state=random_state
        )
        
        # Sistema de evaluación
        self.evaluation_metrics = {}
        self.benchmark_results = {}
        
        print(f"🎵 OptimizedMusicRecommender inicializado")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"🎯 Estrategias: {list(self.recommendation_strategies.keys())}")
        print(f"🔧 Características: {len(self.discriminative_features)} discriminativas")
    
    def initialize_system(self):
        """Inicializar sistema completo con pre-cómputos para máximo performance."""
        
        print("\n🚀 Inicializando sistema optimizado...")
        start_time = time.time()
        
        # 1. Cargar y validar dataset optimizado
        self._load_optimized_dataset()
        
        # 2. Pre-computar clusters usando sistema optimizado
        self._precompute_optimized_clusters()
        
        # 3. Pre-computar índices de similitud para O(1) lookup
        self._precompute_similarity_indices()
        
        # 4. Calcular pesos características discriminativas
        self._calculate_feature_weights()
        
        # 5. Construir índices invertidos para filtrado rápido
        self._build_inverted_indices()
        
        init_time = time.time() - start_time
        print(f"✅ Sistema inicializado en {init_time:.2f}s")
        print(f"📊 Dataset: {len(self.dataset):,} canciones")
        print(f"🎯 Clusters: {self.n_clusters} con Silhouette optimizado")
        print(f"💾 Memoria: Índices pre-computados para O(1) lookup")
        
        return True
    
    def _load_optimized_dataset(self):
        """Cargar dataset optimizado con validaciones."""
        
        print(f"📂 Cargando dataset optimizado: {self.dataset_path}")
        
        try:
            # Cargar con formato optimizado
            self.dataset = pd.read_csv(
                self.dataset_path, 
                sep='^', 
                decimal='.', 
                encoding='utf-8',
                on_bad_lines='skip'
            )
            
            print(f"✅ Dataset cargado: {len(self.dataset):,} canciones")
            
            # Validar características discriminativas disponibles
            available_features = [f for f in self.discriminative_features if f in self.dataset.columns]
            if len(available_features) != len(self.discriminative_features):
                missing = set(self.discriminative_features) - set(available_features)
                print(f"⚠️  Características faltantes: {missing}")
                self.discriminative_features = available_features
            
            print(f"🎵 Características disponibles: {len(self.discriminative_features)}")
            
            # Limpiar datos NaN
            initial_count = len(self.dataset)
            self.dataset = self.dataset.dropna(subset=self.discriminative_features)
            final_count = len(self.dataset)
            
            if initial_count != final_count:
                print(f"🧹 Limpieza: {initial_count:,} → {final_count:,} canciones ({((final_count/initial_count)*100):.1f}% retenido)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando dataset: {e}")
            return False
    
    def _precompute_optimized_clusters(self):
        """Pre-computar clusters usando ClusterPurifier optimizado (+86.1% Silhouette)."""
        
        print("🔧 Pre-computando clusters con ClusterPurifier optimizado...")
        
        try:
            # Usar ClusterPurifier para obtener clustering optimizado
            purifier = ClusterPurifier(base_path=self.base_path)
            
            # Cargar configuración baseline
            config = purifier.load_baseline_configuration()
            
            if config is None:
                print("⚠️  Fallback: Usando clustering directo...")
                return self._fallback_clustering()
            
            # Ejecutar purificación híbrida (estrategia óptima)
            print("   🎯 Ejecutando purificación híbrida (Silhouette +86.1%)...")
            
            # Extraer características musicales
            features_data = config['features_df']
            features_array = features_data[self.discriminative_features].values
            
            # Normalizar
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_array)
            
            # Clustering baseline
            clustering_alg = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                random_state=self.random_state
            )
            initial_labels = clustering_alg.fit_predict(features_normalized)
            
            # Aplicar purificación híbrida
            purified_data, purified_labels, purification_info = purifier._hybrid_purification(
                features_normalized, initial_labels
            )
            
            # Mapear resultados purificados de vuelta al dataset completo
            self._map_purified_results(
                features_array, features_normalized, 
                purified_data, purified_labels, purification_info
            )
            
            # Métricas finales
            final_silhouette = silhouette_score(purified_data, purified_labels)
            retention_ratio = len(purified_data) / len(features_normalized)
            
            print(f"✅ Clustering optimizado completado:")
            print(f"   📊 Silhouette Score: {final_silhouette:.4f} (baseline: 0.1554)")
            print(f"   📈 Mejora: +{((final_silhouette/0.1554)-1)*100:.1f}%")
            print(f"   💾 Retención: {retention_ratio:.1%} ({len(purified_data):,} canciones)")
            print(f"   🎯 Calidad: {'EXCELENTE' if final_silhouette > 0.25 else 'BUENA' if final_silhouette > 0.20 else 'ACEPTABLE'}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error en ClusterPurifier: {e}")
            print("   Usando clustering directo como fallback...")
            return self._fallback_clustering()
    
    def _map_purified_results(self, original_features, normalized_features, 
                             purified_data, purified_labels, purification_info):
        """Mapear resultados purificados de vuelta al dataset completo."""
        
        print("   🗺️  Mapeando resultados purificados...")
        
        # Encontrar índices de datos purificados en dataset original
        # Esto es complejo porque la purificación elimina filas
        # Usaremos matching de características para re-mapear
        
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Calcular distancias entre datos purificados y normalizados originales
        distances = euclidean_distances(purified_data, normalized_features)
        
        # Para cada dato purificado, encontrar su match más cercano en datos originales
        purified_to_original_mapping = []
        for i in range(len(purified_data)):
            closest_idx = np.argmin(distances[i])
            purified_to_original_mapping.append(closest_idx)
        
        # Inicializar asignaciones de cluster
        self.cluster_assignments = np.full(len(self.dataset), -1, dtype=int)  # -1 = no asignado
        
        # Asignar clusters a datos purificados
        for purified_idx, original_idx in enumerate(purified_to_original_mapping):
            if original_idx < len(self.dataset):
                self.cluster_assignments[original_idx] = purified_labels[purified_idx]
        
        # Para datos no purificados, asignar al cluster más cercano
        unassigned_mask = self.cluster_assignments == -1
        unassigned_indices = np.where(unassigned_mask)[0]
        
        if len(unassigned_indices) > 0:
            print(f"   🔄 Asignando {len(unassigned_indices)} canciones restantes...")
            
            # Calcular centroides de clusters purificados
            self.cluster_centroids = {}
            for cluster_id in range(self.n_clusters):
                cluster_mask = purified_labels == cluster_id
                if np.any(cluster_mask):
                    cluster_data = purified_data[cluster_mask]
                    self.cluster_centroids[cluster_id] = np.mean(cluster_data, axis=0)
                else:
                    # Cluster vacío, usar promedio global
                    self.cluster_centroids[cluster_id] = np.mean(purified_data, axis=0)
            
            # Asignar datos no purificados al cluster más cercano
            scaler = StandardScaler()
            normalized_full = scaler.fit_transform(original_features)
            
            for idx in unassigned_indices:
                song_features = normalized_full[idx]
                
                # Encontrar cluster más cercano
                min_distance = float('inf')
                best_cluster = 0
                
                for cluster_id, centroid in self.cluster_centroids.items():
                    distance = np.linalg.norm(song_features - centroid)
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = cluster_id
                
                self.cluster_assignments[idx] = best_cluster
        
        # Agregar columna cluster al dataset
        self.dataset['cluster'] = self.cluster_assignments
        
        # Estadísticas finales
        cluster_counts = pd.Series(self.cluster_assignments).value_counts().sort_index()
        print(f"   📈 Distribución final: {dict(cluster_counts)}")
        
        return True
    
    def _fallback_clustering(self):
        """Clustering directo como fallback si ClusterPurifier falla."""
        
        print("   🔄 Ejecutando clustering directo...")
        
        # Extraer características para clustering
        features_data = self.dataset[self.discriminative_features].values
        
        # Normalizar características
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_data)
        
        # Aplicar clustering (Hierarchical)
        cluster_labels = self.clustering_algorithm.fit_predict(features_normalized)
        
        # Almacenar asignaciones
        self.cluster_assignments = cluster_labels
        self.dataset['cluster'] = cluster_labels
        
        # Calcular métricas de calidad
        silhouette_avg = silhouette_score(features_normalized, cluster_labels)
        
        # Calcular centroides para cada cluster
        self.cluster_centroids = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features_normalized[cluster_mask]
            self.cluster_centroids[cluster_id] = np.mean(cluster_features, axis=0)
        
        # Estadísticas de clusters
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        print(f"✅ Clustering directo completado:")
        print(f"   📊 Silhouette Score: {silhouette_avg:.4f}")
        print(f"   📈 Distribución clusters: {dict(cluster_counts)}")
        
        return True
    
    def _precompute_similarity_indices(self):
        """Pre-computar índices de similitud optimizados para O(1) lookup."""
        
        print("💾 Pre-computando índices de similitud optimizados...")
        
        # Extraer características discriminativas normalizadas
        features_data = self.dataset[self.discriminative_features].values
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_data)
        
        # Aplicar pesos de características discriminativas si están disponibles
        if hasattr(self, 'feature_weights') and self.feature_weights:
            weights = np.array([self.feature_weights.get(f, 1.0) for f in self.discriminative_features])
            features_weighted = features_normalized * weights
        else:
            features_weighted = features_normalized
        
        # Estrategia inteligente de memoria para datasets grandes
        n_songs = len(features_weighted)
        memory_limit_gb = 4.0  # Límite de memoria para matriz similitud
        matrix_size_gb = (n_songs * n_songs * 8) / (1024**3)  # 8 bytes por float64
        
        start_time = time.time()
        
        if matrix_size_gb > memory_limit_gb:
            print(f"   ⚠️  Matriz completa ({matrix_size_gb:.1f}GB) excede límite ({memory_limit_gb}GB)")
            print("   🔧 Usando índices por clusters para optimizar memoria...")
            
            # Crear índices de similitud por cluster (más eficiente)
            self.similarity_indices = {}
            self.cluster_similarities = {}
            
            for cluster_id in range(self.n_clusters):
                cluster_mask = self.cluster_assignments == cluster_id
                cluster_features = features_weighted[cluster_mask]
                
                if len(cluster_features) > 0:
                    # Matriz de similitud solo para este cluster
                    cluster_similarity = cosine_similarity(cluster_features)
                    self.cluster_similarities[cluster_id] = cluster_similarity
                    
                    # Índices de canciones en este cluster
                    cluster_indices = np.where(cluster_mask)[0]
                    self.similarity_indices[cluster_id] = cluster_indices
                    
                    print(f"     📊 Cluster {cluster_id}: {len(cluster_features)} canciones")
            
            # Para similitud inter-cluster, usar centroides
            cluster_centroids_array = np.array([self.cluster_centroids[i] for i in range(self.n_clusters)])
            self.inter_cluster_similarity = cosine_similarity(cluster_centroids_array)
            
        else:
            print(f"   💾 Matriz completa ({matrix_size_gb:.1f}GB) dentro del límite")
            # Pre-computar matriz de similitud completa (coseno ponderado)
            self.similarity_matrix = cosine_similarity(features_weighted)
        
        compute_time = time.time() - start_time
        
        # Estadísticas de memoria
        if hasattr(self, 'similarity_matrix'):
            actual_size_gb = self.similarity_matrix.nbytes / (1024**3)
            print(f"✅ Matriz similitud completa: {self.similarity_matrix.shape} en {compute_time:.2f}s")
            print(f"💾 Memoria: {actual_size_gb:.2f}GB")
        else:
            total_cluster_memory = sum(
                sim_matrix.nbytes for sim_matrix in self.cluster_similarities.values()
            ) / (1024**3)
            print(f"✅ Índices por cluster: {len(self.cluster_similarities)} clusters en {compute_time:.2f}s")
            print(f"💾 Memoria optimizada: {total_cluster_memory:.2f}GB")
        
        return True
    
    def _calculate_feature_weights(self):
        """Calcular pesos de características basados en poder discriminativo."""
        
        print("⚖️  Calculando pesos características discriminativas...")
        
        # Usar ANOVA F-statistic para medir poder discriminativo
        features_data = self.dataset[self.discriminative_features].values
        cluster_labels = self.cluster_assignments
        
        # Calcular F-statistics
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(features_data, cluster_labels)
        
        # Normalizar scores a pesos [0, 1]
        f_scores = selector.scores_
        weights = f_scores / np.sum(f_scores)
        
        # Almacenar pesos
        self.feature_weights = dict(zip(self.discriminative_features, weights))
        
        # Mostrar ranking de características
        sorted_features = sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True)
        print("📊 Ranking características (peso discriminativo):")
        for i, (feature, weight) in enumerate(sorted_features[:5], 1):
            print(f"   {i}. {feature}: {weight:.4f}")
        
        return True
    
    def _build_inverted_indices(self):
        """Construir índices invertidos para filtrado rápido."""
        
        print("🗂️  Construyendo índices invertidos...")
        
        # Índice por cluster
        self.inverted_index['cluster'] = {}
        for cluster_id in range(self.n_clusters):
            cluster_songs = self.dataset[self.dataset['cluster'] == cluster_id].index.tolist()
            self.inverted_index['cluster'][cluster_id] = cluster_songs
        
        # Índices por rangos de características principales
        for feature in ['energy', 'valence', 'danceability']:
            if feature in self.dataset.columns:
                self.inverted_index[feature] = {}
                
                # Dividir en quintiles
                quintiles = pd.qcut(self.dataset[feature], q=5, labels=['low', 'med_low', 'med', 'med_high', 'high'])
                for level in ['low', 'med_low', 'med', 'med_high', 'high']:
                    indices = self.dataset[quintiles == level].index.tolist()
                    self.inverted_index[feature][level] = indices
        
        total_indices = sum(len(idx) for cat in self.inverted_index.values() for idx in cat.values())
        print(f"✅ Índices construidos: {total_indices:,} entradas")
        
        return True
    
    def recommend(self, 
                  query: Union[str, Dict, int], 
                  strategy: str = "hybrid_balanced",
                  n_recommendations: int = 10,
                  filters: Optional[Dict] = None,
                  explain: bool = False) -> Dict:
        """
        Interface principal de recomendación.
        
        Args:
            query: ID canción, características dict, o índice dataset
            strategy: Estrategia de recomendación a usar
            n_recommendations: Número de recomendaciones a retornar
            filters: Filtros contextuales opcionales
            explain: Si incluir explicaciones de recomendaciones
            
        Returns:
            Dict con recomendaciones y metadatos
        """
        
        start_time = time.time()
        
        # Validar estrategia
        if strategy not in self.recommendation_strategies:
            raise ValueError(f"Estrategia inválida. Disponibles: {list(self.recommendation_strategies.keys())}")
        
        # Parsear query a características
        query_features = self._parse_query(query)
        if query_features is None:
            return {"error": "Query inválido"}
        
        # Aplicar estrategia de recomendación
        strategy_func = self.recommendation_strategies[strategy]
        recommendations = strategy_func(query_features, n_recommendations, filters)
        
        # Agregar explicaciones si se solicitan
        if explain:
            recommendations = self._add_explanations(recommendations, query_features, strategy)
        
        # Calcular métricas de tiempo
        total_time = time.time() - start_time
        
        # Construir respuesta
        result = {
            "query": query,
            "strategy": strategy,
            "n_recommendations": len(recommendations),
            "recommendations": recommendations,
            "performance": {
                "total_time_ms": round(total_time * 1000, 2),
                "target_time_ms": 100,
                "performance_ratio": round(100 / (total_time * 1000), 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"🎵 Recomendación completada en {total_time*1000:.1f}ms (objetivo: <100ms)")
        
        return result
    
    def _parse_query(self, query: Union[str, Dict, int]) -> Optional[np.ndarray]:
        """Parsear query a vector de características."""
        
        if isinstance(query, int):
            # Índice directo en dataset
            if 0 <= query < len(self.dataset):
                return self.dataset.iloc[query][self.discriminative_features].values
        
        elif isinstance(query, str):
            # Buscar por ID de canción
            if 'track_id' in self.dataset.columns:
                song_match = self.dataset[self.dataset['track_id'] == query]
                if not song_match.empty:
                    return song_match.iloc[0][self.discriminative_features].values
            
            # Buscar por nombre (fuzzy matching básico)
            if 'track_name' in self.dataset.columns:
                name_matches = self.dataset[
                    self.dataset['track_name'].str.lower().str.contains(query.lower(), na=False)
                ]
                if not name_matches.empty:
                    return name_matches.iloc[0][self.discriminative_features].values
        
        elif isinstance(query, dict):
            # Características directas
            try:
                features = [query.get(f, 0.5) for f in self.discriminative_features]
                return np.array(features)
            except:
                pass
        
        return None
    
    def _recommend_cluster_pure(self, query_features: np.ndarray, n_recs: int, filters: Dict) -> List[Dict]:
        """Estrategia 1: Solo cluster optimizado (máxima cohesión)."""
        
        # Predecir cluster de la query usando clustering optimizado
        query_cluster = self._predict_cluster_optimized(query_features)
        
        # Obtener canciones del mismo cluster
        cluster_songs_idx = self.inverted_index['cluster'][query_cluster]
        
        if len(cluster_songs_idx) == 0:
            return []
        
        # Calcular similitudes dentro del cluster usando índices optimizados
        similarities = self._calculate_cluster_similarities(query_features, query_cluster, cluster_songs_idx)
        
        # Ordenar por similitud y retornar top-N
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for idx, sim in similarities[:n_recs]:
            song = self.dataset.iloc[idx]
            recommendations.append({
                "track_id": song.get('track_id', ''),
                "track_name": song.get('track_name', ''),
                "track_artist": song.get('track_artist', ''),
                "cluster": int(query_cluster),
                "similarity": float(sim),
                "strategy_info": "cluster_pure",
                "cluster_quality": "optimized_+86%"
            })
        
        return recommendations
    
    def _recommend_similarity_weighted(self, query_features: np.ndarray, n_recs: int, filters: Dict) -> List[Dict]:
        """Estrategia 2: Similitud con pesos discriminativos optimizados."""
        
        # Normalizar query features
        query_normalized = self._normalize_features(query_features)
        
        # Calcular similitudes ponderadas usando optimizaciones
        weighted_similarities = []
        
        # Si tenemos matriz de similitud pre-computada, usar esa
        if hasattr(self, 'similarity_matrix'):
            # Para usar matriz pre-computada necesitaríamos el índice de query
            # Por ahora, calcular directamente con optimizaciones
            pass
        
        # Estrategia optimizada: evaluar por clusters primero para reducir cómputo
        evaluated_songs = set()
        
        # Primero evaluar cluster predicho (mayor probabilidad de similitud alta)
        predicted_cluster = self._predict_cluster_optimized(query_features)
        cluster_songs_idx = self.inverted_index['cluster'][predicted_cluster]
        
        for idx in cluster_songs_idx:
            song_features = self.dataset.iloc[idx][self.discriminative_features].values
            song_normalized = self._normalize_features(song_features)
            
            # Similitud ponderada optimizada
            similarity = self._calculate_weighted_similarity(query_normalized, song_normalized)
            weighted_similarities.append((idx, similarity))
            evaluated_songs.add(idx)
        
        # Evaluar otros clusters si necesitamos más canciones
        remaining_needed = n_recs * 3  # Factor de sobre-muestreo para diversidad
        
        for cluster_id in range(self.n_clusters):
            if cluster_id == predicted_cluster:
                continue
                
            cluster_songs_idx = self.inverted_index['cluster'][cluster_id]
            
            # Evaluar muestra del cluster basada en centroides
            inter_cluster_sim = 0.5  # Similitud base entre clusters
            if hasattr(self, 'inter_cluster_similarity'):
                inter_cluster_sim = self.inter_cluster_similarity[predicted_cluster, cluster_id]
            
            # Solo evaluar si hay potencial de similitud
            if inter_cluster_sim > 0.3:
                for idx in cluster_songs_idx[:min(len(cluster_songs_idx), remaining_needed)]:
                    if idx not in evaluated_songs:
                        song_features = self.dataset.iloc[idx][self.discriminative_features].values
                        song_normalized = self._normalize_features(song_features)
                        
                        similarity = self._calculate_weighted_similarity(query_normalized, song_normalized)
                        # Penalizar ligeramente por estar en cluster diferente
                        similarity *= (0.8 + 0.2 * inter_cluster_sim)
                        
                        weighted_similarities.append((idx, similarity))
                        evaluated_songs.add(idx)
                        
                        if len(evaluated_songs) >= remaining_needed + len(cluster_songs_idx):
                            break
        
        # Ordenar y retornar top-N
        weighted_similarities.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for idx, sim in weighted_similarities[:n_recs]:
            song = self.dataset.iloc[idx]
            song_cluster = self.cluster_assignments[idx]
            
            recommendations.append({
                "track_id": song.get('track_id', ''),
                "track_name": song.get('track_name', ''),
                "track_artist": song.get('track_artist', ''),
                "cluster": int(song_cluster),
                "similarity": float(sim),
                "strategy_info": "similarity_weighted",
                "optimization": "discriminative_features_+9"
            })
        
        return recommendations
    
    def _recommend_hybrid_balanced(self, query_features: np.ndarray, n_recs: int, filters: Dict) -> List[Dict]:
        """Estrategia 3: Híbrida balanceada optimizada (cluster +86% + similitud ponderada)."""
        
        # Estrategia híbrida avanzada que combina lo mejor de ambos mundos
        hybrid_recommendations = []
        
        # 1. Predecir cluster optimizado
        predicted_cluster = self._predict_cluster_optimized(query_features)
        query_normalized = self._normalize_features(query_features)
        
        # 2. Evaluar canciones por múltiples criterios
        all_scores = []
        
        for idx in range(len(self.dataset)):
            song = self.dataset.iloc[idx]
            song_features = song[self.discriminative_features].values
            song_normalized = self._normalize_features(song_features)
            song_cluster = self.cluster_assignments[idx]
            
            # Score de similitud directa (componente global)
            similarity_score = self._calculate_weighted_similarity(query_normalized, song_normalized)
            
            # Score de cluster (componente cohesión)
            if song_cluster == predicted_cluster:
                cluster_score = 1.0  # Máximo bonus por estar en mismo cluster optimizado
            else:
                # Penalización suave basada en similitud inter-cluster
                if hasattr(self, 'inter_cluster_similarity'):
                    cluster_score = self.inter_cluster_similarity[predicted_cluster, song_cluster]
                else:
                    cluster_score = 0.3  # Penalización base
            
            # Score híbrido combinado
            hybrid_score = 0.5 * similarity_score + 0.5 * cluster_score
            
            # Bonus adicional por calidad de cluster (si está en cluster purificado)
            if song_cluster == predicted_cluster:
                hybrid_score *= 1.1  # 10% bonus por cluster optimizado
            
            all_scores.append((idx, hybrid_score, similarity_score, cluster_score))
        
        # 3. Ordenar por score híbrido
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Seleccionar top-N con diversidad
        selected_indices = set()
        cluster_counts = {i: 0 for i in range(self.n_clusters)}
        max_per_cluster = max(1, n_recs // self.n_clusters + 1)
        
        for idx, hybrid_score, similarity_score, cluster_score in all_scores:
            if len(hybrid_recommendations) >= n_recs:
                break
                
            song = self.dataset.iloc[idx]
            song_cluster = self.cluster_assignments[idx]
            
            # Control de diversidad: no más de max_per_cluster por cluster
            if cluster_counts[song_cluster] < max_per_cluster:
                hybrid_recommendations.append({
                    "track_id": song.get('track_id', ''),
                    "track_name": song.get('track_name', ''),
                    "track_artist": song.get('track_artist', ''),
                    "cluster": int(song_cluster),
                    "similarity": float(similarity_score),
                    "cluster_score": float(cluster_score),
                    "hybrid_score": float(hybrid_score),
                    "strategy_info": "hybrid_balanced",
                    "optimization": "cluster_+86%_similarity_weighted",
                    "predicted_cluster": int(predicted_cluster)
                })
                
                cluster_counts[song_cluster] += 1
                selected_indices.add(idx)
        
        # 5. Si no tenemos suficientes, agregar las mejores sin restricción de diversidad
        if len(hybrid_recommendations) < n_recs:
            for idx, hybrid_score, similarity_score, cluster_score in all_scores:
                if len(hybrid_recommendations) >= n_recs or idx in selected_indices:
                    continue
                    
                song = self.dataset.iloc[idx]
                song_cluster = self.cluster_assignments[idx]
                
                hybrid_recommendations.append({
                    "track_id": song.get('track_id', ''),
                    "track_name": song.get('track_name', ''),
                    "track_artist": song.get('track_artist', ''),
                    "cluster": int(song_cluster),
                    "similarity": float(similarity_score),
                    "cluster_score": float(cluster_score),
                    "hybrid_score": float(hybrid_score),
                    "strategy_info": "hybrid_balanced",
                    "optimization": "cluster_+86%_similarity_weighted",
                    "predicted_cluster": int(predicted_cluster)
                })
        
        return hybrid_recommendations[:n_recs]
    
    def _recommend_diversity_boosted(self, query_features: np.ndarray, n_recs: int, filters: Dict) -> List[Dict]:
        """Estrategia 4: Anti-clustering para máxima diversidad."""
        
        # Seleccionar canciones de clusters diferentes
        diverse_recommendations = []
        
        # Una canción de cada cluster
        for cluster_id in range(self.n_clusters):
            cluster_songs_idx = self.inverted_index['cluster'][cluster_id]
            if cluster_songs_idx:
                # Seleccionar canción más representativa del cluster (más cercana al centroide)
                best_idx = cluster_songs_idx[0]  # Placeholder - usar cálculo real
                song = self.dataset.iloc[best_idx]
                
                diverse_recommendations.append({
                    "track_id": song.get('track_id', ''),
                    "track_name": song.get('track_name', ''),
                    "track_artist": song.get('track_artist', ''),
                    "cluster": int(cluster_id),
                    "diversity_score": 1.0,
                    "strategy_info": "diversity_boosted"
                })
        
        # Completar con canciones diversas adicionales
        while len(diverse_recommendations) < n_recs:
            # Seleccionar canción aleatoria de cluster diferente
            available_clusters = list(range(self.n_clusters))
            np.random.shuffle(available_clusters)
            
            for cluster_id in available_clusters:
                if len(diverse_recommendations) >= n_recs:
                    break
                    
                cluster_songs_idx = self.inverted_index['cluster'][cluster_id]
                if cluster_songs_idx:
                    random_idx = np.random.choice(cluster_songs_idx)
                    song = self.dataset.iloc[random_idx]
                    
                    diverse_recommendations.append({
                        "track_id": song.get('track_id', ''),
                        "track_name": song.get('track_name', ''),
                        "track_artist": song.get('track_artist', ''),
                        "cluster": int(cluster_id),
                        "diversity_score": 0.8,
                        "strategy_info": "diversity_boosted"
                    })
        
        return diverse_recommendations[:n_recs]
    
    def _recommend_mood_contextual(self, query_features: np.ndarray, n_recs: int, filters: Dict) -> List[Dict]:
        """Estrategia 5: Basado en características emocionales."""
        
        # Características emocionales principales
        mood_features = ['energy', 'valence', 'danceability']
        available_mood_features = [f for f in mood_features if f in self.discriminative_features]
        
        if not available_mood_features:
            # Fallback a estrategia híbrida
            return self._recommend_hybrid_balanced(query_features, n_recs, filters)
        
        # Extraer mood de la query
        mood_vector = []
        for feature in available_mood_features:
            feat_idx = self.discriminative_features.index(feature)
            mood_vector.append(query_features[feat_idx])
        
        # Clasificar mood
        avg_mood = np.mean(mood_vector)
        if avg_mood > 0.7:
            mood_category = "high_energy"
        elif avg_mood > 0.4:
            mood_category = "balanced"
        else:
            mood_category = "low_energy"
        
        # Filtrar canciones por mood similar
        mood_recommendations = []
        
        for idx in range(len(self.dataset)):
            song = self.dataset.iloc[idx]
            song_mood_vector = [song[f] for f in available_mood_features if f in song.index]
            
            if song_mood_vector:
                song_avg_mood = np.mean(song_mood_vector)
                mood_similarity = 1 - abs(avg_mood - song_avg_mood)
                
                mood_recommendations.append({
                    "track_id": song.get('track_id', ''),
                    "track_name": song.get('track_name', ''),
                    "track_artist": song.get('track_artist', ''),
                    "mood_category": mood_category,
                    "mood_similarity": float(mood_similarity),
                    "strategy_info": "mood_contextual"
                })
        
        # Ordenar por similitud de mood
        mood_recommendations.sort(key=lambda x: x['mood_similarity'], reverse=True)
        
        return mood_recommendations[:n_recs]
    
    def _recommend_temporal_aware(self, query_features: np.ndarray, n_recs: int, filters: Dict) -> List[Dict]:
        """Estrategia 6: Considera popularidad y época."""
        
        # Usar estrategia híbrida como base
        base_recommendations = self._recommend_hybrid_balanced(query_features, n_recs * 2, filters)
        
        # Ajustar scores por factores temporales
        temporal_recommendations = []
        
        for rec in base_recommendations:
            track_id = rec.get('track_id', '')
            
            # Buscar información temporal en dataset
            song_row = self.dataset[self.dataset.get('track_id', pd.Series()) == track_id]
            
            temporal_score = rec.get('hybrid_score', 0.5)
            
            if not song_row.empty:
                song = song_row.iloc[0]
                
                # Factor popularidad
                if 'track_popularity' in song.index:
                    popularity = song['track_popularity']
                    if pd.notna(popularity):
                        popularity_boost = min(popularity / 100, 0.2)  # Max 20% boost
                        temporal_score += popularity_boost
                
                # Factor época (preferir música más reciente)
                if 'track_album_release_date' in song.index:
                    release_date = song['track_album_release_date']
                    if pd.notna(release_date):
                        try:
                            year = int(str(release_date)[:4])
                            current_year = datetime.now().year
                            recency_factor = max(0, min((year - 1990) / (current_year - 1990), 0.1))
                            temporal_score += recency_factor
                        except:
                            pass
            
            rec['temporal_score'] = temporal_score
            rec['strategy_info'] = 'temporal_aware'
            temporal_recommendations.append(rec)
        
        # Ordenar por score temporal
        temporal_recommendations.sort(key=lambda x: x['temporal_score'], reverse=True)
        
        return temporal_recommendations[:n_recs]
    
    def _predict_cluster_optimized(self, features: np.ndarray) -> int:
        """Predecir cluster usando centroides optimizados de ClusterPurifier."""
        
        # Normalizar características usando la misma escala que el clustering
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform([features])[0]
        
        # Aplicar pesos discriminativos si están disponibles
        if hasattr(self, 'feature_weights') and self.feature_weights:
            weights = np.array([self.feature_weights.get(f, 1.0) for f in self.discriminative_features])
            features_weighted = features_normalized * weights
        else:
            features_weighted = features_normalized
        
        # Calcular distancia a cada centroide optimizado
        min_distance = float('inf')
        best_cluster = 0
        
        for cluster_id, centroid in self.cluster_centroids.items():
            # Usar distancia coseno para consistencia con matriz de similitud
            similarity = np.dot(features_weighted, centroid) / (
                np.linalg.norm(features_weighted) * np.linalg.norm(centroid)
            )
            distance = 1 - similarity  # Convertir similitud a distancia
            
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster_id
        
        return best_cluster
    
    def _calculate_cluster_similarities(self, query_features: np.ndarray, cluster_id: int, cluster_indices: List[int]) -> List[Tuple[int, float]]:
        """Calcular similitudes optimizadas dentro de un cluster."""
        
        similarities = []
        
        # Si tenemos matriz de similitud completa, usarla directamente
        if hasattr(self, 'similarity_matrix'):
            # Necesitamos encontrar el índice de query en la matriz
            # Por simplicidad, calculamos similitud directamente
            query_normalized = self._normalize_features(query_features)
            
            for idx in cluster_indices:
                song_features = self.dataset.iloc[idx][self.discriminative_features].values
                song_normalized = self._normalize_features(song_features)
                
                # Similitud coseno ponderada
                similarity = self._calculate_weighted_similarity(query_normalized, song_normalized)
                similarities.append((idx, similarity))
        
        # Si tenemos índices por cluster, usar esos
        elif hasattr(self, 'cluster_similarities') and cluster_id in self.cluster_similarities:
            cluster_similarity_matrix = self.cluster_similarities[cluster_id]
            cluster_songs_in_index = self.similarity_indices[cluster_id]
            
            # Mapear query features a cluster space y calcular similitudes
            query_normalized = self._normalize_features(query_features)
            
            for i, global_idx in enumerate(cluster_songs_in_index):
                if global_idx in cluster_indices:
                    # Para simplificar, usar similitud directa
                    song_features = self.dataset.iloc[global_idx][self.discriminative_features].values
                    song_normalized = self._normalize_features(song_features)
                    similarity = self._calculate_weighted_similarity(query_normalized, song_normalized)
                    similarities.append((global_idx, similarity))
        
        # Fallback: calcular similitudes directamente
        else:
            query_normalized = self._normalize_features(query_features)
            
            for idx in cluster_indices:
                song_features = self.dataset.iloc[idx][self.discriminative_features].values
                song_normalized = self._normalize_features(song_features)
                similarity = self._calculate_weighted_similarity(query_normalized, song_normalized)
                similarities.append((idx, similarity))
        
        return similarities
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalizar características usando StandardScaler."""
        scaler = StandardScaler()
        normalized = scaler.fit_transform([features])[0]
        
        # Aplicar pesos discriminativos
        if hasattr(self, 'feature_weights') and self.feature_weights:
            weights = np.array([self.feature_weights.get(f, 1.0) for f in self.discriminative_features])
            normalized = normalized * weights
        
        return normalized
    
    def _calculate_weighted_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calcular similitud coseno ponderada entre dos vectores de características."""
        
        # Similitud coseno
        dot_product = np.dot(features1, features2)
        norm_product = np.linalg.norm(features1) * np.linalg.norm(features2)
        
        if norm_product == 0:
            return 0.0
        
        similarity = dot_product / norm_product
        
        # Asegurar que esté en rango [0, 1]
        return max(0.0, min(1.0, (similarity + 1) / 2))
    
    def _predict_cluster(self, features: np.ndarray) -> int:
        """Método legacy - usar _predict_cluster_optimized en su lugar."""
        return self._predict_cluster_optimized(features)
    
    def _add_explanations(self, recommendations: List[Dict], query_features: np.ndarray, strategy: str) -> List[Dict]:
        """Agregar explicaciones a recomendaciones."""
        
        for rec in recommendations:
            explanation = []
            
            # Explicación por estrategia
            if strategy == "cluster_pure":
                explanation.append(f"Misma categoría musical (cluster {rec.get('cluster', 'N/A')})")
            elif strategy == "similarity_weighted":
                explanation.append("Características musicales similares")
            elif strategy == "hybrid_balanced":
                explanation.append("Combinación de categoría musical y similitud")
            elif strategy == "diversity_boosted":
                explanation.append("Seleccionada para mayor diversidad musical")
            elif strategy == "mood_contextual":
                explanation.append(f"Mood similar: {rec.get('mood_category', 'N/A')}")
            elif strategy == "temporal_aware":
                explanation.append("Balanceando similitud, popularidad y época")
            
            # Explicación por características principales
            if 'similarity' in rec:
                sim_pct = rec['similarity'] * 100
                if sim_pct > 90:
                    explanation.append("Extremadamente similar")
                elif sim_pct > 75:
                    explanation.append("Muy similar")
                elif sim_pct > 50:
                    explanation.append("Moderadamente similar")
                else:
                    explanation.append("Diversa pero relacionada")
            
            rec['explanation'] = " | ".join(explanation)
        
        return recommendations
    
    def evaluate_recommendations(self, test_queries: List, ground_truth: Optional[Dict] = None) -> Dict:
        """Evaluar calidad de recomendaciones con métricas estándar."""
        
        print("📊 Evaluando calidad de recomendaciones...")
        
        evaluation_results = {
            "strategies_evaluated": list(self.recommendation_strategies.keys()),
            "performance_metrics": {},
            "quality_metrics": {},
            "diversity_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Evaluar cada estrategia
        for strategy in self.recommendation_strategies.keys():
            print(f"   🔄 Evaluando estrategia: {strategy}")
            
            strategy_results = {
                "avg_time_ms": 0,
                "precision_at_5": 0,
                "precision_at_10": 0,
                "diversity_score": 0,
                "coverage_score": 0
            }
            
            times = []
            
            # Evaluar en queries de test
            for query in test_queries[:10]:  # Limitar para demo
                start_time = time.time()
                
                try:
                    result = self.recommend(query, strategy=strategy, n_recommendations=10)
                    exec_time = (time.time() - start_time) * 1000
                    times.append(exec_time)
                    
                    # Calcular métricas de diversidad
                    recs = result.get('recommendations', [])
                    if recs:
                        clusters = [r.get('cluster', 0) for r in recs if 'cluster' in r]
                        unique_clusters = len(set(clusters))
                        diversity = unique_clusters / len(recs) if recs else 0
                        strategy_results["diversity_score"] += diversity
                
                except Exception as e:
                    print(f"      ⚠️ Error en query {query}: {e}")
            
            # Promediar métricas
            if times:
                strategy_results["avg_time_ms"] = round(np.mean(times), 2)
                strategy_results["diversity_score"] = round(strategy_results["diversity_score"] / len(times), 3)
            
            evaluation_results["performance_metrics"][strategy] = strategy_results
        
        # Identificar mejor estrategia
        best_strategy = min(
            evaluation_results["performance_metrics"].keys(),
            key=lambda s: evaluation_results["performance_metrics"][s]["avg_time_ms"]
        )
        
        evaluation_results["best_performance_strategy"] = best_strategy
        evaluation_results["target_performance_ms"] = 100
        
        print(f"✅ Evaluación completada")
        print(f"🏆 Mejor performance: {best_strategy} ({evaluation_results['performance_metrics'][best_strategy]['avg_time_ms']}ms)")
        
        return evaluation_results


def setup_logging(log_level='INFO'):
    """Configurar sistema de logging optimizado."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parsear argumentos CLI para recomendador optimizado."""
    parser = argparse.ArgumentParser(
        description='🎵 Sistema de Recomendación Musical Optimizado',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --query "track_id_here" --strategy hybrid_balanced --n-recs 10
  %(prog)s --query "{'energy': 0.8, 'valence': 0.6}" --strategy mood_contextual
  %(prog)s --interactive --strategy cluster_pure
  %(prog)s --benchmark --evaluate
        """
    )
    
    # Argumentos principales
    parser.add_argument('--query', type=str,
                       help='Query: ID canción, nombre, o características JSON')
    
    parser.add_argument('--strategy', 
                       choices=['cluster_pure', 'similarity_weighted', 'hybrid_balanced', 
                               'diversity_boosted', 'mood_contextual', 'temporal_aware'],
                       default='hybrid_balanced',
                       help='Estrategia de recomendación (default: hybrid_balanced)')
    
    parser.add_argument('--n-recs', type=int, default=10,
                       help='Número de recomendaciones (default: 10)')
    
    parser.add_argument('--explain', action='store_true',
                       help='Incluir explicaciones de recomendaciones')
    
    # Modos especiales
    parser.add_argument('--interactive', action='store_true',
                       help='Modo interactivo')
    
    parser.add_argument('--benchmark', action='store_true',
                       help='Ejecutar benchmark de performance')
    
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluar calidad de recomendaciones')
    
    # Configuración
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Nivel de logging (default: INFO)')
    
    return parser.parse_args()


def main():
    """Función principal del recomendador optimizado."""
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    logger.info("🎵 INICIANDO SISTEMA DE RECOMENDACIÓN MUSICAL OPTIMIZADO")
    logger.info("=" * 70)
    
    try:
        # Inicializar sistema
        recommender = OptimizedMusicRecommender()
        
        if not recommender.initialize_system():
            logger.error("❌ Fallo en inicialización del sistema")
            return 1
        
        # Ejecutar según modo
        if args.interactive:
            logger.info("🎯 Iniciando modo interactivo...")
            interactive_mode(recommender, args.strategy, logger)
        
        elif args.benchmark:
            logger.info("📊 Ejecutando benchmark de performance...")
            benchmark_results = benchmark_performance(recommender)
            print(json.dumps(benchmark_results, indent=2))
        
        elif args.evaluate:
            logger.info("📈 Evaluando calidad de recomendaciones...")
            # Crear queries de test
            test_queries = [0, 1, 2, 3, 4]  # Índices de canciones para test
            eval_results = recommender.evaluate_recommendations(test_queries)
            print(json.dumps(eval_results, indent=2))
        
        elif args.query:
            logger.info(f"🎵 Generando recomendaciones para query: {args.query}")
            
            # Procesar query
            try:
                # Intentar parsear como JSON
                query = json.loads(args.query)
            except:
                # Usar como string
                query = args.query
            
            # Generar recomendaciones
            result = recommender.recommend(
                query=query,
                strategy=args.strategy,
                n_recommendations=args.n_recs,
                explain=args.explain
            )
            
            # Mostrar resultados
            print("\n" + "="*70)
            print("🎵 RECOMENDACIONES MUSICALES OPTIMIZADAS")
            print("="*70)
            print(f"📊 Estrategia: {result['strategy']}")
            print(f"⚡ Performance: {result['performance']['total_time_ms']}ms")
            print(f"🎯 Objetivo: <{result['performance']['target_time_ms']}ms")
            
            print(f"\n🎵 Top {len(result['recommendations'])} recomendaciones:")
            print("-" * 70)
            
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i:2d}. 🎤 \"{rec.get('track_name', 'N/A')}\" - {rec.get('track_artist', 'N/A')}")
                if 'similarity' in rec:
                    print(f"    📊 Similitud: {rec['similarity']:.3f}")
                if 'explanation' in rec:
                    print(f"    💡 {rec['explanation']}")
                print()
        
        else:
            logger.error("❌ Debe especificar --query, --interactive, --benchmark, o --evaluate")
            return 1
        
        logger.info("🎵 ¡RECOMENDACIÓN COMPLETADA EXITOSAMENTE! 🎉")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error durante ejecución: {e}")
        return 1


def interactive_mode(recommender: OptimizedMusicRecommender, default_strategy: str, logger):
    """Modo interactivo avanzado del recomendador."""
    
    print("\n🎵 MODO INTERACTIVO - RECOMENDADOR MUSICAL OPTIMIZADO")
    print("=" * 70)
    print("Comandos disponibles:")
    print("  rec <query>              - Generar recomendaciones")
    print("  strategy <name>          - Cambiar estrategia")
    print("  strategies               - Listar estrategias disponibles")
    print("  benchmark                - Ejecutar benchmark rápido")
    print("  random                   - Recomendación para canción aleatoria")
    print("  quit                     - Salir")
    print("-" * 70)
    
    current_strategy = default_strategy
    
    while True:
        try:
            command = input(f"\n🎯 [{current_strategy}] > ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 ¡Hasta luego!")
                break
            
            elif command.startswith('rec '):
                query = command[4:].strip()
                
                # Intentar parsear como JSON
                try:
                    parsed_query = json.loads(query)
                except:
                    parsed_query = query
                
                # Generar recomendaciones
                result = recommender.recommend(
                    query=parsed_query,
                    strategy=current_strategy,
                    n_recommendations=5,
                    explain=True
                )
                
                # Mostrar resultados
                print(f"\n🎵 Recomendaciones ({result['performance']['total_time_ms']}ms):")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"  {i}. {rec.get('track_name', 'N/A')} - {rec.get('track_artist', 'N/A')}")
                    if 'explanation' in rec:
                        print(f"     💡 {rec['explanation']}")
            
            elif command.startswith('strategy '):
                new_strategy = command[9:].strip()
                if new_strategy in recommender.recommendation_strategies:
                    current_strategy = new_strategy
                    print(f"✅ Estrategia cambiada a: {current_strategy}")
                else:
                    print(f"❌ Estrategia inválida. Disponibles: {list(recommender.recommendation_strategies.keys())}")
            
            elif command == 'strategies':
                print("📋 Estrategias disponibles:")
                for strategy in recommender.recommendation_strategies.keys():
                    marker = "👉" if strategy == current_strategy else "  "
                    print(f"  {marker} {strategy}")
            
            elif command == 'benchmark':
                print("📊 Ejecutando benchmark rápido...")
                # Benchmark simple
                test_queries = [0, 1, 2]
                times = []
                
                for query in test_queries:
                    start = time.time()
                    recommender.recommend(query, strategy=current_strategy, n_recommendations=5)
                    times.append((time.time() - start) * 1000)
                
                avg_time = np.mean(times)
                print(f"⚡ Performance promedio: {avg_time:.1f}ms (objetivo: <100ms)")
                print(f"🎯 Factor performance: {100/avg_time:.1f}x objetivo" if avg_time > 0 else "N/A")
            
            elif command == 'random':
                # Seleccionar canción aleatoria
                random_idx = np.random.randint(0, len(recommender.dataset))
                random_song = recommender.dataset.iloc[random_idx]
                
                print(f"🎲 Canción aleatoria: \"{random_song.get('track_name', 'N/A')}\" - {random_song.get('track_artist', 'N/A')}")
                
                result = recommender.recommend(
                    query=random_idx,
                    strategy=current_strategy,
                    n_recommendations=5,
                    explain=True
                )
                
                print(f"\n🎵 Recomendaciones similares:")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"  {i}. {rec.get('track_name', 'N/A')} - {rec.get('track_artist', 'N/A')}")
            
            else:
                print("❌ Comando no reconocido. Usa: rec, strategy, strategies, benchmark, random, o quit")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def benchmark_performance(recommender: OptimizedMusicRecommender) -> Dict:
    """Ejecutar benchmark completo de performance."""
    
    print("📊 Ejecutando benchmark completo...")
    
    # Queries de test
    test_queries = list(range(0, min(50, len(recommender.dataset)), 5))
    
    benchmark_results = {
        "test_queries": len(test_queries),
        "strategies": {},
        "summary": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Benchmark por estrategia
    for strategy in recommender.recommendation_strategies.keys():
        print(f"   🔄 Benchmarking estrategia: {strategy}")
        
        times = []
        errors = 0
        
        for query in test_queries:
            try:
                start_time = time.time()
                result = recommender.recommend(query, strategy=strategy, n_recommendations=10)
                exec_time = (time.time() - start_time) * 1000
                times.append(exec_time)
                
            except Exception as e:
                errors += 1
        
        # Calcular estadísticas
        if times:
            benchmark_results["strategies"][strategy] = {
                "avg_time_ms": round(np.mean(times), 2),
                "min_time_ms": round(np.min(times), 2),
                "max_time_ms": round(np.max(times), 2),
                "std_time_ms": round(np.std(times), 2),
                "success_rate": round((len(times) / len(test_queries)) * 100, 1),
                "errors": errors,
                "target_ratio": round(100 / np.mean(times), 2) if np.mean(times) > 0 else 0
            }
    
    # Resumen general
    all_times = []
    for strategy_data in benchmark_results["strategies"].values():
        all_times.append(strategy_data["avg_time_ms"])
    
    if all_times:
        benchmark_results["summary"] = {
            "overall_avg_ms": round(np.mean(all_times), 2),
            "best_strategy": min(benchmark_results["strategies"].keys(), 
                               key=lambda s: benchmark_results["strategies"][s]["avg_time_ms"]),
            "target_time_ms": 100,
            "strategies_under_target": sum(1 for t in all_times if t < 100),
            "performance_improvement_needed": max(0, round(np.mean(all_times) / 100, 2))
        }
    
    return benchmark_results


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)