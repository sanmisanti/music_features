#!/usr/bin/env python3
"""
Motor de Recomendaciones Musicales Híbrido
Sistema de recomendación basado en similitud vectorial multimodal
Validado científicamente mediante experimentación FASE 3

Uso:
    from music_recommender import HybridMusicRecommender
    recommender = HybridMusicRecommender()
    recommendations = recommender.recommend(track_id="ejemplo", n_recommendations=10)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import time
from load_system import MusicDataLoader

class HybridMusicRecommender:
    """
    Motor de recomendaciones musicales híbrido
    Combina similitud musical (12D) y semántica (384D) con pesos validados científicamente
    """
    
    def __init__(self, system_dir: str = None, precompute_similarities: bool = False, 
                 custom_musical_weight: float = None, custom_semantic_weight: float = None):
        """
        Inicializa el motor de recomendaciones
        
        Args:
            system_dir: Directorio del sistema (None para auto-detectar)
            precompute_similarities: Pre-calcular matrices de similitud (mejora velocidad)
            custom_musical_weight: Peso personalizado para componente musical (0.0-1.0)
            custom_semantic_weight: Peso personalizado para componente semántico (0.0-1.0)
        """
        # Cargar datos del sistema
        self.loader = MusicDataLoader(system_dir)
        
        # Configurar pesos híbridos
        if custom_musical_weight is not None and custom_semantic_weight is not None:
            # Validar pesos personalizados
            if not (0.0 <= custom_musical_weight <= 1.0 and 0.0 <= custom_semantic_weight <= 1.0):
                raise ValueError("Los pesos deben estar en rango 0.0-1.0")
            
            total_weight = custom_musical_weight + custom_semantic_weight
            if abs(total_weight - 1.0) > 0.001:
                raise ValueError(f"Los pesos deben sumar 1.0, suma actual: {total_weight:.3f}")
            
            self.musical_weight = custom_musical_weight
            self.semantic_weight = custom_semantic_weight
            print(f"🎯 HybridMusicRecommender inicializado con pesos personalizados")
            print(f"   Pesos: Musical {self.musical_weight:.2f}, Semántico {self.semantic_weight:.2f}")
        else:
            # Cargar configuración de pesos validados FASE 3
            config = self.loader.get_config()
            weights = config['recommendation_weights']
            self.musical_weight = weights['musical_weight']
            self.semantic_weight = weights['semantic_weight']
            print(f"🎯 HybridMusicRecommender inicializado con pesos FASE 3")
            print(f"   Pesos: Musical {self.musical_weight:.2f}, Semántico {self.semantic_weight:.2f}")
        
        # Cache para matrices de similitud pre-calculadas
        self._similarity_cache = {}
        
        if precompute_similarities:
            self._precompute_similarity_matrices()
    
    def _precompute_similarity_matrices(self):
        """Pre-calcula matrices de similitud completas para velocidad máxima"""
        print("⚡ Pre-calculando matrices de similitud...")
        
        start_time = time.time()
        
        # Cargar vectores
        musical_vectors = self.loader.get_musical_vectors()
        semantic_vectors = self.loader.get_semantic_vectors()
        
        # Calcular matrices de similitud completas
        print("   🎵 Calculando similitudes musicales...")
        self._similarity_cache['musical'] = cosine_similarity(musical_vectors)
        
        print("   🧠 Calculando similitudes semánticas...")
        self._similarity_cache['semantic'] = cosine_similarity(semantic_vectors)
        
        elapsed = time.time() - start_time
        print(f"   ✅ Matrices pre-calculadas en {elapsed:.2f}s")
    
    def _get_song_index(self, track_id: str) -> Optional[int]:
        """
        Obtiene el índice de una canción por track_id
        
        Args:
            track_id: ID único de la canción
            
        Returns:
            Índice en los arrays o None si no se encuentra
        """
        track_ids = self.loader.get_track_ids()
        indices = np.where(track_ids == track_id)[0]
        return int(indices[0]) if len(indices) > 0 else None
    
    def _calculate_similarities(self, input_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula similitudes musical y semántica para una canción
        
        Args:
            input_idx: Índice de la canción de entrada
            
        Returns:
            Tupla (musical_similarities, semantic_similarities)
        """
        if 'musical' in self._similarity_cache and 'semantic' in self._similarity_cache:
            # Usar matrices pre-calculadas
            musical_sims = self._similarity_cache['musical'][input_idx]
            semantic_sims = self._similarity_cache['semantic'][input_idx]
        else:
            # Calcular on-the-fly
            musical_vectors = self.loader.get_musical_vectors()
            semantic_vectors = self.loader.get_semantic_vectors()
            
            input_musical = musical_vectors[input_idx].reshape(1, -1)
            input_semantic = semantic_vectors[input_idx].reshape(1, -1)
            
            musical_sims = cosine_similarity(input_musical, musical_vectors)[0]
            semantic_sims = cosine_similarity(input_semantic, semantic_vectors)[0]
        
        return musical_sims, semantic_sims
    
    def recommend(self, track_id: str, n_recommendations: int = 10, 
                  exclude_same_artist: bool = False) -> List[Dict]:
        """
        Genera recomendaciones híbridas para una canción
        
        Args:
            track_id: ID de la canción de entrada
            n_recommendations: Número de recomendaciones a generar
            exclude_same_artist: Excluir canciones del mismo artista
            
        Returns:
            Lista de diccionarios con información de recomendaciones ordenadas por similitud
        """
        start_time = time.time()
        
        # Verificar que la canción existe
        input_idx = self._get_song_index(track_id)
        if input_idx is None:
            return [{'error': f'Track ID {track_id} no encontrado en el dataset'}]
        
        # Obtener información de la canción de entrada
        input_song = self.loader.get_song_by_track_id(track_id)
        if not input_song:
            return [{'error': f'No se pudo cargar información para {track_id}'}]
        
        # Calcular similitudes
        musical_sims, semantic_sims = self._calculate_similarities(input_idx)
        
        # Fusión híbrida con pesos validados FASE 3
        hybrid_scores = (self.musical_weight * musical_sims + 
                        self.semantic_weight * semantic_sims)
        
        # Excluir la canción de entrada
        hybrid_scores[input_idx] = -1.0
        
        # Filtrar mismo artista si se solicita
        if exclude_same_artist:
            metadata_df = self.loader.get_songs_metadata()
            input_artist = input_song['artist_name']
            
            # Encontrar índices del mismo artista
            same_artist_tracks = metadata_df[
                metadata_df['artist_name'] == input_artist
            ]['track_id'].values
            
            track_ids_array = self.loader.get_track_ids()
            for same_track_id in same_artist_tracks:
                same_idx = self._get_song_index(same_track_id)
                if same_idx is not None:
                    hybrid_scores[same_idx] = -1.0
        
        # Obtener top recomendaciones
        top_indices = np.argsort(hybrid_scores)[::-1][:n_recommendations]
        track_ids_array = self.loader.get_track_ids()
        
        # Construir lista de recomendaciones
        recommendations = []
        for rank, idx in enumerate(top_indices):
            if hybrid_scores[idx] < 0:  # Skip excluidas
                continue
                
            rec_track_id = track_ids_array[idx]
            rec_song = self.loader.get_song_by_track_id(rec_track_id)
            
            if rec_song:
                # Calcular métricas adicionales
                musical_contribution = self.musical_weight * musical_sims[idx]
                semantic_contribution = self.semantic_weight * semantic_sims[idx]
                
                # Obtener metadatos musicales adicionales del dataset
                metadata_full = rec_song.get('metadata', {})
                
                recommendation = {
                    'rank': rank + 1,
                    'track_id': rec_track_id,
                    'track_name': rec_song['track_name'],
                    'artist_name': rec_song['artist_name'],
                    'genre': rec_song['genre'],
                    'musical_cluster': rec_song['musical_cluster'],
                    'semantic_cluster': rec_song['semantic_cluster'],
                    'scores': {
                        'hybrid': float(hybrid_scores[idx]),
                        'musical': float(musical_sims[idx]),
                        'semantic': float(semantic_sims[idx]),
                        'musical_contribution': float(musical_contribution),
                        'semantic_contribution': float(semantic_contribution)
                    },
                    'musical_features': {
                        'danceability': metadata_full.get('danceability', 0.0),
                        'energy': metadata_full.get('energy', 0.0),
                        'valence': metadata_full.get('valence', 0.0),
                        'acousticness': metadata_full.get('acousticness', 0.0),
                        'instrumentalness': metadata_full.get('instrumentalness', 0.0),
                        'tempo': metadata_full.get('tempo', 0.0)
                    },
                    'additional_info': {
                        'popularity': metadata_full.get('track_popularity', 0),
                        'duration_ms': metadata_full.get('duration_ms', 0),
                        'album_name': metadata_full.get('track_album_name', 'Unknown'),
                        'release_date': metadata_full.get('track_album_release_date', 'Unknown')
                    }
                }
                recommendations.append(recommendation)
        
        # Calcular tiempo de ejecución
        elapsed_time = time.time() - start_time
        
        # Añadir metadatos de la recomendación
        result = {
            'input_song': {
                'track_id': track_id,
                'track_name': input_song['track_name'],
                'artist_name': input_song['artist_name'],
                'genre': input_song['genre'],
                'musical_cluster': input_song['musical_cluster'],
                'semantic_cluster': input_song['semantic_cluster']
            },
            'recommendations': recommendations[:n_recommendations],
            'metadata': {
                'n_requested': n_recommendations,
                'n_returned': len(recommendations[:n_recommendations]),
                'execution_time_ms': round(elapsed_time * 1000, 2),
                'weights_used': {
                    'musical': self.musical_weight,
                    'semantic': self.semantic_weight
                },
                'exclude_same_artist': exclude_same_artist
            }
        }
        
        return result
    
    def recommend_by_name(self, song_name: str, artist_name: str = None, 
                         n_recommendations: int = 10) -> List[Dict]:
        """
        Genera recomendaciones buscando canción por nombre
        
        Args:
            song_name: Nombre de la canción
            artist_name: Nombre del artista (opcional, para desambiguar)
            n_recommendations: Número de recomendaciones
            
        Returns:
            Resultado de recomendación o error si no se encuentra
        """
        # Buscar canciones por nombre
        matches = self.loader.find_songs_by_name(song_name, limit=20)
        
        if not matches:
            return [{'error': f'No se encontraron canciones con nombre "{song_name}"'}]
        
        # Filtrar por artista si se especifica
        if artist_name:
            artist_matches = [
                song for song in matches 
                if artist_name.lower() in song['artist_name'].lower()
            ]
            if artist_matches:
                matches = artist_matches
        
        # Usar la primera coincidencia
        selected_song = matches[0]
        
        # Si hay múltiples coincidencias, mostrar información
        if len(matches) > 1:
            print(f"ℹ️  Encontradas {len(matches)} coincidencias, usando:")
            print(f"   {selected_song['track_name']} - {selected_song['artist_name']}")
        
        return self.recommend(selected_song['track_id'], n_recommendations)
    
    def get_similar_by_clusters(self, track_id: str, cluster_type: str = 'both', 
                               n_recommendations: int = 10) -> List[Dict]:
        """
        Obtiene recomendaciones basadas únicamente en pertenencia a clusters
        
        Args:
            track_id: ID de la canción de entrada
            cluster_type: 'musical', 'semantic', o 'both'
            n_recommendations: Número de recomendaciones
            
        Returns:
            Lista de canciones del mismo cluster o clusters
        """
        input_song = self.loader.get_song_by_track_id(track_id)
        if not input_song:
            return [{'error': f'Track ID {track_id} no encontrado'}]
        
        # Obtener canciones candidatas según cluster_type
        candidates = []
        
        if cluster_type in ['musical', 'both']:
            musical_cluster_songs = self.loader.get_songs_by_cluster(
                'musical', input_song['musical_cluster']
            )
            candidates.extend(musical_cluster_songs)
        
        if cluster_type in ['semantic', 'both']:
            semantic_cluster_songs = self.loader.get_songs_by_cluster(
                'semantic', input_song['semantic_cluster']
            )
            candidates.extend(semantic_cluster_songs)
        
        # Remover duplicados y la canción de entrada
        seen_track_ids = set()
        unique_candidates = []
        
        for song in candidates:
            if song['track_id'] not in seen_track_ids and song['track_id'] != track_id:
                seen_track_ids.add(song['track_id'])
                unique_candidates.append(song)
        
        # Limitar a n_recommendations
        cluster_recommendations = unique_candidates[:n_recommendations]
        
        return {
            'input_song': input_song,
            'cluster_type': cluster_type,
            'recommendations': cluster_recommendations,
            'metadata': {
                'n_requested': n_recommendations,
                'n_returned': len(cluster_recommendations),
                'total_candidates': len(unique_candidates)
            }
        }
    
    def benchmark_performance(self, n_tests: int = 100) -> Dict:
        """
        Ejecuta benchmark de performance del sistema
        
        Args:
            n_tests: Número de pruebas de recomendación a ejecutar
            
        Returns:
            Estadísticas de performance
        """
        print(f"⏱️  Ejecutando benchmark con {n_tests} pruebas...")
        
        track_ids = self.loader.get_track_ids()
        test_track_ids = np.random.choice(track_ids, size=min(n_tests, len(track_ids)), replace=False)
        
        execution_times = []
        successful_tests = 0
        
        for i, test_track_id in enumerate(test_track_ids):
            try:
                start_time = time.time()
                result = self.recommend(test_track_id, n_recommendations=10)
                elapsed = time.time() - start_time
                
                if not isinstance(result, list) or 'error' not in str(result):
                    execution_times.append(elapsed * 1000)  # Convert to ms
                    successful_tests += 1
                
                if (i + 1) % 20 == 0:
                    print(f"   Progreso: {i + 1}/{n_tests}")
                    
            except Exception as e:
                print(f"   Error en test {i+1}: {e}")
        
        if execution_times:
            stats = {
                'total_tests': n_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / n_tests,
                'execution_times_ms': {
                    'mean': np.mean(execution_times),
                    'median': np.median(execution_times),
                    'min': np.min(execution_times),
                    'max': np.max(execution_times),
                    'std': np.std(execution_times)
                },
                'performance_target_met': np.mean(execution_times) < 100.0  # Target <100ms
            }
            
            print(f"📊 Benchmark completado:")
            print(f"   Tests exitosos: {successful_tests}/{n_tests} ({stats['success_rate']:.1%})")
            print(f"   Tiempo promedio: {stats['execution_times_ms']['mean']:.2f}ms")
            print(f"   Target <100ms: {'✅' if stats['performance_target_met'] else '❌'}")
            
            return stats
        else:
            return {'error': 'No se pudieron ejecutar tests exitosamente'}


if __name__ == "__main__":
    # Test básico del recomendador
    print("🧪 Ejecutando test del motor de recomendaciones...")
    
    try:
        recommender = HybridMusicRecommender()
        
        # Obtener una canción de ejemplo para test
        loader = MusicDataLoader()
        track_ids = loader.get_track_ids()
        
        if len(track_ids) > 0:
            test_track_id = track_ids[0]
            print(f"🎵 Generando recomendaciones para: {test_track_id}")
            
            # Test de recomendación
            start_time = time.time()
            result = recommender.recommend(test_track_id, n_recommendations=5)
            elapsed = time.time() - start_time
            
            if isinstance(result, dict) and 'recommendations' in result:
                print(f"✅ Test exitoso - {len(result['recommendations'])} recomendaciones en {elapsed*1000:.2f}ms")
                print(f"   Canción entrada: {result['input_song']['track_name']} - {result['input_song']['artist_name']}")
                
                if result['recommendations']:
                    top_rec = result['recommendations'][0]
                    print(f"   Top recomendación: {top_rec['track_name']} - {top_rec['artist_name']}")
                    print(f"   Score híbrido: {top_rec['scores']['hybrid']:.3f}")
            else:
                print(f"❌ Error en test: {result}")
                
        else:
            print("❌ No hay canciones disponibles para test")
            
    except Exception as e:
        print(f"❌ Error en test: {e}")
        raise