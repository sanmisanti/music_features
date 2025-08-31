#!/usr/bin/env python3
"""
Sistema de Carga para Componentes de Recomendaciones Musicales
Proporciona acceso centralizado a vectores, clusters y metadatos

Uso:
    from load_system import MusicDataLoader
    loader = MusicDataLoader()
    vectors = loader.get_semantic_vectors()
    clusters = loader.get_musical_clusters()
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class MusicDataLoader:
    """
    Cargador centralizado de datos del sistema de recomendaciones musicales
    Maneja vectores, clusters, metadatos y configuraciones validadas científicamente
    """
    
    def __init__(self, system_dir: str = None):
        """
        Inicializa el cargador de datos
        
        Args:
            system_dir: Directorio del sistema (None para auto-detectar)
        """
        if system_dir is None:
            # Auto-detectar directorio del sistema
            self.system_dir = Path(__file__).parent.parent
        else:
            self.system_dir = Path(system_dir)
            
        # Directorios específicos
        self.data_dir = self.system_dir / "data"
        self.clusters_dir = self.system_dir / "clusters"
        self.config_dir = self.system_dir / "config"
        
        # Cache de datos cargados (para evitar re-cargas)
        self._cache = {}
        self._config = None
        
        print(f"🎵 MusicDataLoader inicializado")
        print(f"   Directorio sistema: {self.system_dir}")
    
    def get_config(self) -> Dict:
        """Obtiene la configuración del sistema"""
        if self._config is None:
            config_path = self.config_dir / "system_config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        return self._config
    
    def get_semantic_vectors(self) -> np.ndarray:
        """
        Obtiene vectores semánticos (embeddings BERT)
        
        Returns:
            Array (7811, 384) con embeddings normalizados L2
        """
        if 'semantic_vectors' not in self._cache:
            semantic_path = self.data_dir / "semantic_embeddings.npy"
            self._cache['semantic_vectors'] = np.load(semantic_path)
            print(f"✅ Vectores semánticos cargados: {self._cache['semantic_vectors'].shape}")
        
        return self._cache['semantic_vectors']
    
    def get_musical_vectors(self) -> np.ndarray:
        """
        Obtiene vectores musicales (características Spotify normalizadas)
        
        Returns:
            Array (7811, 12) con características StandardScaler
        """
        if 'musical_vectors' not in self._cache:
            musical_path = self.data_dir / "musical_features_normalized.npy"
            self._cache['musical_vectors'] = np.load(musical_path)
            print(f"✅ Vectores musicales cargados: {self._cache['musical_vectors'].shape}")
        
        return self._cache['musical_vectors']
    
    def get_track_ids(self) -> np.ndarray:
        """
        Obtiene IDs de canciones para alineación
        
        Returns:
            Array (7811,) con track_ids únicos
        """
        if 'track_ids' not in self._cache:
            ids_path = self.data_dir / "track_ids.npy"
            self._cache['track_ids'] = np.load(ids_path, allow_pickle=True)
            print(f"✅ Track IDs cargados: {len(self._cache['track_ids'])} canciones")
        
        return self._cache['track_ids']
    
    def get_musical_clusters(self) -> np.ndarray:
        """
        Obtiene asignaciones de clusters musicales óptimos
        
        Returns:
            Array (7811,) con valores 0-9 (K=10 clusters)
        """
        if 'musical_clusters' not in self._cache:
            clusters_path = self.clusters_dir / "musical_clusters_k10.npy"
            self._cache['musical_clusters'] = np.load(clusters_path)
            unique_clusters = len(np.unique(self._cache['musical_clusters']))
            print(f"✅ Clusters musicales cargados: {unique_clusters} clusters únicos")
        
        return self._cache['musical_clusters']
    
    def get_semantic_clusters(self) -> np.ndarray:
        """
        Obtiene asignaciones de clusters semánticos óptimos
        
        Returns:
            Array (7811,) con valores 0-5 (K=6 clusters)
        """
        if 'semantic_clusters' not in self._cache:
            clusters_path = self.clusters_dir / "semantic_clusters_k6.npy"
            self._cache['semantic_clusters'] = np.load(clusters_path)
            unique_clusters = len(np.unique(self._cache['semantic_clusters']))
            print(f"✅ Clusters semánticos cargados: {unique_clusters} clusters únicos")
        
        return self._cache['semantic_clusters']
    
    def get_songs_metadata(self) -> pd.DataFrame:
        """
        Obtiene metadatos de canciones
        
        Returns:
            DataFrame con información de canciones (track_name, artist_name, genre, etc.)
        """
        if 'metadata_df' not in self._cache:
            metadata_path = self.data_dir / "songs_metadata.csv"
            self._cache['metadata_df'] = pd.read_csv(metadata_path)
            print(f"✅ Metadatos cargados: {len(self._cache['metadata_df'])} registros")
        
        return self._cache['metadata_df']
    
    def get_song_by_track_id(self, track_id: str) -> Optional[Dict]:
        """
        Obtiene información completa de una canción por track_id
        
        Args:
            track_id: ID único de la canción
            
        Returns:
            Dict con información completa o None si no se encuentra
        """
        try:
            # Buscar índice
            track_ids = self.get_track_ids()
            idx = np.where(track_ids == track_id)[0]
            
            if len(idx) == 0:
                return None
            
            idx = idx[0]
            
            # Obtener datos de todas las fuentes
            metadata_df = self.get_songs_metadata()
            song_row = metadata_df[metadata_df['track_id'] == track_id].iloc[0]
            
            musical_clusters = self.get_musical_clusters()
            semantic_clusters = self.get_semantic_clusters()
            
            return {
                'track_id': track_id,
                'index': int(idx),
                'track_name': song_row['track_name'],
                'artist_name': song_row['artist_name'],
                'genre': song_row.get('playlist_genre', 'unknown'),
                'musical_cluster': int(musical_clusters[idx]),
                'semantic_cluster': int(semantic_clusters[idx]),
                'metadata': song_row.to_dict()
            }
            
        except Exception as e:
            print(f"Error obteniendo canción {track_id}: {e}")
            return None
    
    def get_songs_by_cluster(self, cluster_type: str, cluster_id: int) -> List[Dict]:
        """
        Obtiene todas las canciones de un cluster específico
        
        Args:
            cluster_type: 'musical' o 'semantic'
            cluster_id: ID del cluster
            
        Returns:
            Lista de diccionarios con información de canciones
        """
        if cluster_type == 'musical':
            clusters = self.get_musical_clusters()
            max_cluster = 9  # K=10, rango 0-9
        elif cluster_type == 'semantic':
            clusters = self.get_semantic_clusters()
            max_cluster = 5  # K=6, rango 0-5
        else:
            raise ValueError("cluster_type debe ser 'musical' o 'semantic'")
        
        if not (0 <= cluster_id <= max_cluster):
            raise ValueError(f"cluster_id debe estar en rango 0-{max_cluster}")
        
        # Encontrar todas las canciones del cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        track_ids = self.get_track_ids()
        
        songs = []
        for idx in cluster_indices:
            song_info = self.get_song_by_track_id(track_ids[idx])
            if song_info:
                songs.append(song_info)
        
        return songs
    
    def get_system_stats(self) -> Dict:
        """
        Obtiene estadísticas del sistema cargado
        
        Returns:
            Dict con estadísticas completas
        """
        config = self.get_config()
        track_ids = self.get_track_ids()
        musical_clusters = self.get_musical_clusters()
        semantic_clusters = self.get_semantic_clusters()
        metadata_df = self.get_songs_metadata()
        
        # Distribución de clusters
        musical_distribution = {int(k): int(v) for k, v in 
                              zip(*np.unique(musical_clusters, return_counts=True))}
        semantic_distribution = {int(k): int(v) for k, v in 
                               zip(*np.unique(semantic_clusters, return_counts=True))}
        
        # Distribución de géneros si está disponible
        genre_distribution = {}
        if 'playlist_genre' in metadata_df.columns:
            genre_counts = metadata_df['playlist_genre'].value_counts()
            genre_distribution = genre_counts.to_dict()
        
        return {
            'dataset_size': len(track_ids),
            'vector_dimensions': {
                'semantic': self.get_semantic_vectors().shape[1],
                'musical': self.get_musical_vectors().shape[1]
            },
            'cluster_counts': {
                'musical': len(np.unique(musical_clusters)),
                'semantic': len(np.unique(semantic_clusters))
            },
            'cluster_distributions': {
                'musical': musical_distribution,
                'semantic': semantic_distribution
            },
            'genre_distribution': genre_distribution,
            'recommendation_weights': config.get('recommendation_weights', {}),
            'system_config': config.get('dataset_info', {})
        }
    
    def find_songs_by_name(self, song_name: str, limit: int = 10) -> List[Dict]:
        """
        Busca canciones por nombre (búsqueda parcial)
        
        Args:
            song_name: Nombre o parte del nombre de la canción
            limit: Máximo número de resultados
            
        Returns:
            Lista de canciones que coinciden
        """
        metadata_df = self.get_songs_metadata()
        
        # Búsqueda case-insensitive
        matches = metadata_df[
            metadata_df['track_name'].str.contains(song_name, case=False, na=False)
        ].head(limit)
        
        results = []
        for _, row in matches.iterrows():
            song_info = self.get_song_by_track_id(row['track_id'])
            if song_info:
                results.append(song_info)
        
        return results
    
    def clear_cache(self):
        """Limpia la cache de datos cargados"""
        self._cache = {}
        self._config = None
        print("🧹 Cache limpiada")


# Función de conveniencia para instanciación rápida
def load_music_system(system_dir: str = None) -> MusicDataLoader:
    """
    Instancia rápida del cargador de datos
    
    Args:
        system_dir: Directorio del sistema (None para auto-detectar)
        
    Returns:
        Instancia de MusicDataLoader
    """
    return MusicDataLoader(system_dir)


if __name__ == "__main__":
    # Test básico del cargador
    print("🧪 Ejecutando test del cargador de datos...")
    
    try:
        loader = MusicDataLoader()
        
        # Cargar componentes principales
        semantic = loader.get_semantic_vectors()
        musical = loader.get_musical_vectors()
        track_ids = loader.get_track_ids()
        
        print(f"✅ Test exitoso - Sistema con {len(track_ids)} canciones cargado")
        
        # Mostrar estadísticas
        stats = loader.get_system_stats()
        print(f"📊 Clusters musicales: {stats['cluster_counts']['musical']}")
        print(f"📊 Clusters semánticos: {stats['cluster_counts']['semantic']}")
        
        # Test de búsqueda
        if len(track_ids) > 0:
            sample_song = loader.get_song_by_track_id(track_ids[0])
            if sample_song:
                print(f"🎵 Canción ejemplo: {sample_song['track_name']} - {sample_song['artist_name']}")
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        raise