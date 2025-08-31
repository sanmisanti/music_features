#!/usr/bin/env python3
"""
Script de Exportación de Catálogo Musical
Genera archivo CSV legible con información completa de canciones del sistema

Uso:
    python export_songs_catalog.py
    python export_songs_catalog.py --output custom_catalog.csv
    python export_songs_catalog.py --include_clusters
    python export_songs_catalog.py --sample 100

Propósito:
    - Exportar información legible de canciones para análisis manual
    - Generar catálogo completo con metadatos musicales
    - Facilitar validación y exploración del dataset
    - Proporcionar vista consolidada para investigación
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import sys

# Agregar directorio de scripts al path para importaciones
script_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(script_dir))

try:
    from load_system import MusicDataLoader
except ImportError as e:
    print(f"❌ Error importando load_system: {e}")
    print("   Asegúrate de que load_system.py esté en el directorio 'scripts/'")
    sys.exit(1)

class SongsCatalogExporter:
    """
    Exportador de catálogo musical para análisis y validación
    """
    
    def __init__(self, system_dir: str = None):
        """
        Inicializa el exportador de catálogo
        
        Args:
            system_dir: Directorio del sistema (None para auto-detectar)
        """
        self.system_dir = Path(__file__).parent.parent if system_dir is None else Path(system_dir)
        
        try:
            self.loader = MusicDataLoader(system_dir)
            print("🎵 SongsCatalogExporter inicializado exitosamente")
            
            # Cargar estadísticas del sistema
            stats = self.loader.get_system_stats()
            print(f"   Dataset: {stats['dataset_size']} canciones disponibles")
            print(f"   Clusters musicales: K={stats['cluster_counts']['musical']}")
            print(f"   Clusters semánticos: K={stats['cluster_counts']['semantic']}")
            
        except Exception as e:
            print(f"❌ Error inicializando exportador: {e}")
            raise
    
    def export_basic_catalog(self, output_path: str, sample_size: int = None) -> bool:
        """
        Exporta catálogo básico con información esencial
        
        Args:
            output_path: Ruta del archivo CSV de salida
            sample_size: Número de canciones a incluir (None para todas)
            
        Returns:
            True si la exportación fue exitosa
        """
        try:
            print(f"\n📊 Generando catálogo básico de canciones...")
            
            # Obtener datos del sistema
            track_ids = self.loader.get_track_ids()
            metadata_df = self.loader.get_songs_metadata()
            
            print(f"   Procesando {len(track_ids)} canciones del sistema...")
            
            # Crear lista de canciones con información básica
            songs_data = []
            processed_count = 0
            errors_count = 0
            
            # Aplicar muestreo si se especifica
            if sample_size and sample_size < len(track_ids):
                indices = np.random.choice(len(track_ids), size=sample_size, replace=False)
                track_ids_sample = track_ids[indices]
                print(f"   Muestreo aplicado: {sample_size} canciones seleccionadas aleatoriamente")
            else:
                track_ids_sample = track_ids
            
            for i, track_id in enumerate(track_ids_sample):
                try:
                    song_info = self.loader.get_song_by_track_id(track_id)
                    
                    if song_info and 'metadata' in song_info:
                        metadata = song_info['metadata']
                        
                        # Extraer información básica
                        song_entry = {
                            'track_id': track_id,
                            'track_name': song_info.get('track_name', 'Unknown'),
                            'artist_name': song_info.get('artist_name', 'Unknown Artist'),
                            'genre': song_info.get('genre', 'Unknown'),
                            'popularity': metadata.get('track_popularity', 0),
                            'album_name': metadata.get('track_album_name', 'Unknown Album'),
                            'release_date': metadata.get('track_album_release_date', 'Unknown'),
                            'duration_minutes': self._format_duration(metadata.get('duration_ms', 0)),
                            'danceability': round(metadata.get('danceability', 0.0), 3),
                            'energy': round(metadata.get('energy', 0.0), 3),
                            'valence': round(metadata.get('valence', 0.0), 3),
                            'acousticness': round(metadata.get('acousticness', 0.0), 3),
                            'instrumentalness': round(metadata.get('instrumentalness', 0.0), 3),
                            'tempo': round(metadata.get('tempo', 0.0), 1)
                        }
                        
                        songs_data.append(song_entry)
                        processed_count += 1
                    else:
                        errors_count += 1
                        
                except Exception as e:
                    errors_count += 1
                    continue
                
                # Mostrar progreso cada 1000 canciones
                if (i + 1) % 1000 == 0:
                    print(f"   Progreso: {i + 1}/{len(track_ids_sample)} canciones procesadas")
            
            if not songs_data:
                print("❌ No se pudieron procesar canciones para exportación")
                return False
            
            # Crear DataFrame y exportar
            catalog_df = pd.DataFrame(songs_data)
            
            # Ordenar por artista y luego por nombre de canción
            catalog_df = catalog_df.sort_values(['artist_name', 'track_name'])
            
            # Guardar CSV
            catalog_df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"✅ Catálogo básico exportado exitosamente:")
            print(f"   Archivo: {output_path}")
            print(f"   Canciones procesadas: {processed_count}")
            print(f"   Errores: {errors_count}")
            print(f"   Tasa de éxito: {processed_count/(processed_count+errors_count)*100:.1f}%")
            
            # Mostrar estadísticas del catálogo
            self._show_catalog_stats(catalog_df)
            
            return True
            
        except Exception as e:
            print(f"❌ Error durante exportación: {e}")
            return False
    
    def export_complete_catalog(self, output_path: str, include_clusters: bool = True, 
                              sample_size: int = None) -> bool:
        """
        Exporta catálogo completo con información de clusters y métricas avanzadas
        
        Args:
            output_path: Ruta del archivo CSV de salida
            include_clusters: Incluir información de clusters
            sample_size: Número de canciones a incluir (None para todas)
            
        Returns:
            True si la exportación fue exitosa
        """
        try:
            print(f"\n📊 Generando catálogo completo de canciones...")
            
            # Obtener datos del sistema
            track_ids = self.loader.get_track_ids()
            metadata_df = self.loader.get_songs_metadata()
            
            if include_clusters:
                musical_clusters = self.loader.get_musical_clusters()
                semantic_clusters = self.loader.get_semantic_clusters()
                print(f"   Incluyendo información de clusters musicales y semánticos")
            
            print(f"   Procesando {len(track_ids)} canciones del sistema...")
            
            # Crear lista de canciones con información completa
            songs_data = []
            processed_count = 0
            errors_count = 0
            
            # Aplicar muestreo si se especifica
            if sample_size and sample_size < len(track_ids):
                indices = np.random.choice(len(track_ids), size=sample_size, replace=False)
                track_ids_sample = track_ids[indices]
                print(f"   Muestreo aplicado: {sample_size} canciones seleccionadas aleatoriamente")
            else:
                track_ids_sample = track_ids
                indices = np.arange(len(track_ids))
            
            for i, (idx, track_id) in enumerate(zip(indices, track_ids_sample)):
                try:
                    song_info = self.loader.get_song_by_track_id(track_id)
                    
                    if song_info and 'metadata' in song_info:
                        metadata = song_info['metadata']
                        
                        # Extraer información completa
                        song_entry = {
                            'track_id': track_id,
                            'track_name': song_info.get('track_name', 'Unknown'),
                            'artist_name': song_info.get('artist_name', 'Unknown Artist'),
                            'album_name': metadata.get('track_album_name', 'Unknown Album'),
                            'release_date': metadata.get('track_album_release_date', 'Unknown'),
                            'genre': song_info.get('genre', 'Unknown'),
                            'popularity': metadata.get('track_popularity', 0),
                            'duration_minutes': self._format_duration(metadata.get('duration_ms', 0)),
                            'duration_ms': metadata.get('duration_ms', 0),
                            
                            # Características musicales principales
                            'danceability': round(metadata.get('danceability', 0.0), 3),
                            'energy': round(metadata.get('energy', 0.0), 3),
                            'valence': round(metadata.get('valence', 0.0), 3),
                            'acousticness': round(metadata.get('acousticness', 0.0), 3),
                            'instrumentalness': round(metadata.get('instrumentalness', 0.0), 3),
                            'liveness': round(metadata.get('liveness', 0.0), 3),
                            'speechiness': round(metadata.get('speechiness', 0.0), 3),
                            'tempo': round(metadata.get('tempo', 0.0), 1),
                            'loudness': round(metadata.get('loudness', 0.0), 2),
                            'key': metadata.get('key', -1),
                            'mode': metadata.get('mode', -1),
                        }
                        
                        # Agregar información de clusters si se solicita
                        if include_clusters:
                            song_entry.update({
                                'musical_cluster': int(musical_clusters[idx]),
                                'semantic_cluster': int(semantic_clusters[idx])
                            })
                        
                        songs_data.append(song_entry)
                        processed_count += 1
                    else:
                        errors_count += 1
                        
                except Exception as e:
                    errors_count += 1
                    continue
                
                # Mostrar progreso cada 1000 canciones
                if (i + 1) % 1000 == 0:
                    print(f"   Progreso: {i + 1}/{len(track_ids_sample)} canciones procesadas")
            
            if not songs_data:
                print("❌ No se pudieron procesar canciones para exportación")
                return False
            
            # Crear DataFrame y exportar
            catalog_df = pd.DataFrame(songs_data)
            
            # Ordenar por artista y luego por nombre de canción
            catalog_df = catalog_df.sort_values(['artist_name', 'track_name'])
            
            # Guardar CSV
            catalog_df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"✅ Catálogo completo exportado exitosamente:")
            print(f"   Archivo: {output_path}")
            print(f"   Canciones procesadas: {processed_count}")
            print(f"   Errores: {errors_count}")
            print(f"   Tasa de éxito: {processed_count/(processed_count+errors_count)*100:.1f}%")
            
            # Mostrar estadísticas del catálogo
            self._show_catalog_stats(catalog_df, include_clusters)
            
            return True
            
        except Exception as e:
            print(f"❌ Error durante exportación: {e}")
            return False
    
    def _format_duration(self, duration_ms: int) -> str:
        """
        Formatea duración de milisegundos a formato MM:SS
        
        Args:
            duration_ms: Duración en milisegundos
            
        Returns:
            String formateado como "MM:SS"
        """
        if duration_ms <= 0:
            return "0:00"
        
        minutes = int(duration_ms) // 60000
        seconds = (int(duration_ms) % 60000) // 1000
        return f"{minutes}:{seconds:02d}"
    
    def _show_catalog_stats(self, catalog_df: pd.DataFrame, include_clusters: bool = False):
        """
        Muestra estadísticas del catálogo exportado
        
        Args:
            catalog_df: DataFrame del catálogo exportado
            include_clusters: Si se incluyó información de clusters
        """
        print(f"\n📈 Estadísticas del catálogo exportado:")
        print(f"   Total canciones: {len(catalog_df)}")
        print(f"   Artistas únicos: {catalog_df['artist_name'].nunique()}")
        print(f"   Géneros únicos: {catalog_df['genre'].nunique()}")
        print(f"   Años representados: {len(catalog_df['release_date'].unique())}")
        
        # Top géneros
        top_genres = catalog_df['genre'].value_counts().head(5)
        print(f"\n🎵 Top 5 géneros:")
        for genre, count in top_genres.items():
            print(f"   {genre}: {count} canciones ({count/len(catalog_df)*100:.1f}%)")
        
        # Top artistas
        top_artists = catalog_df['artist_name'].value_counts().head(5)
        print(f"\n🎤 Top 5 artistas:")
        for artist, count in top_artists.items():
            print(f"   {artist}: {count} canciones")
        
        # Estadísticas de características musicales
        print(f"\n🎼 Características musicales (promedio):")
        for feature in ['energy', 'danceability', 'valence', 'acousticness']:
            avg_value = catalog_df[feature].mean()
            print(f"   {feature.capitalize()}: {avg_value:.3f}")
        
        if include_clusters:
            print(f"\n🎪 Distribución de clusters:")
            musical_dist = catalog_df['musical_cluster'].value_counts().sort_index()
            semantic_dist = catalog_df['semantic_cluster'].value_counts().sort_index()
            print(f"   Clusters musicales: {len(musical_dist)} clusters")
            print(f"   Clusters semánticos: {len(semantic_dist)} clusters")

def main():
    """Función principal del exportador"""
    parser = argparse.ArgumentParser(
        description="Exportador de Catálogo Musical - Sistema de Recomendaciones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python export_songs_catalog.py
  python export_songs_catalog.py --output mi_catalogo.csv
  python export_songs_catalog.py --complete --include_clusters
  python export_songs_catalog.py --sample 1000 --output muestra_1k.csv
        """
    )
    
    # Argumentos de salida
    parser.add_argument('--output', type=str, 
                       help='Archivo CSV de salida (default: songs_catalog_TIMESTAMP.csv)')
    
    # Opciones de contenido
    parser.add_argument('--complete', action='store_true',
                       help='Exportar catálogo completo con todas las características')
    parser.add_argument('--include_clusters', action='store_true',
                       help='Incluir información de clusters (requiere --complete)')
    parser.add_argument('--sample', type=int,
                       help='Exportar solo una muestra aleatoria de N canciones')
    
    # Opciones técnicas
    parser.add_argument('--system_dir', type=str,
                       help='Directorio del sistema (para desarrollo)')
    
    args = parser.parse_args()
    
    # Generar nombre de archivo por defecto si no se especifica
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        catalog_type = 'complete' if args.complete else 'basic'
        sample_suffix = f'_sample{args.sample}' if args.sample else ''
        args.output = f'songs_catalog_{catalog_type}{sample_suffix}_{timestamp}.csv'
    
    print("🎵 EXPORTADOR DE CATÁLOGO MUSICAL")
    print("=" * 40)
    print(f"Archivo de salida: {args.output}")
    print(f"Tipo de catálogo: {'Completo' if args.complete else 'Básico'}")
    if args.sample:
        print(f"Muestra: {args.sample} canciones")
    if args.include_clusters:
        print("Incluyendo información de clusters")
    
    try:
        # Inicializar exportador
        exporter = SongsCatalogExporter(args.system_dir)
        
        # Ejecutar exportación según configuración
        if args.complete:
            success = exporter.export_complete_catalog(
                args.output, 
                include_clusters=args.include_clusters,
                sample_size=args.sample
            )
        else:
            success = exporter.export_basic_catalog(
                args.output,
                sample_size=args.sample
            )
        
        if success:
            print(f"\n🎉 Exportación completada exitosamente")
            print(f"   Archivo generado: {args.output}")
            print(f"   Listo para análisis y exploración manual")
        else:
            print(f"\n❌ Exportación fallida")
            return 1
            
    except Exception as e:
        print(f"❌ Error durante exportación: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())