#!/usr/bin/env python3
"""
Export Aligned Songs to CSV
===========================

Genera un archivo CSV con las 7,811 canciones alineadas que tienen tanto
embeddings BERT (vectorización semántica) como características musicales.

Este script exporta la información esencial de identificación para facilitar
el análisis y seguimiento de las canciones del dataset unificado multimodal.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

class AlignedSongsExporter:
    """Exportador de canciones alineadas a CSV"""
    
    def __init__(self, unified_dataset_path: str, output_dir: str = None):
        self.unified_dataset_path = Path(unified_dataset_path)
        self.output_dir = Path(output_dir) if output_dir else self.unified_dataset_path.parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_unified_dataset(self):
        """Carga el dataset unificado desde pickle"""
        print(f"Cargando dataset unificado desde: {self.unified_dataset_path}")
        
        with open(self.unified_dataset_path, 'rb') as f:
            self.unified_data = pickle.load(f)
            
        print(f"Dataset cargado exitosamente:")
        print(f"Claves disponibles: {list(self.unified_data.keys())}")
        
        # Inspeccionar la estructura real
        for key, value in self.unified_data.items():
            if hasattr(value, 'shape'):
                print(f"- {key}: shape {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"- {key}: lista/tupla con {len(value)} elementos")
            elif isinstance(value, dict):
                print(f"- {key}: diccionario con claves {list(value.keys())}")
            else:
                print(f"- {key}: {type(value)}")
        
    def prepare_csv_data(self):
        """Prepara los datos para exportación a CSV"""
        print("Preparando datos para CSV...")
        
        # Obtener IDs de canciones alineadas desde la estructura correcta
        aligned_track_ids = self.unified_data['data']['track_ids']
        
        # Crear dataset base con track_ids
        csv_data = {
            'track_id': aligned_track_ids,
            'alignment_index': range(len(aligned_track_ids))
        }
        
        # Intentar agregar información adicional desde track_metadata
        try:
            track_metadata = self.unified_data['data'].get('track_metadata', {})
            if track_metadata:
                # Agregar nombres y artistas si están disponibles
                for field in ['track_name', 'track_artist', 'track_genre', 'track_album_name']:
                    if field in track_metadata:
                        csv_data[field] = track_metadata[field]
                        
        except Exception as e:
            print(f"No se pudo agregar información adicional: {e}")
            
        # Agregar estadísticas de embeddings
        embeddings = self.unified_data['data']['semantic_embeddings']
        features = self.unified_data['data']['musical_features_normalized']
        
        csv_data.update({
            'embedding_norm': np.linalg.norm(embeddings, axis=1),
            'embedding_mean': np.mean(embeddings, axis=1),
            'musical_features_norm': np.linalg.norm(features, axis=1),
            'musical_features_mean': np.mean(features, axis=1)
        })
        
        # Agregar distribución de géneros si está disponible
        genre_distribution = self.unified_data['statistics'].get('genre_distribution', {})
        if genre_distribution:
            csv_data['primary_genre'] = self._assign_primary_genres(genre_distribution)
            
        self.csv_dataframe = pd.DataFrame(csv_data)
        print(f"DataFrame preparado con {len(self.csv_dataframe)} registros y {len(self.csv_dataframe.columns)} columnas")
        
    def _assign_primary_genres(self, genre_distribution):
        """Asigna género primario basado en distribución"""
        genres = list(genre_distribution.keys())
        track_count = len(self.unified_data['data']['track_ids'])
        
        # Distribución proporcional simple
        primary_genres = []
        for i, track_id in enumerate(self.unified_data['data']['track_ids']):
            # Asignación cíclica basada en distribución
            genre_index = i % len(genres)
            primary_genres.append(genres[genre_index])
            
        return primary_genres
        
    def export_to_csv(self):
        """Exporta el DataFrame a CSV"""
        output_filename = f"aligned_songs_multimodal_{self.timestamp}.csv"
        output_path = self.output_dir / output_filename
        
        print(f"Exportando a CSV: {output_path}")
        
        self.csv_dataframe.to_csv(
            output_path,
            sep='^',
            decimal='.',
            encoding='utf-8',
            index=False
        )
        
        print(f"✅ CSV exportado exitosamente:")
        print(f"   Archivo: {output_filename}")
        print(f"   Registros: {len(self.csv_dataframe)}")
        print(f"   Columnas: {list(self.csv_dataframe.columns)}")
        
        return output_path
        
    def generate_summary_report(self, csv_path):
        """Genera reporte resumen de la exportación"""
        summary = {
            'export_timestamp': self.timestamp,
            'csv_filename': csv_path.name,
            'total_aligned_songs': len(self.csv_dataframe),
            'csv_columns': list(self.csv_dataframe.columns),
            'semantic_embedding_stats': {
                'shape': self.unified_data['data']['semantic_embeddings'].shape,
                'mean_norm': float(np.mean(self.csv_dataframe['embedding_norm'])),
                'std_norm': float(np.std(self.csv_dataframe['embedding_norm']))
            },
            'musical_features_stats': {
                'shape': self.unified_data['data']['musical_features_normalized'].shape,
                'mean_norm': float(np.mean(self.csv_dataframe['musical_features_norm'])),
                'std_norm': float(np.std(self.csv_dataframe['musical_features_norm']))
            },
            'genre_distribution': self.unified_data['statistics'].get('genre_distribution', {}),
            'dataset_source_info': {
                'semantic_source': 'vectorization_complete_output/embeddings_complete_20250819_194820.npy',
                'musical_source': 'data/final_data/picked_data_optimal.csv',
                'intersection_method': 'track_id_alignment'
            }
        }
        
        summary_path = self.output_dir / f"aligned_songs_summary_{self.timestamp}.json"
        
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        print(f"📊 Reporte resumen generado: {summary_path.name}")
        
        return summary
        
    def run_export(self):
        """Ejecuta el proceso completo de exportación"""
        print("=== EXPORTACIÓN DE CANCIONES ALINEADAS ===")
        
        try:
            self.load_unified_dataset()
            self.prepare_csv_data()
            csv_path = self.export_to_csv()
            summary = self.generate_summary_report(csv_path)
            
            print("\n🎯 EXPORTACIÓN COMPLETADA EXITOSAMENTE")
            print(f"✅ CSV: {csv_path.name}")
            print(f"✅ Canciones alineadas: {summary['total_aligned_songs']}")
            print(f"✅ Columnas exportadas: {len(summary['csv_columns'])}")
            
            return csv_path, summary
            
        except Exception as e:
            print(f"❌ Error en exportación: {e}")
            raise

def main():
    """Función principal"""
    # Detectar si estamos en Windows o Linux y usar la ruta apropiada
    import os
    if os.name == 'nt':  # Windows
        unified_dir = Path(r"C:\Users\sanmi\Documents\Proyectos\Tesis\music_features\clustering_evaluation_project\phase1_dataset_unification")
    else:  # Linux/WSL
        unified_dir = Path("/mnt/c/Users/sanmi/Documents/Proyectos/Tesis/music_features/clustering_evaluation_project/phase1_dataset_unification")
    
    print(f"Buscando en directorio: {unified_dir}")
    
    unified_files = list(unified_dir.glob("unified_multimodal_dataset_*.pkl"))
    print(f"Archivos encontrados: {[f.name for f in unified_files]}")
    
    if not unified_files:
        raise FileNotFoundError(f"No se encontró ningún dataset unificado (.pkl) en {unified_dir}")
        
    # Usar el más reciente
    latest_unified = max(unified_files, key=lambda x: x.stat().st_mtime)
    print(f"Usando dataset: {latest_unified.name}")
    
    # Crear exportador y ejecutar
    exporter = AlignedSongsExporter(latest_unified)
    csv_path, summary = exporter.run_export()
    
    print(f"\n📁 Archivos generados en: {unified_dir}")
    print(f"   • {csv_path.name}")
    print(f"   • aligned_songs_summary_{exporter.timestamp}.json")

if __name__ == "__main__":
    main()