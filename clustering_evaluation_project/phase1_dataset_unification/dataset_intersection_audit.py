#!/usr/bin/env python3
"""
FASE 1: AUDITORÍA DE INTERSECCIÓN DE DATASETS MULTIMODALES
=========================================================

Script para identificar la intersección exacta entre canciones con embeddings BERT
y características musicales, cuantificando el grado de alineación entre datasets.

Autor: Clustering Evaluation Project
Fecha: Agosto 2025
Objetivo: Establecer base sólida para análisis comparativo multimodal
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DatasetIntersectionAuditor:
    """
    Auditor especializado para análisis de intersección de datasets multimodales.
    
    Funcionalidades:
    - Análisis de intersección entre embeddings semánticos y características musicales
    - Cuantificación de alineación de datasets
    - Evaluación de completitud de datos multimodales
    - Generación de reportes detallados de cobertura
    """
    
    def __init__(self, base_path=None):
        """Inicializar auditor de datasets."""
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent
        else:
            base_path = Path(base_path)
            
        self.base_path = base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rutas de archivos principales
        self.semantic_embeddings_path = base_path / "vectorization_complete_output" / "embeddings_complete_20250819_194820.npy"
        self.semantic_track_ids_path = base_path / "vectorization_complete_output" / "track_ids_complete_20250819_194820.npy"
        self.semantic_metadata_path = base_path / "vectorization_complete_output" / "vectorization_metadata_20250819_194820.json"
        
        self.musical_dataset_path = base_path / "data" / "final_data" / "picked_data_optimal.csv"
        
        # Resultados de auditoría
        self.audit_results = {}
        self.intersection_data = {}
        
        print(f"DatasetIntersectionAuditor inicializado")
        print(f"Base path: {base_path}")
        print(f"Timestamp: {self.timestamp}")
    
    def load_semantic_data(self):
        """Cargar datos de embeddings semánticos y metadatos."""
        try:
            print("Cargando datos semánticos...")
            
            # Cargar embeddings BERT
            self.semantic_embeddings = np.load(self.semantic_embeddings_path)
            print(f"Embeddings BERT cargados: {self.semantic_embeddings.shape}")
            
            # Cargar track IDs correspondientes
            self.semantic_track_ids = np.load(self.semantic_track_ids_path, allow_pickle=True)
            print(f"Track IDs semánticos cargados: {len(self.semantic_track_ids)}")
            
            # Cargar metadatos de vectorización
            with open(self.semantic_metadata_path, 'r') as f:
                self.semantic_metadata = json.load(f)
                
            # Validar consistencia de datos
            if len(self.semantic_track_ids) != self.semantic_embeddings.shape[0]:
                raise ValueError(f"Inconsistencia: {len(self.semantic_track_ids)} track_ids vs {self.semantic_embeddings.shape[0]} embeddings")
            
            # Filtrar embeddings válidos (no-zero)
            valid_mask = np.any(self.semantic_embeddings != 0, axis=1)
            self.valid_semantic_embeddings = self.semantic_embeddings[valid_mask]
            self.valid_semantic_track_ids = self.semantic_track_ids[valid_mask]
            
            print(f"Embeddings válidos (no-zero): {len(self.valid_semantic_track_ids)}")
            
            return True
            
        except Exception as e:
            print(f"Error cargando datos semánticos: {e}")
            return False
    
    def load_musical_data(self):
        """Cargar dataset de características musicales."""
        try:
            print("Cargando datos musicales...")
            
            # Cargar dataset musical optimizado
            self.musical_dataset = pd.read_csv(self.musical_dataset_path, 
                                             sep='^', 
                                             encoding='utf-8',
                                             low_memory=False)
            
            print(f"Dataset musical cargado: {len(self.musical_dataset)} canciones")
            print(f"Columnas disponibles: {list(self.musical_dataset.columns)}")
            
            # Extraer track IDs musicales
            self.musical_track_ids = self.musical_dataset['track_id'].values
            print(f"Track IDs musicales: {len(self.musical_track_ids)}")
            
            # Verificar características musicales disponibles
            self.musical_feature_columns = [
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
            ]
            
            available_features = [col for col in self.musical_feature_columns if col in self.musical_dataset.columns]
            missing_features = [col for col in self.musical_feature_columns if col not in self.musical_dataset.columns]
            
            print(f"Características musicales disponibles: {len(available_features)}/13")
            if missing_features:
                print(f"Características faltantes: {missing_features}")
            
            self.available_musical_features = available_features
            
            return True
            
        except Exception as e:
            print(f"Error cargando datos musicales: {e}")
            return False
    
    def calculate_intersection(self):
        """Calcular intersección entre datasets semántico y musical."""
        try:
            print("Calculando intersección de datasets...")
            
            # Convertir a sets para operaciones de intersección eficientes
            semantic_set = set(self.valid_semantic_track_ids)
            musical_set = set(self.musical_track_ids)
            
            # Calcular intersecciones y diferencias
            multimodal_intersection = semantic_set.intersection(musical_set)
            semantic_only = semantic_set - musical_set
            musical_only = musical_set - semantic_set
            
            # Estadísticas de intersección
            intersection_stats = {
                'semantic_total': len(semantic_set),
                'musical_total': len(musical_set),
                'multimodal_intersection': len(multimodal_intersection),
                'semantic_only': len(semantic_only),
                'musical_only': len(musical_only),
                'intersection_percentage_semantic': (len(multimodal_intersection) / len(semantic_set)) * 100,
                'intersection_percentage_musical': (len(multimodal_intersection) / len(musical_set)) * 100,
                'dataset_alignment_score': len(multimodal_intersection) / max(len(semantic_set), len(musical_set)),
                'coverage_ratio': len(multimodal_intersection) / min(len(semantic_set), len(musical_set))
            }
            
            self.intersection_stats = intersection_stats
            self.multimodal_track_ids = list(multimodal_intersection)
            self.semantic_only_ids = list(semantic_only)
            self.musical_only_ids = list(musical_only)
            
            print(f"RESULTADOS DE INTERSECCIÓN:")
            print(f"  Canciones semánticas: {intersection_stats['semantic_total']:,}")
            print(f"  Canciones musicales: {intersection_stats['musical_total']:,}")
            print(f"  Intersección multimodal: {intersection_stats['multimodal_intersection']:,}")
            print(f"  Cobertura semántica: {intersection_stats['intersection_percentage_semantic']:.1f}%")
            print(f"  Cobertura musical: {intersection_stats['intersection_percentage_musical']:.1f}%")
            print(f"  Score de alineación: {intersection_stats['dataset_alignment_score']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error calculando intersección: {e}")
            return False
    
    def analyze_data_quality(self):
        """Analizar calidad de datos en la intersección multimodal."""
        try:
            print("Analizando calidad de datos multimodales...")
            
            # Filtrar datos a la intersección multimodal
            multimodal_musical_data = self.musical_dataset[
                self.musical_dataset['track_id'].isin(self.multimodal_track_ids)
            ].copy()
            
            # Análisis de completitud de características musicales
            musical_completeness = {}
            for feature in self.available_musical_features:
                non_null_count = multimodal_musical_data[feature].notna().sum()
                completeness_percentage = (non_null_count / len(multimodal_musical_data)) * 100
                musical_completeness[feature] = {
                    'non_null_count': int(non_null_count),
                    'completeness_percentage': completeness_percentage
                }
            
            # Análisis de distribución de géneros
            if 'playlist_genre' in multimodal_musical_data.columns:
                genre_distribution = multimodal_musical_data['playlist_genre'].value_counts().head(10).to_dict()
                genre_diversity = len(multimodal_musical_data['playlist_genre'].unique())
            else:
                genre_distribution = {}
                genre_diversity = 0
            
            # Análisis de características de letras (si disponible)
            lyrics_analysis = {}
            if 'lyrics' in multimodal_musical_data.columns:
                lyrics_available = multimodal_musical_data['lyrics'].notna().sum()
                lyrics_percentage = (lyrics_available / len(multimodal_musical_data)) * 100
                
                # Análisis básico de longitud de letras
                lyrics_data = multimodal_musical_data['lyrics'].dropna()
                if len(lyrics_data) > 0:
                    lyrics_lengths = lyrics_data.str.len()
                    lyrics_analysis = {
                        'available_count': int(lyrics_available),
                        'availability_percentage': lyrics_percentage,
                        'average_length': float(lyrics_lengths.mean()),
                        'median_length': float(lyrics_lengths.median()),
                        'min_length': int(lyrics_lengths.min()),
                        'max_length': int(lyrics_lengths.max())
                    }
            
            self.quality_analysis = {
                'multimodal_sample_size': len(multimodal_musical_data),
                'musical_features_completeness': musical_completeness,
                'genre_distribution': genre_distribution,
                'genre_diversity': genre_diversity,
                'lyrics_analysis': lyrics_analysis,
                'available_musical_features_count': len(self.available_musical_features),
                'expected_musical_features_count': 13
            }
            
            print(f"ANÁLISIS DE CALIDAD:")
            print(f"  Muestra multimodal: {len(multimodal_musical_data):,} canciones")
            print(f"  Características musicales: {len(self.available_musical_features)}/13")
            print(f"  Diversidad de géneros: {genre_diversity}")
            if lyrics_analysis:
                print(f"  Letras disponibles: {lyrics_analysis.get('availability_percentage', 0):.1f}%")
            
            return True
            
        except Exception as e:
            print(f"Error analizando calidad de datos: {e}")
            return False
    
    def generate_audit_report(self):
        """Generar reporte completo de auditoría."""
        try:
            print("Generando reporte de auditoría...")
            
            # Compilar reporte completo
            audit_report = {
                'audit_metadata': {
                    'timestamp': self.timestamp,
                    'auditor_version': '1.0',
                    'base_path': str(self.base_path),
                    'semantic_embeddings_source': str(self.semantic_embeddings_path),
                    'musical_dataset_source': str(self.musical_dataset_path)
                },
                'semantic_data_info': {
                    'embeddings_shape': list(self.semantic_embeddings.shape),
                    'total_processed': len(self.semantic_track_ids),
                    'valid_embeddings': len(self.valid_semantic_track_ids),
                    'embedding_dimensions': self.semantic_embeddings.shape[1],
                    'embedding_dtype': str(self.semantic_embeddings.dtype),
                    'processing_metadata': self.semantic_metadata
                },
                'musical_data_info': {
                    'total_songs': len(self.musical_dataset),
                    'available_features': self.available_musical_features,
                    'missing_features': [f for f in self.musical_feature_columns if f not in self.available_musical_features],
                    'dataset_columns': list(self.musical_dataset.columns)
                },
                'intersection_analysis': self.intersection_stats,
                'quality_analysis': self.quality_analysis,
                'multimodal_viability': {
                    'viable_for_analysis': len(self.multimodal_track_ids) >= 1000,
                    'recommended_min_sample': 1000,
                    'actual_sample_size': len(self.multimodal_track_ids),
                    'alignment_quality': 'excellent' if self.intersection_stats['dataset_alignment_score'] > 0.8 else
                                       'good' if self.intersection_stats['dataset_alignment_score'] > 0.5 else
                                       'poor',
                    'coverage_assessment': 'sufficient' if len(self.multimodal_track_ids) >= 2000 else
                                         'limited' if len(self.multimodal_track_ids) >= 1000 else
                                         'insufficient'
                }
            }
            
            self.audit_report = audit_report
            
            # Guardar reporte
            output_path = Path(__file__).parent / f"dataset_intersection_report_{self.timestamp}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(audit_report, f, indent=2, ensure_ascii=False)
            
            print(f"Reporte guardado en: {output_path}")
            
            # Guardar lista de track IDs multimodales
            multimodal_ids_path = Path(__file__).parent / f"valid_multimodal_track_ids_{self.timestamp}.npy"
            np.save(multimodal_ids_path, np.array(self.multimodal_track_ids))
            print(f"Track IDs multimodales guardados en: {multimodal_ids_path}")
            
            return True
            
        except Exception as e:
            print(f"Error generando reporte: {e}")
            return False
    
    def print_summary(self):
        """Imprimir resumen de resultados de auditoría."""
        print("\n" + "="*80)
        print("RESUMEN DE AUDITORÍA DE DATASETS MULTIMODALES")
        print("="*80)
        
        print(f"\nDATOS SEMÁNTICOS:")
        print(f"  Embeddings BERT: {self.semantic_embeddings.shape[0]:,} x {self.semantic_embeddings.shape[1]}D")
        print(f"  Embeddings válidos: {len(self.valid_semantic_track_ids):,}")
        print(f"  Tasa de éxito: {(len(self.valid_semantic_track_ids)/len(self.semantic_track_ids))*100:.1f}%")
        
        print(f"\nDATOS MUSICALES:")
        print(f"  Total canciones: {len(self.musical_dataset):,}")
        print(f"  Características: {len(self.available_musical_features)}/13")
        
        print(f"\nINTERSECCIÓN MULTIMODAL:")
        print(f"  Canciones comunes: {len(self.multimodal_track_ids):,}")
        print(f"  Cobertura semántica: {self.intersection_stats['intersection_percentage_semantic']:.1f}%")
        print(f"  Cobertura musical: {self.intersection_stats['intersection_percentage_musical']:.1f}%")
        print(f"  Score alineación: {self.intersection_stats['dataset_alignment_score']:.3f}")
        
        print(f"\nVIABILIDAD PARA ANÁLISIS:")
        viability = self.audit_report['multimodal_viability']
        print(f"  Viable: {'SÍ' if viability['viable_for_analysis'] else 'NO'}")
        print(f"  Calidad alineación: {viability['alignment_quality'].upper()}")
        print(f"  Evaluación cobertura: {viability['coverage_assessment'].upper()}")
        
        if len(self.multimodal_track_ids) >= 1000:
            print(f"\n✅ CONCLUSIÓN: Dataset viable para análisis comparativo multimodal")
        else:
            print(f"\n❌ CONCLUSIÓN: Dataset insuficiente para análisis robusto")
        
        print("="*80)
    
    def run_complete_audit(self):
        """Ejecutar auditoría completa de datasets."""
        print("Iniciando auditoría completa de datasets multimodales...")
        
        # Secuencia de ejecución
        steps = [
            ("Cargar datos semánticos", self.load_semantic_data),
            ("Cargar datos musicales", self.load_musical_data),
            ("Calcular intersección", self.calculate_intersection),
            ("Analizar calidad", self.analyze_data_quality),
            ("Generar reporte", self.generate_audit_report)
        ]
        
        for step_name, step_function in steps:
            print(f"\n[PASO] {step_name}...")
            success = step_function()
            if not success:
                print(f"❌ Error en paso: {step_name}")
                return False
            print(f"✅ {step_name} completado")
        
        # Mostrar resumen final
        self.print_summary()
        
        return True


def main():
    """Función principal de ejecución."""
    print("AUDITORÍA DE INTERSECCIÓN DE DATASETS MULTIMODALES")
    print("="*60)
    
    # Inicializar auditor
    auditor = DatasetIntersectionAuditor()
    
    # Ejecutar auditoría completa
    success = auditor.run_complete_audit()
    
    if success:
        print(f"\n🎉 Auditoría completada exitosamente")
        print(f"📊 Reporte disponible en: phase1_dataset_unification/")
        return auditor
    else:
        print(f"\n💥 Auditoría falló")
        return None


if __name__ == "__main__":
    auditor = main()