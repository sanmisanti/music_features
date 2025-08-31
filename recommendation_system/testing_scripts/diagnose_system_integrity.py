#!/usr/bin/env python3
"""
Script de Diagnóstico para Sistema de Recomendaciones Musicales
Identifica problemas de integridad entre archivos de datos y metadatos

Uso:
    python diagnose_system_integrity.py

Propósito:
    - Verificar existencia y estructura de archivos críticos
    - Analizar integridad referencial entre track_ids y metadatos
    - Identificar problemas específicos en el mapeo de datos
    - Proporcionar reporte detallado para corrección
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

class SystemIntegrityDiagnostic:
    """
    Diagnóstico completo de integridad del sistema de recomendaciones
    """
    
    def __init__(self, system_dir=None):
        if system_dir is None:
            self.system_dir = Path(__file__).parent.parent
        else:
            self.system_dir = Path(system_dir)
            
        self.data_dir = self.system_dir / "data"
        self.clusters_dir = self.system_dir / "clusters" 
        self.config_dir = self.system_dir / "config"
        
        self.issues = []
        self.warnings = []
        self.info = []
        
    def log_issue(self, message):
        """Registra un problema crítico"""
        self.issues.append(message)
        print(f"❌ ISSUE: {message}")
        
    def log_warning(self, message):
        """Registra una advertencia"""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
        
    def log_info(self, message):
        """Registra información"""
        self.info.append(message)
        print(f"ℹ️  INFO: {message}")
        
    def log_success(self, message):
        """Registra éxito"""
        print(f"✅ SUCCESS: {message}")
    
    def check_file_existence(self):
        """Verifica existencia de archivos críticos"""
        print("\n📁 VERIFICACIÓN DE ARCHIVOS CRÍTICOS")
        print("=" * 50)
        
        required_files = {
            'semantic_embeddings.npy': self.data_dir / 'semantic_embeddings.npy',
            'musical_features_normalized.npy': self.data_dir / 'musical_features_normalized.npy',
            'track_ids.npy': self.data_dir / 'track_ids.npy', 
            'songs_metadata.csv': self.data_dir / 'songs_metadata.csv',
            'musical_clusters_k10.npy': self.clusters_dir / 'musical_clusters_k10.npy',
            'semantic_clusters_k6.npy': self.clusters_dir / 'semantic_clusters_k6.npy',
            'system_config.json': self.config_dir / 'system_config.json'
        }
        
        all_exist = True
        for name, path in required_files.items():
            if path.exists():
                file_size = path.stat().st_size
                self.log_success(f"{name} existe ({file_size:,} bytes)")
            else:
                self.log_issue(f"{name} NO EXISTE en {path}")
                all_exist = False
                
        return all_exist
    
    def analyze_track_ids(self):
        """Analiza el archivo track_ids.npy"""
        print("\n🔍 ANÁLISIS DE TRACK_IDS")
        print("=" * 30)
        
        try:
            track_ids_path = self.data_dir / 'track_ids.npy'
            track_ids = np.load(track_ids_path, allow_pickle=True)
            
            self.log_success(f"track_ids.npy cargado exitosamente")
            self.log_info(f"Shape: {track_ids.shape}")
            self.log_info(f"Tipo de datos: {track_ids.dtype}")
            self.log_info(f"Valores únicos: {len(np.unique(track_ids))}")
            
            # Verificar primeros elementos
            self.log_info(f"Primeros 5 elementos: {track_ids[:5].tolist()}")
            
            # Verificar duplicados
            if len(track_ids) != len(np.unique(track_ids)):
                self.log_warning(f"Hay track_ids duplicados: {len(track_ids)} total vs {len(np.unique(track_ids))} únicos")
            else:
                self.log_success("No hay track_ids duplicados")
                
            return track_ids
            
        except Exception as e:
            self.log_issue(f"Error cargando track_ids.npy: {e}")
            return None
    
    def analyze_metadata(self):
        """Analiza el archivo songs_metadata.csv"""
        print("\n📊 ANÁLISIS DE METADATOS")
        print("=" * 30)
        
        try:
            metadata_path = self.data_dir / 'songs_metadata.csv'
            metadata_df = pd.read_csv(metadata_path, sep='^')
            
            self.log_success(f"songs_metadata.csv cargado exitosamente")
            self.log_info(f"Registros: {len(metadata_df)}")
            self.log_info(f"Columnas: {list(metadata_df.columns)}")
            
            # Verificar columna track_id crítica
            if 'track_id' in metadata_df.columns:
                self.log_success("Columna 'track_id' encontrada")
                self.log_info(f"Track IDs únicos en metadatos: {metadata_df['track_id'].nunique()}")
                self.log_info(f"Valores nulos en track_id: {metadata_df['track_id'].isnull().sum()}")
                
                # Mostrar primeros track_ids
                self.log_info(f"Primeros 5 track_ids: {metadata_df['track_id'].head().tolist()}")
                
                # Verificar duplicados
                duplicated_count = metadata_df['track_id'].duplicated().sum()
                if duplicated_count > 0:
                    self.log_warning(f"Track_ids duplicados en metadatos: {duplicated_count}")
                else:
                    self.log_success("No hay track_ids duplicados en metadatos")
                    
            else:
                self.log_issue("Columna 'track_id' NO ENCONTRADA en metadatos")
                
            return metadata_df
            
        except Exception as e:
            self.log_issue(f"Error cargando songs_metadata.csv: {e}")
            return None
    
    def check_referential_integrity(self, track_ids, metadata_df):
        """Verifica integridad referencial entre track_ids y metadatos"""
        print("\n🔗 VERIFICACIÓN DE INTEGRIDAD REFERENCIAL")
        print("=" * 45)
        
        if track_ids is None or metadata_df is None:
            self.log_issue("No se pueden verificar integridades: datos faltantes")
            return False
            
        if 'track_id' not in metadata_df.columns:
            self.log_issue("No se puede verificar integridad: columna track_id faltante")
            return False
        
        try:
            # Convertir a sets para operaciones de conjunto
            track_ids_set = set(track_ids.tolist())
            metadata_ids_set = set(metadata_df['track_id'].tolist())
            
            # Análisis de intersección
            intersection = track_ids_set & metadata_ids_set
            only_in_track_ids = track_ids_set - metadata_ids_set
            only_in_metadata = metadata_ids_set - track_ids_set
            
            self.log_info(f"Track IDs en track_ids.npy: {len(track_ids_set)}")
            self.log_info(f"Track IDs en metadatos: {len(metadata_ids_set)}")
            self.log_info(f"Track IDs en común: {len(intersection)}")
            
            if len(intersection) == len(track_ids_set) == len(metadata_ids_set):
                self.log_success("Integridad referencial PERFECTA")
                return True
            else:
                # Problemas identificados
                if only_in_track_ids:
                    self.log_issue(f"Track IDs en array pero NO en metadatos: {len(only_in_track_ids)}")
                    if len(only_in_track_ids) <= 5:
                        self.log_info(f"Ejemplos: {list(only_in_track_ids)}")
                    else:
                        self.log_info(f"Primeros 5 ejemplos: {list(only_in_track_ids)[:5]}")
                        
                if only_in_metadata:
                    self.log_warning(f"Track IDs en metadatos pero NO en array: {len(only_in_metadata)}")
                    if len(only_in_metadata) <= 5:
                        self.log_info(f"Ejemplos: {list(only_in_metadata)}")
                    else:
                        self.log_info(f"Primeros 5 ejemplos: {list(only_in_metadata)[:5]}")
                
                # Calcular porcentaje de cobertura
                coverage = len(intersection) / len(track_ids_set) * 100
                self.log_info(f"Cobertura de integridad: {coverage:.2f}%")
                
                return coverage > 95  # Consideramos aceptable si >95%
                
        except Exception as e:
            self.log_issue(f"Error verificando integridad referencial: {e}")
            return False
    
    def test_problematic_track_id(self):
        """Prueba específicamente con el track_id que está causando problemas"""
        print("\n🐛 PRUEBA DE TRACK_ID PROBLEMÁTICO")
        print("=" * 35)
        
        problematic_id = "1JIQmOrYNMohZ8oygnm9Bg"
        self.log_info(f"Probando track_id específico: {problematic_id}")
        
        try:
            # Cargar datos
            track_ids = np.load(self.data_dir / 'track_ids.npy', allow_pickle=True)
            metadata_df = pd.read_csv(self.data_dir / 'songs_metadata.csv', sep='^')
            
            # Buscar en track_ids array
            track_indices = np.where(track_ids == problematic_id)[0]
            if len(track_indices) > 0:
                self.log_success(f"Track ID encontrado en array en índice: {track_indices[0]}")
            else:
                self.log_warning(f"Track ID NO encontrado en track_ids array")
                
            # Buscar en metadatos
            if 'track_id' in metadata_df.columns:
                metadata_matches = metadata_df[metadata_df['track_id'] == problematic_id]
                if len(metadata_matches) > 0:
                    self.log_success(f"Track ID encontrado en metadatos: {len(metadata_matches)} registros")
                    self.log_info(f"Info: {metadata_matches[['track_name', 'artist_name']].iloc[0].to_dict()}")
                else:
                    self.log_issue(f"Track ID NO encontrado en metadatos")
            else:
                self.log_issue("No se puede buscar: columna track_id faltante en metadatos")
                
        except Exception as e:
            self.log_issue(f"Error en prueba de track_id problemático: {e}")
    
    def generate_report(self):
        """Genera reporte final de diagnóstico"""
        print("\n📋 REPORTE FINAL DE DIAGNÓSTICO")
        print("=" * 40)
        
        total_issues = len(self.issues)
        total_warnings = len(self.warnings)
        
        if total_issues == 0:
            self.log_success("✨ SISTEMA EN BUEN ESTADO - No se encontraron problemas críticos")
        else:
            print(f"❌ PROBLEMAS CRÍTICOS ENCONTRADOS: {total_issues}")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
                
        if total_warnings > 0:
            print(f"⚠️  ADVERTENCIAS: {total_warnings}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        # Recomendaciones
        print("\n💡 RECOMENDACIONES:")
        if total_issues > 0:
            print("   - Corregir problemas críticos antes de usar el sistema")
            print("   - Verificar integridad de archivos de datos")
            print("   - Considerar regenerar archivos corruptos")
        else:
            print("   - Sistema listo para uso")
            
        return total_issues == 0

def main():
    """Función principal del diagnóstico"""
    print("🔧 DIAGNÓSTICO DE INTEGRIDAD - SISTEMA DE RECOMENDACIONES MUSICALES")
    print("=" * 70)
    
    diagnostic = SystemIntegrityDiagnostic()
    
    # Ejecutar verificaciones
    files_ok = diagnostic.check_file_existence()
    track_ids = diagnostic.analyze_track_ids()
    metadata_df = diagnostic.analyze_metadata()
    
    if files_ok and track_ids is not None and metadata_df is not None:
        integrity_ok = diagnostic.check_referential_integrity(track_ids, metadata_df)
    
    # Prueba específica
    diagnostic.test_problematic_track_id()
    
    # Reporte final
    system_healthy = diagnostic.generate_report()
    
    # Código de salida
    sys.exit(0 if system_healthy else 1)

if __name__ == "__main__":
    main()