#!/usr/bin/env python3
"""
Script para Enriquecer songs_metadata.csv con Metadatos Musicales Completos
Combina metadatos técnicos del sistema con metadatos descriptivos del dataset fuente

Uso:
    python enrich_metadata.py

Propósito:
    - Leer songs_metadata.csv (metadatos técnicos)
    - Leer spotify_songs_fixed.csv (metadatos descriptivos)
    - Realizar join por track_id
    - Crear songs_metadata_enriched.csv con columnas completas
    - Mantener compatibilidad con el sistema existente
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

class MetadataEnricher:
    """
    Enriquecedor de metadatos para sistema de recomendaciones
    """
    
    def __init__(self):
        self.system_dir = Path(__file__).parent.parent
        self.data_dir = self.system_dir / "data"
        self.source_data_dir = Path(__file__).parent.parent.parent / "data" / "with_lyrics"
        
        # Rutas de archivos
        self.technical_metadata_path = self.data_dir / "songs_metadata.csv"
        self.descriptive_metadata_path = self.source_data_dir / "spotify_songs_fixed.csv"
        self.enriched_metadata_path = self.data_dir / "songs_metadata_enriched.csv"
        self.backup_path = self.data_dir / f"songs_metadata_original_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        print("🔧 ENRIQUECIMIENTO DE METADATOS - Sistema de Recomendaciones")
        print("=" * 65)
        
    def load_technical_metadata(self):
        """Carga metadatos técnicos del sistema de recomendaciones"""
        try:
            print("📊 Cargando metadatos técnicos del sistema...")
            technical_df = pd.read_csv(self.technical_metadata_path, sep='^')
            print(f"✅ Metadatos técnicos cargados: {len(technical_df)} registros")
            print(f"   Columnas técnicas: {list(technical_df.columns)}")
            return technical_df
        except Exception as e:
            print(f"❌ Error cargando metadatos técnicos: {e}")
            return None
    
    def load_descriptive_metadata(self):
        """Carga metadatos descriptivos del dataset fuente"""
        try:
            print("\n🎵 Cargando metadatos descriptivos del dataset fuente...")
            descriptive_df = pd.read_csv(self.descriptive_metadata_path, sep='@@')
            print(f"✅ Metadatos descriptivos cargados: {len(descriptive_df)} registros")
            print(f"   Columnas disponibles: {len(descriptive_df.columns)} columnas")
            
            # Seleccionar columnas relevantes para el sistema de recomendaciones
            relevant_columns = [
                'track_id', 'track_name', 'track_artist', 'track_album_name',
                'track_popularity', 'playlist_genre', 'playlist_subgenre'
            ]
            
            # Verificar disponibilidad de columnas
            available_columns = [col for col in relevant_columns if col in descriptive_df.columns]
            missing_columns = [col for col in relevant_columns if col not in descriptive_df.columns]
            
            if missing_columns:
                print(f"⚠️  Columnas faltantes en dataset fuente: {missing_columns}")
                
                # Mapear nombres alternativos
                column_mappings = {
                    'track_artist': 'track_artist',  # Puede ser 'artist_name' en otros datasets
                }
                
                for missing_col in missing_columns:
                    if missing_col in column_mappings:
                        alt_col = column_mappings[missing_col]
                        if alt_col in descriptive_df.columns:
                            print(f"✅ Mapeando {missing_col} <- {alt_col}")
                            available_columns.append(alt_col)
                            
            print(f"✅ Columnas relevantes disponibles: {available_columns}")
            
            # Filtrar solo columnas disponibles
            filtered_df = descriptive_df[available_columns].copy()
            
            # Renombrar artist_name si existe
            if 'track_artist' in filtered_df.columns:
                filtered_df['artist_name'] = filtered_df['track_artist']
                
            return filtered_df
            
        except Exception as e:
            print(f"❌ Error cargando metadatos descriptivos: {e}")
            print(f"   Verificar que existe: {self.descriptive_metadata_path}")
            return None
    
    def merge_metadata(self, technical_df, descriptive_df):
        """Combina metadatos técnicos y descriptivos"""
        try:
            print("\n🔗 Combinando metadatos técnicos y descriptivos...")
            
            # Verificar track_ids únicos en ambos datasets
            technical_track_ids = set(technical_df['track_id'].tolist())
            descriptive_track_ids = set(descriptive_df['track_id'].tolist())
            
            intersection = technical_track_ids & descriptive_track_ids
            print(f"📊 Análisis de intersección:")
            print(f"   Track IDs técnicos: {len(technical_track_ids)}")
            print(f"   Track IDs descriptivos: {len(descriptive_track_ids)}")
            print(f"   Track IDs en común: {len(intersection)}")
            
            if len(intersection) == 0:
                print("❌ Error crítico: No hay track_ids en común entre datasets")
                return None
                
            coverage = len(intersection) / len(technical_track_ids) * 100
            print(f"   Cobertura de enriquecimiento: {coverage:.2f}%")
            
            # Realizar merge (left join para preservar todos los registros técnicos)
            enriched_df = technical_df.merge(
                descriptive_df,
                on='track_id',
                how='left',
                suffixes=('', '_desc')
            )
            
            print(f"✅ Metadatos combinados exitosamente: {len(enriched_df)} registros")
            print(f"   Columnas en dataset enriquecido: {len(enriched_df.columns)}")
            
            # Verificar datos faltantes en columnas críticas
            critical_columns = ['track_name', 'artist_name']
            for col in critical_columns:
                if col in enriched_df.columns:
                    null_count = enriched_df[col].isnull().sum()
                    if null_count > 0:
                        print(f"⚠️  Valores faltantes en {col}: {null_count} ({null_count/len(enriched_df)*100:.1f}%)")
                    else:
                        print(f"✅ {col}: Sin valores faltantes")
            
            return enriched_df
            
        except Exception as e:
            print(f"❌ Error combinando metadatos: {e}")
            return None
    
    def save_enriched_metadata(self, enriched_df):
        """Guarda metadatos enriquecidos"""
        try:
            print("\n💾 Guardando metadatos enriquecidos...")
            
            # Crear backup del archivo original
            if self.technical_metadata_path.exists():
                shutil.copy2(self.technical_metadata_path, self.backup_path)
                print(f"✅ Backup creado: {self.backup_path.name}")
            
            # Guardar archivo enriquecido
            enriched_df.to_csv(self.enriched_metadata_path, sep='^', index=False)
            print(f"✅ Metadatos enriquecidos guardados: {self.enriched_metadata_path.name}")
            
            # Mostrar estructura final
            print(f"\n📊 Estructura final del archivo enriquecido:")
            print(f"   Registros: {len(enriched_df)}")
            print(f"   Columnas: {list(enriched_df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error guardando metadatos enriquecidos: {e}")
            return False
    
    def update_system_to_use_enriched(self):
        """Actualiza el sistema para usar el archivo enriquecido"""
        try:
            print("\n🔄 Actualizando sistema para usar metadatos enriquecidos...")
            
            # Sobrescribir archivo original con enriquecido
            if self.enriched_metadata_path.exists():
                shutil.copy2(self.enriched_metadata_path, self.technical_metadata_path)
                print(f"✅ Sistema actualizado para usar metadatos enriquecidos")
                print(f"   Archivo original: {self.backup_path.name} (backup)")
                print(f"   Archivo activo: {self.technical_metadata_path.name} (enriquecido)")
                return True
            else:
                print("❌ Error: Archivo enriquecido no existe")
                return False
                
        except Exception as e:
            print(f"❌ Error actualizando sistema: {e}")
            return False
    
    def verify_enrichment(self):
        """Verifica que el enriquecimiento fue exitoso"""
        try:
            print("\n✨ Verificando enriquecimiento exitoso...")
            
            # Cargar archivo actualizado
            updated_df = pd.read_csv(self.technical_metadata_path, sep='^')
            
            # Verificar columnas críticas
            critical_columns = ['track_id', 'track_name', 'artist_name']
            missing_critical = [col for col in critical_columns if col not in updated_df.columns]
            
            if missing_critical:
                print(f"❌ Verificación fallida: Columnas críticas faltantes: {missing_critical}")
                return False
            
            # Verificar contenido no nulo
            for col in critical_columns:
                null_count = updated_df[col].isnull().sum()
                null_percentage = null_count / len(updated_df) * 100
                
                if null_percentage > 50:  # Más del 50% de datos faltantes
                    print(f"⚠️  Advertencia: {col} tiene {null_percentage:.1f}% valores faltantes")
                else:
                    print(f"✅ {col}: {null_percentage:.1f}% valores faltantes (aceptable)")
            
            # Mostrar muestra de datos enriquecidos
            print(f"\n📋 Muestra de metadatos enriquecidos:")
            sample_cols = ['track_id', 'track_name', 'artist_name'] if all(col in updated_df.columns for col in ['track_id', 'track_name', 'artist_name']) else updated_df.columns[:3]
            print(updated_df[sample_cols].head(3).to_string(index=False))
            
            return True
            
        except Exception as e:
            print(f"❌ Error en verificación: {e}")
            return False

def main():
    """Función principal de enriquecimiento"""
    enricher = MetadataEnricher()
    
    # Paso 1: Cargar metadatos técnicos
    technical_df = enricher.load_technical_metadata()
    if technical_df is None:
        print("❌ Proceso abortado: No se pudieron cargar metadatos técnicos")
        return False
    
    # Paso 2: Cargar metadatos descriptivos
    descriptive_df = enricher.load_descriptive_metadata()
    if descriptive_df is None:
        print("❌ Proceso abortado: No se pudieron cargar metadatos descriptivos")
        return False
    
    # Paso 3: Combinar metadatos
    enriched_df = enricher.merge_metadata(technical_df, descriptive_df)
    if enriched_df is None:
        print("❌ Proceso abortado: No se pudieron combinar metadatos")
        return False
    
    # Paso 4: Guardar metadatos enriquecidos
    save_success = enricher.save_enriched_metadata(enriched_df)
    if not save_success:
        print("❌ Proceso abortado: No se pudieron guardar metadatos enriquecidos")
        return False
    
    # Paso 5: Actualizar sistema
    update_success = enricher.update_system_to_use_enriched()
    if not update_success:
        print("❌ Proceso abortado: No se pudo actualizar el sistema")
        return False
    
    # Paso 6: Verificar éxito
    verify_success = enricher.verify_enrichment()
    
    if verify_success:
        print("\n🎉 ENRIQUECIMIENTO COMPLETADO EXITOSAMENTE")
        print("   El sistema de recomendaciones ahora tiene acceso a:")
        print("   - track_name (nombres de canciones)")
        print("   - artist_name (nombres de artistas)")
        print("   - Metadatos técnicos originales preservados")
        print("   - Backup del archivo original creado automáticamente")
    else:
        print("\n❌ ENRIQUECIMIENTO COMPLETADO CON ADVERTENCIAS")
        print("   Revise los mensajes de verificación arriba")
    
    return verify_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)