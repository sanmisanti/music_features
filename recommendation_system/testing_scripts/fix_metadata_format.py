#!/usr/bin/env python3
"""
Script para Corregir Formato de songs_metadata.csv
Convierte archivo con separador '^' a formato estándar CSV

Uso:
    python fix_metadata_format.py

Propósito:
    - Leer songs_metadata.csv con separador '^' correcto
    - Verificar integridad de los datos
    - Crear backup del archivo original
    - Generar archivo corregido con formato estándar
"""

import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

def fix_metadata_format():
    """Corrige el formato del archivo songs_metadata.csv"""
    
    # Rutas
    system_dir = Path(__file__).parent.parent
    data_dir = system_dir / "data"
    metadata_path = data_dir / "songs_metadata.csv"
    backup_path = data_dir / f"songs_metadata_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print("🔧 REPARACIÓN DE FORMATO - songs_metadata.csv")
    print("=" * 50)
    
    try:
        # 1. Crear backup del archivo original
        print(f"💾 Creando backup: {backup_path.name}")
        shutil.copy2(metadata_path, backup_path)
        print("✅ Backup creado exitosamente")
        
        # 2. Leer archivo con separador correcto
        print("📖 Leyendo archivo con separador '^'")
        df = pd.read_csv(metadata_path, sep='^')
        
        print(f"✅ Archivo leído exitosamente:")
        print(f"   Registros: {len(df)}")
        print(f"   Columnas: {list(df.columns)}")
        
        # 3. Verificar integridad básica
        print("\n🔍 Verificando integridad de datos:")
        
        # Verificar columna track_id
        if 'track_id' in df.columns:
            print("✅ Columna 'track_id' encontrada")
            print(f"   Track IDs únicos: {df['track_id'].nunique()}")
            print(f"   Valores nulos: {df['track_id'].isnull().sum()}")
            
            if df['track_id'].nunique() == len(df):
                print("✅ Todos los track_ids son únicos")
            else:
                print("⚠️  Advertencia: Hay track_ids duplicados")
                
        else:
            print("❌ Error: Columna 'track_id' no encontrada después de corregir separador")
            return False
            
        # Verificar otras columnas esperadas
        expected_columns = ['track_id', 'alignment_index', 'embedding_norm', 
                          'embedding_mean', 'musical_features_norm', 
                          'musical_features_mean', 'primary_genre']
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️  Columnas faltantes: {missing_columns}")
        else:
            print("✅ Todas las columnas esperadas están presentes")
            
        # 4. Mostrar muestra de datos
        print("\n📊 Muestra de datos corregidos:")
        print(df.head(3).to_string())
        
        # 5. Confirmar antes de sobrescribir
        print(f"\n⚠️  ¿Proceder a sobrescribir {metadata_path.name} con formato corregido?")
        print("   El archivo original se mantiene como backup.")
        
        # Para uso automático, proceder sin confirmación
        # En uso interactivo, se podría pedir confirmación aquí
        
        # 6. Guardar archivo corregido
        print("💾 Guardando archivo con formato corregido...")
        df.to_csv(metadata_path, index=False)
        print("✅ Archivo songs_metadata.csv corregido y guardado")
        
        # 7. Verificación final
        print("\n✨ Verificación final:")
        test_df = pd.read_csv(metadata_path)  # Leer sin separador especial
        print(f"   Registros en archivo corregido: {len(test_df)}")
        print(f"   Columnas en archivo corregido: {list(test_df.columns)}")
        
        if 'track_id' in test_df.columns:
            print("✅ Verificación exitosa: columna 'track_id' disponible")
            
            # Mostrar algunos track_ids para verificar
            sample_track_ids = test_df['track_id'].head(3).tolist()
            print(f"   Primeros track_ids: {sample_track_ids}")
            
            return True
        else:
            print("❌ Error: Verificación falló")
            return False
            
    except Exception as e:
        print(f"❌ Error durante la reparación: {e}")
        
        # Restaurar backup si existe
        if backup_path.exists():
            print("🔄 Restaurando backup...")
            shutil.copy2(backup_path, metadata_path)
            print("✅ Backup restaurado")
            
        return False

def main():
    """Función principal"""
    success = fix_metadata_format()
    
    if success:
        print("\n🎉 REPARACIÓN COMPLETADA EXITOSAMENTE")
        print("   El archivo songs_metadata.csv ahora tiene el formato correcto")
        print("   El sistema de recomendaciones debería funcionar correctamente")
    else:
        print("\n❌ REPARACIÓN FALLIDA")
        print("   El archivo mantiene el formato original")
        print("   Se requiere investigación adicional")
    
    return success

if __name__ == "__main__":
    main()