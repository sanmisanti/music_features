#!/usr/bin/env python3
"""
Script para generar dataset optimizado usando selector mejorado
FASE 1.4 del Plan de Optimización de Clustering

Este script ejecuta la selección optimizada para generar picked_data_optimal.csv
con Hopkins preservation y validación continua.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Añadir paths para importar módulos del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_selection'))

from data_selection.clustering_aware.select_optimal_10k_from_18k import OptimalSelector
from data_selection.clustering_aware.hopkins_validator import HopkinsValidator

def main():
    """Función principal para generar dataset optimizado"""
    
    print("🚀 FASE 1.4: Generando dataset optimizado con selector mejorado")
    print("=" * 70)
    
    # Configuración - rutas relativas que funcionan en Windows y Linux
    base_path = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(base_path, "data", "with_lyrics", "spotify_songs_fixed.csv")
    output_path = os.path.join(base_path, "data", "final_data", "picked_data_optimal.csv")
    target_size = 10000
    
    # Verificar archivo fuente
    if not os.path.exists(source_path):
        print(f"❌ ERROR: Archivo fuente no encontrado: {source_path}")
        return False
    
    # Cargar información del dataset fuente
    try:
        print(f"📊 Cargando dataset fuente: {source_path}")
        source_data = pd.read_csv(source_path, sep='@@', decimal='.', encoding='utf-8')
        print(f"✅ Dataset cargado: {len(source_data)} canciones, {len(source_data.columns)} columnas")
        
        # Mostrar información básica
        print(f"\n📈 Información del dataset fuente:")
        print(f"   - Total canciones: {len(source_data):,}")
        print(f"   - Columnas disponibles: {len(source_data.columns)}")
        
        # Verificar columnas musicales requeridas
        required_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                           'instrumentalness', 'liveness', 'valence', 'tempo']
        
        missing_features = [f for f in required_features if f not in source_data.columns]
        if missing_features:
            print(f"❌ ERROR: Características musicales faltantes: {missing_features}")
            return False
            
        available_features = [f for f in required_features if f in source_data.columns]
        print(f"   - Características musicales disponibles: {len(available_features)}/{len(required_features)}")
        
        # Verificar calidad de datos
        null_counts = source_data[available_features].isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            print(f"⚠️  Valores faltantes encontrados: {total_nulls}")
            for feature, count in null_counts[null_counts > 0].items():
                print(f"     - {feature}: {count} ({count/len(source_data)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ ERROR cargando dataset fuente: {e}")
        return False
    
    # Inicializar selector optimizado
    try:
        print(f"\n🔧 Inicializando selector optimizado...")
        selector = OptimalSelector()
        hopkins_validator = HopkinsValidator()
        print("✅ Selector optimizado inicializado")
        
    except Exception as e:
        print(f"❌ ERROR inicializando selector: {e}")
        return False
    
    # Análisis Hopkins preliminar del dataset completo
    try:
        print(f"\n🧪 Realizando análisis Hopkins preliminar...")
        sample_data = source_data[available_features].dropna().sample(n=min(1000, len(source_data)), random_state=42)
        hopkins_baseline = hopkins_validator.calculate_hopkins_fast(sample_data, sample_size=100)
        print(f"📊 Hopkins Statistic baseline del dataset: {hopkins_baseline:.4f}")
        
        if hopkins_baseline < 0.5:
            print(f"⚠️  Hopkins baseline bajo ({hopkins_baseline:.4f}) - clustering tendency limitada")
        else:
            print(f"✅ Hopkins baseline bueno ({hopkins_baseline:.4f}) - dataset suitable para clustering")
            
    except Exception as e:
        print(f"⚠️  No se pudo calcular Hopkins baseline: {e}")
        hopkins_baseline = 0.0
    
    # Ejecutar selección optimizada
    print(f"\n🎯 Ejecutando selección optimizada...")
    print(f"   - Target size: {target_size:,} canciones")
    print(f"   - Dataset fuente: {len(source_data):,} canciones")
    print(f"   - Porcentaje selección: {target_size/len(source_data)*100:.2f}%")
    
    start_time = time.time()
    
    try:
        # La función maneja automáticamente target_size=10000 según su implementación
        selected_indices, metadata = selector.select_optimal_10k_with_validation(source_data)
        
        execution_time = time.time() - start_time
        
        print(f"\n✅ Selección completada en {execution_time:.1f} segundos")
        print(f"📊 Resultados de la selección:")
        print(f"   - Canciones seleccionadas: {len(selected_indices):,}")
        print(f"   - Hopkins inicial: {metadata.get('hopkins_initial', 'N/A')}")
        print(f"   - Hopkins final: {metadata.get('hopkins_final', 'N/A')}")
        print(f"   - Método utilizado: {metadata.get('selection_method', 'N/A')}")
        print(f"   - Fallback usado: {metadata.get('fallback_used', False)}")
        
        if metadata.get('fallback_used', False):
            print(f"   - Razón fallback: {metadata.get('fallback_reason', 'N/A')}")
        
    except Exception as e:
        print(f"❌ ERROR en selección optimizada: {e}")
        return False
    
    # Generar dataset optimizado
    try:
        print(f"\n💾 Generando dataset optimizado...")
        selected_data = source_data.iloc[selected_indices].copy()
        
        # Verificar calidad del dataset seleccionado
        print(f"📊 Verificación de calidad del dataset seleccionado:")
        print(f"   - Total canciones: {len(selected_data):,}")
        print(f"   - Columnas preservadas: {len(selected_data.columns)}")
        
        # Verificar diversidad musical
        musical_stats = {}
        for feature in available_features:
            if feature in selected_data.columns:
                original_std = source_data[feature].std()
                selected_std = selected_data[feature].std()
                diversity_ratio = selected_std / original_std if original_std > 0 else 0
                musical_stats[feature] = {
                    'original_std': original_std,
                    'selected_std': selected_std, 
                    'diversity_ratio': diversity_ratio
                }
        
        avg_diversity = np.mean([stats['diversity_ratio'] for stats in musical_stats.values()])
        print(f"   - Diversidad musical promedio: {avg_diversity:.3f}")
        
        if avg_diversity < 0.5:
            print(f"⚠️  Diversidad musical baja ({avg_diversity:.3f})")
        else:
            print(f"✅ Diversidad musical buena ({avg_diversity:.3f})")
        
        # Crear directorio de salida si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Guardar dataset optimizado
        selected_data.to_csv(output_path, sep='@@', decimal='.', index=False, encoding='utf-8')
        print(f"✅ Dataset optimizado guardado: {output_path}")
        
    except Exception as e:
        print(f"❌ ERROR generando dataset optimizado: {e}")
        return False
    
    # Generar reporte detallado
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(base_path, "data", "final_data", f"optimization_report_{timestamp}.json")
        
        report = {
            'timestamp': timestamp,
            'phase': 'FASE_1.4_Dataset_Generation',
            'source_dataset': {
                'path': source_path,
                'total_songs': len(source_data),
                'columns': len(source_data.columns),
                'hopkins_baseline': hopkins_baseline
            },
            'selection_process': {
                'target_size': target_size,
                'selected_size': len(selected_indices),
                'execution_time_seconds': execution_time,
                'selection_ratio': len(selected_indices) / len(source_data)
            },
            'hopkins_analysis': {
                'initial': metadata.get('hopkins_initial'),
                'final': metadata.get('hopkins_final'),
                'degradation': (metadata.get('hopkins_initial', 0) - metadata.get('hopkins_final', 0)) / metadata.get('hopkins_initial', 1) if metadata.get('hopkins_initial', 0) > 0 else 0
            },
            'quality_metrics': {
                'selection_method': metadata.get('selection_method'),
                'fallback_used': metadata.get('fallback_used', False),
                'fallback_reason': metadata.get('fallback_reason'),
                'average_musical_diversity': avg_diversity,
                'musical_features_stats': musical_stats
            },
            'output_dataset': {
                'path': output_path,
                'format': 'CSV with @@ separator and . decimal',
                'encoding': 'UTF-8'
            },
            'next_steps': [
                'Ejecutar tests comprehensivos con test_optimal_selector_improved.py',
                'Realizar clustering comparativo (FASE 2)',
                'Evaluar clustering readiness (FASE 3)',
                'Implementar cluster purification (FASE 4)',
                'Análisis final y documentación (FASE 5)'
            ]
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📋 Reporte detallado generado: {report_path}")
        
    except Exception as e:
        print(f"⚠️  Error generando reporte: {e}")
    
    # Resumen final
    print(f"\n🎉 FASE 1.4 COMPLETADA EXITOSAMENTE")
    print(f"=" * 50)
    print(f"✅ Dataset optimizado generado: picked_data_optimal.csv")
    print(f"📊 Selección: {len(selected_indices):,}/{target_size:,} canciones")
    print(f"🧪 Hopkins final: {metadata.get('hopkins_final', 'N/A')}")
    print(f"⏱️  Tiempo ejecución: {execution_time:.1f}s")
    print(f"🎵 Diversidad musical: {avg_diversity:.3f}")
    
    print(f"\n🔄 PRÓXIMOS PASOS:")
    print(f"1. Ejecutar tests: python tests/test_data_selection/test_optimal_selector_improved.py")
    print(f"2. Continuar con FASE 2: Clustering Comparativo")
    print(f"3. Actualizar clustering scripts para usar picked_data_optimal.csv")
    
    return True

if __name__ == '__main__':
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)