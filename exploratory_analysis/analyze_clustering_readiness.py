#!/usr/bin/env python3
"""
Script para analizar clustering readiness del dataset spotify_songs_fixed.csv

Este script evalúa qué tan adecuado es el dataset de 18K canciones para clustering
y proporciona recomendaciones específicas para optimizar la selección de 10K canciones.

Uso:
    python analyze_clustering_readiness.py
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime

# Añadir el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Importar directamente sin pasar por __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "clustering_readiness", 
    os.path.join(os.path.dirname(__file__), "feature_analysis", "clustering_readiness.py")
)
clustering_readiness_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clustering_readiness_module)
ClusteringReadiness = clustering_readiness_module.ClusteringReadiness

def load_dataset():
    """Cargar el dataset spotify_songs_fixed.csv."""
    dataset_path = '../data/with_lyrics/spotify_songs_fixed.csv'
    
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset no encontrado en {dataset_path}")
        return None
    
    try:
        # Intentar diferentes separadores
        separators = ['@@', '^', ',', ';', '\t']
        
        for sep in separators:
            try:
                print(f"🔍 Intentando cargar con separador '{sep}'...")
                df = pd.read_csv(dataset_path, sep=sep, encoding='utf-8', 
                               on_bad_lines='skip', low_memory=False)
                
                # Verificar que tenemos suficientes columnas y filas
                if len(df.columns) > 10 and len(df) > 1000:
                    print(f"✅ Dataset cargado exitosamente con separador '{sep}'")
                    print(f"📊 Dimensiones: {df.shape}")
                    print(f"📋 Columnas: {list(df.columns[:10])}..." if len(df.columns) > 10 else f"📋 Columnas: {list(df.columns)}")
                    return df
                    
            except Exception as e:
                print(f"❌ Falló con separador '{sep}': {e}")
                continue
        
        print("❌ ERROR: No se pudo cargar el dataset con ningún separador")
        return None
        
    except Exception as e:
        print(f"❌ ERROR cargando dataset: {e}")
        return None

def analyze_dataset_info(df):
    """Analizar información básica del dataset."""
    print("\n" + "="*60)
    print("📊 INFORMACIÓN BÁSICA DEL DATASET")
    print("="*60)
    
    print(f"📦 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"💾 Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Identificar características musicales disponibles
    musical_features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'duration_ms', 'time_signature'
    ]
    
    available_features = [f for f in musical_features if f in df.columns]
    missing_features = [f for f in musical_features if f not in df.columns]
    
    print(f"\n🎵 CARACTERÍSTICAS MUSICALES:")
    print(f"✅ Disponibles ({len(available_features)}): {', '.join(available_features)}")
    if missing_features:
        print(f"❌ Faltantes ({len(missing_features)}): {', '.join(missing_features)}")
    
    # Calidad de datos básica
    print(f"\n🔍 CALIDAD DE DATOS:")
    total_nulls = df[available_features].isnull().sum().sum()
    print(f"📊 Valores nulos totales: {total_nulls:,}")
    
    if total_nulls > 0:
        print("📋 Valores nulos por característica:")
        for feature in available_features:
            nulls = df[feature].isnull().sum()
            if nulls > 0:
                print(f"   {feature}: {nulls:,} ({nulls/len(df)*100:.1f}%)")
    
    return available_features

def run_clustering_readiness_analysis(df):
    """Ejecutar análisis completo de clustering readiness."""
    print("\n" + "="*60)
    print("🧮 ANÁLISIS DE CLUSTERING READINESS")
    print("="*60)
    
    # Inicializar analizador
    analyzer = ClusteringReadiness()
    
    # Ejecutar análisis completo
    print("⚙️ Calculando clustering readiness score...")
    results = analyzer.calculate_clustering_readiness_score(df)
    
    if results.get('error'):
        print(f"❌ ERROR: {results['message']}")
        return None
    
    return results

def display_results(results):
    """Mostrar resultados del análisis de forma organizada."""
    print(f"\n🎯 CLUSTERING READINESS SCORE: {results['readiness_score']}/100")
    print(f"📊 NIVEL: {results['readiness_level'].upper()}")
    
    # Desglose del score
    print(f"\n📋 DESGLOSE DEL SCORE:")
    breakdown = results['score_breakdown']
    for component, score in breakdown.items():
        print(f"   {component.replace('_', ' ').title()}: {score:.1f} puntos")
    
    # Análisis de clustering tendency
    print(f"\n🔍 CLUSTERING TENDENCY:")
    tendency = results['component_analysis']['clustering_tendency']
    print(f"   Hopkins Statistic: {tendency['hopkins_statistic']:.3f}")
    print(f"   Interpretación: {tendency['interpretation']}")
    print(f"   Confianza: {tendency['confidence_level']}")
    print(f"   Tamaño muestra: {tendency['sample_size']:,}")
    
    # Análisis de separabilidad
    print(f"\n📐 ANÁLISIS DE SEPARABILIDAD:")
    separability = results['component_analysis']['separability_analysis']
    print(f"   Score de separabilidad: {separability['separability_score']:.3f}")
    print(f"   Silhouette esperado: {separability['expected_silhouette_range']}")
    print(f"   Riesgo de overlap: {separability['overlap_risk']}")
    
    # Análisis de características
    print(f"\n🎵 ANÁLISIS DE CARACTERÍSTICAS:")
    features = results['component_analysis']['feature_analysis']
    print(f"   Características analizadas: {features['n_features_analyzed']}")
    print(f"   Top 5 recomendadas: {', '.join(features['recommended_features'][:5])}")
    
    if features['redundant_features']:
        print(f"   Características redundantes: {', '.join(features['redundant_features'])}")
    
    # Recomendaciones de mejora
    print(f"\n💡 RECOMENDACIONES DE MEJORA:")
    for i, suggestion in enumerate(results['improvement_suggestions'], 1):
        print(f"   {i}. {suggestion}")

def save_results(results, df):
    """Guardar resultados en archivo JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear directorio de resultados si no existe
    results_dir = "../outputs/clustering_readiness"
    os.makedirs(results_dir, exist_ok=True)
    
    # Preparar datos para guardar
    output_data = {
        'analysis_timestamp': timestamp,
        'dataset_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        },
        'clustering_readiness_results': results
    }
    
    # Guardar JSON
    output_file = f"{results_dir}/clustering_readiness_analysis_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Resultados guardados en: {output_file}")
    return output_file

def main():
    """Función principal del análisis."""
    print("🎵 ANÁLISIS DE CLUSTERING READINESS - DATASET SPOTIFY SONGS")
    print("=" * 60)
    
    # 1. Cargar dataset
    print("📂 Cargando dataset...")
    df = load_dataset()
    if df is None:
        return
    
    # 2. Información básica
    available_features = analyze_dataset_info(df)
    
    # 3. Análisis de clustering readiness
    results = run_clustering_readiness_analysis(df)
    if results is None:
        return
    
    # 4. Mostrar resultados
    display_results(results)
    
    # 5. Guardar resultados
    output_file = save_results(results, df)
    
    # 6. Resumen final
    print(f"\n" + "="*60)
    print("📊 RESUMEN EJECUTIVO")
    print("="*60)
    
    score = results['readiness_score']
    level = results['readiness_level']
    
    if score >= 80:
        print("🏆 EXCELENTE: Dataset óptimo para clustering")
        print("✅ Proceder con selección de 10K canciones usando criterios estándar")
    elif score >= 60:
        print("✅ BUENO: Dataset adecuado para clustering con optimización")
        print("💡 Aplicar recomendaciones antes de seleccionar 10K canciones")
    elif score >= 40:
        print("⚠️  ACEPTABLE: Dataset requiere mejoras significativas")
        print("🔧 Implementar feature engineering antes del clustering")
    else:
        print("❌ PROBLEMÁTICO: Dataset inadecuado para clustering")
        print("🚨 Considerar cambiar estrategia de selección o algoritmo")
    
    print(f"\n📁 Archivo de resultados: {output_file}")
    print("🎯 ¡Análisis completado exitosamente!")

if __name__ == "__main__":
    main()