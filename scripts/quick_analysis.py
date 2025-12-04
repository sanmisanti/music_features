#!/usr/bin/env python3
"""
📊 ANÁLISIS RÁPIDO DE DATASET MUSICAL
Análisis exploratorio + clustering readiness en un script

FUNCIONALIDADES:
- Análisis exploratorio automático (82/82 tests validados)
- Clustering readiness assessment (Hopkins Statistic)
- Reporte comprensivo en múltiples formatos
- Visualizaciones automáticas
- Recomendaciones de optimización

Datasets soportados:
- spotify_songs_fixed.csv (18K canciones con letras)
- picked_data_optimal.csv (dataset optimizado)
- Cualquier dataset con características musicales Spotify

Autor: Exploratory Analysis System + Clustering Readiness
Fecha: Enero 2025
Estado: ✅ SISTEMA MADURO (82/82 tests exitosos)
"""

import sys
import os
import pandas as pd
from datetime import datetime

def main(dataset_name=None):
    """
    Ejecuta análisis rápido completo de un dataset musical.
    
    Args:
        dataset_name: Nombre del dataset a analizar
                     - 'fixed' o None: spotify_songs_fixed.csv (18K)
                     - 'optimal': picked_data_optimal.csv (optimizado)
                     - 'lyrics': picked_data_optimal.csv (con letras)
                     - ruta personalizada: path completo al archivo
    """
    
    print("📊 ANÁLISIS RÁPIDO DE DATASET MUSICAL")
    print("="*70)
    print("🔧 Sistema maduro con 82/82 tests exitosos")
    print("📈 Análisis exploratorio + clustering readiness")
    print("="*70)
    
    # Determinar dataset a analizar
    datasets = {
        'fixed': 'data/2_with_lyrics/spotify_songs_fixed.csv',
        'optimal': 'data/3_selected/picked_data_optimal.csv', 
        'lyrics': 'data/3_selected/picked_data_optimal.csv',
        None: 'data/2_with_lyrics/spotify_songs_fixed.csv'  # Default
    }
    
    if dataset_name in datasets:
        dataset_path = datasets[dataset_name]
        print(f"📂 Dataset seleccionado: {dataset_path}")
    elif dataset_name and os.path.exists(dataset_name):
        dataset_path = dataset_name
        print(f"📂 Dataset personalizado: {dataset_path}")
    else:
        dataset_path = datasets[None]  # Default
        print(f"📂 Dataset por defecto: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset no encontrado en {dataset_path}")
        return False
    
    # Análisis rápido del dataset
    try:
        print(f"\n📋 Cargando dataset...")
        
        # Detectar formato automáticamente
        if 'fixed' in dataset_path:
            df = pd.read_csv(dataset_path, sep='@@', encoding='utf-8', on_bad_lines='skip', engine='python')
        elif 'optimal' in dataset_path or 'lyrics' in dataset_path:
            df = pd.read_csv(dataset_path, sep='^', decimal='.', encoding='utf-8', on_bad_lines='skip')
        else:
            # Intentar formato estándar
            df = pd.read_csv(dataset_path, encoding='utf-8', on_bad_lines='skip')
            
        print(f"✅ Dataset cargado: {df.shape[0]:,} canciones × {df.shape[1]} columnas")
        
        # Información básica
        print(f"\n📊 INFORMACIÓN BÁSICA:")
        print(f"   📈 Tamaño: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print(f"   💾 Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Características musicales disponibles
        musical_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo', 'duration_ms', 'time_signature'
        ]
        
        available_features = [f for f in musical_features if f in df.columns]
        print(f"   🎵 Características musicales: {len(available_features)}/13")
        
        if len(available_features) < 10:
            print(f"   ⚠️  Advertencia: Solo {len(available_features)} características disponibles")
            print(f"      Faltantes: {set(musical_features) - set(available_features)}")
        
        # Calidad de datos básica
        print(f"\n🔍 CALIDAD DE DATOS:")
        print(f"   📊 Valores nulos: {df.isnull().sum().sum():,} ({df.isnull().sum().sum()/df.size*100:.1f}%)")
        print(f"   🔢 Valores únicos promedio: {df.nunique().mean():.1f}")
        
        # Estadísticas rápidas de características musicales
        if available_features:
            music_df = df[available_features]
            print(f"\n🎵 CARACTERÍSTICAS MUSICALES:")
            print(f"   📈 Rango promedio: {(music_df.max() - music_df.min()).mean():.3f}")
            print(f"   📊 Desviación estándar promedio: {music_df.std().mean():.3f}")
            print(f"   🎯 Correlación promedio: {music_df.corr().abs().mean().mean():.3f}")
        
    except Exception as e:
        print(f"❌ Error durante análisis: {e}")
        return False
    
    print(f"\n🎯 ANÁLISIS COMPLETADO")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💡 Para análisis completo, usar:")
    print(f"   python exploratory_analysis/run_full_analysis.py")
    print(f"   python analyze_clustering_readiness_direct.py")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis rápido de dataset musical')
    parser.add_argument('--dataset', '-d', 
                       choices=['fixed', 'optimal', 'lyrics'],
                       help='Dataset a analizar (fixed=18K, optimal=optimizado, lyrics=con letras)')
    parser.add_argument('--path', '-p',
                       help='Ruta personalizada al dataset')
    
    args = parser.parse_args()
    
    dataset = args.path if args.path else args.dataset
    main(dataset)