#!/usr/bin/env python3
"""
Análisis profundo del dataset picked_data_optimal.csv enfocado en características musicales
para la implementación de clustering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Características musicales para clustering (sin time_signature según el README original)
MUSICAL_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms'
]

def load_dataset():
    """Cargar el dataset con letras."""
    dataset_path = Path("data/3_selected/picked_data_optimal.csv")
    
    print("🔍 Cargando dataset...")
    df = pd.read_csv(dataset_path, sep='^', encoding='utf-8')
    print(f"✅ Dataset cargado: {df.shape[0]:,} canciones × {df.shape[1]} columnas")
    print(f"💾 Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df

def analyze_dataset_structure(df):
    """Analizar estructura general del dataset."""
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE ESTRUCTURA DEL DATASET")
    print("="*60)
    
    print(f"🔢 Dimensiones totales: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"🎵 Canciones con letras: {df['lyrics'].notna().sum():,}")
    print(f"📝 Cobertura de letras: {df['lyrics'].notna().sum() / len(df) * 100:.1f}%")
    
    print(f"\n📋 Columnas disponibles ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype_str = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        print(f"{i:2d}. {col:25s} | {dtype_str:10s} | {null_count:4d} nulos")

def analyze_musical_features(df):
    """Analizar características musicales específicas."""
    print("\n" + "="*60)
    print("🎼 ANÁLISIS DE CARACTERÍSTICAS MUSICALES")
    print("="*60)
    
    # Verificar disponibilidad de features
    available_features = [f for f in MUSICAL_FEATURES if f in df.columns]
    missing_features = [f for f in MUSICAL_FEATURES if f not in df.columns]
    
    print(f"✅ Features disponibles: {len(available_features)}/{len(MUSICAL_FEATURES)}")
    print(f"🎯 Features para clustering: {available_features}")
    
    if missing_features:
        print(f"❌ Features faltantes: {missing_features}")
    
    # Análisis estadístico por feature
    print(f"\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
    print("-" * 80)
    
    musical_df = df[available_features]
    stats_df = musical_df.describe()
    
    for feature in available_features:
        data = df[feature]
        null_count = data.isnull().sum()
        null_pct = null_count / len(df) * 100
        
        print(f"\n🎵 {feature.upper()}")
        print(f"   Tipo: {data.dtype}")
        print(f"   Nulos: {null_count} ({null_pct:.1f}%)")
        print(f"   Rango: [{data.min():.3f}, {data.max():.3f}]")
        print(f"   Media: {data.mean():.3f} ± {data.std():.3f}")
        print(f"   Mediana: {data.median():.3f}")
        
        # Detectar outliers usando IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        outlier_pct = outliers / len(data) * 100
        
        print(f"   Outliers: {outliers} ({outlier_pct:.1f}%)")

def analyze_feature_distributions(df):
    """Analizar distribuciones de características musicales."""
    print("\n" + "="*60)
    print("📈 ANÁLISIS DE DISTRIBUCIONES")
    print("="*60)
    
    available_features = [f for f in MUSICAL_FEATURES if f in df.columns]
    
    for feature in available_features:
        data = df[feature].dropna()
        
        # Test de normalidad
        if len(data) > 5000:
            # Para datasets grandes, usar muestra
            sample_data = data.sample(5000, random_state=42)
        else:
            sample_data = data
            
        shapiro_stat, shapiro_p = stats.shapiro(sample_data)
        is_normal = shapiro_p > 0.05
        
        # Skewness y Kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        print(f"\n🎵 {feature.upper()}:")
        print(f"   Normalidad: {'✅ Normal' if is_normal else '❌ No normal'} (p={shapiro_p:.3f})")
        print(f"   Asimetría: {skewness:.3f} ({'Simétrica' if abs(skewness) < 0.5 else 'Asimétrica'})")
        print(f"   Curtosis: {kurtosis:.3f}")

def analyze_feature_correlations(df):
    """Analizar correlaciones entre características musicales."""
    print("\n" + "="*60)
    print("🔗 ANÁLISIS DE CORRELACIONES")
    print("="*60)
    
    available_features = [f for f in MUSICAL_FEATURES if f in df.columns]
    musical_df = df[available_features]
    
    # Calcular matriz de correlación
    corr_matrix = musical_df.corr()
    
    # Encontrar correlaciones altas
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:  # Umbral de correlación alta
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_val
                ))
    
    print(f"🔍 Correlaciones altas encontradas (|r| > 0.5): {len(high_corr_pairs)}")
    
    if high_corr_pairs:
        print("\n📊 CORRELACIONES SIGNIFICATIVAS:")
        print("-" * 50)
        for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            direction = "positiva" if corr_val > 0 else "negativa"
            strength = "muy fuerte" if abs(corr_val) > 0.8 else "fuerte" if abs(corr_val) > 0.6 else "moderada"
            print(f"   {feat1:15s} ↔ {feat2:15s}: {corr_val:6.3f} ({strength} {direction})")
    else:
        print("✅ No hay correlaciones altas entre features (bueno para clustering)")

def check_data_quality(df):
    """Verificar calidad de los datos musicales."""
    print("\n" + "="*60)
    print("🔍 ANÁLISIS DE CALIDAD DE DATOS")
    print("="*60)
    
    available_features = [f for f in MUSICAL_FEATURES if f in df.columns]
    
    # Rangos válidos para características de Spotify
    feature_ranges = {
        'danceability': (0, 1),
        'energy': (0, 1),
        'speechiness': (0, 1),
        'acousticness': (0, 1),
        'instrumentalness': (0, 1),
        'liveness': (0, 1),
        'valence': (0, 1),
        'key': (0, 11),
        'mode': (0, 1),
        'tempo': (30, 300),
        'loudness': (-80, 10),
        'duration_ms': (10000, 1800000),  # 10 seg a 30 min
    }
    
    quality_issues = []
    
    for feature in available_features:
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            data = df[feature]
            
            # Valores fuera de rango
            out_of_range = ((data < min_val) | (data > max_val)).sum()
            if out_of_range > 0:
                quality_issues.append(f"{feature}: {out_of_range} valores fuera de rango [{min_val}, {max_val}]")
            
            # Valores nulos
            null_count = data.isnull().sum()
            if null_count > 0:
                quality_issues.append(f"{feature}: {null_count} valores nulos")
    
    if quality_issues:
        print("⚠️  PROBLEMAS DE CALIDAD ENCONTRADOS:")
        for issue in quality_issues:
            print(f"   🔸 {issue}")
    else:
        print("✅ Datos de alta calidad: sin valores fuera de rango ni nulos")

def analyze_genre_distribution(df):
    """Analizar distribución por géneros."""
    print("\n" + "="*60)
    print("🎭 ANÁLISIS POR GÉNEROS")
    print("="*60)
    
    if 'playlist_genre' in df.columns:
        genre_counts = df['playlist_genre'].value_counts()
        print(f"🎵 Géneros únicos: {len(genre_counts)}")
        print(f"📊 Distribución de géneros (top 10):")
        print("-" * 40)
        
        for genre, count in genre_counts.head(10).items():
            percentage = count / len(df) * 100
            print(f"   {genre:20s}: {count:4d} ({percentage:5.1f}%)")
        
        # Análisis de características por género (top 5)
        print(f"\n🎼 CARACTERÍSTICAS PROMEDIO POR GÉNERO:")
        print("-" * 60)
        
        top_genres = genre_counts.head(5).index
        available_features = [f for f in MUSICAL_FEATURES if f in df.columns]
        
        for genre in top_genres:
            genre_data = df[df['playlist_genre'] == genre]
            print(f"\n🎵 {genre.upper()} ({len(genre_data)} canciones):")
            
            for feature in available_features[:6]:  # Mostrar solo 6 features principales
                avg_val = genre_data[feature].mean()
                std_val = genre_data[feature].std()
                print(f"   {feature:15s}: {avg_val:.3f} ± {std_val:.3f}")

def generate_clustering_recommendations(df):
    """Generar recomendaciones para clustering."""
    print("\n" + "="*60)
    print("🎯 RECOMENDACIONES PARA CLUSTERING")
    print("="*60)
    
    available_features = [f for f in MUSICAL_FEATURES if f in df.columns]
    musical_df = df[available_features]
    
    print(f"📊 CONFIGURACIÓN RECOMENDADA:")
    print(f"   🎵 Dataset: {len(df):,} canciones")
    print(f"   🔧 Features: {len(available_features)} características musicales")
    print(f"   📏 Normalización: StandardScaler (requerida)")
    print(f"   🎯 Algoritmo: K-Means con optimización automática de K")
    
    # Estimar K óptimo basado en el tamaño del dataset
    # Regla empírica: K ≈ sqrt(n/2) pero limitado a un rango práctico
    n = len(df)
    k_estimate = int(np.sqrt(n / 2))
    k_min = max(3, min(k_estimate - 2, 5))
    k_max = min(k_estimate + 3, 12)
    
    print(f"   🔍 Rango K recomendado: {k_min}-{k_max} (estimado: {k_estimate})")
    print(f"   ⚡ Algoritmo optimizado: MiniBatchKMeans para >5000 muestras")
    print(f"   🎲 Random state: 42 (reproducibilidad)")
    
    # Recomendaciones de PCA
    print(f"\n🔬 ANÁLISIS DIMENSIONAL:")
    print(f"   📊 Dimensiones originales: {len(available_features)}D")
    print(f"   🔄 PCA recomendado: Sí (experiencia previa exitosa)")
    print(f"   📉 Componentes sugeridos: 5-8 (balance calidad/interpretabilidad)")
    
    # Métricas de evaluación
    print(f"\n📈 MÉTRICAS DE EVALUACIÓN:")
    print(f"   🎯 Principal: Silhouette Score (objetivo: >0.314)")
    print(f"   🔗 Adicional: Calinski-Harabasz Index")
    print(f"   📊 Referencia: Davies-Bouldin Index (menor es mejor)")
    
    # Consideraciones especiales
    print(f"\n⚠️  CONSIDERACIONES ESPECIALES:")
    
    # Verificar si hay características categóricas
    categorical_features = []
    for feature in available_features:
        unique_vals = df[feature].nunique()
        if unique_vals <= 12:  # Posiblemente categórica
            categorical_features.append(f"{feature} ({unique_vals} valores únicos)")
    
    if categorical_features:
        print(f"   🔸 Features categóricas detectadas:")
        for cat_feat in categorical_features:
            print(f"     - {cat_feat}")
    
    # Verificar distribución de géneros
    if 'playlist_genre' in df.columns:
        genre_count = df['playlist_genre'].nunique()
        print(f"   🎭 Géneros disponibles: {genre_count} (usar para validación)")
    
    print(f"\n✅ PASOS SIGUIENTES:")
    print(f"   1. Implementar carga con separador '^'")
    print(f"   2. Aplicar StandardScaler a features musicales")
    print(f"   3. Probar clustering con y sin PCA")
    print(f"   4. Comparar con modelos baseline existentes")
    print(f"   5. Validar clusters usando géneros como referencia")

def main():
    """Función principal del análisis."""
    print("🎵 ANÁLISIS PROFUNDO DEL DATASET PARA CLUSTERING MUSICAL")
    print("=" * 70)
    
    try:
        # Cargar dataset
        df = load_dataset()
        
        # Realizar análisis
        analyze_dataset_structure(df)
        analyze_musical_features(df)
        analyze_feature_distributions(df)
        analyze_feature_correlations(df)
        check_data_quality(df)
        analyze_genre_distribution(df)
        generate_clustering_recommendations(df)
        
        print(f"\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())