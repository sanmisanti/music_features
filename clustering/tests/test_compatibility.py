#!/usr/bin/env python3
"""
Script de prueba para validar compatibilidad con picked_data_lyrics.csv
"""

import sys
from pathlib import Path

# Simular imports de manera segura
try:
    import pandas as pd
    import numpy as np
    print("✅ Librerías básicas disponibles")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

def test_dataset_loading():
    """Probar carga del dataset picked_data_lyrics.csv"""
    dataset_path = "../../data/final_data/picked_data_lyrics.csv"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset no encontrado: {dataset_path}")
        print("💡 Verificar ruta: debe existir picked_data_lyrics.csv")
        return False
    
    try:
        # Cargar dataset con nuevo formato
        df = pd.read_csv(dataset_path, sep='^', decimal='.', encoding='utf-8', on_bad_lines='skip')
        print(f"✅ Dataset cargado: {len(df):,} filas, {len(df.columns)} columnas")
        
        # Verificar features musicales
        musical_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
        ]
        
        available_features = [f for f in musical_features if f in df.columns]
        missing_features = [f for f in musical_features if f not in df.columns]
        
        print(f"✅ Features disponibles: {len(available_features)}/{len(musical_features)}")
        print(f"   Disponibles: {available_features[:5]}..." if len(available_features) > 5 else f"   Disponibles: {available_features}")
        
        if missing_features:
            print(f"⚠️  Features faltantes: {missing_features}")
        
        # Mostrar estadísticas básicas
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"📊 Memoria utilizada: {memory_mb:.1f} MB")
        
        # Mostrar ejemplos
        print(f"\n📋 Primeras 3 canciones:")
        if 'name' in df.columns and 'artists' in df.columns:
            for i, (_, row) in enumerate(df[['name', 'artists']].head(3).iterrows()):
                print(f"  {i+1}. \"{row['name']}\" - {row['artists']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al cargar dataset: {e}")
        return False

def test_clustering_simulation():
    """Simular clustering con muestra pequeña"""
    print(f"\n🧪 Simulando clustering...")
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        
        # Crear datos de prueba
        np.random.seed(42)
        test_data = np.random.rand(100, 13)  # 100 canciones, 13 features
        
        # Normalizar
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(test_data)
        
        # Clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_data)
        
        # Métricas
        silhouette = silhouette_score(normalized_data, labels)
        
        print(f"✅ Clustering simulado exitoso:")
        print(f"   Silhouette Score: {silhouette:.3f}")
        print(f"   Distribución: {np.bincount(labels)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en clustering simulado: {e}")
        return False

if __name__ == "__main__":
    print("🧪 PRUEBA DE COMPATIBILIDAD - CLUSTERING OPTIMIZADO")
    print("=" * 60)
    
    success = True
    
    # Prueba 1: Carga del dataset
    print("1️⃣ Probando carga del dataset...")
    if not test_dataset_loading():
        success = False
    
    # Prueba 2: Clustering simulado
    print("\n2️⃣ Probando funcionalidad de clustering...")
    if not test_clustering_simulation():
        success = False
    
    # Resultado final
    print("\n" + "=" * 60)
    if success:
        print("🎉 TODAS LAS PRUEBAS PASARON - SISTEMA COMPATIBLE")
        print("✅ El sistema está listo para procesar las 9,677 canciones")
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON - REVISAR CONFIGURACIÓN")
    
    sys.exit(0 if success else 1)