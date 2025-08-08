#!/usr/bin/env python3
"""
Script para seleccionar 10K canciones óptimas desde el dataset de 18K

Este script implementa la estrategia clustering-aware recomendada basada en 
el análisis de clustering readiness, preservando la estructura natural 
identificada en spotify_songs_fixed.csv.

Estrategia:
1. Pre-clustering del dataset 18K para identificar estructura natural (K=2)
2. Selección proporcional respetando clusters naturales
3. Muestreo diverso dentro de cada cluster usando top características
4. Preservación de distribuciones naturales

Uso:
    python data_selection/clustering_aware/select_optimal_10k_from_18k.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class OptimalSelector:
    """Selector optimizado que preserva estructura natural de clustering."""
    
    def __init__(self):
        self.musical_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo', 'duration_ms'  # time_signature no está en dataset 18K
        ]
        
        # Top características según clustering readiness analysis
        self.top_features = [
            'instrumentalness',  # Top 1 - Mayor poder discriminativo
            'liveness',          # Top 2 - Alta varianza
            'duration_ms',       # Top 3 - Diversidad temporal
            'energy',            # Top 4 - Característica clave
            'danceability'       # Top 5 - Característica principal
        ]
    
    def load_source_dataset(self):
        """Cargar dataset fuente de 18K canciones."""
        dataset_path = 'data/with_lyrics/spotify_songs_fixed.csv'
        
        if not os.path.exists(dataset_path):
            print(f"❌ ERROR: Dataset fuente no encontrado en {dataset_path}")
            return None
        
        try:
            print("📂 Cargando dataset fuente de 18K canciones...")
            df = pd.read_csv(dataset_path, sep='@@', encoding='utf-8', 
                           on_bad_lines='skip', engine='python')
            
            print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            
            # Verificar características musicales disponibles
            available_features = [f for f in self.musical_features if f in df.columns]
            missing_features = [f for f in self.musical_features if f not in df.columns]
            
            print(f"🎵 Características disponibles: {len(available_features)}/{len(self.musical_features)}")
            if missing_features:
                print(f"⚠️  Características faltantes: {', '.join(missing_features)}")
            
            self.available_features = available_features
            return df
            
        except Exception as e:
            print(f"❌ ERROR cargando dataset: {e}")
            return None
    
    def prepare_clustering_data(self, df):
        """Preparar datos para pre-clustering."""
        print("\n🔧 PREPARANDO DATOS PARA PRE-CLUSTERING")
        print("-" * 50)
        
        # Extraer características musicales
        X = df[self.available_features].copy()
        original_size = len(X)
        
        # Limpiar datos nulos
        X = X.dropna()
        cleaned_size = len(X)
        
        print(f"🧹 Limpieza: {original_size:,} → {cleaned_size:,} filas (-{original_size-cleaned_size:,} nulos)")
        
        if cleaned_size < 1000:
            print("❌ ERROR: Dataset muy pequeño después de limpieza")
            return None, None
        
        # Normalizar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Mantener índices para mapping posterior
        clean_indices = X.index.tolist()
        
        print(f"✅ Datos preparados: {X_scaled.shape[0]:,} canciones × {X_scaled.shape[1]} características")
        
        return X_scaled, clean_indices
    
    def identify_natural_clusters(self, X_scaled, clean_indices):
        """Identificar clusters naturales usando K=2 (óptimo según análisis)."""
        print("\n🎯 IDENTIFICANDO CLUSTERS NATURALES")
        print("-" * 50)
        
        # Pre-clustering con K=2 (óptimo identificado en análisis)
        print("🔍 Ejecutando K-Means con K=2...")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calcular métricas de calidad
        silhouette = silhouette_score(X_scaled, cluster_labels)
        
        print(f"📊 Silhouette Score: {silhouette:.3f}")
        
        # Analizar distribución de clusters
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"📋 Distribución de clusters:")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(cluster_labels) * 100
            print(f"   Cluster {label}: {count:,} canciones ({percentage:.1f}%)")
        
        return cluster_labels, silhouette
    
    def diverse_sampling_within_cluster(self, cluster_data, target_size, feature_data):
        """Muestreo diverso dentro de un cluster usando top características."""
        if len(cluster_data) <= target_size:
            return cluster_data
        
        # Usar top características para diversidad
        available_top_features = [f for f in self.top_features if f in cluster_data.columns]
        
        if not available_top_features:
            # Fallback a muestreo aleatorio si no hay top features
            return cluster_data.sample(n=target_size, random_state=42)
        
        # MaxMin sampling basado en top características
        feature_subset = cluster_data[available_top_features].values
        scaler = StandardScaler()
        feature_subset_scaled = scaler.fit_transform(feature_subset)
        
        # Selección inicial aleatoria
        selected_indices = [np.random.randint(len(feature_subset_scaled))]
        selected_features = [feature_subset_scaled[selected_indices[0]]]
        
        # MaxMin algorithm
        while len(selected_indices) < target_size:
            distances = []
            
            for i, candidate in enumerate(feature_subset_scaled):
                if i in selected_indices:
                    distances.append(-1)  # Ya seleccionado
                    continue
                
                # Distancia mínima a puntos ya seleccionados
                min_distance = float('inf')
                for selected in selected_features:
                    distance = np.linalg.norm(candidate - selected)
                    min_distance = min(min_distance, distance)
                
                distances.append(min_distance)
            
            # Seleccionar punto más lejano
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
            selected_features.append(feature_subset_scaled[next_idx])
        
        return cluster_data.iloc[selected_indices]
    
    def select_optimal_10k(self, df):
        """Ejecutar selección completa de 10K canciones optimizada."""
        print("\n🚀 SELECCIÓN OPTIMIZADA DE 10K CANCIONES")
        print("=" * 60)
        
        # 1. Preparar datos para clustering
        X_scaled, clean_indices = self.prepare_clustering_data(df)
        if X_scaled is None:
            return None
        
        # 2. Identificar estructura natural
        cluster_labels, silhouette = self.identify_natural_clusters(X_scaled, clean_indices)
        
        # 3. Preparar DataFrame con clusters
        df_clean = df.loc[clean_indices].copy()
        df_clean['natural_cluster'] = cluster_labels
        
        print(f"\n📊 SELECCIÓN POR CLUSTERS")
        print("-" * 50)
        
        # 4. Calcular selección proporcional
        target_total = 10000
        selected_parts = []
        
        for cluster_id in sorted(df_clean['natural_cluster'].unique()):
            cluster_data = df_clean[df_clean['natural_cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            # Proporción basada en tamaño natural del cluster
            proportion = cluster_size / len(df_clean)
            target_size = int(target_total * proportion)
            
            print(f"🎵 Cluster {cluster_id}:")
            print(f"   Tamaño original: {cluster_size:,} canciones ({proportion:.1%})")
            print(f"   Target selección: {target_size:,} canciones")
            
            # 5. Muestreo diverso dentro del cluster
            selected_cluster = self.diverse_sampling_within_cluster(
                cluster_data, target_size, X_scaled
            )
            
            selected_parts.append(selected_cluster)
            print(f"   ✅ Seleccionadas: {len(selected_cluster):,} canciones")
        
        # 6. Combinar selecciones
        final_selection = pd.concat(selected_parts, ignore_index=True)
        
        # Ajustar tamaño si es necesario
        if len(final_selection) != target_total:
            if len(final_selection) > target_total:
                final_selection = final_selection.sample(n=target_total, random_state=42)
            else:
                # Completar con muestreo adicional si faltan
                remaining = target_total - len(final_selection)
                available = df_clean[~df_clean.index.isin(final_selection.index)]
                additional = available.sample(n=remaining, random_state=42)
                final_selection = pd.concat([final_selection, additional], ignore_index=True)
        
        print(f"\n✅ SELECCIÓN COMPLETADA")
        print(f"📦 Dataset final: {len(final_selection):,} canciones")
        print(f"📊 Silhouette del pre-clustering: {silhouette:.3f}")
        
        return final_selection
    
    def validate_selection(self, selected_df, original_df):
        """Validar calidad de la selección."""
        print(f"\n🔍 VALIDACIÓN DE LA SELECCIÓN")
        print("-" * 50)
        
        # Comparar distribuciones de top características
        print("📊 Comparación de distribuciones (original vs seleccionada):")
        
        for feature in self.top_features[:3]:  # Top 3 para no saturar output
            if feature not in selected_df.columns:
                continue
                
            orig_mean = original_df[feature].mean()
            sel_mean = selected_df[feature].mean()
            orig_std = original_df[feature].std()
            sel_std = selected_df[feature].std()
            
            print(f"   {feature}:")
            print(f"     Original: μ={orig_mean:.3f}, σ={orig_std:.3f}")
            print(f"     Seleccionado: μ={sel_mean:.3f}, σ={sel_std:.3f}")
            print(f"     Diferencia: Δμ={abs(orig_mean-sel_mean):.3f}, Δσ={abs(orig_std-sel_std):.3f}")
        
        # Verificar diversidad de géneros
        if 'playlist_genre' in selected_df.columns:
            orig_genres = set(original_df['playlist_genre'].unique())
            sel_genres = set(selected_df['playlist_genre'].unique())
            
            print(f"\n🎵 Diversidad de géneros:")
            print(f"   Original: {len(orig_genres)} géneros únicos")
            print(f"   Seleccionado: {len(sel_genres)} géneros únicos")
            print(f"   Conservación: {len(sel_genres)/len(orig_genres)*100:.1f}%")
        
        print(f"\n✅ Validación completada")
    
    def save_dataset(self, selected_df):
        """Guardar dataset seleccionado."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio de salida
        output_dir = "data/final_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar CSV principal
        output_file = f"{output_dir}/picked_data_optimal_{timestamp}.csv"
        selected_df.to_csv(output_file, sep='^', decimal='.', index=False, encoding='utf-8')
        
        # También crear versión con nombre estándar
        standard_file = f"{output_dir}/picked_data_optimal.csv"
        selected_df.to_csv(standard_file, sep='^', decimal='.', index=False, encoding='utf-8')
        
        print(f"\n💾 DATASETS GUARDADOS:")
        print(f"📁 Con timestamp: {output_file}")
        print(f"📁 Versión estándar: {standard_file}")
        
        # Guardar metadatos
        metadata = {
            'timestamp': timestamp,
            'source_dataset': 'data/with_lyrics/spotify_songs_fixed.csv',
            'source_size': 18454,
            'selected_size': len(selected_df),
            'selection_method': 'clustering_aware_optimal',
            'features_used': self.available_features,
            'top_features_prioritized': self.top_features,
            'format': {
                'separator': '^',
                'decimal': '.',
                'encoding': 'utf-8'
            }
        }
        
        metadata_file = f"{output_dir}/picked_data_optimal_metadata_{timestamp}.json"
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Metadatos: {metadata_file}")
        
        return standard_file

def main():
    """Función principal del selector optimizado."""
    print("🎯 SELECTOR OPTIMIZADO DE 10K CANCIONES DESDE 18K")
    print("=" * 60)
    print("Estrategia: Clustering-aware preservando estructura natural")
    print("Basado en: Análisis clustering readiness (Hopkins=0.823)")
    print()
    
    # Inicializar selector
    selector = OptimalSelector()
    
    # 1. Cargar dataset fuente
    df_18k = selector.load_source_dataset()
    if df_18k is None:
        print("❌ ERROR: No se pudo cargar el dataset fuente")
        return
    
    # 2. Selección optimizada
    selected_10k = selector.select_optimal_10k(df_18k)
    if selected_10k is None:
        print("❌ ERROR: Falló la selección optimizada")
        return
    
    # 3. Validación
    selector.validate_selection(selected_10k, df_18k)
    
    # 4. Guardar resultado
    output_file = selector.save_dataset(selected_10k)
    
    # 5. Resumen final
    print(f"\n🎉 SELECCIÓN OPTIMIZADA COMPLETADA")
    print("=" * 60)
    print(f"📊 Dataset final: {len(selected_10k):,} canciones")
    print(f"📁 Archivo: {output_file}")
    print(f"🔄 Formato: Separador '^', decimal '.', UTF-8")
    print(f"🎯 Listo para clustering con métricas mejoradas esperadas:")
    print(f"   • Hopkins Statistic esperado: 0.75-0.80")
    print(f"   • Clustering Readiness esperado: 75-80/100")
    print(f"   • Silhouette Score esperado: 0.140-0.180")
    print()
    print("📋 SIGUIENTE PASO: Ejecutar clustering con nuevo dataset")
    print("   python clustering/algorithms/musical/clustering_optimized.py")

if __name__ == "__main__":
    main()