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

# Importar Hopkins Validator para validación continua
from .hopkins_validator import HopkinsValidator

class OptimalSelector:
    """Selector optimizado que preserva estructura natural de clustering."""
    
    def __init__(self, hopkins_threshold=0.70):
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
        
        # Inicializar validador Hopkins con threshold configurable
        self.hopkins_validator = HopkinsValidator(threshold=hopkins_threshold)
        self.selection_metadata = {
            'hopkins_validations': [],
            'fallback_activations': 0,
            'diversity_fallbacks': 0
        }
    
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
    
    def improved_initial_selection(self, feature_subset_scaled):
        """
        Selección inicial científica para MaxMin sampling.
        
        En lugar de selección aleatoria, usa el punto más lejano del centroide
        del cluster para maximizar diversidad inicial.
        
        Args:
            feature_subset_scaled: Características normalizadas del cluster
            
        Returns:
            int: Índice del punto inicial óptimo
        """
        if len(feature_subset_scaled) == 0:
            return 0
        
        # Calcular centroide del cluster
        centroid = np.mean(feature_subset_scaled, axis=0)
        
        # Distancias de todos los puntos al centroide
        distances_to_centroid = [
            np.linalg.norm(point - centroid) 
            for point in feature_subset_scaled
        ]
        
        # Seleccionar punto más lejano del centroide
        return np.argmax(distances_to_centroid)
    
    def diverse_sampling_within_cluster_improved(self, cluster_data, target_size, X_scaled=None, cluster_indices=None):
        """
        Muestreo diverso mejorado dentro de un cluster.
        
        Mejoras implementadas:
        1. Selección inicial científica (vs aleatoria)
        2. Uso de datos ya normalizados (vs doble normalización)  
        3. Validación Hopkins opcional durante selección
        
        Args:
            cluster_data: Datos del cluster a muestrear
            target_size: Número objetivo de canciones
            X_scaled: Datos ya normalizados (opcional, evita re-normalización)
            cluster_indices: Índices correspondientes en X_scaled
            
        Returns:
            DataFrame: Canciones seleccionadas del cluster
        """
        if len(cluster_data) <= target_size:
            return cluster_data
        
        # Usar top características para diversidad
        available_top_features = [f for f in self.top_features if f in cluster_data.columns]
        
        if not available_top_features:
            # Fallback a muestreo aleatorio si no hay top features
            print(f"⚠️  Sin características top disponibles - fallback aleatorio")
            return cluster_data.sample(n=target_size, random_state=42)
        
        # MEJORA 1: Usar datos ya normalizados si están disponibles
        if X_scaled is not None and cluster_indices is not None:
            # Extraer características correspondientes de datos ya normalizados
            try:
                # Encontrar índices de características top en X_scaled
                all_features = self.available_features if hasattr(self, 'available_features') else self.musical_features
                top_feature_indices = [all_features.index(f) for f in available_top_features if f in all_features]
                
                if top_feature_indices:
                    feature_subset_scaled = X_scaled[cluster_indices][:, top_feature_indices]
                    print(f"✅ Usando datos pre-normalizados ({len(available_top_features)} características)")
                else:
                    raise ValueError("No se encontraron índices de características")
                    
            except (ValueError, IndexError) as e:
                print(f"⚠️  Error usando datos pre-normalizados: {e}")
                # Fallback a normalización local
                feature_subset = cluster_data[available_top_features].values
                scaler = StandardScaler()
                feature_subset_scaled = scaler.fit_transform(feature_subset)
                print(f"🔄 Fallback a normalización local")
        else:
            # Normalización estándar (datos no pre-normalizados disponibles)
            feature_subset = cluster_data[available_top_features].values
            scaler = StandardScaler()
            feature_subset_scaled = scaler.fit_transform(feature_subset)
            print(f"📊 Normalización estándar ({len(available_top_features)} características)")
        
        # MEJORA 2: Selección inicial científica
        initial_idx = self.improved_initial_selection(feature_subset_scaled)
        selected_indices = [initial_idx]
        selected_features = [feature_subset_scaled[initial_idx]]
        
        print(f"🎯 Selección inicial científica: índice {initial_idx}")
        
        # MaxMin algorithm optimizado
        iteration = 0
        while len(selected_indices) < target_size:
            distances = []
            
            for i, candidate in enumerate(feature_subset_scaled):
                if i in selected_indices:
                    distances.append(-1)  # Ya seleccionado
                    continue
                
                # Distancia mínima a puntos ya seleccionados (vectorizado)
                candidate_distances = [
                    np.linalg.norm(candidate - selected)
                    for selected in selected_features
                ]
                min_distance = min(candidate_distances)
                distances.append(min_distance)
            
            # Seleccionar punto más lejano
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
            selected_features.append(feature_subset_scaled[next_idx])
            
            iteration += 1
            
            # Progress logging cada 100 iteraciones
            if iteration % 100 == 0:
                print(f"   MaxMin progress: {len(selected_indices)}/{target_size} seleccionadas")
        
        selected_cluster = cluster_data.iloc[selected_indices]
        
        print(f"✅ MaxMin sampling completado: {len(selected_indices)} canciones seleccionadas")
        
        return selected_cluster
    
    def apply_diversity_fallback(self, cluster_data, target_size):
        """
        Estrategia fallback para mayor diversidad cuando Hopkins es bajo.
        
        Utiliza muestreo estratificado por percentiles extremos en características
        principales para maximizar diversidad cuando el MaxMin estándar no es suficiente.
        
        Args:
            cluster_data: Datos del cluster
            target_size: Número objetivo de canciones
            
        Returns:
            DataFrame: Canciones seleccionadas con estrategia diversidad
        """
        if len(cluster_data) <= target_size:
            return cluster_data
        
        self.selection_metadata['diversity_fallbacks'] += 1
        print(f"🔄 Aplicando fallback de diversidad (activación #{self.selection_metadata['diversity_fallbacks']})")
        
        # Estrategia: Muestreo estratificado por características extremas
        extreme_indices = set()
        
        # Analizar top 3 características para extremos
        for feature in self.top_features[:3]:
            if feature in cluster_data.columns:
                feature_values = cluster_data[feature].dropna()
                
                if len(feature_values) > 10:  # Suficientes datos para percentiles
                    # Percentiles 10% y 90% para capturar extremos
                    p10 = feature_values.quantile(0.10)
                    p90 = feature_values.quantile(0.90)
                    
                    # Seleccionar canciones en extremos
                    extreme_mask = (feature_values <= p10) | (feature_values >= p90)
                    extreme_songs = cluster_data[cluster_data[feature].isin(feature_values[extreme_mask])]
                    extreme_indices.update(extreme_songs.index.tolist())
                    
                    print(f"   {feature}: {len(extreme_songs)} canciones en extremos (p10={p10:.3f}, p90={p90:.3f})")
        
        # Combinar extremos con muestra aleatoria
        if extreme_indices:
            extreme_subset = cluster_data.loc[list(extreme_indices)]
            n_extreme = min(len(extreme_subset), target_size // 2)  # Máximo 50% extremos
            
            if n_extreme > 0:
                selected_extreme = extreme_subset.sample(n=n_extreme, random_state=42)
                remaining_needed = target_size - n_extreme
                
                if remaining_needed > 0:
                    # Completar con muestra aleatoria del resto
                    remaining_data = cluster_data.drop(selected_extreme.index)
                    
                    if len(remaining_data) >= remaining_needed:
                        selected_random = remaining_data.sample(n=remaining_needed, random_state=42)
                        final_selection = pd.concat([selected_extreme, selected_random])
                    else:
                        # No hay suficientes datos restantes, usar todo
                        final_selection = pd.concat([selected_extreme, remaining_data])
                else:
                    final_selection = selected_extreme
            else:
                # Sin extremos válidos, fallback completo
                final_selection = cluster_data.sample(n=target_size, random_state=42)
        else:
            # Sin características extremas identificadas, muestreo aleatorio
            print(f"⚠️  Sin extremos identificados, muestreo aleatorio")
            final_selection = cluster_data.sample(n=target_size, random_state=42)
        
        print(f"✅ Fallback diversidad: {len(final_selection)} canciones (extremos: {len(extreme_indices) if extreme_indices else 0})")
        
        return final_selection
    
    def select_optimal_10k_with_validation(self, df):
        """
        Selección optimizada con validación Hopkins integrada.
        
        Implementa todas las mejoras críticas identificadas:
        1. MaxMin sampling científico con selección inicial optimizada
        2. Validación Hopkins continua durante selección  
        3. Eliminación de normalización doble mediante reutilización de datos
        4. Estrategias de fallback cuando Hopkins se degrada
        
        Returns:
            tuple: (selected_df, selection_metadata)
        """
        print("\n🚀 SELECCIÓN OPTIMIZADA CON VALIDACIÓN HOPKINS")
        print("=" * 60)
        print("Mejoras: Selección científica, validación continua, sin doble normalización")
        
        # Reset validador para nueva sesión
        self.hopkins_validator.reset_validation_history()
        
        # 1. Preparar datos para clustering
        print("\n🔧 1. PREPARANDO DATOS PARA CLUSTERING")
        print("-" * 40)
        
        X_scaled, clean_indices = self.prepare_clustering_data(df)
        if X_scaled is None:
            return None, None
            
        print(f"✅ Datos preparados: {X_scaled.shape[0]:,} canciones × {X_scaled.shape[1]} características")
        
        # 2. Identificar estructura natural
        print("\n🎯 2. IDENTIFICANDO ESTRUCTURA NATURAL")
        print("-" * 40)
        
        cluster_labels, silhouette = self.identify_natural_clusters(X_scaled, clean_indices)
        
        # 3. Preparar DataFrame con clusters
        df_clean = df.loc[clean_indices].copy()
        df_clean['natural_cluster'] = cluster_labels
        
        # Crear mapeo para datos normalizados
        index_mapping = {original_idx: scaled_idx for scaled_idx, original_idx in enumerate(clean_indices)}
        
        print(f"\n📊 3. SELECCIÓN POR CLUSTERS CON VALIDACIÓN")
        print("-" * 50)
        
        # 4. Selección por cluster con validación continua
        target_total = 10000
        selected_parts = []
        selection_log = {}
        
        for cluster_id in sorted(df_clean['natural_cluster'].unique()):
            cluster_data = df_clean[df_clean['natural_cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            # Calcular selección proporcional robusta
            proportion = cluster_size / len(df_clean)
            base_target = int(target_total * proportion)
            
            # MEJORA: Selección proporcional robusta (evitar clusters vacíos)
            min_per_cluster = max(10, target_total // 100)  # Mínimo 1% o 10 canciones
            max_per_cluster = target_total // 2             # Máximo 50%
            target_size = max(min_per_cluster, min(max_per_cluster, base_target))
            
            print(f"\n🎵 Cluster {cluster_id}:")
            print(f"   Tamaño original: {cluster_size:,} canciones ({proportion:.1%})")
            print(f"   Target base: {base_target:,}, ajustado: {target_size:,}")
            
            # Obtener índices correspondientes en X_scaled
            cluster_scaled_indices = [index_mapping[idx] for idx in cluster_data.index if idx in index_mapping]
            
            # 5. MEJORA: Muestreo diverso mejorado
            try:
                selected_cluster = self.diverse_sampling_within_cluster_improved(
                    cluster_data, target_size, X_scaled, cluster_scaled_indices
                )
                
                # 6. NUEVO: Validación Hopkins del cluster seleccionado
                if len(selected_cluster) >= 20:  # Suficientes datos para Hopkins
                    cluster_features = selected_cluster[self.available_features].dropna()
                    
                    if len(cluster_features) >= 10:
                        validation_result = self.hopkins_validator.validate_during_selection(
                            cluster_features, iteration=f"cluster_{cluster_id}", 
                            context=f"Cluster {cluster_id} con {len(selected_cluster)} canciones"
                        )
                        
                        print(f"   📊 Hopkins validation: {validation_result['hopkins']:.3f}")
                        print(f"   💡 {validation_result['recommendation']}")
                        
                        # Guardar validación en metadata
                        self.selection_metadata['hopkins_validations'].append({
                            'cluster_id': cluster_id,
                            'hopkins': validation_result['hopkins'],
                            'action': validation_result['action'],
                            'size': len(selected_cluster)
                        })
                        
                        # Si Hopkins muy bajo, aplicar estrategia fallback
                        if validation_result['action'] == 'fallback':
                            print(f"   🔄 Hopkins bajo - aplicando fallback diversidad")
                            self.selection_metadata['fallback_activations'] += 1
                            
                            selected_cluster = self.apply_diversity_fallback(
                                cluster_data, target_size
                            )
                            
                            # Re-validar después del fallback
                            fallback_features = selected_cluster[self.available_features].dropna()
                            if len(fallback_features) >= 10:
                                fallback_validation = self.hopkins_validator.validate_during_selection(
                                    fallback_features, iteration=f"cluster_{cluster_id}_fallback"
                                )
                                print(f"   🔄 Hopkins post-fallback: {fallback_validation['hopkins']:.3f}")
                
                selection_log[cluster_id] = {
                    'original_size': cluster_size,
                    'selected_size': len(selected_cluster),
                    'selection_ratio': len(selected_cluster) / cluster_size,
                    'target_size': target_size,
                    'method': 'improved_maxmin_with_validation'
                }
                
                selected_parts.append(selected_cluster)
                print(f"   ✅ Seleccionadas: {len(selected_cluster):,} canciones")
                
            except Exception as e:
                print(f"   ❌ Error en cluster {cluster_id}: {e}")
                # Fallback a muestreo aleatorio en caso de error
                selected_cluster = cluster_data.sample(n=min(target_size, len(cluster_data)), random_state=42)
                selected_parts.append(selected_cluster)
                selection_log[cluster_id] = {
                    'original_size': cluster_size,
                    'selected_size': len(selected_cluster),
                    'error': str(e),
                    'method': 'random_fallback'
                }
        
        # 7. Combinar y ajustar tamaño final
        print(f"\n🔧 4. COMBINANDO Y AJUSTANDO SELECCIÓN FINAL")
        print("-" * 40)
        
        final_selection = pd.concat(selected_parts, ignore_index=True)
        initial_size = len(final_selection)
        
        print(f"Tamaño inicial combinado: {initial_size:,}")
        
        if len(final_selection) != target_total:
            if len(final_selection) > target_total:
                print(f"🔧 Reduciendo de {len(final_selection):,} a {target_total:,}")
                final_selection = final_selection.sample(n=target_total, random_state=42)
            else:
                print(f"🔧 Completando de {len(final_selection):,} a {target_total:,}")
                remaining = target_total - len(final_selection)
                available = df_clean[~df_clean.index.isin(final_selection.index)]
                
                if len(available) >= remaining:
                    additional = available.sample(n=remaining, random_state=42)
                    final_selection = pd.concat([final_selection, additional], ignore_index=True)
                else:
                    print(f"⚠️  Solo {len(available)} canciones disponibles para completar")
                    final_selection = pd.concat([final_selection, available], ignore_index=True)
        
        # 8. NUEVO: Validación Hopkins final del dataset completo
        print(f"\n🔍 5. VALIDACIÓN HOPKINS FINAL")
        print("-" * 40)
        
        final_features = final_selection[self.available_features].dropna()
        if len(final_features) >= 50:
            final_validation = self.hopkins_validator.validate_during_selection(
                final_features, iteration='FINAL_DATASET', 
                context=f'Dataset final con {len(final_selection)} canciones'
            )
            
            print(f"📊 Hopkins final: {final_validation['hopkins']:.4f}")
            print(f"💡 {final_validation['recommendation']}")
        else:
            final_validation = {'hopkins': None, 'recommendation': 'Datos insuficientes para validación final'}
            print(f"⚠️  Datos insuficientes para validación Hopkins final")
        
        # 9. Generar metadata comprehensiva
        validation_summary = self.hopkins_validator.get_validation_summary()
        
        selection_metadata = {
            'selection_log': selection_log,
            'hopkins_validation': validation_summary,
            'final_hopkins': final_validation.get('hopkins'),
            'pre_clustering_silhouette': silhouette,
            'fallback_activations': self.selection_metadata['fallback_activations'],
            'diversity_fallbacks': self.selection_metadata['diversity_fallbacks'],
            'method_improvements': [
                'scientific_initial_selection',
                'continuous_hopkins_validation', 
                'elimination_double_normalization',
                'robust_proportional_selection',
                'diversity_fallback_strategy'
            ]
        }
        
        # 10. Resumen final
        print(f"\n✅ SELECCIÓN CON VALIDACIÓN COMPLETADA")
        print("=" * 60)
        print(f"📦 Dataset final: {len(final_selection):,} canciones")
        print(f"📊 Hopkins final: {final_validation.get('hopkins', 'N/A')}")
        print(f"📈 Hopkins validaciones: {validation_summary['total_validations']}")
        print(f"📊 Hopkins promedio: {validation_summary.get('hopkins_statistics', {}).get('avg_hopkins', 'N/A')}")
        print(f"🔄 Fallbacks activados: {self.selection_metadata['fallback_activations']}")
        print(f"🎯 Calidad estimada: {validation_summary.get('quality_assessment', {}).get('overall_quality', 'unknown')}")
        
        return final_selection, selection_metadata
    
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
    """Función principal del selector optimizado con validación Hopkins."""
    print("🎯 SELECTOR OPTIMIZADO DE 10K CANCIONES DESDE 18K")
    print("=" * 60)
    print("Estrategia: Clustering-aware con validación Hopkins continua")
    print("Mejoras: Selección científica, sin doble normalización, fallback inteligente")
    print("Basado en: Análisis clustering readiness (Hopkins=0.823)")
    print()
    
    # Inicializar selector con threshold Hopkins configurable
    selector = OptimalSelector(hopkins_threshold=0.70)
    
    # 1. Cargar dataset fuente
    print("📂 1. CARGANDO DATASET FUENTE")
    print("-" * 30)
    
    df_18k = selector.load_source_dataset()
    if df_18k is None:
        print("❌ ERROR: No se pudo cargar el dataset fuente")
        return
    
    # 2. Selección optimizada con validación
    print("\n🚀 2. EJECUTANDO SELECCIÓN OPTIMIZADA")
    print("-" * 40)
    
    try:
        selected_10k, selection_metadata = selector.select_optimal_10k_with_validation(df_18k)
        
        if selected_10k is None:
            print("❌ ERROR: Falló la selección optimizada")
            return
            
        print(f"\n✅ Selección exitosa: {len(selected_10k):,} canciones")
        
    except Exception as e:
        print(f"❌ ERROR en selección optimizada: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Validación tradicional adicional
    print("\n📊 3. VALIDACIÓN TRADICIONAL ADICIONAL")
    print("-" * 40)
    
    try:
        selector.validate_selection(selected_10k, df_18k)
    except Exception as e:
        print(f"⚠️  Warning en validación tradicional: {e}")
    
    # 4. Guardar resultado con metadatos mejorados
    print("\n💾 4. GUARDANDO DATASET OPTIMIZADO")
    print("-" * 40)
    
    try:
        # Incluir metadatos de selección en el dataset guardado
        output_file = selector.save_dataset_with_metadata(selected_10k, selection_metadata)
        print(f"✅ Dataset guardado: {output_file}")
        
    except AttributeError:
        # Fallback a método original si save_dataset_with_metadata no existe
        output_file = selector.save_dataset(selected_10k)
        print(f"✅ Dataset guardado (método estándar): {output_file}")
        
        # Guardar metadatos por separado
        try:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_file = f"data/final_data/selection_metadata_{timestamp}.json"
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(selection_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Metadatos guardados: {metadata_file}")
            
        except Exception as e:
            print(f"⚠️  No se pudieron guardar metadatos: {e}")
    
    # 5. Resumen final mejorado
    print(f"\n🎉 SELECCIÓN OPTIMIZADA COMPLETADA")
    print("=" * 60)
    
    # Información básica
    print(f"📊 Dataset final: {len(selected_10k):,} canciones")
    print(f"📁 Archivo principal: {output_file}")
    print(f"🔄 Formato: Separador '^', decimal '.', UTF-8")
    
    # Métricas Hopkins
    final_hopkins = selection_metadata.get('final_hopkins', 'N/A')
    hopkins_validations = selection_metadata.get('hopkins_validation', {}).get('total_validations', 0)
    avg_hopkins = selection_metadata.get('hopkins_validation', {}).get('hopkins_statistics', {}).get('avg_hopkins', 'N/A')
    quality = selection_metadata.get('hopkins_validation', {}).get('quality_assessment', {}).get('overall_quality', 'unknown')
    
    print(f"\n📈 MÉTRICAS DE CALIDAD:")
    print(f"   • Hopkins final: {final_hopkins}")
    print(f"   • Hopkins promedio: {avg_hopkins}")
    print(f"   • Validaciones realizadas: {hopkins_validations}")
    print(f"   • Calidad estimada: {quality}")
    print(f"   • Fallbacks activados: {selection_metadata.get('fallback_activations', 0)}")
    
    # Interpretación de calidad
    if isinstance(final_hopkins, (int, float)):
        if final_hopkins >= 0.80:
            status_emoji = "🟢"
            status_text = "EXCELENTE"
        elif final_hopkins >= 0.70:
            status_emoji = "🟡"  
            status_text = "BUENO"
        elif final_hopkins >= 0.60:
            status_emoji = "🟠"
            status_text = "ACEPTABLE"
        else:
            status_emoji = "🔴"
            status_text = "NECESITA MEJORAS"
    else:
        status_emoji = "⚪"
        status_text = "NO EVALUADO"
    
    print(f"\n{status_emoji} ESTADO FINAL: {status_text}")
    
    # Recomendaciones
    print(f"\n🎯 MÉTRICAS ESPERADAS PARA CLUSTERING:")
    expected_clustering_readiness = "75-85/100" if isinstance(final_hopkins, (int, float)) and final_hopkins >= 0.70 else "60-75/100"
    expected_silhouette = "0.15-0.20" if isinstance(final_hopkins, (int, float)) and final_hopkins >= 0.70 else "0.10-0.15"
    
    print(f"   • Clustering Readiness esperado: {expected_clustering_readiness}")
    print(f"   • Silhouette Score esperado: {expected_silhouette}")
    
    print(f"\n📋 PRÓXIMOS PASOS:")
    if status_text in ["EXCELENTE", "BUENO"]:
        print("   ✅ Proceder a FASE 2: Clustering Comparativo")
        print("   📁 Ejecutar: clustering/algorithms/musical/clustering_optimized.py")
    elif status_text == "ACEPTABLE":
        print("   ⚠️  Considerar mejoras opcionales antes de FASE 2")
        print("   🔧 O proceder con monitoreo adicional en clustering")
    else:
        print("   ❌ Revisar selección antes de continuar")
        print("   🔧 Considerar ajustar hopkins_threshold o datos fuente")
    
    print(f"\n📋 ARCHIVOS GENERADOS:")
    print(f"   • Dataset: {output_file}")
    if 'metadata_file' in locals():
        print(f"   • Metadatos: {metadata_file}")
    
    print(f"\n🔄 Para cargar el dataset:")
    print(f"   import pandas as pd")
    print(f"   df = pd.read_csv('{output_file}', sep='^', decimal='.', encoding='utf-8')")


def save_dataset_with_metadata(self, selected_df, selection_metadata):
    """Guardar dataset con metadatos integrados (método mejorado)."""
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
    
    # Guardar metadatos integrados
    import json
    
    complete_metadata = {
        'generation_info': {
            'timestamp': timestamp,
            'generation_date': datetime.now().isoformat(),
            'source_dataset': 'data/with_lyrics/spotify_songs_fixed.csv',
            'selection_method': 'clustering_aware_with_hopkins_validation',
            'improvements_applied': selection_metadata.get('method_improvements', [])
        },
        'dataset_info': {
            'selected_size': len(selected_df),
            'features_count': len(self.available_features),
            'format': {'separator': '^', 'decimal': '.', 'encoding': 'utf-8'}
        },
        'selection_metadata': selection_metadata,
        'quality_summary': {
            'final_hopkins': selection_metadata.get('final_hopkins'),
            'avg_hopkins': selection_metadata.get('hopkins_validation', {}).get('hopkins_statistics', {}).get('avg_hopkins'),
            'quality_level': selection_metadata.get('hopkins_validation', {}).get('quality_assessment', {}).get('overall_quality'),
            'fallback_activations': selection_metadata.get('fallback_activations', 0)
        }
    }
    
    metadata_file = f"{output_dir}/picked_data_optimal_metadata_{timestamp}.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(complete_metadata, f, indent=2, ensure_ascii=False)
    
    # También versión estándar de metadatos
    standard_metadata = f"{output_dir}/picked_data_optimal_metadata.json"
    with open(standard_metadata, 'w', encoding='utf-8') as f:
        json.dump(complete_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 ARCHIVOS GENERADOS:")
    print(f"📁 CSV principal: {standard_file}")
    print(f"📁 CSV con timestamp: {output_file}")
    print(f"📋 Metadatos: {metadata_file}")
    
    return standard_file

# Monkey patch para añadir método mejorado a la clase
OptimalSelector.save_dataset_with_metadata = save_dataset_with_metadata

if __name__ == "__main__":
    main()