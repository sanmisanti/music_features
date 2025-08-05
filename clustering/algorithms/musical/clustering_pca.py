#!/usr/bin/env python3
"""
Clustering con PCA para mejorar Silhouette Score
===============================================

Aplica PCA antes del clustering para reducir dimensionalidad y ruido,
potencialmente mejorando la separación de clusters.
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import json
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description='Clustering con PCA')
    parser.add_argument('--dataset', type=str, default='../../../data/final_data/picked_data_lyrics.csv')
    parser.add_argument('--k-range', type=int, nargs=2, default=[3, 8])
    parser.add_argument('--pca-components', type=int, default=8, help='Número de componentes PCA')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    logging.basicConfig(level=getattr(logging, args.log_level), 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Características musicales
    MUSICAL_FEATURES = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
    ]
    
    logger.info(f"🎵 CLUSTERING CON PCA - {args.pca_components} COMPONENTES")
    logger.info("=" * 60)
    
    # Cargar datos con nuevo formato
    df = pd.read_csv(args.dataset, sep='^', decimal='.', encoding='utf-8', on_bad_lines='skip')
    logger.info(f"✅ Dataset cargado: {len(df):,} canciones")
    
    # Preparar características
    features_matrix = df[MUSICAL_FEATURES].values
    
    # Normalizar
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(features_matrix)
    logger.info(f"✅ Datos normalizados: {normalized_data.shape}")
    
    # Aplicar PCA
    pca = PCA(n_components=args.pca_components, random_state=42)
    pca_data = pca.fit_transform(normalized_data)
    
    # Mostrar varianza explicada
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()
    
    logger.info(f"🔍 PCA aplicado:")
    logger.info(f"   Componentes: {args.pca_components}")
    logger.info(f"   Varianza explicada: {total_variance:.1%}")
    logger.info(f"   Por componente: {[f'{v:.1%}' for v in explained_variance[:5]]}")
    
    # Clustering en espacio PCA
    best_score = -1
    best_result = None
    
    logger.info(f"🔍 Probando K={args.k_range[0]} a {args.k_range[1]} en espacio PCA...")
    
    for k in range(args.k_range[0], args.k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pca_data)
        
        # Calcular Silhouette en espacio PCA
        if len(pca_data) > 2000:
            # Usar muestra para Silhouette
            sample_idx = np.random.choice(len(pca_data), 2000, replace=False)
            silhouette = silhouette_score(pca_data[sample_idx], labels[sample_idx])
        else:
            silhouette = silhouette_score(pca_data, labels)
        
        logger.info(f"   K={k}: Silhouette={silhouette:.3f}")
        
        if silhouette > best_score:
            best_score = silhouette
            best_result = {
                'k': k, 
                'silhouette': silhouette,
                'labels': labels,
                'kmeans': kmeans
            }
    
    logger.info(f"\n🏆 MEJOR RESULTADO CON PCA:")
    logger.info(f"   K óptimo: {best_result['k']}")
    logger.info(f"   Silhouette Score: {best_result['silhouette']:.3f}")
    
    # Distribución de clusters
    unique, counts = np.unique(best_result['labels'], return_counts=True)
    distribution = dict(zip(unique, counts))
    logger.info(f"   Distribución: {list(distribution.values())}")
    
    # Comparar con clustering original (sin PCA)
    logger.info(f"\n📊 COMPARACIÓN SIN PCA:")
    kmeans_original = KMeans(n_clusters=best_result['k'], random_state=42, n_init=10)
    labels_original = kmeans_original.fit_predict(normalized_data)
    
    if len(normalized_data) > 2000:
        sample_idx = np.random.choice(len(normalized_data), 2000, replace=False)  
        silhouette_original = silhouette_score(normalized_data[sample_idx], labels_original[sample_idx])
    else:
        silhouette_original = silhouette_score(normalized_data, labels_original)
    
    logger.info(f"   Sin PCA: {silhouette_original:.3f}")
    logger.info(f"   Con PCA: {best_result['silhouette']:.3f}")
    improvement = ((best_result['silhouette'] - silhouette_original) / silhouette_original) * 100
    logger.info(f"   Mejora: {improvement:+.1f}%")
    
    # Guardar modelos si se solicita
    if args.save_models:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = Path('results/models_pca')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar pipeline completo
        joblib.dump(scaler, models_dir / f'scaler_pca_{timestamp}.pkl')
        joblib.dump(pca, models_dir / f'pca_{args.pca_components}comp_{timestamp}.pkl')
        joblib.dump(best_result['kmeans'], models_dir / f'kmeans_pca_k{best_result["k"]}_{timestamp}.pkl')
        
        logger.info(f"💾 Modelos PCA guardados en: {models_dir}")
        
        # Guardar dataset con clusters PCA (renombrar cluster_pca a cluster)
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = best_result['labels']
        
        results_dir = Path('results/results_pca')
        results_dir.mkdir(parents=True, exist_ok=True)
        data_path = results_dir / f'clustering_pca_results_{timestamp}.csv'
        df_with_clusters.to_csv(data_path, index=False)
        
        logger.info(f"💾 Dataset con clusters PCA: {data_path}")
    
    # Interpretación final
    if best_result['silhouette'] > 0.5:
        quality = "Bueno - Clusters claramente separados"
    elif best_result['silhouette'] > 0.25:
        quality = "Aceptable - Clusters moderadamente definidos"
    else:
        quality = "Mejorable - Clusters poco definidos"
    
    logger.info(f"\n🎯 Calidad final: {quality}")
    logger.info("🎵 ¡CLUSTERING CON PCA COMPLETADO! 🎉")

if __name__ == "__main__":
    main()