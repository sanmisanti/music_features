#!/usr/bin/env python3
"""
Clustering Musical Optimizado para Datasets Grandes
===================================================

Sistema de clustering optimizado para procesar miles de canciones usando 
características musicales de Spotify. Incluye CLI, logging, persistencia 
de modelos y métricas avanzadas.

Uso:
    python clustering_optimized.py --dataset ../data/picked_data_0.csv
    python clustering_optimized.py --dataset ../data/picked_data_0.csv --k-range 3 15 --save-models
    python clustering_optimized.py --predict --song-features song.json --load-model models/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import warnings
import argparse
import logging
import json
import pickle
import time
from datetime import datetime
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from tqdm import tqdm
import joblib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Configuración global
MUSICAL_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
]

DEFAULT_CONFIG = {
    'k_range': (3, 12),
    'random_state': 42,
    'n_init': 10,
    'max_iter': 300,
    'algorithm': 'kmeans',  # 'kmeans' or 'minibatch'
    'batch_size': 1000,  # para MiniBatchKMeans
    'silhouette_sample_size': 2000,  # muestra para silhouette en datasets grandes
}

def setup_logging(log_level='INFO', log_file=None):
    """Configurar sistema de logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parsear argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Sistema de Clustering Musical Optimizado',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --dataset ../data/picked_data_0.csv
  %(prog)s --dataset ../data/picked_data_0.csv --k-range 3 15 --save-models
  %(prog)s --dataset ../data/picked_data_0.csv --algorithm minibatch --batch-size 2000
  %(prog)s --predict --song-features song.json --load-model models/
        """
    )
    
    # Argumentos principales
    parser.add_argument('--dataset', type=str, 
                       default='../../../data/final_data/picked_data_lyrics.csv',
                       help='Ruta al dataset CSV (default: ../../../data/final_data/picked_data_lyrics.csv)')
    
    parser.add_argument('--k-range', type=int, nargs=2, 
                       default=DEFAULT_CONFIG['k_range'],
                       metavar=('MIN', 'MAX'),
                       help='Rango de clusters K a probar (default: 3 12)')
    
    parser.add_argument('--algorithm', choices=['kmeans', 'minibatch'],
                       default=DEFAULT_CONFIG['algorithm'],
                       help='Algoritmo de clustering (default: kmeans)')
    
    parser.add_argument('--batch-size', type=int,
                       default=DEFAULT_CONFIG['batch_size'],
                       help='Tamaño de batch para MiniBatchKMeans (default: 1000)')
    
    # Configuración de modelo
    parser.add_argument('--random-state', type=int,
                       default=DEFAULT_CONFIG['random_state'],
                       help='Semilla aleatoria (default: 42)')
    
    parser.add_argument('--n-init', type=int,
                       default=DEFAULT_CONFIG['n_init'],
                       help='Número de inicializaciones K-Means (default: 10)')
    
    parser.add_argument('--max-iter', type=int,
                       default=DEFAULT_CONFIG['max_iter'],
                       help='Máximo de iteraciones K-Means (default: 300)')
    
    # Opciones de salida
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directorio para guardar resultados (default: results)')
    
    parser.add_argument('--save-models', action='store_true',
                       help='Guardar modelos entrenados (scaler, kmeans)')
    
    parser.add_argument('--save-plots', action='store_true',
                       help='Guardar visualizaciones como archivos')
    
    # Modo predicción
    parser.add_argument('--predict', action='store_true',
                       help='Modo predicción: predecir cluster de nueva canción')
    
    parser.add_argument('--song-features', type=str,
                       help='Archivo JSON con características de la canción')
    
    parser.add_argument('--load-model', type=str,
                       help='Directorio con modelos guardados')
    
    parser.add_argument('--top-n', type=int, default=5,
                       help='Número de canciones similares a retornar (default: 5)')
    
    # Configuración de logging
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Nivel de logging (default: INFO)')
    
    parser.add_argument('--log-file', type=str,
                       help='Archivo para guardar logs')
    
    # Opciones de visualización
    parser.add_argument('--no-plots', action='store_true',
                       help='No mostrar gráficos (útil para ejecución en batch)')
    
    return parser.parse_args()

def load_dataset(dataset_path, logger=None):
    """
    Cargar dataset completo de manera optimizada
    """
    if logger:
        logger.info(f"🔍 Cargando dataset: {dataset_path}")
    else:
        print(f"🔍 Cargando dataset: {dataset_path}")
    
    # Verificar si el archivo existe
    if not Path(dataset_path).exists():
        error_msg = f"❌ Archivo no encontrado: {dataset_path}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None
    
    start_time = time.time()
    
    try:
        # Cargar con configuración optimizada para nuevo dataset
        df = pd.read_csv(
            dataset_path, 
            sep='^',  # CAMBIO: era ';' - nuevo formato con letras
            decimal='.', # CAMBIO: era ',' - formato internacional
            encoding='utf-8', 
            on_bad_lines='skip',
            low_memory=False  # Evitar warnings de tipos mixtos
        )
        
        load_time = time.time() - start_time
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        info_msg = f"✅ Dataset cargado: {len(df):,} canciones en {load_time:.2f}s ({memory_usage:.1f}MB)"
        if logger:
            logger.info(info_msg)
        else:
            print(info_msg)
        
        return df
        
    except Exception as e:
        error_msg = f"❌ Error al cargar dataset: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None

def validate_dataset(df, features, logger=None):
    """
    Validar que el dataset tenga las características necesarias
    """
    if logger:
        logger.info("🔍 Validando estructura del dataset...")
    else:
        print("🔍 Validando estructura del dataset...")
    
    # Verificar columnas disponibles
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    info_msg = f"🎼 Características musicales disponibles: {len(available_features)}/{len(features)}"
    if logger:
        logger.info(info_msg)
        logger.debug(f"Disponibles: {available_features}")
        if missing_features:
            logger.warning(f"Faltantes: {missing_features}")
    else:
        print(info_msg)
        print(f"   📊 Disponibles: {available_features}")
        if missing_features:
            print(f"   ⚠️  Faltantes: {missing_features}")
    
    if len(available_features) < len(features) * 0.8:  # Al menos 80% de features
        error_msg = "❌ Dataset no tiene suficientes características musicales"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None, None
    
    return available_features, missing_features

def clean_musical_features(df, features, logger=None):
    """
    Limpiar y preparar características musicales para clustering (versión optimizada)
    """
    if logger:
        logger.info("🧹 Limpiando características musicales...")
    else:
        print("🧹 Limpiando características musicales...")
    
    cleaned_df = df.copy()
    cleaning_stats = {
        'original_rows': len(df),
        'features_processed': 0,
        'null_values_fixed': 0,
        'outliers_handled': 0
    }
    
    # Rangos válidos para características de Spotify (optimizados)
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
        'time_signature': (1, 7)
    }
    
    # Procesamiento vectorizado por lotes
    for feature in tqdm(features, desc="Limpiando features", disable=logger is not None):
        if feature in cleaned_df.columns:
            # Convertir a numérico de manera eficiente
            cleaned_df[feature] = pd.to_numeric(cleaned_df[feature], errors='coerce')
            
            # Contar y corregir valores nulos
            null_mask = cleaned_df[feature].isnull()
            null_count = null_mask.sum()
            
            if null_count > 0:
                median_val = cleaned_df[feature].median()
                cleaned_df.loc[null_mask, feature] = median_val
                cleaning_stats['null_values_fixed'] += null_count
                
                if logger:
                    logger.debug(f"  {feature}: {null_count} valores nulos → mediana ({median_val:.3f})")
            
            # Manejar outliers con clipping eficiente
            if feature in feature_ranges:
                min_val, max_val = feature_ranges[feature]
                outlier_mask = (cleaned_df[feature] < min_val) | (cleaned_df[feature] > max_val)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    cleaned_df[feature] = cleaned_df[feature].clip(min_val, max_val)
                    cleaning_stats['outliers_handled'] += outlier_count
                    
                    if logger:
                        logger.debug(f"  {feature}: {outlier_count} outliers → [{min_val}, {max_val}]")
            
            cleaning_stats['features_processed'] += 1
    
    # Resumen de limpieza
    summary_msg = f"📊 Limpieza completada: {cleaning_stats['features_processed']} features, {cleaning_stats['null_values_fixed']} nulos, {cleaning_stats['outliers_handled']} outliers"
    if logger:
        logger.info(summary_msg)
    else:
        print(summary_msg)
    
    return cleaned_df, cleaning_stats

def prepare_clustering_data(df, features, logger=None):
    """
    Preparar datos para clustering K-Means (versión optimizada)
    """
    if logger:
        logger.info("🎯 Preparando datos para clustering...")
    else:
        print("🎯 Preparando datos para clustering...")
    
    # Seleccionar características válidas
    clustering_features = [f for f in features if f in df.columns]
    
    if len(clustering_features) == 0:
        error_msg = "❌ No hay características válidas para clustering"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None, None, None
    
    if logger:
        logger.info(f"🔧 Usando {len(clustering_features)} características: {clustering_features}")
    else:
        print(f"🔧 Usando {len(clustering_features)} características")
        for i, feature in enumerate(clustering_features, 1):
            print(f"  {i:2d}. {feature}")
    
    # Extraer matriz de características
    feature_matrix = df[clustering_features].values
    
    # Normalización con StandardScaler
    scaler = StandardScaler()
    normalized_matrix = scaler.fit_transform(feature_matrix)
    
    success_msg = f"✅ Matriz preparada: {normalized_matrix.shape[0]:,} canciones × {normalized_matrix.shape[1]} características"
    if logger:
        logger.info(success_msg)
    else:
        print(success_msg)
    
    return normalized_matrix, scaler, clustering_features

def find_optimal_k_advanced(data, k_range=(3, 12), algorithm='kmeans', 
                           batch_size=1000, random_state=42, n_init=10, 
                           max_iter=300, silhouette_sample_size=2000, logger=None):
    """
    Encontrar K óptimo usando múltiples métricas y algoritmos optimizados
    """
    if logger:
        logger.info(f"🔍 Buscando K óptimo (rango: {k_range[0]}-{k_range[1]}, algoritmo: {algorithm})...")
    else:
        print(f"🔍 Buscando K óptimo (rango: {k_range[0]}-{k_range[1]}, algoritmo: {algorithm})...")
    
    results = []
    k_range_list = list(range(k_range[0], k_range[1] + 1))
    
    # Determinar si usar muestra para silhouette score
    use_silhouette_sample = len(data) > silhouette_sample_size
    if use_silhouette_sample:
        silhouette_indices = np.random.choice(len(data), silhouette_sample_size, replace=False)
        silhouette_data = data[silhouette_indices]
        if logger:
            logger.info(f"📊 Usando muestra de {silhouette_sample_size} para silhouette score")
        else:
            print(f"📊 Usando muestra de {silhouette_sample_size} para silhouette score")
    else:
        silhouette_data = data
        silhouette_indices = None
    
    for k in tqdm(k_range_list, desc="Probando valores de K"):
        start_time = time.time()
        
        # Seleccionar algoritmo
        if algorithm == 'minibatch':
            kmeans = MiniBatchKMeans(
                n_clusters=k, 
                random_state=random_state,
                batch_size=min(batch_size, len(data)),
                n_init=n_init,
                max_iter=max_iter
            )
        else:
            kmeans = KMeans(
                n_clusters=k, 
                random_state=random_state, 
                n_init=n_init,
                max_iter=max_iter
            )
        
        # Entrenar modelo
        labels = kmeans.fit_predict(data)
        
        # Calcular métricas
        inertia = kmeans.inertia_
        
        # Silhouette score (con muestra si es necesario)
        if use_silhouette_sample:
            sample_labels = labels[silhouette_indices]
            silhouette_avg = silhouette_score(silhouette_data, sample_labels)
        else:
            silhouette_avg = silhouette_score(data, labels)
        
        # Métricas adicionales
        calinski_score = calinski_harabasz_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        
        # Distribución de clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_distribution = dict(zip(unique_labels, counts))
        
        execution_time = time.time() - start_time
        
        result = {
            'k': k,
            'inertia': inertia,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_score,
            'davies_bouldin_score': davies_bouldin,
            'cluster_distribution': cluster_distribution,
            'labels': labels,
            'kmeans_model': kmeans,
            'execution_time': execution_time
        }
        
        results.append(result)
        
        if logger:
            logger.debug(f"K={k}: Silhouette={silhouette_avg:.3f}, Calinski={calinski_score:.1f}, DB={davies_bouldin:.3f} ({execution_time:.1f}s)")
        else:
            print(f"  🤖 K={k}: Silhouette={silhouette_avg:.3f}, Calinski={calinski_score:.1f}, DB={davies_bouldin:.3f} ({execution_time:.1f}s)")
    
    return results

def plot_advanced_metrics(results, save_path=None, show_plots=True):
    """
    Crear visualizaciones avanzadas de métricas de clustering
    """
    print("📊 Creando visualizaciones de métricas...")
    
    k_values = [r['k'] for r in results]
    inertias = [r['inertia'] for r in results]
    silhouette_scores = [r['silhouette_score'] for r in results]
    calinski_scores = [r['calinski_harabasz_score'] for r in results]
    davies_bouldin_scores = [r['davies_bouldin_score'] for r in results]
    
    # Crear subplot con 4 gráficos
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎵 Métricas de Clustering Musical Avanzadas', fontsize=16, fontweight='bold')
    
    # 1. Método del codo
    axes[0, 0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Número de Clusters (K)')
    axes[0, 0].set_ylabel('Inercia (WCSS)')
    axes[0, 0].set_title('📈 Método del Codo')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Silhouette Score
    axes[0, 1].plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Número de Clusters (K)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('🎯 Silhouette Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Marcar mejor K por Silhouette
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_k_idx]
    best_score = silhouette_scores[best_k_idx]
    axes[0, 1].scatter([best_k], [best_score], color='gold', s=200, zorder=5)
    axes[0, 1].annotate(f'Mejor K={best_k}\nScore={best_score:.3f}', 
                       xy=(best_k, best_score), xytext=(10, 10),
                       textcoords='offset points', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 3. Calinski-Harabasz Score (mayor es mejor)
    axes[1, 0].plot(k_values, calinski_scores, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Número de Clusters (K)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('📊 Calinski-Harabasz Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Davies-Bouldin Score (menor es mejor)
    axes[1, 1].plot(k_values, davies_bouldin_scores, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Número de Clusters (K)')
    axes[1, 1].set_ylabel('Davies-Bouldin Score')
    axes[1, 1].set_title('📉 Davies-Bouldin Score (menor es mejor)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return best_k

def select_optimal_k(results, logger=None):
    """
    Seleccionar K óptimo usando criterio combinado de múltiples métricas
    """
    if logger:
        logger.info("🎯 Seleccionando K óptimo usando criterio combinado...")
    else:
        print("🎯 Seleccionando K óptimo usando criterio combinado...")
    
    # Normalizar métricas (0-1)
    silhouette_scores = np.array([r['silhouette_score'] for r in results])
    calinski_scores = np.array([r['calinski_harabasz_score'] for r in results])
    davies_bouldin_scores = np.array([r['davies_bouldin_score'] for r in results])
    
    # Normalizar a 0-1
    silhouette_norm = (silhouette_scores - silhouette_scores.min()) / (silhouette_scores.max() - silhouette_scores.min())
    calinski_norm = (calinski_scores - calinski_scores.min()) / (calinski_scores.max() - calinski_scores.min())
    davies_bouldin_norm = 1 - (davies_bouldin_scores - davies_bouldin_scores.min()) / (davies_bouldin_scores.max() - davies_bouldin_scores.min())  # Invertir porque menor es mejor
    
    # Puntaje combinado (pesos: Silhouette 50%, Calinski 30%, Davies-Bouldin 20%)
    combined_scores = 0.5 * silhouette_norm + 0.3 * calinski_norm + 0.2 * davies_bouldin_norm
    
    best_idx = np.argmax(combined_scores)
    best_result = results[best_idx]
    
    if logger:
        logger.info(f"🏆 K óptimo seleccionado: {best_result['k']} (puntaje combinado: {combined_scores[best_idx]:.3f})")
    else:
        print(f"🏆 K óptimo seleccionado: {best_result['k']} (puntaje combinado: {combined_scores[best_idx]:.3f})")
    
    return best_result, combined_scores

def save_models(scaler, kmeans_model, output_dir, timestamp, logger=None):
    """
    Guardar modelos entrenados
    """
    models_dir = Path(output_dir) / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    scaler_path = models_dir / f'scaler_{timestamp}.pkl'
    kmeans_path = models_dir / f'kmeans_k{kmeans_model.n_clusters}_{timestamp}.pkl'
    
    # Guardar con joblib (más eficiente que pickle para sklearn)
    joblib.dump(scaler, scaler_path)
    joblib.dump(kmeans_model, kmeans_path)
    
    if logger:
        logger.info(f"💾 Modelos guardados:")
        logger.info(f"   Scaler: {scaler_path}")
        logger.info(f"   K-Means: {kmeans_path}")
    else:
        print(f"💾 Modelos guardados:")
        print(f"   Scaler: {scaler_path}")
        print(f"   K-Means: {kmeans_path}")
    
    return scaler_path, kmeans_path

def save_results(df_with_clusters, results, best_result, combined_scores, 
                output_dir, timestamp, logger=None):
    """
    Guardar resultados de clustering
    """
    results_dir = Path(output_dir) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar dataset con clusters
    data_path = results_dir / f'clustering_results_{timestamp}.csv'
    df_with_clusters.to_csv(data_path, index=False)
    
    # Convertir tipos numpy a tipos Python nativos para JSON
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # Guardar métricas detalladas
    metrics_data = {
        'timestamp': timestamp,
        'best_k': int(best_result['k']),
        'best_metrics': {
            'silhouette_score': float(best_result['silhouette_score']),
            'calinski_harabasz_score': float(best_result['calinski_harabasz_score']),
            'davies_bouldin_score': float(best_result['davies_bouldin_score']),
            'inertia': float(best_result['inertia'])
        },
        'cluster_distribution': {str(k): int(v) for k, v in best_result['cluster_distribution'].items()},
        'all_results': [
            {
                'k': int(r['k']),
                'silhouette_score': float(r['silhouette_score']),
                'calinski_harabasz_score': float(r['calinski_harabasz_score']),
                'davies_bouldin_score': float(r['davies_bouldin_score']),
                'inertia': float(r['inertia']),
                'combined_score': float(score)
            }
            for r, score in zip(results, combined_scores)
        ]
    }
    
    metrics_path = results_dir / f'clustering_metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    if logger:
        logger.info(f"💾 Resultados guardados:")
        logger.info(f"   Dataset: {data_path}")
        logger.info(f"   Métricas: {metrics_path}")
    else:
        print(f"💾 Resultados guardados:")
        print(f"   Dataset: {data_path}")
        print(f"   Métricas: {metrics_path}")
    
    return data_path, metrics_path

def main():
    """
    Función principal optimizada
    """
    # Parsear argumentos
    args = parse_arguments()
    
    # Configurar logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Timestamp para archivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("🎵 INICIANDO CLUSTERING MUSICAL OPTIMIZADO")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Rango K: {args.k_range[0]}-{args.k_range[1]}")
    logger.info(f"Algoritmo: {args.algorithm}")
    logger.info(f"Semilla aleatoria: {args.random_state}")
    
    # Configurar semillas
    np.random.seed(args.random_state)
    random.seed(args.random_state)
    
    # Crear directorios de salida
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Cargar dataset
        df = load_dataset(args.dataset, logger)
        if df is None:
            return 1
        
        # 2. Validar dataset
        available_features, missing_features = validate_dataset(df, MUSICAL_FEATURES, logger)
        if available_features is None:
            return 1
        
        # 3. Limpiar datos
        logger.info("📋 Iniciando limpieza de datos...")
        cleaned_df, cleaning_stats = clean_musical_features(df, available_features, logger)
        
        # 4. Preparar datos para clustering
        clustering_data, scaler, final_features = prepare_clustering_data(cleaned_df, available_features, logger)
        if clustering_data is None:
            return 1
        
        # 5. Encontrar K óptimo
        logger.info("🔍 Iniciando búsqueda de K óptimo...")
        clustering_results = find_optimal_k_advanced(
            clustering_data,
            k_range=tuple(args.k_range),
            algorithm=args.algorithm,
            batch_size=args.batch_size,
            random_state=args.random_state,
            n_init=args.n_init,
            max_iter=args.max_iter,
            logger=logger
        )
        
        # 6. Seleccionar mejor K
        best_result, combined_scores = select_optimal_k(clustering_results, logger)
        
        # 7. Crear visualizaciones
        if not args.no_plots:
            plot_path = Path(args.output_dir) / f'clustering_metrics_{timestamp}.png' if args.save_plots else None
            plot_advanced_metrics(clustering_results, plot_path, show_plots=not args.save_plots)
        
        # 8. Agregar clusters al dataset
        logger.info("📊 Generando dataset final con clusters...")
        final_df = cleaned_df.copy()
        final_df['cluster'] = best_result['labels']
        
        # 9. Guardar modelos si se solicita
        if args.save_models:
            save_models(scaler, best_result['kmeans_model'], args.output_dir, timestamp, logger)
        
        # 10. Guardar resultados
        save_results(final_df, clustering_results, best_result, combined_scores, 
                    args.output_dir, timestamp, logger)
        
        # 11. Resumen ejecutivo
        logger.info("\n🎉 CLUSTERING COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)
        logger.info(f"🎵 Canciones procesadas: {len(final_df):,}")
        logger.info(f"🎯 K óptimo: {best_result['k']}")
        logger.info(f"📊 Silhouette Score: {best_result['silhouette_score']:.3f}")
        logger.info(f"📈 Calinski-Harabasz: {best_result['calinski_harabasz_score']:.1f}")
        logger.info(f"📉 Davies-Bouldin: {best_result['davies_bouldin_score']:.3f}")
        logger.info(f"🎵 Distribución: {list(best_result['cluster_distribution'].values())}")
        
        # Interpretación de calidad
        score = best_result['silhouette_score']
        if score > 0.7:
            quality = "Excelente - Clusters muy bien definidos"
        elif score > 0.5:
            quality = "Bueno - Clusters claramente separados"
        elif score > 0.25:
            quality = "Aceptable - Clusters moderadamente definidos"
        else:
            quality = "Mejorable - Clusters poco definidos"
        
        logger.info(f"🎯 Calidad del clustering: {quality}")
        logger.info("🎵 ¡PROCESO COMPLETADO EXITOSAMENTE! 🎉")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error durante la ejecución: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)