import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
import pickle

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Configuración global
MUSICAL_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
]

def load_dataset_sample(dataset_path, sample_size=300, random_state=42):
    """
    Cargar muestra aleatoria del dataset de manera eficiente
    """
    try:
        print(f"🔍 Analizando dataset: {dataset_path}")
        
        # Verificar si el archivo existe
        if not Path(dataset_path).exists():
            print(f"❌ Archivo no encontrado: {dataset_path}")
            print("💡 Asegúrate de que la ruta sea correcta")
            return None
        
        # Para datasets grandes, usar muestreo eficiente
        print("📚 Cargando muestra aleatoria...")
        
        # Opción 1: Si el dataset es manejable (< 500MB)
        try:
            df = pd.read_csv(dataset_path, sep=';', decimal=',', encoding='utf-8', on_bad_lines='skip')
            total_rows = len(df)
            print(f"📊 Dataset cargado: {total_rows:,} canciones")
            
            if total_rows >= sample_size:
                sample_df = df.sample(n=sample_size, random_state=random_state)
                print(f"✅ Muestra creada: {len(sample_df)} canciones")
            else:
                sample_df = df
                print(f"⚠️  Dataset pequeño: usando todas las {len(sample_df)} canciones")
            
            return sample_df
            
        except MemoryError:
            print("⚠️  Dataset muy grande, usando muestreo por chunks...")
            
            # Opción 2: Muestreo eficiente para datasets grandes
            # Contar filas primero
            total_rows = sum(1 for line in open(dataset_path)) - 1
            print(f"📊 Total de filas: {total_rows:,}")
            
            # Generar índices aleatorios
            random_indices = sorted(np.random.choice(total_rows, size=sample_size, replace=False))
            
            # Leer en chunks y seleccionar filas
            chunk_size = 10000
            sample_data = []
            current_index = 0
            
            for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, sep=';', decimal=',', encoding='utf-8', on_bad_lines='skip'):
                chunk_start = current_index
                chunk_end = current_index + len(chunk)
                
                relevant_indices = [idx for idx in random_indices 
                                  if chunk_start <= idx < chunk_end]
                
                if relevant_indices:
                    relative_indices = [idx - chunk_start for idx in relevant_indices]
                    selected_rows = chunk.iloc[relative_indices]
                    sample_data.append(selected_rows)
                
                current_index = chunk_end
                
                if len(sample_data) > 0 and sum(len(df) for df in sample_data) >= sample_size:
                    break
            
            if sample_data:
                sample_df = pd.concat(sample_data, ignore_index=True)
                print(f"✅ Muestra por chunks creada: {len(sample_df)} canciones")
                return sample_df
            else:
                print("❌ No se pudo crear la muestra")
                return None
                
    except Exception as e:
        print(f"❌ Error al cargar dataset: {e}")
        return None

def clean_musical_features(df, features):
    """
    Limpiar y preparar características musicales para clustering
    """
    print("🧹 Limpiando características musicales...")
    
    cleaned_df = df.copy()
    cleaning_stats = {
        'original_rows': len(df),
        'features_processed': 0,
        'null_values_fixed': 0,
        'outliers_handled': 0
    }
    
    # Definir rangos válidos para características de Spotify
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
    
    for feature in features:
        if feature in cleaned_df.columns:
            print(f"  🔧 Procesando: {feature}")
            
            # Convertir a numérico
            cleaned_df[feature] = pd.to_numeric(cleaned_df[feature], errors='coerce')
            
            # Contar valores nulos
            null_count = cleaned_df[feature].isnull().sum()
            if null_count > 0:
                # Rellenar con mediana
                median_val = cleaned_df[feature].median()
                cleaned_df[feature].fillna(median_val, inplace=True)
                cleaning_stats['null_values_fixed'] += null_count
                print(f"    ✅ {null_count} valores nulos → mediana ({median_val:.3f})")
            
            # Manejar outliers usando rangos válidos
            if feature in feature_ranges:
                min_val, max_val = feature_ranges[feature]
                outliers = (cleaned_df[feature] < min_val) | (cleaned_df[feature] > max_val)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Clip valores a rangos válidos
                    cleaned_df[feature] = cleaned_df[feature].clip(min_val, max_val)
                    cleaning_stats['outliers_handled'] += outlier_count
                    print(f"    ✅ {outlier_count} outliers → rango [{min_val}, {max_val}]")
            
            cleaning_stats['features_processed'] += 1
    
    print(f"\n📊 Resumen de limpieza:")
    print(f"   🎵 Filas procesadas: {cleaning_stats['original_rows']}")
    print(f"   🔧 Features limpiadas: {cleaning_stats['features_processed']}")
    print(f"   🔄 Valores nulos corregidos: {cleaning_stats['null_values_fixed']}")
    print(f"   📏 Outliers manejados: {cleaning_stats['outliers_handled']}")
    
    return cleaned_df, cleaning_stats

def prepare_clustering_data(df, features):
    """
    Preparar datos para clustering K-Means
    """
    print("🎯 Preparando datos para clustering...")
    
    # Seleccionar solo características musicales válidas
    clustering_features = [f for f in features if f in df.columns]
    
    if len(clustering_features) == 0:
        print("❌ No hay características válidas para clustering")
        return None, None, None
    
    print(f"🔧 Usando {len(clustering_features)} características:")
    for i, feature in enumerate(clustering_features, 1):
        print(f"  {i:2d}. {feature}")
    
    # Extraer matriz de características
    feature_matrix = df[clustering_features].values
    
    # Normalización con StandardScaler
    scaler = StandardScaler()
    normalized_matrix = scaler.fit_transform(feature_matrix)
    
    print(f"✅ Matriz preparada: {normalized_matrix.shape[0]} canciones × {normalized_matrix.shape[1]} características")
    print(f"📊 Datos normalizados: media ≈ 0, std ≈ 1")
    
    return normalized_matrix, scaler, clustering_features

def find_optimal_k(data, k_range=(2, 11), random_state=42):
    """
    Encontrar el número óptimo de clusters usando múltiples métricas
    """
    print("🔍 Buscando número óptimo de clusters...")
    
    results = []
    
    for k in range(k_range[0], k_range[1]):
        print(f"  🤖 Probando K={k}...")
        
        # Aplicar K-Means
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Calcular métricas
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(data, labels)
        
        # Contar elementos por cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_distribution = dict(zip(unique_labels, counts))
        
        results.append({
            'k': k,
            'inertia': inertia,
            'silhouette_score': silhouette_avg,
            'cluster_distribution': cluster_distribution,
            'labels': labels,
            'kmeans_model': kmeans
        })
        
        print(f"    📊 Inercia: {inertia:.1f}, Silhouette: {silhouette_avg:.3f}")
    
    return results

def plot_clustering_metrics(results):
    """
    Visualizar métricas de clustering
    """
    k_values = [r['k'] for r in results]
    inertias = [r['inertia'] for r in results]
    silhouette_scores = [r['silhouette_score'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Método del codo (Elbow Method)
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Clusters (K)')
    ax1.set_ylabel('Inercia (WCSS)')
    ax1.set_title('📈 Método del Codo')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette Score
    ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Número de Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('🎯 Silhouette Score')
    ax2.grid(True, alpha=0.3)
    
    # Marcar el mejor K según Silhouette Score
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_k_idx]
    best_score = silhouette_scores[best_k_idx]
    
    ax2.scatter([best_k], [best_score], color='gold', s=200, zorder=5)
    ax2.annotate(f'Mejor K={best_k}\nScore={best_score:.3f}', 
                xy=(best_k, best_score), xytext=(10, 10),
                textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return best_k

def analyze_music_clusters(df, clustering_result, feature_names, scaler):
    """
    Analizar las características musicales de cada cluster
    """
    print("🎼 ANÁLISIS DETALLADO DE CLUSTERS MUSICALES:")
    print("=" * 60)
    
    labels = clustering_result['labels']
    k = clustering_result['k']
    
    # Añadir labels al dataframe
    analysis_df = df.copy()
    analysis_df['cluster'] = labels
    
    for cluster_id in range(k):
        cluster_songs = analysis_df[analysis_df['cluster'] == cluster_id]
        cluster_size = len(cluster_songs)
        
        print(f"\n🎵 CLUSTER {cluster_id} ({cluster_size} canciones):")
        print("-" * 40)
        
        # Calcular características promedio del cluster
        print("📊 Características musicales promedio:")
        for feature in feature_names:
            if feature in cluster_songs.columns:
                avg_value = cluster_songs[feature].mean()
                std_value = cluster_songs[feature].std()
                print(f"  {feature:16s}: {avg_value:6.3f} (±{std_value:.3f})")
        
        # Mostrar ejemplos de canciones
        print(f"\n🎤 Ejemplos de canciones:")
        display_cols = []
        for col in ['name', 'artists', 'album']:
            if col in cluster_songs.columns:
                display_cols.append(col)
        
        if display_cols:
            examples = cluster_songs[display_cols].head(5)
            for i, (_, song) in enumerate(examples.iterrows()):
                if 'name' in song and 'artists' in song:
                    print(f"  {i+1}. \"{song['name']}\" - {song['artists']}")
                elif 'name' in song:
                    print(f"  {i+1}. \"{song['name']}\"")
                else:
                    print(f"  {i+1}. {song[display_cols[0]]}")
        
        print()
    
    return analysis_df

def create_cluster_visualization(data, labels, feature_names):
    """
    Crear visualización de clusters usando PCA
    """
    print("📊 Creando visualización de clusters...")
    
    # Aplicar PCA para reducir a 2D
    pca = PCA(n_components=2, random_state=42)
    data_2d = pca.fit_transform(data)
    
    # Crear el gráfico
    plt.figure(figsize=(12, 8))
    
    # Colores para cada cluster
    colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(labels))))
    
    for i, color in enumerate(colors):
        cluster_mask = labels == i
        plt.scatter(data_2d[cluster_mask, 0], data_2d[cluster_mask, 1], 
                   c=[color], label=f'Cluster {i}', alpha=0.7, s=50)
    
    plt.xlabel(f'Primera Componente Principal ({pca.explained_variance_ratio_[0]:.1%} varianza)')
    plt.ylabel(f'Segunda Componente Principal ({pca.explained_variance_ratio_[1]:.1%} varianza)')
    plt.title('🎵 Visualización de Clusters Musicales (PCA 2D)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    total_variance = pca.explained_variance_ratio_.sum()
    print(f"✅ Varianza explicada total: {total_variance:.1%}")
    
    return pca, data_2d

def predict_cluster_and_similar_songs(new_song_features, 
                                     clustered_data_path='clustering_results.csv',
                                     scaler_model=None, 
                                     kmeans_model=None, 
                                     top_n=5):
    """
    Predecir cluster de nueva canción y encontrar similares
    
    Args:
        new_song_features: dict con características de la nueva canción
        clustered_data_path: path al CSV con resultados de clustering
        scaler_model: modelo StandardScaler entrenado
        kmeans_model: modelo KMeans entrenado
        top_n: número de canciones similares a retornar
    """
    
    # Cargar datos clusterizados
    df_clustered = pd.read_csv(clustered_data_path)
    
    # Características para clustering
    feature_cols = MUSICAL_FEATURES
    
    # Preparar características de la nueva canción
    new_features = np.array([[new_song_features[col] for col in feature_cols]])
    
    # Normalizar
    normalized_features = scaler_model.transform(new_features)
    
    # Predecir cluster
    predicted_cluster = kmeans_model.predict(normalized_features)[0]
    
    # Obtener canciones del mismo cluster
    cluster_songs = df_clustered[df_clustered['cluster'] == predicted_cluster].copy()
    
    # Calcular similitud con canciones del cluster
    cluster_features = cluster_songs[feature_cols].values
    cluster_features_normalized = scaler_model.transform(cluster_features)
    
    # Usar distancia Manhattan convertida a similitud
    distances = manhattan_distances(normalized_features, cluster_features_normalized)[0]
    similarities = 1 / (1 + distances)

    cluster_songs['similarity'] = similarities
    
    # Ordenar por similitud
    most_similar = cluster_songs.nlargest(top_n, 'similarity')
    
    return predicted_cluster, most_similar[['name', 'artists', 'similarity']]

def main():
    """
    Función principal para ejecutar el clustering musical
    """
    # Configuración
    DATASET_PATH = "tracks_features_500.csv"
    SAMPLE_SIZE = 500
    RANDOM_STATE = 42
    
    print("🎵 INICIANDO CLUSTERING MUSICAL")
    print("=" * 50)
    
    # Establecer semillas
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    
    print(f"🎯 Configuración:")
    print(f"   Dataset: {DATASET_PATH}")
    print(f"   Muestra: {SAMPLE_SIZE} canciones")
    print(f"   Features: {len(MUSICAL_FEATURES)} características")
    
    # 1. Cargar dataset
    sample_df = load_dataset_sample(DATASET_PATH, SAMPLE_SIZE, RANDOM_STATE)
    if sample_df is None:
        print("❌ Error al cargar el dataset. Terminando ejecución.")
        return
    
    # Verificar características disponibles
    available_features = [f for f in MUSICAL_FEATURES if f in sample_df.columns]
    missing_features = [f for f in MUSICAL_FEATURES if f not in sample_df.columns]
    
    print(f"\n🎼 Características musicales:")
    print(f"   ✅ Disponibles: {len(available_features)}/{len(MUSICAL_FEATURES)}")
    print(f"   📊 {available_features}")
    
    if missing_features:
        print(f"   ❌ Faltantes: {missing_features}")
    
    # 2. Limpiar datos
    cleaned_df, stats = clean_musical_features(sample_df, available_features)
    
    # 3. Preparar datos para clustering
    clustering_data, scaler, final_features = prepare_clustering_data(cleaned_df, available_features)
    
    if clustering_data is None:
        print("❌ No se pudieron preparar los datos para clustering")
        return
    
    # 4. Buscar K óptimo
    clustering_results = find_optimal_k(clustering_data, k_range=(4, 12), random_state=RANDOM_STATE)
    
    # 5. Visualizar métricas
    best_k = plot_clustering_metrics(clustering_results)
    
    # 6. Mostrar resultados tabulares
    print(f"\n📊 RESULTADOS DE CLUSTERING:")
    print("=" * 60)
    print("K  | Inercia | Silhouette | Distribución de Clusters")
    print("-" * 60)
    
    for result in clustering_results:
        distribution = [str(count) for count in result['cluster_distribution'].values()]
        distribution_str = f"[{', '.join(distribution)}]"
        print(f"{result['k']:2d} | {result['inertia']:7.1f} | {result['silhouette_score']:10.3f} | {distribution_str}")
    
    # Encontrar y destacar el mejor resultado
    best_result = max(clustering_results, key=lambda x: x['silhouette_score'])
    
    print(f"\n🏆 MEJOR CLUSTERING:")
    print(f"   🎯 K óptimo: {best_result['k']}")
    print(f"   📊 Silhouette Score: {best_result['silhouette_score']:.3f}")
    print(f"   📈 Inercia: {best_result['inertia']:.1f}")
    print(f"   🎵 Distribución: {list(best_result['cluster_distribution'].values())}")
    
    # 7. Analizar clusters musicales
    analyzed_df = analyze_music_clusters(cleaned_df, best_result, final_features, scaler)
    
    # 8. Crear visualización
    pca_model, data_2d = create_cluster_visualization(clustering_data, best_result['labels'], final_features)
    
    # 9. Guardar resultados
    print(f"\n💾 Guardando resultados...")
    analyzed_df.to_csv('clustering_results.csv', index=False)
    print(f"✅ Resultados guardados en: clustering_results.csv")
    
    # 10. Resumen ejecutivo
    print("\n🎉 RESUMEN EJECUTIVO DEL CLUSTERING MUSICAL")
    print("=" * 60)
    
    print(f"📊 CONFIGURACIÓN:")
    print(f"   🎵 Canciones analizadas: {len(sample_df)}")
    print(f"   🔧 Características utilizadas: {len(final_features)}")
    print(f"   🎯 Algoritmo: K-Means")
    print(f"   🎲 Semilla aleatoria: {RANDOM_STATE}")
    
    print(f"\n🏆 RESULTADO ÓPTIMO:")
    print(f"   🎯 Número de clusters: {best_result['k']}")
    print(f"   📊 Silhouette Score: {best_result['silhouette_score']:.3f}")
    print(f"   📈 Inercia: {best_result['inertia']:.1f}")
    print(f"   🎵 Distribución: {list(best_result['cluster_distribution'].values())}")
    
    # Interpretación del Silhouette Score
    score = best_result['silhouette_score']
    if score > 0.7:
        interpretation = "Excelente - Clusters muy bien definidos"
    elif score > 0.5:
        interpretation = "Bueno - Clusters claramente separados"
    elif score > 0.25:
        interpretation = "Aceptable - Clusters moderadamente definidos"
    else:
        interpretation = "Mejorable - Clusters poco definidos"
    
    print(f"   🎯 Calidad: {interpretation}")
    
    print(f"\n✅ LOGROS:")
    print(f"   🎼 Sistema de clustering musical implementado exitosamente")
    print(f"   📊 Patrones musicales identificados automáticamente")
    print(f"   🔍 Análisis estadístico completo realizado")
    print(f"   📈 Visualizaciones generadas")
    print(f"   💾 Resultados guardados para uso futuro")
    
    print(f"\n🎵 ¡CLUSTERING MUSICAL COMPLETADO EXITOSAMENTE! 🎉")

if __name__ == "__main__":
    main()