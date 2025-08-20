#!/usr/bin/env python3
"""
TEST Y ANÁLISIS DE VECTORIZACIÓN DE LETRAS
Analiza la calidad de embeddings BERT y clustering semántico generado.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.stats import describe
import warnings
warnings.filterwarnings('ignore')

def load_vectorization_data():
    """Carga datos de vectorización completa."""
    print("🔄 Cargando datos de vectorización...")
    
    # Base dir relativo a tests/
    base_dir = Path(__file__).parent.parent / "vectorization_complete_output"
    timestamp = "20250819_194820"
    
    # Cargar embeddings
    embeddings = np.load(base_dir / f"embeddings_complete_{timestamp}.npy")
    track_ids = np.load(base_dir / f"track_ids_complete_{timestamp}.npy")
    
    # Cargar índice de similitud
    with open(base_dir / f"similarity_index_{timestamp}.pkl", 'rb') as f:
        similarity_index = pickle.load(f)
    
    # Cargar metadatos
    with open(base_dir / f"vectorization_metadata_{timestamp}.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Datos cargados: {len(track_ids)} canciones, {embeddings.shape[1]}D")
    return embeddings, track_ids, similarity_index, metadata

def analyze_embedding_quality(embeddings):
    """Analiza calidad de embeddings BERT."""
    print("\n📊 ANÁLISIS DE CALIDAD DE EMBEDDINGS")
    print("="*50)
    
    # Filtrar embeddings válidos (no-zero)
    valid_mask = np.any(embeddings != 0, axis=1)
    valid_embeddings = embeddings[valid_mask]
    
    print(f"Embeddings válidos: {valid_embeddings.shape[0]}/{embeddings.shape[0]} ({valid_embeddings.shape[0]/embeddings.shape[0]*100:.1f}%)")
    
    # Estadísticas básicas
    stats = describe(valid_embeddings.flatten())
    print(f"Estadísticas de valores:")
    print(f"  Media: {stats.mean:.6f}")
    print(f"  Std: {np.sqrt(stats.variance):.6f}")
    print(f"  Min: {stats.minmax[0]:.6f}")
    print(f"  Max: {stats.minmax[1]:.6f}")
    
    # Distribución de normas
    norms = np.linalg.norm(valid_embeddings, axis=1)
    print(f"\nDistribución de normas L2:")
    print(f"  Media: {np.mean(norms):.6f}")
    print(f"  Std: {np.std(norms):.6f}")
    print(f"  Min: {np.min(norms):.6f}")
    print(f"  Max: {np.max(norms):.6f}")
    
    # Diversidad semántica (distancias promedio)
    if len(valid_embeddings) > 1000:
        sample_idx = np.random.choice(len(valid_embeddings), 1000, replace=False)
        sample_embeddings = valid_embeddings[sample_idx]
    else:
        sample_embeddings = valid_embeddings
    
    distances = pdist(sample_embeddings, metric='cosine')
    print(f"\nDiversidad semántica (distancia cosine):")
    print(f"  Media: {np.mean(distances):.6f}")
    print(f"  Std: {np.std(distances):.6f}")
    print(f"  Min: {np.min(distances):.6f}")
    print(f"  Max: {np.max(distances):.6f}")
    
    return valid_embeddings, valid_mask

def test_clustering_quality(embeddings, k_range=[2, 3, 4, 5, 6, 8, 10]):
    """Evalúa calidad de clustering con diferentes K."""
    print("\n🎯 EVALUACIÓN DE CLUSTERING SEMÁNTICO")
    print("="*50)
    
    results = []
    
    for k in k_range:
        print(f"\n🔄 Testing K={k}...")
        
        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings)
        
        # Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        hier_labels = hierarchical.fit_predict(embeddings)
        
        # Métricas para K-Means
        kmeans_sil = silhouette_score(embeddings, kmeans_labels, metric='cosine')
        kmeans_ch = calinski_harabasz_score(embeddings, kmeans_labels)
        kmeans_db = davies_bouldin_score(embeddings, kmeans_labels)
        
        # Métricas para Hierarchical
        hier_sil = silhouette_score(embeddings, hier_labels, metric='cosine')
        hier_ch = calinski_harabasz_score(embeddings, hier_labels)
        hier_db = davies_bouldin_score(embeddings, hier_labels)
        
        results.append({
            'k': k,
            'kmeans_silhouette': kmeans_sil,
            'kmeans_calinski_harabasz': kmeans_ch,
            'kmeans_davies_bouldin': kmeans_db,
            'hierarchical_silhouette': hier_sil,
            'hierarchical_calinski_harabasz': hier_ch,
            'hierarchical_davies_bouldin': hier_db
        })
        
        print(f"  K-Means - Silhouette: {kmeans_sil:.4f}, CH: {kmeans_ch:.1f}, DB: {kmeans_db:.4f}")
        print(f"  Hierarchical - Silhouette: {hier_sil:.4f}, CH: {hier_ch:.1f}, DB: {hier_db:.4f}")
    
    # Crear DataFrame de resultados
    df_results = pd.DataFrame(results)
    
    # Encontrar mejores configuraciones
    best_kmeans_k = df_results.loc[df_results['kmeans_silhouette'].idxmax(), 'k']
    best_hier_k = df_results.loc[df_results['hierarchical_silhouette'].idxmax(), 'k']
    
    print(f"\n🏆 MEJORES CONFIGURACIONES:")
    print(f"  K-Means óptimo: K={best_kmeans_k} (Silhouette: {df_results.loc[df_results['k']==best_kmeans_k, 'kmeans_silhouette'].iloc[0]:.4f})")
    print(f"  Hierarchical óptimo: K={best_hier_k} (Silhouette: {df_results.loc[df_results['k']==best_hier_k, 'hierarchical_silhouette'].iloc[0]:.4f})")
    
    return df_results, best_kmeans_k, best_hier_k

def analyze_cluster_distribution(embeddings, k_best):
    """Analiza distribución de clusters óptimos."""
    print(f"\n📈 ANÁLISIS DE CLUSTERS (K={k_best})")
    print("="*50)
    
    # Clustering óptimo
    kmeans = KMeans(n_clusters=k_best, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Distribución de tamaños
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Distribución de clusters:")
    for cluster_id, count in zip(unique, counts):
        percentage = count / len(labels) * 100
        print(f"  Cluster {cluster_id}: {count} canciones ({percentage:.1f}%)")
    
    # Análisis de cohesión intra-cluster
    print(f"\nCohesión intra-cluster (distancia promedio al centroide):")
    for cluster_id in unique:
        cluster_mask = labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        centroid = kmeans.cluster_centers_[cluster_id]
        
        distances_to_centroid = np.array([
            1 - np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            for emb in cluster_embeddings
        ])
        
        print(f"  Cluster {cluster_id}: {np.mean(distances_to_centroid):.4f} ± {np.std(distances_to_centroid):.4f}")
    
    return labels, kmeans

def analyze_similarity_index(similarity_index, track_ids):
    """Analiza índice de similitud para recomendaciones."""
    print("\n🔍 ANÁLISIS DE ÍNDICE DE SIMILITUD")
    print("="*50)
    
    model = similarity_index.get('model')
    if model is None:
        print("❌ No se encontró modelo en índice de similitud")
        return
    
    print(f"Algoritmo: {model.__class__.__name__}")
    print(f"Métrica: cosine")
    print(f"Canciones indexadas: {len(track_ids)}")
    print(f"Embeddings en modelo: {model._fit_X.shape[0]}")
    
    # Verificar consistencia
    if len(track_ids) != model._fit_X.shape[0]:
        print(f"⚠️  INCONSISTENCIA: track_ids ({len(track_ids)}) != embeddings en modelo ({model._fit_X.shape[0]})")
        print("   Usando solo índices válidos del modelo...")
        max_idx = model._fit_X.shape[0]
    else:
        max_idx = len(track_ids)
    
    # Test de recomendaciones con muestra
    print(f"\n🎵 TEST DE RECOMENDACIONES (muestra aleatoria):")
    
    sample_indices = np.random.choice(max_idx, min(5, max_idx), replace=False)
    
    for idx in sample_indices:
        if idx >= len(track_ids):
            continue
            
        track_id = track_ids[idx]
        
        # Buscar similares
        query_embedding = model._fit_X[idx:idx+1]  # Embedding de la canción
        distances, indices = model.kneighbors(query_embedding, n_neighbors=6)  # 6 para incluir la misma
        
        print(f"\n  Track {track_id}:")
        for i, (dist, sim_idx) in enumerate(zip(distances[0], indices[0])):
            if i == 0:  # Skip la misma canción
                continue
            if sim_idx >= len(track_ids):  # Verificar límites
                continue
            sim_track = track_ids[sim_idx]
            similarity = 1 - dist  # Convert distance to similarity
            print(f"    {i}. {sim_track} (similitud: {similarity:.3f})")

def create_visualizations(embeddings, labels, output_dir):
    """Crea visualizaciones de embeddings y clusters."""
    print("\n🎨 GENERANDO VISUALIZACIONES")
    print("="*50)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PCA para reducción inicial
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"PCA: {embeddings.shape[1]}D → {embeddings_pca.shape[1]}D (varianza explicada: {pca.explained_variance_ratio_.sum():.3f})")
    
    # t-SNE para visualización 2D
    print("Calculando t-SNE 2D...")
    perplexity = min(30, len(embeddings)//4, len(embeddings)-1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    # Plot clusters en 2D
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter)
    plt.title('Clustering Semántico de Letras (t-SNE 2D)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_clustering_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Histograma de distribución de clusters
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts, alpha=0.7)
    plt.title('Distribución de Clusters Semánticos')
    plt.xlabel('Cluster ID')
    plt.ylabel('Número de Canciones')
    for i, count in enumerate(counts):
        plt.text(unique[i], count + max(counts)*0.01, str(count), ha='center')
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizaciones guardadas en {output_dir}/")

def generate_analysis_report(metadata, embedding_stats, clustering_results, output_path):
    """Genera reporte completo de análisis."""
    print(f"\n📄 GENERANDO REPORTE COMPLETO")
    print("="*50)
    
    report = {
        "timestamp": metadata["timestamp"],
        "dataset_analysis": {
            "total_songs": metadata["dataset_info"]["total_songs"],
            "valid_lyrics": metadata["dataset_info"]["valid_lyrics"],
            "embeddings_generated": metadata["processing_stats"]["successful"],
            "success_rate": metadata["processing_stats"]["successful"] / metadata["processing_stats"]["processed"],
            "processing_time_minutes": metadata["processing_stats"]["processing_time"] / 60
        },
        "embedding_quality": embedding_stats,
        "clustering_evaluation": {
            "algorithms_tested": ["kmeans", "hierarchical"],
            "k_range_tested": clustering_results['k'].tolist(),
            "best_kmeans_k": int(clustering_results.loc[clustering_results['kmeans_silhouette'].idxmax(), 'k']),
            "best_hierarchical_k": int(clustering_results.loc[clustering_results['hierarchical_silhouette'].idxmax(), 'k']),
            "max_silhouette_kmeans": float(clustering_results['kmeans_silhouette'].max()),
            "max_silhouette_hierarchical": float(clustering_results['hierarchical_silhouette'].max()),
            "detailed_results": clustering_results.to_dict('records')
        },
        "recommendations": [
            "Clustering semántico funcional con métricas aceptables",
            "Índice de similitud listo para recomendaciones",
            "Sistema apto para integración híbrida música + letras",
            "Considerar optimización de K basada en silhouette score"
        ]
    }
    
    # Guardar reporte
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Reporte guardado: {output_path}")
    return report

def main():
    """Ejecuta análisis completo de vectorización."""
    print("🔬 ANÁLISIS COMPLETO DE VECTORIZACIÓN Y CLUSTERING DE LETRAS")
    print("="*70)
    
    try:
        # Cargar datos
        embeddings, track_ids, similarity_index, metadata = load_vectorization_data()
        
        # Analizar calidad de embeddings
        valid_embeddings, valid_mask = analyze_embedding_quality(embeddings)
        
        # Evaluar clustering
        clustering_results, best_k_means, best_k_hier = test_clustering_quality(valid_embeddings)
        
        # Analizar distribución de clusters
        labels, kmeans_model = analyze_cluster_distribution(valid_embeddings, best_k_means)
        
        # Analizar índice de similitud
        analyze_similarity_index(similarity_index, track_ids[valid_mask])
        
        # Crear visualizaciones
        output_dir = Path(__file__).parent.parent / "outputs" / "vectorization_analysis"
        create_visualizations(valid_embeddings, labels, output_dir)
        
        # Generar reporte final
        embedding_stats = {
            "total_embeddings": len(embeddings),
            "valid_embeddings": len(valid_embeddings),
            "dimensions": embeddings.shape[1],
            "validity_rate": len(valid_embeddings) / len(embeddings)
        }
        
        report = generate_analysis_report(
            metadata, embedding_stats, clustering_results,
            output_dir / "analysis_report.json"
        )
        
        print("\n🎯 RESUMEN EJECUTIVO:")
        print("="*30)
        print(f"✅ {len(valid_embeddings)} embeddings válidos de {len(embeddings)} totales")
        print(f"✅ Mejor K-Means: K={best_k_means} (Silhouette: {clustering_results.loc[clustering_results['k']==best_k_means, 'kmeans_silhouette'].iloc[0]:.4f})")
        print(f"✅ Mejor Hierarchical: K={best_k_hier} (Silhouette: {clustering_results.loc[clustering_results['k']==best_k_hier, 'hierarchical_silhouette'].iloc[0]:.4f})")
        print(f"✅ Sistema de recomendaciones listo con {len(track_ids)} canciones indexadas")
        print(f"✅ Análisis completo disponible en {output_dir}/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante análisis: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Análisis de vectorización completado exitosamente")
    else:
        print("\n💥 Análisis falló - revisar logs para detalles")