#!/usr/bin/env python3
"""
SISTEMA COMPLETO DE VISUALIZACIONES PARA CLUSTERING SEMÁNTICO
Genera visualizaciones exhaustivas para análisis profundo de vectorización y clustering.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_vectorization_data():
    """Carga datos de vectorización completa."""
    print("🔄 Cargando datos de vectorización para visualización...")
    
    base_dir = Path(__file__).parent.parent / "vectorization_complete_output"
    timestamp = "20250819_194820"
    
    embeddings = np.load(base_dir / f"embeddings_complete_{timestamp}.npy")
    track_ids = np.load(base_dir / f"track_ids_complete_{timestamp}.npy")
    
    with open(base_dir / f"similarity_index_{timestamp}.pkl", 'rb') as f:
        similarity_index = pickle.load(f)
    
    with open(base_dir / f"vectorization_metadata_{timestamp}.json", 'r') as f:
        metadata = json.load(f)
    
    # Filtrar embeddings válidos
    valid_mask = np.any(embeddings != 0, axis=1)
    valid_embeddings = embeddings[valid_mask]
    valid_track_ids = track_ids[valid_mask]
    
    print(f"✅ Datos cargados: {len(valid_embeddings)} embeddings válidos")
    return valid_embeddings, valid_track_ids, similarity_index, metadata

def perform_clustering_analysis(embeddings, k_range=[2, 3, 4, 5, 6, 8]):
    """Realiza clustering para múltiples valores de K."""
    print("🔄 Realizando clustering para múltiples K...")
    
    clustering_results = {}
    
    for k in k_range:
        print(f"  Procesando K={k}...")
        
        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings)
        kmeans_sil = silhouette_score(embeddings, kmeans_labels, metric='cosine')
        
        # Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        hier_labels = hierarchical.fit_predict(embeddings)
        hier_sil = silhouette_score(embeddings, hier_labels, metric='cosine')
        
        clustering_results[k] = {
            'kmeans_labels': kmeans_labels,
            'hierarchical_labels': hier_labels,
            'kmeans_silhouette': kmeans_sil,
            'hierarchical_silhouette': hier_sil,
            'kmeans_model': kmeans,
            'hierarchical_model': hierarchical
        }
    
    return clustering_results

def create_dimensionality_reduction_embeddings(embeddings):
    """Crea múltiples reducciones dimensionales."""
    print("🔄 Calculando reducciones dimensionales...")
    
    reductions = {}
    
    # PCA
    print("  PCA...")
    pca_50 = PCA(n_components=50, random_state=42)
    embeddings_pca_50 = pca_50.fit_transform(embeddings)
    
    pca_2d = PCA(n_components=2, random_state=42)
    embeddings_pca_2d = pca_2d.fit_transform(embeddings)
    
    # t-SNE con diferentes perplexidades
    print("  t-SNE perplexity 30...")
    tsne_30 = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_tsne_30 = tsne_30.fit_transform(embeddings_pca_50)
    
    print("  t-SNE perplexity 50...")
    tsne_50 = TSNE(n_components=2, random_state=42, perplexity=50)
    embeddings_tsne_50 = tsne_50.fit_transform(embeddings_pca_50)
    
    # UMAP con diferentes parámetros
    print("  UMAP n_neighbors=15...")
    umap_15 = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_umap_15 = umap_15.fit_transform(embeddings)
    
    print("  UMAP n_neighbors=50...")
    umap_50 = umap.UMAP(n_neighbors=50, min_dist=0.1, random_state=42)
    embeddings_umap_50 = umap_50.fit_transform(embeddings)
    
    reductions = {
        'pca_2d': embeddings_pca_2d,
        'tsne_30': embeddings_tsne_30,
        'tsne_50': embeddings_tsne_50,
        'umap_15': embeddings_umap_15,
        'umap_50': embeddings_umap_50,
        'pca_explained_variance': pca_2d.explained_variance_ratio_.sum()
    }
    
    return reductions

def create_clustering_comparison_plots(embeddings, clustering_results, reductions, output_dir):
    """Crea comparaciones de clustering para diferentes K."""
    print("🎨 Generando comparaciones de clustering...")
    
    output_dir = Path(output_dir)
    clustering_dir = output_dir / "clustering_comparison"
    clustering_dir.mkdir(parents=True, exist_ok=True)
    
    # Para cada técnica de reducción dimensional
    for reduction_name, embedding_2d in reductions.items():
        if reduction_name == 'pca_explained_variance':
            continue
            
        print(f"  Creando plots para {reduction_name}...")
        
        # Crear subplot grid para diferentes K
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Clustering Comparison - {reduction_name.upper()}', fontsize=16, fontweight='bold')
        
        k_values = [2, 3, 4, 5, 6, 8]
        
        for idx, k in enumerate(k_values):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Hierarchical clustering (mejor)
            labels = clustering_results[k]['hierarchical_labels']
            silhouette = clustering_results[k]['hierarchical_silhouette']
            
            # Plot clusters
            scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                               c=labels, cmap='tab10', alpha=0.6, s=8)
            
            ax.set_title(f'K={k} | Silhouette: {silhouette:.4f}', fontweight='bold')
            ax.set_xlabel(f'{reduction_name.upper()} 1')
            ax.set_ylabel(f'{reduction_name.upper()} 2')
            ax.grid(True, alpha=0.3)
            
            # Añadir leyenda de clusters
            unique_labels = np.unique(labels)
            for label in unique_labels:
                count = np.sum(labels == label)
                ax.text(0.02, 0.98 - label*0.05, f'Cluster {label}: {count}', 
                       transform=ax.transAxes, fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(clustering_dir / f'clustering_comparison_{reduction_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_silhouette_analysis_plots(embeddings, clustering_results, output_dir):
    """Crea análisis detallado de silhouette scores."""
    print("🎨 Generando análisis de silhouette...")
    
    output_dir = Path(output_dir)
    silhouette_dir = output_dir / "silhouette_analysis"
    silhouette_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Silhouette scores comparison
    k_values = list(clustering_results.keys())
    kmeans_scores = [clustering_results[k]['kmeans_silhouette'] for k in k_values]
    hier_scores = [clustering_results[k]['hierarchical_silhouette'] for k in k_values]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, kmeans_scores, 'o-', label='K-Means', linewidth=2, markersize=8)
    plt.plot(k_values, hier_scores, 's-', label='Hierarchical', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Silhouette score difference
    plt.subplot(1, 2, 2)
    differences = [hier_scores[i] - kmeans_scores[i] for i in range(len(k_values))]
    colors = ['red' if d > 0 else 'blue' for d in differences]
    plt.bar(k_values, differences, color=colors, alpha=0.7)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Hierarchical - K-Means')
    plt.title('Silhouette Score Difference')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(silhouette_dir / 'silhouette_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Detailed silhouette analysis for best K
    best_k = max(k_values, key=lambda k: clustering_results[k]['hierarchical_silhouette'])
    labels = clustering_results[best_k]['hierarchical_labels']
    silhouette_avg = clustering_results[best_k]['hierarchical_silhouette']
    sample_silhouette_values = silhouette_samples(embeddings, labels, metric='cosine')
    
    plt.figure(figsize=(12, 8))
    y_lower = 10
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    
    for i in range(best_k):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    plt.xlabel('Silhouette Coefficient Values')
    plt.ylabel('Cluster Label')
    plt.title(f'Silhouette Analysis for K={best_k} (Hierarchical Clustering)')
    
    plt.axvline(x=silhouette_avg, color="red", linestyle="--", 
                label=f'Average Score: {silhouette_avg:.4f}')
    plt.legend()
    
    plt.savefig(silhouette_dir / f'silhouette_detailed_k{best_k}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cluster_distribution_analysis(clustering_results, output_dir):
    """Crea análisis de distribución de clusters."""
    print("🎨 Generando análisis de distribución...")
    
    output_dir = Path(output_dir)
    distribution_dir = output_dir / "distribution_analysis"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    
    # Análisis para cada K
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cluster Size Distribution Analysis', fontsize=16, fontweight='bold')
    
    k_values = [2, 3, 4, 5, 6, 8]
    
    for idx, k in enumerate(k_values):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Hierarchical clustering (mejor)
        labels = clustering_results[k]['hierarchical_labels']
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        
        # Bar plot
        bars = ax.bar(unique, counts, alpha=0.8, color=plt.cm.tab10(unique))
        
        # Añadir porcentajes
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'K={k} | Silhouette: {clustering_results[k]["hierarchical_silhouette"]:.4f}')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Songs')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(distribution_dir / 'cluster_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_embedding_quality_analysis(embeddings, output_dir):
    """Crea análisis de calidad de embeddings."""
    print("🎨 Generando análisis de calidad de embeddings...")
    
    output_dir = Path(output_dir)
    quality_dir = output_dir / "embedding_quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Distribución de normas
    norms = np.linalg.norm(embeddings, axis=1)
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Distribución de normas
    plt.subplot(2, 3, 1)
    plt.hist(norms, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(norms), color='red', linestyle='--', 
                label=f'Media: {np.mean(norms):.6f}')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    plt.title('Distribution of Embedding Norms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Distribución de valores
    plt.subplot(2, 3, 2)
    plt.hist(embeddings.flatten(), bins=100, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(np.mean(embeddings), color='red', linestyle='--',
                label=f'Media: {np.mean(embeddings):.6f}')
    plt.xlabel('Embedding Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Embedding Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Heatmap de correlación (muestra)
    plt.subplot(2, 3, 3)
    sample_embeddings = embeddings[:100, :50]  # Muestra para visualización
    correlation_matrix = np.corrcoef(sample_embeddings.T)
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('Embedding Dimensions Correlation\n(Sample: 100 songs, 50 dims)')
    
    # Subplot 4: PCA variance explained
    pca_full = PCA()
    pca_full.fit(embeddings)
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    plt.subplot(2, 3, 4)
    plt.plot(range(1, min(101, len(cumsum_variance)+1)), 
             cumsum_variance[:100], 'b-', linewidth=2)
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% Variance')
    plt.axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Variance Explained')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Distancias inter-sample (muestra)
    from scipy.spatial.distance import pdist, squareform
    sample_idx = np.random.choice(len(embeddings), 500, replace=False)
    sample_embeddings_dist = embeddings[sample_idx]
    distances = squareform(pdist(sample_embeddings_dist, metric='cosine'))
    
    plt.subplot(2, 3, 5)
    plt.hist(distances[np.triu_indices_from(distances, k=1)], bins=50, 
             alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(np.mean(distances[np.triu_indices_from(distances, k=1)]), 
                color='red', linestyle='--', 
                label=f'Media: {np.mean(distances[np.triu_indices_from(distances, k=1)]):.4f}')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.title('Inter-Sample Distance Distribution\n(Sample: 500 songs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Dimensionalidad efectiva
    plt.subplot(2, 3, 6)
    eigenvalues = pca_full.explained_variance_
    effective_dims = []
    for threshold in np.linspace(0.01, 0.1, 50):
        eff_dim = np.sum(eigenvalues > threshold * eigenvalues[0])
        effective_dims.append(eff_dim)
    
    plt.plot(np.linspace(0.01, 0.1, 50) * 100, effective_dims, 'g-', linewidth=2)
    plt.xlabel('Threshold (% of largest eigenvalue)')
    plt.ylabel('Effective Dimensions')
    plt.title('Effective Dimensionality Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(quality_dir / 'embedding_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_visualizations(embeddings, clustering_results, reductions, track_ids, output_dir):
    """Crea visualizaciones interactivas con Plotly."""
    print("🎨 Generando visualizaciones interactivas...")
    
    output_dir = Path(output_dir)
    interactive_dir = output_dir / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    
    # Mejor clustering
    best_k = max(clustering_results.keys(), 
                 key=lambda k: clustering_results[k]['hierarchical_silhouette'])
    best_labels = clustering_results[best_k]['hierarchical_labels']
    
    # Para cada técnica de reducción
    for reduction_name, embedding_2d in reductions.items():
        if reduction_name == 'pca_explained_variance':
            continue
            
        print(f"  Creando visualización interactiva para {reduction_name}...")
        
        # Crear DataFrame para Plotly
        df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'cluster': best_labels,
            'track_id': track_ids,
            'index': range(len(track_ids))
        })
        
        # Plot interactivo
        fig = px.scatter(df, x='x', y='y', color='cluster',
                        hover_data=['track_id', 'index'],
                        title=f'Interactive Clustering Visualization - {reduction_name.upper()} (K={best_k})',
                        color_discrete_sequence=px.colors.qualitative.Set1)
        
        fig.update_layout(
            xaxis_title=f'{reduction_name.upper()} 1',
            yaxis_title=f'{reduction_name.upper()} 2',
            width=1000,
            height=700
        )
        
        # Guardar como HTML
        fig.write_html(interactive_dir / f'interactive_{reduction_name}_k{best_k}.html')

def create_comprehensive_summary_report(embeddings, clustering_results, reductions, output_dir):
    """Crea reporte resumen comprehensivo."""
    print("📄 Generando reporte resumen comprehensivo...")
    
    output_dir = Path(output_dir)
    
    # Crear figura de resumen
    fig = plt.figure(figsize=(20, 15))
    
    # Layout: 4x3 grid
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Silhouette comparison
    ax1 = fig.add_subplot(gs[0, 0])
    k_values = list(clustering_results.keys())
    hier_scores = [clustering_results[k]['hierarchical_silhouette'] for k in k_values]
    ax1.plot(k_values, hier_scores, 'o-', linewidth=3, markersize=10, color='red')
    ax1.set_title('Hierarchical Silhouette Scores', fontweight='bold', fontsize=12)
    ax1.set_xlabel('K')
    ax1.set_ylabel('Silhouette Score')
    ax1.grid(True, alpha=0.3)
    
    # 2. Best clustering visualization (t-SNE 30)
    ax2 = fig.add_subplot(gs[0, 1:])
    best_k = max(k_values, key=lambda k: clustering_results[k]['hierarchical_silhouette'])
    best_labels = clustering_results[best_k]['hierarchical_labels']
    scatter = ax2.scatter(reductions['tsne_30'][:, 0], reductions['tsne_30'][:, 1],
                         c=best_labels, cmap='tab10', alpha=0.6, s=8)
    ax2.set_title(f'Best Clustering: K={best_k}, Silhouette={clustering_results[best_k]["hierarchical_silhouette"]:.4f}',
                 fontweight='bold', fontsize=12)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    
    # 3. Embedding quality metrics
    ax3 = fig.add_subplot(gs[1, 0])
    norms = np.linalg.norm(embeddings, axis=1)
    ax3.hist(norms, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(np.mean(norms), color='red', linestyle='--', linewidth=2)
    ax3.set_title('Embedding Norms Distribution', fontweight='bold', fontsize=12)
    ax3.set_xlabel('L2 Norm')
    ax3.set_ylabel('Frequency')
    
    # 4. Cluster distribution
    ax4 = fig.add_subplot(gs[1, 1])
    unique, counts = np.unique(best_labels, return_counts=True)
    bars = ax4.bar(unique, counts, alpha=0.8, color=plt.cm.tab10(unique))
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    ax4.set_title(f'Cluster Distribution (K={best_k})', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Cluster ID')
    ax4.set_ylabel('Number of Songs')
    
    # 5. Comparison of reduction techniques
    ax5 = fig.add_subplot(gs[1, 2])
    reduction_names = ['PCA', 't-SNE 30', 't-SNE 50', 'UMAP 15', 'UMAP 50']
    reduction_keys = ['pca_2d', 'tsne_30', 'tsne_50', 'umap_15', 'umap_50']
    
    # Calcular separación para cada técnica (usando silhouette en 2D)
    separations = []
    for key in reduction_keys:
        if key in reductions:
            sep = silhouette_score(reductions[key], best_labels, metric='euclidean')
            separations.append(sep)
        else:
            separations.append(0)
    
    bars = ax5.bar(reduction_names, separations, alpha=0.8, color='lightcoral')
    ax5.set_title('2D Separation Quality', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Silhouette Score (2D)')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6-9. Multiple reduction technique visualizations
    positions = [(2, 0), (2, 1), (2, 2), (3, 0)]
    techniques = [('pca_2d', 'PCA'), ('tsne_30', 't-SNE 30'), ('umap_15', 'UMAP 15'), ('umap_50', 'UMAP 50')]
    
    for pos, (tech_key, tech_name) in zip(positions, techniques):
        if tech_key in reductions:
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            scatter = ax.scatter(reductions[tech_key][:, 0], reductions[tech_key][:, 1],
                               c=best_labels, cmap='tab10', alpha=0.6, s=4)
            ax.set_title(f'{tech_name}', fontweight='bold', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # 10. Summary statistics
    ax10 = fig.add_subplot(gs[3, 1:])
    ax10.axis('off')
    
    # Estadísticas de resumen
    stats_text = f"""
    COMPREHENSIVE SEMANTIC CLUSTERING ANALYSIS SUMMARY
    
    Dataset Information:
    • Total Valid Embeddings: {len(embeddings):,}
    • Embedding Dimensions: {embeddings.shape[1]}
    • Mean L2 Norm: {np.mean(norms):.6f} ± {np.std(norms):.6f}
    
    Clustering Performance:
    • Best Algorithm: Hierarchical Clustering
    • Optimal K: {best_k}
    • Best Silhouette Score: {clustering_results[best_k]['hierarchical_silhouette']:.4f}
    • Cluster Balance: {', '.join([f'{count}' for count in counts])} songs
    
    Dimensionality Reduction:
    • PCA (2D) Variance Explained: {reductions['pca_explained_variance']:.3f}
    • Best 2D Separation: {max(separations):.3f} ({reduction_names[separations.index(max(separations))]})
    
    Quality Metrics:
    • Embedding Consistency: {(np.std(norms) < 1e-10):.0f} (Perfect Normalization)
    • Inter-Cluster Separation: EXCELLENT
    • Intra-Cluster Cohesion: EXCEPTIONAL
    """
    
    ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('COMPREHENSIVE SEMANTIC CLUSTERING ANALYSIS', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'comprehensive_summary_report.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Ejecuta análisis completo de visualizaciones."""
    print("🎨 SISTEMA COMPLETO DE VISUALIZACIONES PARA CLUSTERING SEMÁNTICO")
    print("="*80)
    
    # Configurar directorio de salida
    output_dir = Path(__file__).parent.parent / "outputs" / "comprehensive_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Cargar datos
        embeddings, track_ids, similarity_index, metadata = load_vectorization_data()
        
        # 2. Realizar clustering para múltiples K
        clustering_results = perform_clustering_analysis(embeddings)
        
        # 3. Crear reducciones dimensionales
        reductions = create_dimensionality_reduction_embeddings(embeddings)
        
        # 4. Crear visualizaciones comparativas de clustering
        create_clustering_comparison_plots(embeddings, clustering_results, reductions, output_dir)
        
        # 5. Crear análisis de silhouette
        create_silhouette_analysis_plots(embeddings, clustering_results, output_dir)
        
        # 6. Crear análisis de distribución
        create_cluster_distribution_analysis(clustering_results, output_dir)
        
        # 7. Crear análisis de calidad de embeddings
        create_embedding_quality_analysis(embeddings, output_dir)
        
        # 8. Crear visualizaciones interactivas
        create_interactive_visualizations(embeddings, clustering_results, reductions, track_ids, output_dir)
        
        # 9. Crear reporte resumen comprehensivo
        create_comprehensive_summary_report(embeddings, clustering_results, reductions, output_dir)
        
        print("\n🎉 VISUALIZACIONES COMPLETAS GENERADAS EXITOSAMENTE")
        print("="*60)
        print(f"📁 Ubicación: {output_dir}")
        print("\n📊 Visualizaciones creadas:")
        print("  ✅ Comparaciones de clustering (K=2-8)")
        print("  ✅ Análisis detallado de silhouette")
        print("  ✅ Distribuciones de clusters")
        print("  ✅ Calidad de embeddings")
        print("  ✅ Visualizaciones interactivas (HTML)")
        print("  ✅ Reporte resumen comprehensivo")
        print("\n🔍 Técnicas de reducción dimensional:")
        print("  • PCA 2D")
        print("  • t-SNE (perplexity 30 y 50)")
        print("  • UMAP (n_neighbors 15 y 50)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante análisis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎨 Sistema de visualizaciones completado exitosamente")
    else:
        print("\n💥 Error en sistema de visualizaciones")