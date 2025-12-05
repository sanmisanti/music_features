#!/usr/bin/env python3
"""
Script para generar visualizaciones de calidad de embeddings BERT.

Este script analiza los embeddings BERT generados y crea visualizaciones para
la sección 1.3.3 "Análisis de Calidad de Embeddings BERT" del informe.

Genera:
1. Distribución de distancias coseno entre embeddings
2. Análisis dimensional y utilización del espacio vectorial 384D
3. Visualización de clustering readiness semántico
4. Proyección t-SNE/UMAP de embeddings para separabilidad visual

Uso:
    python generate_bert_quality_plots.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")

def load_bert_embeddings():
    """
    Cargar embeddings BERT del dataset unificado.
    """
    print("📂 Cargando embeddings BERT...")

    # Buscar archivo de embeddings BERT
    possible_paths = [
        "clustering_evaluation_project/phase1_dataset_unification/arrays_20250822_004929/semantic_embeddings.npy",
        "vectorization_complete_output/bert_embeddings_complete.npy",
        "clustering/models/lyrics_models/bert_embeddings.npy"
    ]

    embeddings = None
    used_path = None

    for path in possible_paths:
        if Path(path).exists():
            try:
                embeddings = np.load(path)
                used_path = path
                break
            except Exception as e:
                print(f"❌ Error cargando {path}: {e}")
                continue

    if embeddings is None:
        raise FileNotFoundError("No se encontraron embeddings BERT válidos")

    print(f"✅ Embeddings cargados desde: {used_path}")
    print(f"   Dimensiones: {embeddings.shape}")
    print(f"   Tipo: {embeddings.dtype}")

    return embeddings

def load_metadata():
    """
    Cargar metadatos para enriquecer visualizaciones.
    """
    print("\n📊 Cargando metadatos...")

    metadata_paths = [
        "clustering_evaluation_project/phase1_dataset_unification/aligned_songs_multimodal_20250822_011617.csv",
        "data/3_selected/picked_data_optimal.csv"
    ]

    for path in metadata_paths:
        if Path(path).exists():
            try:
                if "aligned_songs" in path:
                    df = pd.read_csv(path, sep='^')
                else:
                    df = pd.read_csv(path, sep='^')
                print(f"✅ Metadatos cargados: {df.shape[0]} canciones")
                return df
            except Exception as e:
                print(f"❌ Error cargando metadatos {path}: {e}")
                continue

    print("⚠️  No se pudieron cargar metadatos, usando datos sintéticos")
    return None

def calculate_hopkins_statistic(embeddings, n_samples=1000):
    """
    Calcular Hopkins Statistic para clustering readiness.
    """
    print("\n🎯 Calculando Hopkins Statistic...")

    if len(embeddings) > n_samples:
        # Muestrear para eficiencia
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        data = embeddings[indices]
    else:
        data = embeddings

    # Calcular Hopkins Statistic
    n = len(data)
    m = int(0.1 * n)  # 10% de los datos para el test

    # Seleccionar puntos aleatorios del dataset
    real_indices = np.random.choice(n, m, replace=False)
    real_points = data[real_indices]

    # Generar puntos sintéticos uniformes
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    synthetic_points = np.random.uniform(min_vals, max_vals, (m, data.shape[1]))

    # Calcular distancias mínimas
    distances_real = []
    distances_synthetic = []

    for i in range(m):
        # Distancia del punto real a su vecino más cercano
        real_point = real_points[i].reshape(1, -1)
        other_indices = [j for j in range(n) if j not in [real_indices[i]]]
        other_points = data[other_indices]
        real_dists = cosine_distances(real_point, other_points)[0]
        distances_real.append(real_dists.min())

        # Distancia del punto sintético a su vecino más cercano en datos reales
        synthetic_point = synthetic_points[i].reshape(1, -1)
        synthetic_dists = cosine_distances(synthetic_point, data)[0]
        distances_synthetic.append(synthetic_dists.min())

    # Hopkins Statistic
    sum_synthetic = sum(distances_synthetic)
    sum_real = sum(distances_real)
    hopkins = sum_synthetic / (sum_synthetic + sum_real)

    print(f"   Hopkins Statistic: {hopkins:.3f}")
    return hopkins

def plot_cosine_distances(embeddings):
    """
    Crear gráfico de distribución de distancias coseno.
    """
    print("\n📈 Generando distribución de distancias coseno...")

    # Muestrear para eficiencia computacional
    n_samples = min(2000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_embeddings = embeddings[indices]

    # Calcular matriz de distancias coseno
    cosine_dist_matrix = cosine_distances(sample_embeddings)

    # Extraer distancias (triángulo superior, sin diagonal)
    mask = np.triu(np.ones_like(cosine_dist_matrix, dtype=bool), k=1)
    distances = cosine_dist_matrix[mask]

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histograma de distancias
    ax1.hist(distances, bins=50, alpha=0.7, color='skyblue', density=True, edgecolor='black')
    ax1.axvline(distances.mean(), color='red', linestyle='--',
                label=f'Media: {distances.mean():.3f}')
    ax1.axvline(np.median(distances), color='orange', linestyle='--',
                label=f'Mediana: {np.median(distances):.3f}')

    ax1.set_title('Distribución de Distancias Coseno\nentre Embeddings BERT', fontweight='bold')
    ax1.set_xlabel('Distancia Coseno')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot para evaluar normalidad
    stats.probplot(distances, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot vs Distribución Normal', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Estadísticas en el gráfico
    stats_text = f'''Estadísticas:
Media: {distances.mean():.3f}
Std: {distances.std():.3f}
Min: {distances.min():.3f}
Max: {distances.max():.3f}
Muestras: {len(distances):,}'''

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig

def plot_dimensional_analysis(embeddings):
    """
    Crear análisis dimensional del espacio vectorial 384D.
    """
    print("\n🔍 Generando análisis dimensional...")

    # PCA para análisis de componentes
    pca = PCA()
    pca.fit(embeddings)

    # Calcular entropía dimensional aproximada
    explained_var = pca.explained_variance_ratio_
    entropy = -np.sum(explained_var * np.log2(explained_var + 1e-10))

    # Crear figura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Varianza explicada por componente
    n_components_show = min(50, len(explained_var))
    ax1.bar(range(1, n_components_show + 1), explained_var[:n_components_show],
            alpha=0.7, color='lightcoral')
    ax1.set_title('Varianza Explicada por Componente Principal', fontweight='bold')
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Varianza Explicada')
    ax1.grid(True, alpha=0.3)

    # 2. Varianza acumulada
    cumsum_var = np.cumsum(explained_var)
    ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, color='darkgreen', linewidth=2)
    ax2.axhline(y=0.95, color='red', linestyle='--', label='95% Varianza')
    ax2.axhline(y=0.99, color='orange', linestyle='--', label='99% Varianza')
    ax2.set_title('Varianza Explicada Acumulada', fontweight='bold')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Varianza Acumulada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(200, len(cumsum_var)))

    # 3. Distribución de normas de embeddings
    norms = np.linalg.norm(embeddings, axis=1)
    ax3.hist(norms, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(norms.mean(), color='red', linestyle='--',
                label=f'Media: {norms.mean():.3f}')
    ax3.set_title('Distribución de Normas L2\nde Embeddings BERT', fontweight='bold')
    ax3.set_xlabel('Norma L2')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Heatmap de correlaciones entre dimensiones (muestra)
    n_dims_show = min(20, embeddings.shape[1])
    sample_dims = np.random.choice(embeddings.shape[1], n_dims_show, replace=False)
    corr_matrix = np.corrcoef(embeddings[:, sample_dims].T)

    im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax4.set_title(f'Correlaciones entre Dimensiones\n(Muestra de {n_dims_show} dimensiones)', fontweight='bold')
    ax4.set_xlabel('Dimensión')
    ax4.set_ylabel('Dimensión')
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # Añadir estadísticas
    stats_text = f'''Análisis Dimensional:
Dimensiones: {embeddings.shape[1]}
Entropía: {entropy:.2f} bits
Varianza 95%: {np.where(cumsum_var >= 0.95)[0][0]+1} componentes
Norma L2 media: {norms.mean():.3f}'''

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    return fig

def plot_clustering_readiness(embeddings, metadata=None):
    """
    Crear visualización de clustering readiness.
    """
    print("\n🎯 Generando visualización de clustering readiness...")

    # Calcular Hopkins Statistic
    hopkins = calculate_hopkins_statistic(embeddings)

    # Reducción dimensional para visualización
    print("   Calculando t-SNE...")
    n_samples = min(3000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_embeddings = embeddings[indices]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(sample_embeddings)

    # PCA para comparación
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(sample_embeddings)

    # Crear figura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. t-SNE plot
    scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          alpha=0.6, s=10, c=range(len(embeddings_2d)), cmap='viridis')
    ax1.set_title('Proyección t-SNE de Embeddings BERT\n(Estructura Natural)', fontweight='bold')
    ax1.set_xlabel('t-SNE Dimensión 1')
    ax1.set_ylabel('t-SNE Dimensión 2')
    plt.colorbar(scatter1, ax=ax1, shrink=0.8)

    # 2. PCA plot
    scatter2 = ax2.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                          alpha=0.6, s=10, c=range(len(embeddings_pca)), cmap='plasma')
    ax2.set_title('Proyección PCA de Embeddings BERT\n(Varianza Máxima)', fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
    plt.colorbar(scatter2, ax=ax2, shrink=0.8)

    # 3. Hopkins Statistic visualization
    hopkins_categories = ['Uniforme\n(H<0.5)', 'Bueno\n(0.5-0.7)', 'Excelente\n(>0.7)']
    hopkins_values = [0.3, 0.6, hopkins]  # Comparación
    colors = ['red', 'orange', 'green']

    bars = ax3.bar(hopkins_categories, hopkins_values, color=colors, alpha=0.7)
    ax3.axhline(y=hopkins, color='blue', linestyle='--', linewidth=2,
                label=f'Nuestro Dataset: {hopkins:.3f}')
    ax3.set_title('Hopkins Statistic\n(Clustering Readiness)', fontweight='bold')
    ax3.set_ylabel('Hopkins Statistic')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Añadir valores sobre barras
    for i, (bar, value) in enumerate(zip(bars, hopkins_values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 4. Distribución de distancias intra vs inter-cluster (aproximación)
    from sklearn.cluster import KMeans

    # K-means para estimar clusters
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(sample_embeddings)

    # Calcular distancias intra e inter-cluster
    intra_distances = []
    inter_distances = []

    for i in range(len(cluster_labels)):
        cluster_id = cluster_labels[i]
        point = sample_embeddings[i]

        # Distancias intra-cluster
        same_cluster_mask = cluster_labels == cluster_id
        if np.sum(same_cluster_mask) > 1:
            same_cluster_points = sample_embeddings[same_cluster_mask]
            intra_dists = cosine_distances([point], same_cluster_points)[0]
            intra_distances.extend(intra_dists[intra_dists > 0])  # Excluir distancia a sí mismo

        # Distancias inter-cluster
        diff_cluster_mask = cluster_labels != cluster_id
        if np.sum(diff_cluster_mask) > 0:
            diff_cluster_points = sample_embeddings[diff_cluster_mask]
            inter_dists = cosine_distances([point], diff_cluster_points)[0]
            inter_distances.extend(inter_dists)

    # Plot de distribuciones
    ax4.hist(intra_distances, bins=30, alpha=0.7, label='Intra-cluster',
             color='lightblue', density=True)
    ax4.hist(inter_distances, bins=30, alpha=0.7, label='Inter-cluster',
             color='lightcoral', density=True)

    ratio = np.mean(intra_distances) / np.mean(inter_distances) if inter_distances else 0
    ax4.axvline(np.mean(intra_distances), color='blue', linestyle='--',
                label=f'Media Intra: {np.mean(intra_distances):.3f}')
    ax4.axvline(np.mean(inter_distances), color='red', linestyle='--',
                label=f'Media Inter: {np.mean(inter_distances):.3f}')

    ax4.set_title(f'Separabilidad de Clusters\nRatio Intra/Inter: {ratio:.3f}', fontweight='bold')
    ax4.set_xlabel('Distancia Coseno')
    ax4.set_ylabel('Densidad')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, hopkins

def main():
    """
    Función principal.
    """
    print("🎵 Generando visualizaciones de calidad de embeddings BERT")
    print("=" * 80)

    try:
        # Cargar datos
        embeddings = load_bert_embeddings()
        metadata = load_metadata()

        # Crear directorio de salida
        output_dir = Path("INFORME_FINAL/imagenes_bert_quality")
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"\n📁 Directorio de salida: {output_dir}")

        # Generar visualizaciones
        print("\n🎨 Generando gráficos...")

        # 1. Distribución de distancias coseno
        print("  1️⃣ Distribución de distancias coseno...")
        fig1 = plot_cosine_distances(embeddings)
        fig1.savefig(output_dir / "distribuciones_distancias_coseno.png",
                     dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig1)

        # 2. Análisis dimensional
        print("  2️⃣ Análisis dimensional...")
        fig2 = plot_dimensional_analysis(embeddings)
        fig2.savefig(output_dir / "analisis_dimensional_384d.png",
                     dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)

        # 3. Clustering readiness
        print("  3️⃣ Clustering readiness...")
        fig3, hopkins = plot_clustering_readiness(embeddings, metadata)
        fig3.savefig(output_dir / "clustering_readiness_semantico.png",
                     dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig3)

        print("\n✅ ¡Visualizaciones generadas exitosamente!")
        print(f"📂 Ubicación: {output_dir.absolute()}")
        print(f"\n📋 Archivos creados:")
        for img_file in output_dir.glob("*.png"):
            print(f"  - {img_file.name}")

        print(f"\n🎯 Métricas calculadas:")
        print(f"  📊 Diversidad semántica: distancias coseno analizadas")
        print(f"  📈 Utilización espacio 384D: PCA y entropía dimensional")
        print(f"  🎯 Hopkins Statistic: {hopkins:.3f} ({'EXCELENTE' if hopkins > 0.7 else 'BUENO' if hopkins > 0.5 else 'REGULAR'})")

    except Exception as e:
        print(f"❌ Error durante la generación: {e}")
        raise

if __name__ == "__main__":
    main()