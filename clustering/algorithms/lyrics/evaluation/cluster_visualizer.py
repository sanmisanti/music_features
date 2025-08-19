"""
Visualizador de Clusters Semánticos para Letras Musicales

Sistema completo de visualización 2D/3D de clusters usando UMAP y t-SNE
con interpretación musical y análisis temático interactivo.

Características:
- UMAP y t-SNE para reducción dimensional
- Visualizaciones interactivas con Plotly
- Color-coding por clusters y idiomas
- Análisis temático visual
- Exportación alta resolución

Autor: Sistema de Clustering Musical
Fecha: Agosto 2025 - FASE 5
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP no disponible. Instalar con: pip install umap-learn")

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly no disponible. Instalar con: pip install plotly")

try:
    from ..config.clustering_params import get_clustering_config
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    
    from config.clustering_params import get_clustering_config

# Setup logging
logger = logging.getLogger(__name__)


class ClusterVisualizer:
    """
    Visualizador completo para clustering semántico de letras.
    
    Proporciona múltiples tipos de visualización para interpretación
    y análisis de clusters semánticos musicales.
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 output_dir: Optional[Path] = None):
        """
        Inicializa visualizador de clusters.
        
        Args:
            random_state: Seed para reproducibilidad
            output_dir: Directorio output (None para no guardar)
        """
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración visual
        self.config = get_clustering_config()
        self.color_palette = self._setup_color_palette()
        
        # Cache reducción dimensional
        self._reduction_cache = {}
        
        logger.info(f"🎨 ClusterVisualizer inicializado:")
        logger.info(f"   UMAP disponible: {UMAP_AVAILABLE}")
        logger.info(f"   t-SNE disponible: {TSNE_AVAILABLE}")
        logger.info(f"   Plotly disponible: {PLOTLY_AVAILABLE}")
        logger.info(f"   Output dir: {self.output_dir}")
    
    def visualize_clusters_2d(self,
                             embeddings: np.ndarray,
                             labels: np.ndarray,
                             texts: List[str] = None,
                             languages: List[str] = None,
                             method: str = 'umap',
                             title: str = "Clusters Semánticos Letras Musicales",
                             save_name: str = None) -> Optional[plt.Figure]:
        """
        Visualización 2D de clusters usando UMAP o t-SNE.
        
        Args:
            embeddings: Array embeddings BERT (N, 384)
            labels: Array labels cluster (N,)
            texts: Lista textos originales (opcional)
            languages: Lista idiomas (opcional)
            method: Método reducción ('umap', 'tsne')
            title: Título gráfico
            save_name: Nombre archivo para guardar
            
        Returns:
            Figura matplotlib o None si error
        """
        logger.info(f"🎨 Generando visualización 2D ({method})...")
        
        if method == 'umap' and not UMAP_AVAILABLE:
            logger.error("UMAP no disponible")
            return None
        
        if method == 'tsne' and not TSNE_AVAILABLE:
            logger.error("t-SNE no disponible")
            return None
        
        # Reducción dimensional
        coords_2d = self._reduce_dimensions(embeddings, method=method, n_components=2)
        
        if coords_2d is None:
            return None
        
        # Crear visualización
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Coloreado por clusters
        scatter1 = axes[0].scatter(coords_2d[:, 0], coords_2d[:, 1], 
                                  c=labels, cmap='tab10', alpha=0.7, s=50)
        axes[0].set_title(f'{title}\\nColoreado por Clusters ({method.upper()})')
        axes[0].set_xlabel(f'{method.upper()}-1')
        axes[0].set_ylabel(f'{method.upper()}-2')
        
        # Colorbar
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label('Cluster ID')
        
        # Plot 2: Coloreado por idiomas si disponible
        if languages:
            unique_langs = sorted(set(languages))
            lang_colors = {lang: i for i, lang in enumerate(unique_langs)}
            color_values = [lang_colors[lang] for lang in languages]
            
            scatter2 = axes[1].scatter(coords_2d[:, 0], coords_2d[:, 1],
                                      c=color_values, cmap='Set1', alpha=0.7, s=50)
            axes[1].set_title(f'{title}\\nColoreado por Idiomas')
            axes[1].set_xlabel(f'{method.upper()}-1')
            axes[1].set_ylabel(f'{method.upper()}-2')
            
            # Legend idiomas
            for i, lang in enumerate(unique_langs):
                axes[1].scatter([], [], c=plt.cm.Set1(i), label=lang, s=100)
            axes[1].legend(title="Idiomas", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Plot duplicado sin idiomas
            scatter2 = axes[1].scatter(coords_2d[:, 0], coords_2d[:, 1],
                                      c=labels, cmap='tab10', alpha=0.7, s=50)
            axes[1].set_title(f'{title}\\nVista Alternativa')
            axes[1].set_xlabel(f'{method.upper()}-1')
            axes[1].set_ylabel(f'{method.upper()}-2')
        
        plt.tight_layout()
        
        # Guardar si se especifica
        if save_name and self.output_dir:
            save_path = self.output_dir / f"{save_name}_2d_{method}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Visualización guardada: {save_path}")
        
        return fig
    
    def visualize_clusters_interactive(self,
                                     embeddings: np.ndarray,
                                     labels: np.ndarray,
                                     texts: List[str] = None,
                                     languages: List[str] = None,
                                     method: str = 'umap',
                                     title: str = "Clusters Interactivos",
                                     save_name: str = None) -> Optional[Any]:
        """
        Visualización interactiva usando Plotly.
        
        Args:
            embeddings: Array embeddings BERT
            labels: Array labels cluster
            texts: Textos para hover info
            languages: Idiomas para color coding
            method: Método reducción dimensional
            title: Título gráfico
            save_name: Nombre archivo HTML
            
        Returns:
            Figura Plotly o None si error
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly no disponible para visualización interactiva")
            return None
        
        logger.info(f"🎨 Generando visualización interactiva ({method})...")
        
        # Reducción dimensional
        coords_2d = self._reduce_dimensions(embeddings, method=method, n_components=2)
        
        if coords_2d is None:
            return None
        
        # Preparar datos para Plotly
        hover_texts = []
        for i in range(len(coords_2d)):
            hover_info = f"Cluster: {labels[i]}<br>"
            
            if languages:
                hover_info += f"Idioma: {languages[i]}<br>"
            
            if texts:
                text_preview = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
                hover_info += f"Texto: {text_preview}"
            
            hover_texts.append(hover_info)
        
        # Crear figura interactiva
        fig = px.scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            color=labels,
            hover_name=[f"Sample {i}" for i in range(len(coords_2d))],
            hover_data={"Hover_Info": hover_texts},
            title=f"{title} ({method.upper()})",
            labels={
                "x": f"{method.upper()}-1",
                "y": f"{method.upper()}-2",
                "color": "Cluster ID"
            },
            color_continuous_scale="Viridis"
        )
        
        # Actualizar layout
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" + 
                         f"{method.upper()}-1: %{{x:.2f}}<br>" +
                         f"{method.upper()}-2: %{{y:.2f}}<br>" +
                         "%{customdata[0]}<extra></extra>",
            customdata=[[info] for info in hover_texts]
        )
        
        fig.update_layout(
            width=900,
            height=600,
            hovermode='closest'
        )
        
        # Guardar HTML si se especifica
        if save_name and self.output_dir:
            save_path = self.output_dir / f"{save_name}_interactive_{method}.html"
            fig.write_html(save_path)
            logger.info(f"🌐 Visualización interactiva guardada: {save_path}")
        
        return fig
    
    def visualize_cluster_comparison(self,
                                   embeddings: np.ndarray,
                                   labels1: np.ndarray,
                                   labels2: np.ndarray,
                                   method1_name: str = "Método 1",
                                   method2_name: str = "Método 2",
                                   method: str = 'umap',
                                   save_name: str = None) -> Optional[plt.Figure]:
        """
        Compara visualmente dos métodos de clustering.
        
        Args:
            embeddings: Array embeddings
            labels1: Labels primer método
            labels2: Labels segundo método
            method1_name: Nombre primer método
            method2_name: Nombre segundo método
            method: Método reducción dimensional
            save_name: Nombre archivo
            
        Returns:
            Figura matplotlib
        """
        logger.info(f"🎨 Generando comparación clustering ({method})...")
        
        # Reducción dimensional compartida
        coords_2d = self._reduce_dimensions(embeddings, method=method, n_components=2)
        
        if coords_2d is None:
            return None
        
        # Crear subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot método 1
        scatter1 = axes[0].scatter(coords_2d[:, 0], coords_2d[:, 1],
                                  c=labels1, cmap='tab10', alpha=0.7, s=50)
        axes[0].set_title(f'{method1_name}\\n{len(set(labels1))} clusters')
        axes[0].set_xlabel(f'{method.upper()}-1')
        axes[0].set_ylabel(f'{method.upper()}-2')
        
        # Plot método 2
        scatter2 = axes[1].scatter(coords_2d[:, 0], coords_2d[:, 1],
                                  c=labels2, cmap='tab10', alpha=0.7, s=50)
        axes[1].set_title(f'{method2_name}\\n{len(set(labels2))} clusters')
        axes[1].set_xlabel(f'{method.upper()}-1')
        axes[1].set_ylabel(f'{method.upper()}-2')
        
        plt.tight_layout()
        
        # Guardar si se especifica
        if save_name and self.output_dir:
            save_path = self.output_dir / f"{save_name}_comparison_{method}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Comparación guardada: {save_path}")
        
        return fig
    
    def plot_cluster_metrics_evolution(self,
                                     metrics_history: List[Dict[str, float]],
                                     save_name: str = None) -> plt.Figure:
        """
        Visualiza evolución de métricas durante optimización.
        
        Args:
            metrics_history: Lista de métricas por iteración/K
            save_name: Nombre archivo
            
        Returns:
            Figura matplotlib
        """
        logger.info("📈 Generando evolución métricas...")
        
        if not metrics_history:
            logger.warning("No hay historial de métricas para visualizar")
            return None
        
        # Extraer métricas
        k_values = list(range(len(metrics_history)))
        silhouette_scores = [m.get('silhouette_score', 0) for m in metrics_history]
        db_scores = [m.get('davies_bouldin_index', 0) for m in metrics_history]
        ch_scores = [m.get('calinski_harabasz_index', 0) for m in metrics_history]
        
        # Crear subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Silhouette Score
        axes[0, 0].plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Silhouette Score (mayor = mejor)')
        axes[0, 0].set_xlabel('K o Iteración')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Davies-Bouldin Index
        axes[0, 1].plot(k_values, db_scores, 'ro-', linewidth=2, markersize=6)
        axes[0, 1].set_title('Davies-Bouldin Index (menor = mejor)')
        axes[0, 1].set_xlabel('K o Iteración')
        axes[0, 1].set_ylabel('Davies-Bouldin Index')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz Index
        axes[1, 0].plot(k_values, ch_scores, 'go-', linewidth=2, markersize=6)
        axes[1, 0].set_title('Calinski-Harabasz Index (mayor = mejor)')
        axes[1, 0].set_xlabel('K o Iteración')
        axes[1, 0].set_ylabel('Calinski-Harabasz Index')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Comparación normalizada
        if len(silhouette_scores) > 1:
            # Normalizar métricas para comparación
            sil_norm = np.array(silhouette_scores) / max(silhouette_scores) if max(silhouette_scores) > 0 else np.zeros_like(silhouette_scores)
            db_norm = 1 - (np.array(db_scores) / max(db_scores)) if max(db_scores) > 0 else np.ones_like(db_scores)  # Invertir para que mayor sea mejor
            ch_norm = np.array(ch_scores) / max(ch_scores) if max(ch_scores) > 0 else np.zeros_like(ch_scores)
            
            axes[1, 1].plot(k_values, sil_norm, 'b-', label='Silhouette (norm)', linewidth=2)
            axes[1, 1].plot(k_values, db_norm, 'r-', label='Davies-Bouldin (inv+norm)', linewidth=2)
            axes[1, 1].plot(k_values, ch_norm, 'g-', label='Calinski-Harabasz (norm)', linewidth=2)
            axes[1, 1].set_title('Métricas Normalizadas (mayor = mejor)')
            axes[1, 1].set_xlabel('K o Iteración')
            axes[1, 1].set_ylabel('Score Normalizado')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar si se especifica
        if save_name and self.output_dir:
            save_path = self.output_dir / f"{save_name}_metrics_evolution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📈 Evolución métricas guardada: {save_path}")
        
        return fig
    
    def _reduce_dimensions(self,
                          embeddings: np.ndarray,
                          method: str = 'umap',
                          n_components: int = 2) -> Optional[np.ndarray]:
        """
        Reduce dimensionalidad de embeddings.
        
        Args:
            embeddings: Array embeddings alta dimensión
            method: Método reducción ('umap', 'tsne')
            n_components: Número componentes salida
            
        Returns:
            Array reducido o None si error
        """
        cache_key = f"{method}_{n_components}_{hash(embeddings.tobytes())}"
        
        # Usar cache si existe
        if cache_key in self._reduction_cache:
            logger.debug(f"   📋 Usando cache para {method}")
            return self._reduction_cache[cache_key]
        
        logger.info(f"   🔄 Reduciendo dimensionalidad con {method} ({embeddings.shape} -> {n_components}D)...")
        start_time = time.time()
        
        try:
            if method == 'umap' and UMAP_AVAILABLE:
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=self.random_state,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='cosine'
                )
                coords = reducer.fit_transform(embeddings)
                
            elif method == 'tsne' and TSNE_AVAILABLE:
                # Para t-SNE, usar perplexity adaptiva
                n_samples = len(embeddings)
                perplexity = min(30, max(5, n_samples // 4))
                
                reducer = TSNE(
                    n_components=n_components,
                    random_state=self.random_state,
                    perplexity=perplexity,
                    metric='cosine',
                    learning_rate='auto',
                    init='random'
                )
                coords = reducer.fit_transform(embeddings)
                
            else:
                logger.error(f"Método {method} no disponible o no soportado")
                return None
            
            reduction_time = time.time() - start_time
            logger.info(f"   ✅ Reducción completada en {reduction_time:.2f}s")
            
            # Guardar en cache
            self._reduction_cache[cache_key] = coords
            
            return coords
            
        except Exception as e:
            logger.error(f"Error en reducción dimensional: {e}")
            return None
    
    def _setup_color_palette(self) -> List[str]:
        """Setup paleta de colores para visualizaciones."""
        return [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
    
    def create_cluster_summary_report(self,
                                    embeddings: np.ndarray,
                                    labels: np.ndarray,
                                    texts: List[str] = None,
                                    languages: List[str] = None,
                                    evaluation_results: Dict = None,
                                    save_name: str = "cluster_report") -> Optional[plt.Figure]:
        """
        Crea reporte visual completo del clustering.
        
        Args:
            embeddings: Array embeddings
            labels: Array labels
            texts: Textos originales
            languages: Lista idiomas
            evaluation_results: Resultados evaluación
            save_name: Nombre base archivos
            
        Returns:
            Figura principal o None
        """
        logger.info("📋 Generando reporte visual completo...")
        
        # Crear múltiples visualizaciones
        figures = []
        
        # 1. Visualización 2D con UMAP
        if UMAP_AVAILABLE:
            fig_umap = self.visualize_clusters_2d(
                embeddings, labels, texts, languages,
                method='umap', title="Clusters UMAP",
                save_name=f"{save_name}_umap"
            )
            if fig_umap:
                figures.append(("UMAP", fig_umap))
        
        # 2. Visualización 2D con t-SNE
        if TSNE_AVAILABLE:
            fig_tsne = self.visualize_clusters_2d(
                embeddings, labels, texts, languages,
                method='tsne', title="Clusters t-SNE",
                save_name=f"{save_name}_tsne"
            )
            if fig_tsne:
                figures.append(("t-SNE", fig_tsne))
        
        # 3. Distribución clusters
        fig_dist = self._plot_cluster_distribution(labels, save_name)
        if fig_dist:
            figures.append(("Distribución", fig_dist))
        
        # 4. Métricas si disponibles
        if evaluation_results and "standard_metrics" in evaluation_results:
            fig_metrics = self._plot_metrics_summary(evaluation_results, save_name)
            if fig_metrics:
                figures.append(("Métricas", fig_metrics))
        
        logger.info(f"📊 Reporte generado: {len(figures)} visualizaciones")
        
        return figures[0][1] if figures else None
    
    def _plot_cluster_distribution(self, labels: np.ndarray, save_name: str) -> Optional[plt.Figure]:
        """Plot distribución tamaños clusters."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Barplot distribución
        ax1.bar(unique_labels, counts, color=self.color_palette[:len(unique_labels)])
        ax1.set_title('Distribución Tamaños Clusters')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Número Samples')
        ax1.grid(True, alpha=0.3)
        
        # Pie chart porcentajes
        ax2.pie(counts, labels=[f'Cluster {i}' for i in unique_labels], 
               autopct='%1.1f%%', colors=self.color_palette[:len(unique_labels)])
        ax2.set_title('Proporción Clusters')
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            save_path = self.output_dir / f"{save_name}_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_metrics_summary(self, evaluation_results: Dict, save_name: str) -> Optional[plt.Figure]:
        """Plot resumen métricas evaluación."""
        metrics = evaluation_results.get("standard_metrics", {})
        
        if not metrics:
            return None
        
        # Preparar datos
        metric_names = []
        metric_values = []
        metric_colors = []
        
        # Silhouette Score (0-1, mayor mejor)
        if "silhouette_score" in metrics:
            metric_names.append("Silhouette\\nScore")
            metric_values.append(metrics["silhouette_score"])
            metric_colors.append('#2ca02c' if metrics["silhouette_score"] > 0.5 else 
                               '#ff7f0e' if metrics["silhouette_score"] > 0.2 else '#d62728')
        
        # Davies-Bouldin normalizado (invertido para que mayor sea mejor)
        if "davies_bouldin_index" in metrics:
            db_score = metrics["davies_bouldin_index"]
            db_normalized = max(0, 1 - (db_score / 5))  # Normalizar asumiendo max ~5
            metric_names.append("Davies-Bouldin\\n(inv+norm)")
            metric_values.append(db_normalized)
            metric_colors.append('#2ca02c' if db_normalized > 0.7 else 
                               '#ff7f0e' if db_normalized > 0.4 else '#d62728')
        
        # Calinski-Harabasz normalizado
        if "calinski_harabasz_index" in metrics:
            ch_score = metrics["calinski_harabasz_index"]
            ch_normalized = min(1, ch_score / 1000)  # Normalizar asumiendo good score ~1000
            metric_names.append("Calinski-Harabasz\\n(norm)")
            metric_values.append(ch_normalized)
            metric_colors.append('#2ca02c' if ch_normalized > 0.7 else 
                               '#ff7f0e' if ch_normalized > 0.4 else '#d62728')
        
        if not metric_names:
            return None
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(metric_names, metric_values, color=metric_colors, alpha=0.8)
        ax.set_title('Resumen Métricas Clustering')
        ax.set_ylabel('Score Normalizado (0-1, mayor=mejor)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en barras
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            save_path = self.output_dir / f"{save_name}_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test básico
    print("🧪 Test ClusterVisualizer:")
    
    # Generar datos fake para test
    import numpy as np
    
    np.random.seed(42)
    n_samples = 150
    embeddings = np.random.rand(n_samples, 384)
    labels = np.random.randint(0, 5, n_samples)
    texts = [f"sample_lyric_{i}" for i in range(n_samples)]
    languages = np.random.choice(["en", "es", "de"], n_samples).tolist()
    
    # Test visualizador
    visualizer = ClusterVisualizer()
    
    # Test visualización 2D
    if UMAP_AVAILABLE:
        fig = visualizer.visualize_clusters_2d(embeddings, labels, texts, languages, method='umap')
        if fig:
            print("✅ Visualización UMAP generada")
            plt.close(fig)
    
    if TSNE_AVAILABLE:
        fig = visualizer.visualize_clusters_2d(embeddings, labels, texts, languages, method='tsne')
        if fig:
            print("✅ Visualización t-SNE generada")
            plt.close(fig)
    
    # Test interactivo
    if PLOTLY_AVAILABLE:
        fig_interactive = visualizer.visualize_clusters_interactive(embeddings, labels, texts, languages)
        if fig_interactive:
            print("✅ Visualización interactiva generada")
    
    print(f"✅ Test completado")