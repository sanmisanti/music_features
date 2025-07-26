"""
Correlation Analysis Module

Creates comprehensive correlation visualizations including heatmaps,
clustered heatmaps, and interactive correlation plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
import warnings
import logging
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from ..config.features_config import (
    CLUSTERING_FEATURES, FEATURE_DEFINITIONS, FEATURE_DISPLAY_NAMES,
    get_features_by_type, FeatureType
)
from ..config.analysis_config import config
from ..utils.plot_styles import PlotStyles

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class CorrelationPlotter:
    """
    Comprehensive correlation analysis and visualization
    """
    
    def __init__(self, style_config: Optional[Dict] = None):
        """
        Initialize correlation plotter
        
        Args:
            style_config: Optional style configuration
        """
        self.config = config
        self.style_helper = PlotStyles()
        
        # Set default style for correlation plots
        self._setup_plotting_style()
        
        if style_config:
            self._update_style_config(style_config)
    
    def _setup_plotting_style(self):
        """Setup matplotlib and seaborn plotting style"""
        sns.set_style("whitegrid")
        
        # Correlation-specific settings
        plt.rcParams.update({
            'figure.figsize': (12, 10),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10
        })
    
    def _update_style_config(self, style_config: Dict):
        """Update plotting style with custom configuration"""
        if 'figsize' in style_config:
            plt.rcParams['figure.figsize'] = style_config['figsize']
    
    def create_correlation_heatmap(self, df: pd.DataFrame,
                                   features: Optional[List[str]] = None,
                                   method: str = 'pearson',
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create correlation heatmap
        
        Args:
            df: DataFrame with music features
            features: Features to analyze (default: all clustering features)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            save_path: Optional path to save plot
            
        Returns:
            Figure object with heatmap
        """
        if df.empty:
            logger.error("Cannot create heatmap for empty dataset")
            return None
        
        # Set defaults
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        
        # Filter to numeric features only
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            logger.error("Need at least 2 numeric features for correlation analysis")
            return None
        
        logger.info(f"Creating {method} correlation heatmap for {len(numeric_features)} features")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr(method=method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(8, len(numeric_features)), max(8, len(numeric_features))))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        
        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={"shrink": .8},
            ax=ax
        )
        
        # Customize labels
        display_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in numeric_features]
        ax.set_xticklabels(display_labels, rotation=45, ha='right')
        ax.set_yticklabels(display_labels, rotation=0)
        
        ax.set_title(f'{method.title()} Correlation Matrix\n({len(numeric_features)} Music Features)', 
                    fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_clustered_heatmap(self, df: pd.DataFrame,
                                features: Optional[List[str]] = None,
                                method: str = 'pearson',
                                linkage_method: str = 'average',
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create clustered correlation heatmap with dendrogram
        
        Args:
            df: DataFrame with music features
            features: Features to analyze
            method: Correlation method ('pearson', 'spearman', 'kendall')
            linkage_method: Hierarchical clustering method
            save_path: Optional path to save plot
            
        Returns:
            Figure object with clustered heatmap
        """
        if df.empty:
            logger.error("Cannot create clustered heatmap for empty dataset")
            return None
        
        # Set defaults
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        
        # Filter to numeric features only
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 3:
            logger.error("Need at least 3 numeric features for clustered correlation analysis")
            return None
        
        logger.info(f"Creating clustered {method} correlation heatmap")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr(method=method)
        
        # Create clustermap
        g = sns.clustermap(
            corr_matrix,
            method=linkage_method,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.3f',
            figsize=(max(10, len(numeric_features)), max(10, len(numeric_features))),
            cbar_pos=(0.02, 0.83, 0.03, 0.15)
        )
        
        # Customize labels
        display_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in numeric_features]
        
        # Get reordered indices from clustering
        reordered_labels = [display_labels[i] for i in g.dendrogram_row.reordered_ind]
        
        g.ax_heatmap.set_xticklabels(reordered_labels, rotation=45, ha='right')
        g.ax_heatmap.set_yticklabels(reordered_labels, rotation=0)
        
        g.fig.suptitle(f'Clustered {method.title()} Correlation Matrix\n({len(numeric_features)} Music Features)', 
                       fontsize=16, y=0.98)
        
        if save_path:
            self._save_plot(g.fig, save_path)
        
        return g.fig
    
    def create_correlation_comparison(self, df: pd.DataFrame,
                                     features: Optional[List[str]] = None,
                                     methods: List[str] = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare different correlation methods side by side
        
        Args:
            df: DataFrame with music features
            features: Features to analyze
            methods: Correlation methods to compare
            save_path: Optional path to save plot
            
        Returns:
            Figure object with comparison plots
        """
        if methods is None:
            methods = ['pearson', 'spearman', 'kendall']
        
        if df.empty:
            logger.error("Cannot create correlation comparison for empty dataset")
            return None
        
        # Set defaults
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        
        # Filter to numeric features only
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            logger.error("Need at least 2 numeric features for correlation analysis")
            return None
        
        logger.info(f"Creating correlation method comparison for {len(methods)} methods")
        
        # Create subplots
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 8))
        
        if n_methods == 1:
            axes = [axes]
        
        for i, method in enumerate(methods):
            # Calculate correlation matrix
            corr_matrix = df[numeric_features].corr(method=method)
            
            # Create heatmap
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar=i == n_methods-1,  # Only show colorbar on last plot
                ax=axes[i]
            )
            
            # Customize labels
            display_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in numeric_features]
            axes[i].set_xticklabels(display_labels, rotation=45, ha='right')
            axes[i].set_yticklabels(display_labels, rotation=0)
            axes[i].set_title(f'{method.title()} Correlation', fontsize=14)
        
        fig.suptitle(f'Correlation Methods Comparison\n({len(numeric_features)} Music Features)', 
                     fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_feature_type_correlation(self, df: pd.DataFrame,
                                       method: str = 'pearson',
                                       save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create correlation heatmaps grouped by feature type
        
        Args:
            df: DataFrame with music features
            method: Correlation method
            save_path: Optional path to save plots
            
        Returns:
            Dictionary with figure objects by feature type
        """
        results = {}
        
        for feature_type in FeatureType:
            type_features = get_features_by_type(feature_type)
            available_features = [f for f in type_features if f in df.columns]
            
            # Filter to numeric features only
            numeric_features = [f for f in available_features if pd.api.types.is_numeric_dtype(df[f])]
            
            if len(numeric_features) < 2:
                continue
            
            logger.info(f"Creating correlation heatmap for {feature_type.value} features")
            
            try:
                # Calculate correlation matrix
                corr_matrix = df[numeric_features].corr(method=method)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(max(6, len(numeric_features)), max(6, len(numeric_features))))
                
                # Create heatmap
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    fmt='.3f',
                    cbar_kws={"shrink": .8},
                    ax=ax
                )
                
                # Customize labels
                display_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in numeric_features]
                ax.set_xticklabels(display_labels, rotation=45, ha='right')
                ax.set_yticklabels(display_labels, rotation=0)
                
                ax.set_title(f'{feature_type.value.title()} Features\n{method.title()} Correlation Matrix', 
                           fontsize=14, pad=20)
                
                plt.tight_layout()
                
                results[feature_type.value] = fig
                
                if save_path:
                    self._save_plot(fig, f"{save_path}_{feature_type.value}_correlation.png")
                    
            except Exception as e:
                logger.warning(f"Failed to create correlation heatmap for {feature_type.value} features: {str(e)}")
        
        return results
    
    def analyze_correlation_strength(self, df: pd.DataFrame,
                                    features: Optional[List[str]] = None,
                                    method: str = 'pearson',
                                    threshold: float = 0.7) -> Dict[str, Any]:
        """
        Analyze correlation strength and identify high correlations
        
        Args:
            df: DataFrame with music features
            features: Features to analyze
            method: Correlation method
            threshold: Threshold for high correlation
            
        Returns:
            Dictionary with correlation analysis results
        """
        if df.empty:
            logger.error("Cannot analyze correlation for empty dataset")
            return {}
        
        # Set defaults
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        
        # Filter to numeric features only
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            logger.error("Need at least 2 numeric features for correlation analysis")
            return {}
        
        logger.info(f"Analyzing correlation strength with threshold {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr(method=method)
        
        # Find high correlations
        high_correlations = []
        moderate_correlations = []
        weak_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if not np.isnan(corr_value):
                    abs_corr = abs(corr_value)
                    corr_info = {
                        'feature1': feature1,
                        'feature2': feature2,
                        'correlation': float(corr_value),
                        'abs_correlation': float(abs_corr)
                    }
                    
                    if abs_corr >= threshold:
                        high_correlations.append(corr_info)
                    elif abs_corr >= 0.3:
                        moderate_correlations.append(corr_info)
                    else:
                        weak_correlations.append(corr_info)
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        moderate_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        return {
            'method': method,
            'threshold': threshold,
            'total_pairs': len(high_correlations) + len(moderate_correlations) + len(weak_correlations),
            'high_correlations': high_correlations,
            'moderate_correlations': moderate_correlations,
            'weak_correlations': weak_correlations,
            'correlation_matrix': corr_matrix,
            'summary': {
                'high_count': len(high_correlations),
                'moderate_count': len(moderate_correlations),
                'weak_count': len(weak_correlations),
                'max_correlation': max([c['abs_correlation'] for c in high_correlations + moderate_correlations + weak_correlations]),
                'avg_correlation': np.mean([c['abs_correlation'] for c in high_correlations + moderate_correlations + weak_correlations])
            }
        }
    
    def create_correlation_network(self, df: pd.DataFrame,
                                  features: Optional[List[str]] = None,
                                  method: str = 'pearson',
                                  threshold: float = 0.5,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create network plot showing feature correlations above threshold
        
        Args:
            df: DataFrame with music features
            features: Features to analyze
            method: Correlation method
            threshold: Minimum correlation to show connection
            save_path: Optional path to save plot
            
        Returns:
            Figure object with network plot
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning("NetworkX not available, skipping network plot")
            return None
        
        if df.empty:
            logger.error("Cannot create correlation network for empty dataset")
            return None
        
        # Set defaults
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        
        # Filter to numeric features only
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 3:
            logger.error("Need at least 3 numeric features for network analysis")
            return None
        
        logger.info(f"Creating correlation network with threshold {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr(method=method)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (features)
        for feature in numeric_features:
            G.add_node(feature, label=FEATURE_DISPLAY_NAMES.get(feature, feature))
        
        # Add edges (correlations above threshold)
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if not np.isnan(corr_value) and abs(corr_value) >= threshold:
                    G.add_edge(feature1, feature2, weight=abs(corr_value), correlation=corr_value)
        
        if len(G.edges) == 0:
            logger.warning(f"No correlations above threshold {threshold} found")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Generate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        node_colors = ['lightblue' if G.degree(node) > 2 else 'lightgray' for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.7, ax=ax)
        
        # Draw edges with thickness based on correlation strength
        edges = G.edges(data=True)
        edge_weights = [d['weight'] * 5 for (u, v, d) in edges]  # Scale for visibility
        edge_colors = ['red' if d['correlation'] > 0 else 'blue' for (u, v, d) in edges]
        
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors, alpha=0.6, ax=ax)
        
        # Draw labels
        labels = {node: FEATURE_DISPLAY_NAMES.get(node, node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Feature Correlation Network\n(|correlation| ≥ {threshold})', fontsize=16)
        ax.axis('off')
        
        # Add legend
        red_line = plt.Line2D([0], [0], color='red', linewidth=3, label='Positive correlation')
        blue_line = plt.Line2D([0], [0], color='blue', linewidth=3, label='Negative correlation')
        ax.legend(handles=[red_line, blue_line], loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def _save_plot(self, fig: plt.Figure, filepath: str):
        """Save plot to file"""
        try:
            fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Plot saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save plot to {filepath}: {str(e)}")

# Convenience functions
def correlation_heatmap(df: pd.DataFrame, features: Optional[List[str]] = None,
                       method: str = 'pearson') -> plt.Figure:
    """Convenience function for quick correlation heatmap"""
    plotter = CorrelationPlotter()
    return plotter.create_correlation_heatmap(df, features, method)

def analyze_correlations(df: pd.DataFrame, features: Optional[List[str]] = None,
                        method: str = 'pearson', threshold: float = 0.7) -> Dict[str, Any]:
    """Convenience function for correlation analysis"""
    plotter = CorrelationPlotter()
    return plotter.analyze_correlation_strength(df, features, method, threshold)

# Backward compatibility
class CorrelationHeatmaps(CorrelationPlotter):
    """Backward compatibility alias"""
    pass