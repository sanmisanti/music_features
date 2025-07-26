"""
Distribution Plots Module

Creates comprehensive distribution visualizations for musical features
including histograms, box plots, violin plots, and Q-Q plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
import warnings
from scipy import stats
import logging

from ..config.features_config import (
    CLUSTERING_FEATURES, FEATURE_DEFINITIONS, FEATURE_DISPLAY_NAMES,
    get_features_by_type, FeatureType
)
from ..config.analysis_config import config
from ..utils.plot_styles import PlotStyles

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class DistributionPlotter:
    """
    Comprehensive distribution plotting for music datasets
    """
    
    def __init__(self, style_config: Optional[Dict] = None):
        """
        Initialize distribution plotter
        
        Args:
            style_config: Optional style configuration
        """
        self.config = config
        self.style_helper = PlotStyles()
        
        # Set default style
        self._setup_plotting_style()
        
        if style_config:
            self._update_style_config(style_config)
    
    def _setup_plotting_style(self):
        """Setup matplotlib and seaborn plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Global figure settings
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 100,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
    
    def _update_style_config(self, style_config: Dict):
        """Update plotting style with custom configuration"""
        if 'figsize' in style_config:
            plt.rcParams['figure.figsize'] = style_config['figsize']
        if 'dpi' in style_config:
            plt.rcParams['figure.dpi'] = style_config['dpi']
    
    def plot_feature_distributions(self, df: pd.DataFrame, 
                                   features: Optional[List[str]] = None,
                                   plot_types: List[str] = None,
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive distribution plots for features
        
        Args:
            df: DataFrame with music features
            features: Features to plot (default: all clustering features)
            plot_types: Types of plots to create ['histogram', 'boxplot', 'violin', 'qq']
            save_path: Optional path to save plots
            
        Returns:
            Dictionary with plot information and figure objects
        """
        if df.empty:
            logger.error("Cannot plot distributions for empty dataset")
            return {}
        
        # Set defaults
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        
        if plot_types is None:
            plot_types = ['histogram', 'boxplot']
        
        if not features:
            logger.error("No valid features found for plotting")
            return {}
        
        logger.info(f"Creating distribution plots for {len(features)} features")
        
        plot_results = {}
        
        # Create individual plots for each type
        for plot_type in plot_types:
            try:
                if plot_type == 'histogram':
                    fig = self._create_histogram_grid(df, features)
                    plot_results['histogram'] = {'figure': fig, 'type': 'histogram'}
                    
                elif plot_type == 'boxplot':
                    fig = self._create_boxplot_grid(df, features)
                    plot_results['boxplot'] = {'figure': fig, 'type': 'boxplot'}
                    
                elif plot_type == 'violin':
                    fig = self._create_violin_plots(df, features)
                    plot_results['violin'] = {'figure': fig, 'type': 'violin'}
                    
                elif plot_type == 'qq':
                    fig = self._create_qq_plots(df, features)
                    plot_results['qq'] = {'figure': fig, 'type': 'qq'}
                
                # Save plots if requested
                if save_path and plot_type in plot_results:
                    self._save_plot(plot_results[plot_type]['figure'], 
                                  f"{save_path}_{plot_type}.png")
                
            except Exception as e:
                logger.warning(f"Failed to create {plot_type} plots: {str(e)}")
        
        return plot_results
    
    def _create_histogram_grid(self, df: pd.DataFrame, features: List[str]) -> plt.Figure:
        """Create grid of histograms for features"""
        n_features = len(features)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        fig.suptitle('Feature Distributions - Histograms', fontsize=16, y=0.98)
        
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_features == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i] if n_features > 1 else axes[0]
            
            # Get feature data
            data = df[feature].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No data\nfor {feature}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(FEATURE_DISPLAY_NAMES.get(feature, feature))
                continue
            
            # Create histogram with KDE overlay
            ax.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add KDE if enough data points
            if len(data) > 10:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                    ax.legend()
                except:
                    pass
            
            # Add statistics
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')
            
            ax.set_title(FEATURE_DISPLAY_NAMES.get(feature, feature))
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _create_boxplot_grid(self, df: pd.DataFrame, features: List[str]) -> plt.Figure:
        """Create grid of box plots for features"""
        n_features = len(features)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        fig.suptitle('Feature Distributions - Box Plots', fontsize=16, y=0.98)
        
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_features == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i] if n_features > 1 else axes[0]
            
            # Get feature data
            data = df[feature].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No data\nfor {feature}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(FEATURE_DISPLAY_NAMES.get(feature, feature))
                continue
            
            # Create box plot
            box = ax.boxplot(data, patch_artist=True)
            box['boxes'][0].set_facecolor('lightblue')
            box['boxes'][0].set_alpha(0.7)
            
            # Add mean marker
            mean_val = data.mean()
            ax.plot(1, mean_val, 'ro', markersize=8, label=f'Mean: {mean_val:.3f}')
            
            ax.set_title(FEATURE_DISPLAY_NAMES.get(feature, feature))
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _create_violin_plots(self, df: pd.DataFrame, features: List[str]) -> plt.Figure:
        """Create violin plots for features"""
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        fig.suptitle('Feature Distributions - Violin Plots', fontsize=16, y=0.98)
        
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_features == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i] if n_features > 1 else axes[0]
            
            # Get feature data
            data = df[feature].dropna()
            
            if len(data) < 10:  # Need sufficient data for violin plot
                ax.text(0.5, 0.5, f'Insufficient data\nfor {feature}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(FEATURE_DISPLAY_NAMES.get(feature, feature))
                continue
            
            # Create violin plot
            parts = ax.violinplot([data], positions=[1], showmeans=True, showmedians=True)
            
            # Customize colors
            for pc in parts['bodies']:
                pc.set_facecolor('lightcoral')
                pc.set_alpha(0.7)
            
            ax.set_title(FEATURE_DISPLAY_NAMES.get(feature, feature))
            ax.set_ylabel('Value')
            ax.set_xticks([1])
            ax.set_xticklabels([feature])
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _create_qq_plots(self, df: pd.DataFrame, features: List[str]) -> plt.Figure:
        """Create Q-Q plots to test normality"""
        n_features = len(features)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        fig.suptitle('Normality Assessment - Q-Q Plots', fontsize=16, y=0.98)
        
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_features == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i] if n_features > 1 else axes[0]
            
            # Get feature data
            data = df[feature].dropna()
            
            if len(data) < 10:
                ax.text(0.5, 0.5, f'Insufficient data\nfor {feature}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(FEATURE_DISPLAY_NAMES.get(feature, feature))
                continue
            
            # Create Q-Q plot
            try:
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(f'{FEATURE_DISPLAY_NAMES.get(feature, feature)}\nQ-Q Plot vs Normal')
                ax.grid(True, alpha=0.3)
                
                # Calculate and display R-squared
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
                sample_quantiles = np.sort(data)
                r_squared = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
                ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
            except Exception as e:
                logger.warning(f"Failed to create Q-Q plot for {feature}: {str(e)}")
                ax.text(0.5, 0.5, f'Q-Q plot failed\nfor {feature}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_by_type(self, df: pd.DataFrame, 
                            plot_type: str = 'histogram',
                            save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create distribution plots grouped by feature type
        
        Args:
            df: DataFrame with music features
            plot_type: Type of plot ('histogram', 'boxplot', 'violin')
            save_path: Optional path to save plots
            
        Returns:
            Dictionary with figure objects by feature type
        """
        results = {}
        
        for feature_type in FeatureType:
            type_features = get_features_by_type(feature_type)
            available_features = [f for f in type_features if f in df.columns]
            
            if not available_features:
                continue
            
            logger.info(f"Creating {plot_type} plot for {feature_type.value} features")
            
            try:
                if plot_type == 'histogram':
                    fig = self._create_histogram_grid(df, available_features)
                elif plot_type == 'boxplot':
                    fig = self._create_boxplot_grid(df, available_features)
                elif plot_type == 'violin':
                    fig = self._create_violin_plots(df, available_features)
                else:
                    logger.warning(f"Unknown plot type: {plot_type}")
                    continue
                
                fig.suptitle(f'{feature_type.value.title()} Features - {plot_type.title()}s', 
                           fontsize=16, y=0.98)
                
                results[feature_type.value] = fig
                
                if save_path:
                    self._save_plot(fig, f"{save_path}_{feature_type.value}_{plot_type}.png")
                    
            except Exception as e:
                logger.warning(f"Failed to create {plot_type} for {feature_type.value} features: {str(e)}")
        
        return results
    
    def create_distribution_summary(self, df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive distribution summary dashboard
        
        Args:
            df: DataFrame with music features
            save_path: Optional path to save plot
            
        Returns:
            Figure object with summary dashboard
        """
        logger.info("Creating distribution summary dashboard")
        
        features = [f for f in CLUSTERING_FEATURES if f in df.columns][:8]  # Limit to 8 features
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Music Features Distribution Summary', fontsize=20, y=0.98)
        
        # Create a 4x4 grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        for i, feature in enumerate(features):
            row = i // 2
            col = (i % 2) * 2
            
            data = df[feature].dropna()
            
            if len(data) == 0:
                continue
            
            # Histogram
            ax1 = fig.add_subplot(gs[row, col])
            ax1.hist(data, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'{FEATURE_DISPLAY_NAMES.get(feature, feature)} - Histogram')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2 = fig.add_subplot(gs[row, col + 1])
            box = ax2.boxplot(data, patch_artist=True)
            box['boxes'][0].set_facecolor('lightcoral')
            box['boxes'][0].set_alpha(0.7)
            ax2.set_title(f'{FEATURE_DISPLAY_NAMES.get(feature, feature)} - Box Plot')
            ax2.set_ylabel('Value')
            ax2.grid(True, alpha=0.3)
        
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
def plot_distributions(df: pd.DataFrame, features: Optional[List[str]] = None,
                      plot_types: List[str] = None) -> Dict[str, Any]:
    """Convenience function for quick distribution plotting"""
    plotter = DistributionPlotter()
    return plotter.plot_feature_distributions(df, features, plot_types)

def quick_histogram(df: pd.DataFrame, feature: str) -> plt.Figure:
    """Quick histogram for a single feature"""
    plotter = DistributionPlotter()
    results = plotter.plot_feature_distributions(df, [feature], ['histogram'])
    return results['histogram']['figure'] if 'histogram' in results else None

# Backward compatibility
class DistributionPlots(DistributionPlotter):
    """Backward compatibility alias"""
    pass