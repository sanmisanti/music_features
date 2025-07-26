"""
Dimensionality Reduction Module

Implements various dimensionality reduction techniques for music feature analysis
including PCA, t-SNE, UMAP, and feature selection methods.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple, Any, Union
import logging
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from ..config.features_config import (
    CLUSTERING_FEATURES, FEATURE_DEFINITIONS, FEATURE_DISPLAY_NAMES,
    get_features_by_type, FeatureType
)
from ..config.analysis_config import config

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class DimensionalityReducer:
    """
    Comprehensive dimensionality reduction for music datasets
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize dimensionality reducer
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = config
        if config_override:
            self.config.update_config(**config_override)
        
        self.scaler = StandardScaler()
        self.fitted_models = {}
        self.reduction_results = {}
    
    def fit_pca(self, df: pd.DataFrame, 
                features: Optional[List[str]] = None,
                n_components: Optional[int] = None,
                variance_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Fit PCA model and analyze principal components
        
        Args:
            df: DataFrame with music features
            features: Features to use (default: all clustering features)
            n_components: Number of components (default: auto based on variance)
            variance_threshold: Cumulative variance threshold for auto selection
            
        Returns:
            Dictionary with PCA results and analysis
        """
        if df.empty:
            logger.error("Cannot perform PCA on empty dataset")
            return {}
        
        # Set defaults
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        
        # Filter to numeric features only
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            logger.error("Need at least 2 numeric features for PCA")
            return {}
        
        logger.info(f"Performing PCA on {len(numeric_features)} features")
        
        # Prepare data
        X = df[numeric_features].fillna(df[numeric_features].mean())
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of components
        if n_components is None:
            # Initial PCA to determine optimal components
            pca_full = PCA()
            pca_full.fit(X_scaled)
            
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumvar >= variance_threshold) + 1
            n_components = min(n_components, len(numeric_features), X.shape[0])
        
        # Fit final PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Store model
        self.fitted_models['pca'] = pca
        
        # Analyze components
        component_analysis = self._analyze_pca_components(pca, numeric_features)
        
        # Create results
        results = {
            'model': pca,
            'transformed_data': X_pca,
            'original_features': numeric_features,
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'total_variance_explained': np.sum(pca.explained_variance_ratio_),
            'component_analysis': component_analysis,
            'feature_importance': self._calculate_pca_feature_importance(pca, numeric_features)
        }
        
        self.reduction_results['pca'] = results
        
        logger.info(f"PCA completed: {n_components} components explain {results['total_variance_explained']:.1%} of variance")
        
        return results
    
    def fit_tsne(self, df: pd.DataFrame,
                 features: Optional[List[str]] = None,
                 n_components: int = 2,
                 perplexity: float = 30.0,
                 random_state: int = 42) -> Dict[str, Any]:
        """
        Fit t-SNE model for non-linear dimensionality reduction
        
        Args:
            df: DataFrame with music features
            features: Features to use
            n_components: Number of dimensions (typically 2 or 3)
            perplexity: t-SNE perplexity parameter
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with t-SNE results
        """
        if df.empty:
            logger.error("Cannot perform t-SNE on empty dataset")
            return {}
        
        # Set defaults
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        
        # Filter to numeric features only
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            logger.error("Need at least 2 numeric features for t-SNE")
            return {}
        
        # Check dataset size for t-SNE
        if len(df) < 4:
            logger.error("t-SNE requires at least 4 samples")
            return {}
        
        # Adjust perplexity if necessary
        max_perplexity = (len(df) - 1) / 3
        if perplexity >= max_perplexity:
            perplexity = max(5.0, max_perplexity)
            logger.warning(f"Adjusted perplexity to {perplexity} for dataset size")
        
        logger.info(f"Performing t-SNE on {len(numeric_features)} features with perplexity={perplexity}")
        
        # Prepare data
        X = df[numeric_features].fillna(df[numeric_features].mean())
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            verbose=0
        )
        
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Store model
        self.fitted_models['tsne'] = tsne
        
        # Create results
        results = {
            'model': tsne,
            'transformed_data': X_tsne,
            'original_features': numeric_features,
            'n_components': n_components,
            'perplexity': perplexity,
            'kl_divergence': tsne.kl_divergence_,
            'n_iter': getattr(tsne, 'n_iter_', 1000)  # Fallback for compatibility
        }
        
        self.reduction_results['tsne'] = results
        
        logger.info(f"t-SNE completed: KL divergence = {tsne.kl_divergence_:.4f}")
        
        return results
    
    def fit_umap(self, df: pd.DataFrame,
                 features: Optional[List[str]] = None,
                 n_components: int = 2,
                 n_neighbors: int = 15,
                 min_dist: float = 0.1,
                 random_state: int = 42) -> Dict[str, Any]:
        """
        Fit UMAP model for non-linear dimensionality reduction
        
        Args:
            df: DataFrame with music features
            features: Features to use
            n_components: Number of dimensions
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with UMAP results
        """
        try:
            import umap
        except ImportError:
            logger.warning("UMAP not available, skipping UMAP analysis")
            return {}
        
        if df.empty:
            logger.error("Cannot perform UMAP on empty dataset")
            return {}
        
        # Set defaults
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        
        # Filter to numeric features only
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            logger.error("Need at least 2 numeric features for UMAP")
            return {}
        
        # Adjust n_neighbors if necessary
        if n_neighbors >= len(df):
            n_neighbors = max(2, len(df) - 1)
            logger.warning(f"Adjusted n_neighbors to {n_neighbors} for dataset size")
        
        logger.info(f"Performing UMAP on {len(numeric_features)} features")
        
        # Prepare data
        X = df[numeric_features].fillna(df[numeric_features].mean())
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit UMAP
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        
        X_umap = umap_model.fit_transform(X_scaled)
        
        # Store model
        self.fitted_models['umap'] = umap_model
        
        # Create results
        results = {
            'model': umap_model,
            'transformed_data': X_umap,
            'original_features': numeric_features,
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist
        }
        
        self.reduction_results['umap'] = results
        
        logger.info("UMAP completed successfully")
        
        return results
    
    def perform_feature_selection(self, df: pd.DataFrame,
                                 target_column: Optional[str] = None,
                                 features: Optional[List[str]] = None,
                                 methods: List[str] = None) -> Dict[str, Any]:
        """
        Perform feature selection using multiple methods
        
        Args:
            df: DataFrame with music features
            target_column: Target column for supervised selection
            features: Features to consider
            methods: Selection methods to use
            
        Returns:
            Dictionary with feature selection results
        """
        if methods is None:
            methods = ['variance', 'mutual_info'] if target_column else ['variance']
        
        if df.empty:
            logger.error("Cannot perform feature selection on empty dataset")
            return {}
        
        # Set defaults
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        
        # Filter to numeric features only
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            logger.error("Need at least 2 numeric features for feature selection")
            return {}
        
        logger.info(f"Performing feature selection on {len(numeric_features)} features")
        
        # Prepare data
        X = df[numeric_features].fillna(df[numeric_features].mean())
        y = df[target_column] if target_column and target_column in df.columns else None
        
        selection_results = {}
        
        # Variance threshold
        if 'variance' in methods:
            selector = VarianceThreshold(threshold=0.01)
            X_var = selector.fit_transform(X)
            selected_features = [numeric_features[i] for i in range(len(numeric_features)) if selector.get_support()[i]]
            
            selection_results['variance'] = {
                'method': 'Variance Threshold',
                'selected_features': selected_features,
                'n_selected': len(selected_features),
                'selection_mask': selector.get_support(),
                'variances': selector.variances_
            }
        
        # Mutual information (requires target)
        if 'mutual_info' in methods and y is not None:
            try:
                selector = SelectKBest(score_func=mutual_info_classif, k=min(10, len(numeric_features)))
                X_mi = selector.fit_transform(X, y)
                selected_features = [numeric_features[i] for i in range(len(numeric_features)) if selector.get_support()[i]]
                
                selection_results['mutual_info'] = {
                    'method': 'Mutual Information',
                    'selected_features': selected_features,
                    'n_selected': len(selected_features),
                    'selection_mask': selector.get_support(),
                    'scores': selector.scores_
                }
            except Exception as e:
                logger.warning(f"Mutual information selection failed: {str(e)}")
        
        # Random Forest importance (requires target)
        if 'rf_importance' in methods and y is not None:
            try:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Select top features
                importance_threshold = np.mean(rf.feature_importances_)
                selected_indices = np.where(rf.feature_importances_ >= importance_threshold)[0]
                selected_features = [numeric_features[i] for i in selected_indices]
                
                selection_results['rf_importance'] = {
                    'method': 'Random Forest Importance',
                    'selected_features': selected_features,
                    'n_selected': len(selected_features),
                    'feature_importances': rf.feature_importances_,
                    'threshold': importance_threshold
                }
            except Exception as e:
                logger.warning(f"Random Forest selection failed: {str(e)}")
        
        logger.info(f"Feature selection completed using {len(selection_results)} methods")
        
        return {
            'original_features': numeric_features,
            'selection_results': selection_results,
            'summary': self._summarize_feature_selection(selection_results)
        }
    
    def _analyze_pca_components(self, pca, features: List[str]) -> Dict[str, Any]:
        """Analyze PCA components and their interpretations"""
        components = pca.components_
        n_components = components.shape[0]
        
        component_analysis = {}
        
        for i in range(n_components):
            component = components[i]
            
            # Find top contributing features (positive and negative)
            sorted_indices = np.argsort(np.abs(component))[::-1]
            top_features = []
            
            for j in sorted_indices[:5]:  # Top 5 features
                feature_name = features[j]
                contribution = component[j]
                top_features.append({
                    'feature': feature_name,
                    'display_name': FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
                    'contribution': float(contribution),
                    'abs_contribution': float(abs(contribution))
                })
            
            component_analysis[f'PC{i+1}'] = {
                'explained_variance': float(pca.explained_variance_[i]),
                'explained_variance_ratio': float(pca.explained_variance_ratio_[i]),
                'top_features': top_features,
                'interpretation': self._interpret_component(top_features)
            }
        
        return component_analysis
    
    def _interpret_component(self, top_features: List[Dict]) -> str:
        """Generate interpretation for PCA component based on top features"""
        if not top_features:
            return "No significant features"
        
        # Get dominant feature types
        feature_types = []
        for feat in top_features[:3]:  # Top 3 features
            feature_name = feat['feature']
            for ftype in FeatureType:
                if feature_name in get_features_by_type(ftype):
                    feature_types.append(ftype.value)
                    break
        
        # Generate interpretation
        dominant_type = max(set(feature_types), key=feature_types.count) if feature_types else "mixed"
        top_feature = top_features[0]['display_name']
        
        return f"Primarily {dominant_type} characteristics, led by {top_feature}"
    
    def _calculate_pca_feature_importance(self, pca, features: List[str]) -> Dict[str, float]:
        """Calculate overall feature importance from PCA components"""
        components = pca.components_
        explained_variance = pca.explained_variance_ratio_
        
        # Weight components by explained variance
        weighted_importance = np.zeros(len(features))
        
        for i in range(len(explained_variance)):
            component = components[i]
            weight = explained_variance[i]
            weighted_importance += np.abs(component) * weight
        
        # Normalize to sum to 1
        weighted_importance = weighted_importance / np.sum(weighted_importance)
        
        return {
            features[i]: float(weighted_importance[i]) 
            for i in range(len(features))
        }
    
    def _summarize_feature_selection(self, selection_results: Dict) -> Dict[str, Any]:
        """Summarize feature selection results across methods"""
        if not selection_results:
            return {}
        
        # Count how often each feature was selected
        all_features = set()
        feature_counts = {}
        
        for method_name, method_results in selection_results.items():
            selected = method_results['selected_features']
            all_features.update(selected)
            
            for feature in selected:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Rank features by selection frequency
        feature_ranking = sorted(
            feature_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            'n_methods': len(selection_results),
            'total_unique_features': len(all_features),
            'feature_selection_frequency': feature_counts,
            'top_features': feature_ranking[:10],
            'consensus_features': [f for f, c in feature_ranking if c == len(selection_results)]
        }
    
    def create_dimensionality_reduction_comparison(self, df: pd.DataFrame,
                                                  features: Optional[List[str]] = None,
                                                  methods: List[str] = None,
                                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparison visualization of dimensionality reduction methods
        
        Args:
            df: DataFrame with music features
            features: Features to use
            methods: Reduction methods to compare ['pca', 'tsne', 'umap']
            save_path: Optional path to save plot
            
        Returns:
            Figure object with comparison plots
        """
        if methods is None:
            methods = ['pca', 'tsne']
        
        if df.empty:
            logger.error("Cannot create comparison for empty dataset")
            return None
        
        logger.info(f"Creating dimensionality reduction comparison for {len(methods)} methods")
        
        # Run each method
        results = {}
        for method in methods:
            if method == 'pca':
                results[method] = self.fit_pca(df, features, n_components=2)
            elif method == 'tsne':
                results[method] = self.fit_tsne(df, features, n_components=2)
            elif method == 'umap':
                results[method] = self.fit_umap(df, features, n_components=2)
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if v}
        
        if not successful_results:
            logger.error("No dimensionality reduction methods succeeded")
            return None
        
        # Create subplots
        n_methods = len(successful_results)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method, result) in enumerate(successful_results.items()):
            ax = axes[i]
            
            # Get transformed data
            X_transformed = result['transformed_data']
            
            # Create scatter plot
            scatter = ax.scatter(
                X_transformed[:, 0], 
                X_transformed[:, 1], 
                alpha=0.6, 
                s=30,
                c=range(len(X_transformed)),
                cmap='viridis'
            )
            
            ax.set_xlabel(f'{method.upper()} Dimension 1')
            ax.set_ylabel(f'{method.upper()} Dimension 2')
            ax.set_title(f'{method.upper()} Projection')
            ax.grid(True, alpha=0.3)
            
            # Add method-specific information
            if method == 'pca':
                var_exp = result['explained_variance_ratio']
                info_text = f'Variance explained: {var_exp[0]:.1%}, {var_exp[1]:.1%}'
            elif method == 'tsne':
                info_text = f'KL divergence: {result["kl_divergence"]:.3f}'
            elif method == 'umap':
                info_text = f'n_neighbors: {result["n_neighbors"]}'
            else:
                info_text = ''
            
            if info_text:
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       verticalalignment='top', fontsize=9)
        
        fig.suptitle(f'Dimensionality Reduction Comparison\n({len(df)} samples)', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to: {save_path}")
        
        return fig

# Convenience functions
def quick_pca(df: pd.DataFrame, features: Optional[List[str]] = None, 
              n_components: Optional[int] = None) -> Dict[str, Any]:
    """Quick PCA analysis"""
    reducer = DimensionalityReducer()
    return reducer.fit_pca(df, features, n_components)

def quick_feature_selection(df: pd.DataFrame, target_column: Optional[str] = None,
                           features: Optional[List[str]] = None) -> Dict[str, Any]:
    """Quick feature selection"""
    reducer = DimensionalityReducer()
    return reducer.perform_feature_selection(df, target_column, features)