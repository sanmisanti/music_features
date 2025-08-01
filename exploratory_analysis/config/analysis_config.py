"""
Analysis Configuration

Central configuration for all exploratory analysis parameters, paths, and settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

# === PATH CONFIGURATION ===
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" 
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Data file paths
DATA_PATHS = {
    'original': DATA_DIR / "original_data" / "tracks_features.csv",
    'cleaned_full': DATA_DIR / "cleaned_data" / "tracks_features_clean.csv", 
    'sample_500': DATA_DIR / "cleaned_data" / "tracks_features_500.csv",
    'lyrics_dataset': DATA_DIR / "final_data" / "picked_data_lyrics.csv",
    'clustering_results': PROJECT_ROOT / "clustering" / "clustering_results.csv"
}

# Output directories
OUTPUT_PATHS = {
    'reports': OUTPUTS_DIR / "reports",
    'plots': OUTPUTS_DIR / "plots", 
    'statistics': OUTPUTS_DIR / "statistics",
    'processed_data': OUTPUTS_DIR / "processed_data"
}

# === DATA LOADING CONFIGURATION ===
@dataclass
class DataConfig:
    """Configuration for data loading and processing"""
    
    # CSV parsing parameters
    separator: str = '^'
    decimal: str = '.'
    encoding: str = 'utf-8'
    
    # Sampling parameters
    default_sample_size: int = 10000
    random_state: int = 42
    
    # Memory management
    chunk_size: int = 10000
    low_memory: bool = True
    
    # Data validation
    validate_on_load: bool = True
    handle_bad_lines: str = 'skip'  # 'skip', 'error', 'warn'
    
    # Missing data handling
    missing_value_threshold: float = 0.1  # Drop columns with >10% missing
    imputation_strategy: str = 'median'   # 'mean', 'median', 'mode'

# === STATISTICAL ANALYSIS CONFIGURATION ===
@dataclass
class StatsConfig:
    """Configuration for statistical analysis"""
    
    # Significance levels
    alpha: float = 0.05
    confidence_level: float = 0.95
    
    # Distribution tests
    normality_test: str = 'shapiro'  # 'shapiro', 'kolmogorov', 'anderson'
    normality_sample_size: int = 5000  # Max sample for normality tests
    
    # Correlation analysis
    correlation_methods: List[str] = field(default_factory=lambda: ['pearson', 'spearman', 'kendall'])
    correlation_threshold: float = 0.7  # High correlation threshold
    
    # Outlier detection
    outlier_methods: List[str] = field(default_factory=lambda: ['zscore', 'iqr', 'isolation_forest'])
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    isolation_contamination: float = 0.1

# === VISUALIZATION CONFIGURATION ===
@dataclass
class PlotConfig:
    """Configuration for plots and visualizations"""
    
    # Figure settings
    figure_size: tuple = (12, 8)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    
    # Colors and themes
    color_palette: str = 'Set2'
    primary_color: str = '#1f77b4'
    secondary_color: str = '#ff7f0e'
    
    # Plot types configuration
    histogram_bins: int = 30
    scatter_alpha: float = 0.6
    heatmap_cmap: str = 'RdBu_r'
    
    # Interactive plots (Plotly)
    plotly_theme: str = 'plotly_white'
    show_plots: bool = True
    save_plots: bool = True
    
    # File formats
    image_format: str = 'png'  # 'png', 'svg', 'pdf'
    interactive_format: str = 'html'

# === FEATURE ANALYSIS CONFIGURATION ===
@dataclass
class FeatureConfig:
    """Configuration for feature analysis and engineering"""
    
    # Feature selection
    variance_threshold: float = 0.01  # Remove low-variance features
    correlation_threshold: float = 0.95  # Remove highly correlated features
    
    # Dimensionality reduction
    pca_n_components: float = 0.95  # Retain 95% variance
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    
    # Feature importance
    importance_methods: List[str] = field(default_factory=lambda: ['mutual_info', 'variance', 'correlation'])
    
    # Feature engineering
    create_ratios: bool = True
    create_interactions: bool = True
    polynomial_degree: int = 2

# === CLUSTERING ANALYSIS CONFIGURATION ===
@dataclass  
class ClusteringConfig:
    """Configuration for clustering readiness analysis"""
    
    # Clustering parameters
    k_range: tuple = (2, 20)
    algorithms: List[str] = field(default_factory=lambda: ['kmeans', 'gmm', 'dbscan'])
    
    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: ['silhouette', 'calinski_harabasz', 'davies_bouldin'])
    
    # Preprocessing
    scaling_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    handle_categorical: str = 'onehot'  # 'onehot', 'label', 'target'

# === REPORTING CONFIGURATION ===
@dataclass
class ReportConfig:
    """Configuration for report generation"""
    
    # Report formats
    formats: List[str] = field(default_factory=lambda: ['html', 'pdf'])
    template_dir: str = 'templates'
    
    # Content settings
    include_raw_data: bool = False
    max_examples: int = 10
    decimal_places: int = 4
    
    # Styling
    css_theme: str = 'default'
    logo_path: str = None

# === PERFORMANCE CONFIGURATION ===
@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    
    # Parallel processing
    n_jobs: int = -1  # Use all available cores
    backend: str = 'loky'  # 'loky', 'threading', 'multiprocessing'
    
    # Memory management
    max_memory_usage: str = '8GB'
    enable_caching: bool = True
    cache_dir: str = '.cache'
    
    # Progress tracking
    show_progress: bool = True
    verbose: int = 1  # 0=silent, 1=info, 2=debug

# === MAIN CONFIGURATION OBJECT ===
class AnalysisConfig:
    """Main configuration class that combines all config sections"""
    
    def __init__(self):
        self.data = DataConfig()
        self.stats = StatsConfig()
        self.plots = PlotConfig()
        self.features = FeatureConfig()
        self.clustering = ClusteringConfig()
        self.reports = ReportConfig()
        self.performance = PerformanceConfig()
        
        # Ensure output directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary output directories"""
        for path in OUTPUT_PATHS.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def get_data_path(self, dataset_type: str = 'sample_500') -> Path:
        """Get path for specific dataset type"""
        return DATA_PATHS.get(dataset_type, DATA_PATHS['sample_500'])
    
    def get_output_path(self, output_type: str = 'reports') -> Path:
        """Get path for specific output type"""
        return OUTPUT_PATHS.get(output_type, OUTPUT_PATHS['reports'])
    
    def update_config(self, **kwargs):
        """Update configuration parameters dynamically"""
        for section_name, section_config in kwargs.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'data': self.data.__dict__,
            'stats': self.stats.__dict__,
            'plots': self.plots.__dict__,
            'features': self.features.__dict__,
            'clustering': self.clustering.__dict__,
            'reports': self.reports.__dict__,
            'performance': self.performance.__dict__
        }

# Create global configuration instance
config = AnalysisConfig()

# === UTILITY FUNCTIONS ===
def get_config() -> AnalysisConfig:
    """Get the global configuration instance"""
    return config

def update_global_config(**kwargs):
    """Update global configuration"""
    config.update_config(**kwargs)

def reset_config():
    """Reset configuration to defaults"""
    global config
    config = AnalysisConfig()

# === ENVIRONMENT-SPECIFIC CONFIGURATIONS ===
def configure_for_development():
    """Configure for development environment"""
    update_global_config(
        data={'default_sample_size': 500, 'validate_on_load': True},
        plots={'show_plots': True, 'save_plots': True},
        performance={'verbose': 2, 'show_progress': True}
    )

def configure_for_production():
    """Configure for production environment"""
    update_global_config(
        data={'default_sample_size': 10000, 'validate_on_load': False},
        plots={'show_plots': False, 'save_plots': True},
        performance={'verbose': 0, 'show_progress': False}
    )

def configure_for_large_dataset():
    """Configure for large dataset processing"""
    update_global_config(
        data={'chunk_size': 50000, 'low_memory': True},
        performance={'max_memory_usage': '16GB', 'enable_caching': True},
        stats={'normality_sample_size': 10000}
    )

# Default to development configuration
configure_for_development()