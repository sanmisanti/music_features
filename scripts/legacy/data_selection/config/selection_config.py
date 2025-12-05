"""
Selection Configuration Module

Configuration settings specifically for data selection pipelines,
sampling strategies, and quality thresholds.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class SelectionConfig:
    """Configuration for data selection processes"""
    
    # Pipeline settings
    target_size: int = 10000
    default_batch_size: int = 1000
    random_state: int = 42
    
    # Diversity sampling settings
    diversity_method: str = "maxmin"
    diversity_iterations: int = 100
    diversity_tolerance: float = 1e-6
    
    # Quality filtering settings
    quality_threshold: float = 60.0
    completeness_weight: float = 0.5
    validity_weight: float = 0.3
    popularity_weight: float = 0.2
    
    # Stratified sampling settings
    balance_method: str = "proportional"
    balance_columns: List[str] = None
    min_stratum_size: int = 5
    
    # Hybrid selection settings
    lyrics_availability_weight: float = 0.4
    musical_diversity_weight: float = 0.4
    popularity_factor_weight: float = 0.15
    genre_balance_weight: float = 0.05
    
    # Progressive constraints for lyrics
    progressive_lyrics_ratios: Dict[int, float] = None
    
    # File paths
    default_input_path: str = "data/cleaned_data/tracks_features_clean.csv"
    default_output_dir: str = "data/pipeline_results"
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    
    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.balance_columns is None:
            self.balance_columns = ["key", "mode"]
            
        if self.progressive_lyrics_ratios is None:
            self.progressive_lyrics_ratios = {
                1: 0.70,  # Stage 1: 70% with lyrics
                2: 0.75,  # Stage 2: 75% with lyrics  
                3: 0.78,  # Stage 3: 78% with lyrics
                4: 0.80   # Stage 4: 80% with lyrics (final)
            }

    @classmethod
    def for_large_dataset(cls) -> 'SelectionConfig':
        """Configuration optimized for large datasets (>100K songs)"""
        return cls(
            target_size=10000,
            default_batch_size=5000,
            diversity_iterations=50,  # Reduced for performance
            quality_threshold=70.0,   # Higher threshold for large datasets
        )
    
    @classmethod  
    def for_small_dataset(cls) -> 'SelectionConfig':
        """Configuration optimized for small datasets (<10K songs)"""
        return cls(
            target_size=1000,
            default_batch_size=100,
            diversity_iterations=200,  # More iterations for precision
            quality_threshold=50.0,    # Lower threshold for small datasets
        )
        
    @classmethod
    def for_lyrics_optimization(cls) -> 'SelectionConfig':
        """Configuration optimized for lyrics availability"""
        return cls(
            lyrics_availability_weight=0.6,  # Higher weight for lyrics
            musical_diversity_weight=0.3,    # Lower weight for diversity
            popularity_factor_weight=0.1,
            progressive_lyrics_ratios={
                1: 0.75,  # Start higher
                2: 0.80,
                3: 0.85,
                4: 0.90   # Target 90% lyrics coverage
            }
        )

# Global configuration instance
_config = SelectionConfig()

def get_selection_config() -> SelectionConfig:
    """Get the global selection configuration"""
    return _config

def configure_for_large_dataset():
    """Configure globally for large dataset processing"""
    global _config
    _config = SelectionConfig.for_large_dataset()
    
def configure_for_small_dataset():
    """Configure globally for small dataset processing"""
    global _config
    _config = SelectionConfig.for_small_dataset()
    
def configure_for_lyrics_optimization():
    """Configure globally for lyrics availability optimization"""
    global _config
    _config = SelectionConfig.for_lyrics_optimization()

def set_selection_config(config: SelectionConfig):
    """Set a custom global configuration"""
    global _config
    _config = config