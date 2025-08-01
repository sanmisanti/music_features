"""
Exploratory Analysis Module for Music Features Dataset

This module provides comprehensive tools for exploring and analyzing musical datasets.
It includes statistical analysis, visualization, feature engineering, and reporting capabilities.
"""

__version__ = "1.0.0"
__author__ = "Music Analysis Research Team"

# Core modules imports (lazy loading to avoid import issues)
from . import config
# Other modules loaded on demand to avoid circular imports
try:
    from . import data_loading  
except ImportError:
    pass
try:
    from . import statistical_analysis
except ImportError:
    pass
try:
    from . import visualization
except ImportError:
    pass
try:
    from . import feature_analysis
except ImportError:
    pass
try:
    from . import reporting
except ImportError:
    pass
try:
    from . import utils
except ImportError:
    pass

__all__ = [
    'config',
    'data_loading',
    'statistical_analysis', 
    'visualization',
    'feature_analysis',
    'reporting',
    'utils'
]