"""
Exploratory Analysis Module for Music Features Dataset

This module provides comprehensive tools for exploring and analyzing musical datasets.
It includes statistical analysis, visualization, feature engineering, and reporting capabilities.
"""

__version__ = "1.0.0"
__author__ = "Music Analysis Research Team"

# Core modules imports
from . import config
from . import data_loading  
from . import statistical_analysis
from . import visualization
from . import feature_analysis
from . import reporting
from . import utils

__all__ = [
    'config',
    'data_loading',
    'statistical_analysis', 
    'visualization',
    'feature_analysis',
    'reporting',
    'utils'
]