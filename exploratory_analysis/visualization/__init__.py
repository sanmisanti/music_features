"""
Visualization Module

Comprehensive data visualization for musical datasets including
distribution plots, correlation analysis, and feature relationships.
"""

from .distribution_plots import DistributionPlotter, DistributionPlots
from .correlation_heatmaps import CorrelationPlotter, CorrelationHeatmaps

__all__ = [
    'DistributionPlotter',
    'CorrelationPlotter',
    # Backward compatibility
    'DistributionPlots',
    'CorrelationHeatmaps'
]