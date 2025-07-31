"""
🎯 DATA SELECTION MODULE
=======================
Module for data selection, sampling, and representative dataset generation.

This module contains algorithms and pipelines for selecting representative
subsets from large music datasets, including diversity sampling, stratified
sampling, and hybrid selection with lyrics verification.

Components:
- pipeline/: Selection pipelines and orchestration
- sampling/: Sampling algorithms and strategies  
- config/: Configuration for selection processes
"""

from .pipeline import HybridSelectionPipeline
from .sampling import SamplingStrategies
from .config import SelectionConfig

__version__ = "1.0.0"
__author__ = "Music Features Analysis System"

__all__ = [
    "HybridSelectionPipeline",
    "SamplingStrategies", 
    "SelectionConfig"
]