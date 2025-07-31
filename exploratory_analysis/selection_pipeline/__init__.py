"""
Selection Pipeline Module

Advanced hybrid selection system for representative song sampling
with integrated lyrics availability verification.

Components:
- main_pipeline: Complete pipeline orchestration
- data_processor: Large dataset analysis and processing  
- representative_selector: Hybrid selection with lyrics verification
- selection_validator: Quality validation and statistical testing

Features:
- Progressive constraints for optimal lyrics ratio (70%→75%→78%→80%)
- Multi-criteria scoring (diversity + lyrics + popularity + genre)
- Robust fallback mechanisms for edge cases
- Comprehensive validation and reporting

Author: Music Features Analysis Project
Date: 2025-01-28
"""

from .main_pipeline import MainSelectionPipeline
from .data_processor import LargeDatasetProcessor  
from .representative_selector import RepresentativeSelector
from .selection_validator import SelectionValidator

__version__ = "1.0.0"
__all__ = [
    "MainSelectionPipeline",
    "LargeDatasetProcessor", 
    "RepresentativeSelector",
    "SelectionValidator"
]