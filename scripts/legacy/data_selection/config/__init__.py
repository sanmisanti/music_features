"""
⚙️ SELECTION CONFIGURATION SUBMODULE
====================================
Configuration settings for data selection processes.

Contains configuration classes and settings for selection pipelines,
sampling strategies, and quality thresholds.
"""

from .selection_config import SelectionConfig, get_selection_config

__all__ = [
    "SelectionConfig",
    "get_selection_config"
]