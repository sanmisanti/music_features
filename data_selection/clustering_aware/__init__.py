"""
Clustering-Aware Data Selection Module

This module implements clustering-aware selection strategies that preserve
the natural clustering structure identified in datasets through Hopkins
Statistic and clustering readiness analysis.
"""

from .select_optimal_10k_from_18k import OptimalSelector

__all__ = ['OptimalSelector']