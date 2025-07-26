#!/usr/bin/env python3
"""
🔍 SELECTION VALIDATOR
=====================
Validates the quality and representativeness of selected song subsets.

This script provides comprehensive validation of song selection quality:
1. Statistical distribution comparison
2. Feature space coverage analysis
3. Diversity and representativeness metrics
4. Detailed quality reports and visualizations

Usage:
    python scripts/selection_validator.py --original-path PATH --selected-path PATH [--output-dir DIR]
"""

import sys
import os
import argparse
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exploratory_analysis.config.analysis_config import get_config
from exploratory_analysis.config.features_config import CLUSTERING_FEATURES, FEATURE_METADATA
from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.statistical_analysis.descriptive_stats import DescriptiveStats
from exploratory_analysis.visualization.distribution_plots import DistributionPlotter
from exploratory_analysis.utils.file_utils import format_file_size, get_memory_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/selection_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Represents validation results for a specific test."""
    test_name: str
    passed: bool
    score: float  # 0-100
    details: Dict[str, Any]
    message: str

class SelectionValidator:
    """
    Advanced validator for assessing the quality of song selection.
    
    Provides comprehensive statistical and visual validation to ensure
    the selected subset maintains the properties of the original dataset.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the selection validator.
        
        Args:
            output_dir: Directory for output files
        """
        self.config = get_config()
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.config.get_output_path('reports')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.stats_analyzer = DescriptiveStats()
        self.dist_plotter = DistributionPlotter()
        
        # Validation results storage
        self.validation_results = []
        self.original_data = None
        self.selected_data = None
        self.comparison_data = {}
        
        logger.info(f"🔍 Selection Validator initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def load_datasets(self, original_path: str, selected_path: str) -> bool:
        """
        Load both original and selected datasets.
        
        Args:
            original_path: Path to original dataset
            selected_path: Path to selected dataset
            
        Returns:
            True if both datasets loaded successfully
        """
        logger.info("📂 Loading datasets for validation...")
        
        try:
            # Load original dataset
            logger.info(f"Loading original dataset: {original_path}")
            original_result = self.data_loader.load_dataset(
                file_path=original_path,
                sample_size=None,  # Load what we can
                validation_level='BASIC'
            )
            
            if not original_result.success:
                logger.error(f"❌ Failed to load original dataset: {original_result.error}")
                return False
            
            self.original_data = original_result.data
            logger.info(f"✅ Original dataset loaded: {self.original_data.shape}")
            
            # Load selected dataset
            logger.info(f"Loading selected dataset: {selected_path}")
            selected_result = self.data_loader.load_dataset(
                file_path=selected_path,
                sample_size=None,
                validation_level='BASIC'
            )
            
            if not selected_result.success:
                logger.error(f"❌ Failed to load selected dataset: {selected_result.error}")
                return False
            
            self.selected_data = selected_result.data
            logger.info(f"✅ Selected dataset loaded: {self.selected_data.shape}")
            
            # Basic size validation
            if len(self.selected_data) > len(self.original_data):
                logger.warning("⚠️  Selected dataset is larger than original dataset!")
            
            # Check feature compatibility
            original_features = set(self.original_data.columns)
            selected_features = set(self.selected_data.columns)
            
            if not selected_features.issubset(original_features):
                missing_features = selected_features - original_features
                logger.warning(f"⚠️  Selected dataset has features not in original: {missing_features}")
            
            logger.info(f"📊 Validation setup complete")
            logger.info(f"   Original: {len(self.original_data):,} songs")
            logger.info(f"   Selected: {len(self.selected_data):,} songs")
            logger.info(f"   Ratio: {len(self.selected_data)/len(self.original_data):.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading datasets: {str(e)}")
            return False
    
    def validate_statistical_distributions(self) -> ValidationResult:
        """
        Validate that statistical distributions are preserved.
        
        Returns:
            ValidationResult with distribution comparison
        """
        logger.info("📊 Validating statistical distributions...")
        
        try:
            from scipy import stats
            
            feature_scores = []
            detailed_results = {}
            
            for feature in CLUSTERING_FEATURES:
                if feature in self.original_data.columns and feature in self.selected_data.columns:
                    orig_values = self.original_data[feature].dropna()
                    sel_values = self.selected_data[feature].dropna()
                    
                    if len(orig_values) < 10 or len(sel_values) < 10:
                        continue
                    
                    # Statistical tests
                    feature_result = {}
                    
                    # 1. Mean comparison
                    orig_mean = orig_values.mean()
                    sel_mean = sel_values.mean()
                    mean_diff_pct = abs(orig_mean - sel_mean) / orig_mean * 100 if orig_mean != 0 else 0
                    
                    # 2. Standard deviation comparison
                    orig_std = orig_values.std()
                    sel_std = sel_values.std()
                    std_ratio = sel_std / orig_std if orig_std != 0 else 1
                    
                    # 3. Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = stats.ks_2samp(orig_values, sel_values)
                    
                    # 4. Mann-Whitney U test (non-parametric)
                    mw_stat, mw_pvalue = stats.mannwhitneyu(orig_values, sel_values, alternative='two-sided')
                    
                    # Calculate feature score (0-100)
                    mean_score = max(0, 100 - mean_diff_pct)
                    std_score = max(0, 100 - abs(1 - std_ratio) * 100)
                    ks_score = max(0, 100 - ks_stat * 100)
                    mw_score = min(100, mw_pvalue * 100)  # Higher p-value = better (no significant difference)
                    
                    feature_score = (mean_score + std_score + ks_score + mw_score) / 4
                    feature_scores.append(feature_score)
                    
                    feature_result = {
                        'original_mean': float(orig_mean),
                        'selected_mean': float(sel_mean),
                        'mean_difference_pct': float(mean_diff_pct),
                        'original_std': float(orig_std),
                        'selected_std': float(sel_std),
                        'std_ratio': float(std_ratio),
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_pvalue),
                        'mw_statistic': float(mw_stat),
                        'mw_pvalue': float(mw_pvalue),
                        'score': float(feature_score)
                    }
                    
                    detailed_results[feature] = feature_result
            
            # Overall score
            overall_score = np.mean(feature_scores) if feature_scores else 0
            
            # Determine pass/fail
            passed = overall_score >= 80  # 80% threshold
            
            message = f"Distribution validation: {overall_score:.1f}/100"
            if passed:
                message += " ✅ PASSED"
            else:
                message += " ❌ FAILED"
            
            logger.info(f"📊 {message}")
            
            return ValidationResult(
                test_name="Statistical Distributions",
                passed=passed,
                score=overall_score,
                details={
                    'feature_results': detailed_results,
                    'overall_score': overall_score,
                    'features_tested': len(feature_scores),
                    'score_distribution': {
                        'mean': float(np.mean(feature_scores)) if feature_scores else 0,
                        'std': float(np.std(feature_scores)) if feature_scores else 0,
                        'min': float(np.min(feature_scores)) if feature_scores else 0,
                        'max': float(np.max(feature_scores)) if feature_scores else 0
                    }
                },
                message=message
            )
            
        except Exception as e:
            logger.error(f"❌ Error in distribution validation: {str(e)}")
            return ValidationResult(
                test_name="Statistical Distributions",
                passed=False,
                score=0,
                details={'error': str(e)},
                message=f"Distribution validation failed: {str(e)}"
            )
    
    def validate_feature_space_coverage(self) -> ValidationResult:
        """
        Validate that the selection covers the feature space adequately.
        
        Returns:
            ValidationResult with coverage analysis
        """
        logger.info("🌟 Validating feature space coverage...")
        
        try:
            coverage_scores = []
            detailed_results = {}
            
            for feature in CLUSTERING_FEATURES:
                if feature in self.original_data.columns and feature in self.selected_data.columns:
                    orig_values = self.original_data[feature].dropna()
                    sel_values = self.selected_data[feature].dropna()
                    
                    if len(orig_values) < 10 or len(sel_values) < 10:
                        continue
                    
                    # Range coverage
                    orig_min, orig_max = orig_values.min(), orig_values.max()
                    sel_min, sel_max = sel_values.min(), sel_values.max()
                    
                    range_coverage = min(1.0, (sel_max - sel_min) / (orig_max - orig_min) if orig_max != orig_min else 1.0)
                    
                    # Percentile coverage (check if selection covers various percentiles)
                    orig_percentiles = np.percentile(orig_values, [10, 25, 50, 75, 90])
                    sel_percentiles = np.percentile(sel_values, [10, 25, 50, 75, 90])
                    
                    percentile_diffs = np.abs(orig_percentiles - sel_percentiles) / (orig_max - orig_min) if orig_max != orig_min else np.zeros_like(orig_percentiles)
                    percentile_coverage = max(0, 1 - np.mean(percentile_diffs))
                    
                    # Density coverage (divide range into bins and check coverage)
                    n_bins = min(20, len(orig_values) // 100)
                    if n_bins > 1:
                        orig_hist, bin_edges = np.histogram(orig_values, bins=n_bins)
                        sel_hist, _ = np.histogram(sel_values, bins=bin_edges)
                        
                        # Check what percentage of non-empty bins are covered
                        non_empty_bins = orig_hist > 0
                        covered_bins = (sel_hist > 0) & non_empty_bins
                        density_coverage = np.sum(covered_bins) / np.sum(non_empty_bins) if np.sum(non_empty_bins) > 0 else 1.0
                    else:
                        density_coverage = 1.0
                    
                    # Combined coverage score
                    feature_coverage = (range_coverage * 0.3 + percentile_coverage * 0.4 + density_coverage * 0.3) * 100
                    coverage_scores.append(feature_coverage)
                    
                    detailed_results[feature] = {
                        'range_coverage': float(range_coverage),
                        'percentile_coverage': float(percentile_coverage),
                        'density_coverage': float(density_coverage),
                        'overall_coverage': float(feature_coverage),
                        'original_range': [float(orig_min), float(orig_max)],
                        'selected_range': [float(sel_min), float(sel_max)]
                    }
            
            # Overall coverage score
            overall_score = np.mean(coverage_scores) if coverage_scores else 0
            
            # Determine pass/fail
            passed = overall_score >= 75  # 75% threshold for coverage
            
            message = f"Coverage validation: {overall_score:.1f}/100"
            if passed:
                message += " ✅ PASSED"
            else:
                message += " ❌ FAILED"
            
            logger.info(f"🌟 {message}")
            
            return ValidationResult(
                test_name="Feature Space Coverage",
                passed=passed,
                score=overall_score,
                details={
                    'feature_results': detailed_results,
                    'overall_score': overall_score,
                    'features_tested': len(coverage_scores)
                },
                message=message
            )
            
        except Exception as e:
            logger.error(f"❌ Error in coverage validation: {str(e)}")
            return ValidationResult(
                test_name="Feature Space Coverage",
                passed=False,
                score=0,
                details={'error': str(e)},
                message=f"Coverage validation failed: {str(e)}"
            )
    
    def validate_diversity_preservation(self) -> ValidationResult:
        """
        Validate that the selection preserves diversity.
        
        Returns:
            ValidationResult with diversity analysis
        """
        logger.info("🎨 Validating diversity preservation...")
        
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics.pairwise import euclidean_distances
            
            # Prepare data
            orig_features = self.original_data[CLUSTERING_FEATURES].dropna()
            sel_features = self.selected_data[CLUSTERING_FEATURES].dropna()
            
            if len(orig_features) < 10 or len(sel_features) < 10:
                return ValidationResult(
                    test_name="Diversity Preservation",
                    passed=False,
                    score=0,
                    details={'error': 'Insufficient data for diversity analysis'},
                    message="Diversity validation failed: insufficient data"
                )
            
            # Standardize features
            scaler = StandardScaler()
            orig_scaled = scaler.fit_transform(orig_features)
            sel_scaled = scaler.transform(sel_features)
            
            # Calculate pairwise distances (sample if too large)
            max_sample = 1000
            if len(orig_scaled) > max_sample:
                orig_sample_idx = np.random.choice(len(orig_scaled), max_sample, replace=False)
                orig_sample = orig_scaled[orig_sample_idx]
            else:
                orig_sample = orig_scaled
            
            if len(sel_scaled) > max_sample:
                sel_sample_idx = np.random.choice(len(sel_scaled), max_sample, replace=False)
                sel_sample = sel_scaled[sel_sample_idx]
            else:
                sel_sample = sel_scaled
            
            # Calculate diversity metrics
            orig_distances = euclidean_distances(orig_sample)
            sel_distances = euclidean_distances(sel_sample)
            
            # Remove diagonal (distance to self = 0)
            orig_distances = orig_distances[np.triu_indices_from(orig_distances, k=1)]
            sel_distances = sel_distances[np.triu_indices_from(sel_distances, k=1)]
            
            # Diversity metrics
            orig_mean_dist = np.mean(orig_distances)
            sel_mean_dist = np.mean(sel_distances)
            
            orig_std_dist = np.std(orig_distances)
            sel_std_dist = np.std(sel_distances)
            
            # Compare diversity measures
            mean_dist_ratio = sel_mean_dist / orig_mean_dist if orig_mean_dist > 0 else 1.0
            std_dist_ratio = sel_std_dist / orig_std_dist if orig_std_dist > 0 else 1.0
            
            # Score based on how well diversity is preserved
            mean_score = max(0, 100 - abs(1 - mean_dist_ratio) * 100)
            std_score = max(0, 100 - abs(1 - std_dist_ratio) * 100)
            
            # Additional diversity metrics
            # Effective range coverage
            orig_range = np.max(orig_distances) - np.min(orig_distances)
            sel_range = np.max(sel_distances) - np.min(sel_distances)
            range_ratio = sel_range / orig_range if orig_range > 0 else 1.0
            range_score = max(0, 100 - abs(1 - range_ratio) * 100)
            
            # Overall diversity score
            diversity_score = (mean_score + std_score + range_score) / 3
            
            # Determine pass/fail
            passed = diversity_score >= 70  # 70% threshold for diversity
            
            message = f"Diversity validation: {diversity_score:.1f}/100"
            if passed:
                message += " ✅ PASSED"
            else:
                message += " ❌ FAILED"
            
            logger.info(f"🎨 {message}")
            
            return ValidationResult(
                test_name="Diversity Preservation",
                passed=passed,
                score=diversity_score,
                details={
                    'original_mean_distance': float(orig_mean_dist),
                    'selected_mean_distance': float(sel_mean_dist),
                    'mean_distance_ratio': float(mean_dist_ratio),
                    'original_std_distance': float(orig_std_dist),
                    'selected_std_distance': float(sel_std_dist),
                    'std_distance_ratio': float(std_dist_ratio),
                    'original_distance_range': float(orig_range),
                    'selected_distance_range': float(sel_range),
                    'range_ratio': float(range_ratio),
                    'component_scores': {
                        'mean_score': float(mean_score),
                        'std_score': float(std_score),
                        'range_score': float(range_score)
                    }
                },
                message=message
            )
            
        except Exception as e:
            logger.error(f"❌ Error in diversity validation: {str(e)}")
            return ValidationResult(
                test_name="Diversity Preservation",
                passed=False,
                score=0,
                details={'error': str(e)},
                message=f"Diversity validation failed: {str(e)}"
            )
    
    def validate_correlation_preservation(self) -> ValidationResult:
        """
        Validate that feature correlations are preserved.
        
        Returns:
            ValidationResult with correlation analysis
        """
        logger.info("🔗 Validating correlation preservation...")
        
        try:
            # Calculate correlation matrices
            orig_corr = self.original_data[CLUSTERING_FEATURES].corr()
            sel_corr = self.selected_data[CLUSTERING_FEATURES].corr()
            
            # Extract upper triangular parts (excluding diagonal)
            mask = np.triu(np.ones_like(orig_corr, dtype=bool), k=1)
            orig_corr_values = orig_corr.where(mask).stack().values
            sel_corr_values = sel_corr.where(mask).stack().values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(orig_corr_values) | np.isnan(sel_corr_values))
            orig_corr_clean = orig_corr_values[valid_mask]
            sel_corr_clean = sel_corr_values[valid_mask]
            
            if len(orig_corr_clean) == 0:
                return ValidationResult(
                    test_name="Correlation Preservation",
                    passed=False,
                    score=0,
                    details={'error': 'No valid correlations to compare'},
                    message="Correlation validation failed: no valid correlations"
                )
            
            # Calculate correlation preservation metrics
            from scipy.stats import pearsonr, spearmanr
            
            # Correlation between correlation matrices
            corr_correlation, corr_p_value = pearsonr(orig_corr_clean, sel_corr_clean)
            rank_correlation, rank_p_value = spearmanr(orig_corr_clean, sel_corr_clean)
            
            # Mean absolute difference
            mean_abs_diff = np.mean(np.abs(orig_corr_clean - sel_corr_clean))
            
            # Root mean square error
            rmse = np.sqrt(np.mean((orig_corr_clean - sel_corr_clean)**2))
            
            # Score calculation
            corr_score = max(0, corr_correlation * 100) if not np.isnan(corr_correlation) else 0
            rank_score = max(0, rank_correlation * 100) if not np.isnan(rank_correlation) else 0
            diff_score = max(0, 100 - mean_abs_diff * 100)
            rmse_score = max(0, 100 - rmse * 100)
            
            # Overall correlation preservation score
            correlation_score = (corr_score + rank_score + diff_score + rmse_score) / 4
            
            # Identify significant correlation changes
            significant_changes = []
            corr_diff = np.abs(orig_corr_clean - sel_corr_clean)
            large_changes_mask = corr_diff > 0.2  # Threshold for significant change
            
            if np.any(large_changes_mask):
                # Find feature pairs with large changes
                feature_pairs = [(orig_corr.index[i], orig_corr.columns[j]) 
                               for i in range(len(orig_corr.index)) 
                               for j in range(i+1, len(orig_corr.columns))]
                
                for idx, (feat1, feat2) in enumerate(feature_pairs):
                    if large_changes_mask[idx]:
                        significant_changes.append({
                            'feature_pair': [feat1, feat2],
                            'original_correlation': float(orig_corr_clean[idx]),
                            'selected_correlation': float(sel_corr_clean[idx]),
                            'difference': float(corr_diff[idx])
                        })
            
            # Determine pass/fail
            passed = correlation_score >= 80  # 80% threshold
            
            message = f"Correlation validation: {correlation_score:.1f}/100"
            if passed:
                message += " ✅ PASSED"
            else:
                message += " ❌ FAILED"
            
            logger.info(f"🔗 {message}")
            
            return ValidationResult(
                test_name="Correlation Preservation",
                passed=passed,
                score=correlation_score,
                details={
                    'correlation_of_correlations': float(corr_correlation) if not np.isnan(corr_correlation) else 0,
                    'rank_correlation': float(rank_correlation) if not np.isnan(rank_correlation) else 0,
                    'mean_absolute_difference': float(mean_abs_diff),
                    'rmse': float(rmse),
                    'significant_changes': significant_changes,
                    'component_scores': {
                        'correlation_score': float(corr_score),
                        'rank_score': float(rank_score),
                        'difference_score': float(diff_score),
                        'rmse_score': float(rmse_score)
                    }
                },
                message=message
            )
            
        except Exception as e:
            logger.error(f"❌ Error in correlation validation: {str(e)}")
            return ValidationResult(
                test_name="Correlation Preservation",
                passed=False,
                score=0,
                details={'error': str(e)},
                message=f"Correlation validation failed: {str(e)}"
            )
    
    def generate_comparison_visualizations(self):
        """Generate visual comparisons between original and selected datasets."""
        logger.info("📊 Generating comparison visualizations...")
        
        try:
            # Set up visualization style
            plt.style.use('seaborn-v0_8')
            fig_size = (15, 10)
            
            # 1. Distribution comparison plots
            n_features = len(CLUSTERING_FEATURES)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for i, feature in enumerate(CLUSTERING_FEATURES):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                if feature in self.original_data.columns and feature in self.selected_data.columns:
                    orig_values = self.original_data[feature].dropna()
                    sel_values = self.selected_data[feature].dropna()
                    
                    # Plot histograms
                    ax.hist(orig_values, bins=30, alpha=0.7, label='Original', density=True, color='blue')
                    ax.hist(sel_values, bins=30, alpha=0.7, label='Selected', density=True, color='orange')
                    
                    ax.set_title(f'{feature.title().replace("_", " ")}')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'{feature}\nNot Available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # Hide empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Correlation heatmap comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Original correlations
            orig_corr = self.original_data[CLUSTERING_FEATURES].corr()
            sns.heatmap(orig_corr, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=ax1, cbar_kws={'label': 'Correlation'})
            ax1.set_title('Original Dataset Correlations')
            
            # Selected correlations
            sel_corr = self.selected_data[CLUSTERING_FEATURES].corr()
            sns.heatmap(sel_corr, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=ax2, cbar_kws={'label': 'Correlation'})
            ax2.set_title('Selected Dataset Correlations')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Box plot comparison
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for i, feature in enumerate(CLUSTERING_FEATURES):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                if feature in self.original_data.columns and feature in self.selected_data.columns:
                    orig_values = self.original_data[feature].dropna()
                    sel_values = self.selected_data[feature].dropna()
                    
                    # Create box plot
                    box_data = [orig_values, sel_values]
                    box_labels = ['Original', 'Selected']
                    
                    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                    bp['boxes'][0].set_facecolor('lightblue')
                    bp['boxes'][1].set_facecolor('lightcoral')
                    
                    ax.set_title(f'{feature.title().replace("_", " ")}')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'{feature}\nNot Available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # Hide empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Comparison visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {str(e)}")
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        logger.info("📄 Generating validation report...")
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"selection_validation_report_{timestamp}.md"
            
            # Calculate overall score
            valid_results = [r for r in self.validation_results if r.score > 0]
            overall_score = np.mean([r.score for r in valid_results]) if valid_results else 0
            overall_passed = overall_score >= 75
            
            # Generate report content
            report_content = f"""# 🔍 Selection Validation Report

## Executive Summary

**Validation Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}  
**Overall Score**: {overall_score:.1f}/100  
**Overall Status**: {'✅ PASSED' if overall_passed else '❌ FAILED'}

### Dataset Information
- **Original Dataset**: {len(self.original_data):,} songs
- **Selected Dataset**: {len(self.selected_data):,} songs  
- **Selection Ratio**: {len(self.selected_data)/len(self.original_data):.4f}

## Validation Results

"""
            
            # Add individual test results
            for result in self.validation_results:
                status_icon = "✅" if result.passed else "❌"
                report_content += f"""### {status_icon} {result.test_name}

**Score**: {result.score:.1f}/100  
**Status**: {'PASSED' if result.passed else 'FAILED'}  
**Message**: {result.message}

"""
                
                # Add key details
                if 'overall_score' in result.details:
                    report_content += f"- Overall Score: {result.details['overall_score']:.1f}\n"
                
                if 'features_tested' in result.details:
                    report_content += f"- Features Tested: {result.details['features_tested']}\n"
                
                if 'significant_changes' in result.details and result.details['significant_changes']:
                    report_content += f"- Significant Changes: {len(result.details['significant_changes'])}\n"
                
                report_content += "\n"
            
            # Summary and recommendations
            report_content += f"""## Summary and Recommendations

### Quality Assessment
The selected dataset achieved an overall validation score of **{overall_score:.1f}/100**.

"""
            
            if overall_passed:
                report_content += """**✅ RECOMMENDATION**: The selected dataset maintains excellent representativeness of the original dataset and is suitable for use in the final model.

### Key Strengths:
"""
                for result in self.validation_results:
                    if result.passed:
                        report_content += f"- {result.test_name}: {result.score:.1f}/100\n"
            else:
                report_content += """**⚠️ RECOMMENDATION**: The selected dataset may have some representativeness issues. Consider reviewing the selection process.

### Areas for Improvement:
"""
                for result in self.validation_results:
                    if not result.passed:
                        report_content += f"- {result.test_name}: {result.score:.1f}/100 - {result.message}\n"
            
            report_content += f"""
### Next Steps:
1. Review detailed validation results above
2. Examine comparison visualizations in the output directory
3. {'Proceed with the selected dataset for model development' if overall_passed else 'Consider refining the selection process based on failed tests'}

---
*Report generated by Selection Validator v1.0*
"""
            
            # Save report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"📄 Validation report saved: {report_path}")
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {str(e)}")
            return ""
    
    def run_comprehensive_validation(self, original_path: str, selected_path: str) -> Dict[str, Any]:
        """
        Run comprehensive validation pipeline.
        
        Args:
            original_path: Path to original dataset
            selected_path: Path to selected dataset
            
        Returns:
            Dictionary with validation summary
        """
        logger.info("🚀 Starting comprehensive validation pipeline...")
        start_time = time.time()
        
        try:
            # Load datasets
            if not self.load_datasets(original_path, selected_path):
                return {'success': False, 'error': 'Failed to load datasets'}
            
            # Run validation tests
            logger.info("🧪 Running validation tests...")
            
            # 1. Statistical distributions
            dist_result = self.validate_statistical_distributions()
            self.validation_results.append(dist_result)
            
            # 2. Feature space coverage
            coverage_result = self.validate_feature_space_coverage()
            self.validation_results.append(coverage_result)
            
            # 3. Diversity preservation
            diversity_result = self.validate_diversity_preservation()
            self.validation_results.append(diversity_result)
            
            # 4. Correlation preservation
            corr_result = self.validate_correlation_preservation()
            self.validation_results.append(corr_result)
            
            # Generate visualizations
            self.generate_comparison_visualizations()
            
            # Generate report
            report_path = self.generate_validation_report()
            
            # Calculate summary
            valid_results = [r for r in self.validation_results if r.score > 0]
            overall_score = np.mean([r.score for r in valid_results]) if valid_results else 0
            tests_passed = sum(1 for r in self.validation_results if r.passed)
            total_tests = len(self.validation_results)
            
            end_time = time.time()
            
            summary = {
                'success': True,
                'overall_score': overall_score,
                'tests_passed': tests_passed,
                'total_tests': total_tests,
                'pass_rate': tests_passed / total_tests if total_tests > 0 else 0,
                'validation_time': end_time - start_time,
                'report_path': report_path,
                'original_size': len(self.original_data),
                'selected_size': len(self.selected_data),
                'selection_ratio': len(self.selected_data) / len(self.original_data)
            }
            
            logger.info("🎉 Comprehensive validation completed!")
            logger.info(f"📊 Overall Score: {overall_score:.1f}/100")
            logger.info(f"✅ Tests Passed: {tests_passed}/{total_tests}")
            logger.info(f"⏱️  Validation Time: {end_time - start_time:.2f}s")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Validation pipeline failed: {str(e)}")
            return {'success': False, 'error': str(e)}

def main():
    """Main entry point for the selection validator."""
    parser = argparse.ArgumentParser(description="Validate quality of song selection")
    parser.add_argument('--original-path', type=str, required=True, help='Path to original dataset')
    parser.add_argument('--selected-path', type=str, required=True, help='Path to selected dataset')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = SelectionValidator(output_dir=args.output_dir)
    
    # Run validation
    results = validator.run_comprehensive_validation(args.original_path, args.selected_path)
    
    if results.get('success', False):
        print(f"\n🎉 Selection validation completed successfully!")
        print(f"📊 Overall Score: {results['overall_score']:.1f}/100")
        print(f"✅ Tests Passed: {results['tests_passed']}/{results['total_tests']}")
        print(f"📁 Check results in: {validator.output_dir}")
        
        # Exit code based on validation quality
        if results['overall_score'] >= 75:
            sys.exit(0)
        else:
            print("⚠️  Some validation tests failed - review results")
            sys.exit(1)
    else:
        print(f"\n❌ Selection validation failed!")
        print(f"Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()