#!/usr/bin/env python3
"""
🎯 REPRESENTATIVE SELECTOR
=========================
Intelligently selects 10,000 representative songs from the 1.2M dataset.

This script implements a multi-stage selection process:
1. Diversity-based sampling to cover the feature space
2. Stratified sampling to preserve distributions  
3. Quality-based filtering for data completeness
4. Clustering-based final selection for maximum representativity

Usage:
    python scripts/representative_selector.py [--target-size 10000] [--output-dir DIR]
"""

import sys
import os
import argparse
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exploratory_analysis.config.analysis_config import get_config, configure_for_large_dataset
from exploratory_analysis.config.features_config import CLUSTERING_FEATURES
from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.data_loading.sampling_strategies import SamplingStrategies
from exploratory_analysis.feature_analysis.dimensionality_reduction import DimensionalityReducer
from exploratory_analysis.statistical_analysis.descriptive_stats import DescriptiveStats
from exploratory_analysis.utils.file_utils import format_file_size, get_memory_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/representative_selection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SelectionStage:
    """Represents a stage in the selection process."""
    name: str
    method: str
    input_size: int
    output_size: int
    criteria: str
    success: bool = False
    execution_time: float = 0.0
    notes: str = ""

class RepresentativeSelector:
    """
    Advanced selector for choosing representative songs from large datasets.
    
    Implements sophisticated sampling strategies to ensure the selected subset
    maintains the statistical properties and diversity of the original dataset.
    """
    
    def __init__(self, target_size: int = 10000, output_dir: Optional[str] = None):
        """
        Initialize the representative selector.
        
        Args:
            target_size: Number of songs to select (default: 10000)
            output_dir: Directory for output files
        """
        # Configure for large dataset processing
        configure_for_large_dataset()
        self.config = get_config()
        self.target_size = target_size
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.config.get_output_path('processed_data')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.sampling_strategies = SamplingStrategies()
        self.dim_reducer = DimensionalityReducer()
        self.stats_analyzer = DescriptiveStats()
        
        # Selection tracking
        self.selection_stages = []
        self.original_data = None
        self.selected_data = None
        self.selection_metadata = {}
        
        logger.info(f"🎯 Representative Selector initialized")
        logger.info(f"📊 Target selection size: {target_size:,} songs")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load the complete dataset for selection.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Loaded dataset
        """
        logger.info("📂 Loading complete dataset...")
        
        try:
            # Load without sampling to get full dataset
            load_result = self.data_loader.load_dataset(
                file_path=dataset_path,
                sample_size=None,  # Load full dataset
                validation_level='BASIC'  # Quick validation for large dataset
            )
            
            if load_result.success:
                logger.info(f"✅ Dataset loaded successfully")
                logger.info(f"📊 Full dataset shape: {load_result.data.shape}")
                logger.info(f"🎯 Quality Score: {load_result.quality_score:.1f}/100")
                logger.info(f"💾 Memory Usage: {get_memory_usage():.1f} MB")
                
                self.original_data = load_result.data
                self.selection_metadata['original_size'] = len(load_result.data)
                self.selection_metadata['original_quality'] = load_result.quality_score
                
                return load_result.data
            else:
                logger.error(f"❌ Dataset loading failed: {load_result.error}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {str(e)}")
            return None
    
    def stage_1_diversity_sampling(self, data: pd.DataFrame, sample_size: int = 100000) -> pd.DataFrame:
        """
        Stage 1: Diversity-based sampling to cover the feature space.
        
        Args:
            data: Input dataset
            sample_size: Target size for this stage
            
        Returns:
            Diversely sampled subset
        """
        logger.info(f"🌟 Stage 1: Diversity sampling ({len(data):,} → {sample_size:,})")
        start_time = time.time()
        
        try:
            # Use max-min diversity sampling for good feature space coverage
            sampled_data = self.sampling_strategies.maxmin_diversity_sample(
                data=data,
                features=CLUSTERING_FEATURES,
                sample_size=sample_size,
                random_state=42
            )
            
            execution_time = time.time() - start_time
            
            if sampled_data is not None and len(sampled_data) > 0:
                logger.info(f"✅ Diversity sampling completed: {len(sampled_data):,} songs selected")
                logger.info(f"⏱️  Execution time: {execution_time:.2f}s")
                
                # Record stage
                stage = SelectionStage(
                    name="Diversity Sampling",
                    method="maxmin_diversity",
                    input_size=len(data),
                    output_size=len(sampled_data),
                    criteria="Maximum feature space coverage",
                    success=True,
                    execution_time=execution_time,
                    notes=f"Used {len(CLUSTERING_FEATURES)} features for diversity calculation"
                )
                self.selection_stages.append(stage)
                
                return sampled_data
            else:
                logger.error("❌ Diversity sampling failed")
                return data.sample(n=min(sample_size, len(data)), random_state=42)
                
        except Exception as e:
            logger.error(f"❌ Error in diversity sampling: {str(e)}")
            # Fallback to random sampling
            return data.sample(n=min(sample_size, len(data)), random_state=42)
    
    def stage_2_stratified_sampling(self, data: pd.DataFrame, sample_size: int = 50000) -> pd.DataFrame:
        """
        Stage 2: Stratified sampling to preserve feature distributions.
        
        Args:
            data: Input dataset from stage 1
            sample_size: Target size for this stage
            
        Returns:
            Stratified sample
        """
        logger.info(f"📊 Stage 2: Stratified sampling ({len(data):,} → {sample_size:,})")
        start_time = time.time()
        
        try:
            # Use balanced sampling to preserve distributions
            sampled_data = self.sampling_strategies.balanced_sample(
                data=data,
                features=CLUSTERING_FEATURES,
                sample_size=sample_size,
                random_state=42
            )
            
            execution_time = time.time() - start_time
            
            if sampled_data is not None and len(sampled_data) > 0:
                logger.info(f"✅ Stratified sampling completed: {len(sampled_data):,} songs selected")
                logger.info(f"⏱️  Execution time: {execution_time:.2f}s")
                
                # Record stage
                stage = SelectionStage(
                    name="Stratified Sampling",
                    method="balanced_sample",
                    input_size=len(data),
                    output_size=len(sampled_data),
                    criteria="Distribution preservation",
                    success=True,
                    execution_time=execution_time,
                    notes="Balanced sampling across feature distributions"
                )
                self.selection_stages.append(stage)
                
                return sampled_data
            else:
                logger.error("❌ Stratified sampling failed")
                return data.sample(n=min(sample_size, len(data)), random_state=42)
                
        except Exception as e:
            logger.error(f"❌ Error in stratified sampling: {str(e)}")
            # Fallback to random sampling
            return data.sample(n=min(sample_size, len(data)), random_state=42)
    
    def stage_3_quality_filtering(self, data: pd.DataFrame, target_size: int = 25000) -> pd.DataFrame:
        """
        Stage 3: Quality-based filtering for data completeness.
        
        Args:
            data: Input dataset from stage 2
            target_size: Target size after filtering
            
        Returns:
            Quality-filtered dataset
        """
        logger.info(f"🔍 Stage 3: Quality filtering ({len(data):,} → ~{target_size:,})")
        start_time = time.time()
        
        try:
            # Calculate quality scores for each row
            quality_scores = []
            
            for idx, row in data.iterrows():
                score = 0
                
                # Check for missing values
                missing_count = row[CLUSTERING_FEATURES].isna().sum()
                completeness_score = (len(CLUSTERING_FEATURES) - missing_count) / len(CLUSTERING_FEATURES)
                score += completeness_score * 40
                
                # Check for valid ranges
                valid_range_count = 0
                for feature in CLUSTERING_FEATURES:
                    value = row[feature]
                    if pd.notna(value):
                        # Basic range validation (adapt as needed)
                        if feature in ['danceability', 'energy', 'speechiness', 'acousticness', 
                                     'instrumentalness', 'liveness', 'valence']:
                            if 0 <= value <= 1:
                                valid_range_count += 1
                        elif feature == 'loudness':
                            if -60 <= value <= 0:
                                valid_range_count += 1
                        elif feature == 'tempo':
                            if 50 <= value <= 250:
                                valid_range_count += 1
                        elif feature in ['key', 'mode', 'time_signature']:
                            if value >= 0:
                                valid_range_count += 1
                        elif feature == 'duration_ms':
                            if value > 0:
                                valid_range_count += 1
                
                range_score = valid_range_count / len(CLUSTERING_FEATURES)
                score += range_score * 40
                
                # Diversity bonus (avoid too many similar values)
                diversity_score = min(1.0, len(set(row[CLUSTERING_FEATURES].dropna())) / len(CLUSTERING_FEATURES))
                score += diversity_score * 20
                
                quality_scores.append(score)
            
            # Add quality scores to dataframe
            data_with_quality = data.copy()
            data_with_quality['quality_score'] = quality_scores
            
            # Sort by quality and select top entries
            sorted_data = data_with_quality.sort_values('quality_score', ascending=False)
            filtered_data = sorted_data.head(target_size).drop('quality_score', axis=1)
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ Quality filtering completed: {len(filtered_data):,} songs selected")
            logger.info(f"📊 Average quality score: {np.mean(quality_scores):.1f}/100")
            logger.info(f"🎯 Top quality score: {max(quality_scores):.1f}/100")
            logger.info(f"⏱️  Execution time: {execution_time:.2f}s")
            
            # Record stage
            stage = SelectionStage(
                name="Quality Filtering",
                method="composite_quality_score",
                input_size=len(data),
                output_size=len(filtered_data),
                criteria="Completeness + validity + diversity",
                success=True,
                execution_time=execution_time,
                notes=f"Average quality: {np.mean(quality_scores):.1f}/100"
            )
            self.selection_stages.append(stage)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"❌ Error in quality filtering: {str(e)}")
            # Fallback to simple completeness filtering
            complete_data = data.dropna(subset=CLUSTERING_FEATURES)
            return complete_data.sample(n=min(target_size, len(complete_data)), random_state=42)
    
    def stage_4_clustering_based_selection(self, data: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """
        Stage 4: Clustering-based final selection for maximum representativity.
        
        Args:
            data: Input dataset from stage 3
            target_size: Final target size
            
        Returns:
            Final selected dataset
        """
        logger.info(f"🎯 Stage 4: Clustering-based selection ({len(data):,} → {target_size:,})")
        start_time = time.time()
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare clustering features
            clustering_data = data[CLUSTERING_FEATURES].copy()
            clustering_data = clustering_data.dropna()
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(clustering_data)
            
            # Determine optimal number of clusters (approximately target_size / 50)
            n_clusters = max(10, min(100, target_size // 50))
            logger.info(f"🔬 Using {n_clusters} clusters for representative selection")
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Add cluster labels to data
            data_with_clusters = clustering_data.copy()
            data_with_clusters['cluster'] = cluster_labels
            
            # Calculate cluster sizes and selection quotas
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            total_in_clusters = len(cluster_labels)
            
            # Proportional selection from each cluster
            selected_indices = []
            songs_per_cluster = target_size // n_clusters
            extra_songs = target_size % n_clusters
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) == 0:
                    continue
                
                # Base allocation plus extra songs for some clusters
                cluster_quota = songs_per_cluster
                if cluster_id < extra_songs:
                    cluster_quota += 1
                
                # Select from cluster (randomly or by distance to centroid)
                if len(cluster_indices) <= cluster_quota:
                    # Take all songs from small clusters
                    selected_indices.extend(cluster_indices)
                else:
                    # For larger clusters, select closest to centroid
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    cluster_features = scaled_features[cluster_indices]
                    
                    # Calculate distances to centroid
                    distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                    closest_indices = np.argsort(distances)[:cluster_quota]
                    selected_indices.extend(cluster_indices[closest_indices])
            
            # Get final selection
            final_selection = clustering_data.iloc[selected_indices].drop('cluster', axis=1)
            
            # Map back to original data indices
            original_indices = clustering_data.index[selected_indices]
            final_data = data.loc[original_indices]
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ Clustering-based selection completed: {len(final_data):,} songs selected")
            logger.info(f"🎯 Target achieved: {len(final_data)} / {target_size}")
            logger.info(f"🔬 Clusters used: {n_clusters}")
            logger.info(f"⏱️  Execution time: {execution_time:.2f}s")
            
            # Record stage
            stage = SelectionStage(
                name="Clustering-based Selection",
                method="kmeans_proportional",
                input_size=len(data),
                output_size=len(final_data),
                criteria="Proportional selection from feature clusters",
                success=True,
                execution_time=execution_time,
                notes=f"Used {n_clusters} clusters for representative selection"
            )
            self.selection_stages.append(stage)
            
            return final_data
            
        except Exception as e:
            logger.error(f"❌ Error in clustering-based selection: {str(e)}")
            # Fallback to random sampling
            return data.sample(n=min(target_size, len(data)), random_state=42)
    
    def validate_selection_quality(self, original_data: pd.DataFrame, selected_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the quality of the selection by comparing distributions.
        
        Args:
            original_data: Original dataset
            selected_data: Selected subset
            
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating selection quality...")
        
        try:
            from scipy import stats
            
            validation_results = {
                'sample_size_ratio': len(selected_data) / len(original_data),
                'feature_comparisons': {},
                'overall_quality': 0.0
            }
            
            quality_scores = []
            
            for feature in CLUSTERING_FEATURES:
                if feature in original_data.columns and feature in selected_data.columns:
                    orig_values = original_data[feature].dropna()
                    sel_values = selected_data[feature].dropna()
                    
                    if len(orig_values) > 0 and len(sel_values) > 0:
                        # Statistical comparison
                        mean_diff = abs(orig_values.mean() - sel_values.mean()) / orig_values.std()
                        std_ratio = sel_values.std() / orig_values.std()
                        
                        # Kolmogorov-Smirnov test for distribution similarity
                        ks_stat, ks_pvalue = stats.ks_2samp(orig_values, sel_values)
                        
                        # Quality score (0-100)
                        mean_score = max(0, 100 - mean_diff * 100)
                        std_score = max(0, 100 - abs(1 - std_ratio) * 100)
                        dist_score = max(0, 100 - ks_stat * 100)
                        
                        feature_quality = (mean_score + std_score + dist_score) / 3
                        quality_scores.append(feature_quality)
                        
                        validation_results['feature_comparisons'][feature] = {
                            'original_mean': float(orig_values.mean()),
                            'selected_mean': float(sel_values.mean()),
                            'original_std': float(orig_values.std()),
                            'selected_std': float(sel_values.std()),
                            'mean_difference': float(mean_diff),
                            'std_ratio': float(std_ratio),
                            'ks_statistic': float(ks_stat),
                            'ks_pvalue': float(ks_pvalue),
                            'quality_score': float(feature_quality)
                        }
            
            # Overall quality
            if quality_scores:
                validation_results['overall_quality'] = np.mean(quality_scores)
                validation_results['quality_distribution'] = {
                    'mean': float(np.mean(quality_scores)),
                    'std': float(np.std(quality_scores)),
                    'min': float(np.min(quality_scores)),
                    'max': float(np.max(quality_scores))
                }
            
            logger.info(f"✅ Selection validation completed")
            logger.info(f"🎯 Overall quality score: {validation_results['overall_quality']:.1f}/100")
            logger.info(f"📊 Sample ratio: {validation_results['sample_size_ratio']:.4f}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Error in validation: {str(e)}")
            return {'error': str(e)}
    
    def save_selected_dataset(self, selected_data: pd.DataFrame, filename: Optional[str] = None) -> str:
        """Save the selected dataset to file."""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"selected_songs_{self.target_size}_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        try:
            # Save with same format as original
            selected_data.to_csv(output_path, sep=';', decimal=',', index=False, encoding='utf-8')
            logger.info(f"💾 Selected dataset saved: {output_path}")
            logger.info(f"📊 Saved {len(selected_data):,} songs")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving dataset: {str(e)}")
            return ""
    
    def save_selection_report(self, validation_results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save comprehensive selection report."""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"selection_report_{self.target_size}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        try:
            import json
            
            # Compile comprehensive report
            report = {
                'metadata': {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'target_size': self.target_size,
                    'selector_version': '1.0'
                },
                'selection_process': {
                    'stages': [stage.__dict__ for stage in self.selection_stages],
                    'total_stages': len(self.selection_stages),
                    'total_execution_time': sum(stage.execution_time for stage in self.selection_stages)
                },
                'dataset_info': self.selection_metadata,
                'validation_results': validation_results
            }
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📄 Selection report saved: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving report: {str(e)}")
            return ""
    
    def select_representative_songs(self, dataset_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main pipeline for selecting representative songs.
        
        Args:
            dataset_path: Path to the complete dataset
            
        Returns:
            Tuple of (selected_data, validation_results)
        """
        logger.info("🚀 Starting representative song selection pipeline...")
        total_start_time = time.time()
        
        try:
            # Load complete dataset
            data = self.load_dataset(dataset_path)
            if data is None:
                logger.error("❌ Failed to load dataset")
                return None, {}
            
            original_size = len(data)
            current_data = data
            
            # Stage 1: Diversity sampling (1.2M → 100K)
            stage1_size = min(100000, original_size // 12)
            current_data = self.stage_1_diversity_sampling(current_data, stage1_size)
            
            # Stage 2: Stratified sampling (100K → 50K)
            stage2_size = min(50000, len(current_data) // 2)
            current_data = self.stage_2_stratified_sampling(current_data, stage2_size)
            
            # Stage 3: Quality filtering (50K → 25K)
            stage3_size = min(25000, len(current_data) // 2)
            current_data = self.stage_3_quality_filtering(current_data, stage3_size)
            
            # Stage 4: Clustering-based final selection (25K → target)
            final_data = self.stage_4_clustering_based_selection(current_data, self.target_size)
            
            # Validate selection quality
            validation_results = self.validate_selection_quality(data, final_data)
            
            # Save results
            dataset_path = self.save_selected_dataset(final_data)
            report_path = self.save_selection_report(validation_results)
            
            # Summary
            total_time = time.time() - total_start_time
            logger.info("🎉 Representative selection completed successfully!")
            logger.info(f"📊 Selection: {original_size:,} → {len(final_data):,} songs")
            logger.info(f"🎯 Quality score: {validation_results.get('overall_quality', 0):.1f}/100")
            logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
            logger.info(f"💾 Dataset saved: {dataset_path}")
            logger.info(f"📄 Report saved: {report_path}")
            
            self.selected_data = final_data
            return final_data, validation_results
            
        except Exception as e:
            logger.error(f"❌ Selection pipeline failed: {str(e)}")
            return None, {}

def main():
    """Main entry point for the representative selector."""
    parser = argparse.ArgumentParser(description="Select representative songs from large dataset")
    parser.add_argument('--target-size', type=int, default=10000, help='Number of songs to select (default: 10000)')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset file (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        # Auto-detect dataset path
        config = get_config()
        dataset_path = str(config.get_data_path('original'))
    
    # Initialize selector
    selector = RepresentativeSelector(
        target_size=args.target_size,
        output_dir=args.output_dir
    )
    
    # Select representative songs
    selected_data, validation_results = selector.select_representative_songs(dataset_path)
    
    if selected_data is not None:
        print(f"\n🎉 Representative selection completed successfully!")
        print(f"📊 Selected {len(selected_data):,} songs from original dataset")
        print(f"🎯 Quality score: {validation_results.get('overall_quality', 0):.1f}/100")
        print(f"📁 Check results in: {selector.output_dir}")
        sys.exit(0)
    else:
        print("\n❌ Representative selection failed!")
        print("📋 Check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()