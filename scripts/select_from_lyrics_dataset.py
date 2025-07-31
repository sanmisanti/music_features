#!/usr/bin/env python3
"""
🎯 REPRESENTATIVE SELECTOR FOR LYRICS DATASET
============================================
Selects 10,000 most representative songs from the 18K spotify_songs_fixed.csv
that already contains verified lyrics.

This reuses the existing pipeline architecture but skips lyrics verification.

Usage:
    python scripts/select_from_lyrics_dataset.py [--target-size 10000]
"""

import sys
import os
import argparse
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exploratory_analysis.config.analysis_config import get_config, configure_for_large_dataset
from exploratory_analysis.data_loading.sampling_strategies import SamplingStrategies
from exploratory_analysis.statistical_analysis.descriptive_stats import DescriptiveStats
from exploratory_analysis.utils.file_utils import format_file_size, get_memory_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/lyrics_dataset_selection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Map spotify_songs columns to our standard clustering features
COLUMN_MAPPING = {
    'track_id': 'id',
    'track_name': 'name', 
    'track_artist': 'artists',
    'danceability': 'danceability',
    'energy': 'energy',
    'key': 'key',
    'loudness': 'loudness',
    'mode': 'mode',
    'speechiness': 'speechiness',
    'acousticness': 'acousticness',
    'instrumentalness': 'instrumentalness',
    'liveness': 'liveness',
    'valence': 'valence',
    'tempo': 'tempo',
    'duration_ms': 'duration_ms'
    # Note: time_signature not available in this dataset
}

CLUSTERING_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]

class LyricsDatasetSelector:
    """Selector for representative songs from lyrics dataset."""
    
    def __init__(self, target_size: int = 10000):
        configure_for_large_dataset()
        self.config = get_config()
        self.target_size = target_size
        
        # Components
        self.sampling_strategies = SamplingStrategies()
        self.stats_analyzer = DescriptiveStats()
        
        logger.info(f"🎯 Lyrics Dataset Selector initialized")
        logger.info(f"📊 Target selection size: {target_size:,} songs")
    
    def load_lyrics_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load the fixed lyrics dataset."""
        logger.info(f"📂 Loading lyrics dataset: {dataset_path}")
        
        try:
            # Load with @@ separator (use python engine for multi-char separators)
            df = pd.read_csv(dataset_path, sep='@@', encoding='utf-8', engine='python')
            logger.info(f"✅ Dataset loaded: {df.shape}")
            
            # Rename columns to match our standard
            df_renamed = df.rename(columns=COLUMN_MAPPING)
            
            # Verify we have the needed features
            missing_features = [f for f in CLUSTERING_FEATURES if f not in df_renamed.columns]
            if missing_features:
                logger.warning(f"⚠️ Missing features: {missing_features}")
            
            # Add time_signature as default (not available in spotify_songs)
            if 'time_signature' not in df_renamed.columns:
                df_renamed['time_signature'] = 4  # Default 4/4 time
                logger.info("➕ Added default time_signature=4")
            
            logger.info(f"🎵 Songs with lyrics: {len(df_renamed):,}")
            logger.info(f"📊 Available features: {len([f for f in CLUSTERING_FEATURES if f in df_renamed.columns])}/13")
            
            return df_renamed
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            return None
    
    def stage_1_diversity_sampling(self, data: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """Stage 1: Diversity-based sampling."""
        logger.info(f"🌟 Stage 1: Diversity sampling ({len(data):,} → {target_size:,})")
        start_time = time.time()
        
        try:
            # Use diversity sampling for good feature space coverage
            available_features = [f for f in CLUSTERING_FEATURES if f in data.columns]
            sampled_data = self.sampling_strategies.diversity_sample(
                df=data,
                feature_columns=available_features,
                sample_size=target_size
            )
            
            execution_time = time.time() - start_time
            
            if sampled_data is not None and len(sampled_data) > 0:
                logger.info(f"✅ Diversity sampling completed: {len(sampled_data):,} songs")
                logger.info(f"⏱️ Execution time: {execution_time:.2f}s")
                return sampled_data
            else:
                logger.warning("⚠️ Diversity sampling failed, using random sampling")
                return data.sample(n=min(target_size, len(data)), random_state=42)
                
        except Exception as e:
            logger.error(f"❌ Error in diversity sampling: {e}")
            # Fallback to random sampling
            return data.sample(n=min(target_size, len(data)), random_state=42)
    
    def stage_2_quality_filtering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Quality-based filtering."""
        logger.info(f"🔍 Stage 2: Quality filtering")
        start_time = time.time()
        
        try:
            # Calculate quality scores
            quality_scores = []
            
            for idx, row in data.iterrows():
                score = 0
                
                # Check for missing values in clustering features
                available_features = [f for f in CLUSTERING_FEATURES if f in data.columns]
                missing_count = row[available_features].isna().sum()
                completeness_score = (len(available_features) - missing_count) / len(available_features)
                score += completeness_score * 50
                
                # Check for valid ranges
                valid_range_count = 0
                for feature in available_features:
                    value = row[feature]
                    if pd.notna(value):
                        # Basic range validation
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
                        elif feature in ['key', 'mode']:
                            if value >= 0:
                                valid_range_count += 1
                        elif feature == 'duration_ms':
                            if value > 0:
                                valid_range_count += 1
                
                range_score = valid_range_count / len(available_features)
                score += range_score * 30
                
                # Popularity bonus (if available)
                if 'track_popularity' in row and pd.notna(row['track_popularity']):
                    try:
                        # Convert to numeric if it's a string
                        popularity = float(row['track_popularity'])
                        # Moderate popularity preferred (not too obscure, not too mainstream)
                        pop_score = 1.0 - abs(popularity - 50) / 50
                        score += pop_score * 20
                    except (ValueError, TypeError):
                        # Skip popularity bonus if conversion fails
                        pass
                
                quality_scores.append(score)
            
            # Add quality scores and sort
            data_with_quality = data.copy()
            data_with_quality['quality_score'] = quality_scores
            
            # Remove rows with very low quality (< 60)
            filtered_data = data_with_quality[data_with_quality['quality_score'] >= 60]
            filtered_data = filtered_data.drop('quality_score', axis=1)
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ Quality filtering completed: {len(filtered_data):,} songs")
            logger.info(f"📊 Average quality: {np.mean(quality_scores):.1f}/100")
            logger.info(f"⏱️ Execution time: {execution_time:.2f}s")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"❌ Error in quality filtering: {e}")
            return data
    
    def stage_3_stratified_sampling(self, data: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """Stage 3: Final stratified sampling to exact target size."""
        logger.info(f"📊 Stage 3: Stratified sampling ({len(data):,} → {target_size:,})")
        start_time = time.time()
        
        try:
            # Use balanced sampling to preserve distributions
            if len(data) <= target_size:
                logger.info("📋 Dataset already at target size or smaller")
                return data
            
            # Stratify by key and mode
            balance_columns = ['key', 'mode']
            available_balance_columns = [col for col in balance_columns if col in data.columns]
            
            if available_balance_columns:
                sampled_data = self.sampling_strategies.balanced_sample(
                    df=data,
                    sample_size=target_size,
                    balance_columns=available_balance_columns,
                    balance_method='proportional'
                )
            else:
                # Fallback to random sampling
                sampled_data = data.sample(n=target_size, random_state=42)
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ Stratified sampling completed: {len(sampled_data):,} songs")
            logger.info(f"⏱️ Execution time: {execution_time:.2f}s")
            
            return sampled_data
            
        except Exception as e:
            logger.error(f"❌ Error in stratified sampling: {e}")
            return data.sample(n=min(target_size, len(data)), random_state=42)
    
    def validate_selection(self, original_data: pd.DataFrame, selected_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the selection quality."""
        logger.info("🔍 Validating selection quality...")
        
        try:
            from scipy import stats
            
            validation_results = {
                'original_size': len(original_data),
                'selected_size': len(selected_data),
                'selection_ratio': len(selected_data) / len(original_data),
                'feature_comparisons': {},
                'overall_quality': 0.0
            }
            
            quality_scores = []
            available_features = [f for f in CLUSTERING_FEATURES if f in selected_data.columns]
            
            for feature in available_features:
                if feature in original_data.columns:
                    orig_values = original_data[feature].dropna()
                    sel_values = selected_data[feature].dropna()
                    
                    if len(orig_values) > 0 and len(sel_values) > 0:
                        # Statistical comparison
                        mean_diff = abs(orig_values.mean() - sel_values.mean()) / orig_values.std()
                        std_ratio = sel_values.std() / orig_values.std()
                        
                        # KS test for distribution similarity
                        ks_stat, ks_pvalue = stats.ks_2samp(orig_values, sel_values)
                        
                        # Quality score
                        mean_score = max(0, 100 - mean_diff * 100)
                        std_score = max(0, 100 - abs(1 - std_ratio) * 100)
                        dist_score = max(0, 100 - ks_stat * 100)
                        
                        feature_quality = (mean_score + std_score + dist_score) / 3
                        quality_scores.append(feature_quality)
                        
                        validation_results['feature_comparisons'][feature] = {
                            'original_mean': float(orig_values.mean()),
                            'selected_mean': float(sel_values.mean()),
                            'quality_score': float(feature_quality)
                        }
            
            # Overall quality
            if quality_scores:
                validation_results['overall_quality'] = np.mean(quality_scores)
            
            logger.info(f"✅ Validation completed")
            logger.info(f"🎯 Overall quality: {validation_results['overall_quality']:.1f}/100")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Error in validation: {e}")
            return {'error': str(e)}
    
    def save_selected_dataset(self, selected_data: pd.DataFrame) -> str:
        """Save the selected dataset."""
        # Save in data/final_data/ directory
        final_data_dir = Path("data/final_data")
        final_data_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = final_data_dir / "picked_data_lyrics.csv"
        
        try:
            # Save with ^ separator to avoid conflicts with lyrics content (single char required)
            selected_data.to_csv(output_path, sep='^', index=False, encoding='utf-8')
            logger.info(f"💾 Selected dataset saved: {output_path}")
            logger.info(f"📊 Saved {len(selected_data):,} songs with verified lyrics")
            logger.info(f"📋 Format: ^ separator (safe for lyrics content)")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving dataset: {e}")
            return ""
    
    def select_representative_songs(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main selection pipeline."""
        logger.info("🚀 Starting representative song selection from lyrics dataset...")
        total_start_time = time.time()
        
        try:
            # Load dataset
            dataset_path = "data/with_lyrics/spotify_songs_fixed.csv"
            data = self.load_lyrics_dataset(dataset_path)
            
            if data is None or len(data) == 0:
                logger.error("❌ Failed to load dataset")
                return None, {}
            
            original_data = data.copy()
            
            # Stage 1: Diversity sampling (18K → ~12K)
            current_data = self.stage_1_diversity_sampling(data, int(self.target_size * 1.2))
            
            # Stage 2: Quality filtering
            current_data = self.stage_2_quality_filtering(current_data)
            
            # Stage 3: Final stratified sampling to exact target
            final_data = self.stage_3_stratified_sampling(current_data, self.target_size)
            
            # Validate selection
            validation_results = self.validate_selection(original_data, final_data)
            
            # Save results
            output_path = self.save_selected_dataset(final_data)
            
            # Summary
            total_time = time.time() - total_start_time
            logger.info("🎉 Selection completed successfully!")
            logger.info(f"📊 Selection: {len(original_data):,} → {len(final_data):,} songs")
            logger.info(f"🎵 All selected songs have verified lyrics")
            logger.info(f"🎯 Quality score: {validation_results.get('overall_quality', 0):.1f}/100")
            logger.info(f"⏱️ Total time: {total_time:.2f} seconds")
            logger.info(f"💾 Dataset saved: {output_path}")
            
            return final_data, validation_results
            
        except Exception as e:
            logger.error(f"❌ Selection pipeline failed: {e}")
            return None, {}

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Select representative songs from lyrics dataset")
    parser.add_argument('--target-size', type=int, default=10000, help='Number of songs to select')
    
    args = parser.parse_args()
    
    print("🎵 REPRESENTATIVE SELECTION FROM LYRICS DATASET")
    print("=" * 50)
    print(f"📊 Target size: {args.target_size:,} songs")
    print(f"🎵 Source: 18K songs with verified lyrics")
    print("=" * 50)
    
    # Initialize selector
    selector = LyricsDatasetSelector(target_size=args.target_size)
    
    # Select representative songs
    selected_data, validation_results = selector.select_representative_songs()
    
    if selected_data is not None:
        print(f"\n🎉 Selection completed successfully!")
        print(f"📊 Selected {len(selected_data):,} representative songs")
        print(f"🎵 All songs have verified lyrics")
        print(f"🎯 Quality score: {validation_results.get('overall_quality', 0):.1f}/100")
        print(f"📁 Dataset saved as: data/final_data/picked_data_lyrics.csv")
    else:
        print(f"\n❌ Selection failed!")
        print(f"📋 Check logs for details")

if __name__ == "__main__":
    main()