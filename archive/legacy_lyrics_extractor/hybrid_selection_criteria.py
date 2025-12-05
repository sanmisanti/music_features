#!/usr/bin/env python3
"""
Hybrid Selection Criteria

Multi-criteria scoring system that balances musical diversity with lyrics availability.
Implements progressive constraints to achieve exact 80/20 distribution through pipeline stages.

Author: Music Features Analysis Project
Date: 2025-01-28
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Configuration for hybrid scoring weights."""
    musical_diversity: float = 0.4
    lyrics_availability: float = 0.4
    popularity_factor: float = 0.15
    genre_balance: float = 0.05
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.musical_diversity + self.lyrics_availability + self.popularity_factor + self.genre_balance
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


@dataclass
class StageConstraints:
    """Progressive constraints for each selection stage."""
    stage_1: float = 0.70  # 70% with lyrics
    stage_2: float = 0.75  # 75% with lyrics
    stage_3: float = 0.78  # 78% with lyrics
    stage_4: float = 0.80  # 80% with lyrics (final)


class HybridSelectionCriteria:
    """
    Multi-criteria selection system for balancing musical diversity with lyrics availability.
    
    Features:
    - Configurable scoring weights for different criteria
    - Progressive constraints through pipeline stages (70% → 75% → 78% → 80%)
    - Genre-aware balancing within lyrics/no-lyrics groups
    - Validation and quality metrics for selection results
    """
    
    def __init__(self, target_lyrics_ratio: float = 0.8, 
                 scoring_weights: Optional[ScoringWeights] = None,
                 stage_constraints: Optional[StageConstraints] = None):
        """
        Initialize hybrid selection criteria.
        
        Args:
            target_lyrics_ratio: Final target ratio for songs with lyrics (0.8 = 80%)
            scoring_weights: Custom scoring weights configuration
            stage_constraints: Custom progressive constraints configuration
        """
        self.target_ratio = target_lyrics_ratio
        self.weights = scoring_weights or ScoringWeights()
        self.stage_constraints = stage_constraints or StageConstraints()
        
        # Get progressive ratios as dictionary
        self.stage_ratios = {
            1: self.stage_constraints.stage_1,
            2: self.stage_constraints.stage_2,
            3: self.stage_constraints.stage_3,
            4: self.stage_constraints.stage_4
        }
        
        # Statistics tracking
        self.stats = {
            'total_scored': 0,
            'selections_made': 0,
            'constraints_applied': 0,
            'validation_checks': 0
        }
        
        logger.info(f"HybridSelectionCriteria initialized")
        logger.info(f"Target ratio: {self.target_ratio:.1%}")
        logger.info(f"Progressive ratios: {[f'S{k}:{v:.1%}' for k,v in self.stage_ratios.items()]}")
    
    def calculate_diversity_score(self, song_features: pd.Series, reference_features: pd.DataFrame) -> float:
        """
        Calculate musical diversity score based on distance to existing selections.
        
        Args:
            song_features: Features of the candidate song
            reference_features: Features of already selected songs
            
        Returns:
            Diversity score between 0 and 1 (higher = more diverse)
        """
        if reference_features.empty:
            return 1.0  # First song gets maximum diversity score
        
        # Use key musical features for diversity calculation
        diversity_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'instrumentalness', 'speechiness', 'tempo']
        
        # Filter to available features
        available_features = [f for f in diversity_features if f in song_features.index]
        
        if not available_features:
            return 0.5  # Default score if no features available
        
        try:
            # Calculate minimum distance to all selected songs
            candidate_vector = song_features[available_features].values
            reference_vectors = reference_features[available_features].values
            
            # Euclidean distances
            distances = np.sqrt(np.sum((reference_vectors - candidate_vector) ** 2, axis=1))
            min_distance = np.min(distances)
            
            # Normalize to 0-1 range (assuming max possible distance ~2.0 in normalized space)
            diversity_score = min(min_distance / 2.0, 1.0)
            
            return diversity_score
            
        except Exception as e:
            logger.warning(f"Error calculating diversity score: {e}")
            return 0.5
    
    def calculate_popularity_score(self, song_features: pd.Series) -> float:
        """
        Calculate popularity score based on musical characteristics.
        
        Args:
            song_features: Features of the song
            
        Returns:
            Popularity score between 0 and 1
        """
        try:
            # Use characteristics that correlate with mainstream appeal
            popularity_indicators = {
                'danceability': 0.3,    # Higher danceability = more popular
                'energy': 0.2,          # Moderate energy preferred
                'valence': 0.2,         # Positive songs more popular
                'speechiness': -0.2,    # Less speechiness = more popular
                'instrumentalness': -0.3 # Less instrumental = more popular
            }
            
            score = 0.5  # Base score
            
            for feature, weight in popularity_indicators.items():
                if feature in song_features.index:
                    value = song_features[feature]
                    
                    if weight > 0:
                        # Higher value is better
                        score += weight * value
                    else:
                        # Lower value is better
                        score += abs(weight) * (1 - value)
            
            # Ensure score is in [0, 1] range
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating popularity score: {e}")
            return 0.5
    
    def calculate_hybrid_score(self, song_features: pd.Series, has_lyrics: bool,
                             diversity_score: float, popularity_score: Optional[float] = None) -> float:
        """
        Calculate comprehensive hybrid score combining all criteria.
        
        Args:
            song_features: Features of the song
            has_lyrics: Whether the song has lyrics available
            diversity_score: Pre-calculated diversity score
            popularity_score: Optional pre-calculated popularity score
            
        Returns:
            Hybrid score between 0 and 1
        """
        try:
            # Calculate popularity score if not provided
            if popularity_score is None:
                popularity_score = self.calculate_popularity_score(song_features)
            
            # Lyrics availability bonus/penalty
            lyrics_score = 1.0 if has_lyrics else 0.0
            
            # Genre balance score (placeholder - can be enhanced)
            genre_score = 0.5  # Neutral for now
            
            # Calculate weighted hybrid score
            hybrid_score = (
                self.weights.musical_diversity * diversity_score +
                self.weights.lyrics_availability * lyrics_score +
                self.weights.popularity_factor * popularity_score +
                self.weights.genre_balance * genre_score
            )
            
            self.stats['total_scored'] += 1
            
            return max(0.0, min(1.0, hybrid_score))
            
        except Exception as e:
            logger.error(f"Error calculating hybrid score: {e}")
            return 0.0
    
    def apply_progressive_constraints(self, scored_songs: pd.DataFrame, 
                                   stage_num: int, target_size: int) -> pd.DataFrame:
        """
        Apply progressive constraints to maintain target lyrics ratio.
        
        Args:
            scored_songs: DataFrame with songs and their scores
            stage_num: Current pipeline stage (1-4)
            target_size: Target number of songs to select
            
        Returns:
            Selected songs DataFrame maintaining progressive ratio
        """
        logger.info(f"Applying constraints for Stage {stage_num}: {target_size:,} songs")
        
        if stage_num not in self.stage_ratios:
            raise ValueError(f"Invalid stage number: {stage_num}")
        
        target_lyrics_ratio = self.stage_ratios[stage_num]
        target_with_lyrics = int(target_size * target_lyrics_ratio)
        target_without_lyrics = target_size - target_with_lyrics
        
        logger.info(f"Target distribution: {target_with_lyrics:,} with lyrics, {target_without_lyrics:,} without")
        
        # Separate songs by lyrics availability
        with_lyrics = scored_songs[scored_songs['has_lyrics'] == True].copy()
        without_lyrics = scored_songs[scored_songs['has_lyrics'] == False].copy()
        
        logger.info(f"Available: {len(with_lyrics):,} with lyrics, {len(without_lyrics):,} without")
        
        # Check if we have enough songs in each category
        if len(with_lyrics) < target_with_lyrics:
            logger.warning(f"Not enough songs with lyrics: need {target_with_lyrics}, have {len(with_lyrics)}")
            target_with_lyrics = len(with_lyrics)
            target_without_lyrics = target_size - target_with_lyrics
        
        if len(without_lyrics) < target_without_lyrics:
            logger.warning(f"Not enough songs without lyrics: need {target_without_lyrics}, have {len(without_lyrics)}")
            target_without_lyrics = len(without_lyrics)
            target_with_lyrics = target_size - target_without_lyrics
        
        # Select top scoring songs from each group
        selected_with = with_lyrics.nlargest(target_with_lyrics, 'hybrid_score')
        selected_without = without_lyrics.nlargest(target_without_lyrics, 'hybrid_score')
        
        # Combine and shuffle
        selected_songs = pd.concat([selected_with, selected_without], ignore_index=True)
        selected_songs = selected_songs.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Update statistics
        self.stats['selections_made'] += 1
        self.stats['constraints_applied'] += 1
        
        # Log results
        actual_with = selected_songs['has_lyrics'].sum()
        actual_ratio = actual_with / len(selected_songs) if len(selected_songs) > 0 else 0
        logger.info(f"Selected: {len(selected_songs):,} songs ({actual_with:,} with lyrics, {actual_ratio:.1%})")
        
        return selected_songs
    
    def validate_lyrics_distribution(self, selected_songs: pd.DataFrame, 
                                   expected_ratio: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate the lyrics distribution in selected songs.
        
        Args:
            selected_songs: DataFrame with selected songs
            expected_ratio: Expected ratio of songs with lyrics (defaults to target_ratio)
            
        Returns:
            Dictionary with validation results
        """
        if expected_ratio is None:
            expected_ratio = self.target_ratio
        
        total = len(selected_songs)
        if total == 0:
            return {'error': 'No songs provided for validation'}
        
        with_lyrics = selected_songs['has_lyrics'].sum() if 'has_lyrics' in selected_songs.columns else 0
        without_lyrics = total - with_lyrics
        actual_ratio = with_lyrics / total
        
        # Calculate deviation from expected
        ratio_deviation = abs(actual_ratio - expected_ratio)
        ratio_acceptable = ratio_deviation <= 0.02  # Allow 2% deviation
        
        validation_result = {
            'total_songs': total,
            'songs_with_lyrics': with_lyrics,
            'songs_without_lyrics': without_lyrics,
            'actual_lyrics_ratio': actual_ratio,
            'expected_lyrics_ratio': expected_ratio,
            'ratio_deviation': ratio_deviation,
            'ratio_acceptable': ratio_acceptable,
            'lyrics_percentage': actual_ratio * 100,
            'quality_score': max(0, 100 - (ratio_deviation * 100 * 10))  # Penalty for deviation
        }
        
        self.stats['validation_checks'] += 1
        
        return validation_result
    
    def analyze_selection_quality(self, selected_songs: pd.DataFrame, 
                                original_songs: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the quality of the selection in terms of diversity and representation.
        
        Args:
            selected_songs: DataFrame with selected songs
            original_songs: DataFrame with original song pool
            
        Returns:
            Dictionary with quality analysis
        """
        logger.info("Analyzing selection quality...")
        
        analysis = {
            'selection_ratio': len(selected_songs) / len(original_songs),
            'diversity_preserved': True,  # Placeholder
            'representation_score': 85.0,  # Placeholder
            'lyrics_distribution': self.validate_lyrics_distribution(selected_songs)
        }
        
        # Add feature distribution comparison if features available
        feature_cols = ['danceability', 'energy', 'valence', 'acousticness']
        available_features = [f for f in feature_cols if f in selected_songs.columns and f in original_songs.columns]
        
        if available_features:
            feature_comparison = {}
            for feature in available_features:
                orig_mean = original_songs[feature].mean()
                selected_mean = selected_songs[feature].mean()
                deviation = abs(selected_mean - orig_mean)
                feature_comparison[feature] = {
                    'original_mean': orig_mean,
                    'selected_mean': selected_mean,
                    'deviation': deviation,
                    'preserved': deviation < 0.1  # Within 10% is good
                }
            
            analysis['feature_preservation'] = feature_comparison
        
        return analysis
    
    def print_statistics(self):
        """Print statistics about the hybrid selection process."""
        print("\n" + "="*50)
        print("HYBRID SELECTION STATISTICS")
        print("="*50)
        print(f"Total songs scored: {self.stats['total_scored']}")
        print(f"Selections made: {self.stats['selections_made']}")
        print(f"Constraints applied: {self.stats['constraints_applied']}")
        print(f"Validation checks: {self.stats['validation_checks']}")
        print(f"Target lyrics ratio: {self.target_ratio:.1%}")
        print("="*50)


def main():
    """Test the hybrid selection criteria functionality."""
    pass  # Test will be in separate file


if __name__ == "__main__":
    main()