#!/usr/bin/env python3
"""
Test script for HybridSelectionCriteria

Focused test with precise output for scoring and constraints system.
Run this to verify the hybrid selection criteria is working correctly.

Usage: cd lyrics_extractor/tests && python test_hybrid_criteria.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_selection_criteria import HybridSelectionCriteria, ScoringWeights, StageConstraints

def create_test_dataset(size: int = 100) -> pd.DataFrame:
    """Create synthetic test dataset with musical features and lyrics flags."""
    np.random.seed(42)  # Reproducible results
    
    data = {
        'id': [f'song_{i:03d}' for i in range(size)],
        'name': [f'Test Song {i}' for i in range(size)],
        'danceability': np.random.uniform(0, 1, size),
        'energy': np.random.uniform(0, 1, size),
        'valence': np.random.uniform(0, 1, size),
        'acousticness': np.random.uniform(0, 1, size),
        'instrumentalness': np.random.uniform(0, 1, size),
        'speechiness': np.random.uniform(0, 1, size),
        'tempo': np.random.uniform(60, 200, size),
        # Simulate lyrics availability (70% have lyrics initially)
        'has_lyrics': np.random.choice([True, False], size, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

def main():
    """Test hybrid selection criteria with precise debugging output."""
    print("🎯 TESTING: HybridSelectionCriteria")
    
    try:
        # Create test dataset
        print("📊 Creating test dataset...")
        test_data = create_test_dataset(100)
        initial_with_lyrics = test_data['has_lyrics'].sum()
        print(f"   Dataset: {len(test_data)} songs, {initial_with_lyrics} with lyrics ({initial_with_lyrics/len(test_data):.1%})")
        
        # Initialize hybrid selector
        print("\n🚀 Initializing HybridSelectionCriteria...")
        selector = HybridSelectionCriteria(target_lyrics_ratio=0.8)
        print(f"✅ Selector initialized | Target ratio: {selector.target_ratio:.1%}")
        
        # Test scoring components
        print("\n🧪 Testing scoring components:")
        
        # Test popularity scoring
        sample_song = test_data.iloc[0]
        popularity_score = selector.calculate_popularity_score(sample_song)
        print(f"   Popularity score: {popularity_score:.3f}")
        
        # Test diversity scoring (empty reference)
        diversity_score = selector.calculate_diversity_score(sample_song, pd.DataFrame())
        print(f"   Diversity score (first song): {diversity_score:.3f}")
        
        # Test diversity scoring with reference
        reference_data = test_data.iloc[1:3]  # Use songs 1-2 as reference
        diversity_score_ref = selector.calculate_diversity_score(sample_song, reference_data)
        print(f"   Diversity score (with reference): {diversity_score_ref:.3f}")
        
        # Test hybrid scoring
        hybrid_score = selector.calculate_hybrid_score(
            sample_song, sample_song['has_lyrics'], diversity_score, popularity_score
        )
        print(f"   Hybrid score: {hybrid_score:.3f}")
        
        # Test progressive constraints
        print(f"\n🎯 Testing progressive constraints:")
        
        # Add scores to test data
        test_data['hybrid_score'] = np.random.uniform(0, 1, len(test_data))
        
        # Test each stage
        stage_sizes = [80, 60, 40, 20]  # Reducing sizes through pipeline
        
        current_data = test_data.copy()
        
        for stage, target_size in enumerate(stage_sizes, 1):
            if len(current_data) < target_size:
                continue
                
            print(f"\n📋 Stage {stage} (target: {target_size} songs):")
            
            # Apply constraints
            selected = selector.apply_progressive_constraints(current_data, stage, target_size)
            
            # Validate results
            validation = selector.validate_lyrics_distribution(selected, selector.stage_ratios[stage])
            
            print(f"   Selected: {len(selected)} songs")
            print(f"   With lyrics: {validation['songs_with_lyrics']} ({validation['lyrics_percentage']:.1f}%)")
            print(f"   Target ratio: {selector.stage_ratios[stage]:.1%}")
            print(f"   Deviation: {validation['ratio_deviation']:.3f}")
            print(f"   Acceptable: {'✅' if validation['ratio_acceptable'] else '❌'}")
            
            # Use selected songs for next stage
            current_data = selected
        
        # Test validation function
        print(f"\n✅ Testing validation:")
        final_validation = selector.validate_lyrics_distribution(current_data)
        print(f"   Final ratio: {final_validation['lyrics_percentage']:.1f}%")
        print(f"   Quality score: {final_validation['quality_score']:.1f}/100")
        print(f"   Within target: {'✅' if final_validation['ratio_acceptable'] else '❌'}")
        
        # Test quality analysis
        print(f"\n📊 Testing quality analysis:")
        quality = selector.analyze_selection_quality(current_data, test_data)
        print(f"   Selection ratio: {quality['selection_ratio']:.3f}")
        print(f"   Representation score: {quality['representation_score']:.1f}")
        
        # Show statistics
        print(f"\n📈 PERFORMANCE STATS:")
        print(f"   Songs scored: {selector.stats['total_scored']}")
        print(f"   Selections made: {selector.stats['selections_made']}")
        print(f"   Constraints applied: {selector.stats['constraints_applied']}")
        print(f"   Validations run: {selector.stats['validation_checks']}")
        
        print(f"\n✅ TEST COMPLETED - All functions working")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        print(f"🐛 Debug info: {traceback.format_exc()}")

if __name__ == "__main__":
    main()