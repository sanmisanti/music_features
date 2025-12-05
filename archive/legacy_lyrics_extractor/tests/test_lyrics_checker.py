#!/usr/bin/env python3
"""
Test script for LyricsAvailabilityChecker

Simple, focused test with precise debugging output.
Run this to verify the lyrics checker is working correctly.

Usage: cd lyrics_extractor/tests && python test_lyrics_checker.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lyrics_availability_checker import LyricsAvailabilityChecker
import pandas as pd

def main():
    """Test the lyrics availability checker with precise debugging output."""
    print("🔍 TESTING: LyricsAvailabilityChecker")
    
    # Test data - mix of known songs and edge cases
    test_data = pd.DataFrame([
        {'id': 'test1', 'name': 'Reggaeton en Paris', 'artists': "['Ozuna', 'Dalex', 'Nicky Jam']"},
        {'id': 'test2', 'name': 'Bohemian Rhapsody', 'artists': "['Queen']"},
        {'id': 'test3', 'name': 'NonExistentSong12345', 'artists': "['UnknownArtist']"},
        {'id': 'test4', 'name': 'Shape of You', 'artists': "['Ed Sheeran']"},
        {'id': 'test5', 'name': 'Despacito', 'artists': "['Luis Fonsi', 'Daddy Yankee']"}
    ])
    
    print(f"📊 Test dataset: {len(test_data)} songs")
    
    try:
        # Initialize checker
        print("🚀 Initializing checker...")
        checker = LyricsAvailabilityChecker()
        print(f"✅ Checker initialized | Cache entries: {len(checker.cache)}")
        
        # Test individual functions first
        print("\n🧪 Testing normalization:")
        test_artist = "['Ozuna', 'Dalex', 'Nicky Jam']"
        cleaned = checker._clean_artist_name(test_artist)
        normalized = checker._normalize_text("Reggaetón en París")
        print(f"   Artist clean: '{test_artist}' → '{cleaned}'")
        print(f"   Text normalize: 'Reggaetón en París' → '{normalized}'")
        
        # Test similarity
        similarity = checker._calculate_similarity("reggaeton en paris", "reggaeton en paris")
        print(f"   Similarity (identical): {similarity:.3f}")
        similarity2 = checker._calculate_similarity("reggaeton en paris", "reggaeton en paris ozuna")
        print(f"   Similarity (partial): {similarity2:.3f}")
        
        # Perform batch verification
        print(f"\n🔍 Starting batch verification (batch_size=2)...")
        results = checker.quick_check_batch(test_data, batch_size=2)
        
        # Show key results only
        print(f"\n📋 RESULTS SUMMARY:")
        total_with_lyrics = sum(results.values())
        success_rate = (total_with_lyrics / len(results)) * 100
        print(f"   Songs with lyrics: {total_with_lyrics}/{len(results)} ({success_rate:.1f}%)")
        
        # Show individual results (compact format)
        print(f"\n📝 Individual Results:")
        for song_id, has_lyrics in results.items():
            song_info = test_data[test_data['id'] == song_id].iloc[0]
            status = "✅" if has_lyrics else "❌"
            print(f"   {status} {song_info['name'][:20]:<20} | {checker._clean_artist_name(song_info['artists'])[:15]:<15}")
        
        # Key statistics
        print(f"\n📊 PERFORMANCE STATS:")
        print(f"   API calls: {checker.stats['api_calls']}")
        print(f"   Cache hits: {checker.stats['cache_hits']}")
        print(f"   Errors: {checker.stats['errors']}")
        
        # Cache verification
        cache_stats = checker.get_cache_stats()
        print(f"   Cache entries: {cache_stats['total_entries']}")
        
        print(f"\n✅ TEST COMPLETED - All functions working")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        print(f"🐛 Debug info: {traceback.format_exc()}")

if __name__ == "__main__":
    main()