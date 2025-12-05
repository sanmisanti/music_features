#!/usr/bin/env python3
"""
Test the accent normalization fix for the Ozuna song.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genius_lyrics_extractor import GeniusLyricsExtractor

def test_ozuna_fix():
    """Test that the accent fix works for Ozuna's song."""
    
    print("🔧 TESTING ACCENT NORMALIZATION FIX")
    print("=" * 50)
    
    try:
        extractor = GeniusLyricsExtractor()
        
        # Test the search with our fixed method
        song = extractor.search_song_with_fallbacks("Reggaeton en Paris", "Ozuna")
        
        if song and hasattr(song, 'lyrics') and song.lyrics:
            print("✅ SUCCESS! Song found and verified!")
            print(f"   Title: {song.title}")
            print(f"   Artist: {song.primary_artist.name}")
            print(f"   Lyrics length: {len(song.lyrics)} characters")
            print(f"   URL: {song.url}")
            return True
        else:
            print("❌ Still not working")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_ozuna_fix()
    if success:
        print("\n🎉 FIX SUCCESSFUL! The extractor should now find more songs.")
    else:
        print("\n⚠️  Fix needs more work.")