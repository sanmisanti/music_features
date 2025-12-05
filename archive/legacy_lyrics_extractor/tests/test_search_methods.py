#!/usr/bin/env python3
"""
Test Search Methods for Lyrics Extraction

This script tests the search methods used by genius_lyrics_extractor.py
specifically for the Ozuna song "Reggaeton en Paris" to debug why it's not found.

Author: Music Features Analysis Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genius_lyrics_extractor import GeniusLyricsExtractor
import lyricsgenius

def test_ozuna_song():
    """Test search methods for Ozuna's 'Reggaeton en Paris'."""
    
    print("🔍 TESTING SEARCH FOR: Reggaeton en Paris by Ozuna")
    print("=" * 60)
    
    # Test data from your CSV
    song_name = "Reggaeton en Paris"
    artist_name = "Ozuna"  # From cleaned CSV format: ['Ozuna', 'Dalex', 'Nicky Jam'] -> 'Ozuna'
    
    try:
        # Initialize extractor
        extractor = GeniusLyricsExtractor()
        print("✅ Genius API client initialized\n")
        
        # Step 1: Test artist name cleaning
        print("STEP 1: Artist Name Cleaning")
        print("-" * 30)
        original_artist = "['Ozuna', 'Dalex', 'Nicky Jam']"
        cleaned_artist = extractor.clean_artist_name(original_artist)
        print(f"Original: {original_artist}")
        print(f"Cleaned:  {cleaned_artist}")
        
        # Step 2: Test search term normalization
        print(f"\nSTEP 2: Search Term Normalization")
        print("-" * 30)
        norm_song, norm_artist = extractor.normalize_search_terms(song_name, cleaned_artist)
        print(f"Original song:  '{song_name}'")
        print(f"Normalized song: '{norm_song}'")
        print(f"Original artist: '{cleaned_artist}'")
        print(f"Normalized artist: '{norm_artist}'")
        
        # Step 3: Test each search strategy individually
        print(f"\nSTEP 3: Testing Search Strategies")
        print("-" * 30)
        
        search_strategies = [
            f"{norm_song} {norm_artist}",  # Song + Artist
            f"{norm_artist} {norm_song}",  # Artist + Song
            norm_song,                     # Song only
            f'"{norm_song}" {norm_artist}' # Quoted song + artist
        ]
        
        for i, search_term in enumerate(search_strategies, 1):
            print(f"\nStrategy {i}: '{search_term}'")
            try:
                # Direct search without verification
                song = extractor.genius.search_song(search_term)
                
                if song:
                    print(f"  ✅ FOUND: '{song.title}' by '{song.primary_artist.name}'")
                    print(f"  🎵 Genius ID: {song.id}")
                    print(f"  🔗 URL: {song.url}")
                    
                    # Test verification
                    is_match = extractor._verify_song_match(song, norm_song, norm_artist)
                    print(f"  🔍 Verification: {'✅ PASS' if is_match else '❌ FAIL'}")
                    
                    if not is_match:
                        # Show detailed comparison
                        found_title = song.title.lower().strip()
                        found_artist = song.primary_artist.name.lower().strip()
                        target_song_lower = norm_song.lower().strip()
                        target_artist_lower = norm_artist.lower().strip()
                        
                        print(f"    🎯 Target: '{target_song_lower}' by '{target_artist_lower}'")
                        print(f"    🎵 Found:  '{found_title}' by '{found_artist}'")
                        
                        # Check title similarity
                        title_similarity = extractor._calculate_similarity(found_title, target_song_lower)
                        print(f"    📊 Title similarity: {title_similarity:.3f}")
                        
                        # Check artist similarity  
                        artist_similarity = extractor._calculate_similarity(found_artist, target_artist_lower)
                        print(f"    👤 Artist similarity: {artist_similarity:.3f}")
                    
                    if is_match and hasattr(song, 'lyrics') and song.lyrics:
                        print(f"  📝 Lyrics: {len(song.lyrics)} characters")
                        print(f"  📄 First 100 chars: {song.lyrics[:100]}...")
                        return True
                    elif is_match:
                        print(f"  ❌ No lyrics available")
                        
                else:
                    print(f"  ❌ NOT FOUND")
                    
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
        
        # Step 4: Manual search alternatives
        print(f"\nSTEP 4: Manual Search Alternatives")
        print("-" * 30)
        
        alternative_searches = [
            "Reggaetón en París Ozuna",  # With accents
            "Reggaeton Paris Ozuna",     # Without "en"
            "Ozuna Dalex Nicky Jam Reggaeton",  # All artists
            "Reggaeton en Paris feat Dalex",    # With featured artists
        ]
        
        for alt_search in alternative_searches:
            print(f"\nAlternative: '{alt_search}'")
            try:
                song = extractor.genius.search_song(alt_search)
                if song:
                    print(f"  ✅ FOUND: '{song.title}' by '{song.primary_artist.name}'")
                else:
                    print(f"  ❌ NOT FOUND")
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_direct_genius_search():
    """Test direct lyricsgenius search without our wrapper."""
    print(f"\n🔧 DIRECT GENIUS API TEST")
    print("=" * 60)
    
    try:
        # Initialize basic Genius client
        genius = lyricsgenius.Genius("Kmryp8sRaJ6ZlwRdy_DaUTdR28OXGjEtJ29VikqhUO3eCnA_ovH5gLMajGIzv_qD")
        
        # Test different search terms
        searches = [
            "Reggaeton en Paris",
            "Ozuna Reggaeton en Paris", 
            "Reggaetón en París",
            "Ozuna Dalex Nicky Jam"
        ]
        
        for search_term in searches:
            print(f"\nDirect search: '{search_term}'")
            try:
                song = genius.search_song(search_term)
                if song:
                    print(f"  ✅ Found: {song.title} by {song.primary_artist.name}")
                    if hasattr(song, 'lyrics') and song.lyrics:
                        print(f"  📝 Has lyrics: {len(song.lyrics)} chars")
                else:
                    print(f"  ❌ Not found")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Direct test failed: {e}")

if __name__ == "__main__":
    print("🎵 LYRICS SEARCH DIAGNOSTIC TOOL")
    print("=" * 60)
    print("Testing search methods for: Reggaeton en Paris by Ozuna")
    print("This will help identify why the song isn't being found.\n")
    
    # Run tests
    found = test_ozuna_song()
    test_direct_genius_search()
    
    print(f"\n{'='*60}")
    print("🎯 SUMMARY")
    print("=" * 60)
    if found:
        print("✅ Song found successfully with current methods")
    else:
        print("❌ Song not found - need to adjust search strategy")
        print("\n💡 Possible solutions:")
        print("1. Relax verification criteria")
        print("2. Add accent normalization")
        print("3. Include featured artists in search")
        print("4. Try different search term combinations")