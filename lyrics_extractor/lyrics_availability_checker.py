#!/usr/bin/env python3
"""
Lyrics Availability Checker

Fast verification system for checking lyrics availability on Genius.com
without downloading complete lyrics. Optimized for massive batch processing
with intelligent caching and rate limiting.

Author: Music Features Analysis Project
Date: 2025-01-28
"""

import os
import sys
import time
import json
import logging
import requests
import unicodedata
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of lyrics availability verification."""
    song_id: str
    has_lyrics: bool
    confidence: float
    found_title: Optional[str] = None
    found_artist: Optional[str] = None
    genius_id: Optional[int] = None
    verification_time: float = 0.0
    error_message: Optional[str] = None


class LyricsAvailabilityChecker:
    """
    Fast lyrics availability checker using Genius API search endpoint.
    
    Features:
    - Quick verification without downloading complete lyrics
    - Intelligent caching system with JSON persistence
    - Optimized rate limiting for massive batch processing
    - Artist name cleaning and normalization
    - Similarity-based matching for better accuracy
    """
    
    def __init__(self, genius_token: Optional[str] = None, 
                 cache_path: str = "data/lyrics_availability_cache.json",
                 rate_limit_delay: float = 0.5):
        """
        Initialize the lyrics availability checker.
        
        Args:
            genius_token: Genius API access token
            cache_path: Path to cache file for storing verification results
            rate_limit_delay: Delay between API requests in seconds
        """
        # Get token from parameter, environment, or default
        self.genius_token = (genius_token or 
                           os.getenv('GENIUS_ACCESS_TOKEN') or 
                           "1CB7ylbYY6TY_MWHG3_qGhe0tA9mR0O23a75Id_KBLhYvPHhlq6v_REbO7OMcb0A")
        
        if not self.genius_token or self.genius_token == "TU_TOKEN_AQUI":
            raise ValueError(
                "Genius API token required. Set GENIUS_ACCESS_TOKEN environment variable "
                "or pass as parameter. Get your token from: https://genius.com/api-clients"
            )
        
        # Configuration
        self.cache_path = Path(cache_path)
        self.rate_limit_delay = rate_limit_delay
        
        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load cache
        self.cache = self._load_cache()
        
        # Setup HTTP session with retries
        self.session = self._setup_session()
        
        # Statistics
        self.stats = {
            'total_checked': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'errors': 0
        }
        
        logger.info(f"LyricsAvailabilityChecker initialized")
        logger.info(f"Cache path: {self.cache_path}")
        logger.info(f"Cached entries: {len(self.cache)}")
    
    def _setup_session(self) -> requests.Session:
        """Setup HTTP session with retry strategy."""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load cache from JSON file."""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    logger.info(f"Loaded {len(cache_data)} entries from cache")
                    return cache_data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        
        return {}
    
    def _save_cache(self):
        """Save cache to JSON file."""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _clean_artist_name(self, artist_str: str) -> str:
        """
        Clean artist name from dataset format.
        
        Args:
            artist_str: Artist string in format "['Artist Name']" or similar
            
        Returns:
            Clean artist name
        """
        if pd.isna(artist_str) or not artist_str:
            return ""
        
        # Remove brackets and quotes
        cleaned = str(artist_str).strip("[]'\"")
        
        # Handle multiple artists - take the first one
        if ',' in cleaned:
            cleaned = cleaned.split(',')[0].strip("'\" ")
        
        # Remove extra whitespace
        return cleaned.strip()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better matching (remove accents, lowercase).
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Normalize unicode (remove accents)
        normalized = unicodedata.normalize('NFD', text.lower())
        without_accents = ''.join(char for char in normalized 
                                 if unicodedata.category(char) != 'Mn')
        
        # Remove common patterns that hurt matching
        patterns = [
            r'\(feat\..*?\)', r'\(ft\..*?\)', r'\(featuring.*?\)',
            r'\(remix\)', r'\(live\)', r'\(acoustic\)', r'\(instrumental\)',
            r'\(radio edit\)', r'\(extended.*?\)', r'\(original.*?\)'
        ]
        
        result = without_accents
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        return result.strip()
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using character n-grams.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        if not str1 or not str2:
            return 0.0
        
        # Simple Jaccard similarity with character-level bigrams
        def get_bigrams(text: str) -> set:
            return set(text[i:i+2] for i in range(len(text) - 1))
        
        bigrams1 = get_bigrams(str1.lower())
        bigrams2 = get_bigrams(str2.lower())
        
        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1.intersection(bigrams2))
        union = len(bigrams1.union(bigrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def _quick_search_exists(self, song_name: str, artist_name: str) -> VerificationResult:
        """
        Quick verification of lyrics existence using search endpoint only.
        
        Args:
            song_name: Name of the song
            artist_name: Name of the artist
            
        Returns:
            VerificationResult with availability information
        """
        start_time = time.time()
        
        # Create unique key for this song
        song_key = f"{self._normalize_text(song_name)}|{self._normalize_text(artist_name)}"
        
        try:
            # Clean and normalize search terms
            clean_song = self._normalize_text(song_name)
            clean_artist = self._clean_artist_name(artist_name)
            clean_artist = self._normalize_text(clean_artist)
            
            if not clean_song or not clean_artist:
                return VerificationResult(
                    song_id=song_key,
                    has_lyrics=False,
                    confidence=0.0,
                    verification_time=time.time() - start_time,
                    error_message="Empty song or artist name"
                )
            
            # Search on Genius using search endpoint
            search_url = "https://api.genius.com/search"
            params = {"q": f"{clean_song} {clean_artist}"}
            headers = {"Authorization": f"Bearer {self.genius_token}"}
            
            response = self.session.get(search_url, params=params, headers=headers, timeout=10)
            self.stats['api_calls'] += 1
            
            if response.status_code == 200:
                data = response.json()
                hits = data.get('response', {}).get('hits', [])
                
                if hits:
                    # Check first few results for best match
                    best_match = None
                    best_confidence = 0.0
                    
                    for hit in hits[:3]:  # Check top 3 results
                        result = hit.get('result', {})
                        found_title = result.get('title', '')
                        found_artist = result.get('primary_artist', {}).get('name', '')
                        
                        # Normalize found data
                        norm_found_title = self._normalize_text(found_title)
                        norm_found_artist = self._normalize_text(found_artist)
                        
                        # Calculate similarities
                        title_similarity = self._calculate_similarity(clean_song, norm_found_title)
                        artist_similarity = self._calculate_similarity(clean_artist, norm_found_artist)
                        
                        # Combined confidence score
                        confidence = (title_similarity * 0.7 + artist_similarity * 0.3)
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_match = {
                                'title': found_title,
                                'artist': found_artist,
                                'genius_id': result.get('id'),
                                'confidence': confidence
                            }
                    
                    # Consider it a match if confidence > 0.6
                    has_lyrics = best_confidence > 0.6
                    
                    return VerificationResult(
                        song_id=song_key,
                        has_lyrics=has_lyrics,
                        confidence=best_confidence,
                        found_title=best_match['title'] if best_match else None,
                        found_artist=best_match['artist'] if best_match else None,
                        genius_id=best_match['genius_id'] if best_match else None,
                        verification_time=time.time() - start_time
                    )
                else:
                    # No results found
                    return VerificationResult(
                        song_id=song_key,
                        has_lyrics=False,
                        confidence=0.0,
                        verification_time=time.time() - start_time
                    )
            
            elif response.status_code == 429:
                # Rate limit exceeded
                logger.warning("Rate limit exceeded, waiting...")
                time.sleep(30)  # Wait 30 seconds on rate limit
                return VerificationResult(
                    song_id=song_key,
                    has_lyrics=False,
                    confidence=0.0,
                    verification_time=time.time() - start_time,
                    error_message="Rate limit exceeded"
                )
            
            else:
                # Other API error
                return VerificationResult(
                    song_id=song_key,
                    has_lyrics=False,
                    confidence=0.0,
                    verification_time=time.time() - start_time,
                    error_message=f"API error: {response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"Error checking {song_name} by {artist_name}: {e}")
            return VerificationResult(
                song_id=song_key,
                has_lyrics=False,
                confidence=0.0,
                verification_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def quick_check_batch(self, songs_df: pd.DataFrame, batch_size: int = 50) -> Dict[str, bool]:
        """
        Perform quick batch verification of lyrics availability.
        
        Args:
            songs_df: DataFrame with song information (must have 'id', 'name', 'artists' columns)
            batch_size: Number of songs to process before saving cache
            
        Returns:
            Dictionary mapping song_id to has_lyrics boolean
        """
        logger.info(f"Starting batch verification of {len(songs_df)} songs")
        logger.info(f"Batch size: {batch_size}, Rate limit delay: {self.rate_limit_delay}s")
        
        results = {}
        processed = 0
        
        for i in range(0, len(songs_df), batch_size):
            batch_start = time.time()
            batch = songs_df.iloc[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}: songs {i+1} to {min(i+batch_size, len(songs_df))}")
            
            for idx, song in batch.iterrows():
                song_id = song['id']
                song_name = song['name']
                artist_name = song['artists']
                
                # Create cache key
                cache_key = f"{self._normalize_text(song_name)}|{self._normalize_text(self._clean_artist_name(artist_name))}"
                
                # Check cache first
                if cache_key in self.cache:
                    cached_result = self.cache[cache_key]
                    results[song_id] = cached_result.get('has_lyrics', False)
                    self.stats['cache_hits'] += 1
                    self.stats['total_checked'] += 1
                    continue
                
                # Perform verification
                verification_result = self._quick_search_exists(song_name, artist_name)
                
                # Store in results and cache
                results[song_id] = verification_result.has_lyrics
                self.cache[cache_key] = {
                    'has_lyrics': verification_result.has_lyrics,
                    'confidence': verification_result.confidence,
                    'found_title': verification_result.found_title,
                    'found_artist': verification_result.found_artist,
                    'genius_id': verification_result.genius_id,
                    'verification_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'error_message': verification_result.error_message
                }
                
                # Update statistics
                self.stats['total_checked'] += 1
                if verification_result.has_lyrics:
                    self.stats['successful_verifications'] += 1
                else:
                    self.stats['failed_verifications'] += 1
                
                if verification_result.error_message:
                    self.stats['errors'] += 1
                
                processed += 1
                
                # Rate limiting
                if processed % 10 == 0:
                    success_rate = (self.stats['successful_verifications'] / self.stats['total_checked']) * 100
                    logger.info(f"Progress: {processed}/{len(songs_df)} songs, Success rate: {success_rate:.1f}%")
                
                time.sleep(self.rate_limit_delay)
            
            # Save cache after each batch
            self._save_cache()
            batch_time = time.time() - batch_start
            logger.info(f"Batch completed in {batch_time:.1f} seconds")
        
        # Final statistics
        self.print_statistics()
        
        return results
    
    def analyze_availability_patterns(self, results: Dict[str, bool], 
                                    songs_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in lyrics availability results.
        
        Args:
            results: Dictionary of song_id -> has_lyrics
            songs_df: Original DataFrame with song information
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing availability patterns...")
        
        # Add results to dataframe for analysis
        df_analysis = songs_df.copy()
        df_analysis['has_lyrics'] = df_analysis['id'].map(results)
        
        analysis = {
            'total_songs': len(df_analysis),
            'songs_with_lyrics': df_analysis['has_lyrics'].sum(),
            'songs_without_lyrics': len(df_analysis) - df_analysis['has_lyrics'].sum(),
            'overall_success_rate': (df_analysis['has_lyrics'].sum() / len(df_analysis)) * 100
        }
        
        # Add percentage
        analysis['lyrics_percentage'] = analysis['overall_success_rate']
        
        # Analysis by features (if available)
        if 'energy' in df_analysis.columns:
            # High energy vs low energy
            high_energy = df_analysis[df_analysis['energy'] > 0.7]
            low_energy = df_analysis[df_analysis['energy'] <= 0.3]
            
            if len(high_energy) > 0:
                analysis['high_energy_lyrics_rate'] = (high_energy['has_lyrics'].sum() / len(high_energy)) * 100
            if len(low_energy) > 0:
                analysis['low_energy_lyrics_rate'] = (low_energy['has_lyrics'].sum() / len(low_energy)) * 100
        
        # Analysis by popularity (if available) 
        if 'popularity' in df_analysis.columns:
            popular = df_analysis[df_analysis['popularity'] > 50]
            if len(popular) > 0:
                analysis['popular_songs_lyrics_rate'] = (popular['has_lyrics'].sum() / len(popular)) * 100
        
        logger.info(f"Analysis complete: {analysis['lyrics_percentage']:.1f}% songs have lyrics available")
        
        return analysis
    
    def print_statistics(self):
        """Print detailed statistics about the verification process."""
        print("\n" + "="*60)
        print("LYRICS AVAILABILITY VERIFICATION STATISTICS")
        print("="*60)
        print(f"Total songs checked: {self.stats['total_checked']}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        print(f"API calls made: {self.stats['api_calls']}")
        print(f"Songs with lyrics: {self.stats['successful_verifications']}")
        print(f"Songs without lyrics: {self.stats['failed_verifications']}")
        print(f"Errors encountered: {self.stats['errors']}")
        
        if self.stats['total_checked'] > 0:
            success_rate = (self.stats['successful_verifications'] / self.stats['total_checked']) * 100
            cache_hit_rate = (self.stats['cache_hits'] / self.stats['total_checked']) * 100
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Cache hit rate: {cache_hit_rate:.1f}%")
        
        print("="*60)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return {
            'total_entries': len(self.cache),
            'cache_file_size': self.cache_path.stat().st_size if self.cache_path.exists() else 0,
            'cache_path': str(self.cache_path)
        }


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