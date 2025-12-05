#!/usr/bin/env python3
"""
Genius Lyrics Extractor

This script extracts lyrics from Genius.com for songs in the tracks_features_500.csv dataset.
Uses the lyricsgenius library with best practices for rate limiting and error handling.

Author: Music Features Analysis Project
"""

# ============================================================================
# CONFIGURACIÓN - Modifica tu token de Genius API aquí
# ============================================================================
GENIUS_ACCESS_TOKEN = "Kmryp8sRaJ6ZlwRdy_DaUTdR28OXGjEtJ29VikqhUO3eCnA_ovH5gLMajGIzv_qD"  # Reemplaza con tu token de Genius API
# Obtén tu token gratuito en: https://genius.com/api-clients
# ============================================================================

import os
import sys
import time
import logging
import pandas as pd
import json
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
from lyrics_database import LyricsDatabase

try:
    import lyricsgenius
except ImportError:
    print("Error: lyricsgenius library not found. Install with: pip install lyricsgenius")
    sys.exit(1)


class GeniusLyricsExtractor:
    """
    A robust lyrics extractor using the Genius API with comprehensive error handling,
    rate limiting, and progress tracking.
    """
    
    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize the Genius Lyrics Extractor.
        
        Args:
            access_token: Genius API token. If None, will look for GENIUS_ACCESS_TOKEN env var.
        """
        self.access_token = access_token or GENIUS_ACCESS_TOKEN
        if not self.access_token or self.access_token == "TU_TOKEN_AQUI":
            raise ValueError(
                "Genius API token required. Please modify GENIUS_ACCESS_TOKEN at the top of this file. "
                "Get your free token from: https://genius.com/api-clients"
            )
        
        # Initialize Genius client with rate limiting settings
        self.genius = lyricsgenius.Genius(
            self.access_token,
            sleep_time=1.0,  # Sleep 1 second between requests
            timeout=15,      # 15 second timeout
            retries=3,       # Retry failed requests 3 times
            remove_section_headers=True,  # Clean lyrics format
            skip_non_songs=True,          # Skip non-song results
            excluded_terms=["(Remix)", "(Live)", "(Instrumental)"],  # Skip remixes/live versions
        )
        
        # Configure logging
        self.setup_logging()
        
        # Stats tracking
        self.stats = {
            'total_songs': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'api_errors': 0,
            'network_errors': 0,
            'parsing_errors': 0,
            'skipped_songs': 0
        }
        
        # Cache for failed songs to avoid retry
        self.failed_songs = set()
        
        # Initialize SQLite database
        self.db = LyricsDatabase()
        
    def setup_logging(self):
        """Configure logging for the extraction process."""
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_dir / 'lyrics_extraction.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def clean_artist_name(self, artist_str: str) -> str:
        """
        Clean artist name string from the dataset format.
        
        Args:
            artist_str: Artist string in format "['Artist Name']" or similar
            
        Returns:
            Clean artist name
        """
        if pd.isna(artist_str):
            return ""
        
        # Remove brackets and quotes
        cleaned = str(artist_str).strip("[]'\"")
        
        # Handle multiple artists - take the first one
        if ',' in cleaned:
            cleaned = cleaned.split(',')[0].strip("'\" ")
        
        return cleaned
        
    def normalize_search_terms(self, song_name: str, artist_name: str) -> Tuple[str, str]:
        """
        Normalize song and artist names for better search results.
        
        Args:
            song_name: Original song name
            artist_name: Original artist name
            
        Returns:
            Tuple of normalized (song_name, artist_name)
        """
        # Remove common patterns that hurt search accuracy
        song_patterns = [
            r'\(feat\..*?\)', r'\(ft\..*?\)', r'\(featuring.*?\)',
            r'\(remix\)', r'\(live\)', r'\(acoustic\)', r'\(instrumental\)',
            r'\(radio edit\)', r'\(extended.*?\)', r'\(original.*?\)'
        ]
        
        normalized_song = song_name
        for pattern in song_patterns:
            normalized_song = re.sub(pattern, '', normalized_song, flags=re.IGNORECASE)
        
        normalized_song = normalized_song.strip()
        normalized_artist = artist_name.strip()
        
        return normalized_song, normalized_artist
    
    def normalize_accents(self, text: str) -> str:
        """
        Normalize accents and special characters for better matching.
        
        Args:
            text: Text to normalize
            
        Returns:
            Text with normalized accents
        """
        import unicodedata
        # Normalize to NFD (decomposed) and remove combining characters (accents)
        normalized = unicodedata.normalize('NFD', text)
        # Remove combining characters (accents)
        without_accents = ''.join(char for char in normalized 
                                 if unicodedata.category(char) != 'Mn')
        return without_accents.lower()
        
    def search_song_with_fallbacks(self, song_name: str, artist_name: str) -> Optional[Dict]:
        """
        Search for a song with multiple fallback strategies.
        
        Args:
            song_name: Name of the song
            artist_name: Name of the artist
            
        Returns:
            Song data if found, None otherwise
        """
        search_key = f"{artist_name} - {song_name}"
        
        if search_key in self.failed_songs:
            self.logger.debug(f"Skipping previously failed song: {search_key}")
            self.stats['skipped_songs'] += 1
            return None
        
        # Normalize search terms
        norm_song, norm_artist = self.normalize_search_terms(song_name, artist_name)
        
        # Try multiple search strategies
        search_strategies = [
            f"{norm_song} {norm_artist}",  # Song + Artist
            f"{norm_artist} {norm_song}",  # Artist + Song
            norm_song,                     # Song only
            f'"{norm_song}" {norm_artist}' # Quoted song + artist
        ]
        
        for i, search_term in enumerate(search_strategies):
            try:
                self.logger.debug(f"Search attempt {i+1} for '{search_key}': {search_term}")
                song = self.genius.search_song(search_term)
                
                if song:
                    # Verify it's the right song (basic matching)
                    if self._verify_song_match(song, norm_song, norm_artist):
                        self.logger.info(f"Found song: {song.title} by {song.primary_artist.name}")
                        return song
                    else:
                        self.logger.debug(f"Song match verification failed for: {song.title}")
                
                # Rate limiting delay between search attempts
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.warning(f"Search attempt {i+1} failed for '{search_key}': {str(e)}")
                if "429" in str(e):  # Rate limit error
                    self.logger.info("Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                elif "timeout" in str(e).lower():
                    self.stats['network_errors'] += 1
                else:
                    self.stats['api_errors'] += 1
                
                time.sleep(2)  # Wait before next attempt
        
        # Mark as failed to avoid future attempts
        self.failed_songs.add(search_key)
        return None
        
    def _verify_song_match(self, song, target_song: str, target_artist: str) -> bool:
        """
        Verify if the found song matches the target song and artist.
        
        Args:
            song: Genius song object
            target_song: Target song name
            target_artist: Target artist name
            
        Returns:
            True if it's a good match, False otherwise
        """
        if not song or not hasattr(song, 'title') or not hasattr(song, 'primary_artist'):
            return False
        
        # Normalize for comparison (including accents)
        found_title = self.normalize_accents(song.title.strip())
        found_artist = self.normalize_accents(song.primary_artist.name.strip())
        target_song_lower = self.normalize_accents(target_song.strip())
        target_artist_lower = self.normalize_accents(target_artist.strip())
        
        # Check if song titles are similar (allowing for some differences)
        title_match = (
            target_song_lower in found_title or 
            found_title in target_song_lower or
            self._calculate_similarity(found_title, target_song_lower) > 0.6  # Lowered threshold
        )
        
        # Check if artists are similar
        artist_match = (
            target_artist_lower in found_artist or 
            found_artist in target_artist_lower or
            self._calculate_similarity(found_artist, target_artist_lower) > 0.6  # Lowered threshold
        )
        
        return title_match and artist_match
        
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate simple similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        if not str1 or not str2:
            return 0.0
        
        # Simple Jaccard similarity with character-level n-grams
        def get_ngrams(text: str, n: int = 2) -> set:
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = get_ngrams(str1)
        ngrams2 = get_ngrams(str2)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
        
    def extract_lyrics_batch(self, df: pd.DataFrame, start_idx: int = 0, batch_size: int = 50) -> List[Dict]:
        """
        Extract lyrics for a batch of songs from the dataframe.
        
        Args:
            df: DataFrame with song data
            start_idx: Starting index for extraction
            batch_size: Number of songs to process in this batch
            
        Returns:
            List of dictionaries with song data and lyrics
        """
        results = []
        end_idx = min(start_idx + batch_size, len(df))
        
        self.logger.info(f"Starting batch extraction: songs {start_idx+1} to {end_idx}")
        
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            song_id = row['id']
            song_name = row['name']
            artist_name = self.clean_artist_name(row['artists'])
            
            self.stats['total_songs'] += 1
            
            self.logger.info(f"Processing {idx+1}/{len(df)}: {song_name} by {artist_name}")
            
            try:
                # Search for the song
                song = self.search_song_with_fallbacks(song_name, artist_name)
                
                if song and hasattr(song, 'lyrics') and song.lyrics:
                    # Successfully extracted lyrics
                    result = {
                        'spotify_id': song_id,
                        'song_name': song_name,
                        'artist_name': artist_name,
                        'genius_id': getattr(song, 'id', None),
                        'genius_title': getattr(song, 'title', None),
                        'genius_artist': getattr(song.primary_artist, 'name', None) if hasattr(song, 'primary_artist') else None,
                        'lyrics': song.lyrics,
                        'genius_url': getattr(song, 'url', None),
                        'extraction_status': 'success'
                    }
                    
                    results.append(result)
                    self.stats['successful_extractions'] += 1
                    self.logger.info(f"[SUCCESS] Successfully extracted lyrics for: {song_name}")
                    
                else:
                    # Song not found
                    result = {
                        'spotify_id': song_id,
                        'song_name': song_name,
                        'artist_name': artist_name,
                        'lyrics': None,
                        'extraction_status': 'not_found'
                    }
                    results.append(result)
                    self.stats['failed_extractions'] += 1
                    self.logger.warning(f"[NOT FOUND] Could not find lyrics for: {song_name} by {artist_name}")
                
            except Exception as e:
                # Handle extraction errors
                error_msg = str(e)
                self.logger.error(f"[ERROR] Error extracting lyrics for {song_name}: {error_msg}")
                
                result = {
                    'spotify_id': song_id,
                    'song_name': song_name,
                    'artist_name': artist_name,
                    'lyrics': None,
                    'extraction_status': 'error',
                    'error_message': error_msg
                }
                results.append(result)
                self.stats['failed_extractions'] += 1
                
                if "429" in error_msg:
                    self.logger.info("Rate limit encountered, sleeping for 60 seconds...")
                    time.sleep(60)
                
            # Progress update every 10 songs
            if (idx + 1) % 10 == 0:
                success_rate = (self.stats['successful_extractions'] / self.stats['total_songs']) * 100
                self.logger.info(f"Progress: {idx+1}/{len(df)} songs, Success rate: {success_rate:.1f}%")
            
            # Rate limiting delay between songs
            time.sleep(1.5)
        
        return results
        
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save extraction results to both SQLite database and CSV backup.
        
        Args:
            results: List of extraction results
            output_path: Path to save CSV backup
        """
        if not results:
            self.logger.warning("No results to save")
            return
        
        # Save to SQLite database (primary storage)
        try:
            successful, failed = self.db.insert_lyrics_batch(results)
            self.logger.info(f"Database insert: {successful} successful, {failed} failed")
        except Exception as e:
            self.logger.error(f"Database insert failed: {e}")
        
        # Save CSV backup
        try:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_path, index=False, encoding='utf-8')
            self.logger.info(f"CSV backup saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"CSV backup failed: {e}")
        
    def print_statistics(self):
        """Print extraction statistics."""
        print("\n" + "="*50)
        print("LYRICS EXTRACTION STATISTICS")
        print("="*50)
        print(f"Total songs processed: {self.stats['total_songs']}")
        print(f"Successful extractions: {self.stats['successful_extractions']}")
        print(f"Failed extractions: {self.stats['failed_extractions']}")
        print(f"Skipped songs: {self.stats['skipped_songs']}")
        print(f"API errors: {self.stats['api_errors']}")
        print(f"Network errors: {self.stats['network_errors']}")
        
        if self.stats['total_songs'] > 0:
            success_rate = (self.stats['successful_extractions'] / self.stats['total_songs']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print("="*50)


def find_resume_position(df: pd.DataFrame, db: LyricsDatabase) -> int:
    """
    Find the position in the DataFrame to resume extraction from.
    
    Args:
        df: DataFrame with song data
        db: Database instance
        
    Returns:
        Index position to start extraction from
    """
    # Check if there are any processed songs
    processed_ids = db.get_processed_song_ids()
    
    if not processed_ids:
        print("No previous extraction found. Starting from the beginning.")
        return 0
    
    # Find the last unprocessed song
    for idx, row in df.iterrows():
        song_id = row['id']
        if song_id not in processed_ids:
            print(f"Resuming extraction from position {idx + 1}/{len(df)}")
            print(f"Last processed songs: {len(processed_ids)}")
            print(f"Remaining songs: {len(df) - idx}")
            return idx
    
    # All songs have been processed
    print("All songs have been processed!")
    return len(df)


def main():
    """Main execution function."""
    # Configuration
    DATA_PATH = "../data/picked_data_0.csv"
    
    # Try absolute path if relative doesn't work
    if not os.path.exists(DATA_PATH):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(script_dir, "..", "data", "picked_data_0.csv")
    
    OUTPUT_PATH = "output/lyrics_extraction_results.csv"
    BATCH_SIZE = 100  # Increased for better performance with larger dataset
    
    print("Genius Lyrics Extractor - SQLite Edition")
    print("=" * 45)
    print("Processing 9,677 songs from selected dataset")
    print("Results will be stored in SQLite database")
    
    # Load the dataset
    try:
        df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
        print(f"Loaded {len(df)} songs from dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize extractor
    try:
        extractor = GeniusLyricsExtractor()
        print("Genius API client initialized successfully")
    except Exception as e:
        print(f"Error initializing Genius client: {e}")
        print("Make sure to set your GENIUS_ACCESS_TOKEN environment variable")
        print("Get your token from: https://genius.com/api-clients")
        return
    
    # Find resume position
    resume_idx = find_resume_position(df, extractor.db)
    
    if resume_idx >= len(df):
        print("Extraction complete! All songs have been processed.")
        extractor.db.print_statistics()
        return
    
    # Start extraction from resume position
    all_results = []
    
    try:
        # Process in batches starting from resume position
        for start_idx in range(resume_idx, len(df), BATCH_SIZE):
            batch_results = extractor.extract_lyrics_batch(df, start_idx, BATCH_SIZE)
            all_results.extend(batch_results)
            
            # Save intermediate results
            if batch_results:
                temp_output = f"output/temp_lyrics_batch_{start_idx}.csv"
                extractor.save_results(batch_results, temp_output)
                print(f"Intermediate results saved to: {temp_output}")
        
        # Save final results and show database statistics
        if all_results:
            extractor.save_results(all_results, OUTPUT_PATH)
            print(f"Final CSV backup saved to: {OUTPUT_PATH}")
            print(f"Primary data stored in SQLite database")
            
            # Show database statistics
            extractor.db.print_statistics()
        
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user")
        if all_results:
            interrupted_path = f"output/interrupted_lyrics_extraction_results.csv"
            extractor.save_results(all_results, interrupted_path)
            print(f"Partial results saved to: {interrupted_path}")
    
    except Exception as e:
        print(f"Unexpected error during extraction: {e}")
        if all_results:
            error_path = f"output/error_lyrics_extraction_results.csv"
            extractor.save_results(all_results, error_path)
    
    finally:
        # Print final statistics
        extractor.print_statistics()


if __name__ == "__main__":
    main()