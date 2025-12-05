#!/usr/bin/env python3
"""
SQLite Database Manager for Lyrics Storage

This module handles SQLite database operations for storing and retrieving
song lyrics extracted from Genius.com. Optimized for ~10,000 songs with
efficient indexing and batch operations.

Author: Music Features Analysis Project
"""

import sqlite3
import os
import logging
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json


class LyricsDatabase:
    """
    SQLite database manager for lyrics storage and retrieval.
    
    Features:
    - Optimized schema with indexes for fast queries
    - Batch operations for efficient data insertion
    - Transaction management for data integrity
    - Built-in statistics and reporting
    """
    
    def __init__(self, db_path: str = "../data/lyrics.db"):
        """
        Initialize the lyrics database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = os.path.abspath(db_path)
        self.logger = self._setup_logging()
        
        # Ensure the database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize the database
        self._create_database()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the database operations."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_database(self):
        """Create the database and tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create the main lyrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS lyrics (
                        spotify_id TEXT PRIMARY KEY,
                        song_name TEXT NOT NULL,
                        artist_name TEXT NOT NULL,
                        lyrics TEXT,
                        genius_id INTEGER,
                        genius_title TEXT,
                        genius_artist TEXT,
                        genius_url TEXT,
                        word_count INTEGER,
                        language TEXT,
                        extraction_status TEXT NOT NULL DEFAULT 'pending',
                        extraction_date TIMESTAMP,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_extraction_status ON lyrics(extraction_status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_artist_name ON lyrics(artist_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_word_count ON lyrics(word_count)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_language ON lyrics(language)')
                
                # Create a metadata table for tracking extraction progress
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS extraction_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_songs INTEGER,
                        processed_songs INTEGER,
                        successful_extractions INTEGER,
                        failed_extractions INTEGER,
                        last_batch_processed INTEGER,
                        extraction_start_time TIMESTAMP,
                        extraction_end_time TIMESTAMP,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info(f"Database initialized at: {self.db_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to create database: {e}")
            raise
    
    def insert_lyrics_batch(self, lyrics_data: List[Dict]) -> Tuple[int, int]:
        """
        Insert a batch of lyrics data into the database.
        
        Args:
            lyrics_data: List of dictionaries with lyrics information
            
        Returns:
            Tuple of (successful_inserts, failed_inserts)
        """
        if not lyrics_data:
            return 0, 0
        
        successful = 0
        failed = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for record in lyrics_data:
                    try:
                        # Calculate word count if lyrics are present
                        word_count = len(record.get('lyrics', '').split()) if record.get('lyrics') else 0
                        
                        # Prepare the data for insertion
                        data = (
                            record['spotify_id'],
                            record['song_name'],
                            record['artist_name'],
                            record.get('lyrics'),
                            record.get('genius_id'),
                            record.get('genius_title'),
                            record.get('genius_artist'),
                            record.get('genius_url'),
                            word_count,
                            record.get('language'),
                            record.get('extraction_status', 'success'),
                            datetime.now().isoformat(),
                            record.get('error_message'),
                            datetime.now().isoformat()
                        )
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO lyrics (
                                spotify_id, song_name, artist_name, lyrics,
                                genius_id, genius_title, genius_artist, genius_url,
                                word_count, language, extraction_status, extraction_date,
                                error_message, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', data)
                        
                        successful += 1
                        
                    except Exception as e:
                        self.logger.error(f"Failed to insert record for {record.get('spotify_id', 'unknown')}: {e}")
                        failed += 1
                
                conn.commit()
                self.logger.info(f"Batch insert completed: {successful} successful, {failed} failed")
                
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            raise
        
        return successful, failed
    
    def get_lyrics_by_id(self, spotify_id: str) -> Optional[Dict]:
        """
        Retrieve lyrics for a specific song by Spotify ID.
        
        Args:
            spotify_id: Spotify track ID
            
        Returns:
            Dictionary with lyrics data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM lyrics WHERE spotify_id = ?', (spotify_id,))
                result = cursor.fetchone()
                
                if result:
                    return dict(result)
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve lyrics for {spotify_id}: {e}")
            return None
    
    def get_lyrics_by_status(self, status: str = 'success') -> pd.DataFrame:
        """
        Retrieve all lyrics with a specific extraction status.
        
        Args:
            status: Extraction status ('success', 'not_found', 'error', 'pending')
            
        Returns:
            DataFrame with lyrics data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM lyrics WHERE extraction_status = ?'
                return pd.read_sql_query(query, conn, params=(status,))
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve lyrics by status {status}: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the lyrics database.
        
        Returns:
            Dictionary with various statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute('SELECT COUNT(*) FROM lyrics')
                total_songs = cursor.fetchone()[0]
                
                cursor.execute('SELECT extraction_status, COUNT(*) FROM lyrics GROUP BY extraction_status')
                status_counts = dict(cursor.fetchall())
                
                # Word count statistics
                cursor.execute('SELECT AVG(word_count), MIN(word_count), MAX(word_count) FROM lyrics WHERE word_count > 0')
                word_stats = cursor.fetchone()
                
                # Language distribution
                cursor.execute('SELECT language, COUNT(*) FROM lyrics WHERE language IS NOT NULL GROUP BY language ORDER BY COUNT(*) DESC LIMIT 10')
                language_dist = dict(cursor.fetchall())
                
                # Top artists by lyrics count
                cursor.execute('SELECT artist_name, COUNT(*) FROM lyrics WHERE extraction_status = "success" GROUP BY artist_name ORDER BY COUNT(*) DESC LIMIT 10')
                top_artists = dict(cursor.fetchall())
                
                stats = {
                    'total_songs': total_songs,
                    'status_distribution': status_counts,
                    'word_count_stats': {
                        'average': round(word_stats[0] or 0, 2),
                        'minimum': word_stats[1] or 0,
                        'maximum': word_stats[2] or 0
                    },
                    'language_distribution': language_dist,
                    'top_artists': top_artists,
                    'success_rate': round((status_counts.get('success', 0) / total_songs * 100), 2) if total_songs > 0 else 0
                }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def export_to_csv(self, output_path: str, status: str = None) -> bool:
        """
        Export lyrics data to CSV format.
        
        Args:
            output_path: Path for the output CSV file
            status: Optional status filter ('success', 'not_found', 'error')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if status:
                    query = 'SELECT * FROM lyrics WHERE extraction_status = ?'
                    df = pd.read_sql_query(query, conn, params=(status,))
                else:
                    query = 'SELECT * FROM lyrics'
                    df = pd.read_sql_query(query, conn)
                
                df.to_csv(output_path, index=False, encoding='utf-8')
                self.logger.info(f"Exported {len(df)} records to {output_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export to CSV: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for the backup file
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Database backed up to: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False
    
    def vacuum_database(self) -> bool:
        """
        Optimize the database by running VACUUM command.
        
        Returns:
            True if vacuum successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('VACUUM')
                self.logger.info("Database vacuumed successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to vacuum database: {e}")
            return False
    
    def get_pending_songs(self, limit: int = None) -> List[str]:
        """
        Get list of songs that haven't been processed yet.
        
        Args:
            limit: Optional limit on number of results
            
        Returns:
            List of Spotify IDs for pending songs
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = 'SELECT spotify_id FROM lyrics WHERE extraction_status = "pending"'
                if limit:
                    query += f' LIMIT {limit}'
                
                cursor.execute(query)
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Failed to get pending songs: {e}")
            return []
    
    def get_last_processed_song(self) -> Optional[str]:
        """
        Get the Spotify ID of the last successfully processed song.
        This is used for resuming extraction from the correct position.
        
        Returns:
            Spotify ID of the last processed song, or None if no songs processed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the last song that was processed (any status except pending)
                cursor.execute('''
                    SELECT spotify_id FROM lyrics 
                    WHERE extraction_status != 'pending'
                    ORDER BY updated_at DESC 
                    LIMIT 1
                ''')
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            self.logger.error(f"Failed to get last processed song: {e}")
            return None
    
    def get_processed_song_ids(self) -> set:
        """
        Get set of all Spotify IDs that have been processed (any status except pending).
        Useful for checking which songs to skip during extraction.
        
        Returns:
            Set of processed Spotify IDs
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT spotify_id FROM lyrics WHERE extraction_status != "pending"')
                return {row[0] for row in cursor.fetchall()}
                
        except Exception as e:
            self.logger.error(f"Failed to get processed song IDs: {e}")
            return set()
    
    def delete_song_by_id(self, spotify_id: str) -> bool:
        """
        Delete a specific song from the database by Spotify ID.
        
        Args:
            spotify_id: Spotify ID of the song to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM lyrics WHERE spotify_id = ?', (spotify_id,))
                deleted_count = cursor.rowcount
                
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Deleted song with ID: {spotify_id}")
                    return True
                else:
                    self.logger.warning(f"No song found with ID: {spotify_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to delete song {spotify_id}: {e}")
            return False
    
    def clear_test_data(self) -> bool:
        """
        Clear test data from the database (songs with 'test' in spotify_id).
        
        Returns:
            True if clearing successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM lyrics WHERE spotify_id LIKE '%test%'")
                deleted_count = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(f"Deleted {deleted_count} test records")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to clear test data: {e}")
            return False

    def print_statistics(self):
        """Print formatted statistics about the database."""
        stats = self.get_statistics()
        
        if not stats:
            print("Unable to retrieve statistics")
            return
        
        print("\n" + "="*60)
        print("LYRICS DATABASE STATISTICS")
        print("="*60)
        
        print(f"Total songs: {stats['total_songs']}")
        print(f"Success rate: {stats['success_rate']}%")
        
        print(f"\nStatus Distribution:")
        for status, count in stats['status_distribution'].items():
            percentage = (count / stats['total_songs'] * 100) if stats['total_songs'] > 0 else 0
            print(f"  {status:12}: {count:5} ({percentage:5.1f}%)")
        
        print(f"\nWord Count Statistics:")
        print(f"  Average: {stats['word_count_stats']['average']} words")
        print(f"  Range: {stats['word_count_stats']['minimum']} - {stats['word_count_stats']['maximum']} words")
        
        if stats['language_distribution']:
            print(f"\nTop Languages:")
            for lang, count in list(stats['language_distribution'].items())[:5]:
                print(f"  {lang or 'Unknown':10}: {count}")
        
        if stats['top_artists']:
            print(f"\nTop Artists (by lyrics count):")
            for artist, count in list(stats['top_artists'].items())[:5]:
                print(f"  {artist[:30]:30}: {count}")
        
        print("="*60)


def main():
    """Test the database functionality."""
    print("Testing Lyrics Database")
    print("=" * 30)
    
    # Initialize database
    db = LyricsDatabase()
    
    # Test data
    test_data = [
        {
            'spotify_id': 'test123',
            'song_name': 'Test Song',
            'artist_name': 'Test Artist',
            'lyrics': 'This is a test song with some lyrics for testing purposes.',
            'extraction_status': 'success'
        }
    ]
    
    # Insert test data
    success, failed = db.insert_lyrics_batch(test_data)
    print(f"Inserted: {success} successful, {failed} failed")
    
    # Get statistics
    db.print_statistics()
    

if __name__ == "__main__":
    main()