#!/usr/bin/env python3
"""
CSV to Parquet Converter for Music Data

This script converts the tracks_features_500.csv file to Parquet format,
preparing the structure to include lyrics data in the future.

Author: Music Features Analysis Project
"""

import os
import sys
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_music_data(csv_path: str) -> pd.DataFrame:
    """
    Load music data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with music data
    """
    try:
        # Try to load with semicolon separator and comma decimal (cleaned data format)
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        logger.info(f"Successfully loaded {len(df)} tracks with semicolon separator")
        return df
    except Exception as e:
        logger.warning(f"Failed to load with semicolon separator: {e}")
        try:
            # Fallback to standard CSV format
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully loaded {len(df)} tracks with comma separator")
            return df
        except Exception as e2:
            logger.error(f"Failed to load CSV file: {e2}")
            raise

def prepare_complete_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the complete data structure with placeholders for lyrics.
    
    Args:
        df: Original music data DataFrame
        
    Returns:
        DataFrame with lyrics columns added
    """
    # Make a copy to avoid modifying the original
    df_complete = df.copy()
    
    # Add lyrics-related columns with default values
    df_complete['lyrics'] = None  # Will contain the actual lyrics text
    df_complete['genius_id'] = None  # Genius song ID
    df_complete['genius_title'] = None  # Title from Genius
    df_complete['genius_artist'] = None  # Artist from Genius
    df_complete['genius_url'] = None  # Genius song URL
    df_complete['lyrics_status'] = 'pending'  # Status: pending, success, not_found, error
    df_complete['lyrics_extraction_date'] = None  # When lyrics were extracted
    
    logger.info(f"Prepared structure with {len(df_complete.columns)} columns")
    logger.info(f"Music features columns: {len(df.columns)}")
    logger.info(f"Lyrics columns added: 7")
    
    return df_complete

def save_to_parquet(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to Parquet format with optimal settings.
    
    Args:
        df: DataFrame to save
        output_path: Path where to save the Parquet file
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with compression for optimal storage
        df.to_parquet(
            output_path,
            engine='pyarrow',  # Use PyArrow for better performance
            compression='snappy',  # Good balance of speed and compression
            index=False  # Don't save the pandas index
        )
        
        # Get file size for logging
        file_size = os.path.getsize(output_path)
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Successfully saved to: {output_path}")
        logger.info(f"File size: {file_size_mb:.2f} MB")
        logger.info(f"Compression ratio: {len(df) / file_size_mb:.0f} rows per MB")
        
    except Exception as e:
        logger.error(f"Failed to save Parquet file: {e}")
        raise

def display_data_info(df: pd.DataFrame):
    """
    Display information about the DataFrame structure.
    
    Args:
        df: DataFrame to analyze
    """
    print("\n" + "="*60)
    print("DATA STRUCTURE INFORMATION")
    print("="*60)
    
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    print("\nColumn Information:")
    print("-" * 40)
    
    # Separate music features from lyrics columns
    music_columns = [col for col in df.columns if not col.startswith(('lyrics', 'genius'))]
    lyrics_columns = [col for col in df.columns if col.startswith(('lyrics', 'genius'))]
    
    print(f"\nMusic Features ({len(music_columns)} columns):")
    for i, col in enumerate(music_columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nLyrics Structure ({len(lyrics_columns)} columns):")
    for i, col in enumerate(lyrics_columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Show sample of data
    print(f"\nSample data (first 3 rows):")
    print("-" * 40)
    sample_cols = ['id', 'name', 'artists', 'danceability', 'energy', 'lyrics_status']
    if all(col in df.columns for col in sample_cols):
        print(df[sample_cols].head(3).to_string(index=False))
    
    print("="*60)

def main():
    """Main execution function."""
    print("CSV to Parquet Converter for Music Data")
    print("=" * 40)
    
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(script_dir, "..", "data", "cleaned_data", "tracks_features_500.csv")
    OUTPUT_PATH = os.path.join(script_dir, "output", "complete_music_data.parquet")
    
    # Check if input file exists
    if not os.path.exists(CSV_PATH):
        logger.error(f"Input CSV file not found: {CSV_PATH}")
        logger.info("Please ensure the tracks_features_500.csv file exists in the data/cleaned_data/ directory")
        return
    
    try:
        # Step 1: Load the original music data
        logger.info("Loading music data from CSV...")
        df_music = load_music_data(CSV_PATH)
        
        # Step 2: Prepare complete structure with lyrics placeholders
        logger.info("Preparing complete data structure...")
        df_complete = prepare_complete_structure(df_music)
        
        # Step 3: Save to Parquet format
        logger.info("Saving to Parquet format...")
        save_to_parquet(df_complete, OUTPUT_PATH)
        
        # Step 4: Display information about the created structure
        display_data_info(df_complete)
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"📁 Output file: {OUTPUT_PATH}")
        print(f"🎵 Ready for {len(df_complete)} songs")
        print(f"📝 Structure prepared for lyrics integration")
        
        # Usage instructions
        print(f"\n📋 Next steps:")
        print(f"1. Use the lyrics extractor to populate the lyrics columns")
        print(f"2. Load the data with: pd.read_parquet('{OUTPUT_PATH}')")
        print(f"3. Filter by lyrics_status for analysis")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        print(f"\n❌ Error during conversion: {e}")
        return

if __name__ == "__main__":
    main()