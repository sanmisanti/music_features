#!/usr/bin/env python3
"""
🔧 CSV SEPARATOR FIXER
======================
Fixes CSV with lyrics by replacing column separators (,) with (@@) 
while preserving commas inside lyrics content.

Usage:
    python scripts/fix_csv_separators.py
"""

import pandas as pd
import re
from pathlib import Path

def intelligent_csv_fix(input_path: str, output_path: str, expected_columns: int = 25):
    """
    Intelligently fixes CSV separators by analyzing column structure.
    
    Args:
        input_path: Path to original CSV
        output_path: Path for fixed CSV  
        expected_columns: Expected number of columns (25 for this dataset)
    """
    print(f"🔧 Fixing CSV separators: {input_path}")
    print(f"📊 Expected columns: {expected_columns}")
    
    fixed_lines = []
    header_processed = False
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip('\n\r')
            
            if line_num == 1:  # Header line
                # Simple replacement for header (no lyrics)
                fixed_line = line.replace(',', '@@')
                fixed_lines.append(fixed_line)
                header_columns = fixed_line.split('@@')
                print(f"📋 Header columns: {len(header_columns)}")
                header_processed = True
                continue
            
            # Data lines - need intelligent processing
            parts = line.split(',')
            
            if len(parts) == expected_columns:
                # Perfect line, just replace
                fixed_line = line.replace(',', '@@')
            else:
                # Line has extra commas (likely in lyrics)
                fixed_line = fix_line_with_lyrics(line, expected_columns)
            
            fixed_lines.append(fixed_line)
            
            # Progress every 1000 lines
            if line_num % 1000 == 0:
                print(f"📈 Processed {line_num:,} lines...")
    
    # Write fixed CSV
    print(f"💾 Writing fixed CSV: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    print(f"✅ CSV fixed successfully!")
    print(f"📊 Total lines processed: {len(fixed_lines):,}")
    
    # Verify the fix
    verify_fixed_csv(output_path, expected_columns)

def fix_line_with_lyrics(line: str, expected_columns: int):
    """
    Fixes a line that has extra commas due to lyrics content.
    
    Strategy:
    1. Split by comma
    2. Identify lyrics field (field 4, after track_artist)
    3. Merge excess parts into lyrics field
    4. Reconstruct with @@ separators
    """
    parts = line.split(',')
    
    if len(parts) <= expected_columns:
        # No extra commas, simple replacement
        return line.replace(',', '@@')
    
    # Structure: track_id, track_name, track_artist, lyrics, ... (21 more fields)
    fixed_parts = []
    
    # First 3 fields are safe (track_id, track_name, track_artist)
    fixed_parts.extend(parts[:3])
    
    # Calculate how many extra parts we have
    extra_parts = len(parts) - expected_columns
    
    # Merge lyrics field (parts[3] through parts[3 + extra_parts])
    lyrics_parts = parts[3:3 + extra_parts + 1]
    merged_lyrics = ','.join(lyrics_parts)  # Keep commas in lyrics
    fixed_parts.append(merged_lyrics)
    
    # Add remaining fields (should be exactly 21 more)
    remaining_start = 3 + extra_parts + 1
    fixed_parts.extend(parts[remaining_start:])
    
    # Join with @@ separator
    return '@@'.join(fixed_parts)

def verify_fixed_csv(csv_path: str, expected_columns: int):
    """Verify the fixed CSV has correct structure."""
    print(f"🔍 Verifying fixed CSV...")
    
    line_count = 0
    correct_columns = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_count += 1
            parts = line.rstrip().split('@@')
            
            if len(parts) == expected_columns:
                correct_columns += 1
            elif line_num <= 5:  # Show first few problematic lines
                print(f"⚠️  Line {line_num} has {len(parts)} columns (expected {expected_columns})")
    
    success_rate = (correct_columns / line_count) * 100
    print(f"📊 Verification Results:")
    print(f"   Total lines: {line_count:,}")
    print(f"   Correct columns: {correct_columns:,}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if success_rate >= 99:
        print(f"✅ CSV structure fixed successfully!")
    else:
        print(f"⚠️  Some lines still have issues")

def test_pandas_loading(csv_path: str):
    """Test loading the fixed CSV with pandas."""
    print(f"🐼 Testing pandas loading...")
    
    try:
        # Try loading with @@ separator
        df = pd.read_csv(csv_path, sep='@@', encoding='utf-8')
        print(f"✅ Pandas loading successful!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns[:5])}... (showing first 5)")
        
        # Check for lyrics content
        if 'lyrics' in df.columns:
            lyrics_with_content = df['lyrics'].notna().sum()
            print(f"🎵 Songs with lyrics: {lyrics_with_content:,}")
        
        return df
        
    except Exception as e:
        print(f"❌ Pandas loading failed: {e}")
        return None

def main():
    """Main execution."""
    print("🎵 SPOTIFY SONGS WITH LYRICS - CSV FIXER")
    print("=" * 50)
    
    # Paths
    input_path = "../data/with_lyrics/spotify_songs.csv"
    output_path = "../data/with_lyrics/spotify_songs_fixed.csv"
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Fix CSV separators
    intelligent_csv_fix(input_path, output_path, expected_columns=25)
    
    # Test pandas loading
    df = test_pandas_loading(output_path)
    
    if df is not None:
        print(f"\n🎯 READY FOR REPRESENTATIVE SELECTION!")
        print(f"📊 Available songs: {len(df):,}")
        print(f"🎵 All songs have verified lyrics")
        print(f"📁 Fixed dataset: {output_path}")

if __name__ == "__main__":
    main()