#!/usr/bin/env python3
"""
Clean Test Data Script

This script removes test data from the lyrics database before starting
the actual extraction process.

Author: Music Features Analysis Project
"""

from lyrics_database import LyricsDatabase

def main():
    """Clean test data from the database."""
    print("Cleaning test data from lyrics database...")
    print("=" * 45)
    
    # Initialize database
    db = LyricsDatabase()
    
    # Show current statistics
    print("BEFORE CLEANING:")
    db.print_statistics()
    
    # Clear test data
    success = db.clear_test_data()
    
    if success:
        print("\n✅ Test data cleared successfully!")
    else:
        print("\n❌ Failed to clear test data")
        return
    
    # Show updated statistics
    print("\nAFTER CLEANING:")
    db.print_statistics()
    
    print("\n🎵 Database is now ready for lyrics extraction!")

if __name__ == "__main__":
    main()