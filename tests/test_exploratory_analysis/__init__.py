"""
Test Suite for Exploratory Analysis Module

This package contains comprehensive tests for the exploratory analysis system,
including data loading, statistical analysis, visualization, and reporting.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_PATH = project_root / "data" / "final_data" / "picked_data_lyrics.csv"
TEST_OUTPUT_PATH = project_root / "tests" / "test_outputs"

# Ensure test output directory exists
TEST_OUTPUT_PATH.mkdir(exist_ok=True)