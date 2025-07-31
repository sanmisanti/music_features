#!/usr/bin/env python3
"""
🎯 HYBRID SELECTION PIPELINE EXECUTOR
=====================================
Simple executable script for running the complete hybrid selection pipeline.

This script serves as the main entry point for executing the hybrid selection
pipeline with lyrics verification from the reorganized architecture.

Usage:
    python scripts/run_hybrid_selection_pipeline.py [--target-size 10000] [--output-dir DIR] [--skip-analysis]

Examples:
    # Basic execution
    python scripts/run_hybrid_selection_pipeline.py
    
    # Custom target size
    python scripts/run_hybrid_selection_pipeline.py --target-size 5000
    
    # Custom output directory
    python scripts/run_hybrid_selection_pipeline.py --output-dir /path/to/results
    
    # Skip initial analysis (faster)
    python scripts/run_hybrid_selection_pipeline.py --skip-analysis
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point - delegates to the actual pipeline implementation."""
    try:
        # Import the actual pipeline implementation
        from exploratory_analysis.selection_pipeline.main_pipeline import main as pipeline_main
        
        # Display header
        print("🎯 HYBRID SELECTION PIPELINE WITH LYRICS VERIFICATION")
        print("=" * 60)
        print("📍 Architecture: Reorganized modular structure")
        print("🎵 Target: 80% songs with lyrics, 20% without")
        print("📊 Default size: 10,000 representative songs") 
        print("=" * 60)
        print()
        
        # Execute the main pipeline
        pipeline_main()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("🔍 Make sure you're running from the project root directory")
        print("📁 Expected structure: music_features/scripts/run_hybrid_selection_pipeline.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        print("📋 Check logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()