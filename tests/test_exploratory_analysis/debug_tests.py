"""
Debug Test Runner - Shows detailed error information
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_single_test(test_module_name):
    """Run a single test module with full error details"""
    print(f"🔍 DEBUGGING: {test_module_name}")
    print("=" * 60)
    
    try:
        # Import test module
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        test_module = __import__(test_module_name)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        # Run with maximum verbosity
        runner = unittest.TextTestRunner(verbosity=2, buffer=False)
        result = runner.run(suite)
        
        print(f"\n📊 SUMMARY:")
        print(f"Tests run: {result.testsRun}")
        print(f"Errors: {len(result.errors)}")
        print(f"Failures: {len(result.failures)}")
        print(f"Success: {result.wasSuccessful()}")
        
        return result
        
    except Exception as e:
        print(f"❌ IMPORT ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug specific test module')
    parser.add_argument('module', help='Test module name (e.g., test_basic_functionality)')
    
    args = parser.parse_args()
    
    result = run_single_test(args.module)
    
    if result:
        exit_code = 0 if result.wasSuccessful() else 1
        sys.exit(exit_code)
    else:
        sys.exit(1)