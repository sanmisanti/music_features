"""
Test Runner for Exploratory Analysis Module

Comprehensive test runner that executes all test suites and provides
detailed reporting on the exploratory analysis system.
"""

import unittest
import sys
import time
from pathlib import Path
from io import StringIO
import importlib
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
test_modules = [
    'test_basic_functionality',    # Start with basic tests
    'test_data_loading',
    'test_statistical_analysis', 
    'test_visualization',
    'test_feature_analysis',
    'test_reporting',
    'test_integration'
]


class TestResult:
    """Container for test results"""
    def __init__(self, module_name, success, errors, failures, tests_run, time_taken):
        self.module_name = module_name
        self.success = success
        self.errors = errors
        self.failures = failures
        self.tests_run = tests_run
        self.time_taken = time_taken


class ExploratoryAnalysisTestRunner:
    """Comprehensive test runner for exploratory analysis"""
    
    def __init__(self):
        self.results = []
        self.total_start_time = None
        
    def run_all_tests(self, verbosity=1):
        """Run all test modules"""
        print("=" * 80)
        print("EXPLORATORY ANALYSIS MODULE - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Testing dataset compatibility with picked_data_lyrics.csv")
        print(f"Configuration: separator='^', decimal='.', encoding='utf-8'")
        print("=" * 80)
        
        self.total_start_time = time.time()
        
        for module_name in test_modules:
            self._run_module_tests(module_name, verbosity)
        
        self._print_summary()
    
    def _run_module_tests(self, module_name, verbosity):
        """Run tests for a specific module"""
        print(f"\n📋 TESTING MODULE: {module_name.upper()}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Import the test module with better error handling
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            
            # Try to import and handle different import issues
            test_module = None
            import_error = None
            
            try:
                test_module = importlib.import_module(module_name)
            except ImportError as ie:
                import_error = str(ie)
                # Try alternative import methods
                try:
                    # Try direct file execution
                    module_file = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
                    if os.path.exists(module_file):
                        spec = importlib.util.spec_from_file_location(module_name, module_file)
                        test_module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = test_module
                        spec.loader.exec_module(test_module)
                except Exception:
                    pass
            
            if test_module is None:
                raise ImportError(import_error or f"Could not import {module_name}")
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            if suite.countTestCases() == 0:
                print("⚠️  No tests found in module")
                test_result = TestResult(module_name, True, 0, 0, 0, time.time() - start_time)
                self.results.append(test_result)
                return
            
            # Capture output
            stream = StringIO()
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=verbosity,
                buffer=True
            )
            
            # Run tests
            result = runner.run(suite)
            
            # Calculate results
            end_time = time.time()
            time_taken = end_time - start_time
            
            success = result.wasSuccessful()
            errors = len(result.errors)
            failures = len(result.failures)
            tests_run = result.testsRun
            
            # Store results
            test_result = TestResult(
                module_name, success, errors, failures, tests_run, time_taken
            )
            self.results.append(test_result)
            
            # Print module summary
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status} - {tests_run} tests, {errors} errors, {failures} failures ({time_taken:.2f}s)")
            
            # Print details if there were issues
            if not success:
                output = stream.getvalue()
                if output.strip():
                    print("\n📄 Detailed Output:")
                    print(output)
                
                # Print specific error details
                if result.errors:
                    print(f"\n🔥 ERRORS ({len(result.errors)}):")
                    for i, (test, error) in enumerate(result.errors, 1):
                        print(f"  {i}. {test}: {error.strip()}")
                
                if result.failures:
                    print(f"\n❌ FAILURES ({len(result.failures)}):")
                    for i, (test, failure) in enumerate(result.failures, 1):
                        print(f"  {i}. {test}: {failure.strip()}")
            
            # Print test names if high verbosity
            if verbosity > 1:
                self._print_test_details(test_module)
                
        except ImportError as e:
            print(f"❌ IMPORT ERROR - Could not import {module_name}: {e}")
            test_result = TestResult(module_name, False, 1, 0, 0, 0)
            self.results.append(test_result)
            
        except Exception as e:
            print(f"❌ EXECUTION ERROR - {module_name}: {e}")
            test_result = TestResult(module_name, False, 1, 0, 0, 0)
            self.results.append(test_result)
    
    def _print_test_details(self, test_module):
        """Print details about tests in module"""
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        test_names = []
        for test_group in suite:
            if hasattr(test_group, '_tests'):
                for test in test_group._tests:
                    test_names.append(test._testMethodName)
        
        if test_names:
            print(f"   Tests: {', '.join(test_names[:5])}")
            if len(test_names) > 5:
                print(f"          ... and {len(test_names) - 5} more")
    
    def _print_summary(self):
        """Print comprehensive test summary"""
        total_time = time.time() - self.total_start_time
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        # Calculate totals
        total_tests = sum(r.tests_run for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_failures = sum(r.failures for r in self.results)
        successful_modules = sum(1 for r in self.results if r.success)
        total_modules = len(self.results)
        
        # Overall status
        overall_success = total_errors == 0 and total_failures == 0
        overall_status = "✅ ALL TESTS PASSED" if overall_success else "❌ SOME TESTS FAILED"
        
        print(f"🎯 OVERALL STATUS: {overall_status}")
        print(f"📈 MODULES: {successful_modules}/{total_modules} passed")
        print(f"🧪 TESTS: {total_tests} total, {total_errors} errors, {total_failures} failures")
        print(f"⏱️  TIME: {total_time:.2f} seconds")
        
        print("\n📋 MODULE BREAKDOWN:")
        print("-" * 80)
        
        for result in self.results:
            status_icon = "✅" if result.success else "❌"
            module_display = result.module_name.replace('_', ' ').title()
            
            print(f"{status_icon} {module_display:<25} "
                  f"{result.tests_run:>3} tests, "
                  f"{result.errors:>2} errors, "
                  f"{result.failures:>2} failures "
                  f"({result.time_taken:>5.2f}s)")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        print("-" * 40)
        
        if self.results:
            fastest = min(self.results, key=lambda r: r.time_taken)
            slowest = max(self.results, key=lambda r: r.time_taken)
            avg_time = sum(r.time_taken for r in self.results) / len(self.results)
            
            print(f"Fastest: {fastest.module_name} ({fastest.time_taken:.2f}s)")
            print(f"Slowest: {slowest.module_name} ({slowest.time_taken:.2f}s)")
            print(f"Average: {avg_time:.2f}s per module")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        
        if overall_success:
            print("✅ All tests passing! The exploratory analysis system is ready for use.")
            print("🚀 You can now proceed with full dataset analysis.")
        else:
            failed_modules = [r.module_name for r in self.results if not r.success]
            print(f"⚠️  Fix issues in: {', '.join(failed_modules)}")
            print("🔧 Review error details above and update code as needed.")
        
        if total_time > 30:
            print("⏰ Consider optimizing slow test modules for better development experience.")
        
        print("\n" + "=" * 80)
        print("TEST EXECUTION COMPLETED")
        print("=" * 80)
    
    def run_specific_module(self, module_name, verbosity=2):
        """Run tests for a specific module only"""
        print(f"Running tests for {module_name} only...")
        self.total_start_time = time.time()
        self._run_module_tests(module_name, verbosity)
        self._print_summary()
    
    def run_quick_tests(self):
        """Run a subset of critical tests for quick validation"""
        print("Running quick validation tests...")
        
        critical_modules = ['test_basic_functionality', 'test_data_loading']
        self.total_start_time = time.time()
        
        for module_name in critical_modules:
            if module_name in test_modules:
                self._run_module_tests(module_name, verbosity=1)
        
        self._print_summary()


def main():
    """Main test runner function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run exploratory analysis tests')
    parser.add_argument('--module', '-m', help='Run specific module tests only')
    parser.add_argument('--quick', '-q', action='store_true', help='Run quick validation tests')
    parser.add_argument('--verbose', '-v', action='count', default=1, help='Increase verbosity')
    
    args = parser.parse_args()
    
    runner = ExploratoryAnalysisTestRunner()
    
    try:
        if args.module:
            runner.run_specific_module(args.module, args.verbose)
        elif args.quick:
            runner.run_quick_tests()
        else:
            runner.run_all_tests(args.verbose)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        print("Partial results:")
        runner._print_summary()
    
    except Exception as e:
        print(f"\n\n❌ Test runner error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Return appropriate exit code
    overall_success = all(r.success for r in runner.results)
    return 0 if overall_success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)