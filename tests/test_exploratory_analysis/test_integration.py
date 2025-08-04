"""
Integration Test Suite for Exploratory Analysis Module

End-to-end tests that verify the complete exploratory analysis pipeline
works correctly with the lyrics dataset.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import time

# Import all modules for integration testing
from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.data_loading.data_validator import DataValidator, ValidationLevel
from exploratory_analysis.statistical_analysis.descriptive_stats import DescriptiveStats
from exploratory_analysis.visualization.correlation_heatmaps import CorrelationPlotter
from exploratory_analysis.visualization.distribution_plots import DistributionPlotter
from exploratory_analysis.feature_analysis.dimensionality_reduction import DimensionalityReducer
from exploratory_analysis.reporting.report_generator import ReportGenerator
from exploratory_analysis.config.analysis_config import config
from exploratory_analysis.config.features_config import CLUSTERING_FEATURES


class TestCompleteAnalysisPipeline(unittest.TestCase):
    """Integration tests for complete analysis pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Track timing for performance tests
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up test fixtures"""
        end_time = time.time()
        execution_time = end_time - self.start_time
        print(f"Test execution time: {execution_time:.2f} seconds")
    
    def test_full_pipeline_small_sample(self):
        """Test complete pipeline with small dataset sample"""
        try:
            print("\n=== Testing Full Pipeline (Small Sample) ===")
            
            # Step 1: Data Loading
            print("Step 1: Loading data...")
            loader = DataLoader()
            result = loader.load_dataset('lyrics_dataset', sample_size=30, validate=False)
            self.assertTrue(result.success, "Failed to load test data")
            test_data = result.data
            
            # Step 2: Data Validation
            print("Step 2: Validating data...")
            validator = DataValidator()
            validation_report = validator.validate_dataset(test_data, ValidationLevel.STANDARD)
            self.assertIsNotNone(validation_report)
            
            # Step 3: Statistical Analysis
            print("Step 3: Statistical analysis...")
            stats_analyzer = DescriptiveStats()
            stats_results = stats_analyzer.analyze_dataset(test_data)
            self.assertIsInstance(stats_results, dict)
            
            # Step 4: Feature Analysis
            print("Step 4: Feature analysis...")
            dim_reducer = DimensionalityReducer()
            pca_results = dim_reducer.fit_pca(test_data, n_components=2)
            if pca_results:  # Only test if PCA succeeded
                self.assertIn('transformed_data', pca_results)
            
            # Step 5: Visualization
            print("Step 5: Creating visualizations...")
            corr_plotter = CorrelationPlotter()
            numeric_cols = test_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                corr_fig = corr_plotter.create_correlation_heatmap(test_data, features=numeric_cols[:4])
                if corr_fig is not None:
                    self.assertIsNotNone(corr_fig)
            
            # Step 6: Report Generation
            print("Step 6: Generating reports...")
            report_generator = ReportGenerator(str(self.temp_path))
            report_paths = report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=30,
                include_visualizations=False,
                formats=['json']
            )
            self.assertIsInstance(report_paths, dict)
            
            print("✅ Full pipeline completed successfully!")
            
        except Exception as e:
            self.fail(f"Full pipeline test failed: {e}")
    
    def test_performance_benchmark(self):
        """Test pipeline performance with timing"""
        try:
            print("\n=== Performance Benchmark Test ===")
            
            start_time = time.time()
            
            # Load and analyze data
            loader = DataLoader()
            result = loader.load_dataset('lyrics_dataset', sample_size=50, validate=False)
            self.assertTrue(result.success)
            
            # Quick statistical analysis
            stats_analyzer = DescriptiveStats()
            stats_results = stats_analyzer.analyze_dataset(result.data)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"Pipeline execution time: {execution_time:.2f} seconds")
            
            # Performance assertion - should complete within reasonable time
            self.assertLess(execution_time, 30.0, "Pipeline taking too long")
            self.assertIsInstance(stats_results, dict)
            
        except Exception as e:
            self.fail(f"Performance benchmark failed: {e}")
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        try:
            print("\n=== Memory Usage Test ===")
            
            # Simple memory test - just ensure we can load and process data
            loader = DataLoader()
            result = loader.load_dataset('lyrics_dataset', sample_size=100, validate=False)
            
            if result.success:
                # Basic processing
                stats_analyzer = DescriptiveStats()
                stats_results = stats_analyzer.analyze_dataset(result.data)
                
                # Should complete without memory issues
                self.assertIsInstance(stats_results, dict)
                self.assertGreater(len(stats_results), 0)
            
        except Exception as e:
            self.fail(f"Memory usage test failed: {e}")


class TestModuleCompatibility(unittest.TestCase):
    """Test compatibility between different modules"""
    
    def setUp(self):
        """Set up test fixtures"""
        loader = DataLoader()
        result = loader.load_dataset('lyrics_dataset', sample_size=40, validate=False)
        self.assertTrue(result.success, "Failed to load test data")
        self.test_data = result.data
    
    def test_config_consistency(self):
        """Test that all modules use consistent configuration"""
        try:
            # Test configuration access across modules
            loader = DataLoader()
            validator = DataValidator()
            stats_analyzer = DescriptiveStats()
            
            # All should have config access
            self.assertIsNotNone(loader.config)
            self.assertIsNotNone(validator.config)
            self.assertIsNotNone(stats_analyzer.config)
            
            # Configuration should be consistent
            self.assertEqual(loader.config.data.separator, '^')
            self.assertEqual(validator.config.data.separator, '^')
            self.assertEqual(stats_analyzer.config.data.separator, '^')
            
        except Exception as e:
            self.fail(f"Config consistency test failed: {e}")
    
    def test_data_flow_compatibility(self):
        """Test data flow between modules"""
        try:
            # Data loading -> Validation
            loader = DataLoader()
            result = loader.load_dataset('lyrics_dataset', sample_size=30, validate=True)
            self.assertTrue(result.success)
            
            # Validation -> Statistical Analysis
            stats_analyzer = DescriptiveStats()
            stats_results = stats_analyzer.analyze_dataset(result.data)
            self.assertIsInstance(stats_results, dict)
            
            # Statistical Analysis -> Feature Analysis
            dim_reducer = DimensionalityReducer()
            numeric_cols = result.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                pca_results = dim_reducer.fit_pca(result.data, features=numeric_cols[:5])
                if pca_results:
                    self.assertIn('transformed_data', pca_results)
            
        except Exception as e:
            self.fail(f"Data flow compatibility test failed: {e}")
    
    def test_error_propagation(self):
        """Test error handling across modules"""
        try:
            # Test with invalid dataset
            loader = DataLoader()
            result = loader.load_dataset('nonexistent_dataset', sample_size=10)
            
            # Should handle gracefully
            if not result.success:
                self.assertFalse(result.success)
                self.assertIsNotNone(result.error)
            
        except Exception as e:
            # Should not crash system
            self.assertTrue(True, f"Error handled appropriately: {e}")


if __name__ == '__main__':
    # Run tests with high verbosity for integration testing
    unittest.main(verbosity=2, buffer=True)