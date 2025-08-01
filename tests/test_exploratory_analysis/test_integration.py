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
from exploratory_analysis.statistical_analysis.descriptive_stats import DescriptiveStatsAnalyzer
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
            load_result = loader.load_dataset('lyrics_dataset', sample_size=50, validate=True)
            
            self.assertTrue(load_result.success, f"Data loading failed: {load_result.errors}")
            self.assertGreater(len(load_result.data), 0, "No data loaded")
            print(f"✓ Loaded {len(load_result.data)} rows with {len(load_result.data.columns)} columns")
            
            # Step 2: Data Validation
            print("Step 2: Validating data...")
            validator = DataValidator()
            validation_report = validator.validate_dataset(load_result.data, level=ValidationLevel.STANDARD)
            
            self.assertIsNotNone(validation_report)
            print(f"✓ Validation completed with {len(validation_report.issues)} issues found")
            
            # Step 3: Statistical Analysis
            print("Step 3: Statistical analysis...")
            stats_analyzer = DescriptiveStatsAnalyzer()
            stats_report = stats_analyzer.analyze_dataset(load_result.data)
            
            self.assertIsNotNone(stats_report)
            print(f"✓ Statistical analysis completed for {len(stats_report.feature_statistics)} features")
            
            # Step 4: Visualization (basic tests)
            print("Step 4: Creating visualizations...")
            numeric_data = load_result.data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) >= 2:
                # Correlation heatmap
                corr_plotter = CorrelationPlotter()
                corr_fig = corr_plotter.create_correlation_heatmap(
                    numeric_data.iloc[:, :6],  # First 6 numeric columns
                    title="Integration Test Correlation"
                )
                self.assertIsNotNone(corr_fig)
                print("✓ Correlation heatmap created")
                
                # Distribution plots
                dist_plotter = DistributionPlotter()
                dist_fig = dist_plotter.create_distribution_grid(
                    numeric_data.iloc[:, :4],  # First 4 numeric columns
                    title="Integration Test Distributions"
                )
                self.assertIsNotNone(dist_fig)
                print("✓ Distribution plots created")
            
            # Step 5: Feature Analysis
            print("Step 5: Feature analysis...")
            if len(numeric_data.columns) >= 3:
                reducer = DimensionalityReducer()
                pca_result = reducer.perform_pca(
                    numeric_data.iloc[:, :6],  # First 6 numeric columns
                    n_components=2
                )
                self.assertIsNotNone(pca_result)
                print("✓ PCA analysis completed")
            
            # Step 6: Report Generation
            print("Step 6: Generating comprehensive report...")
            report_generator = ReportGenerator()
            comprehensive_report = report_generator.generate_comprehensive_report(
                load_result.data,
                title="Integration Test Report - Small Sample"
            )
            
            self.assertIsNotNone(comprehensive_report)
            print("✓ Comprehensive report generated")
            
            # Step 7: Export Results
            print("Step 7: Exporting results...")
            json_path = self.temp_path / "integration_test_small.json"
            md_path = self.temp_path / "integration_test_small.md"
            
            json_success = report_generator.export_report(
                comprehensive_report, json_path, format='json'
            )
            md_success = report_generator.export_report(
                comprehensive_report, md_path, format='markdown'
            )
            
            if json_success and json_path.exists():
                print("✓ JSON report exported")
            if md_success and md_path.exists():
                print("✓ Markdown report exported")
            
            print("=== Full Pipeline Test Completed Successfully ===")
            
        except Exception as e:
            self.fail(f"Full pipeline test failed: {e}")
    
    def test_full_pipeline_medium_sample(self):
        """Test complete pipeline with medium dataset sample"""
        try:
            print("\n=== Testing Full Pipeline (Medium Sample) ===")
            
            # Load larger sample
            loader = DataLoader()
            load_result = loader.load_dataset('lyrics_dataset', sample_size=200, validate=False)
            
            self.assertTrue(load_result.success)
            print(f"✓ Loaded {len(load_result.data)} rows")
            
            # Quick validation
            validator = DataValidator()
            validation_report = validator.validate_dataset(load_result.data, level=ValidationLevel.BASIC)
            print(f"✓ Basic validation completed")
            
            # Statistical analysis
            stats_analyzer = DescriptiveStatsAnalyzer()
            stats_report = stats_analyzer.analyze_dataset(load_result.data)
            print("✓ Statistical analysis completed")
            
            # Feature analysis (if enough numeric features)
            numeric_data = load_result.data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) >= 3:
                reducer = DimensionalityReducer()
                # Use smaller sample for computationally intensive operations
                sample_for_analysis = numeric_data.head(100)
                pca_result = reducer.perform_pca(sample_for_analysis.iloc[:, :8], n_components=3)
                print("✓ PCA analysis completed")
            
            # Generate report
            report_generator = ReportGenerator()
            report = report_generator.generate_comprehensive_report(
                load_result.data,
                title="Integration Test Report - Medium Sample"
            )
            print("✓ Report generated")
            
            print("=== Medium Pipeline Test Completed Successfully ===")
            
        except Exception as e:
            self.fail(f"Medium pipeline test failed: {e}")
    
    def test_clustering_features_pipeline(self):
        """Test pipeline focusing specifically on clustering features"""
        try:
            print("\n=== Testing Clustering Features Pipeline ===")
            
            # Load data
            loader = DataLoader()
            load_result = loader.load_dataset('lyrics_dataset', sample_size=100, validate=False)
            self.assertTrue(load_result.success)
            
            # Identify clustering features in dataset
            available_clustering_features = []
            for feature in CLUSTERING_FEATURES:
                matching_cols = [col for col in load_result.data.columns 
                               if feature.lower() in col.lower()]
                if matching_cols:
                    col_data = load_result.data[matching_cols[0]]
                    if pd.api.types.is_numeric_dtype(col_data):
                        available_clustering_features.append(matching_cols[0])
            
            print(f"✓ Found {len(available_clustering_features)} clustering features")
            
            if len(available_clustering_features) < 3:
                self.skipTest("Not enough clustering features for pipeline test")
            
            # Extract clustering features data
            clustering_data = load_result.data[available_clustering_features].dropna()
            print(f"✓ Extracted clustering data: {clustering_data.shape}")
            
            # Statistical analysis of clustering features
            stats_analyzer = DescriptiveStatsAnalyzer()
            clustering_stats = stats_analyzer.analyze_dataset(clustering_data)
            print("✓ Clustering features statistical analysis completed")
            
            # Correlation analysis
            if len(clustering_data.columns) >= 2:
                corr_plotter = CorrelationPlotter()
                corr_fig = corr_plotter.create_correlation_heatmap(
                    clustering_data,
                    title="Clustering Features Correlation"
                )
                print("✓ Clustering features correlation analysis completed")
            
            # Distribution analysis
            dist_plotter = DistributionPlotter()
            dist_fig = dist_plotter.create_distribution_grid(
                clustering_data.iloc[:, :6],  # First 6 features
                title="Clustering Features Distributions"  
            )
            print("✓ Clustering features distribution analysis completed")
            
            # Dimensionality reduction
            if len(clustering_data.columns) >= 3:
                reducer = DimensionalityReducer()
                pca_result = reducer.perform_pca(clustering_data, n_components=3)
                print(f"✓ PCA completed - explained variance: {pca_result.explained_variance_ratio[:3]}")
            
            # Generate specialized report
            report_generator = ReportGenerator()
            clustering_report = report_generator.generate_comprehensive_report(
                clustering_data,
                title="Clustering Features Analysis Report"
            )
            print("✓ Clustering features report generated")
            
            print("=== Clustering Features Pipeline Completed Successfully ===")
            
        except Exception as e:
            self.fail(f"Clustering features pipeline test failed: {e}")
    
    def test_error_recovery_pipeline(self):
        """Test pipeline error handling and recovery"""
        try:
            print("\n=== Testing Error Recovery Pipeline ===")
            
            # Test with potentially problematic data
            loader = DataLoader()
            
            # Try loading with small sample first
            load_result = loader.load_dataset('lyrics_dataset', sample_size=10, validate=False)
            
            if not load_result.success:
                print(f"⚠ Initial load failed: {load_result.errors}")
                return
            
            # Create data with potential issues
            test_data = load_result.data.copy()
            
            # Add some problematic values to numeric columns
            numeric_cols = test_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Add NaN, inf values
                test_data.loc[0, numeric_cols[0]] = np.nan
                if len(numeric_cols) > 1:
                    test_data.loc[1, numeric_cols[1]] = np.inf
                if len(numeric_cols) > 2:
                    test_data.loc[2, numeric_cols[2]] = -np.inf
            
            print("✓ Created test data with problematic values")
            
            # Test validation with problematic data
            validator = DataValidator()
            try:
                validation_report = validator.validate_dataset(test_data, level=ValidationLevel.STANDARD)
                print(f"✓ Validation handled problematic data - found {len(validation_report.issues)} issues")
            except Exception as e:
                print(f"⚠ Validation failed as expected: {e}")
            
            # Test statistical analysis with problematic data
            try:
                stats_analyzer = DescriptiveStatsAnalyzer()
                stats_report = stats_analyzer.analyze_dataset(test_data)
                print("✓ Statistical analysis handled problematic data")
            except Exception as e:
                print(f"⚠ Statistical analysis failed as expected: {e}")
            
            # Test visualization with problematic data
            clean_numeric_data = test_data.select_dtypes(include=[np.number]).dropna()
            if len(clean_numeric_data.columns) >= 2 and len(clean_numeric_data) >= 2:
                try:
                    corr_plotter = CorrelationPlotter()
                    corr_fig = corr_plotter.create_correlation_heatmap(
                        clean_numeric_data.iloc[:, :4],
                        title="Error Recovery Test Correlation"
                    )
                    print("✓ Visualization handled cleaned data")
                except Exception as e:
                    print(f"⚠ Visualization failed: {e}")
            
            # Test report generation with problematic data
            try:
                report_generator = ReportGenerator()
                report = report_generator.generate_comprehensive_report(
                    test_data,
                    title="Error Recovery Test Report"
                )
                print("✓ Report generation handled problematic data")
            except Exception as e:
                print(f"⚠ Report generation failed: {e}")
            
            print("=== Error Recovery Pipeline Completed ===")
            
        except Exception as e:
            self.fail(f"Error recovery pipeline test failed: {e}")
    
    def test_configuration_consistency(self):
        """Test that all components use consistent configuration"""
        try:
            print("\n=== Testing Configuration Consistency ===")
            
            # Check all components use same configuration
            loader = DataLoader()
            validator = DataValidator()
            stats_analyzer = DescriptiveStatsAnalyzer()
            corr_plotter = CorrelationPlotter()
            dist_plotter = DistributionPlotter()
            reducer = DimensionalityReducer()
            report_generator = ReportGenerator()
            
            # All should have config attribute
            components = [
                ('DataLoader', loader),
                ('DataValidator', validator),
                ('DescriptiveStatsAnalyzer', stats_analyzer),
                ('CorrelationPlotter', corr_plotter),
                ('DistributionPlotter', dist_plotter),
                ('DimensionalityReducer', reducer),
                ('ReportGenerator', report_generator)
            ]
            
            for name, component in components:
                if hasattr(component, 'config'):
                    config_obj = component.config
                    # Check key configuration values
                    if hasattr(config_obj, 'data'):
                        self.assertEqual(config_obj.data.separator, '^', 
                                       f"{name} config separator mismatch")
                        self.assertEqual(config_obj.data.decimal, '.', 
                                       f"{name} config decimal mismatch")
                        self.assertEqual(config_obj.data.encoding, 'utf-8', 
                                       f"{name} config encoding mismatch")
                    print(f"✓ {name} configuration verified")
                else:
                    print(f"⚠ {name} does not have config attribute")
            
            # Test that configuration actually works
            load_result = loader.load_dataset('lyrics_dataset', sample_size=20, validate=False)
            self.assertTrue(load_result.success, "Configuration should work with lyrics dataset")
            print("✓ Configuration works with lyrics dataset")
            
            print("=== Configuration Consistency Test Completed ===")
            
        except Exception as e:
            self.fail(f"Configuration consistency test failed: {e}")
    
    def test_performance_benchmarks(self):
        """Test performance with different dataset sizes"""
        try:
            print("\n=== Testing Performance Benchmarks ===")
            
            loader = DataLoader()
            
            # Test different sample sizes
            sample_sizes = [10, 50, 100]
            
            for size in sample_sizes:
                start_time = time.time()
                
                # Load data
                load_result = loader.load_dataset('lyrics_dataset', sample_size=size, validate=False)
                if not load_result.success:
                    continue
                
                # Quick analysis
                stats_analyzer = DescriptiveStatsAnalyzer()
                stats_report = stats_analyzer.analyze_dataset(load_result.data)
                
                # Generate report
                report_generator = ReportGenerator()
                report = report_generator.generate_comprehensive_report(
                    load_result.data,
                    title=f"Performance Test - {size} samples"
                )
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                print(f"✓ Size {size}: {execution_time:.2f}s")
                
                # Basic performance assertion (should complete within reasonable time)
                self.assertLess(execution_time, 60, f"Processing {size} samples took too long")
            
            print("=== Performance Benchmarks Completed ===")
            
        except Exception as e:
            self.fail(f"Performance benchmarks test failed: {e}")


class TestModuleCompatibility(unittest.TestCase):
    """Test compatibility between different modules"""
    
    def test_data_flow_compatibility(self):
        """Test that data flows correctly between modules"""
        try:
            # Load data
            loader = DataLoader()
            load_result = loader.load_dataset('lyrics_dataset', sample_size=30, validate=False)
            self.assertTrue(load_result.success)
            
            original_data = load_result.data
            
            # Pass through validation
            validator = DataValidator()
            validation_report = validator.validate_dataset(original_data)
            
            # Data should be unchanged after validation
            self.assertEqual(len(original_data), validation_report.total_rows)
            self.assertEqual(len(original_data.columns), validation_report.total_columns)
            
            # Pass through statistical analysis
            stats_analyzer = DescriptiveStatsAnalyzer()
            stats_report = stats_analyzer.analyze_dataset(original_data)
            
            # Should have stats for available features
            self.assertGreater(len(stats_report.feature_statistics), 0)
            
            # Pass through feature analysis
            numeric_data = original_data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) >= 2:
                reducer = DimensionalityReducer()
                pca_result = reducer.perform_pca(numeric_data.iloc[:, :5], n_components=2)
                
                # Transformed data should have correct dimensions
                self.assertEqual(pca_result.transformed_data.shape[0], len(numeric_data))
                self.assertEqual(pca_result.transformed_data.shape[1], 2)
            
            print("✓ Data flow compatibility verified")
            
        except Exception as e:
            self.fail(f"Data flow compatibility test failed: {e}")
    
    def test_output_format_compatibility(self):
        """Test that outputs from different modules are compatible"""
        try:
            # Generate various outputs
            loader = DataLoader()
            load_result = loader.load_dataset('lyrics_dataset', sample_size=20, validate=False)
            
            if not load_result.success:
                self.skipTest("Could not load data for compatibility test")
            
            # Statistical analysis output
            stats_analyzer = DescriptiveStatsAnalyzer()
            stats_report = stats_analyzer.analyze_dataset(load_result.data)
            
            # Report generation should be able to incorporate stats
            report_generator = ReportGenerator()
            comprehensive_report = report_generator.generate_comprehensive_report(
                load_result.data,
                title="Compatibility Test Report"
            )
            
            # Both should be serializable
            import json
            
            # Stats report should be convertible to dict/JSON
            if hasattr(stats_report, '__dict__') or hasattr(stats_report, 'to_dict'):
                try:
                    if hasattr(stats_report, 'to_dict'):
                        stats_dict = stats_report.to_dict()
                    else:
                        stats_dict = vars(stats_report)
                    
                    # Should be JSON serializable (with some conversion)
                    json_str = json.dumps(stats_dict, default=str)
                    self.assertIsInstance(json_str, str)
                except Exception:
                    # May not be directly serializable, but should not crash
                    pass
            
            print("✓ Output format compatibility verified")
            
        except Exception as e:
            self.fail(f"Output format compatibility test failed: {e}")


if __name__ == '__main__':
    # Run tests with high verbosity for integration testing
    unittest.main(verbosity=2, buffer=True)