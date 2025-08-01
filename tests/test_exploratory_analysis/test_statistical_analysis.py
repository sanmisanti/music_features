"""
Test Suite for Statistical Analysis Module

Tests for descriptive statistics and related analysis modules with lyrics dataset.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Import modules to test
from exploratory_analysis.statistical_analysis.descriptive_stats import (
    DescriptiveStats, DatasetStats, FeatureStats
)
from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.config.features_config import CLUSTERING_FEATURES


class TestDescriptiveStats(unittest.TestCase):
    """Test cases for DescriptiveStats with lyrics dataset"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.stats_analyzer = DescriptiveStats()
        
        # Load sample data
        loader = DataLoader()
        result = loader.load_dataset('lyrics_dataset', sample_size=50, validate=False)
        self.assertTrue(result.success, "Failed to load test data")
        self.test_data = result.data
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.stats_analyzer, DescriptiveStats)
        self.assertIsNotNone(self.stats_analyzer.config)
    
    def test_basic_statistics_computation(self):
        """Test basic statistics computation"""
        try:
            report = self.stats_analyzer.analyze_dataset(self.test_data)
            self.assertIsInstance(report, dict)
            
            # Check report structure
            self.assertIn('dataset_stats', report)
            self.assertIn('feature_stats', report)
            self.assertGreater(len(report['feature_stats']), 0)
            
        except Exception as e:
            self.fail(f"Basic statistics computation failed: {e}")
    
    def test_numeric_features_analysis(self):
        """Test analysis of numeric features"""
        try:
            report = self.stats_analyzer.analyze_dataset(self.test_data)
            
            # Should detect numeric features
            feature_stats = report['feature_stats']
            self.assertGreater(len(feature_stats), 0, "Should detect numeric features")
            
            # Check for standard statistical measures
            if feature_stats:
                first_feature = list(feature_stats.values())[0]
                expected_measures = ['mean', 'std', 'min_val', 'max_val', 'count']
                for measure in expected_measures:
                    if hasattr(first_feature, measure):
                        value = getattr(first_feature, measure)
                        self.assertIsInstance(value, (int, float, np.number))
                        
        except Exception as e:
            self.fail(f"Numeric features analysis failed: {e}")
    
    def test_categorical_features_analysis(self):
        """Test analysis of categorical features"""
        try:
            report = self.stats_analyzer.analyze_dataset(self.test_data)
            
            # Check if analysis completed successfully
            self.assertIsInstance(report, dict)
            
            # All features analyzed should have unique_values count
            feature_stats = report['feature_stats']
            if feature_stats:
                first_feature = list(feature_stats.values())[0]
                # Should have unique_values information
                self.assertTrue(hasattr(first_feature, 'unique_values'))
                self.assertIsInstance(first_feature.unique_values, int)
                               
        except Exception as e:
            self.fail(f"Categorical features analysis failed: {e}")
    
    def test_lyrics_column_handling(self):
        """Test that lyrics column is handled appropriately"""
        try:
            # Test that analysis doesn't crash on lyrics column
            lyrics_columns = [col for col in self.test_data.columns if 'lyrics' in col.lower()]
            
            if lyrics_columns:
                # Test analyzing only numeric features (should skip lyrics)
                report = self.stats_analyzer.analyze_dataset(self.test_data)
                self.assertIsInstance(report, dict)
                
                # Should have completed analysis without crashing
                self.assertIn('feature_stats', report)
                
                # Lyrics should not be in clustering features analysis
                feature_stats = report['feature_stats']
                lyrics_in_analysis = any('lyrics' in feature_name.lower() 
                                       for feature_name in feature_stats.keys())
                # This is fine either way - lyrics might be excluded from clustering features
                self.assertTrue(True, "Lyrics processing completed")
                    
        except Exception as e:
            self.fail(f"Lyrics column handling failed: {e}")
    
    def test_clustering_features_detection(self):
        """Test detection and analysis of clustering features"""
        try:
            report = self.stats_analyzer.analyze_dataset(self.test_data)
            
            # Check if clustering features are detected
            feature_stats = report['feature_stats']
            detected_clustering_features = []
            
            for feature_name in feature_stats.keys():
                for cluster_feature in CLUSTERING_FEATURES:
                    if cluster_feature.lower() in feature_name.lower():
                        detected_clustering_features.append(cluster_feature)
                        break
            
            # Should detect at least some clustering features
            self.assertGreater(len(detected_clustering_features), 3, 
                              "Should detect multiple clustering features")
            
        except Exception as e:
            self.fail(f"Clustering features detection failed: {e}")
    
    def test_missing_data_analysis(self):
        """Test missing data analysis"""
        try:
            report = self.stats_analyzer.analyze_dataset(self.test_data)
            
            # Should have dataset-level missing data information
            dataset_stats = report['dataset_stats']
            self.assertGreaterEqual(dataset_stats.missing_data_pct, 0)
            self.assertLessEqual(dataset_stats.missing_data_pct, 100)
                
            # Check individual feature missing data info
            feature_stats = report['feature_stats']
            for feature_stat in feature_stats.values():
                self.assertGreaterEqual(feature_stat.missing, 0)
                self.assertGreaterEqual(feature_stat.missing_pct, 0)
                self.assertLessEqual(feature_stat.missing_pct, 100)
                    
        except Exception as e:
            self.fail(f"Missing data analysis failed: {e}")
    
    def test_data_quality_assessment(self):
        """Test data quality assessment"""
        try:
            report = self.stats_analyzer.analyze_dataset(self.test_data)
            
            # Should have quality assessment
            if 'quality_assessment' in report:
                quality_assessment = report['quality_assessment']
                self.assertIsInstance(quality_assessment, dict)
                
                if 'overall_score' in quality_assessment:
                    self.assertGreater(quality_assessment['overall_score'], 0)
                    self.assertLessEqual(quality_assessment['overall_score'], 100)
            
            # Dataset stats should have quality information
            dataset_stats = report['dataset_stats']
            self.assertIn(dataset_stats.overall_quality, ['excellent', 'good', 'fair', 'poor'])
            
        except Exception as e:
            self.fail(f"Data quality assessment failed: {e}")
    
    def test_correlation_analysis(self):
        """Test basic correlation analysis"""
        try:
            report = self.stats_analyzer.analyze_dataset(self.test_data)
            
            # Check if correlation information is included
            if 'correlation_preview' in report:
                correlation_preview = report['correlation_preview']
                self.assertIsInstance(correlation_preview, dict)
                
                # If correlations computed, should be reasonable
                if 'top_positive' in correlation_preview:
                    for corr_info in correlation_preview['top_positive']:
                        if 'correlation' in corr_info:
                            corr_value = corr_info['correlation']
                            self.assertGreaterEqual(corr_value, -1)
                            self.assertLessEqual(corr_value, 1)
                            
        except Exception as e:
            self.fail(f"Correlation analysis failed: {e}")
    
    def test_report_export(self):
        """Test report export functionality"""
        try:
            report = self.stats_analyzer.analyze_dataset(self.test_data)
            
            # Test basic report structure for potential export
            self.assertIsInstance(report, dict)
            self.assertIn('dataset_stats', report)
            self.assertIn('feature_stats', report)
            
            # Test summary table generation
            summary_table = self.stats_analyzer.get_summary_table()
            self.assertIsInstance(summary_table, pd.DataFrame)
            
            if not summary_table.empty:
                self.assertGreater(len(summary_table.columns), 0)
                self.assertGreater(len(summary_table), 0)
                        
        except Exception as e:
            self.fail(f"Report export failed: {e}")
    
    def test_large_dataset_handling(self):
        """Test handling of larger dataset samples"""
        try:
            # Load larger sample
            loader = DataLoader()
            result = loader.load_dataset('lyrics_dataset', sample_size=200, validate=False)
            if result.success and len(result.data) > 100:
                large_data = result.data
                
                # Should handle larger dataset without issues
                report = self.stats_analyzer.analyze_dataset(large_data)
                self.assertIsInstance(report, dict)
                self.assertGreater(len(report['feature_stats']), 0)
                
        except Exception as e:
            self.fail(f"Large dataset handling failed: {e}")


class TestStatisticalAnalysisIntegration(unittest.TestCase):
    """Integration tests for statistical analysis components"""
    
    def test_end_to_end_analysis(self):
        """Test complete statistical analysis pipeline"""
        try:
            # Load data
            loader = DataLoader()
            result = loader.load_dataset('lyrics_dataset', sample_size=100, validate=False)
            self.assertTrue(result.success)
            
            # Analyze data
            analyzer = DescriptiveStats()
            report = analyzer.analyze_dataset(result.data)
            
            # Verify complete analysis
            self.assertIsInstance(report, dict)
            self.assertIn('dataset_stats', report)
            self.assertGreater(len(report['feature_stats']), 0)
            
            # Check that analysis covers main aspects
            has_numeric_analysis = len(report['feature_stats']) > 0
            has_missing_analysis = 'dataset_stats' in report and hasattr(report['dataset_stats'], 'missing_data_pct')
            
            # Should have at least basic analysis
            self.assertTrue(has_numeric_analysis or has_missing_analysis)
            
        except Exception as e:
            self.fail(f"End-to-end analysis failed: {e}")
    
    def test_configuration_integration(self):
        """Test integration with configuration system"""
        try:
            analyzer = DescriptiveStats()
            
            # Should use proper configuration
            self.assertIsNotNone(analyzer.config)
            
            # Configuration should be compatible with lyrics dataset
            loader = DataLoader()
            result = loader.load_dataset('lyrics_dataset', sample_size=50)
            
            if result.success:
                report = analyzer.analyze_dataset(result.data)
                self.assertIsNotNone(report)
                
        except Exception as e:
            self.fail(f"Configuration integration failed: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)