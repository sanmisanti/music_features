"""
Test Suite for Data Loading Module

Tests for data_loader and data_validator modules with the lyrics dataset.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Import modules to test
from exploratory_analysis.data_loading.data_loader import DataLoader, LoadResult
from exploratory_analysis.data_loading.data_validator import DataValidator, ValidationLevel
from exploratory_analysis.config.analysis_config import config, DATA_PATHS
from exploratory_analysis.config.features_config import CLUSTERING_FEATURES


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class with lyrics dataset"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
        self.lyrics_dataset_path = DATA_PATHS['lyrics_dataset']
        
    def test_lyrics_dataset_exists(self):
        """Test that the lyrics dataset file exists"""
        self.assertTrue(self.lyrics_dataset_path.exists(), 
                       f"Lyrics dataset not found: {self.lyrics_dataset_path}")
    
    def test_load_lyrics_dataset_basic(self):
        """Test basic loading of lyrics dataset"""
        result = self.loader.load_dataset('lyrics_dataset', sample_size=100, validate=False)
        
        self.assertIsInstance(result, LoadResult)
        self.assertTrue(result.success, f"Loading failed: {result.errors}")
        self.assertIsInstance(result.data, pd.DataFrame)
        self.assertGreater(len(result.data), 0, "No data loaded")
        self.assertLessEqual(len(result.data), 100, "Sample size not respected")
    
    def test_lyrics_dataset_structure(self):
        """Test dataset structure and columns"""
        result = self.loader.load_dataset('lyrics_dataset', sample_size=10, validate=False)
        
        self.assertTrue(result.success)
        df = result.data
        
        # Check expected columns
        expected_columns = ['id', 'name', 'artists', 'lyrics', 'track_popularity'] + CLUSTERING_FEATURES
        for col in expected_columns:
            if col in df.columns:  # Some columns might have different names
                continue
            # Check for similar column names
            similar_cols = [c for c in df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
            self.assertTrue(len(similar_cols) > 0, f"Column '{col}' or similar not found in dataset")
    
    def test_lyrics_column_handling(self):
        """Test that lyrics column is handled correctly"""
        result = self.loader.load_dataset('lyrics_dataset', sample_size=5, validate=False)
        
        self.assertTrue(result.success)
        df = result.data
        
        # Check if lyrics column exists (might be named differently)
        lyrics_cols = [col for col in df.columns if 'lyrics' in col.lower()]
        self.assertGreater(len(lyrics_cols), 0, "No lyrics column found")
        
        lyrics_col = lyrics_cols[0]
        
        # Check lyrics are strings and not empty
        self.assertTrue(df[lyrics_col].dtype == 'object', "Lyrics should be object/string type")
        non_null_lyrics = df[lyrics_col].dropna()
        if len(non_null_lyrics) > 0:
            self.assertTrue(all(isinstance(lyric, str) for lyric in non_null_lyrics), 
                           "All lyrics should be strings")
            self.assertTrue(all(len(lyric.strip()) > 0 for lyric in non_null_lyrics), 
                           "Lyrics should not be empty")
    
    def test_caret_separator_handling(self):
        """Test that caret separator is handled correctly"""
        result = self.loader.load_dataset('lyrics_dataset', sample_size=5, validate=False)
        
        self.assertTrue(result.success, f"Failed to load with caret separator: {result.errors}")
        
        # Check that we got reasonable number of columns
        expected_min_cols = 20  # Should have at least 20 columns
        self.assertGreater(len(result.data.columns), expected_min_cols, 
                          f"Too few columns ({len(result.data.columns)}), separator might be wrong")
    
    def test_decimal_point_handling(self):
        """Test that decimal points in numeric features are handled correctly"""
        result = self.loader.load_dataset('lyrics_dataset', sample_size=10, validate=False)
        
        self.assertTrue(result.success)
        df = result.data
        
        # Find numeric columns that should be floats
        numeric_features = ['danceability', 'energy', 'valence', 'acousticness', 'loudness', 'tempo']
        
        for feature in numeric_features:
            # Find column with this feature name (might have slight variations)
            matching_cols = [col for col in df.columns if feature in col.lower()]
            if matching_cols:
                col = matching_cols[0]
                if col in df.columns:
                    # Check if it's numeric
                    try:
                        numeric_values = pd.to_numeric(df[col], errors='coerce')
                        non_null_numeric = numeric_values.dropna()
                        if len(non_null_numeric) > 0:
                            self.assertTrue(len(non_null_numeric) > 0, 
                                          f"Feature {feature} should have numeric values")
                            # Check for reasonable ranges
                            if feature in ['danceability', 'energy', 'valence', 'acousticness']:
                                self.assertTrue(all(0 <= val <= 1 for val in non_null_numeric), 
                                              f"Feature {feature} should be in range [0,1]")
                    except Exception as e:
                        self.fail(f"Failed to process numeric feature {feature}: {e}")
    
    def test_load_statistics(self):
        """Test load statistics tracking"""
        initial_stats = self.loader.get_load_statistics()
        
        result = self.loader.load_dataset('lyrics_dataset', sample_size=50, validate=False)
        self.assertTrue(result.success)
        
        final_stats = self.loader.get_load_statistics()
        
        # Check that stats were updated
        self.assertGreater(final_stats['files_loaded'], initial_stats['files_loaded'])
        self.assertGreater(final_stats['total_rows'], 0)
        self.assertGreater(final_stats['memory_usage'], 0)
        self.assertGreater(final_stats['load_time'], 0)
    
    def test_sampling_functionality(self):
        """Test sampling with different sizes"""
        # Test small sample
        result_small = self.loader.load_dataset('lyrics_dataset', sample_size=10, validate=False)
        self.assertTrue(result_small.success)
        self.assertEqual(len(result_small.data), 10)
        
        # Test larger sample
        result_large = self.loader.load_dataset('lyrics_dataset', sample_size=100, validate=False)
        self.assertTrue(result_large.success)
        self.assertLessEqual(len(result_large.data), 100)
        
        # Results should be different (random sampling)
        if len(result_small.data) > 0 and len(result_large.data) > 0:
            # Check first IDs are different (random sampling)
            id_cols = [col for col in result_small.data.columns if 'id' in col.lower()]
            if id_cols:
                id_col = id_cols[0]
                small_ids = set(result_small.data[id_col])
                large_ids = set(result_large.data[id_col])
                # Large sample should contain some different songs (unless dataset is very small)
                intersection = large_ids.intersection(small_ids)
                if len(large_ids) > len(small_ids):
                    self.assertLess(len(intersection), len(large_ids), 
                                   "Large sample should have some different songs")
                else:
                    # If samples are similar size, just check we got valid data
                    self.assertGreater(len(small_ids), 0, "Should have valid IDs")


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class with lyrics dataset"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
        self.loader = DataLoader()
        
        # Load a small sample for testing
        result = self.loader.load_dataset('lyrics_dataset', sample_size=20, validate=False)
        self.assertTrue(result.success, "Failed to load test data")
        self.test_data = result.data
    
    def test_basic_validation(self):
        """Test basic validation level"""
        report = self.validator.validate_dataset(self.test_data, level=ValidationLevel.BASIC)
        
        self.assertIsNotNone(report)
        self.assertGreater(report.total_rows, 0)
        self.assertGreater(report.total_columns, 0)
        self.assertIsInstance(report.issues, list)
    
    def test_standard_validation(self):
        """Test standard validation level"""
        report = self.validator.validate_dataset(self.test_data, level=ValidationLevel.STANDARD)
        
        self.assertIsNotNone(report)
        self.assertTrue(hasattr(report, 'data_quality_score'))
        self.assertTrue(hasattr(report, 'feature_coverage'))
        
        # Should have more comprehensive checks than basic
        self.assertIsInstance(report.issues, list)
    
    def test_lyrics_column_validation(self):
        """Test validation of lyrics column specifically"""
        report = self.validator.validate_dataset(self.test_data, level=ValidationLevel.STANDARD)
        
        # Check if lyrics-related issues are detected appropriately
        lyrics_issues = [issue for issue in report.issues if 'lyrics' in issue.feature.lower()]
        
        # Should not have critical errors for lyrics (it's a text column)
        critical_lyrics_issues = [issue for issue in lyrics_issues 
                                 if issue.severity.value == 'critical']
        self.assertEqual(len(critical_lyrics_issues), 0, 
                        "Lyrics column should not have critical validation errors")
    
    def test_clustering_features_validation(self):
        """Test validation of clustering features"""
        report = self.validator.validate_dataset(self.test_data, level=ValidationLevel.STANDARD)
        
        # Check clustering features coverage
        clustering_feature_names = CLUSTERING_FEATURES
        detected_features = []
        
        for feature in clustering_feature_names:
            matching_cols = [col for col in self.test_data.columns if feature in col.lower()]
            if matching_cols:
                detected_features.append(feature)
        
        # Should detect at least some clustering features
        self.assertGreater(len(detected_features), 5, 
                          "Should detect most clustering features in dataset")
    
    def test_validation_report_structure(self):
        """Test that validation report has expected structure"""
        report = self.validator.validate_dataset(self.test_data, level=ValidationLevel.STANDARD)
        
        # Check required report attributes
        required_attrs = ['total_rows', 'total_columns', 'issues', 'data_quality_score']
        for attr in required_attrs:
            self.assertTrue(hasattr(report, attr), f"Report missing attribute: {attr}")
        
        # Check issues structure
        if report.issues:
            first_issue = report.issues[0]
            issue_attrs = ['feature', 'severity', 'issue_type', 'description', 'count']
            for attr in issue_attrs:
                self.assertTrue(hasattr(first_issue, attr), f"Issue missing attribute: {attr}")


class TestDataLoadingIntegration(unittest.TestCase):
    """Integration tests for data loading components"""
    
    def test_load_and_validate_pipeline(self):
        """Test complete load and validate pipeline"""
        loader = DataLoader()
        validator = DataValidator()
        
        # Load data
        load_result = loader.load_dataset('lyrics_dataset', sample_size=30, validate=True)
        self.assertTrue(load_result.success, f"Load failed: {load_result.errors}")
        
        # Additional validation
        validation_report = validator.validate_dataset(load_result.data, level=ValidationLevel.STANDARD)
        
        self.assertIsNotNone(validation_report)
        self.assertGreater(validation_report.total_rows, 0)
        
        # Check data quality
        if hasattr(validation_report, 'data_quality_score'):
            self.assertGreater(validation_report.data_quality_score, 0.5, 
                              "Data quality score should be reasonable")
    
    def test_configuration_compatibility(self):
        """Test that configuration works with lyrics dataset"""
        # Test current config settings
        self.assertEqual(config.data.separator, '^', "Separator should be caret")
        self.assertEqual(config.data.decimal, '.', "Decimal should be dot")
        self.assertEqual(config.data.encoding, 'utf-8', "Encoding should be UTF-8")
        
        # Test that these settings work
        loader = DataLoader()
        result = loader.load_dataset('lyrics_dataset', sample_size=5, validate=False)
        self.assertTrue(result.success, "Configuration should work with lyrics dataset")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)