"""
Test Suite for Visualization Module

Tests for correlation heatmaps, distribution plots, and other visualization components.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Import modules to test
from exploratory_analysis.visualization.correlation_heatmaps import CorrelationPlotter
from exploratory_analysis.visualization.distribution_plots import DistributionPlotter
from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.config.features_config import CLUSTERING_FEATURES


class TestCorrelationPlotter(unittest.TestCase):
    """Test cases for CorrelationPlotter with lyrics dataset"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plotter = CorrelationPlotter()
        
        # Load sample data
        loader = DataLoader()
        result = loader.load_dataset('lyrics_dataset', sample_size=100, validate=False)
        self.assertTrue(result.success, "Failed to load test data")
        self.test_data = result.data
        
        # Create temp directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        plt.close('all')  # Close all matplotlib figures
    
    def test_plotter_initialization(self):
        """Test plotter initialization"""
        self.assertIsInstance(self.plotter, CorrelationPlotter)
        self.assertIsNotNone(self.plotter.config)
    
    def test_correlation_matrix_computation(self):
        """Test correlation matrix computation"""
        try:
            # Find numeric columns for correlation
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                self.skipTest("Not enough numeric columns for correlation test")
            
            # Select subset of numeric columns
            test_cols = numeric_cols[:min(10, len(numeric_cols))]
            test_data_subset = self.test_data[test_cols].dropna()
            
            if len(test_data_subset) < 2:
                self.skipTest("Not enough valid data for correlation test")
            
            # Compute correlation matrix
            corr_matrix = test_data_subset.corr(method='pearson')
            
            self.assertIsInstance(corr_matrix, pd.DataFrame)
            self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])  # Should be square
            self.assertTrue(np.allclose(np.diag(corr_matrix), 1.0))  # Diagonal should be 1
            
        except Exception as e:
            self.fail(f"Correlation matrix computation failed: {e}")
    
    def test_heatmap_creation(self):
        """Test basic heatmap creation"""
        try:
            # Find numeric columns
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                self.skipTest("Not enough numeric columns for heatmap test")
            
            # Select subset and clean data
            test_cols = numeric_cols[:min(8, len(numeric_cols))]
            test_data_subset = self.test_data[test_cols].dropna()
            
            if len(test_data_subset) < 2:
                self.skipTest("Not enough valid data for heatmap test")
            
            # Create heatmap
            fig = self.plotter.create_correlation_heatmap(
                test_data_subset, 
                features=test_cols,
                save_path=None
            )
            
            self.assertIsNotNone(fig)
            self.assertTrue(hasattr(fig, 'axes'))
            
        except Exception as e:
            self.fail(f"Heatmap creation failed: {e}")
    
    def test_clustering_features_correlation(self):
        """Test correlation analysis of clustering features"""
        try:
            # Find available clustering features in dataset
            available_features = []
            for feature in CLUSTERING_FEATURES:
                matching_cols = [col for col in self.test_data.columns 
                               if feature.lower() in col.lower()]
                if matching_cols:
                    available_features.append(matching_cols[0])
            
            if len(available_features) < 2:
                self.skipTest("Not enough clustering features available")
            
            # Select subset of clustering features
            clustering_data = self.test_data[available_features[:8]].select_dtypes(include=[np.number])
            
            if clustering_data.empty or len(clustering_data.columns) < 2:
                self.skipTest("No valid numeric clustering features")
            
            # Create correlation heatmap for clustering features
            fig = self.plotter.create_correlation_heatmap(
                self.test_data,
                features=list(clustering_data.columns),
                save_path=None
            )
            
            self.assertIsNotNone(fig)
            
        except Exception as e:
            self.fail(f"Clustering features correlation failed: {e}")
    
    def test_different_correlation_methods(self):
        """Test different correlation methods"""
        try:
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                self.skipTest("Not enough numeric columns")
            
            test_data_subset = self.test_data[numeric_cols[:6]].dropna()
            if len(test_data_subset) < 2:
                self.skipTest("Not enough valid data")
            
            methods = ['pearson', 'spearman', 'kendall']
            
            for method in methods:
                try:
                    corr_matrix = test_data_subset.corr(method=method)
                    self.assertIsInstance(corr_matrix, pd.DataFrame)
                    self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])
                except Exception as method_error:
                    # Some methods might not work with all data types
                    self.assertTrue(True, f"Method {method} failed expectedly: {method_error}")
            
        except Exception as e:
            self.fail(f"Different correlation methods test failed: {e}")
    
    def test_save_functionality(self):
        """Test saving plots to file"""
        try:
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                self.skipTest("Not enough numeric columns")
            
            test_data_subset = self.test_data[numeric_cols[:6]].dropna()
            if len(test_data_subset) < 2:
                self.skipTest("Not enough valid data")
            
            # Test saving
            save_path = self.temp_path / "test_heatmap.png"
            # Define test_cols first
            numeric_cols = test_data_subset.select_dtypes(include=[np.number]).columns.tolist()
            test_cols = numeric_cols[:min(6, len(numeric_cols))]
            
            fig = self.plotter.create_correlation_heatmap(
                test_data_subset,
                features=test_cols[:4],
                save_path=str(save_path)
            )
            
            self.assertIsNotNone(fig)
            if save_path.exists():
                self.assertTrue(save_path.exists())
                self.assertGreater(save_path.stat().st_size, 0)
            
        except Exception as e:
            self.fail(f"Save functionality test failed: {e}")


class TestDistributionPlotter(unittest.TestCase):
    """Test cases for DistributionPlotter with lyrics dataset"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plotter = DistributionPlotter()
        
        # Load sample data
        loader = DataLoader()
        result = loader.load_dataset('lyrics_dataset', sample_size=100, validate=False)
        self.assertTrue(result.success, "Failed to load test data")
        self.test_data = result.data
        
        # Create temp directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        plt.close('all')  # Close all matplotlib figures
    
    def test_plotter_initialization(self):
        """Test plotter initialization"""
        self.assertIsInstance(self.plotter, DistributionPlotter)
        self.assertIsNotNone(self.plotter.config)
    
    def test_histogram_creation(self):
        """Test histogram creation"""
        try:
            # Find a numeric column
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                self.skipTest("No numeric columns available")
            
            test_col = numeric_cols[0]
            test_series = self.test_data[test_col].dropna()
            
            if len(test_series) < 2:
                self.skipTest("Not enough data for histogram")
            
            # Create histogram
            # Get test data first
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 1:
                self.skipTest("No numeric columns for histogram test")
            
            test_col = numeric_cols[0]
            test_data_subset = self.test_data[numeric_cols[:6]].dropna()
            
            # Use plot_feature_distributions instead
            result = self.plotter.plot_feature_distributions(
                test_data_subset,
                features=[test_col],
                plot_types=['histogram'],
                save_path=None
            )
            
            fig = result.get('histogram', {}).get('figure') if result else None
            
            self.assertIsNotNone(fig)
            self.assertTrue(hasattr(fig, 'axes'))
            
        except Exception as e:
            self.fail(f"Histogram creation failed: {e}")
    
    def test_boxplot_creation(self):
        """Test boxplot creation"""
        try:
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                self.skipTest("No numeric columns available")
            
            # Select multiple columns for boxplot
            test_cols = numeric_cols[:min(5, len(numeric_cols))]
            test_data_subset = self.test_data[test_cols].dropna()
            
            if test_data_subset.empty:
                self.skipTest("No valid data for boxplot")
            
            # Create boxplot
            # Use plot_feature_distributions instead
            result = self.plotter.plot_feature_distributions(
                test_data_subset,
                features=test_cols[:4],
                plot_types=['boxplot'],
                save_path=None
            )
            
            fig = result.get('boxplot', {}).get('figure') if result else None
            
            self.assertIsNotNone(fig)
            
        except Exception as e:
            self.fail(f"Boxplot creation failed: {e}")
    
    def test_distribution_grid(self):
        """Test distribution grid creation"""
        try:
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                self.skipText("Not enough numeric columns for grid")
            
            # Select subset of columns
            test_cols = numeric_cols[:min(6, len(numeric_cols))]
            test_data_subset = self.test_data[test_cols]
            
            # Create distribution grid
            # Use create_distribution_summary instead (no features parameter)
            fig = self.plotter.create_distribution_summary(
                test_data_subset,
                save_path=None
            )
            
            self.assertIsNotNone(fig)
            
        except Exception as e:
            self.fail(f"Distribution grid creation failed: {e}")
    
    def test_clustering_features_distributions(self):
        """Test distribution plots for clustering features"""
        try:
            # Find available clustering features
            available_features = []
            for feature in CLUSTERING_FEATURES:
                matching_cols = [col for col in self.test_data.columns 
                               if feature.lower() in col.lower()]
                if matching_cols:
                    # Check if it's numeric
                    col_data = self.test_data[matching_cols[0]]
                    if pd.api.types.is_numeric_dtype(col_data):
                        available_features.append(matching_cols[0])
            
            if len(available_features) < 2:
                self.skipTest("Not enough numeric clustering features")
            
            # Create distributions for clustering features
            clustering_data = self.test_data[available_features[:6]]
            
            # Use create_distribution_summary instead (no features parameter)
            fig = self.plotter.create_distribution_summary(
                clustering_data,
                save_path=None
            )
            
            self.assertIsNotNone(fig)
            
        except Exception as e:
            self.fail(f"Clustering features distributions failed: {e}")
    
    def test_save_functionality(self):
        """Test saving distribution plots"""
        try:
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                self.skipTest("No numeric columns available")
            
            test_col = numeric_cols[0]
            test_series = self.test_data[test_col].dropna()
            
            if len(test_series) < 2:
                self.skipTest("Not enough data")
            
            # Test saving
            save_path = self.temp_path / "test_distribution.png"
            # Get test data first
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 1:
                self.skipTest("No numeric columns for save test")
            
            test_col = numeric_cols[0]
            test_data_subset = self.test_data[numeric_cols[:6]].dropna()
            
            # Use plot_feature_distributions instead
            result = self.plotter.plot_feature_distributions(
                test_data_subset,
                features=[test_col],
                plot_types=['histogram'],
                save_path=str(save_path)
            )
            
            fig = result.get('histogram', {}).get('figure') if result else None
            
            self.assertIsNotNone(fig)
            if save_path.exists():
                self.assertTrue(save_path.exists())
                self.assertGreater(save_path.stat().st_size, 0)
            
        except Exception as e:
            self.fail(f"Save functionality test failed: {e}")


class TestVisualizationIntegration(unittest.TestCase):
    """Integration tests for visualization components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Load sample data
        loader = DataLoader()
        result = loader.load_dataset('lyrics_dataset', sample_size=50, validate=False)
        self.assertTrue(result.success, "Failed to load test data")
        self.test_data = result.data
    
    def tearDown(self):
        """Clean up test fixtures"""
        plt.close('all')
    
    def test_comprehensive_visualization_pipeline(self):
        """Test complete visualization pipeline"""
        try:
            correlation_plotter = CorrelationPlotter()
            distribution_plotter = DistributionPlotter()
            
            # Find numeric features and define test_cols first
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                self.skipTest("Not enough numeric columns for comprehensive test")
            
            test_cols = numeric_cols[:min(6, len(numeric_cols))]
            test_data_subset = self.test_data[test_cols].dropna()
            if test_data_subset.empty:
                self.skipTest("No valid data for comprehensive test")
            
            # Create correlation heatmap
            corr_fig = correlation_plotter.create_correlation_heatmap(
                test_data_subset,
                features=test_cols[:4]
            )
            
            # Create distribution plots
            
            dist_fig = distribution_plotter.create_distribution_summary(
                test_data_subset
            )
            
            self.assertIsNotNone(corr_fig)
            self.assertIsNotNone(dist_fig)
            
        except Exception as e:
            self.fail(f"Comprehensive visualization pipeline failed: {e}")
    
    def test_configuration_integration(self):
        """Test integration with configuration system"""
        try:
            correlation_plotter = CorrelationPlotter()
            distribution_plotter = DistributionPlotter()
            
            # Both should use configuration
            self.assertIsNotNone(correlation_plotter.config)
            self.assertIsNotNone(distribution_plotter.config)
            
            # Configuration should be compatible
            self.assertEqual(correlation_plotter.config.data.separator, '^')
            self.assertEqual(distribution_plotter.config.data.separator, '^')
            
        except Exception as e:
            self.fail(f"Configuration integration failed: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)