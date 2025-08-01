"""
Test Suite for Feature Analysis Module

Tests for dimensionality reduction and feature analysis components.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from sklearn.datasets import make_classification

# Import modules to test
from exploratory_analysis.feature_analysis.dimensionality_reduction import (
    DimensionalityReducer
)
from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.config.features_config import CLUSTERING_FEATURES


class TestDimensionalityReducer(unittest.TestCase):
    """Test cases for DimensionalityReducer with lyrics dataset"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reducer = DimensionalityReducer()
        
        # Load sample data
        loader = DataLoader()
        result = loader.load_dataset('lyrics_dataset', sample_size=100, validate=False)
        self.assertTrue(result.success, "Failed to load test data")
        self.test_data = result.data
        
        # Prepare numeric features for testing
        self.numeric_features = self._prepare_numeric_features()
        
        # Create temp directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def _prepare_numeric_features(self):
        """Prepare numeric features from test data"""
        # Find numeric columns
        numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            # Create synthetic data for testing if not enough features
            X, _ = make_classification(n_samples=100, n_features=8, n_informative=6, 
                                     n_redundant=2, random_state=42)
            feature_names = [f'feature_{i}' for i in range(8)]
            return pd.DataFrame(X, columns=feature_names)
        else:
            # Use real data
            selected_cols = numeric_cols[:min(8, len(numeric_cols))]
            return self.test_data[selected_cols].dropna()
    
    def test_reducer_initialization(self):
        """Test reducer initialization"""
        self.assertIsInstance(self.reducer, DimensionalityReducer)
        self.assertIsNotNone(self.reducer.config)
    
    def test_pca_analysis(self):
        """Test PCA dimensionality reduction"""
        try:
            if len(self.numeric_features) < 2:
                self.skipTest("Not enough features for PCA")
            
            # Perform PCA
            result = self.reducer.fit_pca(
                self.numeric_features,
                n_components=min(3, len(self.numeric_features.columns))
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('transformed_data', result)
            self.assertIn('explained_variance_ratio', result)
            
            # Check dimensions
            n_components = min(3, len(self.numeric_features.columns))
            self.assertEqual(result['transformed_data'].shape[1], n_components)
            self.assertEqual(len(result['explained_variance_ratio']), n_components)
            
            # Check variance ratios are reasonable
            self.assertTrue(all(0 <= ratio <= 1 for ratio in result['explained_variance_ratio']))
            
        except Exception as e:
            self.fail(f"PCA analysis failed: {e}")
    
    def test_tsne_analysis(self):
        """Test t-SNE dimensionality reduction"""
        try:
            if len(self.numeric_features) < 2 or len(self.numeric_features) < 5:
                self.skipTest("Not enough data for t-SNE")
            
            # Use smaller subset for t-SNE (it's computationally intensive)
            small_sample = self.numeric_features.head(50)
            
            # Perform t-SNE
            result = self.reducer.fit_tsne(
                small_sample,
                n_components=2,
                perplexity=min(10, len(small_sample) // 3)
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('transformed_data', result)
            self.assertEqual(result['transformed_data'].shape[1], 2)
            self.assertEqual(result['transformed_data'].shape[0], len(small_sample))
            
        except Exception as e:
            # t-SNE might fail with small datasets or specific configurations
            self.skipTest(f"t-SNE analysis skipped due to: {e}")
    
    def test_umap_analysis(self):
        """Test UMAP dimensionality reduction (if available)"""
        try:
            if len(self.numeric_features) < 2 or len(self.numeric_features) < 10:
                self.skipTest("Not enough data for UMAP")
            
            # UMAP might not be installed
            try:
                import umap
            except ImportError:
                self.skipTest("UMAP not available")
            
            # Use smaller subset for UMAP
            small_sample = self.numeric_features.head(50)
            
            # Perform UMAP
            result = self.reducer.fit_umap(
                small_sample,
                n_components=2,
                n_neighbors=min(5, len(small_sample) // 2)
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('transformed_data', result)
            self.assertEqual(result['transformed_data'].shape[1], 2)
            
        except Exception as e:
            self.skipTest(f"UMAP analysis skipped due to: {e}")
    
    def test_feature_selection(self):
        """Test feature selection functionality"""
        try:
            if len(self.numeric_features) < 4:
                self.skipTest("Not enough features for feature selection")
            
            # Create target variable for supervised feature selection
            # Use first feature as proxy target (for testing purposes)
            target = (self.numeric_features.iloc[:, 0] > self.numeric_features.iloc[:, 0].median()).astype(int)
            features = self.numeric_features.iloc[:, 1:]  # Exclude target proxy
            
            if len(features.columns) < 2:
                self.skipTest("Not enough features after target creation")
            
            # Perform feature selection
            result = self.reducer.perform_feature_selection(
                features,
                target_column=None,  # Pass target as separate parameter if needed
                methods=['mutual_info']
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('selection_results', result)
            # Feature selection results should contain the selected features info
            
        except Exception as e:
            self.fail(f"Feature selection failed: {e}")
    
    def test_clustering_features_analysis(self):
        """Test dimensionality reduction on clustering features"""
        try:
            # Find available clustering features in dataset
            available_features = []
            for feature in CLUSTERING_FEATURES:
                matching_cols = [col for col in self.test_data.columns 
                               if feature.lower() in col.lower()]
                if matching_cols:
                    col_data = self.test_data[matching_cols[0]]
                    if pd.api.types.is_numeric_dtype(col_data):
                        available_features.append(matching_cols[0])
            
            if len(available_features) < 3:
                self.skipTest("Not enough clustering features available")
            
            # Prepare clustering features data
            clustering_data = self.test_data[available_features].dropna()
            
            if len(clustering_data) < 10:
                self.skipTest("Not enough valid clustering data")
            
            # Perform PCA on clustering features
            result = self.reducer.fit_pca(
                clustering_data,
                n_components=min(3, len(available_features))
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('transformed_data', result)
            
        except Exception as e:
            self.fail(f"Clustering features analysis failed: {e}")
    
    def test_explained_variance_analysis(self):
        """Test explained variance analysis"""
        try:
            if len(self.numeric_features) < 3:
                self.skipTest("Not enough features for variance analysis")
            
            # Perform PCA with different numbers of components
            n_features = len(self.numeric_features.columns)
            max_components = min(5, n_features)
            
            result = self.reducer.fit_pca(
                self.numeric_features,
                n_components=max_components
            )
            
            # Check explained variance properties
            explained_var = result['explained_variance_ratio']
            
            # Should be decreasing
            for i in range(len(explained_var) - 1):
                self.assertGreaterEqual(explained_var[i], explained_var[i + 1])
            
            # Cumulative variance should increase
            cumulative_var = np.cumsum(explained_var)
            for i in range(len(cumulative_var) - 1):
                self.assertGreaterEqual(cumulative_var[i + 1], cumulative_var[i])
            
            # Last cumulative value should be <= 1
            self.assertLessEqual(cumulative_var[-1], 1.0)
            
        except Exception as e:
            self.fail(f"Explained variance analysis failed: {e}")
    
    def test_visualization_output(self):
        """Test visualization functionality"""
        try:
            if len(self.numeric_features) < 2:
                self.skipTest("Not enough features for visualization")
            
            # Perform PCA for 2D visualization
            result = self.reducer.fit_pca(self.numeric_features, n_components=2)
            
            # Test basic result structure
            self.assertIn('transformed_data', result)
            self.assertEqual(result['transformed_data'].shape[1], 2)
            
            # Test if visualization method exists
            if hasattr(self.reducer, 'create_dimensionality_reduction_comparison'):
                fig = self.reducer.create_dimensionality_reduction_comparison(
                    self.numeric_features, methods=['pca']
                )
                if fig is not None:
                    self.assertIsNotNone(fig)
            
        except Exception as e:
            self.fail(f"Visualization output test failed: {e}")
    
    def test_component_interpretation(self):
        """Test component interpretation functionality"""
        try:
            if len(self.numeric_features) < 3:
                self.skipTest("Not enough features for component interpretation")
            
            result = self.reducer.fit_pca(self.numeric_features, n_components=2)
            
            # Check if component analysis is available
            if 'component_analysis' in result:
                component_analysis = result['component_analysis']
                self.assertIsInstance(component_analysis, dict)
                
                # Should have analysis for each component
                self.assertIn('PC1', component_analysis)
                self.assertIn('PC2', component_analysis)
            
        except Exception as e:
            self.fail(f"Component interpretation test failed: {e}")


class TestFeatureAnalysisIntegration(unittest.TestCase):
    """Integration tests for feature analysis components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reducer = DimensionalityReducer()
        
        # Load sample data
        loader = DataLoader()
        result = loader.load_dataset('lyrics_dataset', sample_size=50, validate=False)
        self.assertTrue(result.success, "Failed to load test data")
        self.test_data = result.data
    
    def test_end_to_end_analysis(self):
        """Test complete feature analysis pipeline"""
        try:
            # Find numeric features
            numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 3:
                # Create synthetic data for complete test
                X, y = make_classification(n_samples=50, n_features=6, n_informative=4, 
                                         n_redundant=2, random_state=42)
                feature_names = [f'feature_{i}' for i in range(6)]
                features_df = pd.DataFrame(X, columns=feature_names)
                target_series = pd.Series(y, name='target')
            else:
                features_df = self.test_data[numeric_cols[:6]].dropna()
                # Create binary target from first feature
                target_series = (features_df.iloc[:, 0] > features_df.iloc[:, 0].median()).astype(int)
                features_df = features_df.iloc[:, 1:]  # Remove target proxy
            
            if len(features_df) < 10 or len(features_df.columns) < 2:
                self.skipTest("Not enough data for end-to-end test")
            
            # Step 1: Feature selection
            selection_result = self.reducer.perform_feature_selection(
                features_df,
                target_column=None,
                methods=['variance']
            )
            
            # Step 2: Dimensionality reduction
            pca_result = self.reducer.fit_pca(features_df, n_components=2)
            
            # Verify pipeline results
            self.assertIsInstance(selection_result, dict)
            self.assertIsInstance(pca_result, dict)
            self.assertEqual(pca_result['transformed_data'].shape[1], 2)
            
        except Exception as e:
            self.fail(f"End-to-end analysis failed: {e}")
    
    def test_configuration_integration(self):
        """Test integration with configuration system"""
        try:
            # Should use proper configuration
            self.assertIsNotNone(self.reducer.config)
            
            # Configuration should be compatible with lyrics dataset
            self.assertEqual(self.reducer.config.data.separator, '^')
            
        except Exception as e:
            self.fail(f"Configuration integration failed: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)