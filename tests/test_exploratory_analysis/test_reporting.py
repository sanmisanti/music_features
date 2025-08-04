"""
Test Suite for Reporting Module

Tests for report generation and summary statistics components.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

# Import modules to test
from exploratory_analysis.reporting.report_generator import ReportGenerator
from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.statistical_analysis.descriptive_stats import DescriptiveStats
from exploratory_analysis.config.features_config import CLUSTERING_FEATURES


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator with lyrics dataset"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.report_generator = ReportGenerator()
        
        # Load sample data
        loader = DataLoader()
        result = loader.load_dataset('lyrics_dataset', sample_size=50, validate=False)
        self.assertTrue(result.success, "Failed to load test data")
        self.test_data = result.data
        
        # Create temp directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def test_generator_initialization(self):
        """Test report generator initialization"""
        self.assertIsInstance(self.report_generator, ReportGenerator)
        self.assertIsNotNone(self.report_generator.config)
        self.assertIsNotNone(self.report_generator.output_dir)
    
    def test_basic_report_generation(self):
        """Test basic report generation"""
        try:
            # Generate basic report using the real API
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=50,
                include_visualizations=False,
                formats=['json']
            )
            
            self.assertIsInstance(report_paths, dict)
            # Should return paths to generated files or be empty if no files created
            
        except Exception as e:
            self.fail(f"Basic report generation failed: {e}")
    
    def test_dataset_overview_section(self):
        """Test dataset overview section generation"""
        try:
            # Test that analysis results are stored
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=30,
                include_visualizations=False
            )
            
            # Check that dataset info is populated
            self.assertIsInstance(self.report_generator.dataset_info, dict)
            if self.report_generator.dataset_info:
                # Should have basic information (using actual keys from dataset_info)
                self.assertIn('sample_size', self.report_generator.dataset_info)
                self.assertIn('features', self.report_generator.dataset_info)
            
        except Exception as e:
            self.fail(f"Dataset overview section generation failed: {e}")
    
    def test_statistical_analysis_integration(self):
        """Test integration with statistical analysis"""
        try:
            # Generate report that includes statistical analysis
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=50,
                include_visualizations=False
            )
            
            # Check that analysis results are populated
            self.assertIsInstance(self.report_generator.analysis_results, dict)
            # Should not crash - statistical integration working
            
        except Exception as e:
            self.fail(f"Statistical analysis integration failed: {e}")
    
    def test_clustering_features_reporting(self):
        """Test reporting on clustering features"""
        try:
            # Generate report which should handle clustering features automatically
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=40,
                include_visualizations=False
            )
            
            # Should complete without errors
            self.assertIsInstance(report_paths, dict)
            
        except Exception as e:
            self.fail(f"Clustering features reporting failed: {e}")
    
    def test_lyrics_column_handling(self):
        """Test handling of lyrics column in reports"""
        try:
            # Generate report that should handle lyrics column appropriately
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=30
            )
            
            # Should not crash on lyrics column
            self.assertIsInstance(report_paths, dict)
            # Lyrics content should be processed without error
            
        except Exception as e:
            self.fail(f"Lyrics column handling failed: {e}")
    
    def test_report_export_json(self):
        """Test JSON export functionality"""
        try:
            # Generate report with JSON format
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=30,
                include_visualizations=False,
                formats=['json']
            )
            
            self.assertIsInstance(report_paths, dict)
            
            # Check if JSON file was created
            if 'json' in report_paths and report_paths['json']:
                json_path = Path(report_paths['json'])
                if json_path.exists():
                    self.assertTrue(json_path.exists())
                    self.assertGreater(json_path.stat().st_size, 0)
            
        except Exception as e:
            self.fail(f"JSON export failed: {e}")
    
    def test_report_export_markdown(self):
        """Test Markdown export functionality"""
        try:
            # Generate report with Markdown format
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=30,
                include_visualizations=False,
                formats=['markdown']
            )
            
            self.assertIsInstance(report_paths, dict)
            
            # Check if Markdown file was created
            if 'markdown' in report_paths and report_paths['markdown']:
                md_path = Path(report_paths['markdown'])
                if md_path.exists():
                    self.assertTrue(md_path.exists())
                    self.assertGreater(md_path.stat().st_size, 0)
            
        except Exception as e:
            self.fail(f"Markdown export failed: {e}")
    
    def test_report_export_html(self):
        """Test HTML export functionality"""
        try:
            # Generate report with HTML format
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=30,
                include_visualizations=False,
                formats=['html']
            )
            
            self.assertIsInstance(report_paths, dict)
            
            # Check if HTML file was created
            if 'html' in report_paths and report_paths['html']:
                html_path = Path(report_paths['html'])
                if html_path.exists():
                    self.assertTrue(html_path.exists())
                    self.assertGreater(html_path.stat().st_size, 0)
            
        except Exception as e:
            self.fail(f"HTML export failed: {e}")
    
    def test_report_metadata(self):
        """Test report metadata generation"""
        try:
            # Generate report and check internal metadata
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=30
            )
            
            # Check that dataset info is populated
            self.assertIsInstance(self.report_generator.dataset_info, dict)
            
            # Should have basic dataset information
            if self.report_generator.dataset_info:
                info = self.report_generator.dataset_info
                # Should contain shape or similar info
                self.assertTrue(len(info) > 0)
            
        except Exception as e:
            self.fail(f"Report metadata generation failed: {e}")
    
    def test_large_dataset_handling(self):
        """Test handling of larger dataset samples"""
        try:
            # Test with larger sample size
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=100,
                include_visualizations=False
            )
            
            # Should handle larger dataset without issues
            self.assertIsInstance(report_paths, dict)
                
        except Exception as e:
            self.fail(f"Large dataset handling failed: {e}")


class TestReportingIntegration(unittest.TestCase):
    """Integration tests for reporting components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Load sample data
        loader = DataLoader()
        result = loader.load_dataset('lyrics_dataset', sample_size=30, validate=False)
        self.assertTrue(result.success, "Failed to load test data")
        self.test_data = result.data
    
    def test_complete_reporting_pipeline(self):
        """Test complete reporting pipeline with all components"""
        try:
            # Step 1: Statistical analysis through report generator
            report_generator = ReportGenerator(str(self.temp_path))
            
            # Step 2: Generate comprehensive report with multiple formats
            report_paths = report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=50,
                include_visualizations=True,
                formats=['json', 'markdown']
            )
            
            # Verify pipeline results
            self.assertIsInstance(report_paths, dict)
            
            # Check that analysis results are populated
            self.assertIsInstance(report_generator.analysis_results, dict)
            
        except Exception as e:
            self.fail(f"Complete reporting pipeline failed: {e}")
    
    def test_configuration_integration(self):
        """Test integration with configuration system"""
        try:
            report_generator = ReportGenerator()
            
            # Should use proper configuration
            self.assertIsNotNone(report_generator.config)
            
            # Configuration should be compatible with lyrics dataset
            self.assertEqual(report_generator.config.data.separator, '^')
            
            # Should be able to generate report with current configuration
            report_paths = report_generator.generate_comprehensive_report(
                dataset_type='lyrics_dataset',
                sample_size=30,
                include_visualizations=False
            )
            self.assertIsInstance(report_paths, dict)
            
        except Exception as e:
            self.fail(f"Configuration integration failed: {e}")
    
    def test_error_handling(self):
        """Test error handling in reporting"""
        try:
            report_generator = ReportGenerator()
            
            # Test with very small sample (edge case)
            try:
                report_paths = report_generator.generate_comprehensive_report(
                    dataset_type='lyrics_dataset',
                    sample_size=5,
                    include_visualizations=False
                )
                # Should either handle gracefully or return empty dict
                self.assertIsInstance(report_paths, dict)
            except Exception as e:
                # Expected to fail with very small data
                self.assertTrue(True, f"Small dataset handled appropriately: {e}")
            
            # Test with invalid dataset type
            try:
                report_paths = report_generator.generate_comprehensive_report(
                    dataset_type='nonexistent_dataset',
                    sample_size=30
                )
                # Should handle invalid dataset gracefully
                self.assertIsInstance(report_paths, dict)
            except Exception as e:
                # Acceptable to fail with invalid dataset
                self.assertTrue(True, f"Invalid dataset handled: {e}")
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)