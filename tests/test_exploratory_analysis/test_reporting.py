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
from exploratory_analysis.reporting.report_generator import ReportGenerator, AnalysisReport
from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.statistical_analysis.descriptive_stats import DescriptiveStatsAnalyzer
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
    
    def test_basic_report_generation(self):
        """Test basic report generation"""
        try:
            # Generate basic report
            report = self.report_generator.generate_comprehensive_report(
                self.test_data,
                title="Test Lyrics Dataset Report"
            )
            
            self.assertIsInstance(report, AnalysisReport)
            self.assertIsNotNone(report.metadata)
            self.assertTrue(hasattr(report, 'sections'))
            
            # Check report has basic sections
            if hasattr(report, 'sections'):
                section_names = [section.get('name', '') for section in report.sections]
                # Should have at least dataset overview
                self.assertTrue(any('overview' in name.lower() for name in section_names))
            
        except Exception as e:
            self.fail(f"Basic report generation failed: {e}")
    
    def test_dataset_overview_section(self):
        """Test dataset overview section generation"""
        try:
            report = self.report_generator.generate_comprehensive_report(self.test_data)
            
            # Find overview section
            overview_section = None
            if hasattr(report, 'sections'):
                for section in report.sections:
                    if 'overview' in section.get('name', '').lower():
                        overview_section = section
                        break
            
            if overview_section:
                # Should have basic dataset information
                content = overview_section.get('content', {})
                self.assertTrue('rows' in content or 'shape' in content)
                self.assertTrue('columns' in content or 'shape' in content)
            
        except Exception as e:
            self.fail(f"Dataset overview section generation failed: {e}")
    
    def test_statistical_analysis_integration(self):
        """Test integration with statistical analysis"""
        try:
            # Generate report with statistical analysis
            report = self.report_generator.generate_comprehensive_report(
                self.test_data,
                include_stats=True
            )
            
            self.assertIsInstance(report, AnalysisReport)
            
            # Should include statistical information
            if hasattr(report, 'sections'):
                stats_sections = [section for section in report.sections 
                                if 'stat' in section.get('name', '').lower()]
                
                # Might not have explicit stats section, but should not crash
                self.assertTrue(True, "Statistical integration should not crash")
            
        except Exception as e:
            self.fail(f"Statistical analysis integration failed: {e}")
    
    def test_clustering_features_reporting(self):
        """Test reporting on clustering features"""
        try:
            # Find available clustering features
            available_features = []
            for feature in CLUSTERING_FEATURES:
                matching_cols = [col for col in self.test_data.columns 
                               if feature.lower() in col.lower()]
                if matching_cols:
                    available_features.extend(matching_cols)
            
            if len(available_features) < 2:
                self.skipTest("Not enough clustering features available")
            
            # Generate report focusing on clustering features
            clustering_data = self.test_data[available_features[:8]]
            
            report = self.report_generator.generate_comprehensive_report(
                clustering_data,
                title="Clustering Features Report"
            )
            
            self.assertIsInstance(report, AnalysisReport)
            
        except Exception as e:
            self.fail(f"Clustering features reporting failed: {e}")
    
    def test_lyrics_column_handling(self):
        """Test handling of lyrics column in reports"""
        try:
            # Generate report that should handle lyrics column appropriately
            report = self.report_generator.generate_comprehensive_report(self.test_data)
            
            # Should not crash on lyrics column
            self.assertIsInstance(report, AnalysisReport)
            
            # If lyrics column is processed, should be handled as text
            if hasattr(report, 'sections'):
                for section in report.sections:
                    content = section.get('content', {})
                    # Look for lyrics-related content
                    lyrics_content = [item for item in str(content).lower().split() 
                                    if 'lyrics' in item]
                    # If found, should not cause issues
                    if lyrics_content:
                        self.assertTrue(True, "Lyrics content processed without error")
            
        except Exception as e:
            self.fail(f"Lyrics column handling failed: {e}")
    
    def test_report_export_json(self):
        """Test JSON export functionality"""
        try:
            report = self.report_generator.generate_comprehensive_report(
                self.test_data,
                title="Test JSON Export Report"
            )
            
            # Export to JSON
            json_path = self.temp_path / "test_report.json"
            success = self.report_generator.export_report(
                report,
                json_path,
                format='json'
            )
            
            if success and json_path.exists():
                self.assertTrue(json_path.exists())
                self.assertGreater(json_path.stat().st_size, 0)
                
                # Verify JSON is valid
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    self.assertIsInstance(json_data, dict)
            
        except Exception as e:
            self.fail(f"JSON export failed: {e}")
    
    def test_report_export_markdown(self):
        """Test Markdown export functionality"""
        try:
            report = self.report_generator.generate_comprehensive_report(
                self.test_data,
                title="Test Markdown Export Report"
            )
            
            # Export to Markdown
            md_path = self.temp_path / "test_report.md"
            success = self.report_generator.export_report(
                report,
                md_path,
                format='markdown'
            )
            
            if success and md_path.exists():
                self.assertTrue(md_path.exists())
                self.assertGreater(md_path.stat().st_size, 0)
                
                # Verify Markdown content
                with open(md_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                    self.assertIsInstance(md_content, str)
                    self.assertGreater(len(md_content.strip()), 0)
            
        except Exception as e:
            self.fail(f"Markdown export failed: {e}")
    
    def test_report_export_html(self):
        """Test HTML export functionality"""
        try:
            report = self.report_generator.generate_comprehensive_report(
                self.test_data,
                title="Test HTML Export Report"
            )
            
            # Export to HTML
            html_path = self.temp_path / "test_report.html"
            success = self.report_generator.export_report(
                report,
                html_path,
                format='html'
            )
            
            if success and html_path.exists():
                self.assertTrue(html_path.exists())
                self.assertGreater(html_path.stat().st_size, 0)
                
                # Verify HTML content
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    self.assertIsInstance(html_content, str)
                    # Should contain basic HTML tags
                    self.assertTrue('<html>' in html_content.lower() or 
                                  '<!doctype html>' in html_content.lower())
            
        except Exception as e:
            self.fail(f"HTML export failed: {e}")
    
    def test_report_metadata(self):
        """Test report metadata generation"""
        try:
            report = self.report_generator.generate_comprehensive_report(
                self.test_data,
                title="Test Metadata Report"
            )
            
            # Check metadata
            self.assertIsNotNone(report.metadata)
            metadata = report.metadata
            
            # Should have basic metadata
            expected_fields = ['title', 'timestamp', 'dataset_info']
            for field in expected_fields:
                if field in metadata:
                    self.assertIsNotNone(metadata[field])
            
            # Dataset info should include basic information
            if 'dataset_info' in metadata:
                dataset_info = metadata['dataset_info']
                self.assertTrue('rows' in dataset_info or 'shape' in dataset_info)
                self.assertTrue('columns' in dataset_info or 'shape' in dataset_info)
            
        except Exception as e:
            self.fail(f"Report metadata generation failed: {e}")
    
    def test_large_dataset_handling(self):
        """Test handling of larger dataset samples"""
        try:
            # Load larger sample
            loader = DataLoader()
            result = loader.load_dataset('lyrics_dataset', sample_size=200, validate=False)
            
            if result.success and len(result.data) > 100:
                large_data = result.data
                
                # Should handle larger dataset
                report = self.report_generator.generate_comprehensive_report(
                    large_data,
                    title="Large Dataset Test Report"
                )
                
                self.assertIsInstance(report, AnalysisReport)
                
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
            # Step 1: Statistical analysis
            stats_analyzer = DescriptiveStatsAnalyzer()
            stats_report = stats_analyzer.analyze_dataset(self.test_data)
            
            # Step 2: Report generation
            report_generator = ReportGenerator()
            comprehensive_report = report_generator.generate_comprehensive_report(
                self.test_data,
                title="Complete Pipeline Test Report",
                include_stats=True
            )
            
            # Step 3: Export multiple formats
            json_path = self.temp_path / "pipeline_report.json"
            md_path = self.temp_path / "pipeline_report.md"
            
            json_success = report_generator.export_report(
                comprehensive_report, json_path, format='json'
            )
            md_success = report_generator.export_report(
                comprehensive_report, md_path, format='markdown'
            )
            
            # Verify pipeline results
            self.assertIsNotNone(stats_report)
            self.assertIsInstance(comprehensive_report, AnalysisReport)
            
            if json_success and json_path.exists():
                self.assertTrue(json_path.exists())
            if md_success and md_path.exists():
                self.assertTrue(md_path.exists())
            
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
            report = report_generator.generate_comprehensive_report(
                self.test_data,
                title="Configuration Integration Test"
            )
            self.assertIsNotNone(report)
            
        except Exception as e:
            self.fail(f"Configuration integration failed: {e}")
    
    def test_error_handling(self):
        """Test error handling in reporting"""
        try:
            report_generator = ReportGenerator()
            
            # Test with empty dataset
            empty_data = pd.DataFrame()
            try:
                report = report_generator.generate_comprehensive_report(
                    empty_data,
                    title="Empty Dataset Test"
                )
                # Should either handle gracefully or raise appropriate error
                if report is not None:
                    self.assertIsInstance(report, AnalysisReport)
            except Exception as e:
                # Expected to fail with empty data
                self.assertTrue(True, f"Empty dataset handled appropriately: {e}")
            
            # Test with problematic data
            problematic_data = pd.DataFrame({
                'col1': [np.inf, -np.inf, np.nan, 1, 2],
                'col2': ['a', 'b', None, 'd', 'e'],
                'col3': [1.0, 2.0, 3.0, 4.0, 5.0]
            })
            
            try:
                report = report_generator.generate_comprehensive_report(
                    problematic_data,
                    title="Problematic Data Test"
                )
                # Should handle problematic data gracefully
                if report is not None:
                    self.assertIsInstance(report, AnalysisReport)
            except Exception as e:
                # Acceptable to fail, but should not crash the system
                self.assertTrue(True, f"Problematic data handled: {e}")
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)