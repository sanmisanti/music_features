#!/usr/bin/env python3
"""
🔍 LARGE DATASET PROCESSOR
==========================
Processes the complete 1.2M songs dataset for comprehensive analysis.

This script:
1. Configures system for large dataset processing
2. Loads and analyzes the full tracks_features.csv
3. Generates comprehensive statistical profiles
4. Prepares data for intelligent song selection
5. Creates detailed analysis reports

Usage:
    python scripts/large_dataset_processor.py [--sample-size N] [--output-dir DIR]
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exploratory_analysis.config.analysis_config import get_config, configure_for_large_dataset
from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.statistical_analysis.descriptive_stats import DescriptiveStats
from exploratory_analysis.feature_analysis.dimensionality_reduction import DimensionalityReducer
from exploratory_analysis.reporting.report_generator import ReportGenerator
from exploratory_analysis.utils.file_utils import format_file_size, get_memory_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/large_dataset_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LargeDatasetProcessor:
    """
    Processor for analyzing the complete 1.2M songs dataset.
    
    Optimized for memory efficiency and comprehensive analysis.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the processor with large dataset configuration."""
        # Configure for large dataset processing
        configure_for_large_dataset()
        self.config = get_config()
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.config.get_output_path('reports')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.stats_analyzer = DescriptiveStats()
        self.dim_reducer = DimensionalityReducer()
        self.report_generator = ReportGenerator(output_dir=str(self.output_dir))
        
        # Analysis results storage
        self.dataset_info = {}
        self.analysis_results = {}
        
        logger.info(f"🔍 Large Dataset Processor initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def analyze_dataset_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze the structure and basic properties of the dataset.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Dictionary with dataset structure information
        """
        logger.info("🔍 Analyzing dataset structure...")
        
        file_path = Path(file_path)
        
        # File information
        file_size = file_path.stat().st_size
        structure_info = {
            'file_path': str(file_path),
            'file_size_bytes': file_size,
            'file_size_human': format_file_size(file_size),
            'exists': file_path.exists()
        }
        
        if not file_path.exists():
            logger.error(f"❌ Dataset file not found: {file_path}")
            return structure_info
        
        # Quick peek at the data
        try:
            import pandas as pd
            
            # Read just first few rows to understand structure
            sample_df = pd.read_csv(file_path, sep=';', decimal=',', nrows=10)
            
            structure_info.update({
                'columns': list(sample_df.columns),
                'num_columns': len(sample_df.columns),
                'sample_dtypes': sample_df.dtypes.to_dict(),
                'encoding': 'utf-8',
                'separator': ';',
                'decimal': ',',
                'preview_available': True
            })
            
            # Get total row count (more memory efficient than loading all data)
            with open(file_path, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract header
            
            structure_info['total_rows'] = total_rows
            structure_info['estimated_memory_mb'] = (file_size / 1024 / 1024) * 2  # Rough estimate
            
            logger.info(f"📊 Dataset structure: {total_rows:,} rows × {len(sample_df.columns)} columns")
            logger.info(f"💾 File size: {format_file_size(file_size)}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing dataset structure: {str(e)}")
            structure_info['error'] = str(e)
        
        return structure_info
    
    def load_and_validate_dataset(self, dataset_path: str, sample_size: Optional[int] = None) -> Any:
        """
        Load and validate the large dataset with memory optimization.
        
        Args:
            dataset_path: Path to the dataset
            sample_size: Optional sample size for initial analysis
            
        Returns:
            Loaded and validated dataset
        """
        logger.info("📂 Loading and validating large dataset...")
        
        try:
            # Load with intelligent sampling if needed
            load_result = self.data_loader.load_dataset(
                file_path=dataset_path,
                sample_size=sample_size,
                validation_level='STANDARD'  # Balance between thoroughness and speed
            )
            
            if load_result.success:
                logger.info(f"✅ Dataset loaded successfully")
                logger.info(f"📊 Shape: {load_result.data.shape}")
                logger.info(f"🎯 Quality Score: {load_result.quality_score:.1f}/100")
                logger.info(f"⏱️  Load Time: {load_result.metadata.get('load_time', 'N/A'):.2f}s")
                logger.info(f"💾 Memory Usage: {get_memory_usage():.1f} MB")
                
                # Store dataset info
                self.dataset_info = {
                    'shape': load_result.data.shape,
                    'quality_score': load_result.quality_score,
                    'load_time': load_result.metadata.get('load_time', 0),
                    'memory_usage_mb': get_memory_usage(),
                    'sample_size': sample_size,
                    'validation_level': 'STANDARD'
                }
                
                return load_result.data
            else:
                logger.error(f"❌ Dataset loading failed: {load_result.error}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {str(e)}")
            return None
    
    def perform_comprehensive_analysis(self, data) -> Dict[str, Any]:
        """
        Perform comprehensive statistical and feature analysis.
        
        Args:
            data: The loaded dataset
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info("📈 Performing comprehensive analysis...")
        
        analysis_results = {}
        
        try:
            # 1. Statistical Analysis
            logger.info("📊 Computing descriptive statistics...")
            stats_result = self.stats_analyzer.analyze_dataset(data)
            
            if stats_result['success']:
                analysis_results['statistical_analysis'] = stats_result
                logger.info(f"✅ Statistical analysis completed")
                logger.info(f"📊 Features analyzed: {len(stats_result['feature_statistics'])}")
                
                # Log key correlations
                if 'correlation_analysis' in stats_result:
                    high_corr = stats_result['correlation_analysis'].get('high_correlations', [])
                    if high_corr:
                        logger.info(f"🔗 High correlations found: {len(high_corr)}")
                        for corr in high_corr[:3]:  # Show top 3
                            logger.info(f"   {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")
            else:
                logger.warning(f"⚠️  Statistical analysis had issues: {stats_result.get('error', 'Unknown')}")
                analysis_results['statistical_analysis'] = {'success': False, 'error': stats_result.get('error')}
            
            # 2. Dimensionality Analysis
            logger.info("🔬 Performing dimensionality analysis...")
            dim_results = self.dim_reducer.analyze_dimensionality(data)
            
            if dim_results.get('success', False):
                analysis_results['dimensionality_analysis'] = dim_results
                logger.info(f"✅ Dimensionality analysis completed")
                
                # Log PCA results
                if 'pca_analysis' in dim_results:
                    pca_info = dim_results['pca_analysis']
                    logger.info(f"🎯 PCA: {pca_info.get('n_components', 'N/A')} components explain {pca_info.get('total_variance_ratio', 0)*100:.1f}% variance")
                    
                    # Show top components
                    if 'component_importance' in pca_info:
                        top_components = pca_info['component_importance'][:3]
                        for i, comp in enumerate(top_components):
                            logger.info(f"   PC{i+1}: {comp.get('variance_ratio', 0)*100:.1f}% - {comp.get('interpretation', 'Unknown')}")
            else:
                logger.warning(f"⚠️  Dimensionality analysis failed: {dim_results.get('error', 'Unknown')}")
                analysis_results['dimensionality_analysis'] = {'success': False, 'error': dim_results.get('error')}
            
            # 3. Feature Importance Analysis
            logger.info("🎯 Analyzing feature importance...")
            feature_importance = self.analyze_feature_importance(data)
            analysis_results['feature_importance'] = feature_importance
            
            # 4. Data Distribution Analysis
            logger.info("📊 Analyzing data distributions...")
            distribution_analysis = self.analyze_distributions(data)
            analysis_results['distribution_analysis'] = distribution_analysis
            
            logger.info("✅ Comprehensive analysis completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive analysis: {str(e)}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def analyze_feature_importance(self, data) -> Dict[str, Any]:
        """Analyze feature importance for selection guidance."""
        try:
            from exploratory_analysis.config.features_config import CLUSTERING_FEATURES
            from sklearn.feature_selection import VarianceThreshold
            import numpy as np
            
            # Get clustering features
            clustering_data = data[CLUSTERING_FEATURES]
            
            # Variance-based importance
            selector = VarianceThreshold()
            selector.fit(clustering_data)
            
            variances = selector.variances_
            feature_variance = dict(zip(CLUSTERING_FEATURES, variances))
            
            # Sort by variance (higher = more important)
            sorted_features = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'success': True,
                'method': 'variance_threshold',
                'feature_variances': feature_variance,
                'ranked_features': sorted_features,
                'most_important': sorted_features[:5],
                'least_important': sorted_features[-3:]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_distributions(self, data) -> Dict[str, Any]:
        """Analyze feature distributions for sampling guidance."""
        try:
            from exploratory_analysis.config.features_config import CLUSTERING_FEATURES
            from scipy import stats
            import numpy as np
            
            clustering_data = data[CLUSTERING_FEATURES]
            distribution_info = {}
            
            for feature in CLUSTERING_FEATURES:
                feature_data = clustering_data[feature].dropna()
                
                # Basic distribution statistics
                distribution_info[feature] = {
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std()),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'skewness': float(stats.skew(feature_data)),
                    'kurtosis': float(stats.kurtosis(feature_data)),
                    'quartiles': {
                        'q25': float(feature_data.quantile(0.25)),
                        'q50': float(feature_data.quantile(0.50)),
                        'q75': float(feature_data.quantile(0.75))
                    }
                }
            
            return {
                'success': True,
                'feature_distributions': distribution_info,
                'summary': {
                    'most_skewed': max(distribution_info.items(), key=lambda x: abs(x[1]['skewness']))[0],
                    'most_variable': max(distribution_info.items(), key=lambda x: x[1]['std'])[0],
                    'total_features': len(CLUSTERING_FEATURES)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_processing_report(self) -> str:
        """Generate a comprehensive processing report."""
        logger.info("📄 Generating processing report...")
        
        try:
            # Create comprehensive report using existing system
            report_paths = self.report_generator.generate_comprehensive_report(
                dataset_type='original',  # Use original dataset
                include_visualizations=True,
                formats=['markdown', 'json']
            )
            
            if report_paths:
                logger.info(f"✅ Processing report generated successfully")
                for format_type, path in report_paths.items():
                    logger.info(f"   📄 {format_type.upper()}: {path}")
                
                return report_paths.get('markdown', '')
            else:
                logger.warning("⚠️  Report generation failed")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error generating report: {str(e)}")
            return ""
    
    def save_analysis_results(self, output_path: Optional[str] = None):
        """Save comprehensive analysis results."""
        if not output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"large_dataset_analysis_{timestamp}.json"
        
        try:
            import json
            
            # Combine all results
            full_results = {
                'metadata': {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'processor_version': '1.0',
                    'processing_config': self.config.to_dict()
                },
                'dataset_info': self.dataset_info,
                'analysis_results': self.analysis_results
            }
            
            # Save with proper JSON serialization
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 Analysis results saved: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {str(e)}")
    
    def process_dataset(self, dataset_path: str, sample_size: Optional[int] = None) -> bool:
        """
        Main processing pipeline for the large dataset.
        
        Args:
            dataset_path: Path to the dataset file
            sample_size: Optional sample size for analysis
            
        Returns:
            True if processing was successful, False otherwise
        """
        logger.info("🚀 Starting large dataset processing pipeline...")
        start_time = time.time()
        
        try:
            # 1. Analyze dataset structure
            logger.info("🔍 Step 1: Analyzing dataset structure")
            structure_info = self.analyze_dataset_structure(dataset_path)
            self.dataset_info['structure'] = structure_info
            
            if not structure_info.get('exists', False):
                logger.error("❌ Dataset file not found. Aborting.")
                return False
            
            # 2. Load and validate dataset
            logger.info("📂 Step 2: Loading and validating dataset")
            data = self.load_and_validate_dataset(dataset_path, sample_size)
            
            if data is None:
                logger.error("❌ Dataset loading failed. Aborting.")
                return False
            
            # 3. Perform comprehensive analysis
            logger.info("📈 Step 3: Performing comprehensive analysis")
            self.analysis_results = self.perform_comprehensive_analysis(data)
            
            # 4. Generate reports
            logger.info("📄 Step 4: Generating reports")
            report_path = self.generate_processing_report()
            
            # 5. Save results
            logger.info("💾 Step 5: Saving analysis results")
            self.save_analysis_results()
            
            # Summary
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info("🎉 Large dataset processing completed successfully!")
            logger.info(f"⏱️  Total processing time: {processing_time:.2f} seconds")
            logger.info(f"📊 Dataset shape: {self.dataset_info.get('shape', 'Unknown')}")
            logger.info(f"🎯 Quality score: {self.dataset_info.get('quality_score', 'Unknown')}")
            logger.info(f"📁 Output directory: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Processing pipeline failed: {str(e)}")
            return False

def main():
    """Main entry point for the large dataset processor."""
    parser = argparse.ArgumentParser(description="Process large music dataset (1.2M songs)")
    parser.add_argument('--sample-size', type=int, help='Sample size for initial analysis (default: use full dataset)')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset file (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        # Auto-detect dataset path
        config = get_config()
        dataset_path = str(config.get_data_path('original'))
    
    # Initialize processor
    processor = LargeDatasetProcessor(output_dir=args.output_dir)
    
    # Process dataset
    success = processor.process_dataset(
        dataset_path=dataset_path,
        sample_size=args.sample_size
    )
    
    if success:
        print("\n🎉 Large dataset processing completed successfully!")
        print(f"📁 Check results in: {processor.output_dir}")
        sys.exit(0)
    else:
        print("\n❌ Large dataset processing failed!")
        print("📋 Check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()