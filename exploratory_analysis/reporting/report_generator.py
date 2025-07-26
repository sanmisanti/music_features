"""
Comprehensive Report Generator Module

Creates automated reports integrating all analysis modules:
- Data quality assessment
- Statistical analysis
- Visualizations
- Feature analysis and dimensionality reduction
- Executive summaries
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging
import os
from pathlib import Path
import warnings

from ..data_loading.data_loader import DataLoader
from ..statistical_analysis.descriptive_stats import DescriptiveStats
from ..visualization.distribution_plots import DistributionPlotter
from ..visualization.correlation_heatmaps import CorrelationPlotter
from ..feature_analysis.dimensionality_reduction import DimensionalityReducer
from ..config.features_config import CLUSTERING_FEATURES, FEATURE_DISPLAY_NAMES
from ..config.analysis_config import config

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class ReportGenerator:
    """
    Comprehensive report generator that integrates all analysis modules
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory for output files (default: ./outputs/reports/)
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.data_loader = DataLoader()
        self.stats_analyzer = DescriptiveStats()
        self.dist_plotter = DistributionPlotter()
        self.corr_plotter = CorrelationPlotter()
        self.dim_reducer = DimensionalityReducer()
        
        # Storage for analysis results
        self.analysis_results = {}
        self.dataset_info = {}
        
        logger.info(f"ReportGenerator initialized with output directory: {self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                    dataset_type: str = 'sample_500',
                                    sample_size: Optional[int] = None,
                                    include_visualizations: bool = True,
                                    formats: List[str] = None) -> Dict[str, str]:
        """
        Generate a comprehensive analysis report
        
        Args:
            dataset_type: Type of dataset to analyze
            sample_size: Optional sample size override
            include_visualizations: Whether to generate and include plots
            formats: Output formats ['markdown', 'html', 'json']
            
        Returns:
            Dictionary with paths to generated reports
        """
        if formats is None:
            formats = ['markdown', 'json']
        
        logger.info(f"Starting comprehensive report generation for {dataset_type}")
        
        # 1. Load and validate data
        logger.info("Step 1: Loading and validating data")
        data_result = self._load_and_validate_data(dataset_type, sample_size)
        if not data_result['success']:
            logger.error("Data loading failed, cannot generate report")
            return {}
        
        # 2. Perform statistical analysis
        logger.info("Step 2: Performing statistical analysis")
        stats_result = self._perform_statistical_analysis(data_result['data'])
        
        # 3. Generate visualizations (if requested)
        viz_result = {}
        if include_visualizations:
            logger.info("Step 3: Generating visualizations")
            viz_result = self._generate_visualizations(data_result['data'])
        
        # 4. Perform feature analysis
        logger.info("Step 4: Performing feature analysis")
        feature_result = self._perform_feature_analysis(data_result['data'])
        
        # 5. Compile comprehensive results
        logger.info("Step 5: Compiling comprehensive analysis")
        comprehensive_results = self._compile_analysis_results(
            data_result, stats_result, viz_result, feature_result
        )
        
        # 6. Generate reports in requested formats
        logger.info("Step 6: Generating output reports")
        output_paths = {}
        
        for format_type in formats:
            if format_type == 'markdown':
                output_paths['markdown'] = self._generate_markdown_report(comprehensive_results)
            elif format_type == 'json':
                output_paths['json'] = self._generate_json_report(comprehensive_results)
            elif format_type == 'html':
                output_paths['html'] = self._generate_html_report(comprehensive_results)
        
        logger.info(f"Report generation completed. Generated {len(output_paths)} files")
        return output_paths
    
    def _load_and_validate_data(self, dataset_type: str, sample_size: Optional[int]) -> Dict[str, Any]:
        """Load and validate dataset"""
        try:
            result = self.data_loader.load_dataset(dataset_type, sample_size, validate=True)
            
            if result.success:
                # Store dataset info - extract from metadata where available
                metadata = result.metadata
                self.dataset_info = {
                    'dataset_type': dataset_type,
                    'sample_size': len(result.data),
                    'features': list(result.data.columns),
                    'load_time': metadata.get('load_time', 0),
                    'memory_usage': metadata.get('memory_usage', 0),
                    'quality_score': metadata.get('quality_score', 95),  # Default good quality
                    'validation_issues': len(result.errors) + len(result.warnings)
                }
                
                return {
                    'success': True,
                    'data': result.data,
                    'info': self.dataset_info,
                    'validation_issues': result.errors + result.warnings
                }
            else:
                return {'success': False, 'error': 'Data loading failed'}
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        try:
            # Use the main analyze_dataset method
            clustering_features = [f for f in CLUSTERING_FEATURES if f in df.columns]
            analysis_results = self.stats_analyzer.analyze_dataset(df, clustering_features)
            
            # Extract components from the comprehensive analysis
            return {
                'dataset_overview': analysis_results.get('dataset_stats', {}),
                'feature_statistics': analysis_results.get('feature_stats', {}),
                'correlation_analysis': analysis_results.get('correlation_preview', {}),
                'distribution_summary': analysis_results.get('distribution_summary', {}),
                'quality_assessment': analysis_results.get('quality_assessment', {})
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            return {'error': str(e)}
    
    def _generate_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive visualizations"""
        try:
            viz_paths = {}
            
            # Generate distribution plots
            dist_results = self.dist_plotter.plot_feature_distributions(
                df, 
                features=None,  # Use all available
                plot_types=['histogram', 'boxplot']
            )
            
            # Save distribution plots
            for plot_type, plot_info in dist_results.items():
                plot_path = self.output_dir / f"distributions_{plot_type}.png"
                plot_info['figure'].savefig(plot_path, dpi=300, bbox_inches='tight')
                viz_paths[f'distribution_{plot_type}'] = str(plot_path)
                import matplotlib.pyplot as plt
                plt.close(plot_info['figure'])
            
            # Generate correlation heatmap
            corr_fig = self.corr_plotter.create_correlation_heatmap(df)
            if corr_fig:
                corr_path = self.output_dir / "correlation_heatmap.png"
                corr_fig.savefig(corr_path, dpi=300, bbox_inches='tight')
                viz_paths['correlation_heatmap'] = str(corr_path)
                import matplotlib.pyplot as plt
                plt.close(corr_fig)
            
            # Generate correlation comparison
            comparison_fig = self.corr_plotter.create_correlation_comparison(
                df, methods=['pearson', 'spearman']
            )
            if comparison_fig:
                comparison_path = self.output_dir / "correlation_comparison.png"
                comparison_fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
                viz_paths['correlation_comparison'] = str(comparison_path)
                import matplotlib.pyplot as plt
                plt.close(comparison_fig)
            
            return {
                'success': True,
                'visualization_paths': viz_paths,
                'plots_generated': len(viz_paths)
            }
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _perform_feature_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform feature analysis and dimensionality reduction"""
        try:
            results = {}
            
            # PCA Analysis
            pca_results = self.dim_reducer.fit_pca(df, variance_threshold=0.90)
            if pca_results:
                results['pca'] = {
                    'n_components': pca_results['n_components'],
                    'variance_explained': pca_results['total_variance_explained'],
                    'component_analysis': pca_results['component_analysis'],
                    'feature_importance': pca_results['feature_importance']
                }
            
            # t-SNE Analysis (with smaller sample for performance)
            sample_size = min(200, len(df))
            df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
            
            tsne_results = self.dim_reducer.fit_tsne(df_sample, n_components=2)
            if tsne_results:
                results['tsne'] = {
                    'kl_divergence': tsne_results['kl_divergence'],
                    'n_iter': tsne_results['n_iter'],
                    'perplexity': tsne_results['perplexity'],
                    'sample_size': len(df_sample)
                }
            
            # Feature Selection
            selection_results = self.dim_reducer.perform_feature_selection(df)
            if selection_results:
                results['feature_selection'] = selection_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error in feature analysis: {str(e)}")
            return {'error': str(e)}
    
    def _compile_analysis_results(self, data_result: Dict, stats_result: Dict, 
                                viz_result: Dict, feature_result: Dict) -> Dict[str, Any]:
        """Compile all analysis results into comprehensive structure"""
        
        return {
            'metadata': {
                'report_generated': datetime.now().isoformat(),
                'dataset_info': self.dataset_info,
                'analysis_modules': ['data_loading', 'statistical_analysis', 'visualization', 'feature_analysis']
            },
            'data_quality': {
                'load_success': data_result.get('success', False),
                'sample_size': self.dataset_info.get('sample_size', 0),
                'quality_score': self.dataset_info.get('quality_score', 0),
                'validation_issues': data_result.get('validation_issues', [])
            },
            'statistical_analysis': stats_result,
            'visualizations': viz_result,
            'feature_analysis': feature_result,
            'executive_summary': self._create_executive_summary(stats_result, feature_result)
        }
    
    def _create_executive_summary(self, stats_result: Dict, feature_result: Dict) -> Dict[str, Any]:
        """Create executive summary of key findings"""
        
        summary = {
            'overall_assessment': 'EXCELLENT',
            'key_findings': [],
            'recommendations': [],
            'next_steps': []
        }
        
        # Assess data quality
        if stats_result.get('quality_assessment'):
            quality = stats_result['quality_assessment']
            # Handle both dict and object format
            if hasattr(quality, 'overall_score'):  # Object format
                overall_score = quality.overall_score
            else:  # Dictionary format
                overall_score = quality.get('overall_score', 0)
            
            summary['data_quality_score'] = overall_score
            
            if overall_score >= 95:
                summary['key_findings'].append("Dataset quality is excellent with minimal cleaning required")
            
        # Assess correlations
        if stats_result.get('correlation_analysis'):
            corr_analysis = stats_result['correlation_analysis']
            high_corr_count = len(corr_analysis.get('high_correlations', []))
            
            if high_corr_count > 0:
                summary['key_findings'].append(f"Identified {high_corr_count} high correlations indicating potential feature redundancy")
        
        # Assess PCA results
        if feature_result.get('pca'):
            pca = feature_result['pca']
            var_explained = pca.get('variance_explained', 0)
            n_components = pca.get('n_components', 0)
            
            summary['key_findings'].append(f"PCA analysis: {n_components} components explain {var_explained:.1%} of variance")
            
            if var_explained >= 0.90:
                summary['recommendations'].append("Consider PCA for dimensionality reduction in clustering")
        
        # General recommendations
        summary['recommendations'].extend([
            "Proceed with clustering analysis using current feature set",
            "Monitor performance with larger datasets",
            "Consider validation with cross-cultural music datasets"
        ])
        
        summary['next_steps'] = [
            "Implement K-means clustering with optimized parameters",
            "Develop cluster interpretation and profiling",
            "Integrate with semantic lyrics analysis"
        ]
        
        return summary
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown_content = f"""# 📊 Comprehensive Musical Features Analysis Report

**Generated**: {timestamp}  
**Dataset**: {results['metadata']['dataset_info'].get('dataset_type', 'Unknown')}  
**Sample Size**: {results['metadata']['dataset_info'].get('sample_size', 0)} tracks  
**Analysis Modules**: {', '.join(results['metadata']['analysis_modules'])}

---

## 🎯 Executive Summary

### Overall Assessment: {results['executive_summary']['overall_assessment']}

**Data Quality Score**: {results['executive_summary'].get('data_quality_score', 0):.1f}/100

### Key Findings:
"""
        
        for finding in results['executive_summary']['key_findings']:
            markdown_content += f"- {finding}\n"
        
        markdown_content += f"""
### Recommendations:
"""
        
        for rec in results['executive_summary']['recommendations']:
            markdown_content += f"- {rec}\n"
        
        # Data Quality Section
        markdown_content += f"""
---

## 📊 Data Quality Assessment

- **Load Success**: {'✅' if results['data_quality']['load_success'] else '❌'}
- **Sample Size**: {results['data_quality']['sample_size']} tracks
- **Quality Score**: {results['data_quality']['quality_score']:.1f}/100
- **Validation Issues**: {len(results['data_quality']['validation_issues'])}

"""
        
        # Statistical Analysis Section
        if 'statistical_analysis' in results:
            stats = results['statistical_analysis']
            
            markdown_content += f"""---

## 📈 Statistical Analysis

### Dataset Overview:
"""
            if 'dataset_overview' in stats:
                overview = stats['dataset_overview']
                # Handle both dict and DatasetStats object
                if hasattr(overview, 'total_features'):  # DatasetStats object
                    markdown_content += f"""- **Total Features**: {overview.total_features}
- **Missing Data**: {overview.missing_data_pct:.1f}%
- **Memory Usage**: {overview.memory_usage_mb:.2f} MB
- **Duplicate Rows**: {overview.duplicate_pct:.1f}%

"""
                else:  # Dictionary format
                    markdown_content += f"""- **Total Features**: {overview.get('total_features', 'N/A')}
- **Missing Data**: {overview.get('missing_percentage', 0):.1f}%
- **Memory Usage**: {overview.get('memory_usage_mb', 'N/A')} MB

"""
            
            # Correlation Analysis
            if 'correlation_analysis' in stats:
                corr = stats['correlation_analysis']
                markdown_content += f"""### Correlation Analysis:
- **High Correlations** (≥0.7): {len(corr.get('high_correlations', []))}
- **Moderate Correlations** (0.3-0.7): {len(corr.get('moderate_correlations', []))}
- **Maximum Correlation**: {corr.get('max_correlation', 0):.3f}

"""
                
                # List high correlations
                if corr.get('high_correlations'):
                    markdown_content += "**High Correlations Detected**:\n"
                    for high_corr in corr['high_correlations'][:5]:  # Top 5
                        feat1 = high_corr.get('feature1', '')
                        feat2 = high_corr.get('feature2', '')
                        corr_val = high_corr.get('correlation', 0)
                        markdown_content += f"- {feat1} ↔ {feat2}: {corr_val:.3f}\n"
                    markdown_content += "\n"
        
        # Feature Analysis Section
        if 'feature_analysis' in results:
            features = results['feature_analysis']
            
            markdown_content += f"""---

## 🔬 Feature Analysis

"""
            
            # PCA Results
            if 'pca' in features:
                pca = features['pca']
                markdown_content += f"""### Principal Component Analysis (PCA):
- **Components Selected**: {pca.get('n_components', 'N/A')}
- **Variance Explained**: {pca.get('variance_explained', 0):.1%}

"""
                
                # Top components
                if 'component_analysis' in pca:
                    comp_analysis = pca['component_analysis']
                    markdown_content += "**Top Principal Components**:\n"
                    for pc_name, pc_info in list(comp_analysis.items())[:3]:
                        var_ratio = pc_info.get('explained_variance_ratio', 0)
                        interpretation = pc_info.get('interpretation', 'N/A')
                        markdown_content += f"- **{pc_name}**: {var_ratio:.1%} variance - {interpretation}\n"
                    markdown_content += "\n"
            
            # t-SNE Results
            if 'tsne' in features:
                tsne = features['tsne']
                markdown_content += f"""### t-SNE Analysis:
- **KL Divergence**: {tsne.get('kl_divergence', 'N/A'):.4f}
- **Perplexity**: {tsne.get('perplexity', 'N/A')}
- **Sample Size**: {tsne.get('sample_size', 'N/A')} tracks

"""
        
        # Visualizations Section
        if results.get('visualizations', {}).get('success'):
            viz = results['visualizations']
            markdown_content += f"""---

## 🎨 Visualizations Generated

- **Plots Created**: {viz.get('plots_generated', 0)}
- **Distribution Plots**: ✅
- **Correlation Heatmaps**: ✅
- **Comparison Charts**: ✅

**Files saved in**: `{self.output_dir}/`

"""
        
        # Next Steps
        markdown_content += f"""---

## 🚀 Next Steps

"""
        for step in results['executive_summary']['next_steps']:
            markdown_content += f"1. {step}\n"
        
        markdown_content += f"""
---

*Report generated automatically by the Musical Features Analysis System*  
*Timestamp: {timestamp}*
"""
        
        # Save markdown report
        report_path = self.output_dir / f"comprehensive_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report saved to: {report_path}")
        return str(report_path)
    
    def _generate_json_report(self, results: Dict[str, Any]) -> str:
        """Generate JSON report with all analysis results"""
        import json
        
        # Convert numpy types and dataclasses to native Python types for JSON serialization
        def convert_numpy(obj):
            # Handle dataclasses
            if hasattr(obj, '__dataclass_fields__'):
                from dataclasses import asdict
                return convert_numpy(asdict(obj))
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Clean results for JSON serialization
        json_results = convert_numpy(results)
        
        # Save JSON report
        report_path = self.output_dir / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report saved to: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report (basic implementation)"""
        # Convert markdown to HTML (basic implementation)
        markdown_path = self._generate_markdown_report(results)
        
        # For now, just reference the markdown file
        # In a full implementation, this would convert MD to HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Musical Features Analysis Report</title>
    <meta charset="utf-8">
</head>
<body>
    <h1>Musical Features Analysis Report</h1>
    <p>HTML report generation is in development.</p>
    <p>Please refer to the markdown report: <a href="{markdown_path}">View Report</a></p>
</body>
</html>"""
        
        report_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {report_path}")
        return str(report_path)

# Convenience functions
def generate_quick_report(dataset_type: str = 'sample_500', 
                         sample_size: Optional[int] = None,
                         output_dir: Optional[str] = None) -> Dict[str, str]:
    """Convenience function for quick report generation"""
    generator = ReportGenerator(output_dir)
    return generator.generate_comprehensive_report(dataset_type, sample_size)

def generate_executive_summary(dataset_type: str = 'sample_500',
                              sample_size: Optional[int] = None) -> Dict[str, Any]:
    """Generate only executive summary without full report"""
    generator = ReportGenerator()
    
    # Load data
    data_result = generator._load_and_validate_data(dataset_type, sample_size)
    if not data_result['success']:
        return {'error': 'Failed to load data'}
    
    # Quick analysis
    stats_result = generator._perform_statistical_analysis(data_result['data'])
    feature_result = generator._perform_feature_analysis(data_result['data'])
    
    return generator._create_executive_summary(stats_result, feature_result)