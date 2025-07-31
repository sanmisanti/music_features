#!/usr/bin/env python3
"""
🎯 HYBRID SELECTION PIPELINE WITH LYRICS VERIFICATION
=====================================================
Complete orchestration of the hybrid song selection process with lyrics availability optimization.

This script integrates all components for the complete pipeline:
1. Large dataset analysis (1.2M songs) 
2. Hybrid representative selection with lyrics verification (10K songs, 80% with lyrics)
3. Selection validation and quality assessment
4. Comprehensive reporting and documentation

Usage:
    python scripts/main_selection_pipeline.py [--target-size 10000] [--output-dir DIR] [--skip-analysis]
"""

import sys
import os
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exploratory_analysis.config.analysis_config import get_config, configure_for_large_dataset
from exploratory_analysis.selection_pipeline.data_processor import LargeDatasetProcessor
from exploratory_analysis.selection_pipeline.representative_selector import RepresentativeSelector
from exploratory_analysis.selection_pipeline.selection_validator import SelectionValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/main_selection_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MainSelectionPipeline:
    """
    Orchestrates the complete hybrid song selection pipeline with lyrics verification.
    
    Integrates dataset analysis, hybrid representative selection with lyrics availability,
    validation, and reporting into a seamless automated process targeting 80% lyrics coverage.
    """
    
    def __init__(self, target_size: int = 10000, output_dir: Optional[str] = None, skip_analysis: bool = False):
        """
        Initialize the main selection pipeline.
        
        Args:
            target_size: Number of songs to select
            output_dir: Directory for all output files  
            skip_analysis: Skip large dataset analysis if already done
        """
        # Configure for large dataset processing
        configure_for_large_dataset()
        self.config = get_config()
        self.target_size = target_size
        self.skip_analysis = skip_analysis
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"outputs/selection_pipeline_{timestamp}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.analysis_dir = self.output_dir / "analysis"
        self.selection_dir = self.output_dir / "selection"
        self.validation_dir = self.output_dir / "validation"
        self.reports_dir = self.output_dir / "reports"
        
        for subdir in [self.analysis_dir, self.selection_dir, self.validation_dir, self.reports_dir]:
            subdir.mkdir(exist_ok=True)
        
        # Initialize components
        self.processor = LargeDatasetProcessor(output_dir=str(self.analysis_dir))
        self.selector = RepresentativeSelector(target_size=target_size, output_dir=str(self.selection_dir), target_lyrics_ratio=0.8)
        self.validator = SelectionValidator(output_dir=str(self.validation_dir))
        
        # Pipeline results
        self.pipeline_results = {
            'metadata': {
                'target_size': target_size,
                'skip_analysis': skip_analysis,
                'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'pipeline_version': '1.0'
            },
            'stages': {},
            'final_results': {}
        }
        
        logger.info(f"🎯 Hybrid Selection Pipeline initialized")
        logger.info(f"📊 Target size: {target_size:,} songs")
        logger.info(f"🎵 Target lyrics ratio: 80% (8,000 with lyrics, 2,000 without)")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🔍 Skip analysis: {skip_analysis}")
    
    def detect_dataset_path(self) -> str:
        """Auto-detect the path to the original dataset."""
        try:
            # Try configured path for cleaned dataset first
            config_path = self.config.get_data_path('cleaned_full')
            if config_path.exists():
                return str(config_path)
            
            # Try alternative paths
            alternative_paths = [
                "data/cleaned_data/tracks_features_clean.csv",
                "data/original_data/tracks_features.csv",
                "data/tracks_features.csv",
                "../data/cleaned_data/tracks_features_clean.csv"
            ]
            
            for alt_path in alternative_paths:
                path = Path(alt_path)
                if path.exists():
                    logger.info(f"📂 Found dataset at: {path}")
                    return str(path)
            
            logger.error("❌ Could not auto-detect dataset path")
            return str(config_path)  # Return configured path even if not found
            
        except Exception as e:
            logger.error(f"❌ Error detecting dataset path: {str(e)}")
            return "data/cleaned_data/tracks_features_clean.csv"  # Default fallback
    
    def stage_1_dataset_analysis(self, dataset_path: str) -> bool:
        """
        Stage 1: Comprehensive analysis of the large dataset.
        
        Args:
            dataset_path: Path to the original dataset
            
        Returns:
            True if analysis completed successfully
        """
        logger.info("🔍 STAGE 1: Large Dataset Analysis")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            if self.skip_analysis:
                logger.info("⏭️  Skipping large dataset analysis (--skip-analysis flag)")
                self.pipeline_results['stages']['analysis'] = {
                    'skipped': True,
                    'reason': 'User requested skip',
                    'execution_time': 0
                }
                return True
            
            # Run large dataset processing
            logger.info(f"📂 Processing dataset: {dataset_path}")
            success = self.processor.process_dataset(dataset_path, sample_size=None)
            
            execution_time = time.time() - start_time
            
            if success:
                logger.info("✅ Stage 1 completed successfully")
                logger.info(f"⏱️  Analysis time: {execution_time:.2f} seconds")
                
                self.pipeline_results['stages']['analysis'] = {
                    'success': True,
                    'execution_time': execution_time,
                    'dataset_path': dataset_path,
                    'output_dir': str(self.analysis_dir)
                }
                
                return True
            else:
                logger.error("❌ Stage 1 failed")
                self.pipeline_results['stages']['analysis'] = {
                    'success': False,
                    'execution_time': execution_time,
                    'error': 'Large dataset processing failed'
                }
                return False
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Stage 1 error: {str(e)}")
            self.pipeline_results['stages']['analysis'] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            return False
    
    def stage_2_representative_selection(self, dataset_path: str) -> Optional[str]:
        """
        Stage 2: Hybrid selection with lyrics verification (80% with lyrics target).
        
        Args:
            dataset_path: Path to the original dataset
            
        Returns:
            Path to selected dataset file, or None if failed
        """
        logger.info("🎯 STAGE 2: Hybrid Selection with Lyrics Verification")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Run representative selection
            selected_data, validation_results = self.selector.select_representative_songs(dataset_path)
            
            execution_time = time.time() - start_time
            
            if selected_data is not None:
                # Save selected dataset
                selected_path = self.selector.save_selected_dataset(selected_data)
                
                logger.info("✅ Stage 2 completed successfully")
                logger.info(f"📊 Selected {len(selected_data):,} songs")
                logger.info(f"🎯 Selection quality: {validation_results.get('overall_quality', 0):.1f}/100")
                logger.info(f"⏱️  Selection time: {execution_time:.2f} seconds")
                
                self.pipeline_results['stages']['selection'] = {
                    'success': True,
                    'execution_time': execution_time,
                    'selected_size': len(selected_data),
                    'selection_quality': validation_results.get('overall_quality', 0),
                    'selected_dataset_path': selected_path,
                    'output_dir': str(self.selection_dir),
                    'validation_summary': validation_results
                }
                
                return selected_path
            else:
                logger.error("❌ Stage 2 failed")
                self.pipeline_results['stages']['selection'] = {
                    'success': False,
                    'execution_time': execution_time,
                    'error': 'Representative selection failed'
                }
                return None
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Stage 2 error: {str(e)}")
            self.pipeline_results['stages']['selection'] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            return None
    
    def stage_3_selection_validation(self, original_path: str, selected_path: str) -> bool:
        """
        Stage 3: Validate the quality of the selection.
        
        Args:
            original_path: Path to original dataset
            selected_path: Path to selected dataset
            
        Returns:
            True if validation completed successfully
        """
        logger.info("🔍 STAGE 3: Selection Validation")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Run comprehensive validation
            validation_results = self.validator.run_comprehensive_validation(original_path, selected_path)
            
            execution_time = time.time() - start_time
            
            if validation_results.get('success', False):
                logger.info("✅ Stage 3 completed successfully")
                logger.info(f"📊 Overall validation score: {validation_results['overall_score']:.1f}/100")
                logger.info(f"✅ Tests passed: {validation_results['tests_passed']}/{validation_results['total_tests']}")
                logger.info(f"⏱️  Validation time: {execution_time:.2f} seconds")
                
                self.pipeline_results['stages']['validation'] = {
                    'success': True,
                    'execution_time': execution_time,
                    'validation_score': validation_results['overall_score'],
                    'tests_passed': validation_results['tests_passed'],
                    'total_tests': validation_results['total_tests'],
                    'pass_rate': validation_results['pass_rate'],
                    'output_dir': str(self.validation_dir),
                    'detailed_results': validation_results
                }
                
                return True
            else:
                logger.error("❌ Stage 3 failed")
                self.pipeline_results['stages']['validation'] = {
                    'success': False,
                    'execution_time': execution_time,
                    'error': validation_results.get('error', 'Validation failed')
                }
                return False
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Stage 3 error: {str(e)}")
            self.pipeline_results['stages']['validation'] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            return False
    
    def stage_4_final_reporting(self) -> str:
        """
        Stage 4: Generate comprehensive final report.
        
        Returns:
            Path to final report
        """
        logger.info("📄 STAGE 4: Final Reporting")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"final_selection_report_{timestamp}.md"
            
            # Calculate pipeline summary
            total_execution_time = sum(
                stage_data.get('execution_time', 0) 
                for stage_data in self.pipeline_results['stages'].values()
            )
            
            successful_stages = sum(
                1 for stage_data in self.pipeline_results['stages'].values() 
                if stage_data.get('success', False) or stage_data.get('skipped', False)
            )
            
            total_stages = len(self.pipeline_results['stages'])
            pipeline_success = successful_stages == total_stages
            
            # Extract key metrics
            selection_stage = self.pipeline_results['stages'].get('selection', {})
            validation_stage = self.pipeline_results['stages'].get('validation', {})
            
            selected_size = selection_stage.get('selected_size', 0)
            selection_quality = selection_stage.get('selection_quality', 0)
            validation_score = validation_stage.get('validation_score', 0)
            
            # Generate report content
            report_content = f"""# 🎯 Final Selection Pipeline Report

## Executive Summary

**Pipeline Execution Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}  
**Target Selection Size**: {self.target_size:,} songs  
**Pipeline Status**: {'✅ SUCCESS' if pipeline_success else '❌ FAILED'}  
**Total Execution Time**: {total_execution_time:.2f} seconds

### Key Results
- **Songs Selected**: {selected_size:,}
- **Selection Quality**: {selection_quality:.1f}/100
- **Validation Score**: {validation_score:.1f}/100
- **Stages Completed**: {successful_stages}/{total_stages}

## Pipeline Stages

"""
            
            # Add stage details
            stage_names = {
                'analysis': '🔍 Large Dataset Analysis',
                'selection': '🎯 Hybrid Selection with Lyrics Verification', 
                'validation': '🔍 Selection Validation'
            }
            
            for stage_key, stage_data in self.pipeline_results['stages'].items():
                stage_name = stage_names.get(stage_key, stage_key.title())
                
                if stage_data.get('skipped', False):
                    status = "⏭️ SKIPPED"
                    details = f"Reason: {stage_data.get('reason', 'Unknown')}"
                elif stage_data.get('success', False):
                    status = "✅ SUCCESS"
                    exec_time = stage_data.get('execution_time', 0)
                    details = f"Execution time: {exec_time:.2f}s"
                else:
                    status = "❌ FAILED"
                    error = stage_data.get('error', 'Unknown error')
                    details = f"Error: {error}"
                
                report_content += f"""### {stage_name}
**Status**: {status}  
**Details**: {details}

"""
                
                # Add stage-specific details
                if stage_key == 'selection' and stage_data.get('success', False):
                    report_content += f"""**Selection Results**:
- Selected songs: {stage_data.get('selected_size', 0):,}
- Selection quality: {stage_data.get('selection_quality', 0):.1f}/100
- Output directory: `{stage_data.get('output_dir', 'N/A')}`

"""
                
                if stage_key == 'validation' and stage_data.get('success', False):
                    report_content += f"""**Validation Results**:
- Overall score: {stage_data.get('validation_score', 0):.1f}/100
- Tests passed: {stage_data.get('tests_passed', 0)}/{stage_data.get('total_tests', 0)}
- Pass rate: {stage_data.get('pass_rate', 0):.1%}

"""
            
            # Recommendations
            report_content += """## Recommendations

"""
            
            if pipeline_success:
                if validation_score >= 80:
                    report_content += """### ✅ EXCELLENT QUALITY - READY FOR USE

The selected dataset demonstrates excellent representativeness and is ready for use in the final recommendation system.

**Next Steps**:
1. Use the selected dataset for model training
2. Proceed with multimodal system development
3. Monitor model performance with this representative subset

"""
                elif validation_score >= 70:
                    report_content += """### ✅ GOOD QUALITY - USABLE WITH MONITORING

The selected dataset shows good representativeness with minor areas for improvement.

**Next Steps**:
1. Review validation report for specific recommendations
2. Consider the selected dataset suitable for model development
3. Monitor model performance and adjust if needed

"""
                else:
                    report_content += """### ⚠️ MODERATE QUALITY - REVIEW RECOMMENDED

The selected dataset has some representativeness concerns that should be addressed.

**Next Steps**:
1. Review detailed validation results
2. Consider refining the selection process
3. Evaluate if the dataset is suitable for your specific use case

"""
            else:
                report_content += """### ❌ PIPELINE FAILED

The selection pipeline encountered errors that prevented successful completion.

**Next Steps**:
1. Review error logs for specific issues
2. Check dataset availability and format
3. Retry pipeline with appropriate fixes

"""
            
            # File locations
            report_content += f"""## Output Files and Locations

### Main Output Directory
`{self.output_dir}`

### Subdirectories
- **Analysis Results**: `{self.analysis_dir}`
- **Selection Results**: `{self.selection_dir}`  
- **Validation Results**: `{self.validation_dir}`
- **Reports**: `{self.reports_dir}`

### Key Files
"""
            
            # List key output files
            if selection_stage.get('selected_dataset_path'):
                report_content += f"- **Selected Dataset**: `{selection_stage['selected_dataset_path']}`\n"
            
            if validation_stage.get('detailed_results', {}).get('report_path'):
                report_content += f"- **Validation Report**: `{validation_stage['detailed_results']['report_path']}`\n"
            
            report_content += f"""
### Log Files
- **Pipeline Log**: `outputs/main_selection_pipeline.log`
- **Analysis Log**: `outputs/large_dataset_processing.log`
- **Selection Log**: `outputs/representative_selection.log` 
- **Validation Log**: `outputs/selection_validation.log`

## Technical Details

### Configuration
- Target size: {self.target_size:,} songs
- Skip analysis: {self.skip_analysis}
- Pipeline version: {self.pipeline_results['metadata']['pipeline_version']}

### Performance Metrics
- Total execution time: {total_execution_time:.2f} seconds
- Average stage time: {total_execution_time / max(1, total_stages):.2f} seconds

---
*Report generated by Main Selection Pipeline v1.0*  
*Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}*
"""
            
            # Save report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Also save JSON summary
            json_path = self.reports_dir / f"pipeline_summary_{timestamp}.json"
            self.pipeline_results['final_results'] = {
                'pipeline_success': pipeline_success,
                'total_execution_time': total_execution_time,
                'selected_size': selected_size,
                'selection_quality': selection_quality,
                'validation_score': validation_score,
                'successful_stages': successful_stages,
                'total_stages': total_stages
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_results, f, indent=2, ensure_ascii=False, default=str)
            
            execution_time = time.time() - start_time
            
            logger.info("✅ Stage 4 completed successfully")
            logger.info(f"📄 Final report generated: {report_path}")
            logger.info(f"📊 JSON summary saved: {json_path}")
            logger.info(f"⏱️  Reporting time: {execution_time:.2f} seconds")
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"❌ Stage 4 error: {str(e)}")
            return ""
    
    def run_complete_pipeline(self, dataset_path: Optional[str] = None) -> bool:
        """
        Run the complete selection pipeline.
        
        Args:
            dataset_path: Path to dataset (auto-detected if None)
            
        Returns:
            True if pipeline completed successfully
        """
        logger.info("🚀 STARTING MAIN SELECTION PIPELINE")
        logger.info("=" * 70)
        
        total_start_time = time.time()
        
        try:
            # Auto-detect dataset path if not provided
            if not dataset_path:
                dataset_path = self.detect_dataset_path()
            
            logger.info(f"📂 Dataset path: {dataset_path}")
            logger.info(f"🎯 Target selection size: {self.target_size:,}")
            logger.info(f"📁 Output directory: {self.output_dir}")
            
            # Stage 1: Dataset Analysis
            stage1_success = self.stage_1_dataset_analysis(dataset_path)
            if not stage1_success and not self.skip_analysis:
                logger.error("❌ Pipeline failed at Stage 1")
                return False
            
            # Stage 2: Representative Selection
            selected_path = self.stage_2_representative_selection(dataset_path)
            if not selected_path:
                logger.error("❌ Pipeline failed at Stage 2")
                return False
            
            # Stage 3: Selection Validation
            stage3_success = self.stage_3_selection_validation(dataset_path, selected_path)
            if not stage3_success:
                logger.warning("⚠️  Validation failed, but continuing with reporting")
            
            # Stage 4: Final Reporting
            report_path = self.stage_4_final_reporting()
            
            # Pipeline summary
            total_time = time.time() - total_start_time
            final_results = self.pipeline_results.get('final_results', {})
            
            logger.info("🎉 PIPELINE COMPLETED!")
            logger.info("=" * 70)
            logger.info(f"📊 Final Results:")
            logger.info(f"   Selected songs: {final_results.get('selected_size', 0):,}")
            logger.info(f"   Selection quality: {final_results.get('selection_quality', 0):.1f}/100")
            logger.info(f"   Validation score: {final_results.get('validation_score', 0):.1f}/100")
            logger.info(f"⏱️  Total pipeline time: {total_time:.2f} seconds")
            logger.info(f"📄 Final report: {report_path}")
            logger.info(f"📁 All outputs in: {self.output_dir}")
            
            return True
            
        except Exception as e:
            total_time = time.time() - total_start_time
            logger.error(f"❌ PIPELINE FAILED: {str(e)}")
            logger.error(f"⏱️  Time before failure: {total_time:.2f} seconds")
            return False

def main():
    """Main entry point for the selection pipeline."""
    parser = argparse.ArgumentParser(description="Complete song selection pipeline")
    parser.add_argument('--target-size', type=int, default=10000, 
                       help='Number of songs to select (default: 10000)')
    parser.add_argument('--output-dir', type=str, 
                       help='Output directory for all results')
    parser.add_argument('--dataset-path', type=str, 
                       help='Path to dataset file (default: auto-detect)')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip large dataset analysis stage')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display startup banner
    print("🎯 MAIN SELECTION PIPELINE")
    print("=" * 50)
    print(f"Target size: {args.target_size:,} songs")
    print(f"Skip analysis: {args.skip_analysis}")
    print(f"Output directory: {args.output_dir or 'auto-generated'}")
    print("=" * 50)
    
    # Initialize and run pipeline
    pipeline = MainSelectionPipeline(
        target_size=args.target_size,
        output_dir=args.output_dir,
        skip_analysis=args.skip_analysis
    )
    
    success = pipeline.run_complete_pipeline(args.dataset_path)
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        print(f"📁 Check all results in: {pipeline.output_dir}")
        
        # Show key results
        final_results = pipeline.pipeline_results.get('final_results', {})
        if final_results:
            print(f"📊 Selected: {final_results.get('selected_size', 0):,} songs")
            print(f"🎯 Quality: {final_results.get('validation_score', 0):.1f}/100")
        
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        print("📋 Check logs for details")
        print(f"📁 Partial results may be in: {pipeline.output_dir}")
        sys.exit(1)

if __name__ == "__main__":
    main()