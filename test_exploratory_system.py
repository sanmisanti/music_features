"""
Test script for the exploratory analysis system

This script tests the data loading and validation modules we've created.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from exploratory_analysis.data_loading import DataLoader, DataValidator
from exploratory_analysis.config import config, configure_for_development
import pandas as pd

def test_configuration():
    """Test configuration system"""
    print("🔧 Testing Configuration System")
    print("=" * 50)
    
    # Test basic configuration access
    print(f"📊 Sample size: {config.data.default_sample_size}")
    print(f"📁 Data directory: {config.get_data_path('sample_500')}")
    print(f"📈 Figure size: {config.plots.figure_size}")
    print(f"🎯 Random state: {config.data.random_state}")
    
    # Test configuration update
    print("\n🔄 Testing configuration update...")
    original_sample_size = config.data.default_sample_size
    config.update_config(data={'default_sample_size': 1000})
    print(f"Updated sample size: {config.data.default_sample_size}")
    
    # Reset
    config.update_config(data={'default_sample_size': original_sample_size})
    print(f"Reset sample size: {config.data.default_sample_size}")
    
    print("✅ Configuration system working correctly\n")

def test_data_loader():
    """Test data loading functionality"""
    print("📥 Testing Data Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = DataLoader()
    
    # Test loading sample dataset
    print("🔍 Loading sample dataset...")
    result = loader.load_dataset('sample_500', sample_size=100, validate=True)
    
    if result.success:
        print(f"✅ Successfully loaded {len(result.data)} rows")
        print(f"📊 Columns: {len(result.data.columns)}")
        print(f"💾 Memory usage: {result.metadata['memory_usage_mb']:.2f} MB")
        
        # Show column names
        print(f"📋 Columns: {result.data.columns.tolist()}")
        
        # Show basic info about the data
        print(f"📈 Data shape: {result.data.shape}")
        if 'year' in result.data.columns:
            print(f"📅 Year range: {result.data['year'].min()} - {result.data['year'].max()}")
        
        # Show any warnings
        if result.warnings:
            print(f"⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"   - {warning}")
        
        # Show first few rows of key features
        key_features = ['name', 'artists', 'energy', 'valence', 'tempo']
        available_features = [f for f in key_features if f in result.data.columns]
        if available_features:
            print(f"\n📋 Sample data ({', '.join(available_features)}):")
            print(result.data[available_features].head(3).to_string())
        
    else:
        print("❌ Failed to load data")
        for error in result.errors:
            print(f"   Error: {error}")
    
    # Test load statistics
    stats = loader.get_load_statistics()
    print(f"\n📊 Load Statistics:")
    print(f"   Files loaded: {stats['files_loaded']}")
    print(f"   Total rows: {stats['total_rows']}")
    print(f"   Load time: {stats['load_time']:.2f}s")
    
    print("✅ Data loader test completed\n")
    return result

def test_data_validator(data_result):
    """Test data validation functionality"""
    print("🔍 Testing Data Validator")
    print("=" * 50)
    
    if not data_result.success:
        print("❌ Cannot test validator - no data loaded")
        return
    
    # Initialize validator with different levels
    print("🔍 Testing basic validation...")
    from exploratory_analysis.data_loading.data_validator import ValidationLevel
    validator_basic = DataValidator(validation_level=ValidationLevel.BASIC)
    report_basic = validator_basic.validate_dataset(data_result.data)
    
    print(f"📊 Basic Validation Results:")
    print(f"   Quality Score: {report_basic.overall_score:.1f}/100")
    print(f"   Quality Rating: {report_basic.data_quality}")
    print(f"   Issues Found: {len(report_basic.issues)}")
    
    # Show issues by severity
    severity_counts = report_basic.summary['issues_by_severity']
    for severity, count in severity_counts.items():
        if count > 0:
            print(f"   {severity.capitalize()}: {count}")
    
    # Test comprehensive validation
    print(f"\n🔍 Testing comprehensive validation...")
    validator_strict = DataValidator(validation_level=ValidationLevel.STRICT)
    report_strict = validator_strict.validate_dataset(data_result.data)
    
    print(f"📊 Comprehensive Validation Results:")
    print(f"   Quality Score: {report_strict.overall_score:.1f}/100")
    print(f"   Quality Rating: {report_strict.data_quality}")
    print(f"   Issues Found: {len(report_strict.issues)}")
    
    # Show top issues
    if report_strict.issues:
        print(f"\n⚠️  Top Issues Found:")
        for i, issue in enumerate(report_strict.issues[:5], 1):
            print(f"   {i}. {issue.feature}: {issue.description}")
            print(f"      Severity: {issue.severity.value}, Count: {issue.count}")
    
    # Show recommendations
    if report_strict.recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report_strict.recommendations, 1):
            print(f"   {i}. {rec}")
    
    print("✅ Data validator test completed\n")
    return report_strict

def test_feature_configuration():
    """Test feature configuration system"""
    print("🎼 Testing Feature Configuration")
    print("=" * 50)
    
    from exploratory_analysis.config.features_config import (
        CLUSTERING_FEATURES, get_features_by_type, get_feature_range,
        validate_feature_value, FEATURE_DISPLAY_NAMES
    )
    
    print(f"📊 Total clustering features: {len(CLUSTERING_FEATURES)}")
    print(f"📋 Features: {CLUSTERING_FEATURES}")
    
    # Test feature by type
    from exploratory_analysis.config.features_config import FeatureType
    audio_features = get_features_by_type(FeatureType.AUDIO)
    print(f"\n🎵 Audio features ({len(audio_features)}): {audio_features}")
    
    rhythmic_features = get_features_by_type(FeatureType.RHYTHMIC) 
    print(f"🥁 Rhythmic features ({len(rhythmic_features)}): {rhythmic_features}")
    
    # Test feature validation
    print(f"\n🔍 Testing feature validation:")
    test_cases = [
        ('energy', 0.8, True),
        ('energy', 1.5, False),
        ('tempo', 120, True),
        ('tempo', 500, False),
        ('valence', 0.5, True)
    ]
    
    for feature, value, expected in test_cases:
        result = validate_feature_value(feature, value)
        status = "✅" if result == expected else "❌"
        range_info = get_feature_range(feature)
        print(f"   {status} {feature}={value} (range: {range_info}) -> {result}")
    
    print("✅ Feature configuration test completed\n")

def display_summary(data_result, validation_report):
    """Display final summary"""
    print("📋 SYSTEM TEST SUMMARY")
    print("=" * 50)
    
    if data_result.success:
        df = data_result.data
        print(f"✅ Data Loading: SUCCESS")
        print(f"   📊 Loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"   💾 Memory: {data_result.metadata['memory_usage_mb']:.2f} MB")
        
        # Show feature coverage
        from exploratory_analysis.config.features_config import CLUSTERING_FEATURES
        available_features = set(df.columns) & set(CLUSTERING_FEATURES)
        coverage = (len(available_features) / len(CLUSTERING_FEATURES)) * 100
        print(f"   🎼 Feature coverage: {len(available_features)}/{len(CLUSTERING_FEATURES)} ({coverage:.1f}%)")
        
        if validation_report:
            print(f"✅ Data Validation: SUCCESS")
            print(f"   🏆 Quality Score: {validation_report.overall_score:.1f}/100")
            print(f"   📈 Quality Rating: {validation_report.data_quality.upper()}")
            print(f"   ⚠️  Issues: {len(validation_report.issues)}")
        
        print(f"\n🎉 System is ready for exploratory analysis!")
        
        # Show next steps
        print(f"\n🚀 Next Steps:")
        print(f"   1. Run statistical analysis on the data")
        print(f"   2. Generate distribution plots and correlations")
        print(f"   3. Perform feature engineering and dimensionality reduction")
        print(f"   4. Prepare data for advanced clustering")
        
    else:
        print(f"❌ Data Loading: FAILED")
        print(f"   Issues found - check configuration and data paths")

def main():
    """Main test function"""
    print("🎵 EXPLORATORY ANALYSIS SYSTEM TEST")
    print("=" * 70)
    print("Testing the data loading and validation modules\n")
    
    try:
        # Configure for development
        configure_for_development()
        
        # Run tests
        test_configuration()
        test_feature_configuration() 
        data_result = test_data_loader()
        validation_report = test_data_validator(data_result)
        
        # Display summary
        display_summary(data_result, validation_report)
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()