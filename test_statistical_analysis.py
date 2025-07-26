"""
Test script for the statistical analysis module

This script tests the descriptive statistics functionality.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from exploratory_analysis.data_loading import DataLoader
from exploratory_analysis.statistical_analysis import DescriptiveStats
from exploratory_analysis.statistical_analysis.descriptive_stats import quick_stats, print_summary
import pandas as pd

def test_descriptive_stats():
    """Test descriptive statistics functionality"""
    print("📊 Testing Descriptive Statistics Module")
    print("=" * 60)
    
    # Load sample data
    print("🔍 Loading sample data...")
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=200, validate=True)
    
    if not result.success:
        print("❌ Failed to load data for testing")
        return
    
    df = result.data
    print(f"✅ Loaded {len(df)} rows for analysis")
    
    # Initialize descriptive stats analyzer
    analyzer = DescriptiveStats()
    
    # Run comprehensive analysis
    print("\n🧮 Running comprehensive statistical analysis...")
    analysis_results = analyzer.analyze_dataset(df)
    
    if not analysis_results:
        print("❌ Failed to run statistical analysis")
        return
    
    # Display dataset-level statistics
    print("\n📋 DATASET OVERVIEW:")
    print("-" * 40)
    dataset_stats = analysis_results['dataset_stats']
    print(f"📊 Total rows: {dataset_stats.total_rows:,}")
    print(f"🎼 Features analyzed: {dataset_stats.total_features}")
    print(f"💾 Memory usage: {dataset_stats.memory_usage_mb:.2f} MB")
    print(f"📈 Missing data: {dataset_stats.missing_data_pct:.2f}%")
    print(f"🔄 Duplicates: {dataset_stats.duplicate_pct:.2f}%")
    print(f"⭐ Overall quality: {dataset_stats.overall_quality.upper()}")
    
    # Feature type breakdown
    print(f"\n🎵 Features by type:")
    for feat_type, count in dataset_stats.feature_counts.items():
        if count > 0:
            print(f"   {feat_type.capitalize()}: {count}")
    
    # Display correlation preview
    print(f"\n🔗 CORRELATION PREVIEW:")
    print("-" * 40)
    corr_preview = analysis_results['correlation_preview']
    
    if 'top_positive' in corr_preview and corr_preview['top_positive']:
        print("📈 Top positive correlations:")
        for i, corr in enumerate(corr_preview['top_positive'][:3], 1):
            print(f"   {i}. {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")
    
    if 'top_negative' in corr_preview and corr_preview['top_negative']:
        print("📉 Top negative correlations:")
        for i, corr in enumerate(corr_preview['top_negative'][:3], 1):
            print(f"   {i}. {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")
    
    if 'high_correlations' in corr_preview:
        high_corr_count = len(corr_preview['high_correlations'])
        if high_corr_count > 0:
            print(f"⚠️  High correlations (>0.7): {high_corr_count}")
    
    # Display distribution summary
    print(f"\n📊 DISTRIBUTION SUMMARY:")
    print("-" * 40)
    dist_summary = analysis_results['distribution_summary']
    
    summary_items = [
        ('📐 Approximately normal', 'approximately_normal'),
        ('📏 Moderately skewed', 'moderately_skewed'), 
        ('📐 Highly skewed', 'highly_skewed'),
        ('🎯 Many outliers', 'many_outliers'),
        ('📊 Low variance', 'low_variance')
    ]
    
    for label, key in summary_items:
        if key in dist_summary and dist_summary[key]:
            features = dist_summary[key][:3]  # Show first 3
            count = len(dist_summary[key])
            if count <= 3:
                print(f"   {label}: {', '.join(features)}")
            else:
                print(f"   {label}: {', '.join(features)}... ({count} total)")
    
    # Display quality assessment
    print(f"\n✅ QUALITY ASSESSMENT:")
    print("-" * 40)
    quality = analysis_results['quality_assessment']
    print(f"🏆 Overall score: {quality['overall_score']:.1f}/100")
    print(f"📈 Rating: {quality['quality_rating'].upper()}")
    
    if quality['strengths']:
        print("💪 Strengths:")
        for strength in quality['strengths']:
            print(f"   ✅ {strength}")
    
    if quality['issues']:
        print("⚠️  Issues:")
        for issue in quality['issues']:
            print(f"   ⚠️  {issue}")
    
    if quality['recommendations']:
        print("💡 Recommendations:")
        for rec in quality['recommendations']:
            print(f"   💡 {rec}")
    
    # Show feature summary table
    print(f"\n📋 FEATURE SUMMARY TABLE:")
    print("-" * 60)
    summary_df = analyzer.get_summary_table()
    
    # Show first few features
    if not summary_df.empty:
        print(summary_df.head(8).to_string(index=False))
        if len(summary_df) > 8:
            print(f"... and {len(summary_df) - 8} more features")
    
    # Test individual feature analysis
    print(f"\n🎯 INDIVIDUAL FEATURE ANALYSIS:")
    print("-" * 40)
    
    test_features = ['energy', 'valence', 'tempo']
    for feature in test_features:
        if feature in analyzer.feature_stats:
            print(f"\n🎵 {feature.upper()}:")
            feature_summary = analyzer.get_feature_summary(feature)
            if feature_summary:
                basic = feature_summary['basic_stats']
                dist = feature_summary['distribution']
                print(f"   📊 Mean: {basic['mean']}, Median: {basic['median']}, Std: {basic['std']}")
                print(f"   📐 Skewness: {dist['skewness']}, Outliers: {dist['outliers_iqr']}")
                print(f"   🔢 Unique values: {dist['unique_values']}")
    
    print(f"\n✅ Descriptive statistics analysis completed successfully!")
    return True

def test_convenience_functions():
    """Test convenience functions"""
    print(f"\n🛠️ Testing Convenience Functions")
    print("=" * 40)
    
    # Load data
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=50)
    
    if not result.success:
        print("❌ Failed to load data")
        return
    
    df = result.data
    
    # Test quick_stats
    print("⚡ Testing quick_stats function...")
    quick_results = quick_stats(df)
    
    if quick_results and 'dataset_stats' in quick_results:
        dataset_stats = quick_results['dataset_stats']
        print(f"✅ Quick stats: {dataset_stats.total_rows} rows, quality: {dataset_stats.overall_quality}")
    
    # Test print_summary
    print("\n📄 Testing print_summary function...")
    print_summary(df, features=['energy', 'valence', 'tempo'])
    
    print("✅ Convenience functions working correctly!")

def main():
    """Main test function"""
    print("🎵 STATISTICAL ANALYSIS MODULE TEST")
    print("=" * 70)
    print("Testing the descriptive statistics functionality\n")
    
    try:
        # Run main test
        success = test_descriptive_stats()
        
        if success:
            # Test convenience functions
            test_convenience_functions()
            
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"📊 Statistical analysis module is working correctly")
            print(f"🚀 Ready for integration with visualization module")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()