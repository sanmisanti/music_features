#!/usr/bin/env python3
"""
🎨 VISUALIZATION MODULE TEST
======================================================================
Test suite for the exploratory data analysis visualization system

This script validates the visualization capabilities including:
- Distribution plots (histograms, box plots, violin plots)
- Correlation heatmaps and analysis
- Feature type grouping
- Network analysis
- Statistical plots
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from exploratory_analysis.data_loading import DataLoader
from exploratory_analysis.visualization import DistributionPlotter, CorrelationPlotter
import matplotlib.pyplot as plt

def test_distribution_plots():
    """Test distribution plotting functionality"""
    print("📊 Testing Distribution Plots Module")
    print("=" * 60)
    
    # Load sample data
    print("🔍 Loading sample data...")
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=100, validate=True)
    
    if not result.success:
        print("❌ Failed to load data for testing")
        return False
    
    df = result.data
    print(f"✅ Loaded {len(df)} rows for visualization testing")
    
    # Initialize distribution plotter
    plotter = DistributionPlotter()
    
    # Test basic distribution plots
    print("\n🎨 Testing basic distribution plots...")
    try:
        plot_results = plotter.plot_feature_distributions(
            df, 
            features=['energy', 'valence', 'danceability', 'acousticness'],
            plot_types=['histogram', 'boxplot']
        )
        
        print(f"✅ Successfully created {len(plot_results)} plot types")
        
        # Close plots to save memory
        for plot_type, plot_info in plot_results.items():
            plt.close(plot_info['figure'])
            print(f"   📈 {plot_type.capitalize()} plot created and closed")
        
    except Exception as e:
        print(f"❌ Failed to create distribution plots: {str(e)}")
        return False
    
    # Test feature type plots
    print("\n🎵 Testing feature type grouping...")
    try:
        type_plots = plotter.plot_feature_by_type(df, plot_type='histogram')
        
        print(f"✅ Successfully created plots for {len(type_plots)} feature types:")
        for feature_type in type_plots.keys():
            print(f"   🎼 {feature_type.capitalize()} features plot")
            plt.close(type_plots[feature_type])
        
    except Exception as e:
        print(f"❌ Failed to create feature type plots: {str(e)}")
        return False
    
    # Test distribution summary
    print("\n📋 Testing distribution summary dashboard...")
    try:
        summary_fig = plotter.create_distribution_summary(df)
        if summary_fig:
            print("✅ Distribution summary dashboard created successfully")
            plt.close(summary_fig)
        else:
            print("⚠️  Distribution summary returned None")
    
    except Exception as e:
        print(f"❌ Failed to create distribution summary: {str(e)}")
        return False
    
    return True

def test_correlation_plots():
    """Test correlation plotting functionality"""
    print("\n🔗 Testing Correlation Plots Module")
    print("=" * 60)
    
    # Load sample data
    print("🔍 Loading sample data...")
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=150, validate=True)
    
    if not result.success:
        print("❌ Failed to load data for testing")
        return False
    
    df = result.data
    print(f"✅ Loaded {len(df)} rows for correlation testing")
    
    # Initialize correlation plotter
    plotter = CorrelationPlotter()
    
    # Test basic correlation heatmap
    print("\n🔥 Testing correlation heatmap...")
    try:
        corr_fig = plotter.create_correlation_heatmap(
            df, 
            features=['energy', 'valence', 'danceability', 'acousticness', 'loudness', 'tempo'],
            method='pearson'
        )
        
        if corr_fig:
            print("✅ Pearson correlation heatmap created successfully")
            plt.close(corr_fig)
        else:
            print("⚠️  Correlation heatmap returned None")
    
    except Exception as e:
        print(f"❌ Failed to create correlation heatmap: {str(e)}")
        return False
    
    # Test correlation method comparison
    print("\n⚖️  Testing correlation method comparison...")
    try:
        comparison_fig = plotter.create_correlation_comparison(
            df, 
            features=['energy', 'valence', 'danceability', 'acousticness'],
            methods=['pearson', 'spearman']
        )
        
        if comparison_fig:
            print("✅ Correlation method comparison created successfully")
            plt.close(comparison_fig)
        else:
            print("⚠️  Correlation comparison returned None")
    
    except Exception as e:
        print(f"❌ Failed to create correlation comparison: {str(e)}")
        return False
    
    # Test correlation analysis
    print("\n🔍 Testing correlation strength analysis...")
    try:
        corr_analysis = plotter.analyze_correlation_strength(
            df, 
            features=['energy', 'valence', 'danceability', 'acousticness', 'loudness', 'tempo'],
            method='pearson',
            threshold=0.3
        )
        
        if corr_analysis:
            summary = corr_analysis['summary']
            print("✅ Correlation analysis completed successfully:")
            print(f"   📊 Total feature pairs: {corr_analysis['total_pairs']}")
            print(f"   🔴 High correlations (≥0.3): {summary['high_count']}")
            print(f"   🟡 Moderate correlations: {summary['moderate_count']}")
            print(f"   🟢 Weak correlations: {summary['weak_count']}")
            print(f"   📈 Max correlation: {summary['max_correlation']:.3f}")
            print(f"   📊 Average correlation: {summary['avg_correlation']:.3f}")
        else:
            print("⚠️  Correlation analysis returned empty results")
    
    except Exception as e:
        print(f"❌ Failed to perform correlation analysis: {str(e)}")
        return False
    
    # Test feature type correlations
    print("\n🎼 Testing feature type correlations...")
    try:
        type_corr_plots = plotter.create_feature_type_correlation(df, method='pearson')
        
        print(f"✅ Successfully created correlation plots for {len(type_corr_plots)} feature types:")
        for feature_type in type_corr_plots.keys():
            print(f"   🎵 {feature_type.capitalize()} features correlation")
            plt.close(type_corr_plots[feature_type])
        
    except Exception as e:
        print(f"❌ Failed to create feature type correlations: {str(e)}")
        return False
    
    return True

def test_convenience_functions():
    """Test convenience functions"""
    print("\n🛠️ Testing Convenience Functions")
    print("=" * 40)
    
    # Load data
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=75)
    
    if not result.success:
        print("❌ Failed to load data")
        return False
    
    df = result.data
    
    # Test quick distribution plotting
    print("⚡ Testing quick distribution functions...")
    try:
        from exploratory_analysis.visualization.distribution_plots import plot_distributions, quick_histogram
        
        # Test plot_distributions
        quick_plots = plot_distributions(df, features=['energy', 'valence'], plot_types=['histogram'])
        if quick_plots and 'histogram' in quick_plots:
            print("✅ Quick distribution plotting works")
            plt.close(quick_plots['histogram']['figure'])
        
        # Test quick_histogram
        single_hist = quick_histogram(df, 'energy')
        if single_hist:
            print("✅ Quick histogram function works")
            plt.close(single_hist)
        
    except Exception as e:
        print(f"❌ Failed to test convenience functions: {str(e)}")
        return False
    
    # Test quick correlation functions
    print("🔗 Testing quick correlation functions...")
    try:
        from exploratory_analysis.visualization.correlation_heatmaps import correlation_heatmap, analyze_correlations
        
        # Test correlation_heatmap
        quick_corr = correlation_heatmap(df, features=['energy', 'valence', 'danceability'])
        if quick_corr:
            print("✅ Quick correlation heatmap works")
            plt.close(quick_corr)
        
        # Test analyze_correlations
        quick_analysis = analyze_correlations(df, features=['energy', 'valence', 'danceability'], threshold=0.2)
        if quick_analysis and 'summary' in quick_analysis:
            print("✅ Quick correlation analysis works")
            print(f"   📊 Found {quick_analysis['summary']['high_count']} high correlations")
        
    except Exception as e:
        print(f"❌ Failed to test correlation convenience functions: {str(e)}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🎨 VISUALIZATION MODULE TEST")
    print("=" * 70)
    print("Testing the visualization functionality for music data analysis\\n")
    
    try:
        # Run distribution plots test
        dist_success = test_distribution_plots()
        
        # Run correlation plots test
        corr_success = test_correlation_plots()
        
        # Run convenience functions test
        conv_success = test_convenience_functions()
        
        if dist_success and corr_success and conv_success:
            print(f"\\n🎉 ALL VISUALIZATION TESTS PASSED!")
            print(f"📊 Distribution plotting module is working correctly")
            print(f"🔗 Correlation analysis module is working correctly")
            print(f"🛠️ Convenience functions are working correctly")
            print(f"🚀 Ready for integration with feature analysis module")
        else:
            print(f"\\n⚠️  Some tests failed. Please check the output above.")
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()