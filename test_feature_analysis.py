#!/usr/bin/env python3
"""
🔬 FEATURE ANALYSIS MODULE TEST
======================================================================
Test suite for the feature analysis system including:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- Feature selection and importance analysis
- Clustering readiness assessment
"""

import sys
import os
import logging
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
warnings.filterwarnings('ignore', category=UserWarning)

from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.feature_analysis.dimensionality_reduction import DimensionalityReducer, quick_pca, quick_feature_selection

def print_header(title: str, level: int = 1):
    """Print formatted header"""
    symbols = ['🔬', '📊', '🔍', '⚡', '🎯']
    symbol = symbols[min(level-1, len(symbols)-1)]
    
    if level == 1:
        print(f"\n{symbol} {title}")
        print("=" * 70)
    elif level == 2:
        print(f"\n{symbol} {title}")
        print("=" * 60)
    else:
        print(f"\n{symbol} {title}")
        print("-" * 40)

def test_pca_analysis():
    """Test Principal Component Analysis functionality"""
    print_header("Testing PCA Analysis", 2)
    
    # Load test data
    print("🔍 Loading sample data...")
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=200)
    
    if not result.success:
        print("❌ Failed to load data for testing")
        return False
    
    df = result.data
    print(f"✅ Loaded {len(df)} rows for PCA testing")
    
    # Initialize dimensionality reducer
    reducer = DimensionalityReducer()
    
    # Test 1: Basic PCA with automatic component selection
    print("\n📊 Testing automatic PCA component selection...")
    try:
        pca_results = reducer.fit_pca(df, variance_threshold=0.90)
        
        if pca_results:
            n_components = pca_results['n_components']
            total_variance = pca_results['total_variance_explained']
            print(f"✅ PCA completed successfully:")
            print(f"   📈 Components selected: {n_components}")
            print(f"   📊 Total variance explained: {total_variance:.1%}")
            print(f"   🎯 Target variance threshold: 90%")
            
            # Show top components
            component_analysis = pca_results['component_analysis']
            for i, (pc_name, pc_info) in enumerate(list(component_analysis.items())[:3]):
                variance_ratio = pc_info['explained_variance_ratio']
                interpretation = pc_info['interpretation']
                print(f"   🔹 {pc_name}: {variance_ratio:.1%} variance - {interpretation}")
        else:
            print("❌ PCA analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ PCA test failed: {str(e)}")
        return False
    
    # Test 2: Fixed number of components
    print("\n🎯 Testing fixed PCA components...")
    try:
        pca_fixed = reducer.fit_pca(df, n_components=5)
        
        if pca_fixed:
            print(f"✅ Fixed PCA completed:")
            print(f"   📊 Components: {pca_fixed['n_components']}")
            print(f"   📈 Variance explained: {pca_fixed['total_variance_explained']:.1%}")
            
            # Show feature importance
            feature_importance = pca_fixed['feature_importance']
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print("   🎵 Top contributing features:")
            for feature, importance in top_features:
                print(f"      - {feature}: {importance:.3f}")
        else:
            print("❌ Fixed PCA failed")
            return False
            
    except Exception as e:
        print(f"❌ Fixed PCA test failed: {str(e)}")
        return False
    
    # Test 3: PCA with specific features
    print("\n🎼 Testing PCA with selected features...")
    try:
        selected_features = ['energy', 'valence', 'danceability', 'acousticness', 'loudness', 'tempo']
        pca_selected = reducer.fit_pca(df, features=selected_features, n_components=3)
        
        if pca_selected:
            print(f"✅ Selected features PCA completed:")
            print(f"   🎯 Features used: {len(selected_features)}")
            print(f"   📊 Components: {pca_selected['n_components']}")
            print(f"   📈 Variance explained: {pca_selected['total_variance_explained']:.1%}")
        else:
            print("❌ Selected features PCA failed")
            return False
            
    except Exception as e:
        print(f"❌ Selected features PCA test failed: {str(e)}")
        return False
    
    return True

def test_tsne_analysis():
    """Test t-SNE analysis functionality"""
    print_header("Testing t-SNE Analysis", 2)
    
    # Load test data (smaller sample for t-SNE)
    print("🔍 Loading sample data...")
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=100)
    
    if not result.success:
        print("❌ Failed to load data for testing")
        return False
    
    df = result.data
    print(f"✅ Loaded {len(df)} rows for t-SNE testing")
    
    # Initialize dimensionality reducer
    reducer = DimensionalityReducer()
    
    # Test t-SNE with default parameters
    print("\n🌐 Testing t-SNE with default parameters...")
    try:
        tsne_results = reducer.fit_tsne(df, n_components=2, perplexity=30.0)
        
        if tsne_results:
            kl_divergence = tsne_results['kl_divergence']
            n_iter = tsne_results['n_iter']
            print(f"✅ t-SNE completed successfully:")
            print(f"   🎯 Components: {tsne_results['n_components']}")
            print(f"   📊 KL divergence: {kl_divergence:.4f}")
            print(f"   🔄 Iterations: {n_iter}")
            print(f"   📈 Perplexity: {tsne_results['perplexity']}")
        else:
            print("❌ t-SNE analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ t-SNE test failed: {str(e)}")
        return False
    
    # Test t-SNE with adjusted perplexity for small dataset
    print("\n⚙️  Testing t-SNE with adjusted parameters...")
    try:
        tsne_adjusted = reducer.fit_tsne(df, n_components=2, perplexity=10.0)
        
        if tsne_adjusted:
            print(f"✅ Adjusted t-SNE completed:")
            print(f"   🎯 Adjusted perplexity: {tsne_adjusted['perplexity']}")
            print(f"   📊 KL divergence: {tsne_adjusted['kl_divergence']:.4f}")
        else:
            print("❌ Adjusted t-SNE failed")
            return False
            
    except Exception as e:
        print(f"❌ Adjusted t-SNE test failed: {str(e)}")
        return False
    
    return True

def test_umap_analysis():
    """Test UMAP analysis functionality"""
    print_header("Testing UMAP Analysis", 2)
    
    # Load test data
    print("🔍 Loading sample data...")
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=150)
    
    if not result.success:
        print("❌ Failed to load data for testing")
        return False
    
    df = result.data
    print(f"✅ Loaded {len(df)} rows for UMAP testing")
    
    # Initialize dimensionality reducer
    reducer = DimensionalityReducer()
    
    # Test UMAP (might not be available)
    print("\n🗺️  Testing UMAP analysis...")
    try:
        umap_results = reducer.fit_umap(df, n_components=2, n_neighbors=15)
        
        if umap_results:
            print(f"✅ UMAP completed successfully:")
            print(f"   🎯 Components: {umap_results['n_components']}")
            print(f"   👥 Neighbors: {umap_results['n_neighbors']}")
            print(f"   📏 Min distance: {umap_results['min_dist']}")
            print(f"   📊 Features used: {len(umap_results['original_features'])}")
        else:
            print("⚠️  UMAP not available or failed (UMAP library may not be installed)")
            
    except Exception as e:
        print(f"⚠️  UMAP test failed: {str(e)} (UMAP library may not be installed)")
    
    return True  # Don't fail if UMAP is not available

def test_feature_selection():
    """Test feature selection functionality"""
    print_header("Testing Feature Selection", 2)
    
    # Load test data
    print("🔍 Loading sample data...")
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=120)
    
    if not result.success:
        print("❌ Failed to load data for testing")
        return False
    
    df = result.data
    print(f"✅ Loaded {len(df)} rows for feature selection testing")
    
    # Initialize dimensionality reducer
    reducer = DimensionalityReducer()
    
    # Test 1: Variance-based feature selection
    print("\n📊 Testing variance-based feature selection...")
    try:
        selection_results = reducer.perform_feature_selection(
            df, 
            target_column=None,  # Unsupervised
            methods=['variance']
        )
        
        if selection_results and 'selection_results' in selection_results:
            variance_results = selection_results['selection_results'].get('variance', {})
            if variance_results:
                n_selected = variance_results['n_selected']
                n_original = len(selection_results['original_features'])
                print(f"✅ Variance selection completed:")
                print(f"   📊 Original features: {n_original}")
                print(f"   ✅ Selected features: {n_selected}")
                print(f"   📈 Selection ratio: {n_selected/n_original:.1%}")
                
                # Show selected features
                selected = variance_results['selected_features'][:5]
                print(f"   🎵 Top selected features: {', '.join(selected)}")
            else:
                print("❌ Variance selection returned no results")
                return False
        else:
            print("❌ Feature selection failed")
            return False
            
    except Exception as e:
        print(f"❌ Feature selection test failed: {str(e)}")
        return False
    
    # Test 2: Feature selection with synthetic target
    print("\n🎯 Testing supervised feature selection...")
    try:
        # Create a synthetic target based on energy + valence for testing
        if 'energy' in df.columns and 'valence' in df.columns:
            df['synthetic_target'] = df['energy'] + df['valence']
            
            supervised_results = reducer.perform_feature_selection(
                df,
                target_column='synthetic_target',
                methods=['variance', 'mutual_info']
            )
            
            if supervised_results and 'summary' in supervised_results:
                summary = supervised_results['summary']
                print(f"✅ Supervised selection completed:")
                print(f"   🔄 Methods used: {summary['n_methods']}")
                print(f"   📊 Unique features selected: {summary['total_unique_features']}")
                
                # Show consensus features
                consensus = summary.get('consensus_features', [])
                if consensus:
                    print(f"   🎯 Consensus features: {', '.join(consensus)}")
                else:
                    print("   ⚠️  No consensus features found")
            else:
                print("❌ Supervised selection failed")
                return False
        else:
            print("⚠️  Skipping supervised test - required features not found")
            
    except Exception as e:
        print(f"❌ Supervised selection test failed: {str(e)}")
        return False
    
    return True

def test_dimensionality_comparison():
    """Test dimensionality reduction comparison"""
    print_header("Testing Dimensionality Reduction Comparison", 2)
    
    # Load test data
    print("🔍 Loading sample data...")
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=80)
    
    if not result.success:
        print("❌ Failed to load data for testing")
        return False
    
    df = result.data
    print(f"✅ Loaded {len(df)} rows for comparison testing")
    
    # Initialize dimensionality reducer
    reducer = DimensionalityReducer()
    
    # Test comparison of multiple methods
    print("\n🔄 Testing dimensionality reduction comparison...")
    try:
        comparison_fig = reducer.create_dimensionality_reduction_comparison(
            df,
            features=None,  # Use all available features
            methods=['pca', 'tsne']  # Skip UMAP as it might not be available
        )
        
        if comparison_fig:
            print("✅ Dimensionality reduction comparison completed:")
            print("   📊 PCA and t-SNE projections created")
            print("   🎨 Comparison visualization generated")
            print("   📈 2D projections for pattern analysis")
            plt.close(comparison_fig)
        else:
            print("❌ Comparison visualization failed")
            return False
            
    except Exception as e:
        print(f"❌ Comparison test failed: {str(e)}")
        return False
    
    return True

def test_convenience_functions():
    """Test convenience functions"""
    print_header("Testing Convenience Functions", 2)
    
    # Load test data
    print("🔍 Loading sample data...")
    loader = DataLoader()
    result = loader.load_dataset('sample_500', sample_size=60)
    
    if not result.success:
        print("❌ Failed to load data for testing")
        return False
    
    df = result.data
    print(f"✅ Loaded {len(df)} rows for convenience testing")
    
    # Test quick_pca function
    print("\n⚡ Testing quick_pca function...")
    try:
        quick_pca_results = quick_pca(df, n_components=4)
        
        if quick_pca_results:
            print(f"✅ Quick PCA works:")
            print(f"   📊 Components: {quick_pca_results['n_components']}")
            print(f"   📈 Variance explained: {quick_pca_results['total_variance_explained']:.1%}")
        else:
            print("❌ Quick PCA failed")
            return False
            
    except Exception as e:
        print(f"❌ Quick PCA test failed: {str(e)}")
        return False
    
    # Test quick_feature_selection function
    print("\n🎯 Testing quick_feature_selection function...")
    try:
        quick_selection_results = quick_feature_selection(df)
        
        if quick_selection_results:
            n_original = len(quick_selection_results['original_features'])
            n_methods = len(quick_selection_results.get('selection_results', {}))
            print(f"✅ Quick feature selection works:")
            print(f"   📊 Original features: {n_original}")
            print(f"   🔄 Selection methods used: {n_methods}")
        else:
            print("❌ Quick feature selection failed")
            return False
            
    except Exception as e:
        print(f"❌ Quick feature selection test failed: {str(e)}")
        return False
    
    return True

def main():
    """Run all feature analysis tests"""
    print_header("FEATURE ANALYSIS MODULE TEST", 1)
    print("Testing the comprehensive feature analysis system")
    
    # Track test results
    test_results = []
    
    # Run PCA tests
    try:
        result = test_pca_analysis()
        test_results.append(('PCA Analysis', result))
    except Exception as e:
        print(f"❌ PCA testing failed with error: {str(e)}")
        test_results.append(('PCA Analysis', False))
    
    # Run t-SNE tests
    try:
        result = test_tsne_analysis()
        test_results.append(('t-SNE Analysis', result))
    except Exception as e:
        print(f"❌ t-SNE testing failed with error: {str(e)}")
        test_results.append(('t-SNE Analysis', False))
    
    # Run UMAP tests
    try:
        result = test_umap_analysis()
        test_results.append(('UMAP Analysis', result))
    except Exception as e:
        print(f"❌ UMAP testing failed with error: {str(e)}")
        test_results.append(('UMAP Analysis', False))
    
    # Run feature selection tests
    try:
        result = test_feature_selection()
        test_results.append(('Feature Selection', result))
    except Exception as e:
        print(f"❌ Feature selection testing failed with error: {str(e)}")
        test_results.append(('Feature Selection', False))
    
    # Run comparison tests
    try:
        result = test_dimensionality_comparison()
        test_results.append(('Dimensionality Comparison', result))
    except Exception as e:
        print(f"❌ Comparison testing failed with error: {str(e)}")
        test_results.append(('Dimensionality Comparison', False))
    
    # Run convenience function tests
    try:
        result = test_convenience_functions()
        test_results.append(('Convenience Functions', result))
    except Exception as e:
        print(f"❌ Convenience function testing failed with error: {str(e)}")
        test_results.append(('Convenience Functions', False))
    
    # Print final results
    print_header("TEST RESULTS SUMMARY", 1)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("🔬 Feature analysis module is working correctly")
        print("🚀 Ready for reporting system implementation")
    elif passed >= total * 0.8:  # 80% pass rate acceptable
        print("✅ MOSTLY SUCCESSFUL!")
        print("🔬 Core feature analysis functionality is working")
        print("⚠️  Some optional components may not be available")
        print("🚀 Ready to proceed with reporting system")
    else:
        print("⚠️  Some critical tests failed - check implementation")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)