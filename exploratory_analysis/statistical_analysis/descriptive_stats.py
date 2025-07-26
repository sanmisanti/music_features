"""
Descriptive Statistics Module

Comprehensive descriptive statistics for musical datasets including
central tendency, dispersion, distribution shape, and feature summaries.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats
import warnings

from ..config.features_config import (
    CLUSTERING_FEATURES, FEATURE_DEFINITIONS, FEATURE_DISPLAY_NAMES,
    get_features_by_type, FeatureType
)
from ..config.analysis_config import config

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class FeatureStats:
    """Statistics for a single feature"""
    name: str
    display_name: str
    count: int
    missing: int
    missing_pct: float
    
    # Central tendency
    mean: float
    median: float
    mode: Optional[float]
    
    # Dispersion
    std: float
    variance: float
    min_val: float
    max_val: float
    range_val: float
    
    # Quartiles
    q1: float
    q3: float
    iqr: float
    
    # Distribution shape
    skewness: float
    kurtosis: float
    
    # Additional metrics
    unique_values: int
    outliers_iqr: int
    outliers_zscore: int

@dataclass
class DatasetStats:
    """Overall dataset statistics"""
    total_rows: int
    total_features: int
    memory_usage_mb: float
    missing_data_total: int
    missing_data_pct: float
    duplicate_rows: int
    duplicate_pct: float
    
    # Feature type breakdown
    feature_counts: Dict[str, int]
    
    # Quality indicators
    completeness_score: float
    consistency_score: float
    overall_quality: str

class DescriptiveStats:
    """
    Comprehensive descriptive statistics analyzer for music datasets
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize descriptive statistics analyzer
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = config
        if config_override:
            self.config.update_config(**config_override)
        
        self.feature_stats = {}
        self.dataset_stats = None
    
    def analyze_dataset(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive descriptive analysis of dataset
        
        Args:
            df: DataFrame to analyze
            features: Specific features to analyze (default: all clustering features)
            
        Returns:
            Dictionary containing all statistics
        """
        if df.empty:
            logger.error("Cannot analyze empty dataset")
            return {}
        
        logger.info(f"Starting descriptive analysis of {len(df)} rows")
        
        # Determine features to analyze
        if features is None:
            features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        else:
            features = [f for f in features if f in df.columns]
        
        if not features:
            logger.error("No valid features found for analysis")
            return {}
        
        logger.info(f"Analyzing {len(features)} features: {features}")
        
        # Calculate feature-level statistics
        self.feature_stats = {}
        for feature in features:
            try:
                self.feature_stats[feature] = self._calculate_feature_stats(df, feature)
            except Exception as e:
                logger.warning(f"Failed to calculate stats for {feature}: {str(e)}")
        
        # Calculate dataset-level statistics
        self.dataset_stats = self._calculate_dataset_stats(df, features)
        
        # Generate summary
        analysis_results = {
            'dataset_stats': self.dataset_stats,
            'feature_stats': self.feature_stats,
            'summary_by_type': self._group_stats_by_type(),
            'correlation_preview': self._calculate_correlation_preview(df, features),
            'distribution_summary': self._summarize_distributions(),
            'quality_assessment': self._assess_data_quality()
        }
        
        logger.info("Descriptive analysis completed successfully")
        return analysis_results
    
    def _calculate_feature_stats(self, df: pd.DataFrame, feature: str) -> FeatureStats:
        """Calculate comprehensive statistics for a single feature"""
        series = df[feature].copy()
        
        # Basic counts
        count = len(series)
        missing = series.isnull().sum()
        missing_pct = (missing / count) * 100 if count > 0 else 0
        
        # Remove missing values for calculations
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            # Return default stats for completely missing feature
            return self._create_empty_feature_stats(feature, count, missing, missing_pct)
        
        # Central tendency
        mean_val = float(clean_series.mean())
        median_val = float(clean_series.median())
        
        # Mode (most frequent value)
        try:
            mode_result = stats.mode(clean_series, nan_policy='omit', keepdims=True)
            mode_val = float(mode_result.mode[0]) if len(mode_result.mode) > 0 else None
        except:
            mode_val = None
        
        # Dispersion
        std_val = float(clean_series.std())
        variance_val = float(clean_series.var())
        min_val = float(clean_series.min())
        max_val = float(clean_series.max())
        range_val = max_val - min_val
        
        # Quartiles
        q1 = float(clean_series.quantile(0.25))
        q3 = float(clean_series.quantile(0.75))
        iqr = q3 - q1
        
        # Distribution shape
        try:
            skewness = float(stats.skew(clean_series))
            kurtosis_val = float(stats.kurtosis(clean_series))
        except:
            skewness = 0.0
            kurtosis_val = 0.0
        
        # Additional metrics
        unique_values = int(clean_series.nunique())
        
        # Outliers using IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_iqr = int(((clean_series < lower_bound) | (clean_series > upper_bound)).sum())
        
        # Outliers using Z-score method
        if std_val > 0:
            z_scores = np.abs(stats.zscore(clean_series))
            outliers_zscore = int((z_scores > self.config.stats.zscore_threshold).sum())
        else:
            outliers_zscore = 0
        
        return FeatureStats(
            name=feature,
            display_name=FEATURE_DISPLAY_NAMES.get(feature, feature),
            count=count,
            missing=missing,
            missing_pct=missing_pct,
            mean=mean_val,
            median=median_val,
            mode=mode_val,
            std=std_val,
            variance=variance_val,
            min_val=min_val,
            max_val=max_val,
            range_val=range_val,
            q1=q1,
            q3=q3,
            iqr=iqr,
            skewness=skewness,
            kurtosis=kurtosis_val,
            unique_values=unique_values,
            outliers_iqr=outliers_iqr,
            outliers_zscore=outliers_zscore
        )
    
    def _create_empty_feature_stats(self, feature: str, count: int, missing: int, missing_pct: float) -> FeatureStats:
        """Create empty stats for features with no valid data"""
        return FeatureStats(
            name=feature,
            display_name=FEATURE_DISPLAY_NAMES.get(feature, feature),
            count=count,
            missing=missing,
            missing_pct=missing_pct,
            mean=0.0, median=0.0, mode=None,
            std=0.0, variance=0.0,
            min_val=0.0, max_val=0.0, range_val=0.0,
            q1=0.0, q3=0.0, iqr=0.0,
            skewness=0.0, kurtosis=0.0,
            unique_values=0, outliers_iqr=0, outliers_zscore=0
        )
    
    def _calculate_dataset_stats(self, df: pd.DataFrame, features: List[str]) -> DatasetStats:
        """Calculate overall dataset statistics"""
        total_rows = len(df)
        total_features = len(features)
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)
        
        # Missing data
        missing_data_total = df[features].isnull().sum().sum()
        total_cells = total_rows * total_features
        missing_data_pct = (missing_data_total / total_cells) * 100 if total_cells > 0 else 0
        
        # Duplicates
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Feature type breakdown
        feature_counts = {}
        for feature_type in FeatureType:
            type_features = get_features_by_type(feature_type)
            available_type_features = [f for f in type_features if f in features]
            feature_counts[feature_type.value] = len(available_type_features)
        
        # Quality scores
        completeness_score = 100 - missing_data_pct
        consistency_score = 100 - duplicate_pct
        
        # Overall quality assessment
        avg_quality = (completeness_score + consistency_score) / 2
        if avg_quality >= 95:
            overall_quality = "excellent"
        elif avg_quality >= 85:
            overall_quality = "good"
        elif avg_quality >= 70:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        return DatasetStats(
            total_rows=total_rows,
            total_features=total_features,
            memory_usage_mb=memory_usage_mb,
            missing_data_total=missing_data_total,
            missing_data_pct=missing_data_pct,
            duplicate_rows=duplicate_rows,
            duplicate_pct=duplicate_pct,
            feature_counts=feature_counts,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            overall_quality=overall_quality
        )
    
    def _group_stats_by_type(self) -> Dict[str, Dict]:
        """Group feature statistics by feature type"""
        grouped_stats = {}
        
        for feature_type in FeatureType:
            type_features = get_features_by_type(feature_type)
            type_stats = {
                feature: self.feature_stats[feature] 
                for feature in type_features 
                if feature in self.feature_stats
            }
            
            if type_stats:
                # Calculate aggregate statistics for this type
                type_summary = {
                    'feature_count': len(type_stats),
                    'total_missing': sum(stats.missing for stats in type_stats.values()),
                    'avg_missing_pct': np.mean([stats.missing_pct for stats in type_stats.values()]),
                    'avg_skewness': np.mean([stats.skewness for stats in type_stats.values()]),
                    'features': type_stats
                }
                grouped_stats[feature_type.value] = type_summary
        
        return grouped_stats
    
    def _calculate_correlation_preview(self, df: pd.DataFrame, features: List[str]) -> Dict:
        """Calculate correlation preview (top correlations)"""
        try:
            # Select only numeric features for correlation
            numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
            
            if len(numeric_features) < 2:
                return {'message': 'Insufficient numeric features for correlation analysis'}
            
            corr_matrix = df[numeric_features].corr()
            
            # Find top positive and negative correlations
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if not np.isnan(corr_value):
                        correlations.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'correlation': float(corr_value)
                        })
            
            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                'top_positive': [c for c in correlations if c['correlation'] > 0][:5],
                'top_negative': [c for c in correlations if c['correlation'] < 0][:5],
                'high_correlations': [c for c in correlations if abs(c['correlation']) > 0.7],
                'matrix_shape': corr_matrix.shape
            }
        
        except Exception as e:
            logger.warning(f"Failed to calculate correlation preview: {str(e)}")
            return {'error': str(e)}
    
    def _summarize_distributions(self) -> Dict:
        """Summarize distribution characteristics"""
        if not self.feature_stats:
            return {}
        
        distribution_summary = {
            'highly_skewed': [],      # |skewness| > 1
            'moderately_skewed': [],  # 0.5 < |skewness| <= 1  
            'approximately_normal': [],# |skewness| <= 0.5
            'high_kurtosis': [],      # |kurtosis| > 2
            'many_outliers': [],      # >5% outliers
            'low_variance': [],       # std < 0.01
            'wide_range': []          # range > 3*std
        }
        
        for feature, stats in self.feature_stats.items():
            # Skewness classification
            abs_skew = abs(stats.skewness)
            if abs_skew > 1:
                distribution_summary['highly_skewed'].append(feature)
            elif abs_skew > 0.5:
                distribution_summary['moderately_skewed'].append(feature)
            else:
                distribution_summary['approximately_normal'].append(feature)
            
            # Kurtosis
            if abs(stats.kurtosis) > 2:
                distribution_summary['high_kurtosis'].append(feature)
            
            # Outliers
            if stats.count > 0:
                outlier_pct = (stats.outliers_iqr / (stats.count - stats.missing)) * 100
                if outlier_pct > 5:
                    distribution_summary['many_outliers'].append(feature)
            
            # Variance
            if stats.std < 0.01:
                distribution_summary['low_variance'].append(feature)
            
            # Range vs standard deviation
            if stats.std > 0 and stats.range_val > 3 * stats.std:
                distribution_summary['wide_range'].append(feature)
        
        return distribution_summary
    
    def _assess_data_quality(self) -> Dict:
        """Assess overall data quality"""
        if not self.feature_stats or not self.dataset_stats:
            return {}
        
        quality_issues = []
        quality_strengths = []
        
        # Check completeness
        if self.dataset_stats.missing_data_pct > 10:
            quality_issues.append(f"High missing data: {self.dataset_stats.missing_data_pct:.1f}%")
        elif self.dataset_stats.missing_data_pct < 1:
            quality_strengths.append("Very low missing data")
        
        # Check duplicates
        if self.dataset_stats.duplicate_pct > 1:
            quality_issues.append(f"Duplicate rows found: {self.dataset_stats.duplicate_pct:.1f}%")
        else:
            quality_strengths.append("No significant duplicates")
        
        # Check feature quality
        problematic_features = 0
        for feature, stats in self.feature_stats.items():
            if stats.missing_pct > 20:
                problematic_features += 1
            elif stats.std == 0:
                problematic_features += 1
        
        if problematic_features > len(self.feature_stats) * 0.2:
            quality_issues.append(f"Multiple problematic features: {problematic_features}")
        else:
            quality_strengths.append("Most features have good quality")
        
        return {
            'overall_score': (self.dataset_stats.completeness_score + self.dataset_stats.consistency_score) / 2,
            'quality_rating': self.dataset_stats.overall_quality,
            'issues': quality_issues,
            'strengths': quality_strengths,
            'recommendations': self._generate_quality_recommendations(quality_issues)
        }
    
    def _generate_quality_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on quality issues"""
        recommendations = []
        
        for issue in issues:
            if "missing data" in issue.lower():
                recommendations.append("Consider imputation strategies for missing values")
            elif "duplicate" in issue.lower():
                recommendations.append("Remove duplicate rows before analysis")
            elif "problematic features" in issue.lower():
                recommendations.append("Review and potentially exclude low-quality features")
        
        if not recommendations:
            recommendations.append("Data quality is good - proceed with analysis")
        
        return recommendations
    
    def get_feature_summary(self, feature: str) -> Optional[Dict]:
        """Get summary for a specific feature"""
        if feature not in self.feature_stats:
            return None
        
        stats = self.feature_stats[feature]
        return {
            'basic_stats': {
                'count': stats.count,
                'missing': f"{stats.missing} ({stats.missing_pct:.1f}%)",
                'mean': round(stats.mean, 3),
                'median': round(stats.median, 3),
                'std': round(stats.std, 3)
            },
            'distribution': {
                'skewness': round(stats.skewness, 3),
                'kurtosis': round(stats.kurtosis, 3),
                'outliers_iqr': stats.outliers_iqr,
                'unique_values': stats.unique_values
            },
            'range': {
                'min': round(stats.min_val, 3),
                'max': round(stats.max_val, 3),
                'q1': round(stats.q1, 3),
                'q3': round(stats.q3, 3)
            }
        }
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get summary statistics as a DataFrame"""
        if not self.feature_stats:
            return pd.DataFrame()
        
        summary_data = []
        for feature, stats in self.feature_stats.items():
            summary_data.append({
                'Feature': stats.display_name,
                'Count': stats.count,
                'Missing': f"{stats.missing} ({stats.missing_pct:.1f}%)",
                'Mean': round(stats.mean, 3),
                'Std': round(stats.std, 3),
                'Min': round(stats.min_val, 3),
                'Q1': round(stats.q1, 3),
                'Median': round(stats.median, 3),
                'Q3': round(stats.q3, 3),
                'Max': round(stats.max_val, 3),
                'Skewness': round(stats.skewness, 3),
                'Outliers': stats.outliers_iqr
            })
        
        return pd.DataFrame(summary_data)

# Convenience functions
def quick_stats(df: pd.DataFrame, features: Optional[List[str]] = None) -> Dict:
    """Quick descriptive statistics analysis"""
    analyzer = DescriptiveStats()
    return analyzer.analyze_dataset(df, features)

def print_summary(df: pd.DataFrame, features: Optional[List[str]] = None):
    """Print formatted summary statistics"""
    analyzer = DescriptiveStats()
    results = analyzer.analyze_dataset(df, features)
    
    if 'dataset_stats' in results:
        dataset_stats = results['dataset_stats']
        print(f"📊 Dataset Summary:")
        print(f"   Rows: {dataset_stats.total_rows:,}")
        print(f"   Features: {dataset_stats.total_features}")
        print(f"   Missing data: {dataset_stats.missing_data_pct:.1f}%")
        print(f"   Quality: {dataset_stats.overall_quality.upper()}")
    
    if 'feature_stats' in results:
        print(f"\n📋 Feature Statistics:")
        summary_df = analyzer.get_summary_table()
        print(summary_df.to_string(index=False))