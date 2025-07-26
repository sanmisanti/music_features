"""
Data Validator Module

Comprehensive data validation for musical datasets including integrity checks,
outlier detection, and data quality assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ..config.features_config import (
    FEATURE_DEFINITIONS, CLUSTERING_FEATURES, CATEGORICAL_FEATURES,
    validate_feature_value, get_feature_range
)
from ..config.analysis_config import config

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Levels of validation rigor"""
    BASIC = "basic"           # Essential checks only
    STANDARD = "standard"     # Comprehensive validation
    STRICT = "strict"         # Maximum validation

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    feature: str
    severity: ValidationSeverity
    issue_type: str
    description: str
    count: int
    percentage: float
    suggested_action: str

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    total_rows: int
    total_columns: int
    issues: List[ValidationIssue]
    overall_score: float  # 0-100 score
    data_quality: str     # 'excellent', 'good', 'fair', 'poor'
    recommendations: List[str]
    summary: Dict[str, Any]

class DataValidator:
    """
    Comprehensive data validator for music datasets
    """
    
    def __init__(self, validation_level = ValidationLevel.STANDARD):
        """
        Initialize validator
        
        Args:
            validation_level: Level of validation rigor
        """
        self.validation_level = validation_level
        self.config = config
        
    def validate_dataset(self, df: pd.DataFrame) -> ValidationReport:
        """
        Perform comprehensive validation of dataset
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationReport with all findings
        """
        logger.info(f"Starting {self.validation_level.value} validation of dataset")
        
        issues = []
        
        # Basic validations (always performed)
        issues.extend(self._validate_basic_structure(df))
        issues.extend(self._validate_missing_data(df))
        issues.extend(self._validate_data_types(df))
        
        # Standard validations
        if self.validation_level.value in ['standard', 'strict']:
            issues.extend(self._validate_feature_ranges(df))
            issues.extend(self._validate_duplicates(df))
            issues.extend(self._validate_feature_relationships(df))
        
        # Strict validations
        if self.validation_level.value == 'strict':
            issues.extend(self._validate_distribution_anomalies(df))
            issues.extend(self._validate_business_logic(df))
            issues.extend(self._validate_statistical_consistency(df))
        
        # Calculate overall quality score
        overall_score = self._calculate_quality_score(issues, len(df))
        data_quality = self._determine_quality_rating(overall_score)
        recommendations = self._generate_recommendations(issues)
        
        # Create summary statistics
        summary = self._generate_summary(df, issues)
        
        logger.info(f"Validation complete. Overall score: {overall_score:.1f}/100 ({data_quality})")
        
        return ValidationReport(
            total_rows=len(df),
            total_columns=len(df.columns),
            issues=issues,
            overall_score=overall_score,
            data_quality=data_quality,
            recommendations=recommendations,
            summary=summary
        )
    
    def _validate_basic_structure(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate basic structure of dataset"""
        issues = []
        
        # Check if dataframe is empty
        if df.empty:
            issues.append(ValidationIssue(
                feature="dataset",
                severity=ValidationSeverity.CRITICAL,
                issue_type="empty_dataset",
                description="Dataset is completely empty",
                count=0,
                percentage=100.0,
                suggested_action="Load data from source or check data path"
            ))
            return issues
        
        # Check for minimum required columns
        required_features = set(CLUSTERING_FEATURES)
        present_features = set(df.columns)
        missing_features = required_features - present_features
        
        if missing_features:
            issues.append(ValidationIssue(
                feature="columns",
                severity=ValidationSeverity.ERROR,
                issue_type="missing_required_columns",
                description=f"Missing required features: {list(missing_features)}",
                count=len(missing_features),
                percentage=(len(missing_features) / len(required_features)) * 100,
                suggested_action="Check data source or column mapping"
            ))
        
        # Check for unexpected columns
        expected_columns = set(FEATURE_DEFINITIONS.keys()) | {'id', 'name', 'album', 'artists', 'artist_ids'}
        unexpected_columns = present_features - expected_columns
        
        if unexpected_columns:
            issues.append(ValidationIssue(
                feature="columns",
                severity=ValidationSeverity.INFO,
                issue_type="unexpected_columns",
                description=f"Unexpected columns found: {list(unexpected_columns)}",
                count=len(unexpected_columns),
                percentage=(len(unexpected_columns) / len(df.columns)) * 100,
                suggested_action="Review if these columns are needed or can be removed"
            ))
        
        return issues
    
    def _validate_missing_data(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate missing data patterns"""
        issues = []
        
        missing_counts = df.isnull().sum()
        missing_percentages = df.isnull().mean() * 100
        
        for column in df.columns:
            missing_count = missing_counts[column]
            missing_pct = missing_percentages[column]
            
            if missing_count > 0:
                if missing_pct > 50:
                    severity = ValidationSeverity.CRITICAL
                    action = "Consider removing column or investigating data source"
                elif missing_pct > 20:
                    severity = ValidationSeverity.ERROR
                    action = "Investigate missing data pattern and apply appropriate imputation"
                elif missing_pct > 5:
                    severity = ValidationSeverity.WARNING
                    action = "Consider imputation strategy for missing values"
                else:
                    severity = ValidationSeverity.INFO
                    action = "Minor missing data, can be handled with standard imputation"
                
                issues.append(ValidationIssue(
                    feature=column,
                    severity=severity,
                    issue_type="missing_data",
                    description=f"Missing values in {column}",
                    count=missing_count,
                    percentage=missing_pct,
                    suggested_action=action
                ))
        
        return issues
    
    def _validate_data_types(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate data types for features"""
        issues = []
        
        for feature in CLUSTERING_FEATURES:
            if feature in df.columns:
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    issues.append(ValidationIssue(
                        feature=feature,
                        severity=ValidationSeverity.ERROR,
                        issue_type="incorrect_data_type",
                        description=f"{feature} should be numeric but is {df[feature].dtype}",
                        count=len(df),
                        percentage=100.0,
                        suggested_action="Convert to numeric type or check data parsing"
                    ))
        
        # Check categorical features
        for feature in CATEGORICAL_FEATURES:
            if feature in df.columns:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    unique_values = df[feature].nunique()
                    if unique_values > 20:  # Too many categories
                        issues.append(ValidationIssue(
                            feature=feature,
                            severity=ValidationSeverity.WARNING,
                            issue_type="high_cardinality",
                            description=f"{feature} has {unique_values} unique values",
                            count=unique_values,
                            percentage=(unique_values / len(df)) * 100,
                            suggested_action="Consider grouping categories or feature engineering"
                        ))
        
        return issues
    
    def _validate_feature_ranges(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate feature values are within expected ranges"""
        issues = []
        
        for feature in CLUSTERING_FEATURES:
            if feature in df.columns and feature in FEATURE_DEFINITIONS:
                min_val, max_val = get_feature_range(feature)
                
                # Check for values outside range
                below_min = (df[feature] < min_val).sum()
                above_max = (df[feature] > max_val).sum()
                total_outliers = below_min + above_max
                
                if total_outliers > 0:
                    outlier_pct = (total_outliers / len(df)) * 100
                    
                    if outlier_pct > 10:
                        severity = ValidationSeverity.ERROR
                    elif outlier_pct > 5:
                        severity = ValidationSeverity.WARNING
                    else:
                        severity = ValidationSeverity.INFO
                    
                    issues.append(ValidationIssue(
                        feature=feature,
                        severity=severity,
                        issue_type="out_of_range_values",
                        description=f"{feature} has values outside expected range [{min_val}, {max_val}]",
                        count=total_outliers,
                        percentage=outlier_pct,
                        suggested_action="Clip values to valid range or investigate data source"
                    ))
        
        return issues
    
    def _validate_duplicates(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate duplicate records"""
        issues = []
        
        # Check for complete duplicates
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            dup_pct = (duplicate_rows / len(df)) * 100
            
            severity = ValidationSeverity.WARNING if dup_pct < 5 else ValidationSeverity.ERROR
            
            issues.append(ValidationIssue(
                feature="dataset",
                severity=severity,
                issue_type="duplicate_rows",
                description="Duplicate rows found in dataset",
                count=duplicate_rows,
                percentage=dup_pct,
                suggested_action="Remove duplicate rows or investigate data source"
            ))
        
        # Check for duplicate IDs (if ID column exists)
        if 'id' in df.columns:
            duplicate_ids = df['id'].duplicated().sum()
            if duplicate_ids > 0:
                issues.append(ValidationIssue(
                    feature="id",
                    severity=ValidationSeverity.ERROR,
                    issue_type="duplicate_ids",
                    description="Duplicate IDs found",
                    count=duplicate_ids,
                    percentage=(duplicate_ids / len(df)) * 100,
                    suggested_action="Ensure unique identifiers for each record"
                ))
        
        return issues
    
    def _validate_feature_relationships(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate logical relationships between features"""
        issues = []
        
        # Check for impossible combinations
        available_features = set(df.columns) & set(CLUSTERING_FEATURES)
        
        # Energy and acousticness should generally be inversely related
        if {'energy', 'acousticness'} <= available_features:
            high_energy_acoustic = ((df['energy'] > 0.8) & (df['acousticness'] > 0.8)).sum()
            if high_energy_acoustic > 0:
                issues.append(ValidationIssue(
                    feature="energy/acousticness",
                    severity=ValidationSeverity.INFO,
                    issue_type="unusual_relationship",
                    description="Songs with both high energy and high acousticness",
                    count=high_energy_acoustic,
                    percentage=(high_energy_acoustic / len(df)) * 100,
                    suggested_action="Review these tracks as they may be unusual or mislabeled"
                ))
        
        # Instrumentalness and speechiness relationship
        if {'instrumentalness', 'speechiness'} <= available_features:
            high_both = ((df['instrumentalness'] > 0.8) & (df['speechiness'] > 0.8)).sum()
            if high_both > 0:
                issues.append(ValidationIssue(
                    feature="instrumentalness/speechiness",
                    severity=ValidationSeverity.WARNING,
                    issue_type="contradictory_features",
                    description="Songs with both high instrumentalness and high speechiness",
                    count=high_both,
                    percentage=(high_both / len(df)) * 100,
                    suggested_action="Investigate these contradictory feature combinations"
                ))
        
        return issues
    
    def _validate_distribution_anomalies(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate for distribution anomalies (strict validation only)"""
        issues = []
        
        for feature in CLUSTERING_FEATURES:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                # Check for highly skewed distributions
                skewness = df[feature].skew()
                if abs(skewness) > 3:
                    issues.append(ValidationIssue(
                        feature=feature,
                        severity=ValidationSeverity.INFO,
                        issue_type="high_skewness",
                        description=f"{feature} has high skewness ({skewness:.2f})",
                        count=0,
                        percentage=0,
                        suggested_action="Consider log transformation or other normalization"
                    ))
                
                # Check for zero variance
                if df[feature].var() == 0:
                    issues.append(ValidationIssue(
                        feature=feature,
                        severity=ValidationSeverity.ERROR,
                        issue_type="zero_variance",
                        description=f"{feature} has zero variance (all values identical)",
                        count=len(df),
                        percentage=100.0,
                        suggested_action="Remove feature or investigate data source"
                    ))
        
        return issues
    
    def _validate_business_logic(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate business logic constraints"""
        issues = []
        
        # Check for reasonable duration ranges
        if 'duration_ms' in df.columns:
            # Very short songs (< 30 seconds)
            very_short = (df['duration_ms'] < 30000).sum()
            if very_short > 0:
                issues.append(ValidationIssue(
                    feature="duration_ms",
                    severity=ValidationSeverity.INFO,
                    issue_type="very_short_tracks",
                    description="Tracks shorter than 30 seconds",
                    count=very_short,
                    percentage=(very_short / len(df)) * 100,
                    suggested_action="Review if these are legitimate tracks or audio snippets"
                ))
            
            # Very long songs (> 10 minutes)
            very_long = (df['duration_ms'] > 600000).sum()
            if very_long > 0:
                issues.append(ValidationIssue(
                    feature="duration_ms",
                    severity=ValidationSeverity.INFO,
                    issue_type="very_long_tracks",
                    description="Tracks longer than 10 minutes",
                    count=very_long,
                    percentage=(very_long / len(df)) * 100,
                    suggested_action="Review if these are legitimate tracks or compilations"
                ))
        
        return issues
    
    def _validate_statistical_consistency(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate statistical consistency"""
        issues = []
        
        # Check for features with extreme outliers using IQR method
        for feature in CLUSTERING_FEATURES:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR  # More strict than standard 1.5
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
                if outliers > 0:
                    outlier_pct = (outliers / len(df)) * 100
                    if outlier_pct > 1:  # More than 1% outliers
                        issues.append(ValidationIssue(
                            feature=feature,
                            severity=ValidationSeverity.INFO,
                            issue_type="statistical_outliers",
                            description=f"{feature} has {outliers} statistical outliers",
                            count=outliers,
                            percentage=outlier_pct,
                            suggested_action="Consider outlier treatment before analysis"
                        ))
        
        return issues
    
    def _calculate_quality_score(self, issues: List[ValidationIssue], total_rows: int) -> float:
        """Calculate overall data quality score (0-100)"""
        if not issues:
            return 100.0
        
        # Weight penalties by severity
        penalty_weights = {
            ValidationSeverity.INFO: 1,
            ValidationSeverity.WARNING: 3,
            ValidationSeverity.ERROR: 10,
            ValidationSeverity.CRITICAL: 25
        }
        
        total_penalty = 0
        for issue in issues:
            weight = penalty_weights[issue.severity]
            # Penalty proportional to percentage affected
            penalty = weight * (issue.percentage / 100)
            total_penalty += penalty
        
        # Convert to 0-100 scale
        score = max(0, 100 - total_penalty)
        return score
    
    def _determine_quality_rating(self, score: float) -> str:
        """Determine quality rating from score"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        else:
            return "poor"
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate overall recommendations based on issues"""
        recommendations = []
        
        # Count issues by severity
        severity_counts = {}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        if ValidationSeverity.CRITICAL in severity_counts:
            recommendations.append("Address critical issues immediately before proceeding with analysis")
        
        if ValidationSeverity.ERROR in severity_counts:
            recommendations.append("Fix data errors to ensure reliable analysis results")
        
        if ValidationSeverity.WARNING in severity_counts:
            recommendations.append("Review warnings and apply appropriate data preprocessing")
        
        # Specific recommendations based on issue types
        issue_types = {issue.issue_type for issue in issues}
        
        if "missing_data" in issue_types:
            recommendations.append("Implement appropriate missing data imputation strategy")
        
        if "out_of_range_values" in issue_types:
            recommendations.append("Apply feature value clipping or investigate data source quality")
        
        if "duplicate_rows" in issue_types:
            recommendations.append("Remove duplicate records before analysis")
        
        if not recommendations:
            recommendations.append("Data quality is good - proceed with analysis")
        
        return recommendations
    
    def _generate_summary(self, df: pd.DataFrame, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            'total_issues': len(issues),
            'issues_by_severity': {
                severity.value: sum(1 for issue in issues if issue.severity == severity)
                for severity in ValidationSeverity
            },
            'issues_by_type': {
                issue_type: sum(1 for issue in issues if issue.issue_type == issue_type)
                for issue_type in set(issue.issue_type for issue in issues)
            },
            'features_with_issues': len(set(issue.feature for issue in issues)),
            'clean_features': len([col for col in df.columns if col not in {issue.feature for issue in issues}])
        }

# Convenience functions
def quick_validate(df: pd.DataFrame) -> ValidationReport:
    """Quick validation with basic checks"""
    validator = DataValidator(ValidationLevel.BASIC)
    return validator.validate_dataset(df)

def comprehensive_validate(df: pd.DataFrame) -> ValidationReport:
    """Comprehensive validation with all checks"""
    validator = DataValidator(ValidationLevel.STRICT)
    return validator.validate_dataset(df)