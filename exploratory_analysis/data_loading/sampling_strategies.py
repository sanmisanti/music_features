"""
Sampling Strategies Module

Provides various sampling strategies for music datasets including
stratified, balanced, and diversity-based sampling.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SamplingMethod(Enum):
    """Available sampling methods"""
    RANDOM = "random"
    STRATIFIED = "stratified"
    BALANCED = "balanced"
    DIVERSITY = "diversity"

class SamplingStrategies:
    """
    Collection of sampling strategies for music datasets
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize sampling strategies
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def random_sample(
        self, 
        df: pd.DataFrame, 
        sample_size: int,
        replace: bool = False
    ) -> pd.DataFrame:
        """
        Simple random sampling
        
        Args:
            df: DataFrame to sample from
            sample_size: Number of samples to draw
            replace: Whether to sample with replacement
            
        Returns:
            Sampled DataFrame
        """
        if sample_size >= len(df):
            logger.warning(f"Sample size ({sample_size}) >= dataset size ({len(df)}). Returning full dataset.")
            return df.copy()
        
        return df.sample(
            n=sample_size, 
            replace=replace, 
            random_state=self.random_state
        ).reset_index(drop=True)
    
    def stratified_sample(
        self,
        df: pd.DataFrame,
        sample_size: int,
        stratify_column: str,
        min_samples_per_stratum: int = 1
    ) -> pd.DataFrame:
        """
        Stratified sampling based on a categorical column
        
        Args:
            df: DataFrame to sample from
            sample_size: Total number of samples to draw
            stratify_column: Column to stratify by
            min_samples_per_stratum: Minimum samples per category
            
        Returns:
            Stratified sample DataFrame
        """
        if stratify_column not in df.columns:
            logger.warning(f"Stratification column '{stratify_column}' not found. Using random sampling.")
            return self.random_sample(df, sample_size)
        
        # Get proportions of each category
        value_counts = df[stratify_column].value_counts()
        proportions = value_counts / len(df)
        
        # Calculate samples per stratum
        samples_per_stratum = {}
        remaining_samples = sample_size
        
        for category, proportion in proportions.items():
            n_samples = max(
                min_samples_per_stratum,
                int(proportion * sample_size)
            )
            n_samples = min(n_samples, remaining_samples, value_counts[category])
            samples_per_stratum[category] = n_samples
            remaining_samples -= n_samples
        
        # Sample from each stratum
        sampled_dfs = []
        for category, n_samples in samples_per_stratum.items():
            if n_samples > 0:
                category_df = df[df[stratify_column] == category]
                if len(category_df) >= n_samples:
                    sampled_df = category_df.sample(
                        n=n_samples, 
                        random_state=self.random_state
                    )
                    sampled_dfs.append(sampled_df)
        
        if sampled_dfs:
            result = pd.concat(sampled_dfs, ignore_index=True)
            return result.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        else:
            logger.warning("Stratified sampling failed. Using random sampling.")
            return self.random_sample(df, sample_size)
    
    def balanced_sample(
        self,
        df: pd.DataFrame,
        sample_size: int,
        balance_columns: List[str],
        balance_method: str = 'equal'
    ) -> pd.DataFrame:
        """
        Balanced sampling across multiple dimensions
        
        Args:
            df: DataFrame to sample from
            sample_size: Total number of samples
            balance_columns: Columns to balance across
            balance_method: 'equal' or 'proportional'
            
        Returns:
            Balanced sample DataFrame
        """
        if not all(col in df.columns for col in balance_columns):
            missing_cols = [col for col in balance_columns if col not in df.columns]
            logger.warning(f"Balance columns {missing_cols} not found. Using random sampling.")
            return self.random_sample(df, sample_size)
        
        if balance_method == 'equal':
            return self._equal_balanced_sample(df, sample_size, balance_columns)
        else:
            return self._proportional_balanced_sample(df, sample_size, balance_columns)
    
    def _equal_balanced_sample(
        self,
        df: pd.DataFrame,
        sample_size: int,
        balance_columns: List[str]
    ) -> pd.DataFrame:
        """Equal representation across balance dimensions"""
        # Create combination groups
        df_copy = df.copy()
        df_copy['_group'] = df_copy[balance_columns].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        
        unique_groups = df_copy['_group'].unique()
        samples_per_group = max(1, sample_size // len(unique_groups))
        
        sampled_dfs = []
        for group in unique_groups:
            group_df = df_copy[df_copy['_group'] == group]
            n_samples = min(samples_per_group, len(group_df))
            if n_samples > 0:
                sampled_df = group_df.sample(n=n_samples, random_state=self.random_state)
                sampled_dfs.append(sampled_df)
        
        if sampled_dfs:
            result = pd.concat(sampled_dfs, ignore_index=True)
            result = result.drop('_group', axis=1)
            return result.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        else:
            return self.random_sample(df, sample_size)
    
    def _proportional_balanced_sample(
        self,
        df: pd.DataFrame,
        sample_size: int,
        balance_columns: List[str]
    ) -> pd.DataFrame:
        """Proportional representation across balance dimensions"""
        # This is essentially stratified sampling on combined columns
        df_copy = df.copy()
        df_copy['_group'] = df_copy[balance_columns].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        
        result = self.stratified_sample(df_copy, sample_size, '_group')
        if '_group' in result.columns:
            result = result.drop('_group', axis=1)
        
        return result
    
    def diversity_sample(
        self,
        df: pd.DataFrame,
        sample_size: int,
        feature_columns: List[str],
        diversity_method: str = 'maxmin'
    ) -> pd.DataFrame:
        """
        Diversity-based sampling to maximize feature space coverage
        
        Args:
            df: DataFrame to sample from
            sample_size: Number of samples to draw
            feature_columns: Numerical features to consider for diversity
            diversity_method: 'maxmin' or 'clustering'
            
        Returns:
            Diverse sample DataFrame
        """
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            logger.warning("No valid feature columns found for diversity sampling. Using random sampling.")
            return self.random_sample(df, sample_size)
        
        if diversity_method == 'maxmin':
            return self._maxmin_diversity_sample(df, sample_size, available_features)
        else:
            return self._clustering_diversity_sample(df, sample_size, available_features)
    
    def _maxmin_diversity_sample(
        self,
        df: pd.DataFrame,
        sample_size: int,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """MaxMin diversity sampling"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Standardize features
        scaler = StandardScaler()
        features = scaler.fit_transform(df[feature_columns].fillna(df[feature_columns].mean()))
        
        # Start with random point
        selected_indices = [np.random.randint(0, len(df))]
        remaining_indices = list(range(len(df)))
        remaining_indices.remove(selected_indices[0])
        
        # Iteratively select most diverse points
        for _ in range(min(sample_size - 1, len(remaining_indices))):
            if not remaining_indices:
                break
                
            selected_features = features[selected_indices]
            remaining_features = features[remaining_indices]
            
            # Calculate minimum distances to selected points
            distances = euclidean_distances(remaining_features, selected_features)
            min_distances = distances.min(axis=1)
            
            # Select point with maximum minimum distance
            max_min_idx = np.argmax(min_distances)
            selected_idx = remaining_indices[max_min_idx]
            
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
        
        return df.iloc[selected_indices].reset_index(drop=True)
    
    def _clustering_diversity_sample(
        self,
        df: pd.DataFrame,
        sample_size: int,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """Clustering-based diversity sampling"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Standardize features
            scaler = StandardScaler()
            features = scaler.fit_transform(df[feature_columns].fillna(df[feature_columns].mean()))
            
            # Perform clustering
            n_clusters = min(sample_size, 50)  # Reasonable upper limit
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Sample from each cluster
            sampled_indices = []
            samples_per_cluster = sample_size // n_clusters
            remaining_samples = sample_size % n_clusters
            
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    n_samples = samples_per_cluster
                    if remaining_samples > 0:
                        n_samples += 1
                        remaining_samples -= 1
                    
                    n_samples = min(n_samples, len(cluster_indices))
                    selected = np.random.choice(cluster_indices, size=n_samples, replace=False)
                    sampled_indices.extend(selected)
            
            return df.iloc[sampled_indices].reset_index(drop=True)
            
        except ImportError:
            logger.warning("scikit-learn not available for clustering diversity sampling. Using maxmin method.")
            return self._maxmin_diversity_sample(df, sample_size, feature_columns)
    
    def get_sample_statistics(self, original_df: pd.DataFrame, sample_df: pd.DataFrame) -> Dict:
        """
        Calculate statistics comparing original and sampled datasets
        
        Args:
            original_df: Original dataset
            sample_df: Sampled dataset
            
        Returns:
            Dictionary with comparison statistics
        """
        stats = {
            'original_size': len(original_df),
            'sample_size': len(sample_df),
            'sampling_ratio': len(sample_df) / len(original_df),
            'columns_preserved': len(sample_df.columns) == len(original_df.columns)
        }
        
        # Compare distributions for numerical columns
        numerical_cols = original_df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            original_means = original_df[numerical_cols].mean()
            sample_means = sample_df[numerical_cols].mean()
            
            stats['mean_differences'] = {
                col: abs(original_means[col] - sample_means[col])
                for col in numerical_cols
                if col in sample_df.columns
            }
        
        return stats

# Convenience functions
def quick_sample(df: pd.DataFrame, sample_size: int, method: str = 'random') -> pd.DataFrame:
    """Quick sampling with default parameters"""
    sampler = SamplingStrategies()
    
    if method == 'random':
        return sampler.random_sample(df, sample_size)
    elif method == 'stratified' and 'year' in df.columns:
        return sampler.stratified_sample(df, sample_size, 'year')
    else:
        return sampler.random_sample(df, sample_size)