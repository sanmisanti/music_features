"""
Data Loader Module

Handles intelligent loading of musical datasets with memory optimization,
validation, and preprocessing capabilities.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
import logging
from dataclasses import dataclass
import warnings

from ..config.analysis_config import config
from ..config.features_config import FEATURE_DEFINITIONS, CLUSTERING_FEATURES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadResult:
    """Result object for data loading operations"""
    data: pd.DataFrame
    metadata: Dict
    warnings: List[str]
    errors: List[str]
    success: bool

class DataLoader:
    """
    Intelligent data loader for music datasets with memory optimization
    and validation capabilities.
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize DataLoader with configuration
        
        Args:
            config_override: Optional dictionary to override default config
        """
        self.config = config
        if config_override:
            self.config.update_config(**config_override)
        
        self.load_stats = {
            'files_loaded': 0,
            'total_rows': 0,
            'memory_usage': 0,
            'load_time': 0
        }
    
    def load_dataset(
        self, 
        dataset_type: str = 'sample_500',
        sample_size: Optional[int] = None,
        random_state: Optional[int] = None,
        validate: bool = True,
        columns: Optional[List[str]] = None
    ) -> LoadResult:
        """
        Load dataset with intelligent memory management and validation
        
        Args:
            dataset_type: Type of dataset ('sample_500', 'cleaned_full', 'original')
            sample_size: Optional sample size (overrides config)
            random_state: Random state for sampling
            validate: Whether to validate data after loading
            columns: Specific columns to load (None for all)
            
        Returns:
            LoadResult object with data and metadata
        """
        import time
        start_time = time.time()
        
        warnings_list = []
        errors_list = []
        
        try:
            # Get file path
            file_path = self.config.get_data_path(dataset_type)
            
            if not file_path.exists():
                error_msg = f"Dataset file not found: {file_path}"
                logger.error(error_msg)
                return LoadResult(
                    data=pd.DataFrame(),
                    metadata={},
                    warnings=[],
                    errors=[error_msg],
                    success=False
                )
            
            logger.info(f"Loading dataset: {file_path}")
            
            # Determine loading strategy based on file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
            
            if file_size_mb > 500:  # Large file
                df = self._load_large_file(file_path, sample_size, random_state, columns)
            else:  # Small to medium file
                df = self._load_standard_file(file_path, columns)
            
            # Apply sampling if requested
            if sample_size and len(df) > sample_size:
                df = self._apply_sampling(df, sample_size, random_state)
                logger.info(f"Sampled to {len(df)} rows")
            
            # Data validation
            if validate:
                validation_warnings, validation_errors = self._validate_data(df)
                warnings_list.extend(validation_warnings)
                errors_list.extend(validation_errors)
            
            # Calculate metadata
            metadata = self._calculate_metadata(df, file_path)
            
            # Update load stats
            self.load_stats.update({
                'files_loaded': self.load_stats['files_loaded'] + 1,
                'total_rows': len(df),
                'memory_usage': df.memory_usage(deep=True).sum() / (1024**2),  # MB
                'load_time': time.time() - start_time
            })
            
            logger.info(f"Successfully loaded {len(df)} rows in {self.load_stats['load_time']:.2f}s")
            
            return LoadResult(
                data=df,
                metadata=metadata,
                warnings=warnings_list,
                errors=errors_list,
                success=len(errors_list) == 0
            )
            
        except Exception as e:
            error_msg = f"Failed to load dataset: {str(e)}"
            logger.error(error_msg)
            return LoadResult(
                data=pd.DataFrame(),
                metadata={},
                warnings=warnings_list,
                errors=[error_msg],
                success=False
            )
    
    def _load_standard_file(self, file_path: Path, columns: Optional[List[str]]) -> pd.DataFrame:
        """Load file using standard pandas read_csv"""
        return pd.read_csv(
            file_path,
            sep=self.config.data.separator,
            decimal=self.config.data.decimal,
            encoding=self.config.data.encoding,
            usecols=columns,
            on_bad_lines=self.config.data.handle_bad_lines,
            low_memory=self.config.data.low_memory
        )
    
    def _load_large_file(
        self, 
        file_path: Path, 
        sample_size: Optional[int],
        random_state: Optional[int],
        columns: Optional[List[str]]
    ) -> pd.DataFrame:
        """Load large file using chunking strategy"""
        logger.info("Using chunked loading for large file")
        
        # First, get total number of rows
        total_rows = sum(1 for _ in open(file_path, 'r', encoding=self.config.data.encoding)) - 1
        logger.info(f"Total rows in file: {total_rows:,}")
        
        # If sample size is specified and much smaller than total, use random sampling
        if sample_size and sample_size < total_rows * 0.1:
            return self._load_random_sample(file_path, sample_size, random_state, columns)
        
        # Otherwise, load in chunks
        chunks = []
        chunk_iter = pd.read_csv(
            file_path,
            sep=self.config.data.separator,
            decimal=self.config.data.decimal,
            encoding=self.config.data.encoding,
            usecols=columns,
            chunksize=self.config.data.chunk_size,
            on_bad_lines=self.config.data.handle_bad_lines,
            low_memory=self.config.data.low_memory
        )
        
        rows_loaded = 0
        for chunk in chunk_iter:
            chunks.append(chunk)
            rows_loaded += len(chunk)
            
            # Stop if we have enough data for sampling
            if sample_size and rows_loaded >= sample_size * 2:
                break
                
            logger.info(f"Loaded {rows_loaded:,} rows...")
        
        return pd.concat(chunks, ignore_index=True)
    
    def _load_random_sample(
        self,
        file_path: Path,
        sample_size: int,
        random_state: Optional[int],
        columns: Optional[List[str]]
    ) -> pd.DataFrame:
        """Load random sample from large file efficiently"""
        logger.info(f"Loading random sample of {sample_size} rows")
        
        # Get total rows (excluding header)
        total_rows = sum(1 for _ in open(file_path, 'r', encoding=self.config.data.encoding)) - 1
        
        # Generate random row indices
        np.random.seed(random_state or self.config.data.random_state)
        skip_rows = sorted(np.random.choice(total_rows, size=total_rows - sample_size, replace=False))
        
        # Load data skipping random rows
        return pd.read_csv(
            file_path,
            sep=self.config.data.separator,
            decimal=self.config.data.decimal,
            encoding=self.config.data.encoding,
            usecols=columns,
            skiprows=skip_rows,
            on_bad_lines=self.config.data.handle_bad_lines,
            low_memory=self.config.data.low_memory
        )
    
    def _apply_sampling(
        self, 
        df: pd.DataFrame, 
        sample_size: int, 
        random_state: Optional[int]
    ) -> pd.DataFrame:
        """Apply sampling to loaded dataframe"""
        return df.sample(
            n=min(sample_size, len(df)),
            random_state=random_state or self.config.data.random_state
        ).reset_index(drop=True)
    
    def _validate_data(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate loaded data for common issues"""
        warnings_list = []
        errors_list = []
        
        # Check for completely empty dataframe
        if df.empty:
            errors_list.append("Loaded dataframe is empty")
            return warnings_list, errors_list
        
        # Check for missing columns
        expected_features = set(CLUSTERING_FEATURES)
        actual_features = set(df.columns)
        missing_features = expected_features - actual_features
        
        if missing_features:
            warnings_list.append(f"Missing expected features: {missing_features}")
        
        # Check for high missing data
        missing_percentages = df.isnull().mean()
        high_missing = missing_percentages[missing_percentages > self.config.data.missing_value_threshold]
        
        if not high_missing.empty:
            warnings_list.append(f"High missing data in columns: {high_missing.to_dict()}")
        
        # Check data types
        for feature in CLUSTERING_FEATURES:
            if feature in df.columns:
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    warnings_list.append(f"Non-numeric data in feature: {feature}")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings_list.append(f"Found {duplicate_count} duplicate rows")
        
        # Check for outliers in key features
        for feature in ['tempo', 'loudness', 'duration_ms']:
            if feature in df.columns:
                min_val, max_val = FEATURE_DEFINITIONS[feature]['range']
                outliers = (df[feature] < min_val) | (df[feature] > max_val)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    warnings_list.append(f"Found {outlier_count} outliers in {feature}")
        
        return warnings_list, errors_list
    
    def _calculate_metadata(self, df: pd.DataFrame, file_path: Path) -> Dict:
        """Calculate metadata about loaded dataset"""
        metadata = {
            'file_path': str(file_path),
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'dtypes': df.dtypes.to_dict(),
            'missing_data': df.isnull().sum().to_dict(),
            'missing_percentages': df.isnull().mean().to_dict(),
            'column_names': df.columns.tolist()
        }
        
        # Add feature-specific metadata
        if 'year' in df.columns:
            metadata['year_range'] = (df['year'].min(), df['year'].max())
        
        if 'artists' in df.columns:
            metadata['unique_artists'] = df['artists'].nunique()
        
        # Add statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            metadata['numeric_stats'] = df[numeric_columns].describe().to_dict()
        
        return metadata
    
    def get_load_statistics(self) -> Dict:
        """Get statistics about loading operations"""
        return self.load_stats.copy()
    
    def reset_statistics(self):
        """Reset loading statistics"""
        self.load_stats = {
            'files_loaded': 0,
            'total_rows': 0,
            'memory_usage': 0,
            'load_time': 0
        }

# Convenience functions
def load_sample_dataset(sample_size: int = 500) -> LoadResult:
    """Quick function to load sample dataset"""
    loader = DataLoader()
    return loader.load_dataset('sample_500', sample_size=sample_size)

def load_full_dataset(sample_size: Optional[int] = None) -> LoadResult:
    """Quick function to load full cleaned dataset"""
    loader = DataLoader()
    return loader.load_dataset('cleaned_full', sample_size=sample_size)

def load_clustering_features_only(dataset_type: str = 'sample_500') -> LoadResult:
    """Quick function to load only clustering-relevant features"""
    loader = DataLoader()
    return loader.load_dataset(dataset_type, columns=CLUSTERING_FEATURES)