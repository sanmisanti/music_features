"""
Musical Features Configuration

This module defines all musical features and their properties for analysis.
Based on Spotify Audio Features API and additional derived features.
"""

from typing import Dict, List, Tuple
from enum import Enum

class FeatureType(Enum):
    """Types of musical features"""
    AUDIO = "audio"           # Core audio characteristics
    RHYTHMIC = "rhythmic"     # Rhythm and tempo related
    HARMONIC = "harmonic"     # Harmonic and tonal features
    STRUCTURAL = "structural" # Song structure features
    METADATA = "metadata"     # Non-audio metadata
    DERIVED = "derived"       # Engineered features

# Core Spotify Audio Features (13 features)
SPOTIFY_AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
]

# Feature definitions with properties
FEATURE_DEFINITIONS = {
    # === AUDIO CHARACTERISTICS ===
    'danceability': {
        'type': FeatureType.AUDIO,
        'range': (0.0, 1.0),
        'description': 'How suitable a track is for dancing',
        'spotify_feature': True,
        'normalization': 'none',  # Already normalized
        'outlier_method': 'iqr'
    },
    'energy': {
        'type': FeatureType.AUDIO,
        'range': (0.0, 1.0),
        'description': 'Perceptual measure of intensity and power',
        'spotify_feature': True,
        'normalization': 'none',
        'outlier_method': 'iqr'
    },
    'valence': {
        'type': FeatureType.AUDIO,
        'range': (0.0, 1.0),
        'description': 'Musical positiveness conveyed by a track',
        'spotify_feature': True,
        'normalization': 'none',
        'outlier_method': 'iqr'
    },
    'acousticness': {
        'type': FeatureType.AUDIO,
        'range': (0.0, 1.0),
        'description': 'Confidence measure of whether track is acoustic',
        'spotify_feature': True,
        'normalization': 'none',
        'outlier_method': 'iqr'
    },
    'instrumentalness': {
        'type': FeatureType.AUDIO,
        'range': (0.0, 1.0),
        'description': 'Predicts whether a track contains no vocals',
        'spotify_feature': True,
        'normalization': 'none',
        'outlier_method': 'iqr'
    },
    'liveness': {
        'type': FeatureType.AUDIO,
        'range': (0.0, 1.0),
        'description': 'Detects the presence of an audience in recording',
        'spotify_feature': True,
        'normalization': 'none',
        'outlier_method': 'iqr'
    },
    'speechiness': {
        'type': FeatureType.AUDIO,
        'range': (0.0, 1.0),
        'description': 'Detects the presence of spoken words in a track',
        'spotify_feature': True,
        'normalization': 'none',
        'outlier_method': 'iqr'
    },
    
    # === RHYTHMIC FEATURES ===
    'tempo': {
        'type': FeatureType.RHYTHMIC,
        'range': (30.0, 300.0),
        'description': 'Overall estimated tempo in BPM',
        'spotify_feature': True,
        'normalization': 'standard',
        'outlier_method': 'zscore'
    },
    'time_signature': {
        'type': FeatureType.RHYTHMIC,
        'range': (1, 7),
        'description': 'Estimated time signature (beats per bar)',
        'spotify_feature': True,
        'normalization': 'none',
        'outlier_method': 'none'
    },
    
    # === HARMONIC FEATURES ===
    'key': {
        'type': FeatureType.HARMONIC,
        'range': (0, 11),
        'description': 'Estimated key of the track (pitch class notation)',
        'spotify_feature': True,
        'normalization': 'none',
        'outlier_method': 'none'
    },
    'mode': {
        'type': FeatureType.HARMONIC,
        'range': (0, 1),
        'description': 'Modality (major=1, minor=0)',
        'spotify_feature': True,
        'normalization': 'none',
        'outlier_method': 'none'
    },
    'loudness': {
        'type': FeatureType.HARMONIC,
        'range': (-80.0, 10.0),
        'description': 'Overall loudness in decibels (dB)',
        'spotify_feature': True,
        'normalization': 'standard',
        'outlier_method': 'zscore'
    },
    
    # === STRUCTURAL FEATURES ===
    'duration_ms': {
        'type': FeatureType.STRUCTURAL,
        'range': (10000, 1800000),  # 10 seconds to 30 minutes
        'description': 'Duration of track in milliseconds',
        'spotify_feature': True,
        'normalization': 'log',  # Highly skewed distribution
        'outlier_method': 'iqr'
    },
    
    # === METADATA FEATURES ===
    'year': {
        'type': FeatureType.METADATA,
        'range': (1900, 2030),
        'description': 'Release year of the track',
        'spotify_feature': False,
        'normalization': 'none',
        'outlier_method': 'none'
    },
    'explicit': {
        'type': FeatureType.METADATA,
        'range': (0, 1),
        'description': 'Whether track has explicit lyrics',
        'spotify_feature': False,
        'normalization': 'none',
        'outlier_method': 'none'
    }
}

# Feature groupings for analysis
FEATURE_GROUPS = {
    'audio_characteristics': ['danceability', 'energy', 'valence', 'acousticness'],
    'vocal_instrumental': ['speechiness', 'instrumentalness', 'liveness'],
    'rhythmic': ['tempo', 'time_signature'],
    'harmonic': ['key', 'mode', 'loudness'],
    'structural': ['duration_ms'],
    'metadata': ['year', 'explicit']
}

# Features for clustering (exclude metadata and IDs)
CLUSTERING_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'time_signature'
]

# Features that need special handling
LOG_TRANSFORM_FEATURES = ['duration_ms']
CATEGORICAL_FEATURES = ['key', 'mode', 'time_signature', 'explicit']
CONTINUOUS_FEATURES = [f for f in CLUSTERING_FEATURES if f not in CATEGORICAL_FEATURES]

# Correlation analysis groups
HIGH_CORRELATION_PAIRS = [
    ('energy', 'loudness'),      # Often correlated
    ('energy', 'acousticness'),  # Inversely correlated
    ('valence', 'danceability'), # Often positively correlated
]

# Features for dimensionality reduction
PCA_FEATURES = CLUSTERING_FEATURES
UMAP_FEATURES = CLUSTERING_FEATURES

def get_features_by_type(feature_type: FeatureType) -> List[str]:
    """Get all features of a specific type"""
    return [name for name, props in FEATURE_DEFINITIONS.items() 
            if props['type'] == feature_type]

def get_spotify_features() -> List[str]:
    """Get only Spotify audio features"""
    return [name for name, props in FEATURE_DEFINITIONS.items() 
            if props.get('spotify_feature', False)]

def get_feature_range(feature_name: str) -> Tuple[float, float]:
    """Get valid range for a feature"""
    return FEATURE_DEFINITIONS.get(feature_name, {}).get('range', (0, 1))

def get_normalization_method(feature_name: str) -> str:
    """Get recommended normalization method for feature"""
    return FEATURE_DEFINITIONS.get(feature_name, {}).get('normalization', 'standard')

def get_outlier_method(feature_name: str) -> str:
    """Get recommended outlier detection method for feature"""
    return FEATURE_DEFINITIONS.get(feature_name, {}).get('outlier_method', 'iqr')

def validate_feature_value(feature_name: str, value: float) -> bool:
    """Validate if a feature value is within expected range"""
    if feature_name not in FEATURE_DEFINITIONS:
        return True  # Unknown feature, assume valid
    
    min_val, max_val = get_feature_range(feature_name)
    return min_val <= value <= max_val

# Display names for plots and reports
FEATURE_DISPLAY_NAMES = {
    'danceability': 'Danceability',
    'energy': 'Energy',
    'key': 'Key',
    'loudness': 'Loudness (dB)',
    'mode': 'Mode (Major/Minor)',
    'speechiness': 'Speechiness',
    'acousticness': 'Acousticness',
    'instrumentalness': 'Instrumentalness',
    'liveness': 'Liveness',
    'valence': 'Valence (Positivity)',
    'tempo': 'Tempo (BPM)',
    'duration_ms': 'Duration (ms)',
    'time_signature': 'Time Signature',
    'year': 'Release Year',
    'explicit': 'Explicit Content'
}

# Color schemes for different feature types
FEATURE_COLORS = {
    FeatureType.AUDIO: '#1f77b4',      # Blue
    FeatureType.RHYTHMIC: '#ff7f0e',   # Orange  
    FeatureType.HARMONIC: '#2ca02c',   # Green
    FeatureType.STRUCTURAL: '#d62728', # Red
    FeatureType.METADATA: '#9467bd',   # Purple
    FeatureType.DERIVED: '#8c564b'     # Brown
}