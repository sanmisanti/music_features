# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Read Project Context Files

**🔗 ALWAYS READ THESE FILES FIRST**:
1. **FULL_PROJECT.md** - Complete vision, architecture, and technical roadmap for the multimodal music recommendation system
2. **ANALYSIS_RESULTS.md** - Comprehensive analysis results, test outcomes, technical interpretations, and progress tracking for all implemented modules
3. **DOCS.md** - Academic technical documentation with theoretical foundations, methodologies, algorithms, and formal analysis for thesis-level understanding
4. **DIRECTIVAS.md** - Development workflow guidelines, documentation requirements, and mandatory procedures for consistent project execution

The current repository focuses on the musical characteristics analysis module within the larger multimodal system. All development progress and test results are tracked in ANALYSIS_RESULTS.md, while theoretical foundations and academic explanations are maintained in DOCS.md.

## Project Overview

This repository implements the **Musical Characteristics Analysis Module** - one component of a larger multimodal music recommendation system that combines audio features with semantic lyrics analysis. This specific module performs clustering and similarity analysis on music tracks using Spotify audio features and deep learning embeddings.

## Architecture & Components

### Core Data Pipeline
- **Data Processing**: `clean.py` handles large dataset cleaning and creates standardized CSV formats
- **Feature Analysis**: Uses 13 Spotify audio features (danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature)
- **Clustering System**: `clustering/cluster.ipynb` implements K-Means clustering with optimization
- **Recommendation Engine**: `pred.ipynb` provides cluster-based music recommendations
- **Audio Analysis**: `audio_analysis/aa_openl3.py` generates deep learning embeddings using OpenL3

### Data Structure
- **Original Dataset**: ~1.2M music tracks with Spotify features (tracks_features.csv)
- **Cleaned Versions**: Standardized datasets with proper encoding (tracks_features_clean.csv)
- **Sample Dataset**: 500-track subset for development (tracks_features_500.csv)
- **Selected Dataset**: 9,677 representative songs for final model (data/pipeline_results/final_selection_results/)
- **Clustering Results**: CSV with cluster assignments (clustering_results.csv)

### Machine Learning Workflow
1. Data cleaning and feature normalization using StandardScaler
2. K-Means clustering optimization (tested K=4-11, optimal K=7)
3. Cluster analysis and visualization using PCA
4. Similarity-based recommendations using cosine/euclidean/manhattan distance

## Common Development Commands

### Data Processing
```bash
python clean.py  # Clean original dataset and create standardized versions
```

### Selection Pipeline (NEW)
```bash
# Complete pipeline for selecting representative songs
python scripts/main_selection_pipeline.py --target-size 10000

# Individual components
python scripts/large_dataset_processor.py      # Analyze complete dataset
python scripts/representative_selector.py      # Select representative subset
python scripts/selection_validator.py          # Validate selection quality
```

### Jupyter Notebooks
```bash
jupyter notebook clustering/cluster.ipynb  # Run clustering analysis
jupyter notebook pred.ipynb               # Test recommendation system
```

### Audio Analysis
```bash
python audio_analysis/aa_openl3.py  # Generate OpenL3 embeddings for audio files
```

## Key Technical Details

### Dependencies
- **Core ML**: pandas, numpy, scikit-learn (KMeans, StandardScaler, PCA)
- **Visualization**: matplotlib, seaborn, plotly
- **Audio Processing**: openl3, librosa, soundfile

### File Encoding Notes
- Original data uses UTF-8 with comma separators
- Cleaned data uses semicolon separators with comma decimals (Spanish locale)
- Always use `pd.read_csv(path, sep=';', decimal=',')` for cleaned datasets

### Clustering Performance
- Optimal configuration: K=7 clusters with silhouette score 0.177
- Cluster distribution typically uneven (e.g., [42, 197, 110, 7, 37, 26, 81])
- Uses StandardScaler normalization before clustering

### Recommendation System
- Assigns new songs to existing clusters using trained KMeans model
- Calculates similarity within cluster using multiple distance metrics
- Returns top-N most similar tracks from the same cluster

## Research Context

This system demonstrates a complete ML pipeline for music information retrieval, combining traditional audio features with advanced deep learning embeddings. The clustering approach identifies musical patterns beyond simple genre classification, enabling sophisticated similarity-based recommendations.

## Data File Locations

- `data/original_data/tracks_features.csv` - Original 1.2M track dataset
- `data/cleaned_data/tracks_features_clean.csv` - Full cleaned dataset  
- `data/cleaned_data/tracks_features_500.csv` - 500-track sample
- `data/pipeline_results/final_selection_results/selection/selected_songs_10000_20250726_181954.csv` - **9,677 representative songs for final model**
- `data/pipeline_results/final_selection_results/` - Complete pipeline results (analysis, validation, reports)
- `clustering/clustering_results.csv` - Results with cluster assignments
- `audio_analysis/we_will_rock_you_openl3.npy` - Example OpenL3 embeddings

## Pipeline Results Summary

**Latest Execution (2025-01-26)**:
- **Input**: 1,204,025 songs from cleaned dataset
- **Output**: 9,677 representative songs (0.8% of original)
- **Quality Score**: 88.6/100 (EXCELLENT)
- **Validation**: 3/4 tests passed
- **Execution Time**: 245 seconds
- **Status**: ✅ READY FOR CLUSTERING AND RECOMMENDATION SYSTEM