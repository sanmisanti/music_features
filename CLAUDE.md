# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Read Full Project Context

**🔗 ALWAYS READ FULL_PROJECT.md FIRST** - This file contains the complete vision, architecture, and technical roadmap for the multimodal music recommendation system. The current repository focuses on the musical characteristics analysis module within this larger system.

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
- `clustering/clustering_results.csv` - Results with cluster assignments
- `audio_analysis/we_will_rock_you_openl3.npy` - Example OpenL3 embeddings