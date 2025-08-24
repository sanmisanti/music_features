#!/usr/bin/env python3
# Script de carga rápida para dataset unificado multimodal
# Generado automáticamente: 20250822_004929

import numpy as np
import pickle
from pathlib import Path

def load_unified_dataset():
    """Cargar dataset unificado multimodal completo."""
    base_path = Path(__file__).parent
    dataset_path = base_path / "unified_multimodal_dataset_20250822_004929.pkl"
    
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)

def load_arrays_only():
    """Cargar solo arrays principales para análisis rápido."""
    base_path = Path(__file__).parent / "arrays_20250822_004929"
    
    return {
        'track_ids': np.load(base_path / "track_ids.npy"),
        'semantic_embeddings': np.load(base_path / "semantic_embeddings.npy"),
        'musical_features_raw': np.load(base_path / "musical_features_raw.npy"),
        'musical_features_normalized': np.load(base_path / "musical_features_normalized.npy")
    }

if __name__ == "__main__":
    print("Cargando dataset unificado...")
    dataset = load_unified_dataset()
    print(f"Dataset cargado: {dataset['metadata']['sample_size']:,} canciones")
    print(f"Dimensiones: {dataset['metadata']['semantic_dimensions']}D semántico, {dataset['metadata']['musical_dimensions']}D musical")
