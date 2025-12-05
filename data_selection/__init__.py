"""
DATA SELECTION MODULE - Clustering-Aware
=========================================

Modulo para seleccion de datos con preservacion de estructura natural de clustering.

Este modulo contiene el pipeline clustering-aware que genero el dataset
picked_data_optimal.csv (10,000 canciones) desde spotify_songs_fixed.csv (18,454).

Componentes activos:
- clustering_aware/: Pipeline de seleccion con validacion Hopkins
  - select_optimal_10k_from_18k.py: Script principal de generacion
  - hopkins_validator.py: Validador Hopkins Statistic

Componentes legacy (movidos a scripts/legacy/data_selection/):
- pipeline/: Pipeline 1.2M canciones (nunca ejecutado)
- sampling/: Estrategias reimplementadas en clustering_aware
- config/: Configuracion obsoleta

Dataset generado: data/3_selected/picked_data_optimal.csv
Documentacion: PIPELINE.md
"""

from .clustering_aware.hopkins_validator import HopkinsValidator, quick_hopkins_check

__version__ = "2.0.0"
__author__ = "Music Features Analysis System"

__all__ = [
    "HopkinsValidator",
    "quick_hopkins_check"
]
