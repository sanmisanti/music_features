# 4_vectorized - Embeddings BERT de Letras

## Archivos

| Archivo | Tamano | Descripcion |
|---------|--------|-------------|
| embeddings_bert_9753x384.npy | 29 MB | Embeddings BERT [9753, 384] |
| track_ids_9753.npy | 839 KB | IDs correspondientes |
| similarity_index.pkl | 30 MB | Indice k-NN pre-computado (k=50) |
| vectorization_report.json | 1.4 KB | Reporte del proceso |
| vectorization_metadata.json | 1 KB | Metadatos tecnicos |
| load_vectorization.py | 2 KB | Script de carga con ejemplos |

## Estadisticas de Vectorizacion

| Metrica | Valor |
|---------|-------|
| Canciones de entrada | 10,000 |
| Letras validas | 9,753 |
| Embeddings exitosos (no-zero) | 8,567 |
| Embeddings fallidos | 1,186 (12.2%) |
| Dimensiones | 384 |
| Modelo BERT | paraphrase-multilingual-MiniLM-L12-v2 |
| Tiempo de procesamiento | ~20 minutos |

## Carga

```python
import numpy as np

# Cargar embeddings
embeddings = np.load('data/4_vectorized/embeddings_bert_9753x384.npy')
track_ids = np.load('data/4_vectorized/track_ids_9753.npy', allow_pickle=True)

# IMPORTANTE: Filtrar embeddings validos (no-zero)
valid_mask = np.any(embeddings != 0, axis=1)
valid_embeddings = embeddings[valid_mask]  # Shape: [8567, 384]
valid_track_ids = track_ids[valid_mask]    # Length: 8567
```

## ADVERTENCIA CRITICA

El archivo `embeddings_bert_9753x384.npy` contiene **9,753 registros** pero solo **8,567 tienen embeddings validos**.

Los 1,186 restantes son **vectores de ceros** (fallos de vectorizacion por:
- Letras muy cortas
- Idiomas no soportados
- Caracteres especiales problematicos
- Errores de procesamiento)

**SIEMPRE filtrar con**: `embeddings[np.any(embeddings != 0, axis=1)]`

## Indice de Similitud

El archivo `similarity_index.pkl` contiene un indice pre-computado para busqueda de vecinos cercanos:

```python
import pickle

with open('data/4_vectorized/similarity_index.pkl', 'rb') as f:
    similarity_index = pickle.load(f)

# Configuracion del indice
# - Algoritmo: brute force
# - Metrica: cosine
# - Max vecinos: 50
```

## Origen

Generado desde `3_selected/picked_data_optimal.csv` usando el script:
`clustering/algorithms/lyrics/scripts/run_complete_vectorization.py`

## Dataset Derivado

Estos embeddings se combinan con caracteristicas musicales en `5_unified/` para crear el dataset multimodal final.
