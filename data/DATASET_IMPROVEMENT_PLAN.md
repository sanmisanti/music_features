# Plan de Mejora de Dataset Multimodal

## Objetivo

Obtener un dataset multimodal metodologicamente validado a partir de los datasets existentes, resolviendo las limitaciones identificadas en el dataset actual de 7,811 canciones.

---

## Estado Actual

### Datasets Disponibles

| Dataset | Registros | Ubicacion | Estado |
|---------|-----------|-----------|--------|
| spotify_songs_fixed.csv | 18,454 | 2_with_lyrics/ | Fuente validada (Hopkins 0.823) |
| picked_data_optimal.csv | 10,000 | 3_selected/ | Seleccion musical optimizada |
| embeddings_bert_9753x384.npy | 9,753 (8,567 validos) | 4_vectorized/ | Embeddings con 12.2% fallos |
| unified_multimodal_7811.pkl | 7,811 | 5_unified/ | Resultado residual NO OPTIMIZADO |

### Problema Central

El dataset de 7,811 canciones es el **resultado residual de filtros tecnicos**, no un dataset disenado con criterios de calidad explicitos:

```
10,000 (seleccion inicial)
   |
   v  -247 sin letras validas
9,753
   |
   v  -1,186 fallos BERT (vectores cero)
8,567
   |
   v  -756 deduplicacion/no-match
7,811 (residual)
```

**Perdida total**: 21.9% de los datos sin analisis de impacto.

---

## Plan de Mejora en 4 Fases

### FASE A: Validacion del Dataset Actual

**Objetivo**: Establecer baseline con metricas objetivas antes de modificar.

**Tareas**:

1. **Calcular Hopkins Statistic del dataset unificado**
   ```python
   from sklearn.neighbors import NearestNeighbors
   import numpy as np

   # Cargar embeddings semanticos
   semantic = np.load('data/5_unified/arrays/semantic_embeddings.npy')

   # Hopkins en espacio 384D
   # Implementar calculo o usar sklearn-extra
   ```

2. **Comparar distribucion de generos vs dataset original**
   ```python
   import pandas as pd

   original = pd.read_csv('data/2_with_lyrics/spotify_songs_fixed.csv',
                          sep='@@', engine='python')
   unified = pd.read_csv('data/5_unified/aligned_songs.csv')

   # Comparar proporciones
   original_dist = original['playlist_genre'].value_counts(normalize=True)
   unified_dist = unified['playlist_genre'].value_counts(normalize=True)

   # Calcular KL-divergence
   from scipy.stats import entropy
   kl_div = entropy(unified_dist, original_dist)
   ```

3. **Analizar caracteristicas de canciones excluidas**
   ```python
   # Identificar track_ids excluidos
   original_ids = set(original['track_id'])
   unified_ids = set(unified['track_id'])
   excluded_ids = original_ids - unified_ids

   # Comparar features musicales
   excluded = original[original['track_id'].isin(excluded_ids)]
   included = original[original['track_id'].isin(unified_ids)]

   # Test estadistico por feature
   from scipy.stats import mannwhitneyu
   for feature in clustering_features:
       stat, pval = mannwhitneyu(excluded[feature], included[feature])
       print(f"{feature}: p-value = {pval:.4f}")
   ```

4. **Test de estabilidad de clustering**
   ```python
   from sklearn.cluster import KMeans
   from sklearn.metrics import silhouette_score

   scores = []
   for seed in range(10):
       kmeans = KMeans(n_clusters=6, random_state=seed)
       labels = kmeans.fit_predict(semantic)
       score = silhouette_score(semantic, labels)
       scores.append(score)

   print(f"Silhouette: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
   print(f"CV: {np.std(scores)/np.mean(scores):.4f}")
   ```

**Metricas a obtener**:
- Hopkins Statistic (objetivo: >0.75)
- KL-divergence de generos (objetivo: <0.1)
- p-values de features excluidas vs incluidas
- CV de Silhouette entre seeds (objetivo: <0.10)

**Script**: `scripts/validation/validate_unified_dataset.py`

---

### FASE B: Analisis de Sesgo

**Objetivo**: Cuantificar sesgos introducidos por filtros tecnicos.

**Tareas**:

1. **Correlacion entre exito BERT y caracteristicas musicales**
   ```python
   # Cargar embeddings completos (con ceros)
   embeddings = np.load('data/4_vectorized/embeddings_bert_9753x384.npy')
   track_ids = np.load('data/4_vectorized/track_ids_9753.npy', allow_pickle=True)

   # Identificar exitos y fallos
   success_mask = np.any(embeddings != 0, axis=1)

   # Merge con features musicales
   df = pd.read_csv('data/3_selected/picked_data_optimal.csv', sep='^')
   df['bert_success'] = df['track_id'].isin(track_ids[success_mask])

   # Correlacion punto-biserial
   from scipy.stats import pointbiserialr
   for feature in clustering_features:
       corr, pval = pointbiserialr(df['bert_success'], df[feature])
       print(f"{feature}: r = {corr:.4f}, p = {pval:.4f}")
   ```

2. **Tasa de fallo por genero**
   ```python
   failure_rates = df.groupby('playlist_genre')['bert_success'].apply(
       lambda x: 1 - x.mean()
   ).sort_values(ascending=False)
   print("Tasa de fallo BERT por genero:")
   print(failure_rates)
   ```

3. **Analisis de longitud de letras vs exito BERT**
   ```python
   df['lyrics_length'] = df['lyrics'].str.len()
   df['lyrics_words'] = df['lyrics'].str.split().str.len()

   # Comparar longitudes
   success = df[df['bert_success']]
   failure = df[~df['bert_success']]

   print(f"Longitud media (exito): {success['lyrics_length'].mean():.0f}")
   print(f"Longitud media (fallo): {failure['lyrics_length'].mean():.0f}")
   ```

**Entregable**: `reports/dataset_bias_analysis.md`

**Script**: `scripts/validation/analyze_dataset_bias.py`

---

### FASE C: Dataset Alternativo Optimizado

**Objetivo**: Crear dataset con criterios metodologicos explicitos.

#### Estrategia 1: Subsampling Balanceado por Genero

**Justificacion**: Equilibrar distribucion de generos para evitar sesgo en clustering.

```python
# Stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit

# Objetivo: ~1,200 canciones por genero (7,200 total)
samples_per_genre = 1200

balanced_df = df.groupby('playlist_genre').apply(
    lambda x: x.sample(n=min(len(x), samples_per_genre), random_state=42)
).reset_index(drop=True)

print(f"Dataset balanceado: {len(balanced_df)} canciones")
print(balanced_df['playlist_genre'].value_counts())
```

#### Estrategia 2: Re-vectorizacion Selectiva

**Justificacion**: Recuperar canciones con vectorizacion fallida mediante preprocessing mejorado.

```python
# Identificar fallos
failed_ids = track_ids[~success_mask]

# Analizar causas
failed_lyrics = df[df['track_id'].isin(failed_ids)]['lyrics']

# Causas potenciales:
# 1. Letras muy cortas (<50 caracteres)
# 2. Caracteres especiales problematicos
# 3. Idiomas no soportados

# Preprocessing mejorado
def clean_lyrics(text):
    import re
    text = re.sub(r'[^\w\s]', ' ', text)  # Eliminar puntuacion
    text = re.sub(r'\s+', ' ', text)       # Normalizar espacios
    text = text.strip()
    return text if len(text) > 50 else None

# Re-intentar vectorizacion
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

recoverable = []
for idx, row in df[df['track_id'].isin(failed_ids)].iterrows():
    cleaned = clean_lyrics(row['lyrics'])
    if cleaned:
        recoverable.append(row['track_id'])

print(f"Potencialmente recuperables: {len(recoverable)} canciones")
```

**Potencial**: Recuperar 500-800 canciones adicionales.

#### Estrategia 3: Seleccion Hopkins-Aware

**Justificacion**: Maximizar clusterabilidad del dataset final.

```python
# Cargar dataset de 10,000
df = pd.read_csv('data/3_selected/picked_data_optimal.csv', sep='^')

# Features para Hopkins
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo']

X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcular Hopkins por subconjunto
def hopkins_statistic(X, sample_size=100):
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    d = X.shape[1]

    # Muestrear puntos
    sample_idx = np.random.choice(n, sample_size, replace=False)
    X_sample = X[sample_idx]

    # Generar puntos aleatorios
    X_random = np.random.uniform(X.min(axis=0), X.max(axis=0),
                                  (sample_size, d))

    # Distancias al vecino mas cercano
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)

    u_dist = nn.kneighbors(X_random, return_distance=True)[0][:, 1]
    w_dist = nn.kneighbors(X_sample, return_distance=True)[0][:, 1]

    return u_dist.sum() / (u_dist.sum() + w_dist.sum())

# Filtrar a canciones con embeddings validos y maximizar Hopkins
valid_df = df[df['track_id'].isin(unified_ids)]
hopkins = hopkins_statistic(valid_df[features].values)
print(f"Hopkins del subconjunto valido: {hopkins:.4f}")
```

**Script**: `scripts/generation/create_optimized_multimodal_dataset.py`

---

### FASE D: Validacion Comparativa

**Objetivo**: Verificar que dataset mejorado supera al actual.

**Metricas de comparacion**:

| Metrica | Dataset Actual | Dataset Mejorado | Objetivo |
|---------|----------------|------------------|----------|
| Hopkins Statistic | ? (calcular) | >0.80 | +5% min |
| Silhouette Score | ? (calcular) | >0.15 | Mejorar |
| Balance generos | Desbalanceado | max ratio <3:1 | Equilibrar |
| Estabilidad (CV) | ? (calcular) | <0.10 | Reducir |
| Cobertura | 7,811 | 7,500+ | Mantener |

**Comparacion formal**:

```python
def compare_datasets(original, improved):
    metrics = {}

    for name, data in [('original', original), ('improved', improved)]:
        # Hopkins
        metrics[f'{name}_hopkins'] = hopkins_statistic(data['musical'])

        # Silhouette (K=6)
        kmeans = KMeans(n_clusters=6, random_state=42)
        labels = kmeans.fit_predict(data['semantic'])
        metrics[f'{name}_silhouette'] = silhouette_score(data['semantic'], labels)

        # Balance (ratio max/min generos)
        genre_counts = data['metadata']['playlist_genre'].value_counts()
        metrics[f'{name}_balance'] = genre_counts.max() / genre_counts.min()

        # Estabilidad
        scores = []
        for seed in range(10):
            km = KMeans(n_clusters=6, random_state=seed)
            labels = km.fit_predict(data['semantic'])
            scores.append(silhouette_score(data['semantic'], labels))
        metrics[f'{name}_cv'] = np.std(scores) / np.mean(scores)

    return metrics

results = compare_datasets(current_dataset, improved_dataset)
```

**Script**: `scripts/validation/compare_datasets.py`

---

## Comandos de Ejecucion

```bash
# FASE A: Validacion (prerequisito)
python scripts/validation/validate_unified_dataset.py

# FASE B: Analisis de sesgo
python scripts/validation/analyze_dataset_bias.py

# FASE C: Crear dataset optimizado
python scripts/generation/create_optimized_multimodal_dataset.py --strategy balanced

# FASE D: Comparacion
python scripts/validation/compare_datasets.py --original data/5_unified/unified_multimodal_7811.pkl \
                                              --improved data/5_unified/unified_multimodal_optimized.pkl
```

---

## Criterios de Exito

| Criterio | Umbral | Justificacion |
|----------|--------|---------------|
| Hopkins Statistic | >0.80 | Clusterabilidad excelente |
| Silhouette Score | >0.15 | Separacion de clusters aceptable |
| Balance de generos | ratio <3:1 | Representatividad |
| Estabilidad (CV) | <0.10 | Reproducibilidad |
| Cobertura minima | 7,500 | Mantener tamaño util |
| Documentacion | 100% | Trazabilidad metodologica |

---

## Estructura de Scripts a Crear

**ESTADO: PENDIENTE DE IMPLEMENTACION**

Los siguientes scripts deben crearse para ejecutar el plan:

```
scripts/
├── validation/
│   ├── validate_unified_dataset.py    # FASE A [TO-DO]
│   ├── analyze_dataset_bias.py        # FASE B [TO-DO]
│   └── compare_datasets.py            # FASE D [TO-DO]
└── generation/
    └── create_optimized_multimodal_dataset.py  # FASE C [TO-DO]
```

**Nota**: Los comandos de ejecucion listados arriba fallaran hasta que estos scripts sean implementados.

---

## Notas Metodologicas

### Principio de Transparencia

Cada modificacion al dataset debe documentar:
1. **Que** se modifico (registros afectados)
2. **Por que** (justificacion tecnica)
3. **Como** (metodo aplicado)
4. **Impacto** (metricas antes/despues)

### Reproducibilidad

Todos los scripts deben:
- Usar `random_state=42` para determinismo
- Guardar logs de ejecucion
- Exportar metricas en formato JSON

### Versionado

Cada version del dataset debe nombrarse:
```
unified_multimodal_{n_songs}_{version}_{date}.pkl
```

Ejemplo: `unified_multimodal_7500_v2_20250915.pkl`
