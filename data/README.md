# DATA - Estructura Jerarquica de Datasets

Este directorio organiza los datasets del proyecto segun su nivel de madurez y procesamiento.

---

## Estructura de Carpetas

```
data/
├── 0_raw/              # Nivel 0: Datos originales (Spotify API)
├── 1_cleaned/          # Nivel 1: Datos con formato corregido
├── 2_with_lyrics/      # Nivel 2: Datos con letras (Kaggle)
├── 3_selected/         # Nivel 3: Dataset PRODUCTION
└── auxiliary/          # Archivos auxiliares
```

---

## Origen de los Datos

```
FUENTE 1: Spotify API                    FUENTE 2: Kaggle
(1.2M canciones, sin letras)             (18K canciones, con letras)
          |                                        |
          v                                        v
    0_raw/ y 1_cleaned/                    2_with_lyrics/
                                                   |
                                                   v
                                             3_selected/
                                      (picked_data_optimal.csv)
```

**Interseccion**: Solo 12.8% de IDs comunes entre ambas fuentes.

---

## 0_raw/ - Datos Originales

| Archivo | Registros | Separador |
|---------|-----------|-----------|
| tracks_features.csv | 1,204,025 | , |

```python
df = pd.read_csv('data/0_raw/tracks_features.csv', sep=',', encoding='utf-8')
```

---

## 1_cleaned/ - Datos Corregidos

| Archivo | Registros | Separador |
|---------|-----------|-----------|
| tracks_features_clean.csv | 1,204,025 | ; |
| tracks_features_500.csv | 500 | ; |

```python
df = pd.read_csv('data/1_cleaned/tracks_features_clean.csv', sep=';', encoding='utf-8')
```

---

## 2_with_lyrics/ - Datos con Letras

| Archivo | Registros | Separador | Hopkins |
|---------|-----------|-----------|---------|
| spotify_songs_fixed.csv | 18,454 | @@ | 0.823 |

```python
df = pd.read_csv('data/2_with_lyrics/spotify_songs_fixed.csv', sep='@@', engine='python')
```

---

## 3_selected/ - Dataset PRODUCTION

**EL UNICO DATASET QUE DEBE USARSE EN PRODUCCION**

| Archivo | Registros | Separador | Hopkins | Silhouette |
|---------|-----------|-----------|---------|------------|
| **picked_data_optimal.csv** | 10,000 | ^ | 0.823 | 0.289 |

### Comando de Carga

```python
import pandas as pd
df = pd.read_csv('data/3_selected/picked_data_optimal.csv', sep='^', encoding='utf-8')
```

### Generador

- Script: scripts/generation/generate_optimal_dataset.py
- Metodologia: Clustering-aware con validacion Hopkins

### Datasets Legacy (en archive/legacy_data/)

| Archivo | Motivo |
|---------|--------|
| picked_data_lyrics.csv | Hopkins ~0.45 (problematico) |
| picked_data_0.csv | Fuente diferente, obsoleto |

---

## Separadores por Nivel

| Nivel | Carpeta | Separador |
|-------|---------|-----------|
| 0 | 0_raw/ | , |
| 1 | 1_cleaned/ | ; |
| 2 | 2_with_lyrics/ | @@ |
| 3 | 3_selected/ | ^ |

---

## Notas

1. **PRODUCTION**: Usar siempre 3_selected/picked_data_optimal.csv
2. **Testing**: Usar 1_cleaned/tracks_features_500.csv
3. **Legacy**: Datasets obsoletos en archive/legacy_data/

---

**Ultima actualizacion**: Diciembre 2025
