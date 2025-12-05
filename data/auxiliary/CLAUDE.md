# auxiliary - Cache y Datos Auxiliares

## Archivos

| Archivo | Tamano | Descripcion |
|---------|--------|-------------|
| lyrics.db | 148 KB | Base de datos SQLite con letras de Genius |
| lyrics_availability_cache.json | 449 KB | Cache de disponibilidad de letras |

## lyrics.db

Base de datos SQLite que almacena letras extraidas de Genius.com.

### Estructura

```sql
CREATE TABLE lyrics (
    track_id TEXT PRIMARY KEY,
    artist TEXT,
    title TEXT,
    lyrics TEXT,
    genius_id INTEGER,
    extraction_date TEXT
);
```

### Uso

```python
import sqlite3

conn = sqlite3.connect('data/auxiliary/lyrics.db')
cursor = conn.cursor()

# Buscar letra por track_id
cursor.execute("SELECT lyrics FROM lyrics WHERE track_id = ?", (track_id,))
result = cursor.fetchone()
```

## lyrics_availability_cache.json

Cache JSON que almacena resultados de busquedas en Genius API para evitar consultas repetidas.

### Estructura

```json
{
  "track_title|artist_name": {
    "has_lyrics": true,
    "confidence": 0.95,
    "found_title": "Titulo encontrado",
    "found_artist": "Artista encontrado",
    "genius_id": 12345,
    "verification_date": "2025-08-19 14:30:00",
    "error_message": null
  }
}
```

### Uso

```python
import json

with open('data/auxiliary/lyrics_availability_cache.json', 'r') as f:
    cache = json.load(f)

# Verificar si una cancion esta en cache
key = f"{track_title}|{artist_name}"
if key in cache:
    has_lyrics = cache[key]['has_lyrics']
```

## Proposito

Estos archivos son **cache de sistema interno** utilizados durante el proceso de extraccion de letras. No requieren modificacion directa.

## Origen

Generados por el modulo `lyrics_extractor/` durante la fase de enriquecimiento del dataset con letras de Genius.com.
