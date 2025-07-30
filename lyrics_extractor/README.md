# Genius Lyrics Extractor

Este módulo extrae letras de canciones desde Genius.com para el dataset final de 9,677 canciones representativas (`picked_data_0.csv`) del sistema de análisis de características musicales multimodal.

## Configuración

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Obtener token de Genius API

1. Ve a [https://genius.com/api-clients](https://genius.com/api-clients)
2. Crea una nueva aplicación
3. Copia el "Client Access Token"
4. Configura la variable de entorno:

```bash
export GENIUS_ACCESS_TOKEN="tu_token_aqui"
```

## Uso

```bash
cd lyrics_extractor
python genius_lyrics_extractor.py
```

## Características del Script

### ✅ Rate Limiting y Mejores Prácticas
- **Delay entre peticiones**: 1.5 segundos entre canciones
- **Timeout**: 15 segundos por petición
- **Reintentos**: 3 intentos automáticos
- **Manejo de rate limits**: Pausa automática de 60 segundos cuando se alcanza el límite

### ✅ Estrategias de Búsqueda Robustas
- **Múltiples estrategias**: 4 enfoques diferentes de búsqueda por canción
- **Normalización**: Limpia términos como "(feat.)", "(remix)", etc.
- **Verificación de coincidencias**: Confirma que la canción encontrada coincide con la buscada
- **Cache de fallos**: Evita reintentar canciones que ya fallaron

### ✅ Manejo de Errores Comprehensivo
- **Logging detallado**: Archivo de log con todos los eventos
- **Manejo de excepciones**: Captura y maneja diferentes tipos de errores
- **Guardado incremental**: Guarda resultados por lotes para evitar pérdida de datos
- **Interrupción segura**: Permite interrumpir con Ctrl+C sin perder progreso

### ✅ Monitoreo y Estadísticas
- **Progreso en tiempo real**: Muestra avance cada 10 canciones
- **Estadísticas completas**: Contadores de éxito, errores, etc.
- **Tasa de éxito**: Porcentaje de letras extraídas exitosamente

## Almacenamiento de Datos

### Decisión de Arquitectura: SQLite Database

Para optimizar el almacenamiento y acceso de ~10,000 letras de canciones, se utiliza **SQLite** como solución principal:

**Ventajas:**
- ✅ Archivo único comprimido (~50-100MB vs 500MB+ en CSV)
- ✅ Índices automáticos para búsquedas rápidas por `spotify_id`
- ✅ Consultas SQL para análisis semántico avanzado
- ✅ Transacciones seguras durante extracción por lotes
- ✅ Sin dependencias adicionales (SQLite incluido en Python)

### Archivos de Salida

- `../data/lyrics.db`: Base de datos SQLite principal con letras extraídas
- `output/temp_lyrics_batch_*.csv`: Archivos de respaldo temporales por lotes
- `logs/lyrics_extraction.log`: Log detallado del proceso

### Estructura de la Base de Datos

```sql
CREATE TABLE lyrics (
    spotify_id TEXT PRIMARY KEY,
    song_name TEXT,
    artist_name TEXT,
    lyrics TEXT,
    genius_id INTEGER,
    genius_title TEXT,
    genius_artist TEXT,
    genius_url TEXT,
    word_count INTEGER,
    language TEXT,
    extraction_status TEXT,
    extraction_date TIMESTAMP
);
```

## Optimizaciones Implementadas

1. **Limpieza de nombres de artistas**: Maneja el formato `['Artist Name']` del dataset
2. **Normalización de términos de búsqueda**: Elimina patrones que dificultan la búsqueda
3. **Verificación de similitud**: Algoritmo de similitud para confirmar coincidencias
4. **Procesamiento por lotes**: Divide el trabajo en lotes manejables
5. **Exclusión de términos problemáticos**: Evita remixes, versiones live, etc.

## Estadísticas Esperadas

Basado en las mejores prácticas implementadas para el dataset completo de 9,677 canciones:
- **Tasa de éxito**: 60-80% de las canciones del dataset (~6,000-7,700 letras extraídas)
- **Tiempo estimado**: ~4-5 horas para 9,677 canciones (con rate limiting)
- **Procesamiento**: 100 canciones por lote para optimizar rendimiento
- **Manejo de errores**: <5% de errores técnicos
- **Tamaño de DB**: ~50-100MB comprimido en SQLite

## Troubleshooting

### Error: "Genius API token required"
- Configura la variable de entorno `GENIUS_ACCESS_TOKEN`

### Error: "lyricsgenius library not found"
```bash
pip install lyricsgenius
```

### Rate limiting excesivo
- El script maneja automáticamente los rate limits
- Si necesitas ajustar, modifica `sleep_time` en la configuración del cliente

### Resultados parciales
- Los archivos `temp_lyrics_batch_*.csv` contienen resultados intermedios como respaldo
- La base de datos SQLite se actualiza automáticamente por lotes
- Para consultar datos: `sqlite3 ../data/lyrics.db "SELECT COUNT(*) FROM lyrics;"`

## Uso del Sistema SQLite

### Consultas Básicas
```python
import sqlite3
import pandas as pd

# Conectar a la base de datos
conn = sqlite3.connect('data/lyrics.db')

# Cargar letras de canciones específicas
df = pd.read_sql("SELECT * FROM lyrics WHERE extraction_status='success'", conn)

# Estadísticas rápidas
stats = pd.read_sql("SELECT extraction_status, COUNT(*) as count FROM lyrics GROUP BY extraction_status", conn)
```

### Integración con Datos Musicales
```python
# Cargar datos musicales
music_df = pd.read_csv('data/picked_data_0.csv', sep=';', decimal=',')

# Cargar letras desde SQLite
lyrics_df = pd.read_sql("SELECT spotify_id, lyrics, word_count FROM lyrics WHERE extraction_status='success'", conn)

# Fusionar datasets
complete_df = music_df.merge(lyrics_df, left_on='id', right_on='spotify_id', how='left')
```