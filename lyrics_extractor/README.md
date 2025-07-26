# Genius Lyrics Extractor

Este módulo extrae letras de canciones desde Genius.com para el dataset `tracks_features_500.csv` del sistema de análisis de características musicales.

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

## Archivos de Salida

- `lyrics_extraction_results.csv`: Resultados finales con letras extraídas
- `temp_lyrics_batch_*.csv`: Archivos temporales por lotes
- `logs/lyrics_extraction.log`: Log detallado del proceso

## Estructura de Datos de Salida

```csv
spotify_id,song_name,artist_name,genius_id,genius_title,genius_artist,lyrics,genius_url,extraction_status
```

## Optimizaciones Implementadas

1. **Limpieza de nombres de artistas**: Maneja el formato `['Artist Name']` del dataset
2. **Normalización de términos de búsqueda**: Elimina patrones que dificultan la búsqueda
3. **Verificación de similitud**: Algoritmo de similitud para confirmar coincidencias
4. **Procesamiento por lotes**: Divide el trabajo en lotes manejables
5. **Exclusión de términos problemáticos**: Evita remixes, versiones live, etc.

## Estadísticas Esperadas

Basado en las mejores prácticas implementadas, se espera:
- **Tasa de éxito**: 60-80% de las canciones del dataset
- **Tiempo estimado**: ~15-20 minutos para 500 canciones
- **Manejo de errores**: <5% de errores técnicos

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
- Los archivos `temp_lyrics_batch_*.csv` contienen resultados intermedios
- Pueden combinarse manualmente si es necesario