# legacy_lyrics_extractor (ARCHIVADO)

## Estado: OBSOLETO

Modulo de extraccion de letras via Genius API. Archivado tras adopcion del dataset Kaggle con letras pre-incluidas.

## Razon de Archivado

El dataset `data/2_with_lyrics/spotify_songs_fixed.csv` (18,454 canciones) incluye letras de Genius.com con 100% cobertura, eliminando la necesidad de extraccion dinamica.

## Contenido del Modulo

| Archivo | Funcion Original |
|---------|------------------|
| genius_lyrics_extractor.py | Extraccion via Genius API |
| lyrics_database.py | Almacenamiento SQLite |
| lyrics_availability_checker.py | Verificacion de disponibilidad |
| hybrid_selection_criteria.py | Criterios de seleccion |

## Referencias Historicas

- README.md - Documentacion original del modulo
- IMPLEMENTACION_CON_LETRAS.md - Plan de implementacion historico
- data/lyrics.db - Base SQLite con extracciones parciales

## Fecha de Archivado

Diciembre 2025
