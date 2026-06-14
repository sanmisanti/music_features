# Frontend — Recomendador Musical Híbrido (RESONANCE)

Interfaz web para explorar el sistema de recomendación de la Etapa 5. Permite
seleccionar una canción del catálogo, ejecutar el recomendador híbrido y
visualizar el vecindario recomendado con el desglose de las componentes
semántica (letras) y musical (audio).

## Arquitectura

```
app/
├── backend/
│   ├── main.py          # Servidor FastAPI (API REST + sirve el frontend)
│   └── recommender.py   # RecommenderService: carga el NPZ y resuelve consultas
├── frontend/
│   ├── index.html       # Estructura de la interfaz
│   ├── styles.css       # Estética (editorial oscuro)
│   └── app.js           # Lógica del cliente (búsqueda, slider α, render)
└── requirements.txt
```

El backend carga `data/5_unified/unified_dataset.npz` una sola vez al iniciar y
calcula `sigma` del kernel gaussiano con `src.recommendation.engine.compute_sigma`
(misma semilla que la Etapa 5). Cada consulta se resuelve por **fila única**:
producto punto en el espacio semántico de 384D y distancia euclídea + kernel
gaussiano en el espacio musical de 13D, fusionados con
`fused = α · semántica + (1 − α) · musical`. No se materializan matrices N×N.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/meta` | Tamaño de catálogo, α por defecto, dimensiones, géneros |
| GET | `/api/search?q=<texto>&limit=20` | Búsqueda por nombre o artista |
| GET | `/api/random?n=1` | Canciones aleatorias |
| GET | `/api/recommend?idx=<i>&alpha=0.80&k=10` | Recomendaciones para la canción `idx` |

## Cómo ejecutarlo

Desde la **raíz del repositorio** (`music_features/`):

```bash
# 1. Instalar dependencias (una sola vez)
pip install -r app/requirements.txt

# 2. Levantar el servidor
uvicorn app.backend.main:app --reload

# 3. Abrir en el navegador
#    http://127.0.0.1:8000
```

El arranque tarda unos segundos: carga el NPZ (~28 MB) y calcula `sigma`.

## Notas

- El reproductor embebido usa el `track_id` de Spotify de cada canción. Requiere
  conexión a internet; si una pista no está disponible en Spotify, el reproductor
  queda vacío pero los metadatos y las recomendaciones funcionan igual.
- El slider `α` permite explorar la fusión en vivo: 0 = sólo musical,
  1 = sólo semántica, 0.80 = óptimo hallado por grid search.
- Hacer clic en una recomendación despliega su reproductor de Spotify en línea
  (acordeón: uno a la vez) para escucharla sin alterar la semilla ni la lista.
- El botón `↻ similares` de cada fila la usa como nueva semilla, permitiendo
  navegar el grafo de similitudes.
