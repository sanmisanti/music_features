"""
Servidor FastAPI para el frontend del sistema de recomendacion musical.

Expone el recomendador hibrido (Etapa 5) como API REST y sirve la interfaz
estatica. El dataset se carga una unica vez al iniciar el proceso.

Ejecucion (desde la raiz del repositorio):
    uvicorn app.backend.main:app --reload

Luego abrir http://127.0.0.1:8000 en el navegador.
"""

import sys
from pathlib import Path
from typing import Optional

# Permitir importar el paquete `src` cuando se ejecuta uvicorn desde la raiz.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.backend.recommender import DEFAULT_ALPHA, RecommenderService  # noqa: E402

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(
    title="Recomendador Musical Hibrido",
    description="API del sistema de recomendacion semantico-musical (Etapa 5).",
    version="1.0.0",
)

# Servicio global, inicializado en el evento de arranque.
service: Optional[RecommenderService] = None


@app.on_event("startup")
def _load_service() -> None:
    global service
    service = RecommenderService()


def _require_service() -> RecommenderService:
    if service is None:  # pragma: no cover - solo si se invoca antes del startup
        raise HTTPException(status_code=503, detail="Servicio no inicializado.")
    return service


# ---------------------------------------------------------------------- #
# API
# ---------------------------------------------------------------------- #

@app.get("/api/meta")
def meta() -> dict:
    """Metadatos del catalogo: tamano, alpha por defecto, distribucion de generos."""
    svc = _require_service()
    return {
        "catalog_size": svc.n,
        "default_alpha": DEFAULT_ALPHA,
        "semantic_dim": int(svc.semantic.shape[1]),
        "musical_dim": int(svc.musical.shape[1]),
        "genres": svc.genre_counts(),
    }


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, description="Texto a buscar (nombre o artista)."),
    limit: int = Query(20, ge=1, le=50),
) -> dict:
    """Busca canciones por nombre o artista."""
    svc = _require_service()
    return {"results": svc.search(q, limit=limit)}


@app.get("/api/random")
def random_song(n: int = Query(1, ge=1, le=20)) -> dict:
    """Devuelve canciones aleatorias del catalogo."""
    svc = _require_service()
    return {"results": svc.random_songs(n)}


@app.get("/api/recommend")
def recommend(
    idx: int = Query(..., ge=0, description="Indice de la cancion consultada."),
    alpha: float = Query(DEFAULT_ALPHA, ge=0.0, le=1.0),
    k: int = Query(10, ge=1, le=50),
) -> dict:
    """Genera recomendaciones para la cancion indicada."""
    svc = _require_service()
    try:
        return svc.recommend(idx, alpha=alpha, k=k)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------- #
# Frontend estatico
# ---------------------------------------------------------------------- #

@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# Recursos estaticos (styles.css, app.js, fuentes, etc.).
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
