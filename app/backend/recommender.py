"""
Servicio de recomendacion para el frontend.

Carga el dataset unificado una sola vez y resuelve recomendaciones por
consulta individual (una fila), reutilizando exactamente la matematica del
motor de la Etapa 5 (`src.recommendation.engine`). A diferencia de
`precompute_similarities`, que materializa matrices N x N (~2.6 GB para
N=17.964), aqui se calcula la similitud de la cancion consultada contra el
resto del catalogo bajo demanda: una multiplicacion matriz-vector en el
espacio semantico y una distancia euclidea + kernel gaussiano en el espacio
musical. El resultado es identico al de `recommend_all` para esa fila.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

from src.config import RECOMMENDATION_TOP_K, UNIFIED_DATASET
from src.recommendation.engine import compute_sigma, gaussian_kernel

logger = logging.getLogger(__name__)

# Peso de fusion optimo hallado por grid search en la Etapa 5 (peso semantico).
DEFAULT_ALPHA = 0.80


class RecommenderService:
    """Encapsula la carga de datos y la generacion de recomendaciones."""

    def __init__(self, dataset_path=UNIFIED_DATASET):
        logger.info("Cargando dataset unificado desde %s", dataset_path)
        data = np.load(str(dataset_path), allow_pickle=True)

        # Embeddings semanticos ya estan L2-normalizados: coseno = producto punto.
        self.semantic: np.ndarray = np.ascontiguousarray(
            data["semantic_embeddings"], dtype=np.float32
        )
        self.musical: np.ndarray = np.ascontiguousarray(
            data["musical_features"], dtype=np.float32
        )
        self.track_ids: np.ndarray = data["track_ids"]
        self.genres: np.ndarray = data["genre_labels"]
        self.names: np.ndarray = data["track_names"]
        self.artists: np.ndarray = data["track_artists"]

        self.n: int = len(self.track_ids)

        # Cadenas en minuscula pre-computadas para busqueda rapida por substring.
        self._names_lower = np.array(
            [str(x).lower() for x in self.names], dtype=object
        )
        self._artists_lower = np.array(
            [str(x).lower() for x in self.artists], dtype=object
        )

        # Sigma del kernel gaussiano: misma funcion y seed que el motor original,
        # garantizando que las similitudes musicales sean identicas a la Etapa 5.
        logger.info("Calculando sigma del kernel gaussiano (muestra reproducible)...")
        self.sigma: float = compute_sigma(self.musical)

        logger.info(
            "Servicio listo: %d canciones, dim semantica=%d, dim musical=%d, sigma=%.4f",
            self.n, self.semantic.shape[1], self.musical.shape[1], self.sigma,
        )

    # ------------------------------------------------------------------ #
    # Metadatos
    # ------------------------------------------------------------------ #

    def song(self, idx: int) -> Dict:
        """Devuelve los metadatos de una cancion por indice."""
        return {
            "idx": int(idx),
            "track_id": str(self.track_ids[idx]),
            "name": str(self.names[idx]),
            "artist": str(self.artists[idx]),
            "genre": str(self.genres[idx]),
        }

    def genre_counts(self) -> Dict[str, int]:
        """Distribucion de generos en el catalogo."""
        values, counts = np.unique(self.genres.astype(str), return_counts=True)
        return {str(v): int(c) for v, c in zip(values, counts)}

    # ------------------------------------------------------------------ #
    # Busqueda
    # ------------------------------------------------------------------ #

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Busqueda case-insensitive por nombre o artista, priorizando
        coincidencias al inicio de la cadena sobre coincidencias internas.
        """
        q = query.strip().lower()
        if not q:
            return []

        prefix_hits: List[int] = []
        substr_hits: List[int] = []

        for i in range(self.n):
            name = self._names_lower[i]
            artist = self._artists_lower[i]
            if name.startswith(q) or artist.startswith(q):
                prefix_hits.append(i)
            elif q in name or q in artist:
                substr_hits.append(i)
            if len(prefix_hits) >= limit:
                break

        ordered = prefix_hits + substr_hits
        return [self.song(i) for i in ordered[:limit]]

    def random_songs(self, n: int = 1, seed: Optional[int] = None) -> List[Dict]:
        """Devuelve n canciones aleatorias del catalogo."""
        rng = np.random.default_rng(seed)
        idxs = rng.choice(self.n, size=min(n, self.n), replace=False)
        return [self.song(int(i)) for i in idxs]

    # ------------------------------------------------------------------ #
    # Recomendacion
    # ------------------------------------------------------------------ #

    def recommend(
        self,
        idx: int,
        alpha: float = DEFAULT_ALPHA,
        k: int = RECOMMENDATION_TOP_K,
    ) -> Dict:
        """
        Genera las top-k recomendaciones para la cancion `idx` aplicando
        fusion tardia: fused = alpha * coseno_semantico + (1 - alpha) * gauss_musical.

        Replica `recommend_all` del motor para una unica fila, sin construir
        matrices N x N.
        """
        if not 0 <= idx < self.n:
            raise IndexError(f"Indice fuera de rango: {idx}")
        alpha = float(np.clip(alpha, 0.0, 1.0))
        k = int(max(1, min(k, self.n - 1)))

        # Componente semantica: producto punto contra todo el catalogo (coseno).
        sim_semantic = self.semantic @ self.semantic[idx]  # [N]

        # Componente musical: distancia euclidea -> kernel gaussiano.
        dist_musical = cdist(
            self.musical[idx : idx + 1], self.musical, metric="euclidean"
        )[0]  # [N]
        sim_musical = gaussian_kernel(dist_musical, self.sigma)  # [N]

        # Fusion tardia.
        fused = alpha * sim_semantic + (1.0 - alpha) * sim_musical
        fused[idx] = -np.inf  # excluir self-recommendation

        # Top-k eficiente (O(N)) y orden descendente sobre los candidatos.
        top = np.argpartition(fused, -k)[-k:]
        top = top[np.argsort(-fused[top])]

        recommendations = []
        for rank, j in enumerate(top, start=1):
            recommendations.append({
                "rank": rank,
                "idx": int(j),
                "track_id": str(self.track_ids[j]),
                "name": str(self.names[j]),
                "artist": str(self.artists[j]),
                "genre": str(self.genres[j]),
                "score": float(fused[j]),
                "score_semantic": float(sim_semantic[j]),
                "score_musical": float(sim_musical[j]),
                "same_genre": bool(self.genres[j] == self.genres[idx]),
            })

        return {
            "query": self.song(idx),
            "alpha": alpha,
            "k": k,
            "recommendations": recommendations,
        }
