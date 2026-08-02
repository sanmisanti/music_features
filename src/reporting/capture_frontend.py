"""
Captura de la interfaz web del recomendador para el anexo del informe.

Conduce un navegador Chrome mediante Playwright sobre la aplicacion servida
por `app.backend.main` y registra los estados de la interfaz que documentan
el sistema en funcionamiento: pantalla inicial, busqueda incremental, tablero
de recomendacion, ablacion del peso de fusion alpha y variacion de k.

Requisitos previos:
    1. `pip install playwright`
    2. Servidor levantado desde la raiz del repositorio:
       `python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000`

Ejecucion (desde la raiz del repositorio):
    python -m src.reporting.capture_frontend

Salidas:
    - PNG en `thesis/figures/app/`
    - Registro de trazabilidad en `results/metrics/etapa6_frontend_captures.json`
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = PROJECT_ROOT / "thesis" / "figures" / "app"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "etapa6_frontend_captures.json"

# Ventana de captura. El factor de escala 2 produce imagenes nitidas al
# insertarlas en el PDF, donde se reducen a un ancho de pagina.
VIEWPORT = {"width": 1440, "height": 900}
SCALE = 2

# Esperas (ms). Los reproductores de Spotify son iframes externos: sin margen
# suficiente la captura los muestra vacios.
WAIT_RENDER = 900
WAIT_EMBED = 5000

# Semillas de las capturas. Se ubican por busqueda textual, replicando lo que
# haria una persona usando la interfaz.
SEED_VUELVE = "Vuelve"            # idx 8323, Ricky Martin, latin
SEED_LOSE_YOURSELF = "Lose Yourself"  # idx 998, Eminem, rap


# --------------------------------------------------------------------------- #
# Utilidades de interaccion
# --------------------------------------------------------------------------- #

def _settle(page: Page, embeds: bool = False) -> None:
    """Espera a que el DOM se estabilice antes de capturar."""
    page.wait_for_timeout(WAIT_EMBED if embeds else WAIT_RENDER)


def _wait_recommendations(page: Page) -> None:
    """Bloquea hasta que la lista deja de mostrar el estado de calculo."""
    page.wait_for_function(
        "() => { const l = document.getElementById('recsList');"
        " return l && l.querySelectorAll('.rec-item').length > 0; }",
        timeout=20000,
    )
    # Las filas entran con animacion escalonada (45 ms por posicion).
    page.wait_for_timeout(WAIT_RENDER)


def _type_search(page: Page, query: str) -> None:
    """Escribe en el buscador reproduciendo la escritura del usuario."""
    page.click("#searchInput")
    page.fill("#searchInput", "")
    page.type("#searchInput", query, delay=45)
    page.wait_for_selector(".autocomplete.open .ac-item", timeout=10000)
    page.wait_for_timeout(300)


def _select_first_result(page: Page) -> None:
    """Selecciona la primera coincidencia del desplegable."""
    page.click('.ac-item[data-i="0"]')
    _wait_recommendations(page)


def _set_alpha(page: Page, alpha: float) -> None:
    """
    Desplaza el control de fusion. El elemento es un `input[type=range]`: se
    asigna el valor y se emiten los eventos `input` (actualiza la etiqueta) y
    `change` (dispara la consulta al backend), tal como haria el arrastre.
    """
    page.evaluate(
        """(a) => {
            const s = document.getElementById('alphaSlider');
            s.value = String(a);
            s.dispatchEvent(new Event('input',  { bubbles: true }));
            s.dispatchEvent(new Event('change', { bubbles: true }));
        }""",
        alpha,
    )
    _wait_recommendations(page)


def _set_k(page: Page, k: int) -> None:
    """Pulsa el boton de longitud de lista."""
    page.click(f'#kButtons button[data-k="{k}"]')
    _wait_recommendations(page)


def _query_state(page: Page) -> Dict[str, str]:
    """Lee la semilla activa y los controles, para el registro de trazabilidad."""
    return page.evaluate(
        """() => ({
            name:   document.getElementById('queryName').textContent,
            artist: document.getElementById('queryArtist').textContent,
            genre:  document.getElementById('queryGenre').textContent,
            alpha:  document.getElementById('alphaValue').textContent,
            k:      document.querySelector('#kButtons button.active').dataset.k,
            recommendations: Array.from(
                document.querySelectorAll('#recsList .rec-item')
            ).map(li => ({
                rank:   li.querySelector('.rec-rank').textContent,
                name:   li.querySelector('.rec-name').textContent,
                artist: li.querySelector('.rec-artist').textContent,
                genre:  li.querySelector('.rec-genre').textContent,
                score:  li.querySelector('.rec-score').textContent,
            })),
        })"""
    )


def _shot(page: Page, name: str, selector: Optional[str] = None,
          full_page: bool = False) -> Path:
    """
    Guarda una captura. Con `selector` recorta un elemento (util para aislar
    el tablero de recomendacion); con `full_page` extiende la captura mas alla
    del alto de la ventana.
    """
    path = OUTPUT_DIR / f"{name}.png"
    if selector is not None:
        page.locator(selector).screenshot(path=str(path))
    else:
        page.screenshot(path=str(path), full_page=full_page)
    print(f"  [ok] {path.relative_to(PROJECT_ROOT)}")
    return path


# --------------------------------------------------------------------------- #
# Guion de capturas
# --------------------------------------------------------------------------- #

def run() -> List[Dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry: List[Dict] = []

    def record(figure: str, description: str, page: Page,
               with_state: bool = True) -> None:
        entry: Dict = {"figure": figure, "description": description}
        if with_state:
            entry["state"] = _query_state(page)
        registry.append(entry)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=SCALE,
            locale="es-AR",
        )
        page = context.new_page()

        print("Cargando la aplicacion...")
        page.goto(BASE_URL, wait_until="networkidle")
        page.wait_for_function("() => document.fonts.status === 'loaded'", timeout=15000)
        page.wait_for_function(
            "() => document.getElementById('catalogSize').textContent !== '—'",
            timeout=20000,
        )
        _settle(page)

        # --- 1. Pantalla inicial ------------------------------------------- #
        print("[1/8] Pantalla inicial")
        _shot(page, "app_01_estado_inicial")
        registry.append({
            "figure": "app_01_estado_inicial",
            "description": "Pantalla inicial con los metadatos del catalogo.",
        })

        # --- 2. Busqueda incremental --------------------------------------- #
        print("[2/8] Busqueda incremental")
        _type_search(page, SEED_VUELVE)
        _shot(page, "app_02_busqueda")
        registry.append({
            "figure": "app_02_busqueda",
            "description": f"Autocompletado para la consulta '{SEED_VUELVE}'.",
            "state": {"query_text": SEED_VUELVE,
                      "results": page.evaluate(
                          """() => Array.from(document.querySelectorAll('.ac-item'))
                                .map(e => ({
                                    name:   e.querySelector('.ac-name').textContent,
                                    artist: e.querySelector('.ac-artist').textContent,
                                    genre:  e.querySelector('.ac-genre').textContent,
                                }))""")},
        })

        # --- 3. Tablero completo (alpha optimo) ---------------------------- #
        print("[3/8] Tablero completo, alpha = 0,80")
        _select_first_result(page)
        _settle(page, embeds=True)
        _shot(page, "app_03_tablero_alpha080", full_page=True)
        record("app_03_tablero_alpha080",
               "Tablero de recomendacion para la semilla latina al alpha optimo.", page)

        # --- 4-5. Ablacion del peso de fusion ------------------------------ #
        print("[4/8] Ablacion alpha = 0,00 (solo musical)")
        _set_alpha(page, 0.0)
        _shot(page, "app_04_ablacion_alpha000", selector="#board")
        record("app_04_ablacion_alpha000",
               "Misma semilla con alpha = 0,00: recomendacion puramente musical.", page)

        print("[5/8] Ablacion alpha = 1,00 (solo semantica)")
        _set_alpha(page, 1.0)
        _shot(page, "app_05_ablacion_alpha100", selector="#board")
        record("app_05_ablacion_alpha100",
               "Misma semilla con alpha = 1,00: recomendacion puramente semantica.", page)

        # Restaurar el optimo antes de cambiar de semilla.
        _set_alpha(page, 0.80)

        # --- 6-7. Variacion de k ------------------------------------------- #
        print("[6/8] Longitud de lista k = 5")
        _type_search(page, SEED_LOSE_YOURSELF)
        _select_first_result(page)
        _settle(page, embeds=True)
        _set_k(page, 5)
        _shot(page, "app_06_k5", selector="#board")
        record("app_06_k5", "Semilla de rap con lista corta (k = 5).", page)

        print("[7/8] Longitud de lista k = 20")
        _set_k(page, 20)
        _shot(page, "app_07_k20", selector="#board")
        record("app_07_k20", "Misma semilla con lista extendida (k = 20).", page)

        # Restaurar la longitud por defecto.
        _set_k(page, 10)

        # --- 8. Semilla aleatoria ------------------------------------------ #
        print("[8/8] Semilla aleatoria")
        page.click("#randomBtn")
        _wait_recommendations(page)
        _settle(page, embeds=True)
        _shot(page, "app_08_aleatoria", full_page=True)
        record("app_08_aleatoria",
               "Semilla obtenida con el boton de seleccion aleatoria.", page)

        context.close()
        browser.close()

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": BASE_URL,
        "viewport": VIEWPORT,
        "device_scale_factor": SCALE,
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "captures": registry,
    }
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRegistro: {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    return registry


if __name__ == "__main__":
    run()
