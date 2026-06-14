/* =========================================================================
   RESONANCE — lógica del cliente
   ========================================================================= */

const els = {
  search:      document.getElementById("searchInput"),
  results:     document.getElementById("results"),
  random:      document.getElementById("randomBtn"),
  empty:       document.getElementById("empty"),
  board:       document.getElementById("board"),
  queryName:   document.getElementById("queryName"),
  queryArtist: document.getElementById("queryArtist"),
  queryGenre:  document.getElementById("queryGenre"),
  embed:       document.getElementById("spotifyEmbed"),
  alphaSlider: document.getElementById("alphaSlider"),
  alphaValue:  document.getElementById("alphaValue"),
  kButtons:    document.getElementById("kButtons"),
  recsList:    document.getElementById("recsList"),
  catalogSize: document.getElementById("catalogSize"),
  dimInfo:     document.getElementById("dimInfo"),
};

const state = {
  queryIdx: null,
  alpha: 0.80,
  k: 10,
  acItems: [],
  acActive: -1,
};

// ----------------------------------------------------------------- utils
const debounce = (fn, ms) => {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
};

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function spotifyUrl(trackId) {
  return `https://open.spotify.com/track/${trackId}`;
}

// ----------------------------------------------------------------- meta
async function loadMeta() {
  try {
    const m = await api("/api/meta");
    els.catalogSize.textContent = m.catalog_size.toLocaleString("es-AR");
    els.dimInfo.textContent = `${m.semantic_dim} · ${m.musical_dim}`;
  } catch (e) {
    els.catalogSize.textContent = "—";
  }
}

// ----------------------------------------------------------------- búsqueda
const runSearch = debounce(async (q) => {
  if (!q.trim()) { closeAutocomplete(); return; }
  try {
    const { results } = await api(`/api/search?q=${encodeURIComponent(q)}&limit=20`);
    renderAutocomplete(results);
  } catch (e) { /* silencioso */ }
}, 160);

function renderAutocomplete(items) {
  state.acItems = items;
  state.acActive = -1;
  if (!items.length) {
    els.results.innerHTML = `<div class="loading">Sin coincidencias.</div>`;
    els.results.classList.add("open");
    return;
  }
  els.results.innerHTML = items.map((s, i) => `
    <div class="ac-item" data-i="${i}" role="option">
      <span class="ac-name">${escapeHtml(s.name)}</span>
      <span class="ac-artist">${escapeHtml(s.artist)}</span>
      <span class="ac-genre">${escapeHtml(s.genre)}</span>
    </div>
  `).join("");
  els.results.classList.add("open");

  els.results.querySelectorAll(".ac-item").forEach((el) => {
    el.addEventListener("click", () => selectSong(items[+el.dataset.i]));
  });
}

function closeAutocomplete() {
  els.results.classList.remove("open");
  els.results.innerHTML = "";
  state.acItems = [];
  state.acActive = -1;
}

// navegación con teclado
els.search.addEventListener("keydown", (e) => {
  const n = state.acItems.length;
  if (!n) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    state.acActive = (state.acActive + 1) % n;
    highlightActive();
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    state.acActive = (state.acActive - 1 + n) % n;
    highlightActive();
  } else if (e.key === "Enter") {
    const pick = state.acActive >= 0 ? state.acActive : 0;
    selectSong(state.acItems[pick]);
  } else if (e.key === "Escape") {
    closeAutocomplete();
  }
});

function highlightActive() {
  els.results.querySelectorAll(".ac-item").forEach((el, i) => {
    el.classList.toggle("active", i === state.acActive);
    if (i === state.acActive) el.scrollIntoView({ block: "nearest" });
  });
}

els.search.addEventListener("input", (e) => runSearch(e.target.value));
els.search.addEventListener("focus", (e) => {
  if (state.acItems.length) els.results.classList.add("open");
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-wrap")) closeAutocomplete();
});

// canción aleatoria
els.random.addEventListener("click", async () => {
  try {
    const { results } = await api("/api/random?n=1");
    if (results.length) selectSong(results[0]);
  } catch (e) { /* noop */ }
});

// ----------------------------------------------------------------- selección
function selectSong(song) {
  if (!song) return;
  state.queryIdx = song.idx;
  closeAutocomplete();
  els.search.value = "";
  els.search.blur();

  els.empty.hidden = true;
  els.board.hidden = false;

  els.queryName.textContent = song.name;
  els.queryArtist.textContent = song.artist;
  els.queryGenre.textContent = song.genre;
  els.queryGenre.style.cssText = genreChipStyle(song.genre);

  els.embed.src =
    `https://open.spotify.com/embed/track/${song.track_id}?utm_source=generator&theme=0`;

  fetchRecommendations();
  els.board.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ----------------------------------------------------------------- controles
els.alphaSlider.addEventListener("input", (e) => {
  state.alpha = parseFloat(e.target.value);
  els.alphaValue.textContent = state.alpha.toFixed(2);
});
els.alphaSlider.addEventListener("change", () => fetchRecommendations());

els.kButtons.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-k]");
  if (!btn) return;
  state.k = parseInt(btn.dataset.k, 10);
  els.kButtons.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  fetchRecommendations();
});

// ----------------------------------------------------------------- recomendación
async function fetchRecommendations() {
  if (state.queryIdx === null) return;
  els.recsList.innerHTML = `<li class="loading">Calculando vecindario…</li>`;
  try {
    const data = await api(
      `/api/recommend?idx=${state.queryIdx}&alpha=${state.alpha}&k=${state.k}`
    );
    renderRecs(data.recommendations);
  } catch (e) {
    els.recsList.innerHTML = `<li class="loading">Error al recomendar.</li>`;
  }
}

function renderRecs(recs) {
  if (!recs.length) {
    els.recsList.innerHTML = `<li class="loading">Sin resultados.</li>`;
    return;
  }
  const maxScore = Math.max(...recs.map((r) => r.score)) || 1;
  const alpha = state.alpha;

  els.recsList.innerHTML = recs.map((r, i) => {
    // contribuciones a la fusión: alpha*sem y (1-alpha)*mus
    const semC = alpha * r.score_semantic;
    const musC = (1 - alpha) * r.score_musical;
    const total = semC + musC || 1;
    const barW = Math.max(6, (r.score / maxScore) * 100);   // ancho relativo al top
    const semPct = (semC / total) * 100;
    const musPct = 100 - semPct;

    return `
      <li class="rec-item" data-idx="${r.idx}" data-track="${escapeHtml(r.track_id)}"
          style="animation-delay:${i * 45}ms">
        <div class="rec" title="Reproducir">
          <div class="rec-rank">${r.rank}</div>
          <div class="rec-main">
            <div class="rec-title-row">
              <span class="rec-name">${escapeHtml(r.name)}</span>
              ${r.same_genre ? `<span class="rec-same" title="mismo género que la semilla"></span>` : ""}
            </div>
            <div class="rec-artist">${escapeHtml(r.artist)}</div>
            <div class="rec-bar" style="width:${barW}%"
                 title="semántica ${semC.toFixed(3)} · musical ${musC.toFixed(3)}">
              <span class="seg-sem" style="width:${semPct}%"></span>
              <span class="seg-mus" style="width:${musPct}%"></span>
            </div>
          </div>
          <div class="rec-meta">
            <span class="rec-score">${r.score.toFixed(3)}</span>
            <span class="rec-genre">${escapeHtml(r.genre)}</span>
            <button class="rec-reseed" type="button"
                    title="Recomendar canciones similares a esta">↻ similares</button>
          </div>
        </div>
        <div class="rec-player"></div>
      </li>`;
  }).join("");

  els.recsList.querySelectorAll(".rec-item").forEach((li) => {
    const idx = +li.dataset.idx;
    const trackId = li.dataset.track;
    const row = li.querySelector(".rec");

    // precargar el reproductor al pasar el mouse: para cuando se hace clic,
    // el iframe ya está cargado y el despliegue se siente instantáneo.
    row.addEventListener("mouseenter", () => ensurePlayer(li, trackId));
    // foco por teclado también precarga
    row.addEventListener("focusin", () => ensurePlayer(li, trackId));

    // clic en la fila = reproducir (acordeón), sin cambiar la semilla
    row.addEventListener("click", () => togglePlayer(li, trackId));

    // botón dedicado = re-sembrar
    li.querySelector(".rec-reseed").addEventListener("click", (e) => {
      e.stopPropagation();
      reseedFrom(idx);
    });
  });
}

// construye el iframe una sola vez (precarga). El esqueleto aparece al instante;
// el iframe se revela con fundido cuando termina de cargar.
function ensurePlayer(li, trackId) {
  const player = li.querySelector(".rec-player");
  if (player.dataset.built) return;
  player.dataset.built = "1";
  player.innerHTML =
    `<div class="player-skeleton"></div>` +
    `<iframe title="Reproductor de Spotify" allow="encrypted-media" ` +
    `src="https://open.spotify.com/embed/track/${trackId}?utm_source=generator&theme=0"></iframe>`;
  const iframe = player.querySelector("iframe");
  iframe.addEventListener("load", () => player.classList.add("ready"));
}

// reproductor en línea (acordeón: solo uno abierto a la vez)
function togglePlayer(li, trackId) {
  const isOpen = li.classList.contains("playing");

  // cerrar cualquier otro (se conserva el iframe ya cargado para reaperturas instantáneas)
  els.recsList.querySelectorAll(".rec-item.playing").forEach((el) => {
    if (el !== li) el.classList.remove("playing");
  });

  if (isOpen) {
    li.classList.remove("playing");
  } else {
    ensurePlayer(li, trackId);
    li.classList.add("playing");
  }
}

// usar una recomendación como nueva semilla
async function reseedFrom(idx) {
  try {
    // la recomendación ya trae los metadatos; los recuperamos desde el DOM
    // pidiendo una nueva recomendación con esa semilla directamente.
    state.queryIdx = idx;
    const data = await api(`/api/recommend?idx=${idx}&alpha=${state.alpha}&k=${state.k}`);
    const q = data.query;
    els.queryName.textContent = q.name;
    els.queryArtist.textContent = q.artist;
    els.queryGenre.textContent = q.genre;
    els.queryGenre.style.cssText = genreChipStyle(q.genre);
    els.embed.src =
      `https://open.spotify.com/embed/track/${q.track_id}?utm_source=generator&theme=0`;
    renderRecs(data.recommendations);
    els.board.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { /* noop */ }
}

// ----------------------------------------------------------------- helpers
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// paleta por género para los chips
const GENRE_HUES = {
  edm: 280, latin: 18, pop: 330, "r&b": 200, rap: 50, rock: 0,
};
function genreChipStyle(genre) {
  const h = GENRE_HUES[String(genre).toLowerCase()] ?? 40;
  return `display:inline-block;font-family:var(--font-mono);font-size:0.66rem;` +
    `letter-spacing:0.12em;text-transform:uppercase;padding:0.25rem 0.6rem;` +
    `border-radius:6px;color:hsl(${h} 70% 72%);background:hsl(${h} 60% 50% / 0.12);` +
    `border:1px solid hsl(${h} 60% 50% / 0.35);`;
}

// init
loadMeta();
