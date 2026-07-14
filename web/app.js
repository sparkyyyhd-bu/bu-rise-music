// playlistgen web player. Vanilla JS, no framework, no build step.
// Click a row to play it; when a track ends, auto-advance to the next row.
// Rows whose stream URL fails to load are greyed out and skipped.

const form = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt");
const useLlm = document.getElementById("use-llm");
const generateBtn = document.getElementById("generate");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const interpretation = document.getElementById("interpretation");
const interpretationBody = document.getElementById("interpretation-body");
const player = document.getElementById("player");

let playlist = [];       // current playlist items, in sequence order
let currentIndex = -1;   // index into playlist of the playing track

function setStatus(text, isError) {
  statusEl.hidden = !text;
  statusEl.textContent = text || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function renderInterpretation(result) {
  const exp = result.expansion;
  const parts = [];
  parts.push(`<p><strong>source:</strong> ${exp.source}</p>`);
  parts.push("<ul>" + exp.captions.map((c) => `<li>${escapeHtml(c)}</li>`).join("") + "</ul>");
  for (const key of ["genres", "moods", "instruments"]) {
    if (exp[key] && exp[key].length) {
      parts.push(`<p><strong>${key}:</strong> ${exp[key].map(escapeHtml).join(", ")}</p>`);
    }
  }
  if (exp.bpm_range) parts.push(`<p><strong>bpm:</strong> ${exp.bpm_range[0]}–${exp.bpm_range[1]}</p>`);
  if (exp.energy) parts.push(`<p><strong>energy:</strong> ${exp.energy}</p>`);
  parts.push(`<p><strong>vocals:</strong> ${exp.vocals}</p>`);
  for (const note of result.notes || []) {
    parts.push(`<p class="note">${escapeHtml(note)}</p>`);
  }
  interpretationBody.innerHTML = parts.join("");
  interpretation.hidden = false;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function renderPlaylist() {
  resultsEl.innerHTML = "";
  playlist.forEach((item, i) => {
    const li = document.createElement("li");
    li.dataset.index = i;
    li.innerHTML =
      `<span class="title">${escapeHtml(item.artist)} – ${escapeHtml(item.title)}</span>` +
      `<span class="meta">sim ${item.similarity.toFixed(3)}` +
      (item.matched_tags.length ? ` · ${item.matched_tags.map(escapeHtml).join(", ")}` : "") +
      `</span>`;
    li.addEventListener("click", () => play(i));
    resultsEl.appendChild(li);
  });
}

function rowEl(i) {
  return resultsEl.querySelector(`li[data-index="${i}"]`);
}

function play(i) {
  if (i < 0 || i >= playlist.length) return;
  currentIndex = i;
  for (const li of resultsEl.children) li.classList.remove("playing");
  const row = rowEl(i);
  if (row) row.classList.add("playing");
  player.src = playlist[i].stream_url;
  player.play().catch(() => handleBrokenTrack(i));
}

function handleBrokenTrack(i) {
  const row = rowEl(i);
  if (row) {
    row.classList.add("broken");
    row.classList.remove("playing");
  }
  if (i === currentIndex) playNext();
}

function playNext() {
  for (let j = currentIndex + 1; j < playlist.length; j++) {
    if (!rowEl(j)?.classList.contains("broken")) {
      play(j);
      return;
    }
  }
}

player.addEventListener("ended", playNext);
player.addEventListener("error", () => {
  if (currentIndex >= 0) handleBrokenTrack(currentIndex);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  generateBtn.disabled = true;
  setStatus("generating…");
  interpretation.hidden = true;
  resultsEl.innerHTML = "";
  try {
    const resp = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptInput.value, n: 20, use_llm: useLlm.checked }),
    });
    const contentType = resp.headers.get("content-type") || "";
    const body = contentType.includes("application/json")
      ? await resp.json()
      : { detail: (await resp.text()).trim() || `HTTP ${resp.status}` };
    if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`);
    playlist = body.playlist;
    currentIndex = -1;
    renderPlaylist();
    renderInterpretation(body);
    setStatus(playlist.length ? "" : "no results");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    generateBtn.disabled = false;
  }
});
