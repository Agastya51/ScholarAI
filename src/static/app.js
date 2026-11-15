/* Shared behaviour for every ScholarAI page. */

// ---------- Session state (per browser tab) ----------
const SS = {
  get: k => { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } },
  set: (k, v) => sessionStorage.setItem(k, JSON.stringify(v)),
  del: k => sessionStorage.removeItem(k),
  // Values written by SS.set are JSON-encoded; decode them, but accept raw strings too
  str: k => {
    const raw = sessionStorage.getItem(k);
    if (!raw) return "";
    try { const v = JSON.parse(raw); return typeof v === "string" ? v : raw; } catch { return raw; }
  }
};
const PAPER_KEYS = ["paper_id", "title", "stats", "uploadTrace", "summary", "vision", "figures", "topic", "comparison", "papers",
  "compareRun", "improvements", "edits", "improveRun", "qaHistory", "review", "analysis", "insights", "practice",
  "starters", "explain"];

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const $ = id => document.getElementById(id);
const hasPaper = () => !!SS.str("paper_id");
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// ---------- Markdown: rendered, sanitised, and with page citations made clickable ----------
function md(text) {
  const raw = String(text || "");
  let html;
  if (window.marked) {
    html = marked.parse(raw, { breaks: true, gfm: true });
  } else {
    html = "<p>" + esc(raw).replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br>") + "</p>";
  }
  if (window.DOMPurify) html = DOMPurify.sanitize(html, { FORBID_TAGS: ["style", "form", "input", "button"] });
  return linkCitations(html);
}

// "[p. 4]", "[p. 4 · 3 Method]" and "(p. 4)" become buttons that open that page of the paper
function linkCitations(html) {
  return html
    .replace(/\[p\.\s*(\d{1,3})((?:\s*[·,]\s*[^\]<]{0,80})?)\]/g,
      (m, page, rest) => `<button type="button" class="cite" data-page="${page}" title="Open page ${page}${esc(rest)}">p. ${page}</button>`)
    .replace(/\(p\.\s*(\d{1,3})\)/g,
      (m, page) => `<button type="button" class="cite" data-page="${page}" title="Open page ${page}">p. ${page}</button>`);
}

// ---------- Buttons, status, requests ----------
function setBtn(id, loading, label) {
  const b = $(id);
  if (!b) return;
  b.disabled = loading;
  b.innerHTML = loading ? `<span class="spinner"></span> ${esc(label)}` : esc(label);
}
function say(id, kind, text) {     // kind: ok | bad | warn | "" (clear)
  const el = $(id);
  if (!el) return;
  el.innerHTML = kind ? `<span class="tag tag-${kind}">${esc(text)}</span>` : "";
}
async function postJSON(url, body) {
  let res;
  try {
    res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  } catch {
    throw new Error("Could not reach the server. Check that it is running and try again.");
  }
  let data;
  try { data = await res.json(); } catch {
    if (res.status === 502 || res.status === 504) throw new Error("The server took too long and gave up. Try again in a moment.");
    throw new Error(`The server sent an unreadable reply (${res.status}).`);
  }
  if (!res.ok || data.error) throw new Error(friendlyError(data.error || `Server error (${res.status})`));
  return data;
}
function friendlyError(message) {
  const m = String(message);
  if (/rate.?limit|429/i.test(m)) return "The language model is busy (rate limit). Wait about a minute and try again.";
  if (/budget/i.test(m)) return "This request needed more model calls than allowed. Try a narrower request.";
  if (/Could not connect to Groq|Connection error/i.test(m))
    return "The server could not reach the language model (Groq). Try again in a moment. If it keeps happening, check GROQ_API_KEY in the server settings for spaces or line breaks.";
  if (/GROQ_API_KEY|api key|401/i.test(m))
    return "The server has no working Groq API key. Set GROQ_API_KEY (in .env locally, or under Environment on Render) and restart.";
  return m.replace(/^LLM error:\s*/i, "");
}

// ---------- Agent trace, sources, evidence ----------
function traceItemsHtml(trace) {
  return (trace || []).map(e => `
    <div class="trace-item">
      <span class="trace-agent" title="${esc(e.agent)}">${esc(String(e.agent || "").replace(/_/g, " "))}</span>
      <span class="trace-kind k-${esc(e.kind)}">${esc(String(e.kind || "").replace("_", " "))}</span>
      <span class="trace-detail">${esc(e.detail)}</span>
    </div>`).join("");
}
function renderTrace(trace, llmCalls, open) {
  if (!trace || !trace.length) return "";
  const calls = llmCalls != null ? `, ${plural(llmCalls, "model call")}` : "";
  return `<details class="trace"${open ? " open" : ""}><summary>How this was produced (${plural(trace.length, "step")}${calls})</summary>
    <div class="trace-list">${traceItemsHtml(trace)}</div></details>`;
}
function renderSources(sources) {
  if (!sources || !sources.length) return "";
  const seen = new Set();
  const chips = sources.map(s => {
    if (s.type === "literature")
      return `<a class="cite lit" href="${esc(s.link || "#")}" target="_blank" rel="noopener" title="${esc(s.title || "")}">${esc((s.title || "related paper").slice(0, 48))}</a>`;
    const key = `${s.type}:${s.page}:${s.chunk_id ?? s.figure_id ?? s.table_id ?? s.snippet}`;
    if (seen.has(key) || !s.page) return "";
    seen.add(key);
    const what = s.type === "figure" ? (s.figure_id || "figure").replace(/^fig(\d+)$/, "Fig. $1")
      : s.type === "table" || s.kind === "table" ? "table" : s.kind === "caption" ? "caption" : s.kind === "figure" ? "figure" : "";
    return `<button type="button" class="cite" data-page="${esc(s.page)}"${s.chunk_id != null ? ` data-chunk="${esc(s.chunk_id)}"` : ""}` +
      `${s.figure_id ? ` data-figure="${esc(s.figure_id)}"` : ""} data-snippet="${esc(s.snippet || "")}" ` +
      `title="${esc(s.section || "")}">p. ${esc(s.page)}${what ? " " + esc(what) : ""}</button>`;
  }).join("");
  return chips ? `<div class="sources"><span class="sources-label">Read from</span>${chips}</div>` : "";
}
// How strong the evidence behind an answer was, and whether the answer stays inside it
function renderConfidence(r) {
  if (!r || !r.confidence) return "";
  const kind = { high: "ok", medium: "", low: "bad", none: "bad" }[r.confidence];
  const words = { high: "Strong evidence", medium: "Some evidence", low: "Weak evidence", none: "No evidence found" }[r.confidence];
  const tip = `Closest passage similarity ${r.top_score != null ? Number(r.top_score).toFixed(2) : "n/a"}` +
    (r.expanded ? ". The question was reworded to find better passages." : ".");
  let html = `<span class="tag ${kind ? "tag-" + kind : ""}" title="${esc(tip)}">${words}</span>`;
  const g = r.grounding || {};
  if ((g.unsupported_numbers || []).length)
    html += `<span class="tag tag-warn" title="These numbers are in the answer but not in the passages it read. Check them in the paper.">Check these numbers: ${esc(g.unsupported_numbers.slice(0, 4).join(", "))}</span>`;
  if ((g.unsupported_citations || []).length)
    html += `<span class="tag tag-warn" title="The answer cites pages it did not actually read.">Unread pages cited: ${esc(g.unsupported_citations.join(", "))}</span>`;
  return html;
}

// ---------- Source drawer: the real page, with the passage highlighted ----------
const viewer = { page: 1, pages: 0, params: {} };
function openSource(opts) {
  const paperId = SS.str("paper_id");
  if (!paperId) return;
  viewer.pages = (SS.get("stats") || {}).pages || 0;
  viewer.page = Number(opts.page) || 1;
  viewer.params = { chunk: opts.chunk, figure: opts.figure, quote: opts.quote };
  viewer.snippet = opts.snippet || opts.quote || "";
  viewer.home = viewer.page;
  $("drawerQuote").classList.toggle("hidden", !viewer.snippet);
  $("drawerQuote").textContent = viewer.snippet ? viewer.snippet + (viewer.snippet.length >= 220 ? "…" : "") : "";
  loadViewerPage();
  $("sourceDrawer").classList.add("open");
  $("scrim").classList.add("open");
  $("drawerClose").focus();
}
function loadViewerPage() {
  const paperId = SS.str("paper_id");
  const q = new URLSearchParams();
  if (viewer.page === viewer.home) {          // highlights belong to the cited page only
    for (const [k, v] of Object.entries(viewer.params)) if (v != null && v !== "") q.set(k, v);
  }
  const img = $("drawerPage");
  $("drawerNote").textContent = "Loading the page…";
  img.style.opacity = ".4";
  img.onload = () => { img.style.opacity = "1"; $("drawerNote").textContent = viewer.page === viewer.home && (q.toString()) ? "Highlighted: what the answer relied on." : ""; };
  img.onerror = () => { $("drawerNote").textContent = "This page could not be loaded. The server may have restarted: upload the paper again."; };
  img.src = `/papers/${encodeURIComponent(paperId)}/pages/${viewer.page}?${q}`;
  img.alt = `Page ${viewer.page} of the paper`;
  $("drawerTitle").textContent = `Page ${viewer.page}${viewer.pages ? " of " + viewer.pages : ""}`;
  $("drawerPrev").disabled = viewer.page <= 1;
  $("drawerNext").disabled = viewer.pages ? viewer.page >= viewer.pages : false;
}
function stepViewer(d) { viewer.page = Math.max(1, viewer.page + d); loadViewerPage(); }
function closeSource() { $("sourceDrawer").classList.remove("open"); $("scrim").classList.remove("open"); }

document.addEventListener("click", e => {
  const c = e.target.closest(".cite[data-page]");
  if (c) openSource({ page: c.dataset.page, chunk: c.dataset.chunk, figure: c.dataset.figure, quote: c.dataset.quote, snippet: c.dataset.snippet });
});
document.addEventListener("keydown", e => { if (e.key === "Escape") closeSource(); });

// ---------- Sidebar: which paper is open, and is it still on the server ----------
function refreshNav() {
  const title = SS.str("title") || SS.str("topic");
  const box = $("readingBox");
  if (!box) return;
  box.innerHTML = title ? `Reading<b>${esc(title)}</b>` : `No paper open yet`;
}
async function checkServerPaper() {
  if (!hasPaper()) return;
  try {
    const data = await (await fetch("/api/paper")).json();
    if (!data.paper || data.paper.id !== SS.str("paper_id")) {
      const b = $("lostBanner");
      if (b) b.classList.remove("hidden");
    }
  } catch { /* offline: the page's own requests will report it */ }
}
function forgetPaper() { PAPER_KEYS.forEach(SS.del); }

// Pages that need a paper call this first; returns false (and shows the empty state) when none is open
function requirePaper() {
  if (hasPaper()) return true;
  const main = $("mainContent"), empty = $("noPaper");
  if (main) main.classList.add("hidden");
  if (empty) empty.classList.remove("hidden");
  return false;
}

// ---------- Theme ----------
function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch { /* private mode */ }
}

// ---------- Read aloud (browser speech synthesis) ----------
function listen(btn, text) {
  if (!("speechSynthesis" in window)) { btn.textContent = "Not supported in this browser"; return; }
  if (speechSynthesis.speaking) { speechSynthesis.cancel(); btn.textContent = "Listen"; return; }
  const plain = String(text || "").replace(/[#*_`>|\[\]]/g, " ").replace(/\(p\.\s*\d+\)/g, "").replace(/\s+/g, " ");
  const u = new SpeechSynthesisUtterance(plain);
  u.rate = 1.02;
  u.onend = () => { btn.textContent = "Listen"; };
  speechSynthesis.speak(u);
  btn.textContent = "Stop";
}

// ---------- Tabs ----------
function wireTabs(container, onShow) {
  const tabs = [...container.querySelectorAll(".tab")];
  tabs.forEach(t => t.addEventListener("click", () => {
    tabs.forEach(o => { o.classList.toggle("active", o === t); $(o.dataset.panel)?.classList.toggle("hidden", o !== t); });
    if (onShow) onShow(t.dataset.panel);
  }));
}

function downloadText(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type: type || "text/plain" }));
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

document.addEventListener("DOMContentLoaded", () => { refreshNav(); checkServerPaper(); });
