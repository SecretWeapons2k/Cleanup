#!/usr/bin/env python3
"""Build ranker.html for one pipeline run (best result per text box).

Usage:  python build_ranker.py ~/cleanup_compare/RUN_NAME

Reads  <run>/debug_candidates/regions/*.json  (written by yolo_mask_auto_select.py unless
--no-debug-images is used) and writes <run>/ranker.html. Open that file in a browser
(double-click, no server needed). Pixel-identical candidate crops are merged into one tile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranker_server import DEFAULT_PORT, SAVE_JS, serve  # noqa: E402

HTML = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ranker – bestes Ergebnis pro Textbox</title>
<style>
  :root { --bg:#f4f4f6; --panel:#fff; --text:#1d1d22; --muted:#6b6b76; --line:#d9d9e0; --accent:#2b6cff; --ok:#1c9c58; --warn:#d9822b; --sel:#2b6cff; --bad:#d63a3a; }
  @media (prefers-color-scheme: dark) { :root { --bg:#17171b; --panel:#222228; --text:#ececf1; --muted:#9a9aa6; --line:#34343d; --accent:#6c9bff; --ok:#38c27a; --warn:#e8a04c; --sel:#6c9bff; --bad:#ef6a6a; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--panel); border-bottom:1px solid var(--line); padding:8px 14px; display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  .bar { flex:1 1 220px; min-width:160px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
  .bar > div { height:100%; background:var(--ok); width:0; transition:width .2s; }
  label.opt { display:flex; gap:6px; align-items:center; color:var(--muted); cursor:pointer; user-select:none; }
  button { font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:5px 11px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  main { padding:14px; max-width:2200px; margin:0 auto; }
  .titleline { display:flex; flex-wrap:wrap; gap:6px 16px; align-items:baseline; margin-bottom:10px; }
  .titleline h2 { margin:0; font-size:18px; }
  .meta { color:var(--muted); }
  .status { font-weight:600; }
  .status.done { color:var(--ok); } .status.none { color:var(--warn); } .status.skip { color:var(--muted); }
  .grid { display:grid; gap:12px; grid-template-columns:repeat(auto-fill, minmax(var(--tile,320px), 1fr)); }
  .tile { background:var(--panel); border:2px solid var(--line); border-radius:8px; overflow:hidden; display:flex; flex-direction:column; }
  .tile.cand { cursor:pointer; }
  .tile.cand:hover { border-color:var(--accent); }
  .tile.chosen { border-color:var(--sel); box-shadow:0 0 0 3px color-mix(in srgb, var(--sel) 35%, transparent); }
  .tile.orig { border-style:dashed; }
  .tile.bad { border-color:var(--bad); }
  .tile.bad .imgwrap img { opacity:.4; }
  .badge-bad { margin-left:auto; background:var(--bad); color:#fff; border-radius:4px; padding:0 6px; font-size:11px; font-weight:700; }
  .cap .badge-bad + .auto { margin-left:6px; }
  .cap { display:flex; gap:8px; align-items:center; padding:6px 8px; border-bottom:1px solid var(--line); min-height:34px; }
  .key { background:var(--text); color:var(--bg); border-radius:4px; padding:0 6px; font-weight:700; font-size:12px; }
  .cap .names { font-weight:600; word-break:break-word; }
  .cap .auto { margin-left:auto; background:var(--warn); color:#fff; border-radius:4px; padding:0 6px; font-size:11px; font-weight:700; }
  .imgwrap { padding:6px; display:flex; justify-content:center; align-items:center; flex:1;
    background-color:#e8e8ec; background-image:linear-gradient(45deg,#cfcfd6 25%,transparent 25%,transparent 75%,#cfcfd6 75%),linear-gradient(45deg,#cfcfd6 25%,transparent 25%,transparent 75%,#cfcfd6 75%);
    background-size:16px 16px; background-position:0 0,8px 8px; }
  .imgwrap img { max-width:100%; height:auto; display:block; image-rendering:auto; }
  .actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:14px; }
  .actions input[type=text] { flex:1 1 260px; font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:6px 9px; }
  .hint { color:var(--muted); margin-top:10px; font-size:12px; }
  #lightbox { position:fixed; inset:0; background:rgba(0,0,0,.85); display:none; z-index:20; overflow:auto; cursor:zoom-out; }
  #lightbox img { display:block; margin:30px auto; image-rendering:pixelated; background:#888; }
  .empty { padding:40px; text-align:center; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Ranker</h1>
  <div class="bar" title="Fortschritt"><div id="barfill"></div></div>
  <span id="progress" class="meta"></span><span id="saveState" class="meta"></span>
  <label class="opt"><input type="checkbox" id="optBlind" checked> Auto-Gewinner verbergen</label>
  <label class="opt"><input type="checkbox" id="optAdvance" checked> nach Wahl weiter</label>
  <label class="opt"><input type="checkbox" id="optOpen"> nur ungeprüfte</label>
  <label class="opt">Größe <input type="range" id="optSize" min="160" max="700" value="320"></label>
  <button id="btnImport" title="Früher exportierte CSV laden (z. B. altes Ranking)">CSV importieren</button><input type="file" id="fileImport" accept=".csv" hidden>
  <button id="btnExport" class="primary">CSV exportieren</button>
  <button id="btnReset" title="Alle Wahlen dieses Laufs löschen">Zurücksetzen</button>
</header>
<main>
  <div id="view"></div>
</main>
<div id="lightbox"><img id="lbimg" alt=""></div>

<script>
const DATA = __DATA__;
const RUN_ID = __RUN_ID__;
const STORE_KEY = "ranker:" + RUN_ID;
const AUTOSAVE_NAME = "ranking_" + RUN_ID + ".csv";
__SAVE_JS__

let picks = {};   // "image|region" -> {status:"pick"|"none"|"skip"|"open", ids:[best group], bad:[unbrauchbare ids], reviewed:bool, note}
try { picks = JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch (e) { picks = {}; }
function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(picks)); } catch (e) {} scheduleAutosave(); }

const rows = DATA.rows;
let order = rows.map((_, i) => i);
let pos = 0;
const $ = id => document.getElementById(id);
const keyOf = r => r.image + "|" + r.region;
const isDone = p => !!p && ["pick", "none", "skip", "checked"].includes(p.status);
const isReviewed = p => isDone(p) && p.reviewed === true;
const allIds = r => r.tiles.flatMap(t => t.ids);
const tileBad = (p, t) => !!p && Array.isArray(p.bad) && t.ids.every(id => p.bad.includes(id));

function label(id) { return /^\d+$/.test(id) ? id + "px" : id.replace(/_/g, " "); }
function tileName(t) { return t.ids.map(label).join(" = "); }
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

function rebuildOrder(keepCurrent) {
  const cur = order[pos];
  order = rows.map((_, i) => i).filter(i => !$("optOpen").checked || !isReviewed(picks[keyOf(rows[i])]) || i === cur);
  if (keepCurrent && cur !== undefined && order.includes(cur)) pos = order.indexOf(cur);
  else pos = Math.min(pos, Math.max(0, order.length - 1));
}

function updateProgress() {
  const done = rows.filter(r => isDone(picks[keyOf(r)])).length;
  const rev = rows.filter(r => isReviewed(picks[keyOf(r)])).length;
  $("progress").textContent = rev + " / " + rows.length + " vollständig geprüft" + (done > rev ? " · " + (done - rev) + " nur Bestwahl" : "");
  $("barfill").style.width = (rows.length ? 100 * rev / rows.length : 0) + "%";
}

function render() {
  updateProgress();
  const view = $("view");
  if (!order.length) { view.innerHTML = '<div class="empty">Nichts zu zeigen (alles geprüft?).</div>'; return; }
  const r = rows[order[pos]];
  const p = picks[keyOf(r)];
  const blind = $("optBlind").checked;
  const nBad = p && p.bad ? r.tiles.filter(t => tileBad(p, t)).length : 0;
  let statusText = "offen", statusCls = "";
  if (p && p.status === "pick") {
    statusText = "beste: " + p.ids.map(label).join(" = ") + " · " + (r.tiles.length - nBad) + " von " + r.tiles.length + " brauchbar" + (p.reviewed ? "" : " (Brauchbarkeit ungeprüft)");
    statusCls = p.reviewed ? "done" : "none";
  } else if (p && p.status === "checked") { statusText = "geprüft (ohne Bestwahl): " + (r.tiles.length - nBad) + " von " + r.tiles.length + " brauchbar"; statusCls = "done";
  } else if (p && p.status === "none") { statusText = "kein Kandidat brauchbar"; statusCls = "none"; }
  else if (p && p.status === "skip") { statusText = "übersprungen"; statusCls = "skip"; }
  else if (nBad) { statusText = nBad + " unbrauchbar markiert – Beste noch wählen"; }

  let html = '<div class="titleline"><h2>' + esc(r.image) + ' · Textbox ' + r.region + ' / ' + r.regionCount + '</h2>' +
    '<span class="meta">Eintrag ' + (pos + 1) + ' von ' + order.length + ' · bbox ' + r.bbox.join(", ") + '</span>' +
    '<span class="status ' + statusCls + '">' + esc(statusText) + '</span></div><div class="grid">';
  html += tile('orig', 'Original', r.original, null, false, false, false);
  r.tiles.forEach((t, i) => {
    const chosen = p && p.status === "pick" && p.ids.some(id => t.ids.includes(id));
    const isAuto = !blind && t.ids.includes(r.auto_winner);
    html += tile('cand', tileName(t), t.file, i, chosen, isAuto, tileBad(p, t));
  });
  html += '</div><div class="actions">' +
    '<button id="btnChecked" class="primary">Fertig, ohne Bestwahl (Enter)</button><button id="btnNone">Keiner brauchbar (n)</button><button id="btnSkip">Überspringen (s)</button>' +
    '<button id="btnPrev">← Zurück</button><button id="btnNext">Weiter →</button>' +
    '<input type="text" id="note" placeholder="Notiz (optional)" value="' + esc((p && p.note) || "") + '"></div>' +
    '<div class="hint"><b>Klick / Taste 1–9, q, w, e</b> = beste Kachel (zählt als brauchbar) · <b>Shift+Klick / Shift+Taste</b> = Kachel als <b>unbrauchbar</b> markieren (nochmal = zurücknehmen). ' +
    'Alles, was du nicht als unbrauchbar markierst, gilt als brauchbar. ' +
    '<i>Brauchbar = kein erkennbarer Textrest, keine Flecken/Schlieren, Kontur und Muster des Sprites intakt, Fläche passt zum Asset.</i> ' +
    '<b>Enter</b> = Brauchbarkeit fertig geprüft (ohne Bestwahl) · Doppelklick = Zoom · n = alle unbrauchbar · s = überspringen · ←/→ = navigieren. Pixelgleiche Ergebnisse sind eine Kachel.</div>';
  view.innerHTML = html;

  view.querySelectorAll(".tile.cand").forEach(el => {
    el.addEventListener("click", e => { const idx = +el.dataset.idx; e.shiftKey ? toggleBad(idx) : choose(idx); });
    el.addEventListener("dblclick", e => { e.preventDefault(); zoom(el.querySelector("img").src); });
  });
  view.querySelector(".tile.orig img").addEventListener("dblclick", e => zoom(e.target.src));
  $("btnChecked").onclick = () => setStatus("checked");
  $("btnNone").onclick = () => setStatus("none");
  $("btnSkip").onclick = () => setStatus("skip");
  $("btnPrev").onclick = () => go(-1);
  $("btnNext").onclick = () => go(1);
  $("note").addEventListener("input", e => { const q = ensure(r); q.note = e.target.value; save(); });
}

function tile(kind, name, file, idx, chosen, isAuto, bad) {
  const keyBadge = kind === 'cand' && idx < KEYS.length ? '<span class="key">' + KEYS[idx].toUpperCase() + '</span>' : '';
  return '<div class="tile ' + kind + (chosen ? ' chosen' : '') + (bad ? ' bad' : '') + '"' + (kind === 'cand' ? ' data-idx="' + idx + '"' : '') + '>' +
    '<div class="cap">' + keyBadge + '<span class="names">' + esc(name) + '</span>' +
    (bad ? '<span class="badge-bad">✗ unbrauchbar</span>' : '') + (isAuto ? '<span class="auto">AUTO</span>' : '') + '</div>' +
    '<div class="imgwrap"><img src="' + encodeURI(file) + '" alt="' + esc(name) + '"></div></div>';
}

function noteNow() { const n = $("note"); return n ? n.value : ""; }
function ensure(r) {
  const k = keyOf(r);
  if (!picks[k]) picks[k] = { status: "open", ids: [], bad: [], reviewed: false, note: "" };
  if (!Array.isArray(picks[k].bad)) picks[k].bad = [];
  return picks[k];
}

function choose(idx) {
  const r = rows[order[pos]]; const t = r.tiles[idx]; if (!t) return;
  const p = ensure(r);
  p.status = "pick"; p.ids = t.ids.slice(); p.bad = p.bad.filter(id => !t.ids.includes(id));
  p.reviewed = true; p.note = noteNow();
  save(); afterPick();
}
function toggleBad(idx) {
  const r = rows[order[pos]]; const t = r.tiles[idx]; if (!t) return;
  const p = ensure(r);
  if (tileBad(p, t)) p.bad = p.bad.filter(id => !t.ids.includes(id));
  else {
    t.ids.forEach(id => { if (!p.bad.includes(id)) p.bad.push(id); });
    if (p.status === "pick" && p.ids.some(id => t.ids.includes(id))) { p.status = "open"; p.ids = []; }
  }
  if (p.status === "none" && !r.tiles.every(x => tileBad(p, x))) p.status = "open";
  p.reviewed = true; p.note = noteNow();
  save(); render();
}
function setStatus(status) {
  const r = rows[order[pos]];
  const p = ensure(r);
  p.status = status; p.ids = []; p.reviewed = true; p.note = noteNow();
  if (status === "none") p.bad = allIds(r);
  save(); afterPick();
}
function afterPick() { if ($("optAdvance").checked) go(1, true); else render(); }
function go(delta, afterChoice) {
  if ($("optOpen").checked && afterChoice) { rebuildOrder(false); pos = Math.min(pos, order.length - 1); }
  else { pos = Math.max(0, Math.min(order.length - 1, pos + delta)); }
  render(); window.scrollTo(0, 0);
}
function zoom(src) { $("lbimg").src = src; $("lightbox").style.display = "block"; }
$("lightbox").addEventListener("click", () => { $("lightbox").style.display = "none"; });

const KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "q", "w", "e"];
const CODES = ["Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Digit6", "Digit7", "Digit8", "Digit9", "KeyQ", "KeyW", "KeyE"];
document.addEventListener("keydown", e => {
  if (e.target && e.target.tagName === "INPUT" && e.target.type === "text") { if (e.key === "Escape") e.target.blur(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "Escape") { $("lightbox").style.display = "none"; return; }
  if (e.key === "ArrowRight") { go(1); return; }
  if (e.key === "ArrowLeft") { go(-1); return; }
  const i = CODES.indexOf(e.code);
  if (i >= 0) { e.shiftKey ? toggleBad(i) : choose(i); return; }
  if (e.shiftKey) return;
  const k = e.key.toLowerCase();
  if (e.key === "Enter") { e.preventDefault(); setStatus("checked"); return; }
  if (k === "n") setStatus("none");
  else if (k === "s") setStatus("skip");
});

$("optOpen").addEventListener("change", () => { rebuildOrder(true); render(); });
$("optBlind").addEventListener("change", render);
$("optSize").addEventListener("input", e => { document.documentElement.style.setProperty("--tile", e.target.value + "px"); });

function csvCell(v) { v = String(v == null ? "" : v); return /[",\n;]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function buildCsv() {
  const head = ["image", "region", "region_count", "status", "chosen", "chosen_group", "bad_ids", "usable_ids", "reviewed", "auto_winner", "auto_agrees", "auto_usable", "note"];
  const lines = [head.join(",")];
  rows.forEach(r => {
    const p = picks[keyOf(r)]; if (!isDone(p)) return;
    const ids = p.ids || [];
    const bad = p.reviewed ? (p.bad || []) : [];
    const usable = p.reviewed ? allIds(r).filter(id => !bad.includes(id)) : [];
    const agrees = p.status === "pick" ? (ids.includes(r.auto_winner) ? 1 : 0) : "";
    const autoUsable = p.reviewed && p.status !== "skip" ? (bad.includes(r.auto_winner) ? 0 : 1) : "";
    lines.push([r.image, r.region, r.regionCount, p.status, ids[0] || "", ids.join("+"), bad.join("+"), usable.join("+"),
      p.reviewed ? 1 : 0, r.auto_winner, agrees, autoUsable, p.note || ""].map(csvCell).join(","));
  });
  return "\ufeff" + lines.join("\n") + "\n";
}
$("btnExport").addEventListener("click", () => exportCsv("ranking"));

function parseCsv(text) {
  text = text.replace(/^\ufeff/, "");
  const out = []; let row = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\n") { row.push(cell); out.push(row); row = []; cell = ""; }
    else if (c !== "\r") cell += c;
  }
  if (cell !== "" || row.length) { row.push(cell); out.push(row); }
  return out;
}
$("btnImport").addEventListener("click", () => $("fileImport").click());
$("fileImport").addEventListener("change", async e => {
  const f = e.target.files[0]; if (!f) return;
  const table = parseCsv(await f.text()); const head = table.shift() || [];
  const col = n => head.indexOf(n);
  const known = new Set(rows.map(keyOf)); let n = 0, skipped = 0;
  table.forEach(cells => {
    if (cells.length < 4) return;
    const image = cells[col("image")], region = cells[col("region")];
    const k = image + "|" + region;
    if (!known.has(k)) { skipped++; return; }
    const split = v => (v || "").split("+").filter(Boolean);
    const hasBad = col("bad_ids") >= 0;
    picks[k] = {
      status: cells[col("status")], ids: split(cells[col("chosen_group")]),
      bad: hasBad ? split(cells[col("bad_ids")]) : [],
      reviewed: hasBad && cells[col("reviewed")] === "1",
      note: col("note") >= 0 ? cells[col("note")] : "",
    };
    n++;
  });
  save(); rebuildOrder(false); render();
  alert(n + " Einträge importiert" + (skipped ? ", " + skipped + " nicht zu diesem Lauf passend" : "") +
    (head.indexOf("bad_ids") < 0 ? "\nAlte CSV ohne Brauchbarkeit: Bestwahl übernommen, Brauchbarkeit gilt als ungeprüft (Filter „nur ungeprüfte“)." : ""));
  e.target.value = "";
});
$("btnReset").addEventListener("click", () => {
  if (confirm("Alle Wahlen dieses Laufs wirklich löschen?")) { picks = {}; save(); rebuildOrder(false); render(); }
});

(function init() {
  const first = rows.findIndex(r => !isReviewed(picks[keyOf(r)]));
  pos = first >= 0 ? first : 0;
  render();
})();
</script>
</body>
</html>
"""


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build(run_dir: Path) -> tuple[Path, int, int]:
    regions_dir = run_dir / "debug_candidates" / "regions"
    if not regions_dir.is_dir():
        raise SystemExit(
            f"Keine Region-Ausschnitte gefunden: {regions_dir}\n"
            "Der Lauf braucht Debug-Bilder (ohne --no-debug-images) und die Version mit Regionsmodus."
        )
    rows: list[dict] = []
    images = 0
    for jpath in sorted(regions_dir.glob("*.json"), key=lambda p: p.name.casefold()):
        doc = json.loads(jpath.read_text(encoding="utf-8"))
        images += 1
        regions = doc["regions"]
        for reg in sorted(regions, key=lambda r: int(r["region"])):
            groups: dict[str, dict] = {}
            for cid, fname in reg["candidates"].items():
                fpath = regions_dir / fname
                if not fpath.is_file():
                    continue
                key = md5(fpath)
                g = groups.setdefault(key, {"ids": [], "file": f"debug_candidates/regions/{fname}"})
                g["ids"].append(cid)
            rows.append({
                "image": doc["image"],
                "region": int(reg["region"]),
                "regionCount": len(regions),
                "bbox": reg["bbox"],
                "auto_winner": reg["auto_winner"],
                "original": f"debug_candidates/regions/{reg['original']}",
                "tiles": list(groups.values()),
            })
    rows.sort(key=lambda r: (r["image"].casefold(), r["region"]))
    run_id = hashlib.md5(f"{run_dir.resolve().name}|{len(rows)}".encode()).hexdigest()[:10]
    payload = json.dumps({"rows": rows}, ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("__SAVE_JS__", SAVE_JS).replace("__DATA__", payload).replace("__RUN_ID__", json.dumps(run_id))
    out = run_dir / "ranker.html"
    out.write_text(html, encoding="utf-8")
    return out, images, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Ausgabeordner eines Pipeline-Laufs (enthält debug_candidates/)")
    ap.add_argument("--export-dir", type=Path, default=Path(__file__).resolve().parent / "ranker_out",
                    help="Ordner für exportierte/gesicherte CSVs (Standard: ranker_out neben diesem Skript)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-serve", action="store_true", help="Nur ranker.html bauen, keinen Server starten")
    ap.add_argument("--no-browser", action="store_true", help="Server starten, Browser nicht automatisch öffnen")
    args = ap.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    out, images, regions = build(run_dir)
    print(f"OK: {regions} Textbox-Regionen aus {images} Bildern -> {out}")
    if not args.no_serve:
        serve(run_dir, args.export_dir.expanduser().resolve(), out.name, args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
