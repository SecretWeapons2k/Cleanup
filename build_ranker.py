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

HTML = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ranker – bestes Ergebnis pro Textbox</title>
<style>
  :root { --bg:#f4f4f6; --panel:#fff; --text:#1d1d22; --muted:#6b6b76; --line:#d9d9e0; --accent:#2b6cff; --ok:#1c9c58; --warn:#d9822b; --sel:#2b6cff; }
  @media (prefers-color-scheme: dark) { :root { --bg:#17171b; --panel:#222228; --text:#ececf1; --muted:#9a9aa6; --line:#34343d; --accent:#6c9bff; --ok:#38c27a; --warn:#e8a04c; --sel:#6c9bff; } }
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
  <span id="progress" class="meta"></span>
  <label class="opt"><input type="checkbox" id="optBlind" checked> Auto-Gewinner verbergen</label>
  <label class="opt"><input type="checkbox" id="optAdvance" checked> nach Wahl weiter</label>
  <label class="opt"><input type="checkbox" id="optOpen"> nur offene</label>
  <label class="opt">Größe <input type="range" id="optSize" min="160" max="700" value="320"></label>
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
const KEYS = ["1","2","3","4","5","6","7","8","9","q","w","e"];

let picks = {};           // "image|region" -> {status:"pick"|"none"|"skip", ids:[...], note:""}
try { picks = JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch (e) { picks = {}; }
function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(picks)); } catch (e) {} }

const rows = DATA.rows;
let order = rows.map((_, i) => i);   // indices shown (filtered)
let pos = 0;
const $ = id => document.getElementById(id);
const keyOf = r => r.image + "|" + r.region;

function label(id) { return /^\d+$/.test(id) ? id + "px" : id.replace(/_/g, " "); }
function tileName(t) { return t.ids.map(label).join(" = "); }

function rebuildOrder(keepCurrent) {
  const cur = order[pos];
  order = rows.map((_, i) => i).filter(i => !$("optOpen").checked || !picks[keyOf(rows[i])] || i === cur);
  if (keepCurrent && cur !== undefined && order.includes(cur)) pos = order.indexOf(cur);
  else pos = Math.min(pos, Math.max(0, order.length - 1));
}

function updateProgress() {
  const done = rows.filter(r => picks[keyOf(r)]).length;
  $("progress").textContent = done + " / " + rows.length + " bewertet";
  $("barfill").style.width = (rows.length ? 100 * done / rows.length : 0) + "%";
}

function render() {
  updateProgress();
  const view = $("view");
  if (!order.length) { view.innerHTML = '<div class="empty">Nichts zu zeigen (alle bewertet?).</div>'; return; }
  const r = rows[order[pos]];
  const p = picks[keyOf(r)];
  const blind = $("optBlind").checked;
  let statusText = "offen", statusCls = "";
  if (p && p.status === "pick") { statusText = "gewählt: " + p.ids.map(label).join(" = "); statusCls = "done"; }
  else if (p && p.status === "none") { statusText = "kein Kandidat brauchbar"; statusCls = "none"; }
  else if (p && p.status === "skip") { statusText = "übersprungen"; statusCls = "skip"; }

  let html = '<div class="titleline"><h2>' + esc(r.image) + ' · Textbox ' + r.region + ' / ' + r.regionCount + '</h2>' +
    '<span class="meta">Eintrag ' + (pos + 1) + ' von ' + order.length + ' · bbox ' + r.bbox.join(", ") + '</span>' +
    '<span class="status ' + statusCls + '">' + esc(statusText) + '</span></div><div class="grid">';
  html += tile('orig', 'Original', r.original, null, false, false);
  r.tiles.forEach((t, i) => {
    const chosen = p && p.status === "pick" && p.ids.some(id => t.ids.includes(id));
    const isAuto = !blind && t.ids.includes(r.auto_winner);
    html += tile('cand', tileName(t), t.file, i, chosen, isAuto);
  });
  html += '</div><div class="actions">' +
    '<button id="btnNone">Keiner brauchbar (n)</button><button id="btnSkip">Überspringen (s)</button>' +
    '<button id="btnPrev">← Zurück</button><button id="btnNext">Weiter →</button>' +
    '<input type="text" id="note" placeholder="Notiz (optional)" value="' + esc((p && p.note) || "") + '"></div>' +
    '<div class="hint">Klick oder Taste 1–9, q, w, e wählt die Kachel · Doppelklick = Vollbild-Zoom · n = keiner brauchbar · s = überspringen · ←/→ = navigieren. Identische Ergebnisse sind zu einer Kachel zusammengefasst.</div>';
  view.innerHTML = html;

  view.querySelectorAll(".tile.cand").forEach(el => {
    el.addEventListener("click", () => choose(+el.dataset.idx));
    el.addEventListener("dblclick", e => { e.preventDefault(); zoom(el.querySelector("img").src); });
  });
  view.querySelector(".tile.orig img").addEventListener("dblclick", e => zoom(e.target.src));
  $("btnNone").onclick = () => setStatus("none");
  $("btnSkip").onclick = () => setStatus("skip");
  $("btnPrev").onclick = () => go(-1);
  $("btnNext").onclick = () => go(1);
  $("note").addEventListener("input", e => {
    const k = keyOf(r); picks[k] = picks[k] || { status: "skip", ids: [], note: "" };
    picks[k].note = e.target.value; save();
  });
}

function tile(kind, name, file, idx, chosen, isAuto) {
  const keyBadge = kind === 'cand' && idx < KEYS.length ? '<span class="key">' + KEYS[idx].toUpperCase() + '</span>' : '';
  return '<div class="tile ' + kind + (chosen ? ' chosen' : '') + '"' + (kind === 'cand' ? ' data-idx="' + idx + '"' : '') + '>' +
    '<div class="cap">' + keyBadge + '<span class="names">' + esc(name) + '</span>' + (isAuto ? '<span class="auto">AUTO</span>' : '') + '</div>' +
    '<div class="imgwrap"><img src="' + encodeURI(file) + '" alt="' + esc(name) + '"></div></div>';
}

function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

function noteNow() { const n = $("note"); return n ? n.value : ""; }

function choose(idx) {
  const r = rows[order[pos]]; const t = r.tiles[idx]; if (!t) return;
  picks[keyOf(r)] = { status: "pick", ids: t.ids.slice(), note: noteNow() };
  save(); afterPick();
}
function setStatus(status) {
  const r = rows[order[pos]];
  picks[keyOf(r)] = { status, ids: [], note: noteNow() };
  save(); afterPick();
}
function afterPick() {
  if ($("optAdvance").checked) { go(1, true); } else { render(); }
}
function go(delta, afterChoice) {
  const cur = order[pos];
  if ($("optOpen").checked && afterChoice) { rebuildOrder(false); pos = Math.min(pos, order.length - 1); }
  else { pos = Math.max(0, Math.min(order.length - 1, pos + delta)); }
  render(); window.scrollTo(0, 0);
}
function zoom(src) { $("lbimg").src = src; $("lightbox").style.display = "block"; }
$("lightbox").addEventListener("click", () => { $("lightbox").style.display = "none"; });

document.addEventListener("keydown", e => {
  if (e.target && e.target.tagName === "INPUT" && e.target.type === "text") { if (e.key === "Escape") e.target.blur(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "Escape") { $("lightbox").style.display = "none"; return; }
  if (e.key === "ArrowRight") { go(1); return; }
  if (e.key === "ArrowLeft") { go(-1); return; }
  const k = e.key.toLowerCase();
  if (k === "n") { setStatus("none"); return; }
  if (k === "s") { setStatus("skip"); return; }
  const i = KEYS.indexOf(k);
  if (i >= 0) choose(i);
});

$("optOpen").addEventListener("change", () => { rebuildOrder(true); render(); });
$("optBlind").addEventListener("change", render);
$("optSize").addEventListener("input", e => { document.documentElement.style.setProperty("--tile", e.target.value + "px"); });

function csvCell(v) { v = String(v == null ? "" : v); return /[",\n;]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
$("btnExport").addEventListener("click", () => {
  const head = ["image", "region", "region_count", "status", "chosen", "chosen_group", "auto_winner", "auto_agrees", "note"];
  const lines = [head.join(",")];
  rows.forEach(r => {
    const p = picks[keyOf(r)]; if (!p) return;
    const ids = p.ids || [];
    const agrees = p.status === "pick" ? (ids.includes(r.auto_winner) ? 1 : 0) : "";
    lines.push([r.image, r.region, r.regionCount, p.status, ids[0] || "", ids.join("+"), r.auto_winner, agrees, p.note || ""].map(csvCell).join(","));
  });
  const blob = new Blob(["\ufeff" + lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "ranking_" + RUN_ID + ".csv";
  document.body.appendChild(a); a.click(); a.remove();
});
$("btnReset").addEventListener("click", () => {
  if (confirm("Alle Wahlen dieses Laufs wirklich löschen?")) { picks = {}; save(); rebuildOrder(false); render(); }
});

// Start at first open entry.
(function init() {
  const firstOpen = rows.findIndex(r => !picks[keyOf(r)]);
  pos = firstOpen >= 0 ? firstOpen : 0;
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
    html = HTML.replace("__DATA__", payload).replace("__RUN_ID__", json.dumps(run_id))
    out = run_dir / "ranker.html"
    out.write_text(html, encoding="utf-8")
    return out, images, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Ausgabeordner eines Pipeline-Laufs (enthält debug_candidates/)")
    args = ap.parse_args()
    out, images, regions = build(args.run_dir.expanduser().resolve())
    print(f"OK: {regions} Textbox-Regionen aus {images} Bildern -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
