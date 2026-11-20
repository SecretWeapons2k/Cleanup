#!/usr/bin/env python3
"""Build gate_labeler.html for one stage-1 run: label every Box-only result usable / unusable.

Usage:  python build_gate_labeler.py ~/cleanup_compare/STAGE1_RUN

Reads <run>/labeling/*.json (written by yolo_stage1.py) and writes <run>/gate_labeler.html.
Open that file in a browser (double-click). One screen per text region: original, Box-only
result and the mask overlay. J = usable, N or 1-5 = unusable (with reason), U = unsure.
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
<title>Gate-Labeler – Box-only brauchbar?</title>
<style>
  :root { --bg:#f4f4f6; --panel:#fff; --text:#1d1d22; --muted:#6b6b76; --line:#d9d9e0; --accent:#2b6cff; --ok:#1c9c58; --bad:#d63a3a; --warn:#d9822b; }
  @media (prefers-color-scheme: dark) { :root { --bg:#17171b; --panel:#222228; --text:#ececf1; --muted:#9a9aa6; --line:#34343d; --accent:#6c9bff; --ok:#38c27a; --bad:#ef6a6a; --warn:#e8a04c; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--panel); border-bottom:1px solid var(--line); padding:8px 14px; display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  .bar { flex:1 1 200px; min-width:140px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
  .bar > div { height:100%; background:var(--ok); width:0; transition:width .2s; }
  label.opt { display:flex; gap:6px; align-items:center; color:var(--muted); cursor:pointer; user-select:none; }
  button { font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:5px 11px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.ok { background:var(--ok); border-color:var(--ok); color:#fff; font-weight:600; }
  button.bad { background:var(--bad); border-color:var(--bad); color:#fff; font-weight:600; }
  main { padding:14px; max-width:2400px; margin:0 auto; }
  .titleline { display:flex; flex-wrap:wrap; gap:6px 16px; align-items:baseline; margin-bottom:10px; }
  .titleline h2 { margin:0; font-size:18px; }
  .meta { color:var(--muted); }
  .status { font-weight:600; }
  .status.usable { color:var(--ok); } .status.unusable { color:var(--bad); } .status.unsure { color:var(--warn); }
  .panes { display:flex; gap:12px; align-items:flex-start; overflow-x:auto; }
  .pane { flex:1 1 0; min-width:0; background:var(--panel); border:2px solid var(--line); border-radius:8px; overflow:hidden; }
  .pane.result.usable { border-color:var(--ok); } .pane.result.unusable { border-color:var(--bad); } .pane.result.unsure { border-color:var(--warn); }
  .cap { padding:6px 10px; font-weight:600; border-bottom:1px solid var(--line); }
  .imgwrap { padding:6px; display:flex; justify-content:center; overflow:auto;
    background-color:#e8e8ec; background-image:linear-gradient(45deg,#cfcfd6 25%,transparent 25%,transparent 75%,#cfcfd6 75%),linear-gradient(45deg,#cfcfd6 25%,transparent 25%,transparent 75%,#cfcfd6 75%);
    background-size:16px 16px; background-position:0 0,8px 8px; }
  .imgwrap img { display:block; max-width:100%; height:auto; image-rendering:auto; }
  body.zoomed .pane { flex:0 0 auto; } body.zoomed .imgwrap img { max-width:none; image-rendering:pixelated; }
  .actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:14px; }
  .actions input[type=text] { flex:1 1 220px; font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:6px 9px; }
  .hint { color:var(--muted); margin-top:10px; font-size:12px; }
  .imgbox { position:relative; display:inline-block; line-height:0; }
  .imgbox.drawable { cursor:crosshair; }
  .frame { position:absolute; border:2px solid #ff2a2a; background:rgba(255,42,42,.12); pointer-events:none; box-sizing:border-box; }
  .frame.tmp { border-style:dashed; }
  .pane.minimap { flex:0 0 230px; }
  .mini .cur { position:absolute; border:2px solid #ffb400; background:rgba(255,180,0,.2); pointer-events:none; box-sizing:border-box; }
  .mini .oth { position:absolute; border:1px solid rgba(120,120,130,.9); pointer-events:none; box-sizing:border-box; }
  .empty { padding:40px; text-align:center; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Gate-Labeler</h1>
  <div class="bar"><div id="barfill"></div></div>
  <span id="progress" class="meta"></span><span id="saveState" class="meta"></span>
  <label class="opt"><input type="checkbox" id="optAdvance" checked> nach Label weiter</label>
  <label class="opt"><input type="checkbox" id="optOpen"> nur ungelabelte</label>
  <label class="opt"><input type="checkbox" id="optOverlay" checked> Masken-Ansicht</label>
  <label class="opt"><input type="checkbox" id="optZoom"> Zoom 1:1 / 2:1</label>
  <button id="btnImport">CSV importieren</button><input type="file" id="fileImport" accept=".csv" hidden>
  <button id="btnExport" class="primary">CSV exportieren</button>
  <button id="btnReset">Zurücksetzen</button>
</header>
<main><div id="view"></div></main>

<script>
const DATA = __DATA__;
const RUN_ID = __RUN_ID__;
const STORE_KEY = "gate:" + RUN_ID;
const AUTOSAVE_NAME = "gate_labels_" + RUN_ID + ".csv";
__SAVE_JS__
const REASONS = { "1": "Textrest", "2": "Fleck/Artefakt", "3": "Struktur/Kontur kaputt", "4": "Naht/Farbsprung", "5": "Sonstiges" };

let labels = {};   // "image|region" -> {label:"usable"|"unusable"|"unsure", reason:"", note:""}
try { labels = JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch (e) { labels = {}; }
function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(labels)); } catch (e) {} scheduleAutosave(); }

const rows = DATA.rows;
let order = rows.map((_, i) => i), pos = 0;
const $ = id => document.getElementById(id);
const keyOf = r => r.image + "|" + r.region;
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function rebuildOrder(keepCurrent) {
  const cur = order[pos];
  order = rows.map((_, i) => i).filter(i => !$("optOpen").checked || !labels[keyOf(rows[i])] || i === cur);
  if (keepCurrent && cur !== undefined && order.includes(cur)) pos = order.indexOf(cur);
  else pos = Math.min(pos, Math.max(0, order.length - 1));
}
function counts() {
  const c = { usable: 0, unusable: 0, unsure: 0 };
  rows.forEach(r => { const l = labels[keyOf(r)]; if (l) c[l.label]++; });
  return c;
}
function updateProgress() {
  const c = counts(), done = c.usable + c.unusable + c.unsure;
  $("progress").textContent = done + " / " + rows.length + " · brauchbar " + c.usable + " · unbrauchbar " + c.unusable + (c.unsure ? " · unsicher " + c.unsure : "");
  $("barfill").style.width = (rows.length ? 100 * done / rows.length : 0) + "%";
}

function render() {
  updateProgress();
  document.body.classList.toggle("zoomed", $("optZoom").checked);
  const view = $("view");
  if (!order.length) { view.innerHTML = '<div class="empty">Nichts zu zeigen (alles gelabelt?).</div>'; return; }
  const r = rows[order[pos]], l = labels[keyOf(r)];
  let st = "offen", cls = "";
  if (l) { cls = l.label; st = l.label === "usable" ? "brauchbar" : l.label === "unusable" ? "unbrauchbar" + (l.reason ? ": " + l.reason : "") : "unsicher"; }
  const scale = $("optZoom").checked ? 2 : 0;
  const W = r.window[2] - r.window[0] + 1, H = r.window[3] - r.window[1] + 1;
  const boxes = (l && l.boxes) || [];
  const framesHtml = boxes.map(b => '<div class="frame" style="left:' + (100 * b[0] / W) + '%;top:' + (100 * b[1] / H) + '%;width:' + (100 * b[2] / W) + '%;height:' + (100 * b[3] / H) + '%"></div>').join("");
  const pane = (title, file, extra, draw) => '<div class="pane ' + (extra || "") + '"><div class="cap">' + title + '</div><div class="imgwrap"><div class="imgbox' + (draw ? " drawable" : "") + '"' + (draw ? ' id="drawbox"' : "") + '><img src="' + encodeURI(file) + '" alt="' + title + '" draggable="false">' + framesHtml + '</div></div></div>';
  const pct = (v, t) => (100 * v / t) + "%";
  const hasMini = !!(r.imageSize && r.full);
  const [IW, IH] = hasMini ? r.imageSize : [1, 1];
  const others = r.allWindows.filter(w => w.region !== r.region).map(w => '<div class="oth" style="left:' + pct(w.window[0], IW) + ';top:' + pct(w.window[1], IH) + ';width:' + pct(w.window[2] - w.window[0] + 1, IW) + ';height:' + pct(w.window[3] - w.window[1] + 1, IH) + '"></div>').join("");
  const mini = !hasMini ? '' : '<div class="pane minimap"><div class="cap">Lage im Bild (' + IW + '×' + IH + ')</div><div class="imgwrap"><div class="imgbox mini"><img src="' + encodeURI(r.full) + '" alt="Gesamtbild" draggable="false">' + others +
    '<div class="cur" style="left:' + pct(r.window[0], IW) + ';top:' + pct(r.window[1], IH) + ';width:' + pct(W, IW) + ';height:' + pct(H, IH) + '"></div></div></div></div>';
  let html = '<div class="titleline"><h2>' + esc(r.image) + ' · Textbox ' + r.region + ' / ' + r.regionCount + '</h2>' +
    '<span class="meta">Eintrag ' + (pos + 1) + ' von ' + order.length + '</span><span class="status ' + cls + '">' + esc(st) + '</span></div><div class="panes">';
  html += pane("Original", r.original);
  html += pane("Ergebnis (Box-only) – hier Rahmen ziehen", r.result, "result " + cls, true);
  if ($("optOverlay").checked) html += pane("Maske (magenta)", r.overlay);
  html += mini;
  html += '</div><div class="actions">' +
    '<button class="ok" id="bOk">Brauchbar (J)</button><button class="bad" id="bBad">Unbrauchbar (N)</button>' +
    Object.keys(REASONS).map(k => '<button class="bad" data-reason="' + k + '" style="opacity:.8">' + k + ' ' + REASONS[k] + '</button>').join("") +
    '<button id="bUnsure">Unsicher (U)</button><button id="bClear">Löschen (⌫)</button><button id="bFrame">Letzten Rahmen löschen (X) · ' + boxes.length + '</button><button id="bPrev">←</button><button id="bNext">→</button>' +
    '<input type="text" id="note" placeholder="Notiz (optional)" value="' + esc((l && l.note) || "") + '"></div>' +
    '<div class="hint"><b>Rahmen:</b> im Ergebnis mit der Maus aufziehen, was dich stört (zählt dann als unbrauchbar), X = letzten Rahmen löschen · <b>J</b>/Enter = brauchbar · <b>N</b> = unbrauchbar · <b>1–5</b> = unbrauchbar mit Grund (Textrest, Fleck, Struktur, Naht, Sonstiges) · <b>U</b> = unsicher · ⌫ = Label löschen · ←/→ = navigieren · M = Masken-Ansicht · Z = Zoom. ' +
    '<i>Brauchbar = kein erkennbarer Textrest, keine Flecken/Schlieren, Kontur und Muster des Sprites intakt, Fläche passt zum Asset.</i> Im Zweifel: „unsicher“ oder „unbrauchbar“ – die Prüfung soll lieber zu streng als zu lasch sein.</div>';
  view.innerHTML = html;
  if (scale) view.querySelectorAll("img").forEach(im => { im.onload = () => { im.style.width = (im.naturalWidth * scale) + "px"; }; if (im.complete) im.onload(); });
  $("bOk").onclick = () => setLabel("usable"); $("bBad").onclick = () => setLabel("unusable");
  view.querySelectorAll("[data-reason]").forEach(b => b.onclick = () => setLabel("unusable", REASONS[b.dataset.reason]));
  $("bUnsure").onclick = () => setLabel("unsure"); $("bClear").onclick = clearLabel; $("bFrame").onclick = delFrame;
  attachDraw($("drawbox"), r, W, H);
  $("bPrev").onclick = () => go(-1); $("bNext").onclick = () => go(1);
  $("note").addEventListener("input", e => { const k = keyOf(r); labels[k] = labels[k] || { label: "unsure", reason: "", note: "", boxes: [] }; labels[k].note = e.target.value; save(); });
}

function noteNow() { const n = $("note"); return n ? n.value : ""; }
function setLabel(label, reason) {
  const r = rows[order[pos]];
  const old = labels[keyOf(r)];
  labels[keyOf(r)] = { label, reason: reason || "", note: noteNow(), boxes: (old && old.boxes) || [] };
  save(); if ($("optAdvance").checked) go(1, true); else render();
}
function clearLabel() { delete labels[keyOf(rows[order[pos]])]; save(); render(); }
function delFrame() {
  const k = keyOf(rows[order[pos]]), l = labels[k];
  if (l && l.boxes && l.boxes.length) { l.boxes.pop(); save(); render(); }
}
let drawing = null;
function attachDraw(box, r, W, H) {
  if (!box) return;
  box.addEventListener("mousedown", e => {
    if (e.button !== 0) return;
    const rc = box.getBoundingClientRect();
    const tmp = document.createElement("div"); tmp.className = "frame tmp"; box.appendChild(tmp);
    drawing = { box, r, W, H, rc, x0: e.clientX - rc.left, y0: e.clientY - rc.top, tmp };
    e.preventDefault();
  });
}
window.addEventListener("mousemove", e => {
  if (!drawing) return;
  const d = drawing, x = Math.max(0, Math.min(d.rc.width, e.clientX - d.rc.left)), y = Math.max(0, Math.min(d.rc.height, e.clientY - d.rc.top));
  const L = Math.min(d.x0, x), T = Math.min(d.y0, y);
  Object.assign(d.tmp.style, { left: L + "px", top: T + "px", width: Math.abs(x - d.x0) + "px", height: Math.abs(y - d.y0) + "px" });
});
window.addEventListener("mouseup", e => {
  if (!drawing) return;
  const d = drawing; drawing = null;
  const x = Math.max(0, Math.min(d.rc.width, e.clientX - d.rc.left)), y = Math.max(0, Math.min(d.rc.height, e.clientY - d.rc.top));
  d.tmp.remove();
  const sx = d.W / d.rc.width, sy = d.H / d.rc.height;
  const bx = Math.round(Math.min(d.x0, x) * sx), by = Math.round(Math.min(d.y0, y) * sy);
  const bw = Math.round(Math.abs(x - d.x0) * sx), bh = Math.round(Math.abs(y - d.y0) * sy);
  if (bw < 3 || bh < 3) return;                      // a plain click is not a frame
  const k = keyOf(d.r);
  const l = labels[k] || (labels[k] = { label: "unusable", reason: "", note: noteNow(), boxes: [] });
  if (l.label === "usable") l.label = "unusable";
  l.boxes = l.boxes || []; l.boxes.push([bx, by, bw, bh]);
  save(); render();
});
function go(delta, after) {
  if ($("optOpen").checked && after) { rebuildOrder(false); pos = Math.min(pos, order.length - 1); }
  else pos = Math.max(0, Math.min(order.length - 1, pos + delta));
  render(); window.scrollTo(0, 0);
}

document.addEventListener("keydown", e => {
  if (e.target && e.target.tagName === "INPUT" && e.target.type === "text") { if (e.key === "Escape") e.target.blur(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1);
  else if (k === "j" || e.key === "Enter") { e.preventDefault(); setLabel("usable"); }
  else if (k === "n") setLabel("unusable");
  else if (REASONS[e.key]) setLabel("unusable", REASONS[e.key]);
  else if (k === "u") setLabel("unsure");
  else if (k === "x") delFrame();
  else if (e.key === "Backspace") { e.preventDefault(); clearLabel(); }
  else if (k === "m") { $("optOverlay").checked = !$("optOverlay").checked; render(); }
  else if (k === "z") { $("optZoom").checked = !$("optZoom").checked; render(); }
});
$("optOpen").addEventListener("change", () => { rebuildOrder(true); render(); });
["optOverlay", "optZoom"].forEach(id => $(id).addEventListener("change", render));

const cell = v => { v = String(v == null ? "" : v); return /[",\n;]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
function buildCsv() {
  const lines = ["image,region,label,reason,note,boxes_crop,boxes_image"];
  rows.forEach(r => {
    const l = labels[keyOf(r)]; if (!l) return;
    const bx = l.boxes || [];
    lines.push([r.image, r.region, l.label, l.reason, l.note,
      bx.map(b => b.join(",")).join(";"),
      bx.map(b => [b[0] + r.window[0], b[1] + r.window[1], b[2], b[3]].join(",")).join(";")].map(cell).join(","));
  });
  return "\ufeff" + lines.join("\n") + "\n";
}
$("btnExport").addEventListener("click", () => exportCsv("gate_labels"));
function parseCsv(text) {
  text = text.replace(/^\ufeff/, ""); const out = []; let row = [], c = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) { if (ch === '"') { if (text[i + 1] === '"') { c += '"'; i++; } else q = false; } else c += ch; }
    else if (ch === '"') q = true; else if (ch === ",") { row.push(c); c = ""; }
    else if (ch === "\n") { row.push(c); out.push(row); row = []; c = ""; } else if (ch !== "\r") c += ch;
  }
  if (c !== "" || row.length) { row.push(c); out.push(row); }
  return out;
}
$("btnImport").addEventListener("click", () => $("fileImport").click());
$("fileImport").addEventListener("change", async e => {
  const f = e.target.files[0]; if (!f) return;
  const t = parseCsv(await f.text()), h = t.shift() || [], col = n => h.indexOf(n), known = new Set(rows.map(keyOf)); let n = 0;
  t.forEach(c => { const k = c[col("image")] + "|" + c[col("region")]; if (known.has(k)) { const bc = col("boxes_crop") >= 0 ? (c[col("boxes_crop")] || "") : "";
      labels[k] = { label: c[col("label")], reason: c[col("reason")] || "", note: c[col("note")] || "",
        boxes: bc ? bc.split(";").map(t => t.split(",").map(Number)).filter(b => b.length === 4 && b.every(Number.isFinite)) : [] }; n++; } });
  save(); rebuildOrder(false); render(); alert(n + " Labels importiert"); e.target.value = "";
});
$("btnReset").addEventListener("click", () => { if (confirm("Alle Labels dieses Laufs löschen?")) { labels = {}; save(); rebuildOrder(false); render(); } });

(function init() { const first = rows.findIndex(r => !labels[keyOf(r)]); pos = first >= 0 ? first : 0; render(); })();
</script>
</body>
</html>
"""


def build(run_dir: Path) -> tuple[Path, int, int]:
    lab = run_dir / "labeling"
    if not lab.is_dir():
        raise SystemExit(f"Kein Ordner {lab}. Zuerst yolo_stage1.py ausführen und den Ausgabeordner (ggf. run_...-Unterordner) angeben.")
    rows: list[dict] = []
    images = 0
    for jp in sorted(lab.glob("*.json"), key=lambda p: p.name.casefold()):
        doc = json.loads(jp.read_text(encoding="utf-8"))
        images += 1
        regs = sorted(doc["regions"], key=lambda r: int(r["region"]))
        all_windows = [{"region": int(g["region"]), "window": g["window"]} for g in regs]
        for reg in regs:
            rows.append({
                "image": doc["image"], "region": int(reg["region"]), "regionCount": len(regs), "bbox": reg["bbox"],
                "window": reg["window"], "imageSize": doc.get("imageSize"),
                "full": f"labeling/{doc['full']}" if doc.get("full") and (lab / doc["full"]).is_file() else None,
                "allWindows": all_windows,
                "original": f"labeling/{reg['original']}", "result": f"labeling/{reg['result']}", "overlay": f"labeling/{reg['overlay']}",
            })
    rows.sort(key=lambda r: (r["image"].casefold(), r["region"]))
    run_id = hashlib.md5(f"{run_dir.resolve().name}|{len(rows)}".encode()).hexdigest()[:10]
    payload = json.dumps({"rows": rows}, ensure_ascii=False).replace("</", "<\\/")
    out = run_dir / "gate_labeler.html"
    out.write_text(HTML.replace("__SAVE_JS__", SAVE_JS).replace("__DATA__", payload).replace("__RUN_ID__", json.dumps(run_id)), encoding="utf-8")
    return out, images, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Ausgabeordner von yolo_stage1.py (enthält labeling/)")
    ap.add_argument("--export-dir", type=Path, default=Path(__file__).resolve().parent / "ranker_out",
                    help="Ordner für exportierte/gesicherte CSVs (Standard: ranker_out neben diesem Skript)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-serve", action="store_true", help="Nur ranker/labeler-HTML bauen, keinen Server starten")
    ap.add_argument("--no-browser", action="store_true", help="Server starten, Browser nicht automatisch öffnen")
    a = ap.parse_args()
    run_dir = a.run_dir.expanduser().resolve()
    out, images, regions = build(run_dir)
    print(f"OK: {regions} Regionen aus {images} Bildern -> {out}")
    if not a.no_serve:
        serve(run_dir, a.export_dir.expanduser().resolve(), out.name, a.port, open_browser=not a.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
