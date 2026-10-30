#!/usr/bin/env python3
"""Local labeling tool: mark which cleaned candidates are acceptable.

Standalone (stdlib + numpy + Pillow). Does not touch the pipeline.

    python label_tool.py \
        --run-dir /home/secret/Downloads/Cleanup/yolo_mask_winners/full_recheck_after_alpha_island_025_20261025_022053

Then open http://127.0.0.1:8765 . Labels are saved to <run-dir>/labels.csv after
every image (one row per candidate), so you can stop and resume at any time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

HOME = Path.home()
DEFAULT_IMAGES = HOME / "Downloads/yabai proj/yabaiproject - Copy/mit_kanji"
DEFAULT_ANCHORS = HOME / "Downloads/yabai proj/processed_output/anchors.csv"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# file suffix -> candidate id used in candidate_scores.csv ("distance_px" column)
SPECIAL = {"box_only": "BOX_ONLY", "contrast_t38": "CONTRAST_T38",
           "hybrid_d2": "HYBRID_D2", "hybrid_d6": "HYBRID_D6", "smart_ink": "SMART_INK"}
CAND_ORDER = ["0", "2", "4", "6", "8", "10", "HYBRID_D2", "HYBRID_D6",
              "CONTRAST_T38", "SMART_INK", "BOX_ONLY"]


def suffix_to_id(suffix: str) -> str | None:
    if suffix in SPECIAL:
        return SPECIAL[suffix]
    m = re.fullmatch(r"distance_(\d+)px", suffix)
    return str(int(m.group(1))) if m else None


class Store:
    def __init__(self, a):
        self.run = a.run_dir.resolve()
        self.cleaned = self.run / "debug_candidates" / "cleaned_images"
        if not self.cleaned.is_dir():
            raise SystemExit(f"Not found: {self.cleaned}")
        self.labels_path = a.labels or (self.run / "labels.csv")
        self.margin = a.margin
        self.snap = a.snap
        self.lock = threading.Lock()

        # candidate files per image
        self.cands: dict[str, dict[str, Path]] = defaultdict(dict)
        for p in sorted(self.cleaned.glob("*.png")):
            stem, sep, suf = p.stem.partition("__")
            cid = suffix_to_id(suf) if sep else None
            if cid:
                self.cands[stem][cid] = p
        if not self.cands:
            raise SystemExit("No candidate images recognised.")

        # originals
        print("Indexing source images ...")
        self.orig: dict[str, Path] = {}
        want = {f"{s}" for s in self.cands}
        for p in a.images_dir.expanduser().rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS and p.stem in want:
                self.orig.setdefault(p.stem, p)
        print(f"  originals found: {len(self.orig)}/{len(self.cands)}")

        # anchors
        self.boxes: dict[str, list[dict]] = defaultdict(list)
        if a.anchors.is_file():
            with a.anchors.open(encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    try:
                        name = Path((r.get("image") or "").strip()).stem
                        self.boxes[name].append({
                            "type": (r.get("type") or "").strip().lower(),
                            **{k: float(r[k]) for k in ("x1", "y1", "x2", "y2")}})
                    except (ValueError, KeyError, TypeError):
                        continue
        else:
            print(f"WARN: anchors not found ({a.anchors}); showing full images")

        # current winners (hidden in UI by default to avoid biasing labels)
        self.winner: dict[str, str] = {}
        wr = self.run / "winner_report.csv"
        if wr.is_file():
            with wr.open(encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    self.winner[Path(r["image"]).stem] = str(r.get("selected_distance_px", "")).strip()

        self.order = self._make_order(a)
        self.labels: dict[str, dict] = {}
        self._load_labels()
        self.ranker: dict[str, str] = {}
        if a.queue:
            if not a.queue.is_file():
                raise SystemExit(f"--queue file not found: {a.queue}")
            rows = []
            with a.queue.open(encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    n = Path(r["image"]).stem
                    self.ranker[n] = str(r.get("ranker_pick", ""))
                    if str(r.get("needs_review", "0")) == "1" and n in self.cands:
                        rows.append((float(r.get("top_prob") or 1), n))
            self.order = [n for _, n in sorted(rows)]
            print(f"Queue: {len(self.order)} images flagged needs_review (least confident first)")
        if not a.include_labeled:
            self.order = [n for n in self.order if n not in self.labels]
        self.hash_cache: dict[str, dict] = {}

    # ---- ordering: hard images interleaved with random ones (avoids bias)
    def _make_order(self, a) -> list[str]:
        names = sorted(self.cands)
        if a.order == "name":
            return names
        rng = random.Random(a.seed)
        rnd = names[:]
        rng.shuffle(rnd)
        if a.order == "random":
            return rnd
        spread: dict[str, float] = {n: 0.0 for n in names}
        cs = self.run / "candidate_scores.csv"
        if not cs.is_file():
            cs = self.run / "debug_candidates" / "candidate_scores.csv"
        if not cs.is_file() and a.scores:
            cs = Path(a.scores)
        if not cs.is_file():
            print("WARN: candidate_scores.csv not found; 'hard first' ordering disabled")
        if cs.is_file():
            vals = defaultdict(lambda: defaultdict(list))
            with cs.open(encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    n = Path(r["image"]).stem
                    for k in ("text_quality_score", "score"):
                        try:
                            vals[n][k].append(float(r[k]))
                        except (KeyError, ValueError):
                            pass
            for n, d in vals.items():
                s = sum((max(v) - min(v)) for v in d.values() if v)
                tq = d.get("text_quality_score", [])
                if tq and max(tq) == 0:
                    s += 0.5  # all-zero ties: scorer cannot discriminate
                spread[n] = s
        hard = sorted(names, key=lambda n: -spread[n])
        if a.order == "hard":
            return hard
        out, seen = [], set()
        for h, r in zip(hard, rnd):
            for n in (h, r):
                if n not in seen:
                    seen.add(n); out.append(n)
        return out

    # ---- labels
    def _load_labels(self):
        if not self.labels_path.is_file():
            return
        with self.labels_path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                d = self.labels.setdefault(r["image"], {"good": set(), "none": False, "skipped": False})
                if r["good"] == "1":
                    d["good"].add(r["candidate"])
                if r.get("none_good") == "1":
                    d["none"] = True
                if r.get("skipped") == "1":
                    d["skipped"] = True

    def _write_labels(self):
        tmp = self.labels_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "candidate", "good", "none_good", "skipped", "labeled_at"])
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for img, d in self.labels.items():
                for cid in CAND_ORDER:
                    if cid in self.cands.get(img, {}):
                        w.writerow([img, cid, int(cid in d["good"]), int(d["none"]),
                                    int(d["skipped"]), stamp])
        tmp.replace(self.labels_path)

    def save_label(self, img, good, none, skipped):
        with self.lock:
            self.labels[img] = {"good": set(good), "none": bool(none), "skipped": bool(skipped)}
            self._write_labels()

    # ---- cropping / dedupe
    def crop_rect(self, name: str, m: str = "") -> tuple[int, int, int, int]:
        margin = self.margin
        if m == "full":
            margin = None
        elif m.isdigit():
            margin = int(m)
        any_path = next(iter(self.cands[name].values()))
        with Image.open(any_path) as im:
            w, h = im.size
        bs = self.boxes.get(name)
        if not bs or margin is None:
            return 0, 0, w, h
        x1 = min(b["x1"] for b in bs) - margin
        y1 = min(b["y1"] for b in bs) - margin
        x2 = max(b["x2"] for b in bs) + margin
        y2 = max(b["y2"] for b in bs) + margin
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0, 0, w, h
        # Snap to image edges so near-edge content is never trimmed by a few px.
        snap = self.snap
        if x1 <= snap: x1 = 0
        if y1 <= snap: y1 = 0
        if w - x2 <= snap: x2 = w
        if h - y2 <= snap: y2 = h
        if (x2 - x1) * (y2 - y1) >= 0.85 * w * h:
            return 0, 0, w, h
        return x1, y1, x2, y2

    def info(self, name: str, m: str = "") -> dict:
        key = f"{name}|{m}"
        if key in self.hash_cache:
            return self.hash_cache[key]
        rect = self.crop_rect(name, m)
        groups: dict[str, list[str]] = {}
        for cid in CAND_ORDER:
            p = self.cands[name].get(cid)
            if not p:
                continue
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGBA").crop(rect))
            groups.setdefault(hashlib.md5(arr.tobytes()).hexdigest(), []).append(cid)
        info = {"rect": rect, "groups": list(groups.values())}
        self.hash_cache[key] = info
        return info

    def png(self, name: str, which: str, m: str = "") -> bytes:
        path = self.orig.get(name) if which == "orig" else self.cands[name].get(which)
        if path is None:
            raise FileNotFoundError(which)
        with Image.open(path) as im:
            im = im.convert("RGBA").crop(self.crop_rect(name, m))
        buf = BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()


HTML = r"""<!doctype html><meta charset=utf-8><title>Label candidates</title>
<style>
body{margin:0;font:14px system-ui;background:#1b1b1f;color:#ddd}
#bar{position:sticky;top:0;z-index:5;background:#26262c;padding:8px 12px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
#bar b{color:#fff}.k{background:#3a3a44;border-radius:3px;padding:0 5px;font-family:monospace}
#grid{display:flex;flex-wrap:wrap;gap:10px;padding:10px}
.t{border:3px solid #444;border-radius:6px;padding:4px;background:#2a2a30;cursor:pointer}
.t.good{border-color:#3ddc84;background:#203428}
.t.win{outline:3px dashed #ff9f1a}
.t img{display:block;image-rendering:pixelated;
 background:repeating-conic-gradient(#e9e9e9 0 25%,#cfcfcf 0 50%) 0 0/16px 16px}
.l{font-size:12px;margin-bottom:3px;display:flex;justify-content:space-between;gap:8px}
.l span:first-child{color:#fff}
#msg{color:#ff9f1a}
</style>
<div id=bar></div><div id=grid></div>
<script>
let Q=[],idx=0,cur=null,good=new Set(),none=false,cols=4,zm=1,mi=0,MS=['','64','128','full'],showWin=false,flick=false,showBoxes=false,autoNext=true;
const $=s=>document.querySelector(s);
async function j(u,o){return (await fetch(u,o)).json()}
function status(){
  const done=Q.filter(q=>q.labeled).length;
  $('#bar').innerHTML=`<b>${cur?cur.name:''}</b> <span>${idx+1}/${Q.length} · labeled ${done}</span>
  <span><span class=k>1-9,0,-</span> toggle tile</span><span><span class=k>Enter</span> save+next</span>
  <span><span class=k>N</span> none good${none?' ✔':''}</span><span><span class=k>S</span> skip</span>
  <span><span class=k>←/→</span> prev/next</span><span><span class=k>O</span> hold: original</span>
  <span><span class=k>[ ]</span> columns</span><span><span class=k>M</span> area: ${MS[mi]===''?'boxes+margin':MS[mi]==='full'?'FULL image':'margin '+MS[mi]}</span><span><span class=k>Z</span>/<span class=k>X</span> zoom in/out (${Math.round(zm*100)}%) <span class=k>R</span> reset</span><span><span class=k>W</span> show current winner ${showWin?'✔':''}</span>
  <span><span class=k>U</span> next unlabeled</span><span id=msg></span>`;
}
function render(){
  if(!cur)return;
  const g=$('#grid');g.innerHTML='';
  const [x1,y1,x2,y2]=cur.rect,cw=x2-x1,ch=y2-y1;
  const avail=Math.floor(innerWidth/cols)-30;
  let z=avail/cw;if(z>=1)z=Math.floor(z);
  const w=Math.max(40,Math.round(cw*z*zm));
  const items=[{ids:['orig'],label:'ORIGINAL'},...cur.groups.map(ids=>({ids,label:ids.join(' · ')}))];
  items.forEach((it,i)=>{
    const d=document.createElement('div');d.className='t';
    const isG=it.ids[0]!=='orig';
    if(isG&&it.ids.every(x=>good.has(x)))d.classList.add('good');
    if(isG&&showWin&&(it.ids.includes(cur.winner)||it.ids.includes(cur.ranker)))d.classList.add('win');
    const key=i===0?'':('1234567890-'[i-1]||'');
    d.innerHTML=`<div class=l><span>${key?`[${key}] `:''}${it.label}${isG&&showWin&&it.ids.includes(cur.winner)?' (current winner)':''}${isG&&showWin&&it.ids.includes(cur.ranker)?' (ranker pick)':''}</span></div>`;
    const im=new Image();im.width=w;im.dataset.src=`/crop/${encodeURIComponent(cur.name)}/${it.ids[0]}.png?m=${MS[mi]}`;
    im.src=flick?`/crop/${encodeURIComponent(cur.name)}/orig.png?m=${MS[mi]}`:im.dataset.src;
    d.appendChild(im);
    if(isG)d.onclick=()=>toggle(it.ids);
    g.appendChild(d);
  });
  status();
}
function toggle(ids){const on=ids.every(x=>good.has(x));ids.forEach(x=>on?good.delete(x):good.add(x));none=false;render()}
async function load(i){
  idx=Math.max(0,Math.min(Q.length-1,i));
  cur=await j('/api/image/'+encodeURIComponent(Q[idx].name)+'?m='+MS[mi]);
  good=new Set(cur.good);none=cur.none;render();scrollTo(0,0);
}
async function save(skipped){
  if(!skipped&&!none&&good.size===0){$('#msg').textContent='select tiles or press N';return}
  await j('/api/save',{method:'POST',body:JSON.stringify({image:cur.name,good:[...good],none:none,skipped:!!skipped})});
  Q[idx].labeled=true;
  load(idx+1);
}
function nextUnlabeled(){const n=Q.findIndex((q,i)=>i>idx&&!q.labeled);if(n>=0)load(n)}
addEventListener('keydown',e=>{
  if(e.repeat&&(e.key==='o'||e.key==='O'))return;
  const k=e.key;
  if('1234567890-'.includes(k)&&k.length===1){const n=('1234567890-').indexOf(k);
    if(cur&&cur.groups[n])toggle(cur.groups[n]);}
  else if(k==='Enter'||k===' '){e.preventDefault();save(false)}
  else if(k==='n'||k==='N'){none=!none;if(none)good.clear();render()}
  else if(k==='s'||k==='S')save(true);
  else if(k==='ArrowRight')load(idx+1);else if(k==='ArrowLeft')load(idx-1);
  else if(k==='u'||k==='U')nextUnlabeled();
  else if(k==='w'||k==='W'){showWin=!showWin;render()}
  else if(k==='z'||k==='Z'||k==='+'||k==='='){zm=Math.min(8,zm*1.25);render()}
  else if(k==='x'||k==='X'){zm=Math.max(0.25,zm/1.25);render()}
  else if(k==='r'||k==='R'){zm=1;render()}
  else if(k==='m'||k==='M'){mi=(mi+1)%MS.length;load(idx)}
  else if(k==='['){cols=Math.max(1,cols-1);render()}else if(k===']'){cols=Math.min(8,cols+1);render()}
  else if(k==='o'||k==='O'){flick=true;render()}
});
addEventListener('keyup',e=>{if(e.key==='o'||e.key==='O'){flick=false;render()}});
addEventListener('resize',render);
addEventListener('wheel',e=>{if(e.ctrlKey||e.shiftKey){e.preventDefault();
  zm=e.deltaY<0?Math.min(8,zm*1.15):Math.max(0.25,zm/1.15);render()}},{passive:false});
(async()=>{Q=await j('/api/queue');if(!Q.length){document.body.innerHTML='<p style="padding:20px">Nothing left to label in this queue.</p>';return}
  const f=Q.findIndex(q=>!q.labeled);await load(f<0?0:f)})();
</script>"""


def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(200, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            m = (parse_qs(parsed.query).get("m") or [""])[0]
            try:
                if path == "/":
                    return self._send(200, HTML.encode(), "text/html; charset=utf-8")
                if path == "/api/queue":
                    return self._json([{"name": n, "labeled": n in store.labels} for n in store.order])
                if path.startswith("/api/image/"):
                    n = path[len("/api/image/"):]
                    inf = store.info(n, m)
                    d = store.labels.get(n, {"good": set(), "none": False})
                    return self._json({"name": n, "rect": inf["rect"], "groups": inf["groups"],
                                       "good": sorted(d["good"]), "none": d["none"],
                                       "winner": store.winner.get(n, ""), "ranker": store.ranker.get(n, "")})
                mm = re.fullmatch(r"/crop/([^/]+)/([^/]+)\.png", path)
                if mm:
                    return self._send(200, store.png(mm.group(1), mm.group(2), m), "image/png")
            except Exception as e:  # noqa: BLE001
                return self._send(500, str(e).encode(), "text/plain")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path == "/api/save":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                store.save_label(body["image"], body["good"], body["none"], body["skipped"])
                return self._json({"ok": True})
            self._send(404, b"not found", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    ap.add_argument("--scores", type=Path, help="candidate_scores.csv if not inside run-dir")
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--order", choices=("mixed", "hard", "random", "name"), default="mixed",
                    help="hard = only by difficulty; mixed = hardest interleaved with random ones")
    ap.add_argument("--queue", type=Path,
                    help="ranker_picks.csv: show only images with needs_review=1, least confident first")
    ap.add_argument("--include-labeled", action="store_true",
                    help="keep already labeled images in the list (default: hide them)")
    ap.add_argument("--margin", type=int, default=24, help="crop margin around boxes (px)")
    ap.add_argument("--snap", type=int, default=40,
                    help="extend crop to image edge if within this many px (default 40)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    store = Store(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(store))
    url = f"http://127.0.0.1:{a.port}"
    print(f"{len(store.order)} images; labels -> {store.labels_path}\nOpen {url}  (Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
