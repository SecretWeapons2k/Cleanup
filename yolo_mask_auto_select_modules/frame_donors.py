"""Donor ranker: pick the best clean version of every frame once, use it everywhere.

Sprite sheets repeat one frame (same shape, same colours) in many cells that differ only in the text.
After a normal ranker pass every labelled region has a chosen result. This tool

  build     groups all regions of a stage 2 run into FRAMES (same cell shape + same colours outside the
            text), and writes a small ranker (index.html) with ONE page per frame. A page shows the
            results you already chose for the cells of that frame; you pick the best one.
  approve   puts the picked cells into a SiblingBank as approved frames.

Afterwards method FRAME_DONOR (stage 2, ``--sibling-bank DIR``) stamps the approved frame onto every cell of
that frame in every image, text or no text. The result is an ordinary candidate in the normal ranker.

    python -m yolo_mask_auto_select_modules.frame_donors build   --run STAGE2_RUN --ranking ranking.csv
    python -m yolo_mask_auto_select_modules.frame_donors approve --ranker STAGE2_RUN/donor_ranker \\
                                                                  --choices donor_choices.csv --bank BANK_DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:                                                      # inside the package ...
    from . import siblings as S
except ImportError:                                       # ... or next to the file (tests)
    import siblings as S

MAX_CANDIDATES = 24                  # candidates shown per frame (best automatic score first)
ZONE_CHANGE_LEVEL = 6                # a pixel that a candidate changed by more than this belongs to the text zone
ZONE_DILATE_PX = 3
GLOW_PX = 10                         # extra margin around the text zone for the interior comparison (text glow)
MIN_INTERIOR_PX = 200                # fewer interior pixels than this: interior unknown (rim decides)
INTERIOR_MEDIAN_MAX = 4.0            # median visible colour difference inside the frame, outside text + glow
INTERIOR_P90_MAX = 16.0
ALPHA_CORE_MISMATCH_MAX = 0.02       # share of the frame whose hard shape (alpha >= 128) differs
ALPHA_MEAN_DIFF_MAX = 3.0            # mean |alpha difference| over the frame (soft edges may be smoothed differently)
RIM_MEDIAN_MAX = 3.0                 # median visible difference along the outline
RIM_TAIL_LEVEL = 12.0                # a rim pixel differing by more than this counts as 'tail' (text glow reaching the rim)
RIM_TAIL_FRAC_MAX = 0.25             # tail share allowed when judged on the whole rim
RIM_CLEAN_TAIL_FRAC_MAX = 0.10       # tail share allowed on rim pixels away from text and glow
MIN_RIM_CLEAN_PX = 400
FILL_BY_RESULT_MEDIAN_MAX = 6.0      # fallback when no clean interior is left: compare the CLEANED results, blurred
FILL_BY_RESULT_P90_MAX = 20.0
FILL_BLUR_SIGMA = 6.0                # inpainting differs in detail from cell to cell; only the fill COLOUR must agree
RIM_PX = 8                           # frame identity is judged on this band along the outline (text/glow never reaches it)


@dataclass
class Member:
    image: str
    region: int
    window: tuple[int, int, int, int]
    source: str
    orig: Path
    cands: dict[str, Path]
    chosen: str | None = None                  # method picked in the ranking (status "pick")
    usable: list = field(default_factory=list)  # every method not marked bad in the ranking (chosen first)
    _feat: tuple | None = field(default=None, repr=False)
    _zbig: np.ndarray | None = field(default=None, repr=False)
    _cand: tuple | None = field(default=None, repr=False)

    @property
    def size(self) -> tuple[int, int]:
        return self.window[2] - self.window[0] + 1, self.window[3] - self.window[1] + 1


def _rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(Image.open(path).convert("RGBA"))
    return a[..., :3], a[..., 3]


def load_members(run: Path, ranking_csv: Path | None) -> list[Member]:
    picks: dict[tuple[str, int], str] = {}
    usable_by: dict[tuple[str, int], list[str]] = {}
    bad_by: dict[tuple[str, int], set[str]] = {}
    if ranking_csv is not None:
        with ranking_csv.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["image"], int(row["region"]))
                if row.get("status") == "pick" and row.get("chosen"):
                    picks[key] = row["chosen"]
                usable_by[key] = [x for x in (row.get("usable_ids") or "").split("+") if x]
                bad_by[key] = {x for x in (row.get("bad_ids") or "").split("+") if x}
    reg_dir = run / "debug_candidates" / "regions"
    members: list[Member] = []
    for jf in sorted(reg_dir.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        for e in data.get("regions", []):
            cands = {m: reg_dir / fn for m, fn in e.get("candidates", {}).items() if (reg_dir / fn).is_file()}
            orig = reg_dir / e.get("original", "")
            if not cands or not orig.is_file():
                continue
            key = (data["image"], int(e["region"]))
            chosen = picks.get(key)
            chosen = chosen if chosen in cands else None
            # all results the person did not mark as bad; the chosen one always counts
            use = [m for m in usable_by.get(key, []) if m in cands and m not in bad_by.get(key, set())]
            if chosen and chosen not in use:
                use.append(chosen)
            if chosen:
                use.sort(key=lambda m: m != chosen)
            members.append(Member(data["image"], int(e["region"]), tuple(e["window"]), e.get("window_source", ""),
                                  orig, cands, chosen, use))
    return members


def _features(m: Member) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(original rgb, alpha, text zone). The zone is where any candidate changed the original."""
    if m._feat is None:
        rgb, alpha = _rgba(m.orig)
        zone = np.zeros(alpha.shape, bool)
        for p in m.cands.values():
            c, ca = _rgba(p)
            if c.shape == rgb.shape:
                zone |= np.abs(c.astype(np.int16) - rgb.astype(np.int16)).max(axis=-1) > ZONE_CHANGE_LEVEL
                zone |= np.abs(ca.astype(np.int16) - alpha.astype(np.int16)) > ZONE_CHANGE_LEVEL
        zone = cv2.dilate(zone.astype(np.uint8), np.ones((2 * ZONE_DILATE_PX + 1,) * 2, np.uint8)) > 0
        m._feat = (rgb, alpha, zone)
    return m._feat


def _cand_pixels(m: Member):
    """(rgb, alpha) of the preferred cleaned result of the cell (BOX_CELL if present)."""
    if not m.cands:
        return None
    if m._cand is None:
        name = "BOX_CELL" if "BOX_CELL" in m.cands else sorted(m.cands)[0]
        m._cand = _rgba(m.cands[name])
    return m._cand


def _zone_big(m: Member) -> np.ndarray:
    if m._zbig is None:
        z = _features(m)[2].astype(np.uint8)
        m._zbig = cv2.dilate(z, np.ones((2 * GLOW_PX + 1,) * 2, np.uint8)) > 0
    return m._zbig


def fill_by_result(a: Member, b: Member, fill: np.ndarray):
    """(median, p90) of the visible colour difference of the two CLEANED results, blurred, over the frame fill.
    None when it cannot be measured. Inpainting differs in detail, so only the colour of the fill is compared."""
    ca, cb = _cand_pixels(a), _cand_pixels(b)
    if ca is None or cb is None or ca[0].shape != cb[0].shape or fill.sum() < MIN_INTERIOR_PX:
        return None
    qa = ca[0].astype(np.float32) * ca[1][..., None] / 255.0
    qb = cb[0].astype(np.float32) * cb[1][..., None] / 255.0
    qa = cv2.GaussianBlur(qa, (0, 0), FILL_BLUR_SIGMA)
    qb = cv2.GaussianBlur(qb, (0, 0), FILL_BLUR_SIGMA)
    dq = np.abs(qa[fill] - qb[fill]).max(axis=1)
    return round(float(np.median(dq)), 1), round(float(np.percentile(dq, 90)), 1)


def gate_metrics(a: Member, b: Member) -> dict:
    """All numbers behind the same-frame decision (also used by the diagnose command)."""
    out = {"size_a": a.size, "size_b": b.size}
    if a.size != b.size:
        out["ok"] = False
        out["why"] = "size differs"
        return out
    ra, aa, za = _features(a)
    rb, ab, zb = _features(b)
    inside = aa > S.TILE_ALPHA_LEVEL
    use = inside & ~za & ~zb                                  # everything outside the changed text zones
    dist = cv2.distanceTransform(np.pad(inside, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    clean = ~_zone_big(a) & ~_zone_big(b)                     # away from text AND its glow (glow also spills over the outline)
    rim = inside & ~za & ~zb & (dist <= RIM_PX)               # outline band outside the text zones
    out["zone_a"] = float(za.mean())
    out["zone_b"] = float(zb.mean())

    # visible colour = RGB weighted by alpha: where a pixel is nearly transparent its raw RGB is irrelevant
    pa = (ra.astype(np.float32) * aa[..., None] / 255.0)
    pb = (rb.astype(np.float32) * ab[..., None] / 255.0)

    def agree(mask: np.ndarray) -> tuple[int, float, int]:
        if not mask.any():
            return 0, 0.0, 0
        d = np.abs(pa[mask] - pb[mask]).max(axis=1)
        return int(mask.sum()), float((d <= S.GATE_LEVEL).mean()), int(round(float(d.max())))

    same_a = np.abs(aa.astype(np.int16) - ab.astype(np.int16)) <= S.ALPHA_SAME_LEVEL
    out["alpha_same"] = float(same_a[clean].mean()) if clean.sum() >= 1000 else float(same_a.mean())
    out["alpha_same_all"] = float(same_a.mean())              # whole tile, for comparison
    # decision: the HARD shape must agree, soft edge values (anti-aliasing) may differ a little between sheets
    ins_any = inside | (ab > S.TILE_ALPHA_LEVEL)
    n_ins = max(int(ins_any.sum()), 1)
    adiff = np.abs(aa.astype(np.int16) - ab.astype(np.int16))
    out["alpha_core_mismatch"] = round(float(((aa >= 128) != (ab >= 128)).sum()) / n_ins, 4)
    out["alpha_mean_diff"] = round(float(adiff[ins_any].mean()), 2) if ins_any.any() else 0.0
    alpha_ok = out["alpha_core_mismatch"] <= ALPHA_CORE_MISMATCH_MAX and out["alpha_mean_diff"] <= ALPHA_MEAN_DIFF_MAX
    diff_a = ~same_a
    out["alpha_diff_px"] = int(diff_a.sum())
    if diff_a.any():
        ys, xs = np.nonzero(diff_a)
        out["alpha_diff_bbox"] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        out["alpha_diff_in_text_zone_pct"] = round(100.0 * float((diff_a & ~clean).sum()) / diff_a.sum(), 1)
        out["alpha_diff_max"] = int(np.abs(aa.astype(np.int16) - ab.astype(np.int16)).max())
    out["rim_px"], out["rim_agree"], out["rim_diff_max"] = agree(rim)
    out["rim_mean"] = round(float(np.abs(pa[rim] - pb[rim]).max(axis=1).mean()), 1) if rim.any() else None
    # robust: a glow tail touches SOME rim pixels, a real colour variant differs on (almost) all of them
    if rim.any():
        dr = np.abs(pa[rim] - pb[rim]).max(axis=1)
        out["rim_median"] = round(float(np.median(dr)), 1)
        out["rim_tail_frac"] = round(float((dr > RIM_TAIL_LEVEL).mean()), 3)
        # glow tails sit near the text, a colour variant differs ALONG the whole outline:
        # so judge the tail on rim pixels away from text + glow when enough of them exist
        rc = rim & clean
        out["rim_clean_px"] = int(rc.sum())
        if out["rim_clean_px"] >= MIN_RIM_CLEAN_PX:
            dc = np.abs(pa[rc] - pb[rc]).max(axis=1)
            out["rim_clean_tail_frac"] = round(float((dc > RIM_TAIL_LEVEL).mean()), 3)
            tail_ok = out["rim_clean_tail_frac"] <= RIM_CLEAN_TAIL_FRAC_MAX
        else:
            tail_ok = out["rim_tail_frac"] <= RIM_TAIL_FRAC_MAX
        rim_ok = out["rim_median"] <= RIM_MEDIAN_MAX and tail_ok
    else:
        rim_ok = False
    out["use_px"], out["agree"], out["diff_max"] = agree(use)   # info only: includes pixels the text glow reaches
    # interior: the fill colour of the frame, judged away from the text AND its glow, robust statistics
    interior = inside & (dist > RIM_PX) & clean
    out["interior_px"] = int(interior.sum())
    fill_unverifiable = False
    if out["interior_px"] >= MIN_INTERIOR_PX:
        di = np.abs(pa[interior] - pb[interior]).max(axis=1)
        out["interior_median"] = round(float(np.median(di)), 1)
        out["interior_p90"] = round(float(np.percentile(di, 90)), 1)
        interior_ok = out["interior_median"] <= INTERIOR_MEDIAN_MAX and out["interior_p90"] <= INTERIOR_P90_MAX
    else:
        # no clean interior left (text covers it): compare the CLEANED results instead, text is gone there.
        # Without this the fill colour was never checked for such cells.
        fr = fill_by_result(a, b, inside & (dist > RIM_PX))
        if fr is not None:
            out["fill_by_result_median"], out["fill_by_result_p90"] = fr
            interior_ok = (fr[0] <= FILL_BY_RESULT_MEDIAN_MAX and fr[1] <= FILL_BY_RESULT_P90_MAX)
        else:
            interior_ok = False                               # cannot verify the fill: do not merge
            fill_unverifiable = True
    out["ok"] = (alpha_ok and out["rim_px"] >= S.MIN_GATE_PIXELS
                 and rim_ok and interior_ok)
    out["why"] = "" if out["ok"] else ("alpha silhouette" if not alpha_ok else
                                       "too few rim pixels" if out["rim_px"] < S.MIN_GATE_PIXELS else
                                       "outline colours differ" if not rim_ok else
                                       "fill not verifiable" if fill_unverifiable else
                                       "fill colour differs")
    return out


def same_frame(a: Member, b: Member) -> bool:
    """Same cell shape and the same colours outside the text of both (the gate of the sibling bank)."""
    return bool(gate_metrics(a, b)["ok"])


MAX_FRAME_MEMBERS = 15               # cells used for a group's consensus frame (cleanest first)


def frame_member(cluster: list[Member]) -> Member:
    """Consensus frame of a group, as a pseudo cell that gate_metrics can compare like any other.
    Per pixel: the median over the cells in which that pixel is free of text AND glow (their texts sit in
    different places, so almost every pixel is clean in at least one cell). Pixels clean in no cell are 'unknown'
    (treated as text zone) and are filled from the cleaned results of the cells, only as a fallback guess."""
    mem = sorted(cluster, key=lambda m: (int(_features(m)[2].sum()), m.image, m.region))[:MAX_FRAME_MEMBERS]
    rgbs = np.stack([_features(m)[0] for m in mem]).astype(np.float32)          # (n, H, W, 3)
    alphas = np.stack([_features(m)[1] for m in mem]).astype(np.float32)        # (n, H, W)
    clean = np.stack([~_zone_big(m) for m in mem])                              # (n, H, W)
    known = clean.any(axis=0)
    masked = np.where(clean[..., None], rgbs, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(masked, axis=0)                                  # (H, W, 3), NaN where unknown
    cands = [_cand_pixels(m) for m in mem]
    cands = [c for c in cands if c is not None and c[0].shape == med.shape]
    if cands:
        guess = np.median(np.stack([c[0] for c in cands]).astype(np.float32), axis=0)
    else:
        guess = np.median(rgbs, axis=0)
    full = np.where(known[..., None], med, guess)
    alpha = np.median(alphas, axis=0).astype(np.uint8)
    rgb_u8 = np.clip(full, 0, 255).astype(np.uint8)
    first = mem[0]
    fm = Member(first.image, first.region, first.window, "cell", first.orig, {"FRAME": first.orig})
    fm._feat = (rgb_u8, alpha, ~known)
    fm._zbig = ~known
    fm._cand = (rgb_u8, alpha)
    return fm


def build_groups(members: list[Member]) -> list[list[Member]]:
    """1) cells join the first frame whose representative (cleanest cell) they match (no chains A~B~C),
    2) groups are merged when their CONSENSUS frames match: the consensus is built from all cells of a group,
    so it is known on (almost) every pixel, also where every single cell has text. Only cell-sized windows match."""
    ms = [m for m in members if not (m.source and m.source != "cell")]
    by_size: dict[tuple[int, int], list[Member]] = {}
    for m in ms:
        by_size.setdefault(m.size, []).append(m)
    groups: list[list[Member]] = []
    for bucket in by_size.values():
        bucket.sort(key=lambda m: (int(_features(m)[2].sum()), m.image, m.region))   # cleanest cell first
        clusters: list[list[Member]] = []
        for m in bucket:
            for cl in clusters:
                if same_frame(cl[0], m):
                    cl.append(m)
                    break
            else:
                clusters.append([m])
        # merge pass on consensus frames; pair results are cached and only redone for a changed group
        ids = list(range(len(clusters)))
        frames = {i: frame_member(clusters[i]) for i in ids}
        cache: dict[tuple[int, int], bool] = {}
        nxt = len(clusters)
        while True:
            hit = None
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    key = (ids[x], ids[y])
                    if key not in cache:
                        cache[key] = same_frame(frames[ids[x]], frames[ids[y]])
                    if cache[key]:
                        hit = (x, y)
                        break
                if hit:
                    break
            if hit is None:
                break
            x, y = hit
            ix, iy = ids[x], ids[y]
            merged = clusters[ix] + clusters[iy]
            clusters.append(merged)
            frames[nxt] = frame_member(merged)
            ids = [i for i in ids if i not in (ix, iy)] + [nxt]
            nxt += 1
        groups.extend(clusters[i] for i in ids)
    for g in groups:
        g.sort(key=lambda x: (x.image, x.region))
    return groups


def roughness(m: Member, method: str | None = None) -> float:
    """Automatic score of the chosen result: high-frequency energy in and around the changed text zone
    (lower = smoother = fewer smears). Only used to order candidates, the person decides."""
    rgb, alpha, zone = _features(m)
    res, _ = _rgba(m.cands[method or m.chosen])
    luma = res.astype(np.float32).mean(axis=-1)
    lap = np.abs(cv2.Laplacian(luma, cv2.CV_32F, ksize=3))
    area = cv2.dilate(zone.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    area &= alpha > S.TILE_ALPHA_LEVEL
    return float(lap[area].mean()) if area.any() else 1e9


# ----------------------------------------------------------------------------- ranker

PAGE = r"""<!doctype html><html lang="de"><meta charset="utf-8"><title>Donor-Ranker</title>
<style>
:root{color-scheme:dark;--bg:#16161d;--card:#1f1f2a;--ln:#33334a;--ac:#5b8cff;--ok:#3ecf8e;--bad:#e5484d;--tx:#e8e8f0;--mut:#9a9ab0}
body{margin:0;background:var(--bg);color:var(--tx);font:14px system-ui,sans-serif}
header{display:flex;gap:14px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--ln);flex-wrap:wrap}
header b{font-size:15px} .mut{color:var(--mut)} button{background:var(--card);color:var(--tx);border:1px solid var(--ln);border-radius:6px;padding:6px 12px;cursor:pointer}
button.pri{background:var(--ac);border-color:var(--ac)} button:hover{filter:brightness(1.15)}
#bar{height:6px;background:var(--card);flex:1;min-width:120px;border-radius:3px;overflow:hidden} #bar i{display:block;height:100%;background:var(--ok);width:0}
main{padding:14px 16px} h2{margin:0 0 4px;font-size:17px}
#grid{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
.card{background:var(--card);border:2px solid var(--ln);border-radius:8px;cursor:pointer;position:relative}
.card.sel{border-color:var(--ok);box-shadow:0 0 0 2px var(--ok) inset}
.card .k{position:absolute;left:6px;top:6px;background:#000a;border-radius:4px;padding:1px 6px;font-weight:700}
.card .cap{padding:5px 8px;font-size:12px;color:var(--mut);max-width:var(--w);line-height:1.35;word-break:break-all}.card .cap b{color:var(--tx)}
.card .img{background-color:#bbb;background-image:linear-gradient(45deg,#999 25%,transparent 25%,transparent 75%,#999 75%),linear-gradient(45deg,#999 25%,transparent 25%,transparent 75%,#999 75%);background-size:16px 16px;background-position:0 0,8px 8px;border-radius:6px 6px 0 0;overflow:hidden}
.card img{display:block;image-rendering:auto}
#help{margin-top:16px;color:var(--mut);font-size:12px;line-height:1.5}
</style>
<header>
 <b>Donor-Ranker</b><span id="pos" class="mut"></span><span id="bar"><i></i></span><span id="cnt" class="mut"></span>
 <label class="mut">Größe <input id="zoom" type="range" min="0.5" max="2.5" step="0.1" value="1"></label>
 <button id="imp">CSV importieren</button><button id="exp" class="pri">CSV exportieren</button><button id="rst">Zurücksetzen</button>
 <input id="file" type="file" accept=".csv" hidden>
</header>
<main>
 <h2 id="title"></h2><div id="sub" class="mut"></div>
 <div id="grid"></div>
 <div style="margin-top:14px;display:flex;gap:8px;flex-wrap:wrap">
  <button id="prev">← Zurück</button><button id="none">Keiner brauchbar (n)</button><button id="skip">Überspringen (s)</button>
  <button id="next" class="pri">Fertig / Weiter (Enter)</button>
 </div>
 <div id="help">Klick oder Taste 1–9, 0, q, w = beste Version dieses Rahmens (sie wird Donor für ALLE Zellen dieses Rahmens). Nochmal klicken = abwählen.
 Brauchbar heißt: kein Textrest, keine Schlieren, Verlauf und Kontur intakt. Die Kandidaten sind nach automatischem Glätte-Score vorsortiert (beste zuerst). Der Stand wird im Browser gesichert.</div>
</main>
<script>
const DATA=__DATA__;
const KEYS="1234567890qw";
const store="donor_ranker_"+DATA.id;
let st=JSON.parse(localStorage.getItem(store)||"{}"); // group id -> {status, file}
DATA.groups.forEach(g=>{if(g.prefill&&!st[g.id])st[g.id]={status:"pick",file:g.prefill,auto:true}});
let cur=0;
const $=id=>document.getElementById(id);
const save=()=>localStorage.setItem(store,JSON.stringify(st));
function done(){return Object.values(st).filter(v=>v.status&&v.status!=="open").length}
function firstOpen(){for(let i=0;i<DATA.groups.length;i++){const s=st[DATA.groups[i].id];if(!s||s.status==="open")return i}return 0}
function render(){
  const g=DATA.groups[cur];const s=st[g.id]||{status:"open"};const z=parseFloat($("zoom").value);
  $("pos").textContent=`Rahmen ${cur+1} / ${DATA.groups.length}`;
  $("cnt").textContent=`${done()} entschieden`;$("bar").firstElementChild.style.width=(100*done()/DATA.groups.length)+"%";
  $("title").textContent=`Rahmen ${g.id} · ${g.size[0]}×${g.size[1]}`;
  $("sub").textContent=`${g.members_total} Zellen mit diesem Rahmen in allen Bildern · ${g.candidates.length} von ${g.pool_total} brauchbaren Ergebnissen (★ = in der normalen Wertung gewählt)`+(s.auto?" · aus deiner alten Abstimmung übernommen (zum Ändern andere Kachel wählen)":"")+(s.status==="none"?" · Status: keiner brauchbar":s.status==="skip"?" · Status: übersprungen":"");
  const grid=$("grid");grid.innerHTML="";
  const mk=(file,label,key,sel,onclick)=>{
    const c=document.createElement("div");c.className="card"+(sel?" sel":"");c.style.setProperty("--w",(g.size[0]*z)+"px");
    c.innerHTML=`<div class="img"><img src="${file}" width="${Math.round(g.size[0]*z)}"></div><div class="cap">${label}</div>`+(key?`<span class="k">${key}</span>`:"");
    if(onclick)c.onclick=onclick;grid.appendChild(c)};
  mk(g.original,"<b>Original</b><br>(Beispiel)","",false,null);
  g.candidates.forEach((c,i)=>mk(c.file,`${c.image.replace(".png","")} · r${c.region}<br><b>${c.method}</b>${c.was_chosen?" ★":""}`,KEYS[i]||"",s.file===c.file,()=>pick(i)));
}
function pick(i){const g=DATA.groups[cur];const c=g.candidates[i];if(!c)return;const s=st[g.id]||{};
  st[g.id]=(s.file===c.file)?{status:"open"}:{status:"pick",file:c.file};save();render()}
function setStatus(x){const g=DATA.groups[cur];st[g.id]={status:x};save();go(1)}
function go(d){cur=Math.max(0,Math.min(DATA.groups.length-1,cur+d));render()}
function parseLine(l){const out=[];let cur="",q=false;
  for(let i=0;i<l.length;i++){const ch=l[i];
    if(q){if(ch==='"'){if(l[i+1]==='"'){cur+='"';i++}else q=false}else cur+=ch}
    else if(ch==='"')q=true;else if(ch===","){out.push(cur);cur=""}else cur+=ch}
  out.push(cur);return out}
function csv(){const rows=[["group","status","chosen_file","image","region","method","members_total","candidates"]];
  DATA.groups.forEach(g=>{const s=st[g.id];if(!s||s.status==="open")return;const c=g.candidates.find(c=>c.file===s.file)||{};
    rows.push([g.id,s.status,s.file||"",c.image||"",c.region||"",c.method||"",g.members_total,g.candidates.length])});
  return "\ufeff"+rows.map(r=>r.map(v=>`"${String(v).replace(/"/g,'""')}"`).join(",")).join("\n")}
$("exp").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv()],{type:"text/csv"}));a.download="donor_choices.csv";a.click()};
$("imp").onclick=()=>$("file").click();
$("file").onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(t=>{
  t.replace(/^\ufeff/,"").split(/\r?\n/).slice(1).filter(Boolean).forEach(l=>{const p=parseLine(l);
    if(p[0])st[p[0]]={status:p[1],file:p[2]||undefined}});save();cur=firstOpen();render()})};
$("rst").onclick=()=>{if(confirm("Alle Entscheidungen löschen?")){st={};save();cur=0;render()}};
$("prev").onclick=()=>go(-1);$("next").onclick=()=>go(1);$("none").onclick=()=>setStatus("none");$("skip").onclick=()=>setStatus("skip");
$("zoom").oninput=render;
addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;
  if(e.key==="Enter")go(1);else if(e.key==="ArrowRight")go(1);else if(e.key==="ArrowLeft")go(-1);
  else if(e.key==="n")setStatus("none");else if(e.key==="s")setStatus("skip");
  else{const i=KEYS.indexOf(e.key.toLowerCase());if(i>=0)pick(i)}});
cur=firstOpen();render();
</script></html>"""


def build_ranker(run: Path, ranking_csv: Path, out: Path | None = None, max_candidates: int = MAX_CANDIDATES,
                 prefill_csv: Path | None = None) -> Path:
    import hashlib
    members = load_members(run, ranking_csv)
    groups = build_groups(members)
    old_picks: set[tuple[str, int, str]] = set()          # earlier votes (image, region, method) to carry over
    if prefill_csv is not None:
        with prefill_csv.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "pick" and row.get("image"):
                    old_picks.add((row["image"], int(row["region"]), row["method"]))
    out = out or (run / "donor_ranker")
    if out.exists():
        shutil.rmtree(out)
    (out / "tiles").mkdir(parents=True)
    entries = []
    skipped_no_cand = skipped_single = 0
    gid = 0
    for g in sorted(groups, key=lambda g: (-len(g), g[0].image, g[0].region)):
        if len(g) < 2:
            skipped_single += 1
            continue
        pool = [(m, meth) for m in g for meth in m.usable]      # EVERY usable result of EVERY cell of the frame
        if not pool:
            skipped_no_cand += 1
            continue
        scored = sorted(pool, key=lambda p: roughness(p[0], p[1]))
        gid += 1
        cands = []
        for i, (m, meth) in enumerate(scored[:max_candidates]):
            fn = f"tiles/g{gid:03d}_c{i:02d}.png"
            shutil.copyfile(m.cands[meth], out / fn)
            cands.append({"file": fn, "image": m.image, "region": m.region, "method": meth,
                          "was_chosen": meth == m.chosen, "window": list(m.window)})
        orig_fn = f"tiles/g{gid:03d}_orig.png"
        shutil.copyfile(g[0].orig, out / orig_fn)
        pre = next((c["file"] for c in cands if (c["image"], c["region"], c["method"]) in old_picks), None)
        entries.append({"id": f"{gid:03d}", "size": list(g[0].size), "members_total": len(g), "prefill": pre,
                        "pool_total": len(pool), "original": orig_fn, "candidates": cands,
                        "members": [f"{m.image}#r{m.region}" for m in g]})
    # the id keys the browser's saved decisions: a different grouping must not reuse old group numbers
    sig = hashlib.sha1(json.dumps([e["members"] for e in entries]).encode()).hexdigest()[:8]
    data = {"id": f"{run.name}_{sig}", "groups": entries}
    (out / "donor_groups.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (out / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
    sizes = sorted((len(g) for g in groups if len(g) >= 2), reverse=True)
    print(f"{len(members)} Regionen, {len(groups)} Rahmengruppen ({len(entries)} im Ranker, {skipped_single} Einzelzellen, "
          f"{skipped_no_cand} ohne brauchbares Ergebnis)")
    if prefill_csv is not None:
        n_pre = sum(1 for e in entries if e["prefill"])
        print(f"Aus der alten Abstimmung uebernommen: {n_pre} von {len(entries)} Gruppen vorgewaehlt, "
              f"{len(entries) - n_pre} muessen neu entschieden werden")
    print(f"Gruppengroessen (Zellen): {sizes[:25]}{' ...' if len(sizes) > 25 else ''}")
    print(f"Ranker: {out / 'index.html'}")
    return out / "index.html"


def near_report(run: Path, ranking_csv: Path | None, top: int, dump: Path | None) -> None:
    """The closest pairs of frames that were kept APART: if they look the same to you, the gate is too strict for
    them and the numbers show by how much. Compares the representatives (cleanest cell) of every pair of frames."""
    members = load_members(run, ranking_csv)
    groups = build_groups(members)
    by_size: dict[tuple[int, int], list[list[Member]]] = {}
    for g in groups:
        by_size.setdefault(g[0].size, []).append(g)
    pairs = []
    for gl in by_size.values():
        gl.sort(key=lambda g: int(_features(g[0])[2].sum()))
        fr = [frame_member(g) for g in gl]
        for i in range(len(gl)):
            for j in range(i + 1, len(gl)):
                m = gate_metrics(fr[i], fr[j])
                if m.get("ok") or m.get("rim_mean") is None:
                    continue
                score = max(m["rim_mean"], m.get("interior_median", 0.0))
                pairs.append((score, m, gl[i], gl[j]))
    pairs.sort(key=lambda p: p[0])
    print(f"{len(groups)} Gruppen; {len(pairs)} getrennte Paare gleicher Zellgroesse. Die {min(top, len(pairs))} naechsten:")
    for k, (score, m, ga, gb) in enumerate(pairs[:top], 1):
        ra_, rb_ = ga[0], gb[0]
        print(f"{k:>2}. {ra_.image[:-4]}:{ra_.region} ({len(ga)} Zellen) | {rb_.image[:-4]}:{rb_.region} ({len(gb)} Zellen)  "
              f"Rand-Abstand mittel {m['rim_mean']} (median {m.get('rim_median')}, Schweif {m.get('rim_tail_frac')}, fern vom Text {m.get('rim_clean_tail_frac')} bei {m.get('rim_clean_px')} px), Fuellung median {m.get('interior_median', '?')} p90 {m.get('interior_p90', '?')}, Fuellung (Ergebnisbild, weichgezeichnet) median {m.get('fill_by_result_median', '-')} p90 {m.get('fill_by_result_p90', '-')}, "
              f"Alpha: Kern-Abweichung {m.get('alpha_core_mismatch')}, mittlere Diff {m.get('alpha_mean_diff')}, Grund: {m['why']}")
        if dump is not None:
            dump.mkdir(parents=True, exist_ok=True)
            ra, aa, _ = _features(frame_member(ga))
            rb, ab, _ = _features(frame_member(gb))
            bg = np.full(ra.shape, 128, np.float32)
            comp = lambda r, al: (r * (al[..., None] / 255.0) + bg * (1 - al[..., None] / 255.0)).astype(np.uint8)
            diff = np.clip(np.abs(comp(ra, aa).astype(np.int16) - comp(rb, ab).astype(np.int16)) * 5, 0, 255).astype(np.uint8)
            Image.fromarray(np.hstack([comp(ra, aa), comp(rb, ab), diff])).save(
                dump / f"near_{k:02d}_{ra_.image[:-4]}-{ra_.region}__{rb_.image[:-4]}-{rb_.region}.png")
    if dump is not None:
        print(f"Bilder (links A, Mitte B, rechts Differenz x5): {dump}")


def calibrate(run: Path, ranking_csv: Path | None, per_group: int = 8) -> None:
    """How big is the result-vs-result colour difference for cells that are CERTAINLY the same frame?
    Pairs inside a group that were confirmed on real original pixels (clean interior) are the reference; their
    result difference is pure inpainting noise. A threshold above that noise separates real fill differences."""
    members = load_members(run, ranking_csv)
    groups = build_groups(members)
    meds, p90s, checked = [], [], 0
    for g in groups:
        if len(g) < 2:
            continue
        sel = sorted(g, key=lambda m: int(_features(m)[2].sum()))[:per_group]
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                m = gate_metrics(sel[i], sel[j])
                checked += 1
                if not m.get("ok") or m.get("interior_px", 0) < MIN_INTERIOR_PX:
                    continue                                   # only pairs verified on original pixels
                ra, aa, _ = _features(sel[i])
                inside = aa > S.TILE_ALPHA_LEVEL
                dist = cv2.distanceTransform(np.pad(inside, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
                fr = fill_by_result(sel[i], sel[j], inside & (dist > RIM_PX))
                if fr:
                    meds.append(fr[0])
                    p90s.append(fr[1])
    print(f"{len(groups)} Gruppen, {checked} Zellpaare geprueft, {len(meds)} davon auf Originalpixeln als gleich bestaetigt")
    if not meds:
        print("Keine Referenzpaare gefunden.")
        return
    for name, v in (("median", meds), ("p90", p90s)):
        v = np.array(v)
        print(f"  Ergebnisbild-Abstand {name}: p50 {np.percentile(v, 50):.1f}  p90 {np.percentile(v, 90):.1f}  "
              f"p99 {np.percentile(v, 99):.1f}  max {v.max():.1f}")
    rec_m = max(6.0, float(np.ceil(np.percentile(meds, 99) * 1.2)))          # never stricter than the default
    rec_p = max(20.0, float(np.ceil(max(np.percentile(p90s, 99) * 1.2, rec_m + 10))))
    print(f"Empfehlung: --fill-median-max {rec_m:g} --fill-p90-max {rec_p:g}   (aktuell {FILL_BY_RESULT_MEDIAN_MAX:g} / {FILL_BY_RESULT_P90_MAX:g})")


def diagnose(run: Path, ranking_csv: Path | None, a: str, b: str, dump: Path | None = None) -> None:
    """Print why two cells ('m003_007:13' = image:region) are, or are not, one frame."""
    members = {f"{m.image.replace('.png', '')}:{m.region}": m for m in load_members(run, ranking_csv)}
    for k in (a, b):
        if k not in members:
            raise SystemExit(f"unbekannt: {k} (Beispiel: {next(iter(members))})")
    ma, mb = members[a], members[b]
    print(f"{a}: window_source={ma.source!r} size={ma.size}   {b}: window_source={mb.source!r} size={mb.size}")
    for k, v in gate_metrics(ma, mb).items():
        print(f"  {k}: {v}")
    ra, aa, za = _features(ma)
    rb, ab, zb = _features(mb)
    if ma.size == mb.size:
        use = (aa > S.TILE_ALPHA_LEVEL) & ~za & ~zb
        if use.any():
            diff = ra.astype(np.int16) - rb.astype(np.int16)
            print(f"  Mittelwert RGB auf gemeinsamen Pixeln: a={ra[use].mean(axis=0).round(1).tolist()} "
                  f"b={rb[use].mean(axis=0).round(1).tolist()}")
            print(f"  mittlere Differenz a-b je Kanal (R,G,B): {diff[use].mean(axis=0).round(1).tolist()}")
            ys, xs = np.nonzero(use)
            print(f"  gemeinsame Pixel liegen in x={xs.min()}..{xs.max()}, y={ys.min()}..{ys.max()}")
            dm = np.abs(diff).max(axis=-1)
            dv = np.abs(ra.astype(np.float32) * aa[..., None] / 255 - rb.astype(np.float32) * ab[..., None] / 255).max(axis=-1)
            print("  Abstand roh (RGB) | sichtbar (alpha-gewichtet), auf gemeinsamen Pixeln:")
            for lo, hi in ((0, 3), (4, 10), (11, 30), (31, 255)):
                sel = (dm[use] >= lo) & (dm[use] <= hi)
                selv = (dv[use] >= lo) & (dv[use] <= hi)
                print(f"    {lo:>2}-{hi:<3}: {100 * sel.mean():5.1f} % | {100 * selv.mean():5.1f} %")
            print("  nach Alpha-Bereich (Pixel, roher Abstand mittel, sichtbarer Abstand mittel, Mittelwert Alpha-Abweichung):")
            for lo, hi in ((16, 64), (65, 192), (193, 255)):
                sel = use & (aa >= lo) & (aa <= hi)
                if sel.any():
                    print(f"    alpha {lo:>3}-{hi:<3}: {int(sel.sum()):6d} px | roh {dm[sel].mean():5.1f} | sichtbar {dv[sel].mean():5.1f}")
        if dump is not None:
            dump.mkdir(parents=True, exist_ok=True)
            vis = np.clip(np.abs(ra.astype(np.int16) - rb.astype(np.int16)) * 5, 0, 255).astype(np.uint8)
            vis[~use] = 0
            for name, arr in (("a_orig", ra), ("b_orig", rb), ("diff_x5_nur_gemeinsame_pixel", vis),
                              ("gemeinsame_pixel_maske", (use * 255).astype(np.uint8))):
                Image.fromarray(arr).save(dump / f"{a.replace(':', '_')}__{b.replace(':', '_')}__{name}.png")
            print(f"  Bilder: {dump}")
    print(f"  Schwellen: ALPHA_SAME_LEVEL={S.ALPHA_SAME_LEVEL} ALPHA_SAME_FRAC={S.ALPHA_SAME_FRAC} GATE_LEVEL={S.GATE_LEVEL} "
          f"GATE_AGREE_MIN={S.GATE_AGREE_MIN} MIN_GATE_PIXELS={S.MIN_GATE_PIXELS} TILE_ALPHA_LEVEL={S.TILE_ALPHA_LEVEL}")


# ----------------------------------------------------------------------------- approve

def approve(ranker_dir: Path, choices_csv: Path, bank_dir: Path) -> tuple[int, int]:
    """Put every picked frame into the bank as an approved entry. Returns (approved, refused)."""
    bank = S.SiblingBank(bank_dir)
    ok = bad = 0
    # which cells each approved donor is meant for: the members of its group (from the ranker's donor_groups.json)
    members_of: dict[str, list[str]] = {}
    gj = ranker_dir / "donor_groups.json"
    if gj.is_file():
        data = json.loads(gj.read_text(encoding="utf-8"))
        members_of = {g["id"]: g.get("members", []) for g in data.get("groups", [])}
    donor_members: dict[str, dict] = {}
    with choices_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "pick" or not row.get("chosen_file"):
                continue
            path = ranker_dir / row["chosen_file"]
            if not path.is_file():
                print(f"  fehlt: {path}")
                bad += 1
                continue
            rgb, alpha = _rgba(path)
            tiles = S.find_tiles(alpha)
            if not tiles:
                print(f"  kein Sprite erkannt: {path.name}")
                bad += 1
                continue
            t = max(tiles, key=lambda r: S._size(r)[0] * S._size(r)[1])          # the sprite of the cell
            key = f"approved:{row['image']}#r{row['region']}:{row['method']}"
            bank.approve(S._crop(alpha, t), S._crop(rgb, t), key)
            gid = f"{int(row['group']):03d}" if str(row.get("group", "")).isdigit() else str(row.get("group", ""))
            donor_members[key] = {"group": gid, "donor": f"{row['image']}#r{row['region']}", "tile": row["chosen_file"],
                                  "members": members_of.get(gid, [])}
            ok += 1
    new = bank.save()
    if donor_members:
        out = Path(bank_dir) / "frame_donor_members.json"
        out.write_text(json.dumps(donor_members, indent=1), encoding="utf-8")
        n_cells = sum(len(v["members"]) for v in donor_members.values())
        print(f"Zuordnung Donor -> Zellen der Gruppe: {out}  ({len(donor_members)} Donoren, {n_cells} Zellen)")
    print(f"{ok} Rahmen freigegeben ({new} neue Eintraege, der Rest ersetzt bestehende), {bad} uebersprungen -> {bank.path}")
    return ok, bad


def main(argv: list[str] | None = None) -> int:
    global RIM_MEDIAN_MAX, FILL_BY_RESULT_MEDIAN_MAX, FILL_BY_RESULT_P90_MAX
    p = argparse.ArgumentParser(description="Donor ranker: best clean version per frame")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", type=Path, required=True, help="Stage 2 run folder (with debug_candidates/regions)")
    b.add_argument("--ranking", type=Path, required=True, help="Ranking CSV exported from the normal ranker")
    b.add_argument("--out", type=Path, default=None)
    b.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    b.add_argument("--rim-median-max", type=float, default=None,
                   help=f"max median colour difference along the outline for the same frame (default {RIM_MEDIAN_MAX})")
    b.add_argument("--prefill", type=Path, default=None,
                   help="old donor_choices.csv: picks that are still a candidate of the new group are preselected")
    b.add_argument("--gate-level", type=float, default=None,
                   help="max visible colour difference (0-255) along the outline to still count as the same frame "
                        f"(default {S.GATE_LEVEL}); a higher value merges faint brightness variants")
    c = sub.add_parser("calibrate", help="noise level of the result comparison for certainly-equal frames")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--ranking", type=Path, default=None)
    for p_ in (b,):
        p_.add_argument("--fill-median-max", type=float, default=None, help="max median fill-colour difference of the cleaned results")
        p_.add_argument("--fill-p90-max", type=float, default=None)
    n = sub.add_parser("near", help="closest pairs of frames kept apart (are they really different?)")
    n.add_argument("--run", type=Path, required=True)
    n.add_argument("--ranking", type=Path, default=None)
    n.add_argument("--top", type=int, default=20)
    n.add_argument("--gate-level", type=float, default=None)
    n.add_argument("--rim-median-max", type=float, default=None)
    n.add_argument("--fill-median-max", type=float, default=None)
    n.add_argument("--fill-p90-max", type=float, default=None)
    n.add_argument("--dump", type=Path, default=None, help="folder for side-by-side images")
    d = sub.add_parser("diagnose", help="why are two cells (image:region) one frame or not")
    d.add_argument("--run", type=Path, required=True)
    d.add_argument("--ranking", type=Path, default=None)
    d.add_argument("--gate-level", type=float, default=None, help="as in build")
    d.add_argument("--rim-median-max", type=float, default=None, help="as in build")
    d.add_argument("--fill-median-max", type=float, default=None)
    d.add_argument("--fill-p90-max", type=float, default=None)
    d.add_argument("--dump", type=Path, default=None, help="folder for diff images")
    d.add_argument("a", help="e.g. m003_007:13")
    d.add_argument("b", help="e.g. m003_004:12")
    a = sub.add_parser("approve")
    a.add_argument("--ranker", type=Path, required=True, help="donor_ranker folder")
    a.add_argument("--choices", type=Path, required=True, help="donor_choices.csv exported from the donor ranker")
    a.add_argument("--bank", type=Path, required=True)
    args = p.parse_args(argv)
    if getattr(args, "fill_median_max", None) is not None:
        FILL_BY_RESULT_MEDIAN_MAX = args.fill_median_max
        print(f"FILL_BY_RESULT_MEDIAN_MAX ueberschrieben: {FILL_BY_RESULT_MEDIAN_MAX}")
    if getattr(args, "fill_p90_max", None) is not None:
        FILL_BY_RESULT_P90_MAX = args.fill_p90_max
        print(f"FILL_BY_RESULT_P90_MAX ueberschrieben: {FILL_BY_RESULT_P90_MAX}")
    if getattr(args, "rim_median_max", None) is not None:
        RIM_MEDIAN_MAX = args.rim_median_max
        print(f"RIM_MEDIAN_MAX ueberschrieben: {RIM_MEDIAN_MAX}")
    if getattr(args, "gate_level", None) is not None:
        S.GATE_LEVEL = args.gate_level
        print(f"GATE_LEVEL ueberschrieben: {S.GATE_LEVEL}")
    if args.cmd == "build":
        build_ranker(args.run.expanduser(), args.ranking.expanduser(), args.out, args.max_candidates,
                     args.prefill.expanduser() if args.prefill else None)
    elif args.cmd == "calibrate":
        calibrate(args.run.expanduser(), args.ranking.expanduser() if args.ranking else None)
    elif args.cmd == "near":
        near_report(args.run.expanduser(), args.ranking.expanduser() if args.ranking else None, args.top,
                    args.dump.expanduser() if args.dump else None)
    elif args.cmd == "diagnose":
        diagnose(args.run.expanduser(), args.ranking.expanduser() if args.ranking else None, args.a, args.b,
                 args.dump.expanduser() if args.dump else None)
    else:
        approve(args.ranker.expanduser(), args.choices.expanduser(), args.bank.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
