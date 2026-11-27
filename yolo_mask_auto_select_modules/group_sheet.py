"""group_sheet: Seite-an-Seite-Ansicht aller Rahmengruppen, um uebersehene Zusammengehoerigkeit zu finden.

Importiert Gruppierung und Vergleich aus frame_donors (gleiche Logik, gleiche Schwellen wie `build`) und schreibt
einen Ordner mit index.html:
  A  alle Gruppen als Konsensus-Rahmen, nach Zellgroesse gruppiert (gleich grosse Rahmen stehen nebeneinander)
  B  die naechsten getrennten Paare gleicher Groesse: A | B | Differenz x5 samt Messwerten
  C  Gruppen mit fast gleicher Zellgroesse (+-Toleranz), die nie verglichen werden

    python -m yolo_mask_auto_select_modules.group_sheet --run RUN [--top 60] [--out DIR]
"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from . import frame_donors as FD
except ImportError:
    import frame_donors as FD
S = FD.S


def _comp(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32) / 255.0
    return (rgb.astype(np.float32) * a + 128.0 * (1 - a)).astype(np.uint8)


def _fit(fa, fb):
    """Wie weit sind zwei Rahmen nach einem Gain/Offset je Kanal noch voneinander entfernt? Nur Pixel, die in
    beiden gemessen sind (nicht geraten). Ergebnis: (Median vorher, Median nachher, p90 nachher, Gains, Offsets, Pixel)."""
    ra, aa, za = FD._features(fa)
    rb, ab, zb = FD._features(fb)
    m = (aa > S.TILE_ALPHA_LEVEL) & (ab > S.TILE_ALPHA_LEVEL) & ~za & ~zb
    if int(m.sum()) < 500:
        return None
    x = (ra.astype(np.float32) * aa[..., None] / 255.0)[m]
    y = (rb.astype(np.float32) * ab[..., None] / 255.0)[m]
    if len(x) > 30000:
        sel = np.random.default_rng(0).choice(len(x), 30000, replace=False)
        x, y = x[sel], y[sel]
    raw = np.abs(x - y).max(axis=1)
    res, gains, offs = [], [], []
    for c in range(3):
        A = np.stack([x[:, c], np.ones(len(x), np.float32)], axis=1)
        sol = np.linalg.lstsq(A, y[:, c], rcond=None)[0]
        res.append(np.abs(A @ sol - y[:, c]))
        gains.append(round(float(sol[0]), 2))
        offs.append(round(float(sol[1]), 1))
    r = np.max(res, axis=0)
    return (round(float(np.median(raw)), 1), round(float(np.median(r)), 1), round(float(np.percentile(r, 90)), 1),
            gains, offs, int(m.sum()))


def _fit_merge(groups, res_med: float, res_p90: float, raw_max: float = 20.0, off_max: float = 15.0, min_px: int = 2000):
    """Vorschau: Gruppen gleicher Groesse zusammenfuehren, die sich nur durch eine Farbverschiebung unterscheiden.
    Bedingungen (alle): gleiche Alpha-Form, Abstand ROH <= raw_max (sonst waere es eine andere Farbe: ein rotes und
    ein blaues Gitter liessen sich sonst per Gain/Offset 'angleichen'), nach Gain/Offset Median <= res_med und
    p90 <= res_p90, |Offset| <= off_max, genug gemessene Pixel. Nichts davon aendert Dateien der Pipeline."""
    frames = [FD.frame_member(g) for g in groups]
    parent = list(range(len(groups)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_size: dict[tuple[int, int], list[int]] = {}
    for k, g in enumerate(groups):
        by_size.setdefault(g[0].size, []).append(k)
    n_pairs = 0
    for idxs in by_size.values():
        for i, j in combinations(idxs, 2):
            m = FD.gate_metrics(frames[i], frames[j])
            if m.get("alpha_core_mismatch") is None:
                continue
            if m["alpha_core_mismatch"] > FD.ALPHA_CORE_MISMATCH_MAX or m["alpha_mean_diff"] > FD.ALPHA_MEAN_DIFF_MAX:
                continue
            ft = _fit(frames[i], frames[j])
            if m.get("ok") or (ft is not None and ft[5] >= min_px and ft[0] <= raw_max and ft[1] <= res_med
                               and ft[2] <= res_p90 and max(abs(o) for o in ft[4]) <= off_max):
                parent[find(i)] = find(j)
                n_pairs += 1
    clusters: dict[int, list] = {}
    for k, g in enumerate(groups):
        clusters.setdefault(find(k), []).extend(g)
    merged = []
    for root, cells in clusters.items():
        cells.sort(key=lambda x: (x.image, x.region))
        merged.append(cells)
    sizes = sorted(((len(c), c[0].size) for c in merged if len(c) > 0), reverse=True)
    joined = [[len(groups[k]) for k in range(len(groups)) if find(k) == r] for r in clusters if sum(find(k) == r for k in range(len(groups))) > 1]
    print(f"Farbfit-Vorschau: {len(groups)} Gruppen -> {len(merged)} Gruppen ({n_pairs} Paare erfuellen die Bedingungen)")
    for js in sorted(joined, key=lambda j: -sum(j))[:25]:
        print(f"   zusammengefuehrt: {'+'.join(map(str, sorted(js, reverse=True)))} = {sum(js)} Zellen")
    return merged


def build(run: Path, ranking: Path | None, ranker: Path | None, out: Path, top: int, tol: int,
          fit_merge: bool = False, fit_res_median: float = 2.5, fit_res_p90: float = 6.0) -> Path:
    members = FD.load_members(run, ranking)
    groups = FD.build_groups(members)
    if fit_merge:
        groups = _fit_merge(groups, fit_res_median, fit_res_p90)
    groups.sort(key=lambda g: (g[0].size, -len(g), g[0].image, g[0].region))
    print(f"{len(members)} Regionen -> {len(groups)} Gruppen, berechne Konsensus-Rahmen ...")
    frames = [FD.frame_member(g) for g in groups]

    # ids wie im Ranker (donor_groups.json), damit du die Gruppen dort wiederfindest
    ids: dict[frozenset, str] = {}
    gj = (ranker or run / "donor_ranker") / "donor_groups.json"
    if gj.is_file():
        entries = json.loads(gj.read_text(encoding="utf-8")).get("groups", [])
        ids = {frozenset(e["members"]): e["id"] for e in entries}
    sig = [frozenset(f"{m.image}#r{m.region}" for m in g) for g in groups]
    found = sum(1 for s in sig if s in ids)
    if ids:
        print(f"Ranker-Ids: {found} von {len(ids)} Ranker-Gruppen identisch wiedergefunden"
              + ("" if found == len(ids) else "  <-- Abweichung: Schwellen/Version anders als beim damaligen build"))
    else:
        print(f"Keine donor_groups.json gefunden ({gj}), Gruppen werden nur mit Nummer G# benannt")
    label = [("R" + ids[s]) if s in ids else (f"E{k}" if len(g) == 1 else f"G{k}")
             for k, (s, g) in enumerate(zip(sig, groups))]

    (out / "img").mkdir(parents=True, exist_ok=True)
    for old in (out / "img").glob("*.png"):
        old.unlink()
    unk: list[float] = []                  # Anteil der Pixel, die in KEINER Zelle frei von Text+Glow sind (geraten, nicht gemessen)
    for k, fm in enumerate(frames):
        rgb, alpha, z = FD._features(fm)
        Image.fromarray(np.dstack([rgb, alpha])).save(out / "img" / f"g{k:03d}.png")
        tint = rgb.astype(np.float32)
        tint[z] = 0.4 * tint[z] + 0.6 * np.array([255, 0, 255], np.float32)
        Image.fromarray(np.dstack([np.clip(tint, 0, 255).astype(np.uint8), alpha])).save(out / "img" / f"g{k:03d}u.png")
        vis = alpha > S.TILE_ALPHA_LEVEL
        unk.append(float(z[vis].mean()) if vis.any() else 0.0)

    by_size: dict[tuple[int, int], list[int]] = {}
    for k, g in enumerate(groups):
        by_size.setdefault(g[0].size, []).append(k)

    pairs, no_rim = [], 0
    for idxs in by_size.values():
        for i, j in combinations(idxs, 2):
            m = FD.gate_metrics(frames[i], frames[j])
            if m.get("rim_mean") is None:
                no_rim += 1
                continue
            inner = m.get("interior_median")
            if inner is None:
                inner = m.get("fill_by_result_median", 0.0)
            sc = max(m["rim_mean"], inner)
            if m.get("why") == "too few rim pixels":
                sc += 1000.0                                  # Rand nicht beurteilbar: ans Ende
            pairs.append((sc, i, j, m))
    pairs.sort(key=lambda p: p[0])
    fits = {(i, j): _fit(frames[i], frames[j]) for _, i, j, _ in pairs[:max(top, 20)]}
    print(f"Grund der Trennung bei den {min(top, len(pairs))} naechsten Paaren: "
          f"{dict(Counter(str(p[3].get('why', '')) for p in pairs[:top]))}")
    for n, (score, i, j, m) in enumerate(pairs[:20], 1):
        print(f"{n:>2}. {label[i]} ({len(groups[i])}) | {label[j]} ({len(groups[j])})  "
              f"{groups[i][0].size[0]}x{groups[i][0].size[1]}  Score {score % 1000:.1f}  Grund: {m.get('why')}  "
              f"rim_px {m.get('rim_px')} int_px {m.get('interior_px')} unbekannt {unk[i]:.0%}/{unk[j]:.0%}  "
              f"rim_med {m.get('rim_median')}  tail {m.get('rim_tail_frac')}/{m.get('rim_clean_tail_frac')}  "
              f"int_med {m.get('interior_median', '-')} p90 {m.get('interior_p90', '-')}  "
              f"fill_res {m.get('fill_by_result_median', '-')}/{m.get('fill_by_result_p90', '-')}  "
              f"alpha {m.get('alpha_core_mismatch')}/{m.get('alpha_mean_diff')}"
              + (lambda ft: f"  FARBFIT Median {ft[0]}->{ft[1]} p90 {ft[2]} gain {ft[3]} off {ft[4]}" if ft else "  FARBFIT -")(fits.get((i, j))))
    for n, (score, i, j, m) in enumerate(pairs[:top], 1):
        ra, aa, _ = FD._features(frames[i])
        rb, ab, _ = FD._features(frames[j])
        ca, cb = _comp(ra, aa), _comp(rb, ab)
        diff = np.clip(np.abs(ca.astype(np.int16) - cb.astype(np.int16)) * 5, 0, 255).astype(np.uint8)
        both = ~FD._features(frames[i])[2] & ~FD._features(frames[j])[2]
        diff[~both] = (70, 0, 70)                              # violett = in A oder B geraten, nicht vergleichbar
        gap = np.full((ca.shape[0], 6, 3), 40, np.uint8)
        Image.fromarray(np.hstack([ca, gap, cb, gap, diff])).save(out / "img" / f"p{n:03d}.png")

    def card(k: int) -> str:
        g, w = groups[k], groups[k][0].size[0]
        ex = ", ".join(f"{m.image.replace('.png', '')}:{m.region}" for m in g[:3]) + (" ..." if len(g) > 3 else "")
        return (f'<div class="card" id="g{k}" data-k="{label[k]} (#{k})" onclick="mark(this)">'
                f'<img src="img/g{k:03d}u.png" data-plain="img/g{k:03d}.png" data-tint="img/g{k:03d}u.png" data-w="{w}" loading="lazy">'
                f'<div class="cap"><b>{label[k]}</b> · unbekannt {unk[k]:.0%} · '
                f'{len(g)} Zelle{"n" if len(g) != 1 else ""}<br>{html.escape(ex)}</div></div>')

    A = []
    for (w, h), idxs in by_size.items():
        A.append(f'<h3 id="s{w}x{h}">{w}×{h} <span class="mut">· {len(idxs)} Gruppe{"n" if len(idxs) != 1 else ""}, '
                 f'{sum(len(groups[k]) for k in idxs)} Zellen</span></h3><div class="row">'
                 + "".join(card(k) for k in idxs) + "</div>")

    B = []
    for n, (score, i, j, m) in enumerate(pairs[:top], 1):
        w = groups[i][0].size[0] * 3 + 12
        flag = ' <b class="bad">würde mergen!</b>' if m.get("ok") else ""
        B.append(
            f'<div class="pair"><div><a href="#g{i}">{label[i]}</a> ({len(groups[i])} Zellen) | '
            f'<a href="#g{j}">{label[j]}</a> ({len(groups[j])} Zellen) · {groups[i][0].size[0]}×{groups[i][0].size[1]} · '
            f'{"Score %.1f" % score if score < 1000 else "Rand nicht beurteilbar"}{flag}</div><img src="img/p{n:03d}.png" data-w="{w}">'
            f'<div class="cap">Rand mittel {m["rim_mean"]}, median {m.get("rim_median")}, Schweif {m.get("rim_tail_frac")} · '
            f'Füllung median {m.get("interior_median", m.get("fill_by_result_median", "-"))} '
            f'p90 {m.get("interior_p90", m.get("fill_by_result_p90", "-"))} · '
            f'Alpha-Kern {m.get("alpha_core_mismatch")}, Alpha-Diff {m.get("alpha_mean_diff")} · '
            f'Grund: {html.escape(str(m.get("why", "")))}'
            + (lambda ft: f'<br>Farbfit (Gain/Offset je Kanal): Median {ft[0]} → {ft[1]}, p90 {ft[2]}, Gain {ft[3]}, Offset {ft[4]}' if ft else "")(fits.get((i, j)))
            + '</div></div>')

    sizes = sorted(by_size)
    C = []
    for a, b in combinations(sizes, 2):
        if abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol:
            C.append(f'<li><a href="#s{a[0]}x{a[1]}">{a[0]}×{a[1]}</a> ({len(by_size[a])} Gr.) ↔ '
                     f'<a href="#s{b[0]}x{b[1]}">{b[0]}×{b[1]}</a> ({len(by_size[b])} Gr.)</li>')

    page = PAGE.replace("__A__", "".join(A)).replace("__B__", "".join(B) or "<p>Keine Paare.</p>") \
        .replace("__C__", "<ul>" + "".join(C) + "</ul>" if C else "<p>Keine Größen-Nachbarn.</p>") \
        .replace("__NG__", str(len(groups))).replace("__NP__", str(min(top, len(pairs)))) \
        .replace("__TOL__", str(tol)).replace("__NS__", str(len(sizes)))
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(sizes)} Zellgroessen, {len(pairs)} getrennte Paare gleicher Groesse "
          f"({no_rim} ohne messbaren Rand), {len(C)} Groessen-Nachbarn")
    print(f"Fertig: {out / 'index.html'}")
    return out / "index.html"


PAGE = r"""<!doctype html><html lang="de"><meta charset="utf-8"><title>Gruppen nebeneinander</title>
<style>
:root{color-scheme:dark;--bg:#16161d;--card:#1f1f2a;--ln:#33334a;--ac:#5b8cff;--ok:#3ecf8e;--bad:#e5484d;--tx:#e8e8f0;--mut:#9a9ab0}
body{margin:0;background:var(--bg);color:var(--tx);font:14px system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);display:flex;gap:14px;align-items:center;padding:8px 16px;border-bottom:1px solid var(--ln);flex-wrap:wrap}
.mut{color:var(--mut)} .bad{color:var(--bad)} a{color:var(--ac)} main{padding:10px 16px}
h2{margin:26px 0 6px;font-size:18px} h3{margin:16px 0 6px;font-size:15px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start}
.card{background:var(--card);border:2px solid var(--ln);border-radius:8px;cursor:pointer;display:flex;flex-direction:column}
.card.sel{border-color:var(--bad);box-shadow:0 0 0 2px var(--bad) inset}
.card img,.pair img{display:block;background-color:#bbb;background-image:linear-gradient(45deg,#999 25%,transparent 25%,transparent 75%,#999 75%),linear-gradient(45deg,#999 25%,transparent 25%,transparent 75%,#999 75%);background-size:16px 16px;background-position:0 0,8px 8px;border-radius:6px 6px 0 0}
.cap{padding:4px 8px;font-size:12px;color:var(--mut);width:0;min-width:100%;box-sizing:border-box;line-height:1.35;word-break:break-all}
.cap b{color:var(--tx)} .pair{background:var(--card);border:1px solid var(--ln);border-radius:8px;margin:10px 0;display:inline-block;max-width:100%;padding:6px}
.pair>div:first-child{margin-bottom:4px} textarea{width:340px;height:34px;background:var(--card);color:var(--tx);border:1px solid var(--ln)}
button{background:var(--card);color:var(--tx);border:1px solid var(--ln);border-radius:6px;padding:6px 12px;cursor:pointer}
</style>
<header><b>Gruppen nebeneinander</b><span class="mut">__NG__ Gruppen · __NS__ Größen</span>
<label class="mut">Größe <input id="zoom" type="range" min="0.5" max="3" step="0.1" value="1"></label>
<label class="mut"><input type="checkbox" id="tint" checked> unbekannte Pixel violett</label>
<span class="mut">Kacheln anklicken = verdächtig:</span><textarea id="out" readonly></textarea><button onclick="clr()">leeren</button>
<a href="#B">B Paare</a><a href="#C">C Größen-Nachbarn</a></header>
<main>
<h2 id="A">A · Alle Gruppen nach Zellgröße</h2>
<div class="mut">Konsensus-Rahmen jeder Gruppe. VIOLETT = Pixel, die in keiner Zelle der Gruppe frei von Text und Glow sind: dort ist der Rahmen GERATEN (Median der LaMa-Ergebnisse), nicht gemessen. Gleich große Gruppen stehen in einer Zeile: sehen zwei gleich aus, gehören sie evtl. zusammen. R### = Id im Ranker, E# = Einzelzelle, G# = Gruppe ohne Ranker-Id.</div>
__A__
<h2 id="B">B · Nächste getrennte Paare gleicher Größe (__NP__)</h2>
<div class="mut">Links A, Mitte B, rechts Differenz ×5 (schwarz = identisch). Niedriger Score = knapp an der Schwelle.</div>
__B__
<h2 id="C">C · Größen-Nachbarn (± __TOL__ px)</h2>
<div class="mut">Diese Größen werden nie verglichen, weil die Zellgröße nicht exakt gleich ist. Zeigen beide Zeilen gleiche Rahmen, ist es eine Kandidatur.</div>
__C__
</main>
<script>
const $=id=>document.getElementById(id);
function zoom(){const z=parseFloat($("zoom").value);document.querySelectorAll("img[data-w]").forEach(i=>i.style.width=(i.dataset.w*z)+"px")}
$("zoom").oninput=zoom;zoom();
function tint(){const on=$("tint").checked;document.querySelectorAll("img[data-plain]").forEach(i=>i.src=on?i.dataset.tint:i.dataset.plain)}
$("tint").onchange=tint;
function upd(){$("out").value=[...document.querySelectorAll(".card.sel")].map(c=>c.dataset.k).join(", ")}
function mark(c){c.classList.toggle("sel");upd()}
function clr(){document.querySelectorAll(".card.sel").forEach(c=>c.classList.remove("sel"));upd()}
</script></html>"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Seite-an-Seite-Ansicht der Rahmengruppen")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ranking", type=Path, default=None)
    p.add_argument("--ranker", type=Path, default=None, help="donor_ranker-Ordner (Standard RUN/donor_ranker)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--top", type=int, default=60)
    p.add_argument("--size-tol", type=int, default=2)
    p.add_argument("--gate-level", type=float, default=None)
    p.add_argument("--rim-median-max", type=float, default=None)
    p.add_argument("--fill-median-max", type=float, default=None)
    p.add_argument("--fill-p90-max", type=float, default=None)
    p.add_argument("--glow-px", type=int, default=None, help=f"Rand um die Textzone, der NICHT als Rahmen zaehlt (Standard {FD.GLOW_PX}); kleiner = weniger unbekannte Pixel")
    p.add_argument("--zone-dilate-px", type=int, default=None, help=f"Verbreiterung der Textzone (Standard {FD.ZONE_DILATE_PX})")
    p.add_argument("--zone-change-level", type=int, default=None, help=f"ab dieser Aenderung durch einen Kandidaten gilt ein Pixel als Text (Standard {FD.ZONE_CHANGE_LEVEL})")
    p.add_argument("--fit-merge", action="store_true",
                   help="Vorschau: Gruppen zusammenfuehren, die sich nur durch Gain/Offset der Farbe unterscheiden")
    p.add_argument("--fit-res-median", type=float, default=2.5, help="erlaubter Restabstand (Median) nach dem Fit")
    p.add_argument("--fit-res-p90", type=float, default=6.0, help="erlaubter Restabstand (p90) nach dem Fit")
    p.add_argument("--interior-median-max", type=float, default=None, help=f"Standard {FD.INTERIOR_MEDIAN_MAX}")
    p.add_argument("--interior-p90-max", type=float, default=None, help=f"Standard {FD.INTERIOR_P90_MAX}")
    p.add_argument("--tail-frac-max", type=float, default=None,
                   help=f"Anteil Randpixel mit grosser Abweichung, Standard {FD.RIM_CLEAN_TAIL_FRAC_MAX} (fern vom Text) / {FD.RIM_TAIL_FRAC_MAX} (ganzer Rand)")
    p.add_argument("--tail-level", type=float, default=None, help=f"ab dieser Abweichung zaehlt ein Randpixel zum Schweif, Standard {FD.RIM_TAIL_LEVEL}")
    a = p.parse_args(argv)
    if a.glow_px is not None:
        FD.GLOW_PX = a.glow_px
    if a.zone_dilate_px is not None:
        FD.ZONE_DILATE_PX = a.zone_dilate_px
    if a.zone_change_level is not None:
        FD.ZONE_CHANGE_LEVEL = a.zone_change_level
    if a.interior_median_max is not None:
        FD.INTERIOR_MEDIAN_MAX = a.interior_median_max
    if a.interior_p90_max is not None:
        FD.INTERIOR_P90_MAX = a.interior_p90_max
    if a.tail_frac_max is not None:
        FD.RIM_CLEAN_TAIL_FRAC_MAX = a.tail_frac_max
        FD.RIM_TAIL_FRAC_MAX = max(a.tail_frac_max, FD.RIM_TAIL_FRAC_MAX)
    if a.tail_level is not None:
        FD.RIM_TAIL_LEVEL = a.tail_level
    if a.gate_level is not None:     # wirkt in frame_donors nur auf Info-Felder (agree), nicht auf die Gruppenentscheidung
        S.GATE_LEVEL = a.gate_level
    if a.rim_median_max is not None:
        FD.RIM_MEDIAN_MAX = a.rim_median_max
    if a.fill_median_max is not None:
        FD.FILL_BY_RESULT_MEDIAN_MAX = a.fill_median_max
    if a.fill_p90_max is not None:
        FD.FILL_BY_RESULT_P90_MAX = a.fill_p90_max
    run = a.run.expanduser()
    build(run, a.ranking.expanduser() if a.ranking else None, a.ranker.expanduser() if a.ranker else None,
          (a.out or run / "group_sheet").expanduser(), a.top, a.size_tol, a.fit_merge, a.fit_res_median, a.fit_res_p90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
