"""Sibling-cell consensus fill: rebuild the background under text from sibling cells.

Sprite sheets often repeat one frame (button, text bar) in several cells that differ only in the
TEXT. When every cell carries text there is no clean copy to stamp (see donor.py), but the text sits
at different places in different cells, so at most pixels at least two cells show the plain frame.
Per mask pixel the cells vote; the value that several siblings agree on is the background.

A sibling is accepted only if ALL of these hold (nothing game specific):
  shape   same size and the same alpha silhouette (text never changes alpha),
  colour  outside every text zone the sibling equals the target pixel by pixel
          (so recoloured copies, e.g. the rows of a colour sheet, never vote),
  text    the result differs from the target inside the mask (a sibling with the SAME text would
          just hand the text back).
Needs at least MIN_SIBLINGS siblings: with a single one two equal pixels cannot be told from two
equal ink pixels.

Voting per pixel (target value plus every sibling value are the candidates):
  1. pixels on which all cells agree are certain; with the target's ring they give a smooth
     background estimate;
  2. at the other pixels the candidate with the most supporters AND the smallest distance to that
     estimate wins (a supported value far from the estimate is ink shared by two cells, e.g. the same
     glyph at the same place, and is not accepted); a lone candidate counts only if it lies within
     SMOOTH_TOL of the estimate;
  3. pixels without a verdict are holes: they are filled with a smooth (Gaussian) estimate from the voted
     pixels and the target's ring (Telea only where nothing is known nearby). The share
     of holes is reported and capped (MAX_HOLE_FRAC), so a tile that is mostly guesswork is refused.

    res = sibling_fill(rgb, alpha, region_mask, zone_mask)       # SiblingResult or SiblingMiss
    if isinstance(res, SiblingResult): rgb2 = res.rgb ...

Bank across images (SiblingBank)
  The same frame often appears in several image files (one sheet per text set). What the vote has
  verified as background is remembered per frame (<dir>/bank.json, patches/<id>.png, masks/<id>.png):
    known   pixel verified as background (voted by >= 2 cells, or outside every text zone)
    shared  pixel on which all cells agreed but which is far from the background estimate: ink that
            belongs to the frame itself (e.g. a header label); kept as it is when the target shows it too
  A later image whose tile passes the same shape/colour gate takes ``known`` pixels straight from the
  bank, so even a lone tile (no siblings in its own image) can be filled, and every image adds the
  pixels that are still unknown. Like donor.py, a run only uses entries that existed when it started;
  learned pixels are merged into the bank at save(). Entries whose fills were labelled unusable more
  often than usable are not used any more (``labels`` command).

    bank = SiblingBank("~/cleanup_compare/sibling_bank", learn=True)
    res = sibling_fill(rgb, alpha, mask, zone, tiles, bank=bank, source_image="m004_000", region=1)
    bank.save()

CLI:  python -m yolo_mask_auto_select_modules.siblings list   --bank DIR
      python -m yolo_mask_auto_select_modules.siblings labels --bank DIR --ranking RANK.csv --features stage2_features.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

import cv2
import numpy as np

Rect = tuple[int, int, int, int]            # x1, y1, x2, y2 inclusive

TILE_ALPHA_LEVEL = 15        # alpha above this is "opaque" when looking for tiles
TILE_MIN_PX = 400            # a connected opaque area needs this many pixels to be a tile
SIZE_TOL_PX = 0              # tiles of a group must have exactly the same size
ALPHA_SAME_LEVEL = 2         # alpha difference that still counts as the same silhouette
ALPHA_SAME_FRAC = 0.995      # share of the tile that must have the same alpha
ZONE_DILATE_PX = 3           # text zones are widened by this much before the colour gate
BANK_ZONE_DILATE_PX = 8      # ... and by this much when a remembered frame is compared (text glow reaches further than a tight mask)
GATE_LEVEL = 3               # per-pixel colour tolerance (any channel) in the colour gate
GATE_AGREE_MIN = 0.98        # share of non-text pixels that must agree
MIN_GATE_PIXELS = 80         # fewer pixels cannot prove the colours agree
MIN_SIBLINGS = 2             # at least this many siblings (so >= 3 tiles incl. the target)
VOTE_TOL = 3                 # candidates closer than this (any channel) count as the same value
SMOOTH_SIGMA = 4.0
SMOOTH_TOL = 12.0            # a lone candidate must be this close to the smooth estimate
SUPPORT_WEIGHT = 30.0        # one extra supporter is worth this many levels of distance to the estimate
GLOW_TOL = 6.0              # a candidate counts as background only if it is this close to the smooth estimate (weaker glow than this is invisible)
HALO_TOL = 3.0              # next to a hole, a voted pixel must be this close to the estimate
INK_FAR = 12.0               # a winner farther than this from the estimate is ink (two cells may share a glyph)
HOLE_SIGMAS = (2.5, 5.0, 10.0, 20.0)   # scales of the hole fill: the smallest one with enough known pixels around is used
HOLE_MIN_SUPPORT = 0.35     # share of the kernel that must lie on known pixels
HOLE_GUARD_PX = 2.5         # decided pixels this close to a hole do not feed the fill (glow tail)
MAX_HOLE_FRAC = 0.55         # more holes than this: refuse
BANK_VERSION = 1
ASSIGNED_ALPHA_CORE_MAX = 0.02   # an approved donor given to a cell by its group: share of the tile whose hard shape (alpha >= 128) may differ
ASSIGNED_RIM_MEDIAN_MAX = 8.0    # safety net: a group's donor is dropped if its colours differ from the cell by more than this (median, outside the text zones)
ASSIGNED_ALPHA_MEAN_MAX = 3.0    # ... and the mean alpha difference (soft edges are smoothed a little differently between sheets)
MAX_OFF_TILE_FRAC = 0.10     # region pixels on no tile (transparent area) tolerated before the region is refused
MAX_SKIP_PART_FRAC = 0.03    # a failing part this small (share of the region) is left as it is instead of refusing the region
MIN_INK_PX = 20              # a part with less ink than this has nothing to remove: it passes unchanged
MIN_CHANGED_INK_FRAC = 0.2   # the fill must change this many pixels (MIN_CHANGED_PX) or this share of the part's ink, if that is smaller
BODY_EDGE_MARGIN_PX = 10     # shared ink closer than this to the sprite's edge is frame, not text, even inside a body box
SAME_TEXT_SHARED_FRAC = 0.6  # of the body ink of the target, more than this share is identical in all cells: same text
MIN_CHANGED_PX = 40          # (a short line in a big box changes only ~1-2 % of the mask, so the threshold is absolute, not a share of the mask)
CHANGED_LEVEL = 8


@dataclass
class SiblingResult:
    rgb: np.ndarray
    alpha: np.ndarray
    voted: np.ndarray                      # bool HxW: pixels decided by the vote
    holes: np.ndarray                      # bool HxW: pixels filled by Telea
    meta: dict = field(default_factory=dict)


@dataclass
class SiblingMiss:
    reason: str
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- tiles

def find_tiles(alpha: np.ndarray) -> list[Rect]:
    """Opaque connected areas (sprites) as rects. Works without any text information."""
    n, _lab, st, _c = cv2.connectedComponentsWithStats((alpha > TILE_ALPHA_LEVEL).astype(np.uint8), connectivity=8)
    return [(int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2] - 1), int(st[i, 1] + st[i, 3] - 1))
            for i in range(1, n) if st[i, 4] >= TILE_MIN_PX]


def _size(r: Rect) -> tuple[int, int]:
    return r[2] - r[0] + 1, r[3] - r[1] + 1


def _crop(a: np.ndarray, r: Rect) -> np.ndarray:
    return a[r[1]:r[3] + 1, r[0]:r[2] + 1]


def _contains(r: Rect, x1: int, y1: int, x2: int, y2: int) -> bool:
    return r[0] <= x1 and r[1] <= y1 and r[2] >= x2 and r[3] >= y2


def _removed_enough(changed_px: float, ink_px: float) -> bool:
    """Did the fill remove what there was to remove? (False = the siblings carry the same text.)"""
    if ink_px < MIN_INK_PX:
        return True                                         # nothing to remove in this part
    return changed_px >= min(MIN_CHANGED_PX, MIN_CHANGED_INK_FRAC * ink_px)


def _delta_e_p95(a: np.ndarray, b: np.ndarray) -> float:
    la = cv2.cvtColor(a.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    lb = cv2.cvtColor(b.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return float(np.percentile(np.sqrt(((la - lb) ** 2).sum(axis=2)), 95))


def _smooth_estimate(values: np.ndarray, known: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalised Gaussian average of the known pixels. Returns (estimate float32 HxWx3, weight)."""
    k = known.astype(np.float32)
    num = cv2.GaussianBlur(values.astype(np.float32) * k[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(k, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)[..., None], den


# ----------------------------------------------------------------------------- main

def find_siblings(rgb: np.ndarray, alpha: np.ndarray, tile: Rect, tiles: list[Rect],
                  zone: np.ndarray) -> tuple[list[Rect], dict]:
    """Tiles that qualify as siblings of ``tile`` (shape and colour gate) and gate statistics."""
    zone_t = cv2.dilate(_crop(zone, tile).astype(np.uint8), np.ones((2 * ZONE_DILATE_PX + 1,) * 2, np.uint8)) > 0
    a_t = _crop(alpha, tile)
    sibs: list[Rect] = []
    stats = {"same_size_tiles": 0, "alpha_rejected": 0, "colour_rejected": 0, "gate_pixels_min": 10 ** 9}
    for s in tiles:
        if s == tile or abs(_size(s)[0] - _size(tile)[0]) > SIZE_TOL_PX or abs(_size(s)[1] - _size(tile)[1]) > SIZE_TOL_PX:
            continue
        stats["same_size_tiles"] += 1
        a_s = _crop(alpha, s)
        if a_s.shape != a_t.shape or (np.abs(a_s.astype(np.int16) - a_t.astype(np.int16)) <= ALPHA_SAME_LEVEL).mean() < ALPHA_SAME_FRAC:
            stats["alpha_rejected"] += 1
            continue
        zone_s = cv2.dilate(_crop(zone, s).astype(np.uint8), np.ones((2 * ZONE_DILATE_PX + 1,) * 2, np.uint8)) > 0
        use = (a_t > TILE_ALPHA_LEVEL) & ~zone_t & ~zone_s
        stats["gate_pixels_min"] = min(stats["gate_pixels_min"], int(use.sum()))
        if int(use.sum()) < MIN_GATE_PIXELS:
            stats["colour_rejected"] += 1
            continue
        c_t, c_s = _crop(rgb, tile)[use], _crop(rgb, s)[use]
        agree = float((np.abs(c_t.astype(np.int16) - c_s.astype(np.int16)).max(axis=1) <= GATE_LEVEL).mean())
        if agree < GATE_AGREE_MIN:
            stats["colour_rejected"] += 1
            continue
        sibs.append(s)
    if stats["gate_pixels_min"] == 10 ** 9:
        stats["gate_pixels_min"] = 0
    return sibs, stats


def _fill_tile(rgb: np.ndarray, alpha: np.ndarray, region_mask: np.ndarray, zone: np.ndarray,
               tiles: list[Rect], tile: Rect, bank: "SiblingBank | None", source_image: str,
               region: int, body_mask: np.ndarray | None = None) -> SiblingResult | SiblingMiss:
    """Fill ``region_mask`` (lies inside ``tile``) from the sibling tiles and the bank."""
    ys, xs = np.nonzero(region_mask)
    body_here = np.zeros(len(ys), bool) if body_mask is None else np.asarray(body_mask, dtype=bool)[ys, xs]
    if body_here.any():                                    # the frame's outline is shared by all cells and is never text
        a_tile = _crop(alpha, tile) > TILE_ALPHA_LEVEL
        edge = cv2.distanceTransform(np.pad(a_tile, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
        body_here &= edge[ys - tile[1], xs - tile[0]] > BODY_EDGE_MARGIN_PX
    sibs, gate = find_siblings(rgb, alpha, tile, tiles, zone)
    zone_t = _crop(zone, tile)
    entry = bank.match(rgb, alpha, tile, zone_t) if bank is not None else None
    meta = {"sibling_tile": f"{tile[0]},{tile[1]}", "sibling_count": len(sibs), **{f"sibling_{k}": v for k, v in gate.items()}}
    if entry is not None:
        meta["sibling_entry_id"] = entry.entry_id
        meta["sibling_entry_known_frac"] = round(float(entry.known[zone_t].mean()), 4) if zone_t.any() else 0.0
    if len(sibs) < MIN_SIBLINGS and entry is None:
        return SiblingMiss("too_few_siblings", meta)

    h, w = tile[3] - tile[1] + 1, tile[2] - tile[0] + 1
    ty, tx = ys - tile[1], xs - tile[0]
    cands = [rgb[ys, xs].astype(np.int16)]                  # candidate 0 = the target itself
    for s_ in sibs:
        cands.append(rgb[ty + s_[1], tx + s_[0]].astype(np.int16))
    V = np.stack(cands)                                     # N, n, 3
    N = V.shape[0]
    agree = np.abs(V[:, None] - V[None]).max(axis=-1) <= VOTE_TOL      # N, N, n
    support = agree.sum(axis=1)                             # N, n  (includes itself)
    certain = (support.min(axis=0) == N) if N >= 2 else np.zeros(V.shape[1], bool)   # all cells agree

    if entry is not None:
        e_val = entry.rgb[ty, tx].astype(np.int16)
        e_known = entry.known[ty, tx]
        e_shared = entry.shared[ty, tx] & ~e_known
    else:
        e_val = V[0]
        e_known = np.zeros(V.shape[1], bool)
        e_shared = np.zeros(V.shape[1], bool)

    # smooth background estimate from certain pixels, the bank's known pixels and the target's ring
    tile_rgb = _crop(rgb, tile)
    known = np.zeros((h, w), bool)
    known[ty[certain], tx[certain]] = True
    ring = ~zone_t & (_crop(alpha, tile) > TILE_ALPHA_LEVEL)
    known |= ring
    est_vals = tile_rgb.copy()
    if entry is not None:
        known[ty[e_known], tx[e_known]] = True
        est_vals[ty[e_known], tx[e_known]] = e_val[e_known].astype(np.uint8)
    est_img, den = _smooth_estimate(est_vals, known, SMOOTH_SIGMA)
    est = est_img[ty, tx]                                   # n, 3
    est_ok = den[ty, tx] > 0.02
    dist = np.abs(V.astype(np.float32) - est[None]).max(axis=-1)       # N, n
    # Ink that all cells share (a header label, a glyph at the same place) is "certain" but is not background:
    # take it out of the estimate, otherwise it bleeds into the neighbourhood (and hides itself).
    ink_c = certain & est_ok & (dist[0] > INK_FAR)
    kept_ink_t = np.zeros((h, w), bool)
    kept_ink_t[ty[ink_c], tx[ink_c]] = True
    if ink_c.any():
        known &= ~kept_ink_t
        est_img, den = _smooth_estimate(est_vals, known, SMOOTH_SIGMA)
        est = est_img[ty, tx]
        est_ok = den[ty, tx] > 0.02
        dist = np.abs(V.astype(np.float32) - est[None]).max(axis=-1)

    ink_px = int((est_ok & (dist[0] > INK_FAR)).sum())      # pixels of the target that stand out from the background
    meta["sibling_ink_px"] = ink_px

    # choose per pixel: highest support, ties by closeness to the estimate
    score = support.astype(np.float32) * np.where(est_ok, SUPPORT_WEIGHT, 1000.0)[None] - np.where(est_ok[None], dist, 0.0)
    best = score.argmax(axis=0)
    cols = np.arange(V.shape[1])
    best_support = support[best, cols]
    pick = V[best, cols]
    members = agree[best, :, cols].T                        # N, n: who agrees with the pick
    cluster_mean = (V * members[..., None]).sum(axis=0) / np.maximum(members.sum(axis=0), 1)[:, None]
    value = np.rint(cluster_mean).astype(np.int16)
    # Text glow is an additive halo that differs from sibling to sibling and fades out over many levels: agreement
    # between two siblings does not prove "background" (two faint halos can agree). Where the smooth estimate is
    # known, the candidate CLOSEST to it is taken, and only if it is within GLOW_TOL; otherwise the pixel is a hole
    # and gets the smooth estimate instead of a ghost. Without an estimate the vote by support is used.
    arg = dist.argmin(axis=0)
    dmin = dist[arg, cols]
    closest = V[arg, cols]
    glow_ok = est_ok & (dmin <= GLOW_TOL)
    support_ok = ~est_ok & (best_support >= 2)
    far = est_ok & (dist[0] > INK_FAR)                       # the target's own pixel stands out from the background
    value = np.where(glow_ok[:, None], closest, np.where((best_support == 1)[:, None], pick, value))
    # What ALL cells share stays as it is (e.g. a header label) - except inside a body box: there it is text
    # (the same glyph at the same place in every cell, e.g. a final "?") and is treated like text.
    decided = glow_ok | support_ok | (certain & ~(far & body_here))
    dsel = np.where(glow_ok, dmin, dist[best, cols])

    # Halo guard: a voted pixel right next to a hole that is not shared by all cells and is not
    # very close to the smooth estimate may still carry the faint edge of a glyph: make it a hole too.
    holes_flat = ~decided
    hmap = np.zeros((h, w), np.uint8)
    hmap[ty[holes_flat], tx[holes_flat]] = 1
    near_hole = (cv2.dilate(hmap, np.ones((3, 3), np.uint8)) > 0)[ty, tx]
    halo = near_hole & decided & ~certain & est_ok & (dsel > HALO_TOL)
    voted_ok = decided & ~halo

    # bank pixels: verified background wins; ink the frame itself carries is kept when the target shows it too
    shared_keep = e_shared & ~body_here & (np.abs(V[0] - e_val).max(axis=-1) <= VOTE_TOL)
    use_entry = e_known
    final_decided = voted_ok | use_entry | shared_keep
    final_value = np.where(use_entry[:, None], e_val, np.where(shared_keep[:, None], V[0], value))

    # learn (before the accept tests, so that a refused tile still teaches the bank what its vote verified)
    if bank is not None and bank.learn and len(sibs) >= MIN_SIBLINGS:
        changed_vote = float((np.abs(np.where(voted_ok[:, None], value, V[0]) - V[0]).max(axis=-1) > CHANGED_LEVEL).mean())
        if _removed_enough(changed_vote * len(ys), float(ink_px)):                # not "same text everywhere"
            k_new = np.zeros((h, w), bool)
            sh_new = np.zeros((h, w), bool)
            val_tile = tile_rgb.copy()
            good = (glow_ok | support_ok) & ~halo
            k_new[ty[good], tx[good]] = True
            val_tile[ty[good], tx[good]] = np.clip(value[good], 0, 255).astype(np.uint8)
            sh = certain & far & ~body_here
            sh_new[ty[sh], tx[sh]] = True
            val_tile[ty[sh], tx[sh]] = V[0][sh].astype(np.uint8)
            k_new |= ring                                   # outside every text zone: verified by the colour gate
            bank.add(_crop(alpha, tile), val_tile, k_new, sh_new, source_image, region)

    out_rgb = rgb.copy()
    out_rgb[ys[final_decided], xs[final_decided]] = np.clip(final_value[final_decided], 0, 255).astype(np.uint8)
    holes_flat = ~final_decided
    voted = np.zeros(region_mask.shape, bool)
    voted[ys[final_decided], xs[final_decided]] = True
    holes = np.zeros(region_mask.shape, bool)
    holes[ys[holes_flat], xs[holes_flat]] = True
    hole_frac = float(holes_flat.mean())
    changed = float((np.abs(out_rgb[ys, xs].astype(np.int16) - rgb[ys, xs].astype(np.int16)).max(axis=-1) > CHANGED_LEVEL).mean())
    meta.update({"sibling_voted_frac": round(1.0 - hole_frac, 4), "sibling_hole_frac": round(hole_frac, 4),
                 "sibling_changed_frac": round(changed, 4), "sibling_certain_frac": round(float(certain.mean()), 4),
                 "sibling_bank_frac": round(float(use_entry.mean()), 4)})
    if not _removed_enough(changed * len(ys), float(ink_px)):
        return SiblingMiss("sibling_same_text", meta)         # nothing removed: siblings carry the same text
    if hole_frac > MAX_HOLE_FRAC:
        return SiblingMiss("too_many_holes", meta)

    if holes.any():                                          # holes: smooth multi-scale estimate from clean decided pixels
        voted_t = voted[tile[1]:tile[3] + 1, tile[0]:tile[2] + 1]
        holes_t = holes[tile[1]:tile[3] + 1, tile[0]:tile[2] + 1]
        known2 = (voted_t | ring) & ~holes_t & ~kept_ink_t
        # The fading tail of the glow lies right next to the holes: it must not feed the fill (that is the "ghost").
        dist_hole = cv2.distanceTransform((~holes_t).astype(np.uint8), cv2.DIST_L2, 3)
        guarded = known2 & (dist_hole > HOLE_GUARD_PX)
        src = guarded if guarded.sum() >= 50 else known2
        base = _crop(out_rgb, tile)
        fill = np.zeros((h, w, 3), np.float32)
        todo = holes_t.copy()
        for sigma in HOLE_SIGMAS:                            # smallest scale with enough support wins
            est2, den2 = _smooth_estimate(base, src, sigma)
            ok = todo & (den2 >= HOLE_MIN_SUPPORT)
            fill[ok] = est2[ok]
            todo &= ~ok
        est2, den2 = _smooth_estimate(base, src, HOLE_SIGMAS[-1])
        last = todo & (den2 > 0.02)
        fill[last] = est2[last]
        todo &= ~last
        use = holes_t & ~todo
        patch = _crop(out_rgb, tile)
        patch[use] = np.clip(np.rint(fill[use]), 0, 255).astype(np.uint8)
        if todo.any():                                       # no known pixel anywhere near: Telea
            fixed = cv2.inpaint(np.ascontiguousarray(patch), todo.astype(np.uint8), 3, cv2.INPAINT_TELEA)
            patch[todo] = fixed[todo]
    return SiblingResult(out_rgb, alpha.copy(), voted, holes, meta)


def sibling_fill(rgb: np.ndarray, alpha: np.ndarray, region_mask: np.ndarray, zone_mask: np.ndarray | None = None,
                 tiles: list[Rect] | None = None, bank: "SiblingBank | None" = None,
                 source_image: str = "", region: int = 0,
                 body_mask: np.ndarray | None = None) -> SiblingResult | SiblingMiss:
    """Fill ``region_mask`` from the sibling tiles and the bank.

    ``body_mask``: pixels of the text (body) boxes. Ink that all cells share is kept only outside of
    it (frame labels such as a header); inside it identical glyphs are text and are removed.

    ``zone_mask``: union of all text masks of the image (defaults to ``region_mask``); it tells the
    colour gate where text may be. With ``bank`` a lone tile can be filled from remembered frames, and
    (bank.learn) the verified background pixels are remembered for later images.

    A region may cover several tiles (stage 2 merges boxes closer than 2*region_merge_px, and stacked
    tiles touch): it is split per tile and every part is filled on its own. All parts must succeed,
    otherwise the region is refused, so a region is never half cleaned. Region pixels that lie on no
    tile (transparent area) are left as they are; if they are more than MAX_OFF_TILE_FRAC of the
    region the mask does not belong to this sprite sheet and the region is refused (``no_tile``)."""
    region_mask = np.asarray(region_mask, dtype=bool)
    if not region_mask.any():
        return SiblingMiss("empty_mask")
    zone = region_mask if zone_mask is None else (np.asarray(zone_mask, dtype=bool) | region_mask)
    tiles = find_tiles(alpha) if tiles is None else tiles
    parts: list[tuple[Rect, np.ndarray]] = []
    total = int(region_mask.sum())
    covered = 0
    for t in tiles:
        sub = np.zeros_like(region_mask)
        sub[t[1]:t[3] + 1, t[0]:t[2] + 1] = region_mask[t[1]:t[3] + 1, t[0]:t[2] + 1]
        n = int(sub.sum())
        if n:
            # a pixel on two overlapping tiles belongs to the smaller one
            parts.append((t, sub))
            covered += n
    def _no_tile(off: float) -> SiblingMiss:
        vis = float((alpha[region_mask] > TILE_ALPHA_LEVEL).mean())
        biggest = max((_size(t)[0] * _size(t)[1] for t in tiles), default=0)
        return SiblingMiss("no_tile", {"sibling_tiles_in_image": len(tiles), "sibling_off_tile_frac": round(off, 4),
                                       "sibling_region_visible_frac": round(vis, 4), "sibling_biggest_tile_px": biggest,
                                       "sibling_region_px": total})
    if not parts:
        return _no_tile(1.0)
    # make parts disjoint (overlapping tile rects): smaller tile first, pixels taken once
    parts.sort(key=lambda p: _size(p[0])[0] * _size(p[0])[1])
    taken = np.zeros_like(region_mask)
    disjoint: list[tuple[Rect, np.ndarray]] = []
    for t, sub in parts:
        sub = sub & ~taken
        if sub.any():
            disjoint.append((t, sub))
            taken |= sub
    off = 1.0 - float(taken.sum()) / total
    if off > MAX_OFF_TILE_FRAC:
        return _no_tile(off)
    results: list[tuple[np.ndarray, SiblingResult]] = []
    failed: list[tuple[np.ndarray, SiblingMiss]] = []
    for t, sub in disjoint:
        r = _fill_tile(rgb, alpha, sub, zone, tiles, t, bank, source_image, region, body_mask)
        if isinstance(r, SiblingMiss):
            failed.append((sub, r))
        else:
            results.append((sub, r))
    skipped_px = 0
    for sub, miss in failed:
        # a sliver of the region (a few pixels of a neighbouring tile) must not veto the whole region: it stays as it is
        if results and int(sub.sum()) <= MAX_SKIP_PART_FRAC * total:
            skipped_px += int(sub.sum())
            continue
        miss.meta = {**miss.meta, "sibling_parts": len(disjoint), "sibling_off_tile_frac": round(off, 4)}
        return miss
    if not results:
        return SiblingMiss("no_tile", {"sibling_tiles_in_image": len(tiles), "sibling_off_tile_frac": round(off, 4)})
    for _sub, r in results:
        r.meta["sibling_skipped_px"] = skipped_px
    if len(results) == 1:
        results[0][1].meta["sibling_parts"] = 1
        results[0][1].meta["sibling_off_tile_frac"] = round(off, 4)
        return results[0][1]
    out_rgb = rgb.copy()
    voted = np.zeros(region_mask.shape, bool)
    holes = np.zeros(region_mask.shape, bool)
    for sub, r in results:
        out_rgb[sub] = r.rgb[sub]
        voted |= r.voted & sub
        holes |= r.holes & sub
    w = np.array([int(sub.sum()) for sub, _ in results], float)
    w /= w.sum()

    def avg(key: str) -> float:
        return round(float(sum(wi * float(r.meta.get(key, 0.0)) for wi, (_, r) in zip(w, results))), 4)

    ids = [r.meta["sibling_entry_id"] for _, r in results if r.meta.get("sibling_entry_id")]
    meta = {"sibling_parts": len(results), "sibling_off_tile_frac": round(off, 4), "sibling_skipped_px": skipped_px,
            "sibling_count": min(int(r.meta["sibling_count"]) for _, r in results),
            "sibling_voted_frac": avg("sibling_voted_frac"), "sibling_hole_frac": avg("sibling_hole_frac"),
            "sibling_changed_frac": avg("sibling_changed_frac"), "sibling_certain_frac": avg("sibling_certain_frac"),
            "sibling_bank_frac": avg("sibling_bank_frac")}
    if ids:
        meta["sibling_entry_id"] = "+".join(dict.fromkeys(ids))
    return SiblingResult(out_rgb, alpha.copy(), voted, holes, meta)


# ----------------------------------------------------------------------------- frame donor (method FRAME_DONOR)

def frame_donor_fill(rgb: np.ndarray, alpha: np.ndarray, region_mask: np.ndarray, zone_mask: np.ndarray | None,
                     tiles: list[Rect] | None, bank: "SiblingBank", source_image: str = "",
                     region: int = 0, assigned: "BankMatch | None" = None) -> SiblingResult | SiblingMiss:
    """Stamp a frame a person approved (SiblingBank.approve) onto ``region_mask``.

    ``assigned``: the approved frame the donor ranker's group gave to this cell (SiblingBank.assigned_entry). It is
    used for a tile of the same size whose shape is close (_alpha_close) even where the strict colour gate of
    bank.match would refuse; any other tile falls back to bank.match as before.

    The donor is a complete clean cell, so it also covers cells whose text sits somewhere else. A frame
    qualifies only if shape and colours match (same gate as the sibling bank); a region on several tiles is
    split per tile and every part needs its own donor (parts below MAX_SKIP_PART_FRAC may stay as they are)."""
    region_mask = np.asarray(region_mask, dtype=bool)
    if not region_mask.any():
        return SiblingMiss("empty_mask")
    zone = region_mask if zone_mask is None else (np.asarray(zone_mask, dtype=bool) | region_mask)
    tiles = find_tiles(alpha) if tiles is None else tiles
    total = int(region_mask.sum())
    parts: list[tuple[Rect, np.ndarray]] = []
    taken = np.zeros_like(region_mask)
    for t in sorted(tiles, key=lambda r: _size(r)[0] * _size(r)[1]):          # a pixel on two tiles belongs to the smaller
        sub = np.zeros_like(region_mask)
        sub[t[1]:t[3] + 1, t[0]:t[2] + 1] = region_mask[t[1]:t[3] + 1, t[0]:t[2] + 1]
        sub &= ~taken
        if sub.any():
            parts.append((t, sub))
            taken |= sub
    off = 1.0 - float(taken.sum()) / total
    if not parts or off > MAX_OFF_TILE_FRAC:
        return SiblingMiss("no_tile", {"sibling_off_tile_frac": round(off, 4), "sibling_tiles_in_image": len(tiles)})
    out_rgb = rgb.copy()
    stamped = np.zeros_like(region_mask)
    ids: list[str] = []
    agrees: list[float] = []
    skipped = 0
    first_miss: SiblingMiss | None = None
    used_assigned = False
    for t, sub in parts:
        entry = None
        if assigned is not None and assigned.alpha is not None and assigned.rgb.shape[:2] == (_size(t)[1], _size(t)[0]) \
                and _alpha_close(assigned.alpha, _crop(alpha, t)):
            zd = cv2.dilate(_crop(zone, t).astype(np.uint8), np.ones((2 * BANK_ZONE_DILATE_PX + 1,) * 2, np.uint8)) > 0
            cmp_px = ~zd & assigned.known & (_crop(alpha, t) > TILE_ALPHA_LEVEL)
            if int(cmp_px.sum()) >= MIN_GATE_PIXELS:
                diff = np.abs(assigned.rgb[cmp_px].astype(np.int16) - _crop(rgb, t)[cmp_px].astype(np.int16)).max(axis=1)
                if float(np.median(diff)) <= ASSIGNED_RIM_MEDIAN_MAX:         # the group's donor, colours agree
                    entry = BankMatch(assigned.entry_id, assigned.rgb, assigned.known, assigned.shared,
                                      float((diff <= GATE_LEVEL).mean()), int(cmp_px.sum()), True, assigned.alpha)
                    used_assigned = True
        if entry is None:
            entry = bank.match(rgb, alpha, t, _crop(zone, t), approved_only=True)
        if entry is None:
            if first_miss is None:
                first_miss = SiblingMiss("no_frame_donor", {"sibling_tile": f"{t[0]},{t[1]}"})
            skipped += int(sub.sum())
            continue
        ys, xs = np.nonzero(sub)
        ty, tx = ys - t[1], xs - t[0]
        known = entry.known[ty, tx]
        if not known.all():                                    # the donor does not cover the whole part: do not guess
            return SiblingMiss("donor_incomplete", {"sibling_entry_id": entry.entry_id,
                                                    "sibling_hole_frac": round(float(1.0 - known.mean()), 4)})
        out_rgb[ys, xs] = entry.rgb[ty, tx]
        stamped |= sub
        ids.append(entry.entry_id)
        agrees.append(entry.agree)
    if not stamped.any():
        return first_miss or SiblingMiss("no_frame_donor")
    if skipped > MAX_SKIP_PART_FRAC * total:                  # a big part found no donor: refuse, never half clean
        return first_miss or SiblingMiss("no_frame_donor")
    meta = {"sibling_entry_id": "+".join(dict.fromkeys(ids)), "sibling_parts": len(parts), "sibling_skipped_px": skipped,
            "sibling_donor_agree": round(float(min(agrees)), 4), "sibling_bank_frac": round(float(stamped.sum()) / total, 4),
            "sibling_hole_frac": 0.0, "sibling_donor_assigned": int(used_assigned)}
    return SiblingResult(out_rgb, alpha.copy(), stamped, np.zeros_like(region_mask), meta)


# ----------------------------------------------------------------------------- bank across images

@dataclass
class BankMatch:
    entry_id: str
    rgb: np.ndarray            # h, w, 3 uint8: remembered frame
    known: np.ndarray          # bool h, w: verified background
    shared: np.ndarray         # bool h, w: ink of the frame itself (all cells agreed)
    agree: float = 1.0
    gate_pixels: int = 0
    approved: bool = False
    alpha: np.ndarray | None = None   # h, w uint8: alpha of the remembered frame (set for approved frames)


def _alpha_close(a_entry: np.ndarray, a_tile: np.ndarray) -> bool:
    """Same hard shape and about the same soft edge. Looser than the colour gate's alpha test on purpose: a donor
    that a person's group assigned to a cell only stamps RGB, the cell keeps its own alpha."""
    if a_entry.shape != a_tile.shape:
        return False
    ins = (a_entry > TILE_ALPHA_LEVEL) | (a_tile > TILE_ALPHA_LEVEL)
    n = max(int(ins.sum()), 1)
    core = float(((a_entry >= 128) != (a_tile >= 128)).sum()) / n
    mean = float(np.abs(a_entry.astype(np.int16) - a_tile.astype(np.int16))[ins].mean()) if ins.any() else 0.0
    return core <= ASSIGNED_ALPHA_CORE_MAX and mean <= ASSIGNED_ALPHA_MEAN_MAX


def _gate(entry_patch: np.ndarray, entry_known: np.ndarray, alpha: np.ndarray, rgb: np.ndarray,
          use: np.ndarray) -> tuple[bool, float, int]:
    """Shape + colour gate between a remembered frame and a tile. ``use``: pixels allowed to be compared."""
    if entry_patch.shape[:2] != alpha.shape:
        return False, 0.0, 0
    a_e = entry_patch[..., 3]
    if (np.abs(a_e.astype(np.int16) - alpha.astype(np.int16)) <= ALPHA_SAME_LEVEL).mean() < ALPHA_SAME_FRAC:
        return False, 0.0, 0
    cmp_px = use & entry_known & (alpha > TILE_ALPHA_LEVEL)
    n = int(cmp_px.sum())
    if n < MIN_GATE_PIXELS:
        return False, 0.0, n
    agree = float((np.abs(entry_patch[..., :3][cmp_px].astype(np.int16) - rgb[cmp_px].astype(np.int16)).max(axis=1) <= GATE_LEVEL).mean())
    return agree >= GATE_AGREE_MIN, agree, n


class SiblingBank:
    """Frames remembered on disk: <dir>/bank.json, patches/<id>.png (RGBA), masks/<id>.png (255 known, 128 shared)."""

    def __init__(self, path: str | Path, *, learn: bool = False) -> None:
        self.path = Path(path).expanduser()
        self.learn = learn
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = self._read_index()      # frozen snapshot: used for matching
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._pending: list[tuple] = []                           # learned in this run: merged at save()
        self._approved: list[tuple] = []                          # frames chosen by a person: written at save()
        self._seen: dict[str, int] = {}
        self._outcomes: dict[str, list[int]] = {}

    # -- persistence
    def _read_index(self) -> dict[str, dict]:
        f = self.path / "bank.json"
        if not f.is_file():
            return {}
        data = json.loads(f.read_text(encoding="utf-8"))
        return {e["id"]: e for e in data.get("entries", [])}

    def _load(self, eid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            if eid in self._cache:
                return self._cache[eid]
        patch = np.asarray(Image.open(self.path / "patches" / f"{eid}.png").convert("RGBA"))
        code = np.asarray(Image.open(self.path / "masks" / f"{eid}.png").convert("L"))
        out = (patch, code == 255, code == 128)
        with self._lock:
            self._cache[eid] = out
        return out

    def save(self) -> int:
        """Merge what was learned into the bank; returns the number of new entries. No-op without learn."""
        if not self.learn and not self._outcomes and not self._approved:
            return 0
        with self._lock:
            pending = sorted(self._pending, key=lambda p: (p[4], p[5]))
            self._pending = []
            approved = list(self._approved)
            self._approved = []
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "patches").mkdir(exist_ok=True)
        (self.path / "masks").mkdir(exist_ok=True)
        entries = self._read_index()                              # merge with what is on disk now
        work: dict[str, list] = {}                                # id -> [patch, known, shared] being changed
        n_new = 0

        def get(eid: str):
            if eid not in work:
                p, k, sh = self._load_disk(eid)
                work[eid] = [p.copy(), k.copy(), sh.copy()]
            return work[eid]

        for alpha, rgb, known, shared, src, reg in pending:
            target = None
            for eid, meta in entries.items():
                if tuple(meta.get("window", ())) != (alpha.shape[1], alpha.shape[0]):
                    continue
                p, k, _sh = get(eid)
                ok, _a, _n = _gate(p, k, alpha, rgb, known)
                if ok:
                    target = eid
                    break
            if target is None:
                eid = hashlib.sha1(alpha.tobytes() + rgb[known].tobytes() + str(alpha.shape).encode()).hexdigest()[:12]
                patch = np.dstack((rgb, alpha)).astype(np.uint8)
                work[eid] = [patch, known.copy(), shared & ~known]
                entries[eid] = {"id": eid, "window": [int(alpha.shape[1]), int(alpha.shape[0])], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "sources": [f"{src}#r{reg}"], "seen": 1, "ok": 0, "bad": 0, "conflicts": 0}
                n_new += 1
                continue
            p, k, sh = get(target)
            both = k & known
            conflict = both & (np.abs(p[..., :3].astype(np.int16) - rgb.astype(np.int16)).max(axis=-1) > VOTE_TOL)
            fresh = known & ~k
            p[..., :3][fresh] = rgb[fresh]
            k |= fresh
            sh &= ~k                                              # a pixel proven to be background is not shared ink any more
            sh |= shared & ~k
            meta = entries[target]
            meta["conflicts"] = int(meta.get("conflicts", 0)) + int(conflict.sum())
            meta["merges"] = int(meta.get("merges", 0)) + 1
            meta["compared_px"] = int(meta.get("compared_px", 0)) + int(both.sum())
            meta["seen"] = int(meta.get("seen", 1)) + 1
            meta.setdefault("sources", [])
            if len(meta["sources"]) < 30 and f"{src}#r{reg}" not in meta["sources"]:
                meta["sources"].append(f"{src}#r{reg}")
        for alpha, rgb, label in approved:                        # a person's choice replaces what was learned
            opaque = alpha > TILE_ALPHA_LEVEL
            target = None
            for eid, meta in entries.items():
                if tuple(meta.get("window", ())) != (alpha.shape[1], alpha.shape[0]):
                    continue
                p, k, _sh = get(eid)
                if k.any():
                    ok, _a, _n = _gate(p, k, alpha, rgb, opaque)
                else:                                              # entry without known pixels: compare the silhouette only
                    ok = (np.abs(p[..., 3].astype(np.int16) - alpha.astype(np.int16)) <= ALPHA_SAME_LEVEL).mean() >= ALPHA_SAME_FRAC
                if ok:
                    target = eid
                    break
            if target is None:
                target = hashlib.sha1(b"approved" + alpha.tobytes() + rgb.tobytes()).hexdigest()[:12]
                entries[target] = {"id": target, "window": [int(alpha.shape[1]), int(alpha.shape[0])],
                                   "created": time.strftime("%Y-%m-%d %H:%M:%S"), "sources": [], "seen": 1, "ok": 0, "bad": 0,
                                   "conflicts": 0}
                n_new += 1
            work[target] = [np.dstack((rgb, alpha)).astype(np.uint8), opaque.copy(), np.zeros(alpha.shape, bool)]
            entries[target]["approved"] = True
            entries[target]["sources"] = [label]
            labels = entries[target].setdefault("labels", [])          # all approved donors that ended up in this entry
            if label and label not in labels:
                labels.append(label)
        for eid, (p, k, sh) in work.items():
            if eid not in entries:
                continue
            Image.fromarray(p).save(self.path / "patches" / f"{eid}.png")
            code = np.zeros(k.shape, np.uint8)
            code[sh] = 128
            code[k] = 255
            Image.fromarray(code, "L").save(self.path / "masks" / f"{eid}.png")
            entries[eid]["known_px"] = int(k.sum())
            entries[eid]["shared_px"] = int(sh.sum())
        for eid, n in self._seen.items():
            if eid in entries:
                entries[eid]["seen"] = int(entries[eid].get("seen", 1)) + n
        for eid, (ok, bad) in self._outcomes.items():
            if eid in entries:
                entries[eid]["ok"] = int(entries[eid].get("ok", 0)) + ok
                entries[eid]["bad"] = int(entries[eid].get("bad", 0)) + bad
        tmp = self.path / "bank.json.tmp"
        tmp.write_text(json.dumps({"version": BANK_VERSION, "entries": list(entries.values())}, indent=1), encoding="utf-8")
        os.replace(tmp, self.path / "bank.json")
        self._seen, self._outcomes = {}, {}
        return n_new

    def _load_disk(self, eid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        patch = np.asarray(Image.open(self.path / "patches" / f"{eid}.png").convert("RGBA"))
        code = np.asarray(Image.open(self.path / "masks" / f"{eid}.png").convert("L"))
        return patch, code == 255, code == 128

    # -- learning / matching
    def approve(self, alpha: np.ndarray, rgb: np.ndarray, label: str = "") -> None:
        """Remember a frame that a person picked as the best clean version: all its opaque pixels count as known.
        It replaces what was learned for this frame; method FRAME_DONOR only uses such entries."""
        with self._lock:
            self._approved.append((alpha.copy(), rgb.copy(), label))

    def learn_pending(self) -> int:
        return len(self._pending)

    def add(self, alpha: np.ndarray, rgb: np.ndarray, known: np.ndarray, shared: np.ndarray,
              source_image: str = "", region: int = 0) -> None:
        with self._lock:
            self._pending.append((alpha.copy(), rgb.copy(), known.copy(), shared.copy(), source_image, int(region)))

    def match(self, rgb: np.ndarray, alpha: np.ndarray, tile: Rect, zone_tile: np.ndarray,
              approved_only: bool = False) -> BankMatch | None:
        """The remembered frame that fits this tile (same shape, same colours outside the text zones)."""
        if not self._entries:
            return None
        a_t = _crop(alpha, tile)
        t_rgb = _crop(rgb, tile)
        zd = cv2.dilate(zone_tile.astype(np.uint8), np.ones((2 * BANK_ZONE_DILATE_PX + 1,) * 2, np.uint8)) > 0
        best: BankMatch | None = None
        for eid, meta in self._entries.items():
            if tuple(meta.get("window", ())) != (a_t.shape[1], a_t.shape[0]):
                continue
            if approved_only and not meta.get("approved"):
                continue
            if int(meta.get("bad", 0)) > int(meta.get("ok", 0)):
                continue                                          # labelled unusable more often than usable
            patch, known, shared = self._load(eid)
            ok, agree, n = _gate(patch, known, a_t, t_rgb, ~zd)
            if not ok:
                continue
            if best is None or (bool(meta.get("approved")), agree, n) > (best.approved, best.agree, best.gate_pixels):
                best = BankMatch(eid, patch[..., :3], known, shared, agree, n, bool(meta.get("approved")))
        if best is not None:
            with self._lock:
                self._seen[best.entry_id] = self._seen.get(best.entry_id, 0) + 1
        return best

    def assigned_entry(self, source_image: str, region: int) -> BankMatch | None:
        """The approved frame that the donor ranker's group gave to this cell (frame_donor_members.json, written by
        ``frame_donors approve`` next to bank.json), or None. Nothing is read when the file is missing."""
        if not hasattr(self, "_member_label"):
            self._member_label: dict[str, str] = {}
            f = self.path / "frame_donor_members.json"
            if f.is_file():
                for label, info in json.loads(f.read_text(encoding="utf-8")).items():
                    for cell in info.get("members", []):
                        self._member_label.setdefault(cell, label)
        label = self._member_label.get(f"{source_image}#r{int(region)}")
        if not label:
            return None
        for eid, meta in self._entries.items():
            if meta.get("approved") and (label in meta.get("labels", []) or label in meta.get("sources", [])):
                patch, known, shared = self._load(eid)
                return BankMatch(eid, patch[..., :3], known, shared, 1.0, int(known.sum()), True, patch[..., 3])
        return None

    def record_outcome(self, entry_id: str, usable: bool) -> None:
        with self._lock:
            self._outcomes.setdefault(entry_id, [0, 0])[0 if usable else 1] += 1

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[dict]:
        return list(self._entries.values())


# ----------------------------------------------------------------------------- CLI

def _stem(name: str) -> str:
    return Path(name).stem


def apply_labels(bank: SiblingBank, ranking_csv: Path, features_csv: Path) -> tuple[int, int, int]:
    """Feed ranker verdicts back: SIBLING tiles marked usable/unusable update the bank entry they used."""
    entry_of: dict[tuple[str, int], str] = {}
    with features_csv.expanduser().open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") == "SIBLING" and row.get("sibling_entry_id"):
                entry_of[(_stem(row["image"]), int(row["region"]))] = row["sibling_entry_id"]
    ok = bad = skipped = 0
    with ranking_csv.expanduser().open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            eid = entry_of.get((_stem(row["image"]), int(row["region"])))
            if not eid:
                continue
            usable = {x for x in (row.get("usable_ids") or "").split("+") if x}
            unusable = {x for x in (row.get("bad_ids") or "").split("+") if x}
            if "SIBLING" in usable:
                for one in eid.split("+"):
                    bank.record_outcome(one, True)
                ok += 1
            elif "SIBLING" in unusable:
                for one in eid.split("+"):
                    bank.record_outcome(one, False)
                bad += 1
            else:
                skipped += 1
    bank.learn = True            # allow save() to write the counters
    bank.save()
    return ok, bad, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sibling bank tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list"); a.add_argument("--bank", type=Path, required=True)
    b = sub.add_parser("labels"); b.add_argument("--bank", type=Path, required=True)
    b.add_argument("--ranking", type=Path, required=True); b.add_argument("--features", type=Path, required=True)
    args = p.parse_args(argv)
    bank = SiblingBank(args.bank)
    if args.cmd == "list":
        print(f"{len(bank)} Eintraege in {bank.path}")
        for e in sorted(bank.entries(), key=lambda e: e.get("created", "")):
            cmp_px = int(e.get("compared_px", 0))
            rate = f"{100.0 * int(e.get('conflicts', 0)) / cmp_px:.2f} %" if cmp_px else "?"
            print(f"  {e['id']}  Fenster {e.get('window')}  bekannt {e.get('known_px')}  geteilt {e.get('shared_px')}"
                  f"  gesehen {e.get('seen', 1)}  ok {e.get('ok', 0)}  schlecht {e.get('bad', 0)}  Konflikte {e.get('conflicts', 0)}"
                  f"  Merges {e.get('merges', '?')}  Konfliktrate {rate}  Quellen {len(e.get('sources', []))}"
                  + ("  FREIGEGEBEN" if e.get("approved") else ""))
        return 0
    ok, bad, skipped = apply_labels(bank, args.ranking, args.features)
    print(f"Urteile uebernommen: brauchbar {ok}, unbrauchbar {bad}, ohne SIBLING-Urteil {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
