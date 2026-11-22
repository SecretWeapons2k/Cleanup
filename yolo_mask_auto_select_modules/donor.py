"""Stamp fill: copy the same spot from a text-free copy, found in the image or in a donor bank.

Many sheets repeat one sprite (button, text bar) several times; sometimes one copy is empty.
Instead of inventing pixels, the masked pixels are copied from such a clean copy. Nothing is
game specific; a donor is accepted only if ALL of these hold:

  form    the ring around the mask is IDENTICAL at the donor position (proves pixel alignment),
          and that ring has real structure (a flat ring would match everywhere);
  colour  the colours the target has OUTSIDE its text, inside the mask, agree with the donor
          (CIE-Lab delta-E, 95th percentile <= BG_DE_MAX); form alone is not enough;
  clean   the donor's own pixels under the mask contain no text-like structure.

Sources of donors
  find_donors   copies elsewhere in the SAME image (e.g. the empty cell of a text bar)
  DonorBank     clean donors remembered on disk and matched against OTHER images

Bank rules: a run only matches entries that existed when it started (so the result does not
depend on thread timing); entries learned during the run are saved at the end (``learn=True``).
Entries whose stamps were labelled unusable more often than usable are not used any more.

    donors = find_donors(rgb, alpha, region_mask)
    result = stamp(rgb, alpha, region_mask, donors)          # (rgb, alpha, donor) or None
    bank = DonorBank("~/cleanup_compare/donor_bank", learn=True)
    entry_id = bank.add(rgb, alpha, region_mask, donor, source_image="m006_007", region=1)
    matches = bank.match(rgb2, alpha2, region_mask2)         # other image
    bank.save()

CLI:  python -m yolo_mask_auto_select_modules.donor list   --bank DIR
      python -m yolo_mask_auto_select_modules.donor labels --bank DIR --ranking RANK.csv --features stage2_features.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RING_PX = 10                 # width of the ring around the mask that must match exactly
MAX_RING_MSD = 1.0           # mean squared difference per channel over the ring (0 = identical)
MIN_RING_PIXELS = 300        # fewer ring pixels cannot prove alignment
RING_GRAD_LEVEL = 24.0       # Sobel magnitude (about a 6-level step) that counts as structure
MIN_RING_STRUCTURE = 0.03    # share of ring pixels with structure; a flat ring proves nothing
CLEAN_CONTRAST_LEVELS = 14   # thin-stroke contrast that counts as text/structure in a donor
CLEAN_LARGE_LEVELS = 40      # thick-shape contrast (25 px median) that counts as structure in a donor
CLEAN_MAX_FRACTION = 0.002   # share of donor mask pixels above that level allowed
# Structure at the very edge of the mask that is identical in donor and target belongs to the sprite
# (frame glow reaches ~10 px into a box), not to text: it does not count against the donor.
SHARED_EDGE_PX = 4           # distance from the mask outline (inside the mask) in which this applies
SHARED_LEVELS = 2            # donor and target pixel must agree within this many levels
BG_INK_LEVEL = 8             # target pixels above this thin-stroke contrast are "text"
BG_INK_LARGE_LEVEL = 20      # ... or above this thick-shape contrast
BG_INK_DIST = 9              # px around text that are not compared as background
MIN_BG_PIXELS = 80           # fewer background pixels: colour cannot be verified (flag only)
BG_DE_MAX = 2.5              # CIE76 delta-E, 95th percentile, target background vs donor
MAX_DONORS = 12
# Near-exact tier (opt-in, allow_near=True): the ring may differ a little, e.g. the outer edge of a
# strip, but the seam directly around the mask must still be (almost) identical.
NEAR_MSD_MAX = 600.0         # candidates above this are never examined
NEAR_CANDIDATES = 40         # only the best relaxed candidates are examined
NEAR_RING_BAD_MAX = 0.03     # share of ring pixels that may differ clearly
SEAM_PX = 2                  # the band directly around the mask
SEAM_BAD_MAX = 0.005         # share of seam pixels that may differ clearly
BAD_LEVEL = 8                # difference (any channel) that counts as "differs clearly"
BANK_VERSION = 1
BANK_PAD_PX = RING_PX + 2    # context stored around the mask in a bank entry


@dataclass
class Donor:
    dx: int                  # donor position minus target position
    dy: int
    ring_msd: float          # 0.0 = surroundings identical
    structure_frac: float    # share of text-like pixels under the donor's mask area
    clean: bool
    bg_pixels: int = 0       # target pixels inside the mask that are outside its text
    bg_de_p95: float = 0.0   # colour difference of those pixels to the donor (CIE76 delta-E)
    color_ok: bool = True    # False = same form, different colours
    color_checked: bool = False
    tier: str = "exact"      # "exact" (ring identical) or "near" (see NEAR_* constants)
    ring_bad_frac: float = 0.0
    seam_bad_frac: float = 0.0

    @property
    def usable(self) -> bool:
        return self.clean and self.color_ok


# ----------------------------------------------------------------------------- helpers

def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _structure_maps(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local luminance contrast at two scales: thin strokes (9 px median) and thick shapes
    (25 px median), which a small median would mistake for background."""
    lum = rgb.max(axis=2).astype(np.uint8)
    small = np.abs(lum.astype(np.int16) - cv2.medianBlur(lum, 9).astype(np.int16))
    large = np.abs(lum.astype(np.int16) - cv2.medianBlur(lum, 25).astype(np.int16))
    return small, large


def _text_like(maps: tuple[np.ndarray, np.ndarray], small_level: int, large_level: int) -> np.ndarray:
    return (maps[0] > small_level) | (maps[1] > large_level)


def _delta_e(a_rgb: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
    """CIE76 delta-E between two (n, 3) uint8 colour arrays."""
    a = cv2.cvtColor((a_rgb.reshape(-1, 1, 3).astype(np.float32)) / 255.0, cv2.COLOR_RGB2Lab)
    b = cv2.cvtColor((b_rgb.reshape(-1, 1, 3).astype(np.float32)) / 255.0, cv2.COLOR_RGB2Lab)
    return np.sqrt(((a - b) ** 2).sum(axis=2)).ravel()


def _background_pixels(maps: tuple[np.ndarray, np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Mask pixels that are not text and not close to text in the TARGET image."""
    ink = _text_like(maps, BG_INK_LEVEL, BG_INK_LARGE_LEVEL).astype(np.uint8)
    k = 2 * BG_INK_DIST + 1
    near_ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    return mask & ~near_ink


def _colour_check(target_rgb: np.ndarray, donor_rgb_at, bg: np.ndarray) -> tuple[int, float, bool, bool]:
    """(n background pixels, delta-E p95, colour ok, checked). ``donor_rgb_at(ys, xs)`` -> donor colours."""
    ys, xs = np.nonzero(bg)
    if len(ys) < MIN_BG_PIXELS:
        return int(len(ys)), 0.0, True, False
    de = _delta_e(target_rgb[ys, xs], donor_rgb_at(ys, xs))
    p95 = float(np.percentile(de, 95))
    return int(len(ys)), p95, p95 <= BG_DE_MAX, True


def _prepare(mask: np.ndarray, ring_px: int, bounds: tuple[int, int, int, int] | None = None):
    """Window around the mask plus its ring; None when the ring cannot prove an alignment.

    ``bounds`` (x1, y1, x2, y2 inclusive, e.g. the sprite cell from cells.py) cuts the ring off at
    the cell edge: beyond it lie pixels of the neighbouring cell, which say nothing about this sprite
    (and may not even exist at the donor's position, e.g. at the edge of the image)."""
    h, w = mask.shape
    x1, y1, x2, y2 = _bbox(mask)
    wx1, wy1 = max(0, x1 - ring_px), max(0, y1 - ring_px)
    wx2, wy2 = min(w - 1, x2 + ring_px), min(h - 1, y2 + ring_px)
    if bounds is not None:
        bx1, by1, bx2, by2 = bounds
        wx1, wy1 = max(wx1, bx1), max(wy1, by1)
        wx2, wy2 = min(wx2, bx2), min(wy2, by2)
        if wx1 > x1 or wy1 > y1 or wx2 < x2 or wy2 < y2:
            bounds = None                                  # the mask leaves its cell: do not cut
            wx1, wy1 = max(0, x1 - ring_px), max(0, y1 - ring_px)
            wx2, wy2 = min(w - 1, x2 + ring_px), min(h - 1, y2 + ring_px)
    sl = (slice(wy1, wy2 + 1), slice(wx1, wx2 + 1))
    k = 2 * ring_px + 1
    ring = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool) & ~mask
    ring_w = ring[sl]
    if int(ring_w.sum()) < MIN_RING_PIXELS:
        return None
    ks = 2 * SEAM_PX + 1
    seam = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))).astype(bool) & ~mask
    return wx1, wy1, wx2, wy2, sl, ring_w, seam[sl]


def _classify(value: float, diff_fn, ring_w: np.ndarray, seam_w: np.ndarray, max_msd: float, allow_near: bool):
    """("exact" | "near", ring_bad, seam_bad) for a candidate, or None when it does not qualify."""
    if value <= max_msd:
        return "exact", 0.0, 0.0
    if not allow_near or value > NEAR_MSD_MAX:
        return None
    diff = diff_fn()
    ring_bad = float((diff[ring_w] > BAD_LEVEL).mean())
    seam_bad = float((diff[seam_w] > BAD_LEVEL).mean()) if seam_w.any() else 0.0
    if seam_bad <= SEAM_BAD_MAX and ring_bad <= NEAR_RING_BAD_MAX:
        return "near", ring_bad, seam_bad
    return None


def _ring_informative(rgb_window: np.ndarray, ring_w: np.ndarray) -> bool:
    gray = cv2.cvtColor(rgb_window, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return float((mag[ring_w] > RING_GRAD_LEVEL).mean()) >= MIN_RING_STRUCTURE


# ----------------------------------------------------------------------------- same image

def find_donors(
    rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray,
    *, ring_px: int = RING_PX, max_msd: float = MAX_RING_MSD, allow_near: bool = False,
    bounds: tuple[int, int, int, int] | None = None,
) -> list[Donor]:
    """Donor positions whose ring around ``mask`` equals the target's ring; usable ones first.

    With ``allow_near`` also positions whose ring differs a little (tier "near", see NEAR_*)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    h, w = mask.shape
    prep = _prepare(mask, ring_px, bounds)
    if prep is None:
        return []
    wx1, wy1, wx2, wy2, sl, ring_w, seam_w = prep
    if not _ring_informative(rgb[sl], ring_w):
        return []

    img = np.dstack((rgb, alpha)).astype(np.float32)
    templ = np.ascontiguousarray(img[sl])
    th, tw = templ.shape[:2]
    tmask = np.repeat(ring_w[..., None].astype(np.float32), 4, axis=2)
    msd = cv2.matchTemplate(img, templ, cv2.TM_SQDIFF, mask=tmask) / (int(ring_w.sum()) * 4.0)

    maps = _structure_maps(rgb)
    flagged = _text_like(maps, CLEAN_CONTRAST_LEVELS, CLEAN_LARGE_LEVELS)
    bg = _background_pixels(maps, mask)
    my, mx = np.nonzero(mask)
    near_outline = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)[my, mx] <= SHARED_EDGE_PX
    donors: list[Donor] = []
    ys, xs = np.nonzero(msd <= (NEAR_MSD_MAX if allow_near else max_msd))
    relaxed_seen = 0
    for idx in np.argsort(msd[ys, xs]):
        px, py = int(xs[idx]), int(ys[idx])
        dx, dy = px - wx1, py - wy1
        if dx == 0 and dy == 0:
            continue
        if any(abs(dx - d.dx) <= 1 and abs(dy - d.dy) <= 1 for d in donors):
            continue                                   # neighbour of an accepted donor position
        value = max(0.0, float(msd[py, px]))
        if value > max_msd:
            relaxed_seen += 1
            if relaxed_seen > NEAR_CANDIDATES:
                break
        cls = _classify(value, lambda: np.abs(img[py:py + th, px:px + tw] - templ).max(axis=2),
                        ring_w, seam_w, max_msd, allow_near)
        if cls is None:
            continue
        ty, tx = my + dy, mx + dx
        if ty.min() < 0 or tx.min() < 0 or ty.max() >= h or tx.max() >= w:
            continue                                   # donor would reach outside the image
        same = np.abs(rgb[my, mx].astype(np.int16) - rgb[ty, tx].astype(np.int16)).max(axis=1) <= SHARED_LEVELS
        frac = float((flagged[ty, tx] & ~(same & near_outline)).mean())
        n_bg, p95, c_ok, c_checked = _colour_check(rgb, lambda yy, xx: rgb[yy + dy, xx + dx], bg)
        donors.append(Donor(dx, dy, value, frac, frac <= CLEAN_MAX_FRACTION,
                            n_bg, p95, c_ok, c_checked, cls[0], cls[1], cls[2]))
        if len(donors) >= MAX_DONORS:
            break
    donors.sort(key=lambda d: (not d.usable, not d.clean, d.tier != "exact", d.ring_msd, abs(d.dx) + abs(d.dy)))
    return donors


def stamp(
    rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray, donors: list[Donor],
) -> tuple[np.ndarray, np.ndarray, Donor] | None:
    """Copy the masked pixels from the best usable donor (clean AND colour ok); None otherwise."""
    usable = [d for d in donors if d.usable]
    if not usable:
        return None
    d = usable[0]
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    out_rgb, out_alpha = rgb.copy(), alpha.copy()
    out_rgb[ys, xs] = rgb[ys + d.dy, xs + d.dx]
    out_alpha[ys, xs] = alpha[ys + d.dy, xs + d.dx]
    return out_rgb, out_alpha, d


# ----------------------------------------------------------------------------- diagnosis

MISS_FIELDS = ["reason", "ring_pixels", "ring_structure", "best_ring_msd", "best_ring_bad_frac", "best_seam_bad_frac",
               "best_dx", "best_dy", "best_structure_frac", "bank_best_msd"]


def explain_no_donor(rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray,
                     bank: "DonorBank | None" = None, *, ring_px: int = RING_PX,
                     bounds: tuple[int, int, int, int] | None = None) -> dict:
    """Why did this region get no stamp? Returns the fields of MISS_FIELDS (for a report only).

    reason: ring_too_small | ring_flat | no_ring_match | donor_not_clean | colour_mismatch | no_mask
    best_ring_bad_frac is the share of ring pixels that differ clearly (> 8 levels) at the best
    offset: a small value means "almost identical" (e.g. only the strip's outer edge differs).
    """
    out: dict = {k: "" for k in MISS_FIELDS}
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        out["reason"] = "no_mask"
        return out
    prep = _prepare(mask, ring_px, bounds)
    if prep is None:
        out["reason"] = "ring_too_small"
        return out
    wx1, wy1, wx2, wy2, sl, ring_w, seam_w = prep
    n_ring = int(ring_w.sum())
    out["ring_pixels"] = n_ring
    gray = cv2.cvtColor(rgb[sl], cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    out["ring_structure"] = round(float((mag[ring_w] > RING_GRAD_LEVEL).mean()), 4)
    if not _ring_informative(rgb[sl], ring_w):
        out["reason"] = "ring_flat"
        return out
    img = np.dstack((rgb, alpha)).astype(np.float32)
    templ = np.ascontiguousarray(img[sl])
    tmask = np.repeat(ring_w[..., None].astype(np.float32), 4, axis=2)
    msd = cv2.matchTemplate(img, templ, cv2.TM_SQDIFF, mask=tmask) / (n_ring * 4.0)
    msd = np.maximum(msd, 0.0)
    msd[max(0, wy1 - 1):wy1 + 2, max(0, wx1 - 1):wx1 + 2] = np.inf        # the target itself
    py, px = np.unravel_index(int(np.argmin(msd)), msd.shape)
    if np.isfinite(msd[py, px]):
        diff = np.abs(img[py:py + templ.shape[0], px:px + templ.shape[1]] - templ).max(axis=2)
        out.update(best_ring_msd=round(float(msd[py, px]), 3), best_dx=int(px - wx1), best_dy=int(py - wy1),
                   best_ring_bad_frac=round(float((diff[ring_w] > BAD_LEVEL).mean()), 4),
                   best_seam_bad_frac=round(float((diff[seam_w] > BAD_LEVEL).mean()), 4) if seam_w.any() else 0.0)
    if bank is not None and len(bank):
        best = np.inf
        for eid in list(bank._entries):
            patch, _clean = bank._load_patch(eid)
            if patch.shape[0] >= templ.shape[0] and patch.shape[1] >= templ.shape[1]:
                r = cv2.matchTemplate(patch.astype(np.float32), templ, cv2.TM_SQDIFF, mask=tmask) / (n_ring * 4.0)
                best = min(best, float(np.maximum(r, 0.0).min()))
        if np.isfinite(best):
            out["bank_best_msd"] = round(best, 3)
    donors = find_donors(rgb, alpha, mask, ring_px=ring_px, bounds=bounds)
    if not donors:
        out["reason"] = "no_ring_match"
    else:
        out.update(best_structure_frac=round(min(d.structure_frac for d in donors), 4))
        if any(d.usable for d in donors):
            out["reason"] = "usable_donor_exists"
        else:
            out["reason"] = "colour_mismatch" if any(d.clean for d in donors) else "donor_not_clean"
    return out


# ----------------------------------------------------------------------------- bank

@dataclass
class BankMatch:
    entry_id: str
    ring_msd: float
    bg_pixels: int
    bg_de_p95: float
    color_ok: bool
    color_checked: bool
    px: int                  # top-left of the matched window inside the entry's patch
    py: int
    wx1: int                 # top-left of the target window in the target image
    wy1: int
    tier: str = "exact"
    ring_bad_frac: float = 0.0

    @property
    def usable(self) -> bool:
        return self.color_ok


class DonorBank:
    """Clean donors remembered on disk: <dir>/bank.json, patches/<id>.png, masks/<id>.png."""

    def __init__(self, path: str | Path, *, learn: bool = False) -> None:
        self.path = Path(path).expanduser()
        self.learn = learn
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}          # frozen snapshot: used for matching
        self._new: dict[str, dict] = {}              # learned in this run: saved, not matched
        self._new_files: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._seen: dict[str, int] = {}
        self._outcomes: dict[str, list[int]] = {}    # id -> [ok, bad] added in this run
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._entries = self._read_index()

    # -- persistence
    def _read_index(self) -> dict[str, dict]:
        f = self.path / "bank.json"
        if not f.is_file():
            return {}
        data = json.loads(f.read_text(encoding="utf-8"))
        return {e["id"]: e for e in data.get("entries", [])}

    def _load_patch(self, entry_id: str) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if entry_id in self._cache:
                return self._cache[entry_id]
        patch = np.asarray(Image.open(self.path / "patches" / f"{entry_id}.png").convert("RGBA"))
        clean = np.asarray(Image.open(self.path / "masks" / f"{entry_id}.png").convert("L")) > 127
        with self._lock:
            self._cache[entry_id] = (patch, clean)
        return patch, clean

    def save(self) -> int:
        """Write new entries and counters; returns the number of new entries. No-op without learn."""
        if not self.learn and not self._outcomes:
            return 0
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / "patches").mkdir(exist_ok=True)
            (self.path / "masks").mkdir(exist_ok=True)
            entries = self._read_index()                 # merge with what is on disk now
            for eid, meta in self._new.items():
                patch, clean = self._new_files[eid]
                Image.fromarray(patch).save(self.path / "patches" / f"{eid}.png")
                Image.fromarray((clean * 255).astype(np.uint8), "L").save(self.path / "masks" / f"{eid}.png")
                entries.setdefault(eid, meta)
            for eid, n in self._seen.items():
                if eid in entries:
                    entries[eid]["seen"] = int(entries[eid].get("seen", 1)) + n
            for eid, (ok, bad) in self._outcomes.items():
                if eid in entries:
                    entries[eid]["ok"] = int(entries[eid].get("ok", 0)) + ok
                    entries[eid]["bad"] = int(entries[eid].get("bad", 0)) + bad
            tmp = self.path / "bank.json.tmp"
            tmp.write_text(json.dumps({"version": BANK_VERSION, "entries": list(entries.values())}, indent=1),
                           encoding="utf-8")
            os.replace(tmp, self.path / "bank.json")
            n_new = len(self._new)
            self._new, self._new_files, self._seen, self._outcomes = {}, {}, {}, {}
            return n_new

    # -- learning
    def add(self, rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray, donor: Donor,
            *, source_image: str = "", region: int = 0) -> str | None:
        """Remember the clean pixels that ``donor`` supplied for ``mask``; returns the entry id."""
        if not donor.usable:
            return None
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        x1, y1, x2, y2 = _bbox(mask)
        # Context around the mask, clipped so that target window AND donor window stay inside the image.
        wx1, wy1 = max(x1 - BANK_PAD_PX, 0, -donor.dx), max(y1 - BANK_PAD_PX, 0, -donor.dy)
        wx2, wy2 = min(x2 + BANK_PAD_PX, w - 1, w - 1 - donor.dx), min(y2 + BANK_PAD_PX, h - 1, h - 1 - donor.dy)
        if wx1 > x1 or wy1 > y1 or wx2 < x2 or wy2 < y2:
            return None                                   # the donor window would not cover the mask
        dx1, dy1, dx2, dy2 = wx1 + donor.dx, wy1 + donor.dy, wx2 + donor.dx, wy2 + donor.dy
        patch = np.ascontiguousarray(np.dstack((rgb, alpha))[dy1:dy2 + 1, dx1:dx2 + 1])
        clean = np.ascontiguousarray(mask[wy1:wy2 + 1, wx1:wx2 + 1])
        eid = hashlib.sha1(patch.tobytes() + clean.tobytes() + str(patch.shape).encode()).hexdigest()[:12]
        with self._lock:
            known = eid in self._entries or eid in self._new
            if known:
                self._seen[eid] = self._seen.get(eid, 0) + 1
                return eid
            if not self.learn:
                return eid                                # id only (for the report), nothing stored
            self._new[eid] = {
                "id": eid, "source_image": source_image, "source_region": int(region),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "window": [int(patch.shape[1]), int(patch.shape[0])],
                "mask_px": int(clean.sum()), "seen": 1, "ok": 0, "bad": 0,
            }
            self._new_files[eid] = (patch, clean)
        return eid

    def record_outcome(self, entry_id: str, usable: bool) -> None:
        with self._lock:
            ok_bad = self._outcomes.setdefault(entry_id, [0, 0])
            ok_bad[0 if usable else 1] += 1

    # -- matching
    def match(self, rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray,
              *, ring_px: int = RING_PX, max_msd: float = MAX_RING_MSD, allow_near: bool = False,
              bounds: tuple[int, int, int, int] | None = None) -> list[BankMatch]:
        """Entries that fit ``mask`` in this image (form identical, colour ok); best first."""
        mask = np.asarray(mask, dtype=bool)
        if not mask.any() or not self._entries:
            return []
        prep = _prepare(mask, ring_px, bounds)
        if prep is None:
            return []
        wx1, wy1, wx2, wy2, sl, ring_w, seam_w = prep
        if not _ring_informative(rgb[sl], ring_w):
            return []
        templ = np.ascontiguousarray(np.dstack((rgb, alpha)).astype(np.float32)[sl])
        tmask = np.repeat(ring_w[..., None].astype(np.float32), 4, axis=2)
        n_ring = int(ring_w.sum())
        th, tw = templ.shape[:2]
        my, mx = np.nonzero(mask)
        bg = _background_pixels(_structure_maps(rgb), mask)
        found: list[BankMatch] = []
        for eid, meta in self._entries.items():
            if int(meta.get("bad", 0)) > int(meta.get("ok", 0)):
                continue                                  # labelled unusable more often than usable
            patch, clean = self._load_patch(eid)
            if patch.shape[0] < th or patch.shape[1] < tw:
                continue
            patch_f = patch.astype(np.float32)
            res = cv2.matchTemplate(patch_f, templ, cv2.TM_SQDIFF, mask=tmask) / (n_ring * 4.0)
            ys, xs = np.nonzero(res <= (NEAR_MSD_MAX if allow_near else max_msd))
            for idx in np.argsort(res[ys, xs])[:6]:
                px, py = int(xs[idx]), int(ys[idx])
                value = max(0.0, float(res[py, px]))
                cls = _classify(value, lambda: np.abs(patch_f[py:py + th, px:px + tw] - templ).max(axis=2),
                                ring_w, seam_w, max_msd, allow_near)
                if cls is None:
                    continue
                qy, qx = my - wy1 + py, mx - wx1 + px
                if not clean[qy, qx].all():
                    continue                              # the entry is only verified under its own mask
                n_bg, p95, c_ok, c_checked = _colour_check(
                    rgb, lambda yy, xx: patch[yy - wy1 + py, xx - wx1 + px, :3], bg)
                found.append(BankMatch(eid, value, n_bg, p95, c_ok, c_checked, px, py, wx1, wy1, cls[0], cls[1]))
        found.sort(key=lambda m: (not m.usable, m.tier != "exact", m.ring_msd, -int(self._entries[m.entry_id].get("ok", 0))))
        return found

    def stamp(self, rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray,
              match: BankMatch) -> tuple[np.ndarray, np.ndarray]:
        patch, _clean = self._load_patch(match.entry_id)
        ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
        qy, qx = ys - match.wy1 + match.py, xs - match.wx1 + match.px
        out_rgb, out_alpha = rgb.copy(), alpha.copy()
        out_rgb[ys, xs] = patch[qy, qx, :3]
        out_alpha[ys, xs] = patch[qy, qx, 3]
        return out_rgb, out_alpha

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[dict]:
        return list(self._entries.values())


# ----------------------------------------------------------------------------- CLI

def _stem(name: str) -> str:
    return Path(name).stem


def apply_labels(bank: DonorBank, ranking_csv: Path, features_csv: Path) -> tuple[int, int, int]:
    """Feed ranker verdicts back: DONOR tiles marked usable/unusable update their bank entries."""
    entry_of: dict[tuple[str, int], str] = {}
    with features_csv.expanduser().open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") == "DONOR" and row.get("donor_entry_id"):
                entry_of[(_stem(row["image"]), int(row["region"]))] = row["donor_entry_id"]
    ok = bad = skipped = 0
    with ranking_csv.expanduser().open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (_stem(row["image"]), int(row["region"]))
            eid = entry_of.get(key)
            if not eid:
                continue
            usable = {x for x in (row.get("usable_ids") or "").split("+") if x}
            unusable = {x for x in (row.get("bad_ids") or "").split("+") if x}
            if "DONOR" in usable:
                bank.record_outcome(eid, True); ok += 1
            elif "DONOR" in unusable:
                bank.record_outcome(eid, False); bad += 1
            else:
                skipped += 1
    bank.learn = True            # allow save() to write the counters
    bank.save()
    return ok, bad, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Donor bank tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list"); a.add_argument("--bank", type=Path, required=True)
    b = sub.add_parser("labels"); b.add_argument("--bank", type=Path, required=True)
    b.add_argument("--ranking", type=Path, required=True); b.add_argument("--features", type=Path, required=True)
    args = p.parse_args(argv)
    bank = DonorBank(args.bank)
    if args.cmd == "list":
        print(f"{len(bank)} Eintraege in {bank.path}")
        for e in sorted(bank.entries(), key=lambda e: e.get("created", "")):
            print(f"  {e['id']}  aus {e.get('source_image')} r{e.get('source_region')}  Fenster {e.get('window')}"
                  f"  gesehen {e.get('seen', 1)}  ok {e.get('ok', 0)}  schlecht {e.get('bad', 0)}")
        return 0
    ok, bad, skipped = apply_labels(bank, args.ranking, args.features)
    print(f"Urteile uebernommen: brauchbar {ok}, unbrauchbar {bad}, ohne DONOR-Urteil {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
