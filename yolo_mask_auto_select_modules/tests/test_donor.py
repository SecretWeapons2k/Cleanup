"""Tests for donor.py (synthetic strips). Place in yolo_mask_auto_select_modules/tests/."""
from __future__ import annotations

import csv

import numpy as np

from yolo_mask_auto_select_modules import donor


def _strip(n=4, cw=120, h=60, empty=(3,), seed=0, sprite_seed=0, tint=0):
    """n identical framed cells with a noisy background; cells not in ``empty`` carry text blobs."""
    rng = np.random.default_rng(sprite_seed)
    cell = np.zeros((h, cw, 3), np.uint8)
    cell[:] = 20
    cell[:3] = cell[-3:] = 120
    cell[:, :3] = cell[:, -3:] = 120                                  # frame
    cell = np.clip(cell.astype(int) + rng.integers(-3, 4, cell.shape), 0, 255).astype(np.uint8)
    if tint:
        cell[10:-10, 10:-10, 2] = np.clip(cell[10:-10, 10:-10, 2].astype(int) + tint, 0, 255)   # other background colour inside the mask, ring unchanged
    rgb = np.tile(cell, (1, n, 1))
    rng = np.random.default_rng(seed + 1000)                           # text differs per image
    for i in range(n):
        if i in empty:
            continue
        for _ in range(4):
            x = i * cw + int(rng.integers(15, cw - 45)); y = int(rng.integers(15, h - 25))
            rgb[y:y + 10, x:x + 30] = 235                              # bright "text"
    alpha = np.full(rgb.shape[:2], 255, np.uint8)
    return rgb, alpha


def _mask(shape, i, cw=120, inset=10):
    m = np.zeros(shape, bool)
    m[inset:shape[0] - inset, i * cw + inset:(i + 1) * cw - inset] = True
    return m


def test_finds_and_stamps_the_empty_copy():
    rgb, alpha = _strip()
    m = _mask(alpha.shape, 0)
    donors = donor.find_donors(rgb, alpha, m)
    clean = [d for d in donors if d.clean]
    assert len(clean) == 1 and (clean[0].dx, clean[0].dy) == (360, 0) and clean[0].ring_msd == 0.0
    out_rgb, out_alpha, used = donor.stamp(rgb, alpha, m, donors)
    assert np.array_equal(out_rgb[m], rgb[_mask(alpha.shape, 3)])       # identical to the empty cell
    assert np.array_equal(out_rgb[~m], rgb[~m])                         # nothing outside the mask changes


def test_copies_with_text_are_not_clean_donors():
    rgb, alpha = _strip(empty=())
    donors = donor.find_donors(rgb, alpha, _mask(alpha.shape, 1))
    assert donors and not any(d.clean for d in donors)
    assert donor.stamp(rgb, alpha, _mask(alpha.shape, 1), donors) is None


def test_misaligned_copy_is_rejected():
    rgb, alpha = _strip()
    shifted = rgb.copy()
    shifted[:, 360:] = np.roll(rgb[:, 360:], 1, axis=0)                 # the empty copy is off by one pixel
    donors = donor.find_donors(shifted, alpha, _mask(alpha.shape, 0))
    assert not any(d.clean for d in donors)


def test_single_cell_has_no_donor():
    rgb, alpha = _strip(n=1, empty=())
    assert donor.find_donors(rgb, alpha, _mask(alpha.shape, 0)) == []


def test_tiny_mask_without_enough_ring_gives_nothing():
    rgb, alpha = _strip()
    m = np.zeros(alpha.shape, bool); m[30:32, 30:32] = True
    assert donor.find_donors(rgb, alpha, m) == []


# ---- colour -------------------------------------------------------------------------------

def test_same_form_but_other_background_colour_is_not_usable():
    rgb, alpha = _strip(empty=(3,))
    tinted = rgb.copy()
    tinted[10:-10, 370:470, 2] += 6                                    # slightly bluer background (delta-E ~4), no edge
    donors = donor.find_donors(tinted, alpha, _mask(alpha.shape, 0))
    best = [d for d in donors if d.clean]
    assert best and best[0].ring_msd < 1.0 and not best[0].color_ok     # frame identical, colour differs
    assert donor.stamp(tinted, alpha, _mask(alpha.shape, 0), donors) is None


def test_flat_ring_proves_no_alignment():
    rgb = np.full((60, 480, 3), 20, np.uint8)
    alpha = np.full((60, 480), 255, np.uint8)
    assert donor.find_donors(rgb, alpha, _mask(alpha.shape, 0)) == []


def test_thick_bright_shape_in_a_donor_is_not_clean():
    rgb, alpha = _strip(empty=(3,))
    rgb[15:45, 360 + 20:360 + 100] = 230                               # a big logo in the "empty" copy
    donors = donor.find_donors(rgb, alpha, _mask(alpha.shape, 0))
    assert not any(d.clean and d.dx == 360 for d in donors)


# ---- bank ---------------------------------------------------------------------------------

def _learn(tmp_path, seed=0):
    rgb, alpha = _strip(seed=seed)
    m = _mask(alpha.shape, 0)
    donors = donor.find_donors(rgb, alpha, m)
    bank = donor.DonorBank(tmp_path, learn=True)
    eid = bank.add(rgb, alpha, m, donors[0], source_image="a", region=1)
    assert eid and bank.save() == 1
    return eid


def test_bank_cleans_another_image_with_no_empty_copy(tmp_path):
    eid = _learn(tmp_path, seed=0)
    other, alpha = _strip(empty=(), seed=7)                            # same sprite, other text, no empty cell
    bank = donor.DonorBank(tmp_path)
    m = _mask(alpha.shape, 1)
    matches = bank.match(other, alpha, m)
    assert matches and matches[0].entry_id == eid and matches[0].usable and matches[0].ring_msd == 0.0
    out_rgb, out_alpha = bank.stamp(other, alpha, m, matches[0])
    clean, _ = _strip(seed=0)
    assert np.array_equal(out_rgb[m], clean[_mask(alpha.shape, 3)])    # identical to the empty cell
    assert np.array_equal(out_rgb[~m], other[~m])


def test_bank_rejects_other_background_colour(tmp_path):
    _learn(tmp_path)
    other, alpha = _strip(empty=(), seed=7, tint=6)                    # same form, slightly other background colour
    bank = donor.DonorBank(tmp_path)
    got = bank.match(other, alpha, _mask(alpha.shape, 1))
    assert not any(x.usable for x in got)


def test_bank_ignores_entries_learned_in_the_same_run(tmp_path):
    rgb, alpha = _strip()
    m = _mask(alpha.shape, 0)
    bank = donor.DonorBank(tmp_path, learn=True)
    bank.add(rgb, alpha, m, donor.find_donors(rgb, alpha, m)[0])
    other, _ = _strip(empty=(), seed=7)
    assert bank.match(other, alpha, _mask(alpha.shape, 1)) == []       # frozen snapshot: result is order independent
    bank.save()
    assert donor.DonorBank(tmp_path).match(other, alpha, _mask(alpha.shape, 1))


def test_same_donor_is_stored_once(tmp_path):
    rgb, alpha = _strip()
    m = _mask(alpha.shape, 0)
    d = donor.find_donors(rgb, alpha, m)[0]
    bank = donor.DonorBank(tmp_path, learn=True)
    a = bank.add(rgb, alpha, m, d); b = bank.add(rgb, alpha, m, d)
    assert a == b and bank.save() == 1 and len(donor.DonorBank(tmp_path)) == 1


def test_bad_verdicts_switch_an_entry_off(tmp_path):
    eid = _learn(tmp_path)
    bank = donor.DonorBank(tmp_path)
    bank.record_outcome(eid, False)
    bank.learn = True
    bank.save()
    other, alpha = _strip(empty=(), seed=7)
    assert donor.DonorBank(tmp_path).match(other, alpha, _mask(alpha.shape, 1)) == []


def test_ranker_labels_update_the_bank(tmp_path):
    eid = _learn(tmp_path / "bank")
    feats = tmp_path / "f.csv"; rank = tmp_path / "r.csv"
    feats.write_text(f"image,region,method,donor_entry_id\nm1.png,1,DONOR,{eid}\nm1.png,1,SMART_INK,\n", encoding="utf-8")
    rank.write_text("image,region,usable_ids,bad_ids\nm1,1,BOX_ONLY+DONOR,SMART_INK\n", encoding="utf-8")
    ok, bad, _ = donor.apply_labels(donor.DonorBank(tmp_path / "bank"), rank, feats)
    assert (ok, bad) == (1, 0)
    assert donor.DonorBank(tmp_path / "bank").entries()[0]["ok"] == 1
