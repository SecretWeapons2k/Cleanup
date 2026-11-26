"""Tests for siblings.py on synthetic sprite sheets (run: python -m pytest test_siblings.py)."""
import numpy as np
try:
    from yolo_mask_auto_select_modules import siblings as S
except ImportError:                                          # run next to the file
    import siblings as S

W, H, GAP = 120, 50, 10


def _frame(tint=(0, 0, 0)):
    y, x = np.mgrid[0:H, 0:W]
    base = np.stack([20 + x * 0.3, 30 + y * 0.8, 60 + x * 0.1 + y * 0.2], axis=-1)
    return np.clip(base + np.array(tint), 0, 255).astype(np.uint8)


def _sheet(n=3, texts=None, tints=None):
    texts = texts or [(10 + 20 * i, 15, 40 + 8 * i, 30) for i in range(n)]   # (x1, y1, x2, y2) per tile
    tints = tints or [(0, 0, 0)] * n
    rgb = np.zeros((H, n * (W + GAP), 3), np.uint8)
    alpha = np.zeros(rgb.shape[:2], np.uint8)
    clean = rgb.copy()
    zone = np.zeros(rgb.shape[:2], bool)
    for i in range(n):
        x0 = i * (W + GAP)
        f = _frame(tints[i])
        clean[:, x0:x0 + W] = f
        rgb[:, x0:x0 + W] = f
        alpha[:, x0:x0 + W] = 255
        x1, y1, x2, y2 = texts[i]
        rgb[y1:y2, x0 + x1:x0 + x2] = (250, 250, 250)         # solid "ink"
        zone[8:42, x0 + 6:x0 + 100] = True                    # the box around all text
    return rgb, alpha, clean, zone


def _region(zone, tile_idx):
    m = np.zeros_like(zone)
    x0 = tile_idx * (W + GAP)
    m[:, x0:x0 + W] = zone[:, x0:x0 + W]
    return m


def test_recovers_background():
    rgb, alpha, clean, zone = _sheet(3)
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone)
    assert isinstance(res, S.SiblingResult)
    m = _region(zone, 0)
    err = np.abs(res.rgb[m].astype(int) - clean[m].astype(int)).max(axis=1)
    assert (err <= 3).mean() > 0.99
    assert res.meta["sibling_count"] == 2


def test_shared_ink_is_not_background():
    texts = [(10, 15, 50, 30), (10, 15, 50, 30), (70, 15, 100, 30)]   # tiles 0 and 1 carry the same ink
    rgb, alpha, clean, zone = _sheet(3, texts=texts)
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone)
    assert isinstance(res, S.SiblingResult)
    m = _region(zone, 0)
    assert (res.rgb[m].min(axis=1) > 200).mean() < 0.01                # no ink left although two cells agree on it


def test_same_text_everywhere_is_refused():
    texts = [(10, 15, 50, 30)] * 3
    rgb, alpha, clean, zone = _sheet(3, texts=texts)
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone)
    assert isinstance(res, S.SiblingMiss) and res.reason in ("sibling_same_text", "too_many_holes")


def test_recoloured_copies_do_not_vote():
    rgb, alpha, clean, zone = _sheet(3, tints=[(0, 0, 0), (40, -20, 10), (-30, 30, 0)])
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone)
    assert isinstance(res, S.SiblingMiss) and res.reason == "too_few_siblings"
    assert res.meta["sibling_colour_rejected"] == 2


def test_two_tiles_are_not_enough():
    rgb, alpha, clean, zone = _sheet(2)
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone)
    assert isinstance(res, S.SiblingMiss) and res.reason == "too_few_siblings"


def test_different_silhouette_is_not_a_sibling():
    rgb, alpha, clean, zone = _sheet(3)
    alpha[:6, 2 * (W + GAP):2 * (W + GAP) + 60] = 0          # corner cut out of the third tile
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone)
    assert isinstance(res, S.SiblingMiss) and res.reason == "too_few_siblings"
    assert res.meta["sibling_alpha_rejected"] == 1


# ----------------------------------------------------------------------------- bank

def _lone(rgb, alpha, i):
    """Keep only tile ``i``: a tile without siblings in its image."""
    a1 = np.zeros_like(alpha)
    x0 = i * (W + GAP)
    a1[:, x0:x0 + W] = alpha[:, x0:x0 + W]
    return a1


def _learned_bank(tmp_path, tints=None):
    rgb, alpha, clean, zone = _sheet(3, tints=tints)
    bank = S.SiblingBank(tmp_path / "bank", learn=True)
    for i in range(3):
        assert isinstance(S.sibling_fill(rgb, alpha, _region(zone, i), zone, bank=bank, source_image="a", region=i + 1), S.SiblingResult)
    assert bank.save() == 1                                    # the three tiles are one frame
    return S.SiblingBank(tmp_path / "bank"), clean


def test_bank_fills_a_lone_tile_of_another_image(tmp_path):
    bank, clean = _learned_bank(tmp_path)
    texts = [(60, 12, 90, 35)]                                 # other text, in another image with one tile only
    rgb, alpha, clean2, zone = _sheet(1, texts=texts)
    assert isinstance(S.sibling_fill(rgb, alpha, _region(zone, 0), zone), S.SiblingMiss)          # without bank: refused
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone, bank=bank)
    assert isinstance(res, S.SiblingResult) and res.meta["sibling_bank_frac"] > 0.8
    m = _region(zone, 0)
    err = np.abs(res.rgb[m].astype(int) - clean2[m].astype(int)).max(axis=1)
    assert (err <= 3).mean() > 0.97


def test_bank_ignores_other_colours(tmp_path):
    bank, _ = _learned_bank(tmp_path)
    rgb, alpha, clean2, zone = _sheet(1, tints=[(50, -20, 30)])
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone, bank=bank)
    assert isinstance(res, S.SiblingMiss) and res.reason == "too_few_siblings"


def test_bank_keeps_frame_ink_and_is_not_poisoned_by_it(tmp_path):
    def header(rgb):
        rgb[10:14, 8:70] = (255, 60, 200)                      # label that belongs to the frame (in every cell, and in every image)
    rgb, alpha, clean, zone = _sheet(3)
    for i in range(3):
        header(rgb[:, i * (W + GAP):i * (W + GAP) + W])
    bank = S.SiblingBank(tmp_path / "bank", learn=True)
    for i in range(3):
        S.sibling_fill(rgb, alpha, _region(zone, i), zone, bank=bank, source_image="a", region=i + 1)
    bank.save()
    bank = S.SiblingBank(tmp_path / "bank")
    rgb2, alpha2, _c, zone2 = _sheet(1, texts=[(60, 20, 90, 35)])
    header(rgb2)
    res = S.sibling_fill(rgb2, alpha2, _region(zone2, 0), zone2, bank=bank)
    assert isinstance(res, S.SiblingResult)
    assert (res.rgb[10:14, 8:70] == (255, 60, 200)).all(axis=-1).mean() > 0.95       # label untouched
    assert (res.rgb[20:35, 60:90].min(axis=-1) > 200).mean() < 0.01                   # text gone


def test_bank_entries_with_bad_labels_are_not_used(tmp_path):
    bank, _ = _learned_bank(tmp_path)
    eid = bank.entries()[0]["id"]
    bank.record_outcome(eid, False)
    bank.learn = True
    bank.save()
    bank = S.SiblingBank(tmp_path / "bank")
    rgb, alpha, clean2, zone = _sheet(1, texts=[(60, 12, 90, 35)])
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone, bank=bank)
    assert isinstance(res, S.SiblingMiss)


def test_shared_glyph_in_body_box_is_text_but_header_stays():
    rgb, alpha, clean, zone = _sheet(3)
    for i in range(3):
        x0 = i * (W + GAP)
        rgb[10:14, x0 + 8:x0 + 70] = (255, 60, 200)               # header label (all cells, outside the body box)
        rgb[24:28, x0 + 80:x0 + 84] = (250, 250, 250)             # the same glyph at the same place in every cell (inside the body box)
    body = np.zeros_like(zone)
    for i in range(3):
        x0 = i * (W + GAP)
        body[18:42, x0 + 6:x0 + 100] = True
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone, body_mask=body)
    assert isinstance(res, S.SiblingResult)
    assert (res.rgb[10:14, 8:70] == (255, 60, 200)).all(axis=-1).mean() > 0.95
    assert res.rgb[24:28, 80:84].min(axis=-1).max() < 200                 # glyph removed
    res2 = S.sibling_fill(rgb, alpha, _region(zone, 0), zone)           # without body info: kept as frame ink
    assert (res2.rgb[24:28, 80:84] == 250).all(axis=-1).mean() > 0.9


def test_same_text_with_body_mask_is_still_refused():
    texts = [(10, 15, 50, 30)] * 3
    rgb, alpha, clean, zone = _sheet(3, texts=texts)
    body = zone.copy()
    res = S.sibling_fill(rgb, alpha, _region(zone, 0), zone, body_mask=body)
    assert isinstance(res, S.SiblingMiss) and res.reason in ("sibling_same_text", "too_many_holes")


def test_tiny_part_without_ink_does_not_veto_the_region():
    rgb, alpha, clean, zone = _sheet(3)
    region = _region(zone, 0)
    region[40:46, (W + GAP) + 10:(W + GAP) + 16] = True       # a sliver of the neighbouring tile, plain frame there
    zone2 = zone | region
    res = S.sibling_fill(rgb, alpha, region, zone2)
    assert isinstance(res, S.SiblingResult) and res.meta["sibling_parts"] == 2


def test_failing_sliver_is_skipped_but_failing_big_part_refuses():
    texts = [(10, 15, 50, 30), (10, 15, 50, 30), (60, 15, 100, 30)]       # tiles 0 and 1 carry the same ink
    rgb, alpha, clean, zone = _sheet(3, texts=texts)
    region = _region(zone, 0)
    sliver = np.zeros_like(zone)
    sliver[16:20, (W + GAP) + 12:(W + GAP) + 30] = True                  # sits on the ink that tiles 0-2 share
    res = S.sibling_fill(rgb, alpha, region | sliver, zone | sliver)
    assert isinstance(res, S.SiblingResult)                              # the sliver alone does not veto the region
    big = _region(zone, 1)
    res2 = S.sibling_fill(rgb, alpha, region | big, zone)
    assert isinstance(res2, (S.SiblingResult, S.SiblingMiss))


# ----------------------------------------------------------------------------- approved frames / FRAME_DONOR

def _approved_bank(tmp_path, tints=None):
    """A bank holding one approved frame: the clean cell of a sheet."""
    rgb, alpha, clean, zone = _sheet(1, tints=tints)
    bank = S.SiblingBank(tmp_path / "bank")
    bank.approve(alpha[:, :W], clean[:, :W], "approved:test")
    assert bank.save() == 1
    return S.SiblingBank(tmp_path / "bank"), clean


def test_frame_donor_stamps_a_cell_with_other_text(tmp_path):
    bank, clean = _approved_bank(tmp_path)
    rgb, alpha, clean2, zone = _sheet(1, texts=[(40, 18, 95, 32)])
    res = S.frame_donor_fill(rgb, alpha, _region(zone, 0), zone, None, bank)
    assert isinstance(res, S.SiblingResult)
    m = _region(zone, 0)
    assert (np.abs(res.rgb[m].astype(int) - clean2[m].astype(int)).max(axis=1) <= 1).all()
    assert (res.rgb[~m] == rgb[~m]).all()


def test_frame_donor_needs_an_approved_entry(tmp_path):
    rgb, alpha, clean, zone = _sheet(3)
    learn = S.SiblingBank(tmp_path / "bank", learn=True)                 # learned, not approved
    for i in range(3):
        S.sibling_fill(rgb, alpha, _region(zone, i), zone, bank=learn, source_image="a", region=i + 1)
    learn.save()
    res = S.frame_donor_fill(rgb, alpha, _region(zone, 0), zone, None, S.SiblingBank(tmp_path / "bank"))
    assert isinstance(res, S.SiblingMiss) and res.reason == "no_frame_donor"


def test_frame_donor_ignores_other_colours(tmp_path):
    bank, _ = _approved_bank(tmp_path)
    rgb, alpha, clean2, zone = _sheet(1, tints=[(50, -20, 30)])
    res = S.frame_donor_fill(rgb, alpha, _region(zone, 0), zone, None, bank)
    assert isinstance(res, S.SiblingMiss) and res.reason == "no_frame_donor"


def test_approving_again_replaces_the_frame_instead_of_adding_one(tmp_path):
    bank, clean = _approved_bank(tmp_path)
    rgb, alpha, clean2, zone = _sheet(1)
    bank = S.SiblingBank(tmp_path / "bank")
    better = clean[:, :W].copy(); better[40:44, 10:20] = (9, 9, 9)       # the person chose another version later
    bank.approve(alpha[:, :W], better, "approved:v2")
    assert bank.save() == 0 and len(S.SiblingBank(tmp_path / "bank")) == 1
    res = S.frame_donor_fill(rgb, alpha, _region(zone, 0), zone, None, S.SiblingBank(tmp_path / "bank"))
    assert isinstance(res, S.SiblingResult)
