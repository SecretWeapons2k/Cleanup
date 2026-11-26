"""Tests for lama_cell.py with a fake LaMa (run: python test_lama_cell.py or pytest)."""
import numpy as np
try:
    from yolo_mask_auto_select_modules import lama_cell as L
except ImportError:
    import lama_cell as L


def _scene():
    h, w = 60, 200
    rgb = np.zeros((h, w, 3), np.uint8); rgb[...] = (10, 20, 40)
    alpha = np.full((h, w), 255, np.uint8)
    rgb[20:30, 20:60] = (250, 250, 250)        # text of region 1
    rgb[20:30, 120:160] = (250, 250, 250)      # text of region 2
    mask = np.zeros((h, w), bool); mask[15:35, 15:65] = True; mask[15:35, 115:165] = True
    territory = np.ones((h, w), np.int32); territory[:, 100:] = 2
    windows = [(0, 0, 99, 59), (100, 0, 199, 59)]
    return rgb, alpha, mask, territory, windows


def _fake_lama(job):
    """What LaMa would do: fill the masked pixels with the unmasked mean of the crop."""
    keep = job.mask == 0
    fill = job.rgb[keep].mean(axis=0).astype(np.uint8)
    out = job.rgb.copy(); out[job.mask > 0] = fill
    return out, job.alpha_rgb[..., 0]


def test_jobs_cover_regions_and_crop_correctly():
    rgb, alpha, mask, terr, wins = _scene()
    jobs = L.make_jobs(rgb, alpha, mask, terr, [1, 2], wins)
    assert [j.region for j in jobs] == [1, 2]
    assert jobs[0].rgb.shape == (60, 100, 3) and jobs[1].win == (100, 0, 199, 59)
    assert jobs[0].own.sum() == 20 * 50 and jobs[1].own.sum() == 20 * 50


def test_neighbour_mask_inside_window_is_in_the_crop_mask_but_not_pasted():
    rgb, alpha, mask, terr, _ = _scene()
    wins = [(0, 0, 199, 59), (100, 0, 199, 59)]               # window of region 1 reaches into region 2
    jobs = L.make_jobs(rgb, alpha, mask, terr, [1, 2], wins)
    j1 = jobs[0]
    assert (j1.mask[15:35, 115:165] == 255).all()              # neighbour text masked in the crop ...
    assert not j1.own[15:35, 115:165].any()                    # ... but never pasted


def test_paste_replaces_only_own_territory_and_only_masked_pixels():
    rgb, alpha, mask, terr, wins = _scene()
    jobs = L.make_jobs(rgb, alpha, mask, terr, [1, 2], wins)
    out_rgb, out_alpha, st = rgb.copy(), alpha.copy(), np.zeros(mask.shape, bool)
    for j in jobs:
        p_rgb, p_alpha = _fake_lama(j)
        assert L.paste(j, p_rgb, p_alpha, out_rgb, out_alpha, st)
    assert (out_rgb[mask] == (10, 20, 40)).all()               # text gone
    assert (out_rgb[~mask] == rgb[~mask]).all()                # everything else untouched
    assert (st == mask).all()


def test_wrong_size_or_missing_prediction_is_refused():
    rgb, alpha, mask, terr, wins = _scene()
    j = L.make_jobs(rgb, alpha, mask, terr, [1], wins)[0]
    out_rgb, out_alpha, st = rgb.copy(), alpha.copy(), np.zeros(mask.shape, bool)
    assert not L.paste(j, None, None, out_rgb, out_alpha, st)
    assert not L.paste(j, np.zeros((5, 5, 3), np.uint8), np.zeros((5, 5), np.uint8), out_rgb, out_alpha, st)
    assert not st.any() and (out_rgb == rgb).all()


def test_region_without_mask_in_its_territory_gets_no_job():
    rgb, alpha, mask, terr, wins = _scene()
    mask[:, 100:] = False
    assert [j.region for j in L.make_jobs(rgb, alpha, mask, terr, [1, 2], wins)] == [1]


def test_merge_integrity_takes_the_worst():
    assert L.merge_integrity([{"a": 0.1, "b": 2.0}, {"a": 0.5, "b": 1.0}]) == {"a": 0.5, "b": 2.0}


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
