"""Tests for frame_donors.py on a synthetic stage 2 run (python test_frame_donors.py or pytest)."""
import csv, json, subprocess, shutil, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
try:
    from yolo_mask_auto_select_modules import frame_donors as FD, siblings as S
except ImportError:
    import frame_donors as FD, siblings as S
import test_siblings as T

W, H = T.W, T.H


def make_run(tmp: Path, n_images=3, tints=((0, 0, 0), (40, -20, 10))):
    """Fake stage 2 run: every image has one tile per tint (two frames), a BOX_CELL result per region and an original."""
    reg = tmp / "run" / "debug_candidates" / "regions"
    reg.mkdir(parents=True)
    rows = []
    for i in range(n_images):
        texts = [(10 + 7 * i, 15, 45 + 7 * i, 30), (50 - 5 * i, 18, 90 - 5 * i, 33)]
        rgb, alpha, clean, zone = T._sheet(len(tints), texts=texts, tints=list(tints))
        name = f"m999_{i:03d}.png"
        stem = Path(name).stem
        index = {"image": name, "regions": []}
        for k in range(len(tints)):
            sl = (slice(0, H), slice(k * (W + T.GAP), k * (W + T.GAP) + W))
            orig = np.dstack((rgb[sl], alpha[sl]))
            good = np.dstack((clean[sl], alpha[sl]))                      # a perfect result
            smear = good.copy(); smear[20:30, 30:90, :3] = (60, 60, 120)   # a worse one
            Image.fromarray(orig).save(reg / f"{stem}__r{k+1}__original.png")
            Image.fromarray(good).save(reg / f"{stem}__r{k+1}__BOX_CELL.png")
            Image.fromarray(smear).save(reg / f"{stem}__r{k+1}__BOX_ONLY.png")
            index["regions"].append({"region": k + 1, "bbox": [0, 0, 1, 1], "window": [0, 0, W - 1, H - 1], "window_source": "cell",
                                     "candidates": {"BOX_ONLY": f"{stem}__r{k+1}__BOX_ONLY.png", "BOX_CELL": f"{stem}__r{k+1}__BOX_CELL.png"},
                                     "original": f"{stem}__r{k+1}__original.png"})
            rows.append((name, k + 1, "pick", "BOX_CELL" if (i + k) % 2 == 0 else "BOX_ONLY"))
        (reg / f"{stem}.json").write_text(json.dumps(index), encoding="utf-8")
    ranking = tmp / "ranking.csv"
    with ranking.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "region", "status", "chosen"]); w.writerows(rows)
    return tmp / "run", ranking


def test_groups_follow_shape_and_colour(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    run, ranking = make_run(tmp)
    members = FD.load_members(run, ranking)
    groups = FD.build_groups(members)
    assert sorted(len(g) for g in groups) == [3, 3]                       # two frames (two colours), 3 images each
    assert all(len({m.image for m in g}) == 3 for g in groups)


def test_ranker_lists_chosen_results_best_first_and_html_is_valid_js(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    run, ranking = make_run(tmp)
    idx = FD.build_ranker(run, ranking)
    data = json.loads((idx.parent / "donor_groups.json").read_text(encoding="utf-8"))
    assert len(data["groups"]) == 2
    g = data["groups"][0]
    assert g["members_total"] == 3 and 1 <= len(g["candidates"]) <= 3
    assert all((idx.parent / c["file"]).is_file() for c in g["candidates"])
    # the smoother (perfect) results come before the smeared ones
    methods = [c["method"] for c in g["candidates"]]
    assert methods == sorted(methods, key=lambda m: m != "BOX_CELL")
    html = idx.read_text(encoding="utf-8")
    script = html.split("<script>")[1].split("</script>")[0]
    (tmp / "page.js").write_text(script, encoding="utf-8")
    if shutil.which("node"):
        r = subprocess.run(["node", "--check", str(tmp / "page.js")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_approve_then_frame_donor_cleans_a_new_cell(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    run, ranking = make_run(tmp)
    idx = FD.build_ranker(run, ranking)
    data = json.loads((idx.parent / "donor_groups.json").read_text(encoding="utf-8"))
    choices = tmp / "donor_choices.csv"
    with choices.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "status", "chosen_file", "image", "region", "method", "members_total", "candidates"])
        for g in data["groups"]:
            c = g["candidates"][0]
            w.writerow([g["id"], "pick", c["file"], c["image"], c["region"], c["method"], g["members_total"], len(g["candidates"])])
    ok, bad = FD.approve(idx.parent, choices, tmp / "bank")
    assert (ok, bad) == (2, 0)
    bank = S.SiblingBank(tmp / "bank")
    assert len(bank) == 2 and all(e.get("approved") for e in bank.entries())
    # a brand new image with the first frame and text somewhere else
    rgb, alpha, clean, zone = T._sheet(1, texts=[(55, 20, 100, 36)], tints=[(0, 0, 0)])
    res = S.frame_donor_fill(rgb, alpha, T._region(zone, 0), zone, None, bank)
    assert isinstance(res, S.SiblingResult)
    m = T._region(zone, 0)
    assert (np.abs(res.rgb[m].astype(int) - clean[m].astype(int)).max(axis=1) <= 1).all()


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
