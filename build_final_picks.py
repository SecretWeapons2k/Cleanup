#!/usr/bin/env python3
"""Combine your hand labels with the ranker into one final pick per image.

Priority per image:
  1. You labeled it with at least one acceptable candidate
       -> use the ranker's pick if you accepted it, else its runner-up if accepted,
          else the first accepted candidate in a fixed preference order. (source=label)
  2. You marked it "none good"            -> status needs_mask_fix, no winner copied
  3. You skipped it                       -> status needs_review
  4. Not labeled                          -> ranker pick (source=ranker);
                                             low-confidence ones get status needs_review

    python build_final_picks.py --run-dir <RUN> [--copy]

--copy gathers winning images into <RUN>/final_winners/ (copies, never moves/deletes
anything) and writes <RUN>/final_picks.csv.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

PREFER = ["HYBRID_D6", "HYBRID_D2", "BOX_ONLY", "2", "4", "6", "8", "10", "0",
          "CONTRAST_T38", "SMART_INK"]
FILE_SUFFIX = {"BOX_ONLY": "box_only", "CONTRAST_T38": "contrast_t38", "HYBRID_D2": "hybrid_d2",
               "HYBRID_D6": "hybrid_d6", "SMART_INK": "smart_ink"}


def cand_file(image_stem: str, cand: str) -> str:
    suf = FILE_SUFFIX.get(cand) or f"distance_{int(cand):02d}px"
    return f"{image_stem}__{suf}.png"


def stem(s: str) -> str:
    return Path(str(s)).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ranker-picks", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--copy", action="store_true", help="copy winning images to final_winners/")
    a = ap.parse_args()
    run = a.run_dir
    picks = pd.read_csv(a.ranker_picks or run / "ranker_out" / "ranker_picks.csv", encoding="utf-8-sig")
    labels = pd.read_csv(a.labels or run / "labels.csv", encoding="utf-8-sig")
    picks["img"] = picks["image"].map(stem)
    labels["img"] = labels["image"].map(stem)
    labels["candidate"] = labels["candidate"].astype(str)

    lab = {}
    for img, d in labels.groupby("img"):
        lab[img] = {"good": set(d.loc[d.good == 1, "candidate"]),
                    "none": int(d.none_good.max()) == 1, "skip": int(d.skipped.max()) == 1}

    rows = []
    for r in picks.itertuples():
        img = r.img
        pick, runner = str(r.ranker_pick), str(r.runner_up)
        row = {"image": img + ".png", "winner": "", "source": "", "status": "", "ranker_pick": pick,
               "top_prob": r.top_prob}
        L = lab.get(img)
        if L is None:
            row.update(winner=pick, source="ranker",
                       status="needs_review" if r.needs_review else "auto_accepted")
        elif L["none"]:
            row.update(source="label", status="needs_mask_fix")
        elif L["skip"] and not L["good"]:
            row.update(source="label", status="needs_review")
        else:
            good = L["good"]
            if pick in good:
                w = pick
            elif runner in good:
                w = runner
            else:
                w = next((c for c in PREFER if c in good), sorted(good)[0] if good else "")
            row.update(winner=w, source="label", status="labeled_ok" if w else "needs_review")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(run / "final_picks.csv", index=False, encoding="utf-8-sig")

    print(out.status.value_counts().to_string())
    print(f"\nWrote {run/'final_picks.csv'}")

    if a.copy:
        src = run / "debug_candidates" / "cleaned_images"
        dst = run / "final_winners"
        dst.mkdir(exist_ok=True)
        n = miss = 0
        for r in out[out.winner != ""].itertuples():
            f = src / cand_file(Path(r.image).stem, str(r.winner))
            if f.is_file():
                shutil.copy2(f, dst / (Path(r.image).stem + ".png")); n += 1
            else:
                miss += 1
        print(f"Copied {n} winners to {dst} ({miss} source files missing)")
        for status in ("needs_mask_fix", "needs_review"):
            lst = out[out.status == status].image.tolist()
            (run / f"list_{status}.txt").write_text("\n".join(lst), encoding="utf-8")
        print("Wrote list_needs_mask_fix.txt and list_needs_review.txt")


if __name__ == "__main__":
    main()
