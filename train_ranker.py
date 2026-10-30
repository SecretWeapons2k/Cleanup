#!/usr/bin/env python3
"""Train/evaluate a candidate ranker from hand labels (standalone, no pipeline changes).

    pip install scikit-learn pandas      # if missing in the venv
    python train_ranker.py --scores candidate_scores.csv --labels labels.csv \
        [--winner-report winner_report.csv] --out ranker_out

Outputs (in --out):
  cv_report.txt        grouped cross-validation results vs. baselines
  ranker_picks.csv     one row per image: ranker pick, confidence, review flag
  ranker.pkl           final model (trained on all usable labels)

Evaluation unit: "is the picked candidate one you marked acceptable?" measured on
images the model never saw during training (grouped K-fold, repeated).
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CANDS = ["0", "2", "4", "6", "8", "10", "HYBRID_D2", "HYBRID_D6",
         "CONTRAST_T38", "SMART_INK", "BOX_ONLY"]
REL_COLS = [
    "text_quality_score", "score", "structure_score", "visual_text_residual_score",
    "residual_text_detail_ratio", "residual_inside_peak", "residual_inside_energy",
    "sprite_contour_displacement_ratio", "sprite_texture_distortion_ratio",
    "residual_blob_contrast_ratio", "outline_leak_score", "boundary_color_artifact_ratio",
    "max_text_anchor_retained_detail_ratio", "residual_micro_component_area",
    "external_edge_destruction_score",
]
NON_FEATURES = {"image", "distance_px", "img", "cand", "good", "none_good", "skipped"}


def stem(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)


def build_features(S: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    num = [c for c in S.columns if c not in NON_FEATURES
           and pd.api.types.is_numeric_dtype(S[c]) and S[c].nunique() > 1]
    X = S[num].fillna(0.0).copy()
    g = S["img"]
    rel = {}
    for c in REL_COLS:
        if c in X:
            v = X[c]
            rel[c + "_dmin"] = v - v.groupby(g).transform("min")
            rel[c + "_dmed"] = v - v.groupby(g).transform("median")
            rel[c + "_rank"] = v.groupby(g).rank(pct=True)
    X = pd.concat([X, pd.DataFrame(rel)], axis=1)
    for c in CANDS:
        X["is_" + c] = (S["cand"] == c).astype(int)
    return X, list(X.columns)


def make_hgb(seed=0):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                          l2_regularization=1.0, random_state=seed)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))


def pick(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.loc[df.groupby("img")[col].idxmax()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--winner-report", type=Path, help="current selector's choices (optional)")
    ap.add_argument("--out", type=Path, default=Path("ranker_out"))
    ap.add_argument("--repeats", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep: list[str] = []

    def say(s=""):
        print(s); rep.append(s)

    S = pd.read_csv(a.scores, encoding="utf-8-sig")
    S["img"] = stem(S["image"]); S["cand"] = S["distance_px"].astype(str)
    L = pd.read_csv(a.labels, encoding="utf-8-sig")
    L["img"] = stem(L["image"]); L["cand"] = L["candidate"].astype(str)

    X_all, feats = build_features(S)
    S = S.reset_index(drop=True)
    D = S.merge(L[["img", "cand", "good", "none_good", "skipped"]], on=["img", "cand"], how="left")
    D = D.reset_index(drop=True)
    labeled_imgs = set(L["img"])
    D["is_labeled"] = D["img"].isin(labeled_imgs)

    lab = D[D.is_labeled]
    per = lab.groupby("img").agg(g=("good", "sum"), n=("good", "count"),
                                 none=("none_good", "max"), skip=("skipped", "max"))
    usable = per[(per.g > 0) & (per.g < per.n) & (per.none == 0) & (per.skip == 0)].index
    say(f"Labeled images: {len(per)} | none-good: {int(per.none.sum())} | skipped: {int(per.skip.sum())}"
        f" | all-candidates-good: {int((per.g == per.n).sum())} | usable for ranking: {len(usable)}")
    T = D[D.img.isin(usable)].copy()
    X = X_all.loc[T.index]
    y = T["good"].astype(int).values
    imgs = T["img"].unique()

    # ---- baselines
    say("\n== Baselines on usable images (hit rate = pick is acceptable) ==")
    T["_p"] = 0.0
    for c in ["BOX_ONLY", "HYBRID_D6", "HYBRID_D2", "CONTRAST_T38", "SMART_INK"]:
        say(f"  always {c:13s}: {T[T.cand == c].good.mean():.3f}")
    say(f"  random pick         : {T.groupby('img').good.mean().mean():.3f}")
    if "score" in T:
        T["_p"] = -T["score"].values
        say(f"  lowest 'score' column (proxy for current selector): {pick(T, '_p').good.mean():.3f}")
    if a.winner_report and a.winner_report.is_file():
        W = pd.read_csv(a.winner_report, encoding="utf-8-sig")
        W["img"] = stem(W["image"]); W["cand"] = W["selected_distance_px"].astype(str)
        M = W[["img", "cand"]].merge(T[["img", "cand", "good"]], on=["img", "cand"], how="inner")
        say(f"  ACTUAL current selector (winner_report): {M.good.mean():.3f}  (n={len(M)})")

    # ---- grouped CV, repeated
    say(f"\n== Grouped cross-validation ({a.repeats}x 5-fold, images never seen in training) ==")
    results = {"hgb": [], "lr": [], "blend": []}
    oof_sum = np.zeros(len(T))
    for seed in range(a.repeats):
        rng = np.random.RandomState(seed)
        fold_of = {i: rng.randint(0, 5) for i in imgs}
        fold = T["img"].map(fold_of).values
        ph, pl = np.zeros(len(T)), np.zeros(len(T))
        for f in range(5):
            tr, te = fold != f, fold == f
            m1 = make_hgb(seed).fit(X[tr], y[tr]); m2 = make_lr().fit(X[tr], y[tr])
            ph[te] = m1.predict_proba(X[te])[:, 1]; pl[te] = m2.predict_proba(X[te])[:, 1]
        for name, p in (("hgb", ph), ("lr", pl), ("blend", (ph + pl) / 2)):
            T["_p"] = p
            results[name].append(pick(T, "_p").good.mean())
        oof_sum += (ph + pl) / 2
    for k, v in results.items():
        say(f"  {k:6s}: {np.mean(v):.3f} +/- {np.std(v):.3f}")
    say("  (with ~%d images, differences under ~2 points are within noise)" % len(imgs))

    # ---- confidence vs accuracy (out-of-fold)
    T["_p"] = oof_sum / a.repeats
    top = pick(T, "_p").copy()
    sec = T.sort_values("_p", ascending=False).groupby("img")["_p"].nth(1)
    top["margin"] = top["_p"].values - sec.reindex(top["img"]).values
    say("\n== Out-of-fold confidence vs. hit rate (use to set the review threshold) ==")
    top["bucket"] = pd.qcut(top["_p"], 4, duplicates="drop")
    for b, d in top.groupby("bucket", observed=True):
        say(f"  top prob {b}: hit {d.good.mean():.3f}  (n={len(d)})")
    thr = float(top["_p"].quantile(0.25))
    keep = top[top["_p"] >= thr]
    say(f"  auto-accept above top-prob {thr:.2f}: hit {keep.good.mean():.3f} on {len(keep)}/{len(top)} images;"
        f" the rest ({len(top) - len(keep)}) go to review")

    # ---- does the ranker's confidence flag the CURRENT selector's mistakes?
    if a.winner_report and a.winner_report.is_file():
        sel = M.rename(columns={"good": "sel_good"})[["img", "sel_good"]]
        chk = top[["img", "_p"]].merge(sel, on="img", how="inner")
        chk["flagged"] = chk["_p"] < thr
        miss = chk[chk.sel_good == 0]
        say("\n== Using ranker confidence as a review flag on the CURRENT selector ==")
        say(f"  selector misses: {len(miss)} of {len(chk)} images")
        for flag, d in chk.groupby("flagged"):
            say(f"  {'flagged (low confidence)' if flag else 'not flagged':24s}: "
                f"selector hit {d.sel_good.mean():.3f}  (n={len(d)})")
        caught = int(miss.flagged.sum())
        say(f"  misses caught by the flag: {caught}/{len(miss)}"
            f"  | reviewing {int(chk.flagged.sum())} of {len(chk)} images"
            f" ({chk.flagged.mean():.0%})")
        for q in (0.15, 0.35, 0.5):
            t2 = float(top["_p"].quantile(q)); f2 = chk["_p"] < t2
            say(f"  flag lowest {int(q*100):2d}% by confidence: catches "
                f"{int(miss['_p'].lt(t2).sum())}/{len(miss)} misses, reviews {int(f2.sum())} images")

    # ---- final model on all usable labels, predict every image
    m1 = make_hgb(0).fit(X, y); m2 = make_lr().fit(X, y)
    D["_p"] = (m1.predict_proba(X_all)[:, 1] + m2.predict_proba(X_all)[:, 1]) / 2
    rows = []
    for img, d in D.groupby("img"):
        d = d.sort_values("_p", ascending=False)
        best = d.iloc[0]
        rows.append({
            "image": img + ".png", "ranker_pick": best["cand"], "top_prob": round(best["_p"], 4),
            "runner_up": d.iloc[1]["cand"], "runner_up_prob": round(d.iloc[1]["_p"], 4),
            "needs_review": int(best["_p"] < thr),
            "already_labeled": int(img in labeled_imgs),
        })
    P = pd.DataFrame(rows)
    if a.winner_report and a.winner_report.is_file():
        W2 = W[["image", "selected_distance_px"]].rename(columns={"selected_distance_px": "current_pick"})
        P = P.merge(W2, on="image", how="left")
        P["differs_from_current"] = (P["ranker_pick"].astype(str) != P["current_pick"].astype(str)).astype(int)
    P.to_csv(a.out / "ranker_picks.csv", index=False, encoding="utf-8-sig")
    with (a.out / "ranker.pkl").open("wb") as f:
        pickle.dump({"features": feats, "hgb": m1, "lr": m2, "review_threshold": thr}, f)
    say(f"\nWrote {a.out/'ranker_picks.csv'} ({len(P)} images; {int(P.needs_review.sum())} flagged for review)")
    (a.out / "cv_report.txt").write_text("\n".join(rep), encoding="utf-8")


if __name__ == "__main__":
    main()
