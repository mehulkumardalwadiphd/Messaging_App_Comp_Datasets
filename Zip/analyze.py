"""Turn the prediction files into every table the paper quotes.

Outputs (results/)
  fold_metrics.csv        one row per model x regime x seed x fold x condition
  summary.csv             mean and SD over the 10 fold-seed runs, with the
                          change from clean (delta) and its SD
  bootstrap_delta.csv     patient-cluster bootstrap 95% CI of the change from
                          clean (pooled out-of-fold predictions, both seeds)
  regime_gain.csv         compression-aware minus standard, same bootstrap
  seed_floor.csv          prediction flips between seed 0 and seed 1 on clean
                          photographs: the instability floor a flip rate must beat
  bytes_summary.csv       median file size per condition and ratio to the PNG
  seed_identity.csv       frozen heads: are seed 0 and seed 1 predictions identical?
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, COND_NAMES, MALIGNANT, RESULTS, load_metadata

MAL = np.array([c in MALIGNANT for c in CLASSES])
MEL = CLASSES.index("MEL")
B = 2000
RNG = np.random.default_rng(2026)


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (conf > lo) & (conf <= hi)
        if k.any():
            e += k.mean() * abs((pred[k] == y[k]).mean() - conf[k].mean())
    return e


def metrics(p, y, p_clean):
    pred = p.argmax(1)
    mal_true = MAL[y]
    out = dict(bacc=balanced_accuracy_score(y, pred),
               macro_f1=f1_score(y, pred, average="macro"),
               auc=roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(len(CLASSES)))),
               mal_sens=(MAL[pred][mal_true]).mean(),
               mel_recall=(pred[y == MEL] == MEL).mean() if (y == MEL).any() else np.nan,
               flip=(pred != p_clean.argmax(1)).mean(),
               ece=ece(p, y), n=len(y))
    return out


def load_runs():
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "preds" / "*.npz"))):
        base = os.path.basename(f)[:-4]
        fam, rest = base.split("_", 1)
        model, regime, seed, fold = rest.rsplit("_", 3)
        rows.append(dict(file=f, family=fam, model=model, regime=regime, seed=int(seed[1:]), fold=int(fold[1:])))
    return pd.DataFrame(rows)


def main():
    m = load_metadata()
    y_all, pid = m["y"].values, m["patient_id"].values
    runs = load_runs()
    assert len(runs), "no prediction files"
    fm, pooled = [], {}
    for r in runs.itertuples():
        z = np.load(r.file)
        idx = z["idx"]
        y = y_all[idx]
        key = (r.model, r.regime, r.seed)
        pooled.setdefault(key, {c: np.full((len(m), len(CLASSES)), np.nan) for c in COND_NAMES})
        for c in COND_NAMES:
            pooled[key][c][idx] = z[f"p_{c}"]
            fm.append(dict(family=r.family, model=r.model, regime=r.regime, seed=r.seed, fold=r.fold, cond=c,
                           **metrics(z[f"p_{c}"], y, z["p_clean"])))
    fm = pd.DataFrame(fm)
    fm.to_csv(RESULTS / "fold_metrics.csv", index=False)

    # mean / SD over fold-seed runs, and the change from clean within each run
    cl = fm[fm.cond == "clean"].set_index(["model", "regime", "seed", "fold"])
    d = fm.join(cl[["bacc", "macro_f1", "auc", "mal_sens", "mel_recall", "ece"]], on=["model", "regime", "seed", "fold"],
                rsuffix="_clean")
    for k in ["bacc", "macro_f1", "auc", "mal_sens", "mel_recall", "ece"]:
        d[f"d_{k}"] = d[k] - d[f"{k}_clean"]
    agg = {}
    for k in ["bacc", "macro_f1", "auc", "mal_sens", "mel_recall", "flip", "ece",
              "d_bacc", "d_macro_f1", "d_auc", "d_mal_sens", "d_mel_recall", "d_ece"]:
        agg[f"{k}_mean"] = (k, "mean")
        agg[f"{k}_sd"] = (k, "std")
    summ = d.groupby(["model", "regime", "cond"]).agg(n_runs=("bacc", "size"), **agg).reset_index()
    summ["cond"] = pd.Categorical(summ["cond"], COND_NAMES, ordered=True)
    summ.sort_values(["model", "regime", "cond"]).to_csv(RESULTS / "summary.csv", index=False, float_format="%.4f")

    # Patient-cluster bootstrap on pooled out-of-fold predictions (both seeds).
    # Each resample draws patients with replacement; W[b, i] is how many times
    # image i enters resample b, so every statistic is a weighted mean and all
    # B resamples are computed at once.
    up, inv = np.unique(pid, return_inverse=True)
    draws = RNG.integers(0, len(up), size=(B, len(up)))
    Wp = np.stack([np.bincount(d, minlength=len(up)) for d in draws])        # (B, patients)
    W = Wp[:, inv].astype(float)                                             # (B, images)
    W_full = np.ones((1, len(m)))
    onehot = np.eye(len(CLASSES))[y_all]                                     # (N, C)
    seeds = sorted({k[2] for k in pooled})
    complete = {k for k, v in pooled.items() if not np.isnan(v["clean"]).any()}
    models = sorted({(k[0], k[1]) for k in complete if all((k[0], k[1], s) in complete for s in seeds)})
    skipped = sorted({(k[0], k[1]) for k in pooled} - set(models))
    if skipped:
        print("bootstrap skips incomplete model/regime:", skipped)

    def bacc_w(p, Wm):
        corr = (p.argmax(1) == y_all).astype(float)[:, None] * onehot        # (N, C)
        return ((Wm @ corr) / (Wm @ onehot)).mean(1)                         # (B,)

    def mal_w(p, Wm):
        hit = (MAL[p.argmax(1)] & MAL[y_all]).astype(float)
        return (Wm @ hit) / (Wm @ MAL[y_all].astype(float))

    def flip_w(a, b, Wm):
        return (Wm @ (a.argmax(1) != b.argmax(1)).astype(float)) / Wm.sum(1)

    def seed_mean(fn, P, Wm):
        return np.mean([fn(p, Wm) for p in P], axis=0)

    def ci(x):
        return np.percentile(x, 2.5), np.percentile(x, 97.5)

    bd, rg = [], []
    for (mod, reg) in models:
        P = {c: [pooled[(mod, reg, s)][c] for s in seeds] for c in COND_NAMES}
        for c in COND_NAMES:
            if c == "clean":
                continue
            for name, fn in [("bacc", bacc_w), ("mal_sens", mal_w)]:
                pt = (seed_mean(fn, P[c], W_full) - seed_mean(fn, P["clean"], W_full))[0]
                lo, hi = ci(seed_mean(fn, P[c], W) - seed_mean(fn, P["clean"], W))
                bd.append(dict(model=mod, regime=reg, cond=c, metric=name, delta=pt, lo=lo, hi=hi))
            fl = lambda Wm: np.mean([flip_w(P[c][s], P["clean"][s], Wm) for s in range(len(seeds))], axis=0)
            lo, hi = ci(fl(W))
            bd.append(dict(model=mod, regime=reg, cond=c, metric="flip", delta=fl(W_full)[0], lo=lo, hi=hi))
    pd.DataFrame(bd).to_csv(RESULTS / "bootstrap_delta.csv", index=False, float_format="%.4f")

    for mod in sorted({k[0] for k in pooled}):
        if (mod, "standard") not in models or (mod, "compaware") not in models:
            continue
        for c in COND_NAMES:
            A = [pooled[(mod, "compaware", s)][c] for s in seeds]
            S = [pooled[(mod, "standard", s)][c] for s in seeds]
            pt = (seed_mean(bacc_w, A, W_full) - seed_mean(bacc_w, S, W_full))[0]
            lo, hi = ci(seed_mean(bacc_w, A, W) - seed_mean(bacc_w, S, W))
            rg.append(dict(model=mod, cond=c, gain_bacc=pt, lo=lo, hi=hi))
    pd.DataFrame(rg).to_csv(RESULTS / "regime_gain.csv", index=False, float_format="%.4f")

    # Sensitivity: drop every photograph that appears twice under two lesion IDs
    dp = RESULTS / "duplicate_pairs.csv"
    if dp.exists():
        dup = pd.read_csv(dp)
        keep = ~m["img_id"].isin(set(dup.img_a) | set(dup.img_b)).values
        sens = []
        for (mod, reg) in models:
            P = {c: [pooled[(mod, reg, s)][c][keep] for s in seeds] for c in COND_NAMES}
            yk = y_all[keep]
            for c in COND_NAMES:
                b_all = np.mean([balanced_accuracy_score(y_all, pooled[(mod, reg, s)][c].argmax(1)) for s in seeds])
                b_k = np.mean([balanced_accuracy_score(yk, p.argmax(1)) for p in P[c]])
                sens.append(dict(model=mod, regime=reg, cond=c, bacc_all=b_all, bacc_no_dups=b_k,
                                 n_all=len(m), n_no_dups=int(keep.sum())))
        pd.DataFrame(sens).to_csv(RESULTS / "sensitivity_no_duplicates.csv", index=False, float_format="%.4f")

    sf, si = [], []
    for (mod, reg) in models:
        if len(seeds) < 2:
            continue
        a, b = pooled[(mod, reg, seeds[0])]["clean"], pooled[(mod, reg, seeds[1])]["clean"]
        fl = a.argmax(1) != b.argmax(1)
        fb = flip_w(a, b, W)
        sf.append(dict(model=mod, regime=reg, seed_flip=fl.mean(), lo=np.percentile(fb, 2.5), hi=np.percentile(fb, 97.5)))
        si.append(dict(model=mod, regime=reg, max_abs_prob_diff=float(np.nanmax(np.abs(a - b)))))
    pd.DataFrame(sf).to_csv(RESULTS / "seed_floor.csv", index=False, float_format="%.4f")
    pd.DataFrame(si).to_csv(RESULTS / "seed_identity.csv", index=False, float_format="%.6g")

    cb = pd.read_csv(RESULTS / "condition_bytes.csv")
    cb["ratio"] = cb["bytes"] / cb["orig_png_bytes"]
    bsum = cb.groupby("cond").agg(median_bytes=("bytes", "median"), median_ratio=("ratio", "median"),
                                  median_long_edge=("long_edge", "median"),
                                  share_downscaled=("long_edge", lambda s: np.nan)).reset_index()
    down = cb.assign(ds=cb.long_edge < cb.orig_long_edge).groupby("cond").ds.mean()
    bsum["share_downscaled"] = bsum["cond"].map(down)
    bsum["cond"] = pd.Categorical(bsum["cond"], COND_NAMES, ordered=True)
    bsum.sort_values("cond").to_csv(RESULTS / "bytes_summary.csv", index=False, float_format="%.4f")
    print("analysis written:", len(fm), "fold-condition rows")


if __name__ == "__main__":
    main()
