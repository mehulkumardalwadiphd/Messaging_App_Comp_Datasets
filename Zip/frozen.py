"""Frozen foundation-model encoders (DINOv2 ViT-S/14, OpenCLIP ViT-B/16) with a
logistic-regression head, same splits, regimes and conditions as the CNNs.

Stage 1 extracts embeddings once per (encoder, condition) into .cache/.
Stage 2 fits the head per (regime, seed, fold). Logistic regression with the
lbfgs solver is deterministic, so the two seeds are expected to give identical
predictions; both are run and the analysis checks that they match.

  python frozen.py            # extract (if missing) + fit all
  python frozen.py --smoke    # 64 images, one condition, one fold
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Public weights: download anonymously so an expired local Hugging Face token
# cannot block the run (the user's stored token is not touched).
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, CLASSES, COND_NAMES, N_AUG, N_FOLDS, RESULTS, code_hash, get_splits, load_metadata

ENCODERS = ["dinov2_vits14", "clip_vitb16"]
REGIMES = ["standard", "compaware"]
SEEDS = [0, 1]
C_GRID = [0.01, 0.1, 1.0, 10.0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PRED = RESULTS / "preds"
PRED.mkdir(exist_ok=True)
FEAT = CACHE / "features"
FEAT.mkdir(exist_ok=True)
LOG = RESULTS / "run_log.txt"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_encoder(name):
    if name == "dinov2_vits14":
        import timm
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
        mean, std, fn = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), m
    elif name == "clip_vitb16":
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        fn = m.encode_image
    else:
        raise ValueError(name)
    m.eval().to(DEV)
    return fn, torch.tensor(mean).view(1, 3, 1, 1).to(DEV), torch.tensor(std).view(1, 3, 1, 1).to(DEV)


@torch.no_grad()
def embed(fn, mean, std, arr, bs=128):
    out = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(np.ascontiguousarray(arr[i:i + bs])).to(DEV).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            f = fn(x)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def extract(name, limit=None):
    fn = None
    names = COND_NAMES + [f"aug{k}" for k in range(N_AUG)]
    for c in names:
        p = FEAT / f"{name}_{c}.npy"
        if p.exists() and limit is None:
            continue
        if fn is None:
            fn, mean, std = load_encoder(name)
        if c.startswith("aug"):
            arr = np.load(CACHE / "inputs_aug.npy", mmap_mode="r")[:, int(c[3:])]
        else:
            arr = np.load(CACHE / f"inputs_{c}.npy", mmap_mode="r")
        if limit:
            arr = arr[:limit]
        t0 = time.time()
        f = embed(fn, mean, std, arr)
        if limit:
            return f
        np.save(p, f)
        log(f"[feat] {name} {c}: {f.shape} in {time.time() - t0:.0f}s")


def fit_job(name, regime, seed, fold, y, s, tag):
    out_p = PRED / f"fz_{name}_{regime}_s{seed}_f{fold}.npz"
    if out_p.exists() and str(np.load(out_p)["code_hash"]) == tag:
        return "cached"
    te = np.where(s["fold"].values == fold)[0]
    va = np.where(s[f"val_f{fold}"].values)[0]
    tr = np.where((s["fold"].values != fold) & ~s[f"val_f{fold}"].values)[0]
    F = {c: np.load(FEAT / f"{name}_{c}.npy") for c in COND_NAMES}
    Xtr, ytr = F["clean"][tr], y[tr]
    if regime == "compaware":
        Xtr = np.concatenate([Xtr] + [np.load(FEAT / f"{name}_aug{k}.npy")[tr] for k in range(N_AUG)])
        ytr = np.concatenate([ytr] * (N_AUG + 1))
    sc = StandardScaler().fit(Xtr)
    best = (-1, None, None)
    for C in C_GRID:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
        clf.fit(sc.transform(Xtr), ytr)
        b = balanced_accuracy_score(y[va], clf.predict(sc.transform(F["clean"][va])))
        if b > best[0]:
            best = (b, C, clf)
    probs = {c: best[2].predict_proba(sc.transform(F[c][te])) for c in COND_NAMES}
    np.savez_compressed(out_p, idx=te, code_hash=tag, best_C=best[1], best_val_bacc=best[0],
                        **{f"p_{c}": probs[c] for c in COND_NAMES})
    return f"C={best[1]} val_bacc={best[0]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    m, s = load_metadata(), get_splits()
    y = m["y"].values
    if a.smoke:
        for e in ENCODERS:
            t0 = time.time()
            f = extract(e, limit=64)
            print(e, f.shape, f"{time.time() - t0:.0f}s", "finite:", np.isfinite(f).all())
        return
    tag = code_hash("frozen.py")
    (RESULTS / "hyperparameters_frozen.json").write_text(json.dumps(dict(
        encoders={"dinov2_vits14": "timm vit_small_patch14_dinov2.lvd142m, img_size=224",
                  "clip_vitb16": "open_clip ViT-B-16 laion2b_s34b_b88k"},
        head="StandardScaler + LogisticRegression(lbfgs, class_weight=balanced, max_iter=5000)",
        C_grid=C_GRID, selection="balanced accuracy on the clean inner-validation patients",
        regimes=REGIMES, seeds=SEEDS, code_hash=tag), indent=2))
    for e in ENCODERS:
        extract(e)
    for e in ENCODERS:
        for rg in REGIMES:
            for sd in SEEDS:
                for f in range(N_FOLDS):
                    log(f"[fz] {e} {rg} s{sd} f{f}: {fit_job(e, rg, sd, f, y, s, tag)}")
    log("[fz] ALL DONE")


if __name__ == "__main__":
    main()
