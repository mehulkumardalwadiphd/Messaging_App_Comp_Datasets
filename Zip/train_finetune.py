"""Fine-tuned CNNs (ResNet-50, EfficientNet-B0), trained on clean photographs
('standard') or with compression-aware augmentation ('compaware'), evaluated on
every test condition. One job = (model, regime, seed, fold); each job writes its
own prediction file, so an interrupted run resumes where it stopped and a
finished job is never recomputed (files are keyed to a hash of the code).

  python train_finetune.py                 # all jobs
  python train_finetune.py --smoke         # 1 job, 2 epochs, quick check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, CLASSES, COND_NAMES, N_AUG, N_FOLDS, RESULTS, code_hash, get_splits, load_metadata

MODELS = ["resnet50", "efficientnet_b0"]
REGIMES = ["standard", "compaware"]
SEEDS = [0, 1]
HP = dict(epochs=30, patience=7, batch=32, lr=1e-4, wd=1e-4, aug_prob=0.5, label_smoothing=0.0)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PRED = RESULTS / "preds"
PRED.mkdir(exist_ok=True)
WEIGHTS = CACHE / "weights"
WEIGHTS.mkdir(exist_ok=True)
LOG = RESULTS / "run_log.txt"


def log(msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build(name: str) -> nn.Module:
    if name == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    elif name == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    else:
        raise ValueError(name)
    return m


def to_float(x_uint8: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) uint8 on device -> normalised (B, 3, H, W) float."""
    x = x_uint8.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def geom_aug(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Per-sample flips and 90-degree rotations (lesion orientation is arbitrary)
    plus mild brightness/contrast jitter. Same for both regimes."""
    B = x.shape[0]
    r = torch.rand(B, 4, generator=g)
    out = []
    for i in range(B):
        xi = x[i]
        if r[i, 0] < 0.5: xi = xi.flip(2)
        if r[i, 1] < 0.5: xi = xi.flip(1)
        xi = torch.rot90(xi, int(r[i, 2] * 4) % 4, dims=(1, 2))
        out.append(xi)
    x = torch.stack(out)
    b = (torch.rand(B, 1, 1, 1, generator=g) - 0.5) * 0.2   # +-0.1 in normalised units
    c = 1 + (torch.rand(B, 1, 1, 1, generator=g) - 0.5) * 0.2
    return x * c.to(x.device) + b.to(x.device)


@torch.no_grad()
def predict(model, arr: np.ndarray, bs: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(arr), bs):
        xb = to_float(torch.from_numpy(np.ascontiguousarray(arr[i:i + bs])).to(DEV))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            out.append(torch.softmax(model(xb).float(), 1).cpu().numpy())
    return np.concatenate(out)


def run_job(model_name, regime, seed, fold, y, s, clean, aug, hp, tag, smoke=False):
    out_p = PRED / f"ft_{model_name}_{regime}_s{seed}_f{fold}.npz"
    if out_p.exists() and not smoke:
        z = np.load(out_p)  # only numeric + unicode arrays; no pickle needed
        if str(z["code_hash"]) == tag:
            return "cached"
    set_seed(1000 * seed + fold)
    te = np.where(s["fold"].values == fold)[0]
    va = np.where(s[f"val_f{fold}"].values)[0]
    tr = np.where((s["fold"].values != fold) & ~s[f"val_f{fold}"].values)[0]
    Xtr = torch.from_numpy(clean[tr])                       # (n, 224, 224, 3) uint8, CPU
    Atr = torch.from_numpy(aug[tr]) if regime == "compaware" else None
    ytr = torch.from_numpy(y[tr]).long()
    cnt = np.bincount(y[tr], minlength=len(CLASSES)).astype(float)
    w = torch.tensor(cnt.sum() / (len(CLASSES) * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    model = build(model_name).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    epochs = 2 if smoke else hp["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=hp["label_smoothing"])
    g = torch.Generator().manual_seed(seed * 7919 + fold)
    best, best_ep, best_state, hist, bad = -1.0, -1, None, [], 0
    Xva = clean[va]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=g)
        tl, t0 = 0.0, time.time()
        for i in range(0, len(tr), hp["batch"]):
            idx = perm[i:i + hp["batch"]]
            xb = Xtr[idx].clone()
            if Atr is not None:
                swap = torch.rand(len(idx), generator=g) < hp["aug_prob"]
                k = torch.randint(0, N_AUG, (len(idx),), generator=g)
                for j in torch.where(swap)[0].tolist():
                    xb[j] = Atr[idx[j], k[j]]
            xb = geom_aug(to_float(xb.to(DEV)), g)
            yb = ytr[idx].to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(idx)
        sched.step()
        pva = predict(model, Xva)
        bacc = balanced_accuracy_score(y[va], pva.argmax(1))
        hist.append(dict(epoch=ep, train_loss=tl / len(tr), val_bacc=bacc, sec=time.time() - t0))
        if bacc > best:
            best, best_ep, bad = bacc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)   # restore best-validation weights
    probs = {}
    for c in COND_NAMES:
        arr = np.load(CACHE / f"inputs_{c}.npy", mmap_mode="r")
        probs[c] = predict(model, arr[te])
    if smoke:
        return dict(hist=hist, best=best, probs_clean_bacc=balanced_accuracy_score(y[te], probs["clean"].argmax(1)))
    torch.save(best_state, WEIGHTS / f"{model_name}_{regime}_s{seed}_f{fold}.pt")
    np.savez_compressed(out_p, idx=te, code_hash=tag, best_epoch=best_ep, best_val_bacc=best,
                        history=json.dumps(hist), **{f"p_{c}": probs[c] for c in COND_NAMES})
    return f"best_ep={best_ep} val_bacc={best:.3f} epochs_run={len(hist)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()
    m, s = load_metadata(), get_splits()
    y = m["y"].values
    clean = np.load(CACHE / "inputs_clean.npy", mmap_mode="r")
    aug = np.load(CACHE / "inputs_aug.npy", mmap_mode="r")
    tag = code_hash("train_finetune.py")
    (RESULTS / "hyperparameters_finetune.json").write_text(json.dumps(dict(HP, models=MODELS, regimes=REGIMES, seeds=SEEDS,
                                                                            folds=N_FOLDS, code_hash=tag), indent=2))
    if a.smoke:
        t0 = time.time()
        r = run_job(a.models[0], "compaware", 0, 0, y, s, clean, aug, HP, tag, smoke=True)
        print(json.dumps(r, indent=1, default=float))
        print(f"smoke took {time.time() - t0:.0f}s")
        return
    jobs = [(mn, rg, sd, f) for mn in a.models for rg in REGIMES for sd in SEEDS for f in range(N_FOLDS)]
    t0, done_new = time.time(), 0
    for n, (mn, rg, sd, f) in enumerate(jobs, 1):
        tj = time.time()
        r = run_job(mn, rg, sd, f, y, s, clean, aug, HP, tag)
        if r != "cached":
            done_new += 1
        el = time.time() - t0
        eta = el / max(done_new, 1) * (len(jobs) - n) if done_new else float("nan")
        log(f"[ft {n}/{len(jobs)}] {mn} {rg} s{sd} f{f}: {r} ({time.time() - tj:.0f}s)  ETA {eta / 60:.0f} min")
    log("[ft] ALL DONE")


if __name__ == "__main__":
    main()
