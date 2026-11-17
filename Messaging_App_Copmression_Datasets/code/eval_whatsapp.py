"""Evaluate every trained model on the 120 photographs that made a real
WhatsApp round trip, next to (a) the same photographs uncompressed and (b) a
simulation of the profile WhatsApp was measured to apply (long edge capped at
1600 px, JPEG quality 92, 4:2:0). Each image is scored only by the models of
the fold in which it was a test image, so no model sees its own training data.

Writes results/whatsapp_eval.csv (one row per model x regime x seed) and
results/whatsapp_fidelity.csv (pixel difference real vs simulated, per image).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, DATA, N_AUG, N_FOLDS, RESULTS, apply_condition, get_splits, load_metadata, load_rgb, to_input
import frozen
import train_finetune as ft

PROFILE = dict(cap=1600, q=92, gens=1)   # measured: results/whatsapp_roundtrip.csv


def main():
    m, s = load_metadata(), get_splits()
    y = m["y"].values
    rt = pd.read_csv(RESULTS / "whatsapp_roundtrip.csv")
    idx = np.array([int(np.where(m.img_id == k)[0][0]) for k in rt.img_id])
    fold = s.fold.values[idx]
    clean = np.load(CACHE / "inputs_clean.npy", mmap_mode="r")[idx]
    real = np.stack([to_input(Image.open(DATA / "whatsapp" / "matched" / (k[:-4] + ".jpeg")).convert("RGB")) for k in rt.img_id])
    sim = np.stack([to_input(apply_condition(load_rgb(p), **PROFILE)[0]) for p in m.path.values[idx]])
    fid = pd.DataFrame(dict(img_id=rt.img_id, mad_real_vs_clean=np.abs(real.astype(float) - clean).mean((1, 2, 3)),
                            mad_sim_vs_clean=np.abs(sim.astype(float) - clean).mean((1, 2, 3)),
                            mad_real_vs_sim=np.abs(real.astype(float) - sim).mean((1, 2, 3))))
    fid.to_csv(RESULTS / "whatsapp_fidelity.csv", index=False, float_format="%.4f")
    V = dict(clean=clean, real=real, sim=sim)
    rows = []

    def summarise(fam, model, regime, seed, P):
        pc, pr, ps = (P[k].argmax(1) for k in ("clean", "real", "sim"))
        yy = y[idx]
        rows.append(dict(family=fam, model=model, regime=regime, seed=seed, n=len(idx),
                         acc_clean=(pc == yy).mean(), acc_real=(pr == yy).mean(), acc_sim=(ps == yy).mean(),
                         flip_real=(pr != pc).mean(), flip_sim=(ps != pc).mean(), agree_real_sim=(pr == ps).mean(),
                         mean_abs_dprob_real=np.abs(P["real"] - P["clean"]).sum(1).mean() / 2,
                         mean_abs_dprob_sim=np.abs(P["sim"] - P["clean"]).sum(1).mean() / 2))

    for model in ft.MODELS:
        for regime in ft.REGIMES:
            for seed in ft.SEEDS:
                P = {k: np.zeros((len(idx), 6)) for k in V}
                for f in range(N_FOLDS):
                    sel = np.where(fold == f)[0]
                    if not len(sel):
                        continue
                    net = ft.build(model).to(ft.DEV)
                    net.load_state_dict(torch.load(ft.WEIGHTS / f"{model}_{regime}_s{seed}_f{f}.pt", map_location=ft.DEV, weights_only=True))
                    for k in V:
                        P[k][sel] = ft.predict(net, V[k][sel])
                summarise("ft", model, regime, seed, P)
    for enc in frozen.ENCODERS:
        fn, mean, std = frozen.load_encoder(enc)
        F = {k: frozen.embed(fn, mean, std, V[k]) for k in V}
        Fall = {c: np.load(frozen.FEAT / f"{enc}_{c}.npy") for c in ["clean"] + [f"aug{k}" for k in range(N_AUG)]}
        for regime in frozen.REGIMES:
            for seed in frozen.SEEDS:
                P = {k: np.zeros((len(idx), 6)) for k in V}
                for f in range(N_FOLDS):
                    sel = np.where(fold == f)[0]
                    z = np.load(frozen.PRED / f"fz_{enc}_{regime}_s{seed}_f{f}.npz")
                    tr = np.where((s.fold.values != f) & ~s[f"val_f{f}"].values)[0]
                    Xtr, ytr = Fall["clean"][tr], y[tr]
                    if regime == "compaware":
                        Xtr = np.concatenate([Xtr] + [Fall[f"aug{k}"][tr] for k in range(N_AUG)])
                        ytr = np.concatenate([ytr] * (N_AUG + 1))
                    sc = StandardScaler().fit(Xtr)
                    clf = LogisticRegression(C=float(z["best_C"]), class_weight="balanced", max_iter=5000,
                                             random_state=seed).fit(sc.transform(Xtr), ytr)
                    # the refit must reproduce the saved predictions exactly
                    ref = z["p_clean"][np.searchsorted(z["idx"], idx[sel])]
                    got = clf.predict_proba(sc.transform(Fall["clean"][idx[sel]]))
                    assert np.allclose(ref, got, atol=1e-6), "refit does not reproduce saved head"
                    for k in V:
                        P[k][sel] = clf.predict_proba(sc.transform(F[k][sel]))
                summarise("fz", enc, regime, seed, P)
    pd.DataFrame(rows).to_csv(RESULTS / "whatsapp_eval.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).groupby(["model", "regime"])[["acc_clean", "acc_real", "flip_real", "flip_sim", "agree_real_sim"]].mean())


if __name__ == "__main__":
    main()
