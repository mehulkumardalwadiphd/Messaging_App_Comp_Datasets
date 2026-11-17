"""All figures, drawn only from results/*.csv and the cached inputs.

Fig1 workflow block diagram · Fig2 example photograph under compression ·
Fig3 change in balanced accuracy over JPEG quality (95% CI) · Fig4 prediction
flip rate over JPEG quality against the seed-instability floor · Fig5 gain from
compression-aware training. 300 dpi PNG, Arial, colour-blind-checked palette
with a distinct marker and dash per model so print and greyscale still read.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONDITIONS, FIG, RESULTS, apply_condition, load_metadata, load_rgb, to_input

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.labelcolor": "#222222", "xtick.color": "#444444",
                     "ytick.color": "#444444", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
MODELS = [("resnet50", "ResNet-50 (fine-tuned)", "#2a78d6", "o", "-"),
          ("efficientnet_b0", "EfficientNet-B0 (fine-tuned)", "#eb6834", "s", "--"),
          ("dinov2_vits14", "DINOv2 ViT-S/14 (frozen)", "#1baf7a", "^", "-."),
          ("clip_vitb16", "CLIP ViT-B/16 (frozen)", "#eda100", "D", ":")]
QS = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
XP = list(range(len(QS)))   # evenly spaced categories: 95 and 90 must not collide


def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.axis("off")
    w, gap = 0.172, 0.027
    boxes = ["PAD-UFES-20\n2,298 smartphone\nphotographs,\n1,373 patients,\n6 diagnoses",
             "Patient-grouped\nstratified 5-fold\ncross-validation\n+ inner validation\nsplit",
             "Training on original\nphotographs, or with\ncompression-aware\naugmentation",
             "Test photograph\ndegraded at native\nresolution: JPEG,\napp profile or\nrepeated re-send",
             "Resize to 224 x 224,\nclassify, compare\nwith the prediction\non the original"]
    for i, t in enumerate(boxes):
        x = 0.01 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.36), w, 0.58, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc="#eef4fc", ec="#2a78d6", lw=1.0))
        ax.text(x + w / 2, 0.65, t, ha="center", va="center", fontsize=7.5, linespacing=1.2)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.004, 0.65), xytext=(x + w + 0.004, 0.65),
                        arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(0.5, 0.21, "Classifiers: ResNet-50 and EfficientNet-B0 (fine-tuned); DINOv2 ViT-S/14 and CLIP ViT-B/16 "
            "(frozen encoder + logistic regression)", ha="center", fontsize=8.0)
    ax.text(0.5, 0.07, "External check: 120 test photographs sent through WhatsApp and saved back, scored by the "
            "same fold models", ha="center", fontsize=8.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIG / "fig1_workflow.png")
    plt.close(fig)


def fig_examples():
    m = load_metadata()
    big = m.assign(le=[max(Image.open(p).size) for p in m.path]).query("le > 1600 and diagnostic == 'MEL'")
    p = big.path.iloc[0]
    im = load_rgb(p)
    w, h = im.size
    side = int(min(w, h) * 0.30)                       # centre crop: lesions are centred in PAD-UFES-20
    box = ((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2)
    conds = ["clean", "q70", "q30", "q10"]
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.9))
    for j, c in enumerate(conds):
        sp = CONDITIONS[c]
        out, nb = apply_condition(im, sp["cap"], sp["q"], sp["gens"])
        views = [to_input(out), np.asarray(out.crop(box).resize((300, 300), Image.NEAREST))]
        for i, v in enumerate(views):
            a = axs[i, j]
            a.imshow(v); a.set_xticks([]); a.set_yticks([]); a.grid(False)
            for s in a.spines.values():
                s.set_visible(True); s.set_color("#999999")
        axs[0, j].set_title("original (PNG)" if c == "clean" else f"JPEG quality {sp['q']}", fontsize=8)
    axs[0, 0].set_ylabel("model input\n(224 x 224)", fontsize=8)
    axs[1, 0].set_ylabel("centre crop at\nnative resolution", fontsize=8)
    fig.savefig(FIG / "fig2_examples.png")
    plt.close(fig)
    return os.path.basename(p)


def _qcurve(df, model, regime, metric):
    d = df[(df.model == model) & (df.regime == regime) & (df.metric == metric)].set_index("cond")
    return np.array([d.loc[f"q{q}", "delta"] for q in QS]), np.array([d.loc[f"q{q}", "lo"] for q in QS]), \
        np.array([d.loc[f"q{q}", "hi"] for q in QS])


def fig_dose(bd):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for a, reg, ttl in zip(axs, ["standard", "compaware"], ["(a) trained on original photos",
                                                           "(b) compression-aware training"]):
        a.axhline(0, color="#888888", lw=0.8)
        for mk, name, col, mar, ls in MODELS:
            if not ((bd.model == mk) & (bd.regime == reg)).any():
                continue
            y, lo, hi = _qcurve(bd, mk, reg, "bacc")
            a.fill_between(XP, lo, hi, color=col, alpha=0.12, lw=0)
            a.plot(XP, y, color=col, marker=mar, ms=4, lw=1.6, ls=ls, label=name)
        a.set_xlim(-0.4, len(QS) - 0.6); a.set_xticks(XP, [str(q) for q in QS])
        a.set_xlabel("JPEG quality (no downscale)")
        a.set_title(ttl, fontsize=8.5, loc="left")
    axs[0].set_ylabel("Change in balanced accuracy\nvs original photo")
    axs[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.savefig(FIG / "fig3_dose_response.png")
    plt.close(fig)


def fig_flip(bd, sf):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for mk, name, col, mar, ls in MODELS:
        if not ((bd.model == mk) & (bd.regime == "standard")).any():
            continue
        y, lo, hi = _qcurve(bd, mk, "standard", "flip")
        ax.plot(XP, y, color=col, marker=mar, ms=4, lw=1.6, ls=ls, label=name)
        f = sf[(sf.model == mk) & (sf.regime == "standard")]
        if len(f) and f.seed_flip.iloc[0] > 0:
            ax.axhline(f.seed_flip.iloc[0], color=col, lw=0.9, ls=(0, (1, 2)))
    ax.set_xlim(-0.4, len(QS) - 0.6); ax.set_xticks(XP, [str(q) for q in QS])
    ax.set_xlabel("JPEG quality (no downscale)")
    ax.set_ylabel("Share of photos whose\npredicted diagnosis changes")
    ax.text(-0.3, ax.get_ylim()[1] * 0.97, "horizontal dotted lines: share that changes between\ntwo training seeds of the same fine-tuned network",
            fontsize=6.8, va="top", color="#444444")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.savefig(FIG / "fig4_flip_rate.png")
    plt.close(fig)


def fig_gain(rg):
    conds = ["q90", "q70", "q50", "q30", "q20", "q10", "app_1600_q80", "app_960_q50", "resend_1280_q60_x5"]
    labels = ["q90", "q70", "q50", "q30", "q20", "q10", "1600/q80", "960/q50", "1280/q60 x5"]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    present = [mm for mm in MODELS if (rg.model == mm[0]).any()]
    wbar = 0.8 / max(len(present), 1)
    for i, (mk, name, col, mar, ls) in enumerate(present):
        d = rg[rg.model == mk].set_index("cond").loc[conds]
        x = np.arange(len(conds)) + (i - (len(present) - 1) / 2) * wbar
        ax.errorbar(x, d.gain_bacc, yerr=[d.gain_bacc - d.lo, d.hi - d.gain_bacc], fmt=mar, color=col, ms=4,
                    capsize=2, lw=1.0, label=name)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.axvline(5.5, color="#cccccc", lw=0.8)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Balanced accuracy gain\n(compression-aware - standard)")
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper left")
    fig.savefig(FIG / "fig5_regime_gain.png")
    plt.close(fig)


def main():
    bd = pd.read_csv(RESULTS / "bootstrap_delta.csv")
    sf = pd.read_csv(RESULTS / "seed_floor.csv")
    rg = pd.read_csv(RESULTS / "regime_gain.csv")
    fig_workflow()
    ex = fig_examples()
    fig_dose(bd)
    fig_flip(bd, sf)
    fig_gain(rg)
    print("figures written; example photo:", ex)


if __name__ == "__main__":
    main()
