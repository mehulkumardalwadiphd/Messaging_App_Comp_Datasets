"""Write every manuscript table from results/ into the draft.

For each table this script (1) writes a paper-ready CSV, results/table_<name>.csv,
whose first column is the row label printed in the paper and whose values are
rounded exactly as printed, and (2) replaces the block between
<!-- BEGIN table:<name> --> and <!-- END table:<name> --> in the draft section
with a caption, a <!-- results: table_<name>.csv --> binding and the Markdown
table. The numbers gate then re-reads every cell against that CSV.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, CONDITIONS, PAPER, RESULTS

DRAFT = PAPER / "draft"
NAMES = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0", "dinov2_vits14": "DINOv2 ViT-S/14",
         "clip_vitb16": "CLIP ViT-B/16"}
ORDER = list(NAMES)
REG = {"standard": "standard", "compaware": "compression-aware"}
COND_LABEL = {"clean": "Original PNG", **{f"q{q}": f"JPEG q{q}" for q in (95, 90, 80, 70, 60, 50, 40, 30, 20, 10)},
              "app_1600_q80": "App 1600 px / q80", "app_1600_q70": "App 1600 px / q70",
              "app_1280_q60": "App 1280 px / q60", "app_960_q50": "App 960 px / q50",
              "resend_1280_q60_x3": "Re-sent x3 (1280 px / q60)", "resend_1280_q60_x5": "Re-sent x5 (1280 px / q60)"}
KEY_CONDS = ["q95", "app_1600_q80", "app_960_q50", "resend_1280_q60_x5", "q30", "q10"]


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def pm(m, s):
    return f"{f3(m)} ± {f3(s)}"


def put(section: str, name: str, caption: str, df: pd.DataFrame, fmt, bind: bool = True):
    """df: first column = label; fmt(col, row) -> printed cell."""
    df.to_csv(RESULTS / f"table_{name}.csv", index=False)
    cols = list(df.columns)
    lines = [caption, ""] + ([f"<!-- results: table_{name}.csv -->"] if bind else []) + ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join([str(r[cols[0]])] + [fmt(c, r) for c in cols[1:]]) + " |")
    p = DRAFT / section
    t = p.read_text(encoding="utf-8")
    pat = re.compile(rf"<!-- BEGIN table:{name} -->.*?<!-- END table:{name} -->", re.S)
    if not pat.search(t):
        sys.exit(f"marker table:{name} missing in {section}")
    t = pat.sub(lambda _: f"<!-- BEGIN table:{name} -->\n" + "\n".join(lines) + f"\n<!-- END table:{name} -->", t)
    p.write_text(t, encoding="utf-8")


def t_folds():
    s = pd.read_csv(RESULTS / "splits.csv")
    rows = []
    for f in range(5):
        d = s[s.fold == f]
        rows.append({"Fold": f"Fold {f + 1}", "Patients": d.patient_id.nunique(), "Photographs": len(d),
                     **{c: int((d.diagnostic == c).sum()) for c in CLASSES}})
    rows.append({"Fold": "Total", "Patients": s.patient_id.nunique(), "Photographs": len(s),
                 **{c: int((s.diagnostic == c).sum()) for c in CLASSES}})
    put("03-methods.md", "folds", "**Table 1** Photographs per fold and diagnosis (patient-grouped, class-stratified "
        "five-fold split)", pd.DataFrame(rows), lambda c, r: str(int(r[c])))


def t_conditions():
    b = pd.read_csv(RESULTS / "bytes_summary.csv").set_index("cond")
    rows = []
    for c, sp in CONDITIONS.items():
        rows.append({"Condition": COND_LABEL[c], "Longest side cap (px)": sp["cap"] if sp["cap"] else "none",
                     "JPEG quality": sp["q"] if sp["q"] else "–", "Passes": sp["gens"],
                     "Median size (% of PNG)": round(100 * b.loc[c, "median_ratio"], 1),
                     "Photographs downscaled (%)": round(100 * b.loc[c, "share_downscaled"], 1)})
    df = pd.DataFrame(rows)
    put("03-methods.md", "conditions", "**Table 2** Compression conditions applied to every test photograph. File "
        "sizes are medians over all 2,298 photographs", df,
        lambda c, r: f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]))


def t_hyper():
    hp = json.loads((RESULTS / "hyperparameters_finetune.json").read_text())
    fz = json.loads((RESULTS / "hyperparameters_frozen.json").read_text())
    rows = [
        ("Input", "whole photograph resized to 224 × 224, no crop", "keeps the full lesion and its border in view"),
        ("Fine-tuned backbones", "ResNet-50 (ImageNet V2 weights), EfficientNet-B0 (ImageNet V1)",
         "standard, widely used baselines of two sizes"),
        ("Optimiser", f"AdamW, learning rate {hp['lr']:g}, weight decay {hp['wd']:g}",
         "usual fine-tuning settings for pre-trained networks"),
        ("Schedule / epochs", f"cosine decay, at most {hp['epochs']} epochs, batch {hp['batch']}",
         "fits 6 GB of GPU memory"),
        ("Early stopping", f"balanced accuracy on clean validation patients, patience {hp['patience']}, best weights restored",
         "selection never sees test photographs"),
        ("Loss", "cross-entropy with inverse-frequency class weights", "melanoma is the smallest class"),
        ("Training augmentation", "flips, 90° rotations, brightness/contrast ±0.1",
         "lesion orientation carries no diagnostic meaning"),
        ("Compression-aware variant", f"replace image by a pre-drawn degraded variant with probability {hp['aug_prob']}",
         "matched control: everything else identical"),
        ("Frozen encoders", "DINOv2 ViT-S/14; CLIP ViT-B/16 (LAION-2B)", "two pre-training families"),
        ("Linear head", f"standardised features, logistic regression, C chosen from {', '.join(str(c) for c in fz['C_grid'])}",
         "C chosen on validation patients"),
        ("Repeats", f"{len(hp['seeds'])} seeds × {hp['folds']} folds = {len(hp['seeds']) * hp['folds']} runs per model and regime",
         "uncertainty from both data split and training"),
    ]
    df = pd.DataFrame(rows, columns=["Setting", "Value", "Reason"])
    put("03-methods.md", "hyper", "**Table 3** Training settings and the reason for each choice", df,
        lambda c, r: str(r[c]), bind=False)   # text only: nothing numeric to bind


def t_clean():
    s = pd.read_csv(RESULTS / "summary.csv")
    rows = []
    for mk in ORDER:
        for rg in REG:
            d = s[(s.model == mk) & (s.regime == rg) & (s.cond == "clean")]
            if d.empty:
                continue
            d = d.iloc[0]
            row = {"Classifier": f"{NAMES[mk]}, {REG[rg]}"}
            for col, k in [("Balanced accuracy", "bacc"), ("Macro F1", "macro_f1"), ("Macro AUC", "auc"),
                           ("Malignant sensitivity", "mal_sens"), ("ECE", "ece")]:
                row[col] = round(d[f"{k}_mean"], 3)
                row[f"{col}_sd"] = round(d[f"{k}_sd"], 3)
            rows.append(row)
    df = pd.DataFrame(rows)
    vis = [c for c in df.columns if not c.endswith("_sd")]
    df.to_csv(RESULTS / "table_clean.csv", index=False)
    lines = ["**Table 4** Performance on original photographs (mean ± SD over 10 runs: 5 folds × 2 seeds)", "",
             "<!-- results: table_clean.csv -->", "| " + " | ".join(vis) + " |", "|" + "|".join(["---"] * len(vis)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join([r[vis[0]]] + [pm(r[c], r[f"{c}_sd"]) for c in vis[1:]]) + " |")
    _replace("04-results.md", "clean", lines)


def _replace(section, name, lines):
    p = DRAFT / section
    t = p.read_text(encoding="utf-8")
    pat = re.compile(rf"<!-- BEGIN table:{name} -->.*?<!-- END table:{name} -->", re.S)
    if not pat.search(t):
        sys.exit(f"marker table:{name} missing in {section}")
    p.write_text(pat.sub(lambda _: f"<!-- BEGIN table:{name} -->\n" + "\n".join(lines) + f"\n<!-- END table:{name} -->", t),
                 encoding="utf-8")


def t_effects():
    b = pd.read_csv(RESULTS / "bootstrap_delta.csv")
    b = b[b.regime == "standard"]
    rows = []
    for mk in ORDER:
        for c in KEY_CONDS:
            g = b[(b.model == mk) & (b.cond == c)].set_index("metric")
            if g.empty:
                continue
            rows.append({"Classifier and condition": f"{NAMES[mk]} · {COND_LABEL[c]}",
                         "ΔBA": round(g.loc["bacc", "delta"], 3), "ΔBA low": round(g.loc["bacc", "lo"], 3),
                         "ΔBA high": round(g.loc["bacc", "hi"], 3),
                         "ΔSens_M": round(g.loc["mal_sens", "delta"], 3),
                         "Flip rate": round(g.loc["flip", "delta"], 3), "Flip low": round(g.loc["flip", "lo"], 3),
                         "Flip high": round(g.loc["flip", "hi"], 3)})
    put("04-results.md", "effects", "**Table 5** Paired change from the original photograph for classifiers trained on "
        "original photographs: balanced accuracy (ΔBA), malignant sensitivity (ΔSens_M) and flip rate, with 95% "
        "patient-cluster bootstrap intervals (pooled out-of-fold predictions, two seeds)", pd.DataFrame(rows),
        lambda c, r: f3(r[c]))


def t_floor():
    sf = pd.read_csv(RESULTS / "seed_floor.csv")
    b = pd.read_csv(RESULTS / "bootstrap_delta.csv")
    rows = []
    for mk in ["resnet50", "efficientnet_b0"]:   # frozen heads are deterministic: their seed floor is zero by design
        for rg in REG:
            f = sf[(sf.model == mk) & (sf.regime == rg)]
            if f.empty:
                continue
            g = b[(b.model == mk) & (b.regime == rg) & (b.metric == "flip")].set_index("cond")["delta"]
            rows.append({"Classifier": f"{NAMES[mk]}, {REG[rg]}",
                         "Seed-to-seed flip": round(f.seed_flip.iloc[0], 3),
                         "Flip q95": round(g["q95"], 3), "Flip app 960 px / q50": round(g["app_960_q50"], 3),
                         "Flip q10": round(g["q10"], 3)})
    put("04-results.md", "floor", "**Table 6** Fine-tuned classifiers: prediction flips caused by compression compared with the "
        "disagreement between two training seeds on the same original photographs", pd.DataFrame(rows), lambda c, r: f3(r[c]))


def t_whatsapp():
    w = pd.read_csv(RESULTS / "whatsapp_eval.csv")
    g = w.groupby(["model", "regime"]).mean(numeric_only=True).reset_index()
    rows = []
    for mk in ORDER:
        d = g[(g.model == mk) & (g.regime == "standard")]
        if d.empty:
            continue
        d = d.iloc[0]
        rows.append({"Classifier": NAMES[mk], "Accuracy original": round(d.acc_clean, 3),
                     "Accuracy WhatsApp": round(d.acc_real, 3), "Flip WhatsApp": round(d.flip_real, 3),
                     "Flip simulated profile": round(d.flip_sim, 3), "Agreement WhatsApp vs simulated": round(d.agree_real_sim, 3)})
    put("04-results.md", "whatsapp", "**Table 8** The 120 photographs sent through WhatsApp, scored by the fold models "
        "trained on original photographs (mean over two seeds). Simulated profile: longest side 1,600 px, JPEG quality 92",
        pd.DataFrame(rows), lambda c, r: f3(r[c]))


def t_roundtrip():
    d = pd.read_csv(RESULTS / "whatsapp_roundtrip.csv")
    big = d[d.orig_w.combine(d.orig_h, max) > 1600]
    rows = [{"Measured property": "Received files", "Value": len(d)},
            {"Measured property": "Files saved as JPEG with 4:2:0 chroma subsampling", "Value": int((d.subsampling == "4:2:0").sum())},
            {"Measured property": "Estimated JPEG quality (all files)", "Value": int(d.est_quality.median())},
            {"Measured property": "Originals longer than 1,600 px", "Value": len(big)},
            {"Measured property": "of which received with longest side 1,600 px", "Value": int((big[["recv_w", "recv_h"]].max(axis=1) == 1600).sum())},
            {"Measured property": "Originals at or below 1,600 px received at original size", "Value": int(((d.recv_w == d.orig_w) & (d.recv_h == d.orig_h))[d.orig_w.combine(d.orig_h, max) <= 1600].sum())},
            {"Measured property": "Median received size (% of original PNG)", "Value": round(100 * d.byte_ratio.median(), 1)}]
    assert d.est_quality.nunique() == 1, "quality varies - the table text must change"
    put("04-results.md", "roundtrip", "**Table 7** What WhatsApp Desktop (Windows, standard quality) did to the 120 "
        "photographs", pd.DataFrame(rows), lambda c, r: (str(int(r[c])) if float(r[c]).is_integer() else f"{float(r[c]):.1f}"))


if __name__ == "__main__":
    which = sys.argv[1:] or ["folds", "conditions", "hyper", "clean", "roundtrip", "effects", "floor", "whatsapp"]
    for w in which:
        globals()[f"t_{w}"]()
        print("table", w, "written")
