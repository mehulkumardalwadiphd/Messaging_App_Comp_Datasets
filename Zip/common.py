"""Shared paths, metadata loading, split construction and the compression conditions.

Every number in the paper is produced from files written by the scripts in this
folder into ../results/. Nothing here is tuned after looking at test results:
the splits and the condition grid are fixed in this file before any model runs.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "data"
CACHE = PAPER / ".cache"
RESULTS = PAPER / "results"
FIG = PAPER / "figures"
for _p in (CACHE, RESULTS, FIG):
    _p.mkdir(exist_ok=True)

CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
MALIGNANT = {"BCC", "MEL", "SCC"}
INPUT = 224          # model input side (pixels)
SPLIT_SEED = 2026    # fixed before any model was trained
N_FOLDS = 5


# --------------------------------------------------------------------------
# Compression conditions. Applied to the image at its native resolution,
# BEFORE the model's own resize to 224 x 224, which is the order in which a
# teledermatology photo meets a classifier: phone -> messaging app -> server.
#   cap  : longest edge after the app's downscale (None = no downscale)
#   q    : JPEG quality factor (libjpeg scale, 4:2:0 chroma subsampling)
#   gens : how many save-and-resend cycles (each one re-encodes)
# --------------------------------------------------------------------------
CONDITIONS: dict[str, dict] = {
    "clean":          dict(cap=None, q=None, gens=0),
    # quality sweep, no resize (dose-response)
    "q95":            dict(cap=None, q=95, gens=1),   # near-lossless null
    "q90":            dict(cap=None, q=90, gens=1),
    "q80":            dict(cap=None, q=80, gens=1),
    "q70":            dict(cap=None, q=70, gens=1),
    "q60":            dict(cap=None, q=60, gens=1),
    "q50":            dict(cap=None, q=50, gens=1),
    "q40":            dict(cap=None, q=40, gens=1),
    "q30":            dict(cap=None, q=30, gens=1),
    "q20":            dict(cap=None, q=20, gens=1),
    "q10":            dict(cap=None, q=10, gens=1),
    # messaging-app profiles (simulated): downscale + recompress
    "app_1600_q80":   dict(cap=1600, q=80, gens=1),
    "app_1600_q70":   dict(cap=1600, q=70, gens=1),
    "app_1280_q60":   dict(cap=1280, q=60, gens=1),
    "app_960_q50":    dict(cap=960, q=50, gens=1),
    # repeated save-and-resend of the 1280/q60 profile
    "resend_1280_q60_x3": dict(cap=1280, q=60, gens=3),
    "resend_1280_q60_x5": dict(cap=1280, q=60, gens=5),
}
COND_NAMES = list(CONDITIONS)

# Compression-aware training: each training image gets N_AUG pre-drawn variants
# from a distribution that covers the app profiles but stops at q = 25, so the
# q20 and q10 test conditions remain outside the training range.
N_AUG = 4
AUG_Q = (25, 95)
AUG_CAPS = [None, 1600, 1280, 960, 720]
AUG_GENS = [1, 1, 1, 2]


def load_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    im.load()
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def jpeg_roundtrip(im: Image.Image, q: int) -> tuple[Image.Image, int]:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(q), subsampling=2, optimize=False)
    n = buf.tell()
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB"), n


def apply_condition(im: Image.Image, cap, q, gens) -> tuple[Image.Image, int | None]:
    """Return the degraded image and the byte size of the last JPEG written."""
    out, nbytes = im, None
    for _ in range(gens):
        if cap is not None and max(out.size) > cap:
            s = cap / max(out.size)
            out = out.resize((max(1, round(out.size[0] * s)), max(1, round(out.size[1] * s))), Image.LANCZOS)
        out, nbytes = jpeg_roundtrip(out, q)
    return out, nbytes


def to_input(im: Image.Image) -> np.ndarray:
    """The model's own preprocessing: resize the whole photograph to 224 x 224."""
    return np.asarray(im.resize((INPUT, INPUT), Image.BILINEAR), dtype=np.uint8)


def load_metadata() -> pd.DataFrame:
    m = pd.read_csv(DATA / "metadata.csv")
    paths = {p.name: p for p in (DATA / "images").glob("*/*.png")}
    m["path"] = m["img_id"].map(paths)
    assert m["path"].notna().all(), "image file missing for some metadata rows"
    m["y"] = m["diagnostic"].map({c: i for i, c in enumerate(CLASSES)})
    return m.sort_values("img_id").reset_index(drop=True)


def make_splits(m: pd.DataFrame) -> pd.DataFrame:
    """Patient-grouped stratified 5-fold CV; inside each training part one
    patient-grouped sixth is the validation set for early stopping / C choice."""
    from sklearn.model_selection import StratifiedGroupKFold
    s = pd.DataFrame({"img_id": m["img_id"], "patient_id": m["patient_id"], "diagnostic": m["diagnostic"]})
    s["fold"] = -1
    outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    for f, (_, te) in enumerate(outer.split(m, m["y"], m["patient_id"])):
        s.loc[te, "fold"] = f
    for f in range(N_FOLDS):
        tr = np.where(s["fold"] != f)[0]
        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SPLIT_SEED + f)
        _, va = next(inner.split(tr, m["y"].values[tr], m["patient_id"].values[tr]))
        col = np.zeros(len(s), dtype=bool)
        col[tr[va]] = True
        s[f"val_f{f}"] = col
    return s


def get_splits() -> pd.DataFrame:
    p = RESULTS / "splits.csv"
    m = load_metadata()
    if not p.exists():
        make_splits(m).to_csv(p, index=False)
    s = pd.read_csv(p)
    assert (s["img_id"].values == m["img_id"].values).all()
    return s


def code_hash(*files: str) -> str:
    h = hashlib.sha256()
    for f in ("common.py",) + files:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:10]
