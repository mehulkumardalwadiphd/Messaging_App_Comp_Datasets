"""Pair every image saved back from the WhatsApp round trip with the original
it came from (by pixel content, since WhatsApp renames files), estimate the
JPEG quality WhatsApp used from the file's own quantisation table, and write
results/whatsapp_roundtrip.csv. Matched copies are written to
data/whatsapp/matched/<img_id>.jpeg.

Quality estimate: the sum of the luminance quantisation table is compared with
the sum of the IJG standard table scaled for every quality 1..100 (the sum is
independent of the coefficient order the reader returns); the closest wins.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from common import DATA, RESULTS, load_rgb

STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def ijg_table(q):
    s = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((STD_LUMA * s + 50) / 100), 1, 255)


def est_quality(tab):
    t = np.array(list(tab)).sum()
    return int(min(range(1, 101), key=lambda q: abs(ijg_table(q).sum() - t)))


def sig(im, n=32):
    a = np.asarray(im.convert("L").resize((n, n), Image.BILINEAR), dtype=float).ravel()
    return (a - a.mean()) / (a.std() + 1e-9)


W = DATA / "whatsapp"
sample = pd.read_csv(RESULTS / "whatsapp_sample.csv")
orig = {r.img_id: load_rgb(W / "sent" / r.img_id) for r in sample.itertuples()}
from scipy.optimize import linear_sum_assignment

keys = list(orig)
O = np.stack([sig(orig[k], 96) for k in keys])
recv = sorted(p for p in (W / "received").iterdir() if p.name.startswith("wa_"))
R = np.stack([sig(Image.open(p), 96) for p in recv])
C = R @ O.T / O.shape[1]                         # Pearson correlation, received x original
ri, ci = linear_sum_assignment(-C)               # one-to-one assignment
rows = []
for a, b in zip(ri, ci):
    p, k = recv[a], keys[b]
    im = Image.open(p)
    others = np.delete(C[a], b)
    rows.append(dict(wa_file=p.name, img_id=k, match_corr=C[a, b], second_best_corr=others.max(),
                     orig_w=orig[k].size[0], orig_h=orig[k].size[1], recv_w=im.size[0], recv_h=im.size[1],
                     orig_png_bytes=os.path.getsize(W / "sent" / k), recv_bytes=os.path.getsize(p),
                     est_quality=est_quality(im.quantization[0]),
                     subsampling="4:2:0" if im.layer[0][1:3] == (2, 2) else str(im.layer[0][1:3])))
d = pd.DataFrame(rows).merge(sample[["img_id", "diagnostic"]], on="img_id")
assert (abs(d.orig_w / d.orig_h - d.recv_w / d.recv_h) < 0.02).all(), "matched pair with different aspect ratio"
assert d.match_corr.min() > 0.95, "a weak match - inspect by eye"
assert d["img_id"].is_unique, "two received files matched the same original"
assert set(d["img_id"]) == set(sample["img_id"]), "some originals have no received copy"
out = W / "matched"
out.mkdir(exist_ok=True)
for r in d.itertuples():
    shutil.copy2(W / "received" / r.wa_file, out / (r.img_id[:-4] + ".jpeg"))
d["downscaled"] = d.recv_w < d.orig_w
d["byte_ratio"] = d.recv_bytes / d.orig_png_bytes
d.sort_values("img_id").to_csv(RESULTS / "whatsapp_roundtrip.csv", index=False, float_format="%.4f")
print(len(d), "matched; min corr", round(d.match_corr.min(), 3), "max 2nd", round(d.second_best_corr.max(), 3))
print("quality:", d.est_quality.value_counts().sort_index().to_dict())
print("downscaled:", int(d.downscaled.sum()), "; recv long edge max:", int(d[["recv_w", "recv_h"]].max(axis=1).max()))
print("long edge of downscaled:", sorted(d.loc[d.downscaled, ["recv_w", "recv_h"]].max(axis=1).unique().tolist())[:10])
print("median byte ratio", round(d.byte_ratio.median(), 4), "subsampling", d.subsampling.value_counts().to_dict())
