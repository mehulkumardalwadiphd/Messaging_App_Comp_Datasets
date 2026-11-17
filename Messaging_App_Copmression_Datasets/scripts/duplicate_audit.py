"""Audit PAD-UFES-20 for exact and near-duplicate photographs.

Exact: identical decoded pixels (SHA-256 of the RGB array).
Near: Pearson correlation > 0.995 between 64 x 64 greyscale signatures of the
224 x 224 clean inputs, restricted to pairs with the same aspect ratio.
Writes results/duplicate_pairs.csv and prints a summary; it also checks that
every duplicate pair falls in the same cross-validation fold.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from common import CACHE, RESULTS, get_splits, load_metadata, load_rgb

m, s = load_metadata(), get_splits()
X = np.load(CACHE / "inputs_clean.npy", mmap_mode="r")
sizes, hashes = [], []
for p in m["path"]:
    im = load_rgb(p)
    sizes.append(im.size)
    hashes.append(hashlib.sha256(np.asarray(im).tobytes()).hexdigest())
S = []
for i in range(len(m)):
    a = np.asarray(Image.fromarray(X[i]).convert("L").resize((64, 64), Image.BILINEAR), float).ravel()
    S.append((a - a.mean()) / (a.std() + 1e-9))
S = np.stack(S)
C = S @ S.T / S.shape[1]
asp = np.array([w / h for w, h in sizes])
rows = []
iu, ju = np.where(np.triu(C, 1) > 0.995)
for i, j in zip(iu, ju):
    if abs(asp[i] - asp[j]) > 0.02:
        continue
    rows.append(dict(img_a=m.img_id[i], img_b=m.img_id[j], patient_a=m.patient_id[i], patient_b=m.patient_id[j],
                     lesion_a=m.lesion_id[i], lesion_b=m.lesion_id[j], dx_a=m.diagnostic[i], dx_b=m.diagnostic[j],
                     corr=C[i, j], exact=hashes[i] == hashes[j], fold_a=s.fold[i], fold_b=s.fold[j]))
d = pd.DataFrame(rows)
d.to_csv(RESULTS / "duplicate_pairs.csv", index=False, float_format="%.5f")
print("pairs:", len(d), "| exact:", int(d.exact.sum()) if len(d) else 0)
if len(d):
    print("same patient:", int((d.patient_a == d.patient_b).sum()), "| same lesion:", int((d.lesion_a == d.lesion_b).sum()),
          "| label conflicts:", int((d.dx_a != d.dx_b).sum()), "| cross-fold:", int((d.fold_a != d.fold_b).sum()))
    print(d[d.dx_a != d.dx_b][["img_a", "img_b", "dx_a", "dx_b", "corr", "exact"]].to_string())
