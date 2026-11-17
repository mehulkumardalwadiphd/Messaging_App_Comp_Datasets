"""Build the 224 x 224 uint8 input arrays for every image under every condition,
plus N_AUG compression-aware training variants per image, and record the byte
cost of each condition. Run once; everything downstream reads these arrays.

Outputs
  .cache/inputs_<cond>.npy      (N, 224, 224, 3) uint8, one per condition
  .cache/inputs_aug.npy         (N, N_AUG, 224, 224, 3) uint8
  results/condition_bytes.csv   per image x condition: JPEG bytes, long edge
  results/aug_draws.csv         the drawn (q, cap, gens) of every training variant
  results/dataset_info.json     provenance and counts
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (AUG_CAPS, AUG_GENS, AUG_Q, CACHE, COND_NAMES, CONDITIONS, DATA, N_AUG, RESULTS,
                    apply_condition, get_splits, load_metadata, load_rgb, to_input)


def work(args):
    i, path = args
    im = load_rgb(path)
    orig_bytes = os.path.getsize(path)
    arrs, rows = {}, []
    for c in COND_NAMES:
        sp = CONDITIONS[c]
        out, nb = apply_condition(im, sp["cap"], sp["q"], sp["gens"])
        arrs[c] = to_input(out)
        rows.append(dict(idx=i, cond=c, bytes=orig_bytes if nb is None else nb, long_edge=max(out.size),
                         orig_long_edge=max(im.size), orig_png_bytes=orig_bytes))
    rng = np.random.default_rng(10_000 + i)
    aug, draws = [], []
    for k in range(N_AUG):
        q = int(rng.integers(AUG_Q[0], AUG_Q[1] + 1))
        cap = AUG_CAPS[int(rng.integers(len(AUG_CAPS)))]
        gens = AUG_GENS[int(rng.integers(len(AUG_GENS)))]
        out, _ = apply_condition(im, cap, q, gens)
        aug.append(to_input(out))
        draws.append(dict(idx=i, k=k, q=q, cap=cap if cap else 0, gens=gens))
    return i, arrs, np.stack(aug), rows, draws


def main(limit: int | None = None):
    m = load_metadata()
    get_splits()  # freeze the splits before anything else exists
    n = len(m) if limit is None else limit
    jobs = list(zip(range(n), m["path"].tolist()[:n]))
    out = {c: np.lib.format.open_memmap(CACHE / f"inputs_{c}.npy", mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3))
           for c in COND_NAMES}
    aug = np.lib.format.open_memmap(CACHE / "inputs_aug.npy", mode="w+", dtype=np.uint8, shape=(n, N_AUG, 224, 224, 3))
    rows, draws = [], []
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as pool:
        for done, (i, arrs, a, r, d) in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            for c in COND_NAMES:
                out[c][i] = arrs[c]
            aug[i] = a
            rows += r
            draws += d
            if done % 100 == 0 or done == n:
                el = time.time() - t0
                print(f"[precompute] {done}/{n}  {el:.0f}s elapsed  ETA {el / done * (n - done):.0f}s", flush=True)
    for a in list(out.values()) + [aug]:
        a.flush()
    sfx = "" if limit is None else "_smoke"
    pd.DataFrame(rows).sort_values(["idx", "cond"]).to_csv(RESULTS / f"condition_bytes{sfx}.csv", index=False)
    pd.DataFrame(draws).sort_values(["idx", "k"]).to_csv(RESULTS / f"aug_draws{sfx}.csv", index=False)
    if limit is None:
        info = dict(dataset="PAD-UFES-20", doi="https://doi.org/10.17632/zr7vgbcyr2.1", licence="CC BY 4.0",
                    source="https://data.mendeley.com/public-api/zip/zr7vgbcyr2/download/1",
                    zip_sha256="e8c7e17bac1698c97e44d4096ec20ac1b91c135285c1446b7b2e7ebbc9be933c",
                    zip_bytes=3593344681, downloaded="2026-11-12",
                    n_images=int(len(m)), n_patients=int(m.patient_id.nunique()), n_lesions=int(m.lesion_id.nunique()),
                    class_counts={k: int(v) for k, v in m.diagnostic.value_counts().items()},
                    rgba_images_composited_on_white=True)
        (RESULTS / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"[precompute] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
