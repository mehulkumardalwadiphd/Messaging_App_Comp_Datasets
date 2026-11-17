"""Pick the images that go through a real WhatsApp round trip.

20 per class (all six classes), drawn with a fixed seed, at most one image per
patient, and with every image larger than 1600 px on the long edge in the pool
given priority for half of each class's quota so the app's downscale is
actually exercised. Copies the original PNGs into data/whatsapp/sent/ and
writes results/whatsapp_sample.csv.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from common import CLASSES, DATA, RESULTS, load_metadata

PER_CLASS = 20
rng = np.random.default_rng(7)
m = load_metadata()
m["long_edge"] = [max(Image.open(p).size) for p in m["path"]]
picked = []
for c in CLASSES:
    pool = m[m.diagnostic == c].sample(frac=1, random_state=int(rng.integers(1e9)))
    pool = pool.drop_duplicates("patient_id")
    big = pool[pool.long_edge > 1600].head(PER_CLASS // 2)
    rest = pool.drop(big.index).head(PER_CLASS - len(big))
    picked.append(pd.concat([big, rest]))
s = pd.concat(picked)
out = DATA / "whatsapp" / "sent"
out.mkdir(parents=True, exist_ok=True)
for p in s["path"]:
    shutil.copy2(p, out / os.path.basename(p))
s[["img_id", "patient_id", "diagnostic", "long_edge"]].to_csv(RESULTS / "whatsapp_sample.csv", index=False)
print(len(s), "images;", (s.long_edge > 1600).sum(), "above 1600 px;", s.diagnostic.value_counts().to_dict())
