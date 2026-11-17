"""Tests for the compression conditions and the splits (pytest)."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONDITIONS, apply_condition, get_splits, load_metadata, to_input


def _img(w=2000, h=1500, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 50, w // 50, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((w, h), Image.BICUBIC)


def test_clean_is_identity():
    im = _img()
    out, nb = apply_condition(im, **CONDITIONS["clean"])
    assert nb is None and np.array_equal(np.asarray(out), np.asarray(im))


def test_cap_limits_long_edge_and_keeps_aspect():
    im = _img(2000, 1500)
    out, _ = apply_condition(im, cap=1280, q=60, gens=1)
    assert max(out.size) == 1280 and abs(out.size[0] / out.size[1] - 2000 / 1500) < 0.01


def test_cap_never_upscales():
    im = _img(800, 600)
    out, _ = apply_condition(im, cap=1600, q=80, gens=1)
    assert out.size == (800, 600)


def test_bytes_fall_with_quality():
    im = _img()
    sizes = [apply_condition(im, cap=None, q=q, gens=1)[1] for q in (95, 70, 40, 10)]
    assert sizes == sorted(sizes, reverse=True)


def test_deterministic():
    im = _img()
    a, _ = apply_condition(im, cap=1280, q=60, gens=3)
    b, _ = apply_condition(im, cap=1280, q=60, gens=3)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_error_grows_with_compression():
    im = _img()
    ref = to_input(im).astype(float)
    e = [np.abs(to_input(apply_condition(im, None, q, 1)[0]) - ref).mean() for q in (95, 50, 10)]
    assert e[0] < e[1] < e[2]


def test_splits_patient_disjoint_and_complete():
    s = get_splits()
    m = load_metadata()
    assert len(s) == len(m) and set(s["fold"]) == set(range(5))
    for f in range(5):
        te = set(s.loc[s.fold == f, "patient_id"])
        tr = set(s.loc[s.fold != f, "patient_id"])
        va = set(s.loc[s[f"val_f{f}"], "patient_id"])
        assert not (te & tr)
        assert va <= tr and not (va & set(s.loc[(s.fold != f) & ~s[f"val_f{f}"], "patient_id"]))
        assert not s.loc[s.fold == f, f"val_f{f}"].any()
