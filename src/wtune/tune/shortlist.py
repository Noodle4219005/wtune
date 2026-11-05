"""Per-machine layout ranker: which decompositions to measure first.

A ridge model fitted on a machine's own probe measurements (machine/shortlist/<hardware_id>.json)
predicts log(s/step) minus the group mean of the candidates of ONE (case, nodes), so it only orders
layouts; it says nothing about the absolute time. f1-grace: 29 groups, 632 settings, omp=1 dmpar
only -- top-3 contained the fastest layout in 89.7 % of groups (leave-one-case-out), but pruning
by the same signal did not hold (recall 93.1 %), so this orders and never drops.
Off with WTUNE_SHORTLIST=0; a machine without a model file is unchanged.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

_DIR = Path(__file__).resolve().parents[3] / "machine" / "shortlist"
_cache: dict = {}


def load(hardware_id):
    """The model dict of `hardware_id`, or None (no file, or WTUNE_SHORTLIST=0)."""
    if os.environ.get("WTUNE_SHORTLIST") == "0" or not hardware_id:
        return None
    if hardware_id not in _cache:
        p = _DIR / ("%s.json" % hardware_id)
        _cache[hardware_id] = json.loads(p.read_text()) if p.is_file() else None
    return _cache[hardware_id]


def _heaviest_patch_work(c: dict, e_vert: float) -> float:
    """sum over the domains of the fattest patch's cells (time-step ratios are not in the
    candidate, so a nest counts once per d01 step)."""
    return sum(d["patch_x_max"] * d["patch_y_max"] for d in c["domains"]) * e_vert


def _features(c: dict, e_vert: float, hmin: float) -> list[float]:
    d1 = c["domains"][0]
    px, py = d1["patch_x_max"], d1["patch_y_max"]
    h = _heaviest_patch_work(c, e_vert)
    return [math.log(h), math.log(h / hmin), math.log(px / py), 2.0 * (px + py) / (px * py),
            math.log(max(1, int(c["numtiles"]))), 1.0 if int(c.get("tile_strategy", 0) or 0) else 0.0,
            float(c.get("numtiles_x", 0) or 0), float(c.get("numtiles_y", 0) or 0),
            1.0 if c.get("dist") == "block" else 0.0, math.log(c["nx"] / c["ny"])]


def usable(c: dict) -> bool:
    """The model was fitted on omp=1, default-tile layouts; others are not ranked by it."""
    return (int(c.get("omp", 1) or 1) == 1 and not int(c.get("tile_strategy", 0) or 0)
            and bool(c.get("domains")))


def rank(model: dict, cands: list[dict], e_vert: float) -> list[dict]:
    """`cands` (usable ones) best predicted first; ties keep the input order."""
    cands = [c for c in cands if usable(c)]
    if not cands:
        return []
    hmin = min(_heaviest_patch_work(c, e_vert) for c in cands)
    mean, scale, coef = model["scaler_mean"], model["scaler_scale"], model["coef"]

    def score(c):
        x = _features(c, e_vert, hmin)
        return model.get("intercept", 0.0) + sum(
            b * (xi - m) / (s or 1.0) for b, xi, m, s in zip(coef, x, mean, scale))

    return sorted(cands, key=score)
