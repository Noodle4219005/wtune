#!/usr/bin/env python3
"""Per-field RMS( candidate - reference ) for two wrfout netCDF files.

bitcmp.py already answers "are they exactly equal"; it reports max_abs_diff but not RMS,
and the fidelity gate's chaos floor is stated as an RMS.  This runs only on arms that
bitcmp.py has already failed, so its job is to say HOW FAR apart they are in the same
units the PHYSICS-GRAY floor was quoted in (T rms_diff 0.0013, PSFC 0.058).

Masking is off, exactly as in bitcmp.py, so no fill-value substitution changes a number.
"""
import argparse
import json
import sys

import numpy as np
from netCDF4 import Dataset


def raw(ds, name):
    var = ds.variables[name]
    var.set_auto_maskandscale(False)
    return np.asarray(var[...])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--label", default="arm")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    ref, cand = Dataset(a.reference), Dataset(a.candidate)
    rows = []
    for n in sorted(set(ref.variables) & set(cand.variables)):
        x, y = raw(ref, n), raw(cand, n)
        if x.shape != y.shape or x.dtype.kind not in "fiu":
            continue
        d = y.astype("f8") - x.astype("f8")
        finite = np.isfinite(d)
        if not finite.any():
            continue
        d = d[finite]
        n_diff = int(np.count_nonzero(d))
        if n_diff == 0:
            continue
        scale = float(np.sqrt(np.mean(x.astype("f8")[finite] ** 2))) or float("nan")
        rows.append({
            "field": n,
            "rms_diff": float(np.sqrt(np.mean(d ** 2))),
            "max_abs_diff": float(np.max(np.abs(d))),
            "n_diff_elements": n_diff,
            "n_elements": int(d.size),
            "rms_ref": scale,
            "normalized": float(np.sqrt(np.mean(d ** 2)) / scale) if scale == scale else None,
        })

    rows.sort(key=lambda r: -r["rms_diff"])
    print("RMS %s: %d fields differ" % (a.label, len(rows)))
    for r in rows[:a.top]:
        print("   %-12s rms_diff=%.6g  max_abs=%.6g  n_diff=%d/%d  rms_ref=%.6g"
              % (r["field"], r["rms_diff"], r["max_abs_diff"],
                 r["n_diff_elements"], r["n_elements"], r["rms_ref"]))
    for want in ("T", "PSFC", "U", "V", "W", "QVAPOR", "T2"):
        hit = next((r for r in rows if r["field"] == want), None)
        if hit:
            print("   [floor-field] %-7s rms_diff=%.6g" % (want, hit["rms_diff"]))
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump({"label": a.label, "reference": a.reference,
                       "candidate": a.candidate, "fields": rows}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
