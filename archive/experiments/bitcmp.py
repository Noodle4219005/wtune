#!/usr/bin/env python3
"""EXACT (no tolerance) comparison of two wrfout netCDF files.

This is the control check for arm A2 (rk_ord=3 run as a restart chain) against arm A
(rk_ord=3 in one call).  WRF restart is documented to be bit-identical, and if it is not
bit-identical HERE then every other number in the experiment is measuring the restart, not
rk_ord -- so this is reported PASS/FAIL on its own, before any tolerance-based gate.

Exact means exact: numpy array_equal on the raw stored values with masking turned OFF, so
no fill-value substitution, no float tolerance, no roundoff allowance.  NaN is compared
bit-wise too (a NaN in the same place in both files counts as equal; a NaN in only one does
not).
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


def exact_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, None, None
    if a.dtype.kind in "SU":
        return bool(np.array_equal(a, b)), None, None
    same = (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else (a == b)
    n_diff = int(np.count_nonzero(~same))
    if n_diff == 0:
        return True, 0, 0.0
    with np.errstate(invalid="ignore"):
        d = np.abs(a.astype("f8") - b.astype("f8"))
        mx = float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")
    return False, n_diff, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--label", default="A2-vs-A")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    ref = Dataset(a.reference)
    cand = Dataset(a.candidate)
    ref.set_auto_maskandscale(False)
    cand.set_auto_maskandscale(False)

    names = sorted(set(ref.variables) | set(cand.variables))
    only_ref = sorted(set(ref.variables) - set(cand.variables))
    only_cand = sorted(set(cand.variables) - set(ref.variables))
    differing = []
    compared = 0
    for n in names:
        if n in only_ref or n in only_cand:
            continue
        ok, n_diff, mx = exact_equal(raw(ref, n), raw(cand, n))
        compared += 1
        if not ok:
            differing.append({"field": n, "n_diff_elements": n_diff, "max_abs_diff": mx})

    passed = not differing and not only_ref and not only_cand
    out = {
        "label": a.label,
        "reference": a.reference,
        "candidate": a.candidate,
        "n_vars_compared": compared,
        "n_vars_differing": len(differing),
        "vars_only_in_reference": only_ref,
        "vars_only_in_candidate": only_cand,
        "differing": differing[:40],
        "result": "PASS" if passed else "FAIL",
    }
    print("BIT-IDENTITY %s: %s  (%d/%d variables exactly equal)"
          % (a.label, out["result"], compared - len(differing), compared))
    for d in differing[:10]:
        print("   DIFFERS %-12s n_elem=%s max_abs=%s"
              % (d["field"], d["n_diff_elements"], d["max_abs_diff"]))
    if only_ref or only_cand:
        print("   variable sets differ: only_ref=%s only_cand=%s" % (only_ref, only_cand))
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
