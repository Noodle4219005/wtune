#!/usr/bin/env python3
"""One table: arm, what changed vs BASE, bit-identical?, worst per-field RMS."""
import argparse
import csv
import glob
import json
import os

CHANGED = {
    "BASE":   "-- reference -- 56 ranks x 2 thr, nproc 8x7, 12 tiles",
    "REPEAT": "nothing (determinism control)",
    "NPY4":   "Y-CUT ONLY nproc 8x7 -> 8x4 (32 ranks; i-extent unchanged)",
    "NPX4":   "X-CUT ONLY nproc 8x7 -> 4x7 (28 ranks; j-extent unchanged)",
    "SHAPE":  "BOTH AXES nproc 8x7 -> 7x8 (ranks fixed at 56)",
    "TILE8":  "numtiles 12 -> 8",
    "TILE1":  "numtiles 12 -> 1",
    "THR1":   "threads 2 -> 1 (ranks/shape/tiles fixed)",
    "DISTB":  "srun distribution cyclic -> block:block",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()

    arms = []
    csv_path = os.path.join(a.dir, "arms.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            arms = list(csv.DictReader(fh))

    print("%-7s %-52s %-14s %-9s %s" % ("arm", "what changed vs BASE", "bit-identical",
                                        "solver_s", "worst field RMS(arm-BASE)"))
    print("-" * 124)
    for row in arms:
        arm = row["arm"]
        verdict, worst = "-", "-"
        if arm == "BASE":
            verdict = "(reference)"
        else:
            bj = os.path.join(a.dir, "bitident_%s.json" % arm)
            if os.path.exists(bj):
                b = json.load(open(bj))
                if b["result"] == "PASS":
                    verdict = "YES  %d/%d" % (b["n_vars_compared"], b["n_vars_compared"])
                    worst = "0 (exact)"
                else:
                    verdict = "NO   %d/%d differ" % (b["n_vars_differing"], b["n_vars_compared"])
            rj = os.path.join(a.dir, "rms_%s.json" % arm)
            if os.path.exists(rj):
                f = json.load(open(rj))["fields"]
                if f:
                    worst = "%s rms=%.4g (max_abs=%.4g)" % (
                        f[0]["field"], f[0]["rms_diff"], f[0]["max_abs_diff"])
        if row["ok"] != "1":
            verdict = "RUN FAILED"
        print("%-7s %-52s %-14s %-9s %s"
              % (arm, CHANGED.get(arm, "?"), verdict, row.get("solver_s", "-"), worst))

    print()
    print("per-arm namelist.output proof (what WRF ITSELF parsed):")
    for row in arms:
        f = os.path.join(a.dir, "%s.namelist.output" % row["arm"])
        vals = {}
        if os.path.exists(f):
            for line in open(f, errors="ignore"):
                s = line.strip()
                for k in ("nproc_x", "nproc_y", "numtiles"):
                    if s.lower().startswith(k + " ") or s.lower().startswith(k + "="):
                        vals[k] = s.split("=")[-1].strip().strip(",")
        print("  %-7s asked nproc=%sx%s tiles=%s omp=%s dist=%s   |  WRF read %s"
              % (row["arm"], row["nx"], row["ny"], row["numtiles"], row["omp"], row["dist"],
                 vals or "<namelist.output not captured>"))

    print()
    print("differing-field detail (only for arms that failed the exact check):")
    for rj in sorted(glob.glob(os.path.join(a.dir, "rms_*.json"))):
        d = json.load(open(rj))
        print("  %s: %d fields differ; top 5:" % (d["label"], len(d["fields"])))
        for r in d["fields"][:5]:
            print("      %-12s rms=%.6g max_abs=%.6g n_diff=%d/%d"
                  % (r["field"], r["rms_diff"], r["max_abs_diff"],
                     r["n_diff_elements"], r["n_elements"]))


if __name__ == "__main__":
    main()
