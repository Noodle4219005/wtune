#!/usr/bin/env python3
"""Print the one summary table for the rk_ord alternation experiment.

Inputs are the artifacts the sbatch already wrote:
  arms.csv            one row per arm: wall, solver time, step count, decomposition
  fidelity_<ARM>.json fidelity_gate.py output (reference=A, twin=TWIN)
  bitident.json       bitcmp.py output for A2 vs A

Cost is reported three ways on purpose.  `wall` is what the arm actually cost in the
allocation, and for a 4-segment arm it contains 4 WRF start-ups instead of 1; `solver` is
the sum of WRF's own "Timing for main" lines and so excludes start-up and I/O entirely.
A candidate's honest cost comparison is against A2 (same segmentation overhead), not
against A -- the vs-A2 column exists so that mistake cannot be made by reading the table.
"""
import argparse
import csv
import json
import os

RMS_FIELDS = ["T2", "PSFC", "T", "U", "V", "W", "QVAPOR"]


def load_fid(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def num(v, fmt="%.4g"):
    if v is None or v == "":
        return "-"
    if isinstance(v, str):
        return v
    try:
        return fmt % float(v)
    except (TypeError, ValueError):
        return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="the job's output directory")
    a = ap.parse_args()
    d = a.dir

    with open(os.path.join(d, "arms.csv")) as fh:
        arms = list(csv.DictReader(fh))
    by_id = {r["arm"]: r for r in arms}

    def f(arm, key):
        try:
            return float(by_id[arm][key])
        except (KeyError, ValueError, TypeError):
            return None

    wall_a, solver_a = f("A", "wall_s"), f("A", "solver_s")
    wall_a2, solver_a2 = f("A2", "wall_s"), f("A2", "solver_s")

    bit = load_fid(os.path.join(d, "bitident.json"))

    print()
    print("=" * 132)
    case = os.environ.get("RK_ALT_CASE") or (arms[0].get("case") if arms and arms[0].get("case") else "")
    print("rk_ord ALTERNATION ACROSS RESTART BOUNDARIES -- %s%s"
          % (case + ", " if case else "", os.path.basename(d)))
    print("=" * 132)
    hdr = ("%-5s %-10s %-4s %-14s %9s %8s %8s %9s %8s %6s %9s %9s  %s"
           % ("arm", "rk_ord", "segs", "decomp", "wall_s", "d%_vs_A", "d%_vsA2",
              "solver_s", "dS%_vsA", "steps", "max_nd", "maxFloorR", "fidelity"))
    print(hdr)
    print("-" * 132)
    for r in arms:
        arm = r["arm"]
        w, s = f(arm, "wall_s"), f(arm, "solver_s")
        dw = "%+.2f" % (100.0 * (w / wall_a - 1)) if (w and wall_a) else "-"
        dw2 = "%+.2f" % (100.0 * (w / wall_a2 - 1)) if (w and wall_a2 and int(r["nsegs"]) > 1) else "-"
        ds = "%+.2f" % (100.0 * (s / solver_a - 1)) if (s and solver_a) else "-"
        fid = load_fid(os.path.join(d, "fidelity_%s.json" % arm))
        label = fid["label"] if fid else ("reference" if arm == "A" else "-")
        summ = (fid or {}).get("per_field_summary_at_last_time") or {}

        def vmax(key):
            vals = [v.get(key) for v in summ.values()
                    if isinstance(v, dict) and isinstance(v.get(key), (int, float))]
            return max(vals) if vals else None

        ok = "" if r.get("ok") == "1" else "  [RUN NOT OK]"
        # the TWIN legitimately differs in ranks x threads too, not only in nproc_x/y
        shape = "%sx%s %sx%st" % (r["nx"], r["ny"], r.get("ranks", "?"), r.get("omp", "?"))
        print("%-5s %-10s %-4s %-14s %9s %8s %8s %9s %8s %6s %9s %9s  %s%s"
              % (arm, r["rk_pattern"], r["nsegs"], shape,
                 num(w, "%.1f"), dw, dw2, num(s, "%.1f"), ds, r["steps"],
                 num(vmax("nd")), num(vmax("floor_ratio")), label, ok))
    print("-" * 132)

    print()
    print("per-field RMS( arm - A ) at the final frame   (floor = RMS(TWIN - A), the chaos floor)")
    print("%-6s %s" % ("field", " ".join("%13s" % x for x in ["A2", "B", "C", "D", "TWIN(floor)"])))
    fids = {arm: load_fid(os.path.join(d, "fidelity_%s.json" % arm))
            for arm in ("A2", "B", "C", "D", "TWIN")}
    for field in RMS_FIELDS:
        cells = []
        for arm in ("A2", "B", "C", "D", "TWIN"):
            fid = fids.get(arm)
            row = ((fid or {}).get("per_field_summary_at_last_time") or {}).get(field) or {}
            cells.append("%13s" % num(row.get("rms_diff"), "%.6g"))
        print("%-6s %s" % (field, " ".join(cells)))

    print()
    if bit:
        print("BIT-IDENTITY CONTROL  A2 vs A : %s   (%d variables compared, %d differ)"
              % (bit["result"], bit["n_vars_compared"], bit["n_vars_differing"]))
        if bit["result"] != "PASS":
            print("  *** The control failed.  A2 is rk_ord=3 in both arms, so any difference here is")
            print("  *** the restart chain itself, not rk_ord -- the C/D comparison is NOT meaningful.")
            for x in bit["differing"][:10]:
                print("      %-12s n_elem=%s max_abs=%s"
                      % (x["field"], x["n_diff_elements"], x["max_abs_diff"]))
    else:
        print("BIT-IDENTITY CONTROL  A2 vs A : NOT RUN (bitident.json missing)")
    print("=" * 132)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
