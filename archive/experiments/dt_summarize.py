#!/usr/bin/env python3
"""Summary table for the RK2-with-a-shorter-time_step trade-off experiment.

Reads the artifacts dt_tradeoff.sbatch wrote:
  arms.csv              one row per arm: rk_ord, time_step, nss, wall, solver, steps
  fidelity_<ARM>.json   fidelity_gate.py output, reference = the TRUTH arm
  repro.json            bitcmp.py, this job's REF vs job 1080968's arm A

Three things are computed here rather than left to the reader:

1. COST.  Solver seconds and s/step per arm, and cost relative to the REF arm.  Then a
   two-parameter fit  solver = R + steps * c  over the RK2 arms alone (they share rk_ord,
   so c is one number) -- R is the dt-invariant cost, dominated by radiation, which fires
   on a fixed WALL-CLOCK cadence (radt minutes) and therefore the same number of times in
   a 12 h forecast whatever the time step is.  From that fit the BREAK-EVEN time_step is
   solved exactly: the dt at which RK2 costs the same as the REF arm.  This is the number
   the whole hypothesis turns on, and it is not something an arm can be run at directly.

2. ERROR SCALING.  Per field, a least-squares fit of log(RMS vs TRUTH) against log(dt)
   over the RK2 arms.  The exponent p is the claim under test: p ~ 2 means the RK2 time
   truncation dominates and shortening dt buys accuracy at the advertised rate; p ~ 0
   means something else dominates the error and the whole direction is closed.

3. DOMINANCE.  An arm dominates REF only if it is BOTH cheaper (solver) AND closer to
   TRUTH (max nd over the fields).  Reported per arm, with no partial credit.
"""
import argparse
import csv
import json
import math
import os

RMS_FIELDS = ["T2", "PSFC", "T", "U", "V", "W", "QVAPOR"]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def num(v, fmt="%.4g"):
    if v is None or v == "":
        return "-"
    try:
        return fmt % float(v)
    except (TypeError, ValueError):
        return str(v)


def lstsq(xs, ys):
    """slope, intercept of a plain least-squares line; None when under-determined."""
    n = len(xs)
    if n < 2:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = a.dir
    case = os.environ.get("DT_CASE", "")
    truth = os.environ.get("DT_TRUTH", "TRUTH")
    refarm = os.environ.get("DT_REF", "REF")

    with open(os.path.join(d, "arms.csv")) as fh:
        arms = list(csv.DictReader(fh))
    by = {r["arm"]: r for r in arms}
    fid = {r["arm"]: load(os.path.join(d, "fidelity_%s.json" % r["arm"])) for r in arms}

    def fnum(arm, key):
        try:
            return float(by[arm][key])
        except (KeyError, ValueError, TypeError):
            return None

    total_s = None
    if refarm in by:
        total_s = int(by[refarm]["time_step"]) * int(by[refarm]["steps_expected"])
    solver_ref = fnum(refarm, "solver_s")

    def maxnd(arm):
        summ = ((fid.get(arm) or {}).get("per_field_summary_at_last_time") or {})
        vals = [v.get("nd") for v in summ.values() if isinstance(v, dict)
                and isinstance(v.get("nd"), (int, float))]
        return max(vals) if vals else None

    W = 138
    print()
    print("=" * W)
    print("RK2 WITH A SHORTENED time_step vs RK3 -- %s%s" % (case + ", " if case else "",
                                                             os.path.basename(d)))
    print("   every arm one wrf.exe call, same decomposition/binary/allocation;"
          " scored against arm %s" % truth)
    print("=" * W)
    print("%-7s %-7s %-6s %-5s %-7s %-10s %-11s %-10s %-10s %-10s  %s"
          % ("arm", "rk_ord", "dt_s", "nss", "steps", "solver_s", "s/step", "cost/REF",
             "max_nd", "maxFloorR", "fidelity vs " + truth))
    print("-" * W)
    for r in arms:
        arm = r["arm"]
        s = fnum(arm, "solver_s")
        st = int(r["steps"]) if r["steps"] else 0
        summ = ((fid.get(arm) or {}).get("per_field_summary_at_last_time") or {})
        fr = [v.get("floor_ratio") for v in summ.values() if isinstance(v, dict)
              and isinstance(v.get("floor_ratio"), (int, float))]
        label = (fid.get(arm) or {}).get("label") or ("TRUTH (reference)" if arm == truth else "-")
        ok = "" if r.get("ok") == "1" else "  [RUN NOT OK]"
        print("%-7s %-7s %-6s %-5s %-7s %-10s %-11s %-10s %-10s %-10s  %s%s"
              % (arm, r["rk_ord"], r["time_step"], r["nss"], r["steps"],
                 num(s, "%.2f"), num(s / st if (s and st) else None, "%.5f"),
                 num(s / solver_ref if (s and solver_ref) else None, "%.4f"),
                 num(maxnd(arm)), num(max(fr) if fr else None), label, ok))
    print("-" * W)

    # ---------------------------------------------------------------- 1. cost model
    print()
    print("COST MODEL  solver = R + steps * c   fitted over the rk_ord=2 arms only")
    rk2 = [r for r in arms if r["rk_ord"] == "2" and r.get("ok") == "1" and fnum(r["arm"], "solver_s")]
    c = R = None
    if len(rk2) >= 2:
        c, R = lstsq([float(r["steps"]) for r in rk2], [fnum(r["arm"], "solver_s") for r in rk2])
    if c is None:
        print("  not enough rk_ord=2 arms to fit")
    else:
        print("  c = %.5f s/step (per-step cost that scales with the step count)" % c)
        print("  R = %.2f s      (dt-invariant cost: radiation fires on a fixed radt cadence,"
              " so its call count does not change with dt)" % R)
        if solver_ref and total_s:
            margin = solver_ref - R
            if margin > 0 and c > 0:
                dt_be = total_s * c / margin
                print("  BREAK-EVEN time_step = %.2f s  (the dt at which rk_ord=2 costs the same"
                      " %.2f s as %s)" % (dt_be, solver_ref, refarm))
                print("  => a shortened-dt RK2 arm is cheaper than %s only for time_step > %.2f s"
                      % (refarm, dt_be))
            else:
                print("  BREAK-EVEN undefined: the fitted dt-invariant cost R already exceeds"
                      " %s's solver time" % refarm)

    # ---------------------------------------------------------------- 2. error vs TRUTH
    print()
    print("PER-FIELD RMS( arm - %s ) AT THE FINAL FRAME, and the fitted exponent p in"
          " RMS ~ dt^p over the rk_ord=2 arms" % truth)
    cols = [r["arm"] for r in arms if r["arm"] != truth]
    print("%-8s %s %10s" % ("field", " ".join("%13s" % ("%s(dt=%s)" % (x, by[x]["time_step"]))
                                              for x in cols), "p(RK2)"))
    exps = []
    for field in RMS_FIELDS:
        cells, xs, ys = [], [], []
        for arm in cols:
            row = ((fid.get(arm) or {}).get("per_field_summary_at_last_time") or {}).get(field) or {}
            v = row.get("rms_diff")
            cells.append("%13s" % num(v, "%.6g"))
            if by[arm]["rk_ord"] == "2" and isinstance(v, (int, float)) and v > 0:
                xs.append(math.log(float(by[arm]["time_step"]))); ys.append(math.log(v))
        p, _ = lstsq(xs, ys)
        if p is not None:
            exps.append(p)
        print("%-8s %s %10s" % (field, " ".join(cells), num(p, "%.2f")))
    if exps:
        exps.sort()
        med = exps[len(exps) // 2] if len(exps) % 2 else 0.5 * (exps[len(exps) // 2 - 1] + exps[len(exps) // 2])
        print()
        print("  median p over %d fields = %.2f   (p~2 => the RK2 time truncation dominates the"
              " error; p~0 => it does not, and shortening dt buys nothing)" % (len(exps), med))

    # ---------------------------------------------------------------- 3. dominance
    print()
    nd_ref = maxnd(refarm)
    print("DOMINANCE vs %s (solver %s s, max_nd vs %s = %s) -- an arm dominates only if it is"
          " BOTH cheaper AND closer to %s"
          % (refarm, num(solver_ref, "%.2f"), truth, num(nd_ref), truth))
    for r in arms:
        arm = r["arm"]
        if arm in (refarm, truth):
            continue
        s, nd = fnum(arm, "solver_s"), maxnd(arm)
        if s is None or nd is None or solver_ref is None or nd_ref is None:
            verdict = "INCOMPLETE"
        else:
            cheaper, closer = s < solver_ref, nd < nd_ref
            verdict = ("DOMINATES" if (cheaper and closer) else
                       "dominated (worse on both)" if not (cheaper or closer) else
                       "cheaper but less accurate" if cheaper else "more accurate but more expensive")
        print("  %-7s solver %+7s %%   max_nd %-10s   %s"
              % (arm, num(100.0 * (s / solver_ref - 1) if (s and solver_ref) else None, "%+.2f"),
                 num(nd), verdict))

    print()
    rep = load(os.path.join(d, "repro.json"))
    if rep:
        print("CROSS-JOB REPRO CONTROL  %s vs job 1080968 arm A : %s   (%d variables, %d differ)"
              % (refarm, rep["result"], rep["n_vars_compared"], rep["n_vars_differing"]))
        if rep["result"] != "PASS":
            print("  *** identical configuration in a different allocation did NOT reproduce.")
            print("  *** The borrowed TWIN chaos floor is then not comparable: read the raw RMS")
            print("  *** columns and the cost columns only, and ignore maxFloorR.")
            for x in rep["differing"][:10]:
                print("      %-12s n_elem=%s max_abs=%s"
                      % (x["field"], x["n_diff_elements"], x["max_abs_diff"]))
    else:
        print("CROSS-JOB REPRO CONTROL: NOT RUN (repro.json missing)")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
