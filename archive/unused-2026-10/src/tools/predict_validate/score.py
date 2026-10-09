"""Score the wtune predict validation (pre-registered 2026-09-23).

Whole-run wall  : registered K=1 prediction (preregistered_k1.csv) vs ground-truth wall (truth.csv).
Compute component (what restart sampling is meant to fix): normal-step d01 compute of the ground
truth vs s/step x (true number of normal steps), for
  K1_reg  campaign probe s/step (the registered K=1 input)
  K1_job  win_0 (cold start at t0) from the windows job
  K4      trapezoid over win_0..win_3 at 0/25/50/75 % of the run, flat to 100 %
Ground-truth steps are classified from rsl.error.0000: step 1 dropped; a step > 1.8x the median is
a radiation or write step; the rest are normal steps.
"""
import csv
import json
import os
import re
import statistics
import sys

V = os.path.expanduser("~/wtune-runs/predict-validate")
CASES = ["matthew", "severe2023", "sc26p2", "sc26p2narr", "conus12", "karina", "c34", "sc26p1"]


def truth_steps(case):
    t = open(os.path.join(V, case, "rsl.error.0000"), errors="ignore").read()
    st = [float(x) for x in re.findall(r"Timing for main: time \S+ on domain\s+1:\s+([\d.]+) elapsed", t)][1:]
    med = statistics.median(st)
    normal = [s for s in st if s <= 1.8 * med]
    return normal


def main():
    reg = {r["case"]: r for r in csv.DictReader(open(os.path.join(V, "preregistered_k1.csv")))}
    rows = []
    for c in CASES:
        d = os.path.join(V, c)
        if not os.path.exists(os.path.join(d, "truth.csv")):
            continue
        tc = open(os.path.join(d, "truth.csv")).read().strip().split(",")
        wall, ok = float(tc[5]), tc[4]
        if ok != "1":
            rows.append(dict(case=c, note="truth run failed"))
            continue
        pred = float(reg[c]["pred_total_s"])
        lo, hi = float(reg[c]["pred_lo_s"]), float(reg[c]["pred_hi_s"])
        normal = truth_steps(c)
        act_norm = sum(normal)
        n_norm = len(normal)
        p = json.load(open(os.path.join(d, "prediction_k1.json")))
        s_reg = p.get("compute", {}).get("s_normal") or p.get("compute", {}).get("s_per_step")
        row = dict(case=c, wall_s=wall, pred_s=pred, wall_err=pred / wall - 1, in_range=lo <= wall <= hi,
                   n_normal=n_norm, act_normal_s=act_norm, s_truth_median=statistics.median(normal))
        if s_reg:
            row["K1_reg_err"] = s_reg * n_norm / act_norm - 1
        wr = os.path.join(d, "windows_result.csv")
        if os.path.exists(wr):
            w = {r["window"]: r for r in csv.DictReader(open(wr)) if r.get("s_step_normal_median")}
            if all(k in w for k in ("win_0", "win_1", "win_2", "win_3")):
                s = [float(w["win_%d" % k]["s_step_normal_median"]) for k in range(4)]
                x = [0.0, 0.25, 0.5, 0.75, 1.0]
                y = s + [s[3]]
                mean_s = sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2 for i in range(4))
                row.update(K1_job_err=s[0] * n_norm / act_norm - 1, K4_err=mean_s * n_norm / act_norm - 1,
                           s_windows=" / ".join("%.5f" % v for v in s),
                           r_rad_windows=" / ".join(w["win_%d" % k]["r_rad"] for k in range(4)))
        rows.append(row)

    print("%-11s %8s %8s %7s %5s | %8s %8s %8s | %s" % ("case", "wall", "pred", "err", "range", "K1_reg", "K1_job", "K4", "window s/step (t0/25/50/75%)"))
    for r in rows:
        if "note" in r:
            print("%-11s %s" % (r["case"], r["note"]))
            continue
        f = lambda k: ("%+7.1f%%" % (100 * r[k])) if k in r else "      -"
        print("%-11s %7.1fs %7.1fs %+6.1f%% %5s | %8s %8s %8s | %s" % (
            r["case"], r["wall_s"], r["pred_s"], 100 * r["wall_err"], "yes" if r["in_range"] else "no",
            f("K1_reg_err"), f("K1_job_err"), f("K4_err"), r.get("s_windows", "")))
    def med_abs(k):
        v = [abs(r[k]) for r in rows if k in r]
        return (statistics.median(v), len(v)) if v else (float("nan"), 0)
    for k, label in (("wall_err", "whole-run wall, registered K=1"), ("K1_reg_err", "compute, K1 registered probe"),
                     ("K1_job_err", "compute, K1 in-job (win_0)"), ("K4_err", "compute, K4 restart sampling")):
        m, n = med_abs(k)
        print("median |%s| = %.1f%%  (n=%d)" % (label, 100 * m, n))
    if len(sys.argv) > 1:
        keys = sorted({k for r in rows for k in r})
        with open(sys.argv[1], "w", newline="") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=keys)
            wcsv.writeheader()
            wcsv.writerows(rows)


if __name__ == "__main__":
    main()
